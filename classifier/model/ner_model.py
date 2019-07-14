import numpy as np
import os
import tensorflow as tf

from .utils import report
from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel

from tensorflow.contrib.rnn import BasicLSTMCell

class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def _softmax_with_mask(self, logits, lens, axis=-1):
        mask = tf.sequence_mask(lens, maxlen=tf.shape(logits)[axis], dtype=tf.int32)
        max_logits = tf.add(logits, 1000000.0 * tf.cast((mask - 1), tf.float32))
        self.max_logits = tf.reduce_max(max_logits, axis=1, keepdims=True)
        self.red_logits = logits - self.max_logits
        self.red_logits = self.red_logits * tf.cast(mask, tf.float32)

        exp_logits = tf.exp(self.red_logits)
        masked_exp_logits = tf.multiply(exp_logits, tf.cast(mask, tf.float32))
        self.masked_exp_logits_sum = tf.clip_by_value(tf.reduce_sum(masked_exp_logits, axis), 1e-37, 1e37)
        return tf.clip_by_value(tf.div(masked_exp_logits, tf.expand_dims(self.masked_exp_logits_sum, axis)), 1e-37, 1e37)

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max_anchor_len, anchor_feature_len)
        self.anchor_features = tf.placeholder(tf.float32,
                shape=[None, self.config.anchor_types, self.config.input_feature_dim],
                name="anchor_features")

        # shape = (batch size, max_anchor_len, anchor_feature_len)
        self.elmo_features = tf.placeholder(tf.float32,
                shape=[None, 3, self.config.anchor_types, 1024],
                name="elmo_features")

        # shape = (batch size, time_step, 600)
        self.all_context = tf.placeholder(tf.float32, shape=[None, None, 2 * self.config.hidden_size_lstm_2],
                        name="all_context_features")
        # shape = (batch size)
        self.all_context_len = tf.placeholder(tf.int32, shape=[None],
                        name="all_context_len")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                       name="word_lengths")

        # shape = (batch size)
        self.anchor_lens = tf.placeholder(tf.int32, shape=[None],
                        name="anchor_lens")

        # shape = (batch size)
        self.cls_labels = tf.placeholder(tf.int32, shape=[None],
                        name="cls_labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, features, elmo_features, context_features,
            left_context_features, left_context_lens, right_context_features, right_context_lens,
            char_ids, word_lens, lens=None, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # build feed dictionary
        feed = {
            self.anchor_features: np.array(features),
            self.elmo_features: np.array(elmo_features),
        }

        # get all the context
        all_context = []
        for i in range(len(left_context_features)):
            left = left_context_features[i]
            right = right_context_features[i]
            al = np.concatenate((left, right), axis = 0)
            all_context.append(al)
        all_context, all_context_len = pad_sequences(
                all_context, pad_tok=[0]*600) # 600
        feed[self.all_context] = np.array(all_context)
        feed[self.all_context_len] = all_context_len

        if self.config.use_chars:
            char_ids, _ = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lens

        if lens is not None:
            feed[self.anchor_lens] = lens

        if labels is not None:
            feed[self.cls_labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed

    def add_word_embeddings_op(self):
        with tf.variable_scope("chars"): # chars
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([self.anchor_features, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

    def add_lstm_1_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("lstm_1"): # bi-lstm
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_1)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_1)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.anchor_lens, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            # lstm_output shape: [batch_size, time_step, hidden_dim * 2]
            self.lstm_1_output = tf.nn.dropout(output, self.dropout)

    def add_elmo_op(self):
        with tf.variable_scope("elmo"):
            # elmo layer weights
            elmo_layer_weights = tf.get_variable("elmo_layer_weights", dtype=tf.float32,
                    shape=[3], initializer = tf.contrib.layers.xavier_initializer())
            norm_weights = tf.nn.softmax(elmo_layer_weights)

            batch_num = tf.shape(self.elmo_features)[0]
            nsteps = tf.shape(self.elmo_features)[2]
            norm_weights = tf.expand_dims(tf.expand_dims(tf.expand_dims(norm_weights, 0), 2), 3)
            norm_weights = tf.tile(norm_weights, [batch_num, 1, nsteps, 1024])

            # self.elmo_embedding shape: [batch_size, 3, time_step, 1024]
            # elmo_embedding shape: [batch_size, time_step, 1024]
            weighted_elmo = tf.multiply(self.elmo_features, norm_weights)
            elmo_embedding = tf.reduce_sum(self.config.elmo_scale * weighted_elmo, axis=1)
            # concat elmo embedding and lstm_1_output
            # shape: [batch_size, time_step, 1024 + hidden_dim * 2]
            self.concat_hidden = tf.concat([elmo_embedding, self.lstm_1_output], axis=-1)

    def add_lstm_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("cls"): # bi-lstm
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_2)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_2)

            _, (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.concat_hidden,
                    sequence_length=self.anchor_lens, dtype=tf.float32)

            # last output shape: [batch_size, hidden_dim * 2]
            output = tf.concat([output_states_fw[0], output_states_bw[0]], axis = -1)
            #output = tf.reshape(tf.transpose(output, [1, 0, 2]), [-1, 1200])

            self.last_output = tf.nn.dropout(output, self.dropout)
            #self.nsteps = tf.shape(self.lstm_output)[1]

    def add_context_lstm_op(self):
        with tf.variable_scope("context_lstm"): # bi-lstm
            cell_fw = tf.contrib.rnn.LSTMCell(300)
            cell_bw = tf.contrib.rnn.LSTMCell(300)
            _, (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.all_context,
                    sequence_length=self.all_context_len, dtype=tf.float32)
            self.context_output = tf.concat([output_states_fw[0], output_states_bw[0]], axis = -1)

    def add_context_att_lstm_op(self):
        with tf.variable_scope("context_att_lstm"): # bi-lstm

            s = tf.shape(self.all_context)
            time_step = s[1]

            self.WA = tf.get_variable("WA", dtype=tf.float32, shape=[600, 600],
                    initializer = tf.contrib.layers.xavier_initializer())

            entity = tf.expand_dims(tf.matmul(self.last_output, self.WA), 2)
            att_logits = tf.matmul(self.all_context, entity)
            self.att_logits = tf.squeeze(att_logits, [2])
            self.attention = self._softmax_with_mask(self.att_logits, self.all_context_len)

            expand_attention = tf.tile(tf.expand_dims(self.attention, -1), [1, 1, 600])
            attention_context = expand_attention * self.all_context

            cell_fw = tf.contrib.rnn.LSTMCell(300)
            cell_bw = tf.contrib.rnn.LSTMCell(300)
            _, (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, attention_context, sequence_length=self.all_context_len, dtype=tf.float32)
            self.context_output = tf.concat([output_states_fw[0], output_states_bw[0]], axis = -1)

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("cls"):
            W1 = tf.get_variable("W1", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm_2, self.config.cls_hidden_size])

            b1 = tf.get_variable("b1", shape=[self.config.cls_hidden_size],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            W2 = tf.get_variable("W2", dtype=tf.float32,
                    shape=[self.config.cls_hidden_size, self.config.roi_types])

            b2 = tf.get_variable("b2", shape=[self.config.roi_types],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            cls_hidden = tf.nn.relu(tf.matmul(self.last_output, W1) + b1)
            self.logits = tf.matmul(cls_hidden, W2) + b2

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        self.one_hot_labels = tf.one_hot(self.cls_labels, depth=self.config.roi_types)

        # get probs
        #probs = tf.nn.softmax(self.logits)
        # add max margin loss
        pos = tf.ones_like(self.one_hot_labels)
        neg = -tf.ones_like(self.one_hot_labels)
        IND = tf.where(tf.equal(self.one_hot_labels, 1), x = pos, y = neg)
        max_margin_loss = tf.square(tf.maximum(0.0, 5 - tf.multiply(IND, self.logits)))
        self.loss = tf.reduce_mean(max_margin_loss)

        """
        class_weights = tf.constant([[1, 1.2, 1, 1]], dtype=tf.float32)
        weights = tf.reduce_sum(class_weights * self.one_hot_labels, axis=1)
        losses = unweighted_losses * weights
        self.loss = tf.reduce_mean(losses)
        """

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_lstm_1_op()
        self.add_elmo_op()
        self.add_lstm_op()
        self.add_context_att_lstm_op() #add att
        #self.add_context_lstm_op() #add context
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init

    def run_epoch(self, train, dev, epoch, dev_total_entity):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        #train.shuffle_data()

        # iterate over dataset
        for i, (features, elmo_features, lens, labels, chars, word_lens, context_features, lc, ll, rc, rl) in enumerate(minibatches(train, batch_size)):

            fd = self.get_feed_dict(features, elmo_features, context_features,
                    lc, ll, rc, rl,
                    chars, word_lens, lens = lens, labels = labels,
                    lr = self.config.lr, dropout = self.config.dropout)

            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

        metrics = self.run_evaluate(dev, dev_total_entity)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def run_evaluate(self, test, total_entity):
        """
        Evaluate rpn on test dataset
        """
        total_pred = np.array([], dtype=np.int32)
        for features, elmo_features, lens, labels, chars, word_lens, context_features, lc, ll, rc, rl in minibatches(test, self.config.batch_size):

            fd = self.get_feed_dict(features, elmo_features, context_features,
                    lc, ll, rc, rl,
                    chars, word_lens, lens, dropout = 1)
            [pred_labels, pred_logits] = self.sess.run([self.labels_pred,
                self.logits], feed_dict = fd)
            total_pred = np.append(total_pred, pred_labels)

        f1 = report(test.roi_labels, np.array(total_pred), total_entity)
        return {"f1": 100 * f1}
