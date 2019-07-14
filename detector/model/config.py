import os


from .rpn import generate_anchor, load_label_dict, load_cls_dict, load_contain_dict
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word, get_processing_postags


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_postags  = get_processing_postags(self.vocab_tags)
        self.generate_anchor = generate_anchor()

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    dir_ner_model  = "../sequence_tagging/" + dir_output + "shared_var.ckpt/"
    path_log   = dir_output + "log.txt"

    # save features
    dir_saved_roi = "../data/saved_roi/"

    # embeddings
    dim_word = 300
    dim_char = 100
    dim_postags = 300

    # anchor types for each word center
    anchor_types = 6
    rpn_topN = 20

    # glove files
    data_pre = "../data/"
    filename_glove = data_pre + "glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = data_pre + "glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    dataset = 'ace2004'
    filename_dev = data_pre + dataset + "/dev.txt"
    filename_test = data_pre + dataset + "/test.txt"
    filename_train = data_pre + dataset + "/train.txt"

    # elmo file
    elmofile_dev = data_pre + dataset + "/elmo/elmo_dev.hdf5"
    elmofile_test = data_pre + dataset + "/elmo/elmo_test.hdf5"
    elmofile_train = data_pre + dataset + "/elmo/elmo_train.hdf5"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = data_pre + "words.txt"
    filename_tags = data_pre + "tags.txt"
    filename_chars = data_pre + "chars.txt"
    filename_cls = data_pre + "class_id"

    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3
    batch_sample     = 128 # samples that use to calculate rpn loss

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm_1 = 300 # lstm on word embeddings
    hidden_size_lstm_2 = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = False # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU

