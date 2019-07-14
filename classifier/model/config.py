import os


from .rpn import generate_anchor, load_label_dict, load_cls_dict
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


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
        self.label_dict = load_label_dict(self.filename_tags)
        #self.cls_dict = load_cls_dict(self.filename_cls)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

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

    # anchor types for each word center
    anchor_types = 6
    rpn_topN = 20
    roi_types = 8
    # glove files
    filename_glove = "../data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "../data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = dir_saved_roi + "dev_word_ids/"
    filename_test = dir_saved_roi + "test_word_ids/"
    filename_train = dir_saved_roi + "train_word_ids/"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "../data/words.txt"
    filename_tags = "../data/tags.txt"
    filename_chars = "../data/chars.txt"
    filename_cls = "../data/class_id"

    # training
    train_embeddings = False
    nepochs          = 30 # 15
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
    hidden_size_lstm = 300 # lstm on word embeddings
    input_feature_dim = 128 #600 + 1024
    cls_hidden_size = 50

    # evaluataion
    train_total_entity = 24687
    dev_total_entity = 3217
    test_total_entity = 3027

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = False # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU

