import re
import numpy as np


begin_pattern = re.compile(r'^B-*')
mid_pattern = re.compile(r'^I-*')
out_pattern = re.compile(r'^O')

def conflict(anchor_1, anchor_2):
    if (anchor_1[0] < anchor_2[0]) and \
            (anchor_2[0] < anchor_1[1]) and \
            (anchor_1[1] < anchor_2[1]):
            return True
    if (anchor_1[0] > anchor_2[0]) and \
            (anchor_1[0] < anchor_2[1]) and \
            (anchor_2[1] < anchor_1[1]):
            return True
    return False

def detect_conflict(candi_group, prob_group, cls_group,
        roi_feature_group, roi_elmo_feature_group, 
        roi_label_group, roi_len_group, roi_char_ids_group, 
        roi_word_lengths_group, sen_last_hidden_group,
        left_context_word_group, left_context_len_group, 
        right_context_word_group, right_context_len_group):
    """
    Accept the anchor with highest prob
    Delete conflict anchors
    """
    roi_feature_nonconf, roi_elmo_feature_nonconf, roi_label_nonconf, roi_len_nonconf = [], [], [], []
    roi_char_ids_nonconf, roi_word_lengths_nonconf, sen_last_hidden_nonconf = [], [], []
    left_context_word_nonconf, left_context_len_nonconf = [], []
    right_context_word_nonconf, right_context_len_nonconf = [], []

    keep = []
    orders  = np.argsort(-np.array(prob_group))
    while orders.size > 0:
        save_item = list(range(orders.shape[0]))

        # Accept the anchor with hightest prob
        highest_idx = orders[0]
        keep.append(highest_idx)
        save_item.remove(0)

        if __DELETE_CONF__:
        # delete conflict anchors
            for k in range(1, len(orders)):
                if conflict(candi_group[highest_idx], candi_group[orders[k]]):
                    save_item.remove(k)

        orders = orders[save_item]

    for idx in keep:
        # output probs and labels
        roi_feature_nonconf.append(roi_feature_group[idx])
        roi_elmo_feature_nonconf.append(roi_elmo_feature_group[idx])
        roi_label_nonconf.append(roi_label_group[idx])
        roi_len_nonconf.append(roi_len_group[idx])
        roi_char_ids_nonconf.append(roi_char_ids_group[idx])
        roi_word_lengths_nonconf.append(roi_word_lengths_group[idx])
        sen_last_hidden_nonconf.append(sen_last_hidden_group[idx])
        left_context_word_nonconf.append(left_context_word_group[idx])
        left_context_len_nonconf.append(left_context_len_group[idx])
        right_context_word_nonconf.append(right_context_word_group[idx])
        right_context_len_nonconf.append(right_context_len_group[idx])
    return roi_feature_nonconf, roi_elmo_feature_nonconf, roi_label_nonconf, roi_len_nonconf, roi_char_ids_nonconf, roi_word_lengths_nonconf, sen_last_hidden_nonconf, left_context_word_nonconf, left_context_len_nonconf, right_context_word_nonconf, right_context_len_nonconf

def load_contain_dict(file_name):
    contain_dict = {}
    for line in open(file_name, 'r'):
        arr = line.strip().split('\t')
        if arr[0] not in contain_dict.keys():
            contain_dict[arr[0]] = []
        contain_dict[arr[0]].append([int(arr[1]), int(arr[2]), int(arr[3])])
    return contain_dict

def load_cls_dict(file_name):
    cls_dict = dict()
    for line in open(file_name):
        arr = line.strip().split(':')
        cls_dict[arr[0]] = int(arr[1])
    return cls_dict

def load_label_dict(file_name):
    try:
        f = open(file_name)
        d = dict()
        for idx, word in enumerate(f):
            word = word.strip()
            d[idx] = word
        return d
    except:
        pass

def get_contain_entity(words, idx_arr, contain_dict):
    idx_1 = idx_arr[1]
    idx_2 = idx_arr[2]
    entity_words = words[idx_1: idx_2+1]
    long_entity = ' '.join(entity_words)
    if long_entity in contain_dict.keys():
        short_list = contain_dict[long_entity]
        return_list = []
        for s in short_list:
            s_type = s[0]
            s_idx_1 = idx_1 + s[1]
            s_idx_2 = idx_1 + s[2]
            return_list.append([s_type, s_idx_1, s_idx_2])
        return return_list
    return []

def get_anchor_label(idx1, idx2, true_entity, sen_len):
    """
    Return the label of an input idx pair
    1, id: positive pair, class_id
    0, 0: negitive pair
    -1, -1: unvalid pair
    """
    if idx1 >=0 and idx2 < sen_len:
        if idx1 in true_entity:
            for candidate in true_entity[idx1]:
                if candidate[0] == idx2:
                    return 1, candidate[1] # true entity pair
        return 0, 0 # neg pair
    return -1, -1 # labels out of boundary

def k_anchors(true_entity, sen_len, idx):
    """
    Generate 5 types of anchors and labels for each word
    sentence: A B C D E
    word: C
    idx: 2
    """
    anchors = []
    anchor_labels = []
    cls_ids = []
    sample_indexes = []

    entity_tag = False

    # type 1: C
    anchors.append([idx, idx])
    an_label, cls_id = get_anchor_label(
        idx, idx, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)
    if (an_label == 1):
        entity_tag = True

    # type 2: CD
    anchors.append([idx, idx + 1])
    an_label, cls_id = get_anchor_label(
        idx, idx + 1, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)
    if (an_label == 1):
        entity_tag = True

    # type 3: BCD
    anchors.append([idx - 1, idx + 1])
    an_label, cls_id = get_anchor_label(
        idx - 1, idx + 1, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)
    if (an_label == 1):
        entity_tag = True

    # type 4: BCDE
    anchors.append([idx - 1, idx + 2])
    an_label, cls_id = get_anchor_label(
        idx - 1, idx + 2, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)
    if (an_label == 1):
        entity_tag = True

    # type 5: ABCDE
    anchors.append([idx - 2, idx + 2])
    an_label, cls_id = get_anchor_label(
        idx - 2, idx + 2, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    # type 6: ABCDEF
    anchors.append([idx - 2, idx + 3])
    an_label, cls_id = get_anchor_label(
        idx - 2, idx + 3, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    if (an_label == 1):
        entity_tag = True

    if (entity_tag == True):
        # add other entities as negs
        sample_indexes = list(range(idx*5, (idx+1)*5))
        #print("*******word idx", idx, "sample indexes:", sample_indexes)

    return anchors, anchor_labels, cls_ids, sample_indexes

def get_pairs(line):
    """
        Description:
            extract entity pairs from each line
        type: line: str
                    sentence+'\t'+entity_pairs
                    sentence: words (split: ' ')
                    entity pairs: (split: '\t')
                    each pair: entity_type, start_idx, end_idx
        rtype:
            idx: dict
                key: begin_idx
                value: [end_idx, entity_type]
    """

    # get positive anchor pairs
    idx_dict = {}
    arr = line.strip().split('\t')
    for i in range(2, len(arr)):
        info = arr[i].split(' ')
        entity_type = int(info[0])
        begin_idx = int(info[1])
        end_idx = int(info[2])
        if begin_idx not in idx_dict.keys():
            idx_dict[begin_idx] = []
        idx_dict[begin_idx].append([end_idx, entity_type])

    # get sentence length
    words = arr[0].split(' ')
    sen_len = len(words)

    return idx_dict, sen_len

def generate_anchor():
    #label_dict, cls_dict, contain_dict):

    def f(line):
        """
        Generate anchors and anhcor labels for one line
        Input: tags for one line
        Output: anchors and anchor labels for one line
        """
        # get positive index pairs
        pos_pair, sen_len = get_pairs(line)

        # total data
        line_anchor, line_label, line_cls = [], [], []
        sample_indexes = []
        for word_idx in range(sen_len):

            # get anchors and labels for each line
            anchors, labels, cls, s_indexes = k_anchors(pos_pair,  sen_len, word_idx)

            # append word reuslt
            line_anchor += anchors
            line_label += labels
            line_cls += cls
            sample_indexes += s_indexes

        # return total data
        return line_anchor, line_label, line_cls, sample_indexes
    return f
