import re


begin_pattern = re.compile(r'^B-*')
mid_pattern = re.compile(r'^I-*')
out_pattern = re.compile(r'^O')

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

def get_true_entity(tags, label_dict, cls_dict):
    """
    Get positive entity index diccts
    """
    true_idx = []
    tmp = []
    for i in range(len(tags)):
        tag = label_dict[tags[i]]
        if begin_pattern.match(tag):
            if len(tmp) > 0:
                # append end index
                tmp.append(i-1)
                # append to result
                true_idx.append(tmp)
                tmp = []
            # append cls_id and begin index
            tmp.append(cls_dict[tag])
            tmp.append(i)
        elif out_pattern.match(tag):
            if len(tmp) > 0:
                # append end index
                tmp.append(i-1)
                # append to result
                true_idx.append(tmp)
    if len(tmp) > 0:
        # append end index
        tmp.append(len(tags) - 1)
        # append to result
        true_idx.append(tmp)

    # transfer list to dict
    idx_dict = {} # entity begin_idx=>end_idx, class_id
    for en_idx in true_idx:
        #        begin idx  =>  end idx  ,  cls_id
        idx_dict[en_idx[1]] = [en_idx[2], en_idx[0]]
    #print idx_dict
    return idx_dict

def get_anchor_label(idx1, idx2, true_entity, sen_len):
    """
    Return the label of an input idx pair
    1, id: positive pair, class_id
    0, 0: negitive pair
    -1, -1: unvalid pair
    """
    if idx1 >=0 and idx2 < sen_len:
        if true_entity.has_key(idx1) and \
                true_entity[idx1][0] == idx2:
                    return 1, true_entity[idx1][1] # true entity pair
        return 0, 0 # neg pair
    return -1, -1 # labels out of boundary

def k_anchors(tags, true_entity, idx):
    """
    Generate 5 types of anchors and labels for each word
    sentence: A B C D E
    word: C
    idx: 2
    """
    anchors = []
    anchor_labels = []
    cls_ids = []
    sen_len = len(tags)

    # type 1: C
    anchors.append([idx, idx])
    an_label, cls_id = get_anchor_label(
        idx, idx, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    # type 2: CD
    anchors.append([idx, idx + 1])
    an_label, cls_id = get_anchor_label(
        idx, idx + 1, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    # type 3: BCD
    anchors.append([idx - 1, idx + 1])
    an_label, cls_id = get_anchor_label(
        idx - 1, idx + 1, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    # type 4: BCDE
    anchors.append([idx - 1, idx + 2])
    an_label, cls_id = get_anchor_label(
        idx - 1, idx + 2, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    # type 5: ABCDE
    anchors.append([idx - 2, idx + 2])
    an_label, cls_id = get_anchor_label(
        idx - 2, idx + 2, true_entity, sen_len)
    anchor_labels.append(an_label)
    cls_ids.append(cls_id)

    return anchors, anchor_labels, cls_ids

def generate_anchor(label_dict, cls_dict):

    def f(tags):
        """
        Generate anchors and anhcor labels for one line
        Input: tags for one line
        Output: anchors and anchor labels for one line
        """
        # get positive index pairs
        pos_pair = get_true_entity(tags, label_dict, cls_dict)

        # total data
        line_anchor, line_label, line_cls = [], [], []
        for word_idx in range(len(tags)):

            # get anchors and labels for eacch line
            anchors, labels, cls = k_anchors(tags, pos_pair, word_idx)

            # append word reuslt
            line_anchor += anchors
            line_label += labels
            line_cls += cls

        # return total data
        return line_anchor, line_label, line_cls
    return f
