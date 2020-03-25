import os
from collections import Counter
import numpy as np
import re

EPSILON = float(np.finfo(float).eps)


def remove_token(a, token_id):
    res = []
    for a_i in a:
        res_i = [t for t in a_i if t != token_id]
        res.append(res_i)
    
    return res

def get_batched_acc(a,a_r):
    # check input type
    ret = None
    if len(a) > 0 and isinstance(a[0],list):
        accuracy = []
        for i in range(len(a)):
            accuracy.append(get_accuracy(a[i], a_r[i]))
        ret = np.mean(accuracy, 0)
    else:
        ret = get_accuracy(a, a_r)
    return ret

def get_accuracy(a, a_r):
    # F1 score
    # remove end token
    '''if a[-1] == self.end_idx:
        a_new = a[: -1]
    else:
        a_new = a'''
    a_new = a
    #a_r_new = a_r[: -1] # first version action embedding with end token
    a_r_new = a_r
    # both are empty
    if len(a) == 0 and len(a_r) == 0:
        return 1.0
    a_common = list((Counter(a) & Counter(a_r)).elements())
    recall = len(a_common) / (len(a_r) + EPSILON)
    precision = len(a_common) / (len(a) + EPSILON)
    F1_score = 2 * precision * recall / (precision + recall + EPSILON)
    return F1_score


def get_batched_binary_acc(no_output_bi, a_r, action_type):
    ret = None
    if len(no_output_bi) > 0 and isinstance(no_output_bi[0],list):
        accuracy = []
        for i in range(len(no_output_bi)):
            accuracy.append(get_binary_accuracy(no_output_bi[i], a_r[i], action_type))
        ret = np.mean(accuracy, 0)
    else:
        ret = get_binary_accuracy(no_output_bi, a_r, action_type)
    return ret
    

def get_binary_accuracy(no_output_bi, a_r, action_type):
    # 0-1 accuracy of binary classifier
    num_category = len(no_output_bi)
    category_label_r = np.array(get_category_label(a_r, action_type))
    ret = []
    for i in range(num_category):
        ret.append(float(category_label_r[i][0] == no_output_bi[i]))
    return ret


def get_batched_finance_diff(a, a_r, m_start, action_money):
    ret = None
    if len(a) > 0 and isinstance(a[0], list):
        diff = []
        for i in range(len(a)):
            diff.append(get_finance_diff(a[i], a_r[i], m_start[i], action_money))
        ret = np.mean(diff, 0)
    else:
        ret = get_finance_diff(a, a_r, m_start, action_money)
    return ret

def get_finance_diff(a, a_r, m_start, action_money):
    m = sum(action_money[n] for n in a)
    m_r = sum(action_money[n] for n in a_r)
    #print("--------a:", a, "m: ", m, "m_r:", m_r, "m_start:", m_start)
    # assert m_r <= m_start, (a_r, m_r, m_start)
    if int(m_start) == 0:
        m_start = 0.1
    return abs(m_r - m) / float(m_start)


def get_batched_category_label(label, action_type):
    ret = None
    if len(label) > 0 and isinstance(label[0], list):
        ret = []
        for l in label:
            ret.append(get_category_label(l, action_type))
    else:
        ret = get_category_label(label, action_type)
    return ret

def get_category_label(label, action_type):
    t = [0.0, 1.0]
    f = [1.0, 0.0]
    res = [f] * 3
    
    for l in label:
        if action_type[l] == 0: # pistols
            res[0] = t
        elif 1 <= action_type[l] <= 5: # primary guns
            res[0] = t
        elif action_type[l] == 6: # grenades
            res[1] = t
        else: # equipment
            res[2] = t
    return res

def filter_batched_category_actions(label, action_type, num_categories):
    ret = None
    if len(label) > 0 and isinstance(label, list):
        ret = []
        for l in label:
            ret.append(filter_category_actions(l, action_type, num_categories))
    else:
        ret = filter_category_actions(label, action_type, num_categories)
    return ret

def filter_category_actions(label, action_type, num_categories):
    assert num_categories == 3
    
    ret = [[]] * num_categories
    for l in label:
        if action_type[l] <= 5:
            ret[0].append(l)
        elif action_type[l] == 6:
            ret[1].append(l)
        else:
            ret[2].append(l)
    return ret

def reshape_batched_category_actions(label_by_batch):
    num_categories = len(label_by_batch[0])
    label_by_category = [[] for i in range(num_categories)]
    for label in label_by_batch:
        for i in range(num_categories):
            label_by_category[i].append(label[i])
            
            
    return label_by_category

def read_npy(args):
    npy_dict = {}
    
    npy_dict["action_embedding"] = np.load(args.action_embedding)
    npy_dict["action_name"] = np.load(args.action_name, allow_pickle = True)
    npy_dict["action_money"] = np.load(args.action_money, allow_pickle = True)
    npy_dict["action_capacity"] = np.load(args.action_capacity, allow_pickle = True)
    npy_dict["type_capacity"] = np.load(args.type_capacity, allow_pickle = True)
    npy_dict["id2type"] = np.load(args.id2type, allow_pickle = True)
    npy_dict["typeid2name"] =  np.load(args.typeid2name, allow_pickle = True)
    npy_dict["side_mask"] = np.load(args.side_mask)
    
    return npy_dict

def find_latest_file(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        return max(files)[1]
    else:
        return None
