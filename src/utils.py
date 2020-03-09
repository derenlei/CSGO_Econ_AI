import os
from collections import Counter
import numpy as np
import re

EPSILON = float(np.finfo(float).eps)

def get_accuracy(a, a_r):
    # F1 score
    # remove end token
    '''if a[-1] == self.end_idx:
        a_new = a[: -1]
    else:
        a_new = a'''
    a_new = a
    a_r_new = a_r[: -1]
    # both are empty
    if len(a_new) == 0 and len(a_r_new) == 0:
        return 1.0
    a_common = list((Counter(a_new) & Counter(a_r_new)).elements())
    recall = len(a_common) / (len(a_r) + EPSILON)
    precision = len(a_common) / (len(a) + EPSILON)
    F1_score = 2 * precision * recall / (precision + recall + EPSILON)
    return F1_score

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