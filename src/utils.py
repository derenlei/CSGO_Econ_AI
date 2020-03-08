import os
from collections import Counter
import numpy as np


def get_accuracy(a, a_r):
    EPSILON = float(np.finfo(float).eps)
    # F1 score
    a_common = list((Counter(a) & Counter(a_r.numpy())).elements())
    recall = len(a_common) / len(a_r)
    precision = len(a_common) / len(a)
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