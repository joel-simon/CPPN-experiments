import numpy as np

def precision_recall(true, pred):
    assert(len(pred) == len(true))
    tp = 0 # True positives.
    fp = 0 # False positives.
    fn = 0 # False negatives.
    precision = 0
    recall = 0

    for p, t in zip(pred, true):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and t:
            fn += 1

    if tp+fp != 0:
        precision = tp / float(tp + fp)

    if tp+fn != 0:
        recall = tp / float(tp+fn)

    return precision, recall

def f1_score(true, pred):
    precision, recall = precision_recall(true, pred)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def balanced_accuracy_score(true, pred):
    true_pos = 0
    true_neg = 0

    num_pos = float(sum(true))
    num_neg = len(true) - num_pos

    for p, t in zip(pred, true):
        if p and t:
            true_pos += 1
        elif not p and not t:
            true_neg += 1

    a = true_neg / num_neg
    b = true_pos / num_pos

    return (((a+b)/2)-.5)*2

def balanced_accuracy_score_np(true, pred):
    """ For arbitrarily shaped numpy arrays. Must be same size/
    """
    n_true_pos = true.sum()
    n_true_neg = true.size - n_true_pos

    true_pos = np.logical_and(true, pred)
    true_neg = np.logical_not(np.logical_or(true, pred))

    a = true_neg.sum() / n_true_neg
    b = true_pos.sum() / n_true_pos

    return (a+b)/2

