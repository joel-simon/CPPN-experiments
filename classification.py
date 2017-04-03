import numpy as np

def balanced_accuracy(true, pred):
    """ For arbitrarily shaped numpy arrays. Must be same shape.
    """
    n_true_pos = true.sum()
    n_true_neg = true.size - n_true_pos

    true_pos = np.logical_and(true, pred)
    true_neg = np.logical_not(np.logical_or(true, pred))

    a = true_neg.sum() / float(n_true_neg)
    b = true_pos.sum() / float(n_true_pos)

    return (a+b)/2
