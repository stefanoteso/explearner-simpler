import pickle
import numpy as np


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


def move_indices(dst, src, indices):
    dst, src, indices = set(dst), set(src), set(indices)
    assert indices.issubset(src), 'not in src'
    assert indices.isdisjoint(dst), 'in dst'
    dst = np.array(list(sorted(dst | set(indices))))
    src = np.array(list(sorted(src - set(indices))))
    return dst, src
