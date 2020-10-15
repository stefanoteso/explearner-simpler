import pickle
import numpy as np


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


def merge(y, z, result, mid):
    inv_count = 0
    i = 0
    j = 0

    while i < len(y) and j < len(z):
        if y[i] <= z[j]:
            result.append(y[i])
            i += 1
        else:
            result.append(z[j])
            inv_count = inv_count + (mid - i)
            j += 1

    result += y[i:]
    result += z[j:]

    return inv_count, result


def merge_sort(arr):
    result = []
    inv_count = 0

    if len(arr) == 1:
        return inv_count, arr

    mid = int(len(arr) / 2)
    inv_count, y = merge_sort(arr[:mid])
    temp_count, z = merge_sort(arr[mid:])

    inv_count += temp_count

    temp_count, result = merge(y, z, result, mid)

    inv_count += temp_count

    return inv_count, result


def kendall_tau_dist(x, y):
    """
    This is a `O(n\\log n)` implementation to compute kendall tau distance
    as the number of Bubble sort inversions defined in
    `Kendal Tau Distance <https://en.wikipedia.org/wiki/Kendall_tau_distance>`

    The fastest implementation runs in `O(n\\sqrt(\\log n)`
    """

    # Inverse the permutation y
    inverse = [x for x in range(len(y))]
    n = len(x)

    for i in np.arange(len(y)):
        inverse[((y[i]) - 1)] = i

    comp_perm = [x for x in range(len(x))]

    for i in range(0, len(inverse)):
        comp_perm[i] = inverse[x[i] - 1]

    # Compute merge_sort on the permutation
    dist, _ = merge_sort(comp_perm)
    pairs = (n * (n-1))/2
    dist = dist / pairs

    return dist


def rank_items(array):
    """
    Return an array with the ranking of each element.
    For instance if array=[1.0, 3.1, 2.2] the function returns the vector
    [0, 1, 2]

    Parameters
    ----------
    array: The input array

    Returns
    -------
    The array of rankings

    """

    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks
