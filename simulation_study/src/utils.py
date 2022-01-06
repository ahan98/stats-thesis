import numpy as np
from itertools import combinations

def partition(n1, n2):
    """Returns all unique ways to arrange [0..n1+n2) into two groups of size n1 and n2.

    Parameters
    ----------
    n1 : int
        Size of group 1
    n2 : int
        Size of group 2

    Returns
    -------
    nd.array
        A numpy array of size NxM, where N = n1 + n2, and M = choose(N, n1)
    """
    N = n1 + n2
    x1_idxs = list(combinations(range(N), n1))
    x2_idxs = list(combinations(range(N), n2))[::-1]
    return np.hstack((x1_idxs, x2_idxs))
