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

    def get_groups(idxs):
        i = 0
        idxs2 = []
        for j in idxs:
            while i < N and i < j:
                idxs2.append(i)
                i += 1

            if i == j:
                i += 1

        idxs2 += range(i, N)
        return list(idxs) + idxs2

    partitions = np.array([get_groups(idxs) for idxs in combinations(range(N), n1)])
    return partitions
