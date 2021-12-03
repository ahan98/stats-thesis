import numpy as np
import math


def bootstrap_ci(x1, x2, alpha=0.05, delta=0, epochs=10000):
    x1 = x1 - delta  # make sample means equivalent under null hypothesis
    diffs = []
    for _ in range(epochs):
        bs1 = np.random.choice(x1, size=len(x1), replace=True)
        bs2 = np.random.choice(x2, size=len(x2), replace=True)
        diffs.append(np.mean(bs1) - np.mean(bs2))

    diffs.sort()
    tail = (alpha/2) * len(diffs)
    #print(tail)

    # if rounding, make CI more conservative
    lower = math.ceil(tail)
    upper = math.floor(epochs - tail)
    #print(lower, upper)
    assert lower < upper
    return (diffs[lower], diffs[upper])
