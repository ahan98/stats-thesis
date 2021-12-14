from .perm_test import pval_vectorized

"""
Parameters:
x1 - data for group 1
x2 - data for group 2
start, end - initial bounds

This function performs a binary search on the interval [start, end]
corresponding to the (delta0, p_value) distribution.

Assumes that there exists delta in [start, end] such that pval(delta) = alpha.

Convergence criteria:
(1) The p-value from one iteration to the next has a percent change less than the desired
    threshold.
(2) We have an approximation for delta0 such that the corresponding p-value is within
    a desired margin of alpha.
"""
def search(x1, x2, partitions, start, end, margin=0.005, alpha=0.05, threshold=1):
    # Check that the p-values associated with delta = start and delta = end
    # are on opposite sides of alpha.
    p_start = pval_vectorized(x1, x2, partitions, delta=start)
    p_end = pval_vectorized(x1, x2, partitions, delta=end)
    # print("p_start =", p_start, "\np_end=", p_end)
    assert (p_start - alpha) * (p_end - alpha) <= 0

    i = 0
    p = p_new = delta = None
    while True:
        # print("iteration", i)

        delta = (start + end) / 2
        # print("delta =", delta, " in [", start, ",", end, "]")

        p_new = pval_vectorized(x1, x2, partitions, delta=delta)
        # print("p_new =", p_new)

        if p and percent_change(p, p_new) <= threshold:
            # (1) percent change is below threshold
            break

        if p_new > alpha + margin:
            start = delta
        elif p_new < alpha - margin:
            end = delta
        else:
            # (2) p-value is within margin of error
            break

        p = p_new
        i += 1

    return delta


def percent_change(old, new):
    return 100 * abs(new - old) / old

