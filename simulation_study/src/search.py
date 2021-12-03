from .perm_test import pval


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
def search(x1, x2, start, end, margin=0.005, alpha=0.05, threshold=1):
    # Check that the p-values associated with delta = start and delta = end
    # are on opposite sides of alpha.
    p_start = pval(x1, x2, delta=start)
    p_end = pval(x1, x2, delta=end)
    if (p_start - alpha) * (p_end - alpha) >= 0:
        return None

    p = i = 0
    while True:
        print("iteration", i)

        mid = (start + end) / 2
        print("delta0 =", mid, " in [", start, ",", end, "]")

        delta, p_new = pval(x1, x2, delta=mid)

        if p and percent_change(p, p_new) <= threshold:
            # (1) percent change is below threshold
            p = p_new
            break

        if p_new > alpha + margin:
            start = mid
        elif p_new < alpha - margin:
            end = mid
        else:
            # (2) p-value is within margin of error
            p = p_new
            break

        i += 1

    return delta, p


def percent_change(old, new):
    return 100 * abs(new - old) / old
