from .perm_test import pval


def search(x1, x2, partitions, start, end, alpha=0.05, margin=0.005, threshold=1, alternative="two-sided"):
    """Returns the difference in means for which the corresponding permutation
    test outputs a p-value equal to alpha.

    This function performs a binary search on the interval [start, end]
    corresponding to the (delta0, p_value) distribution.

    Assumes that there exists delta in [start, end] such that pval(delta) = alpha.

    Convergence criteria:
    (1) The p-value from one iteration to the next has a percent change less than the desired
        threshold.
    (2) We have an approximation for delta0 such that the corresponding p-value is within
        a desired margin of alpha.

    Parameters
    ----------
    x1 : nd.array
        Data for group 1
    x2 : nd.array
        Data for group 2
    start : float
        Initial lower bound for difference in means
    end : float
        Initial upper bound for difference in means

    Returns
    -------
    float
        The hypothesized true difference in population means.
        The permutation test associated with this difference outputs a p-value
        (approximately) equal to alpha.
    """

    # Check that the p-values associated with delta = start and delta = end
    # are on opposite sides of alpha.
    p_start = pval(x1, x2, partitions, delta=start, alternative=alternative)
    p_end = pval(x1, x2, partitions, delta=end, alternative=alternative)
    # print("p_start =", p_start, ", p_end=", p_end)
    assert (p_start - alpha) * (p_end - alpha) <= 0

    i = 0
    p = p_new = delta = None
    percent_change = lambda old, new : 100 * abs(new - old) / old

    while True:
        # print("iteration", i)
        delta = (start + end) / 2
        # print("delta =", delta, " in [", start, ",", end, "]")
        p_new = pval(x1, x2, partitions, delta=delta, alternative=alternative)
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
