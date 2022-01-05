import numpy as np
import statsmodels.stats.api as sms


def search(x1, x2, partitions, start, end, margin=0.005, threshold=1,
           alpha=0.05, pooled=True, alternative="two-sided"):
    """Returns the difference in means for which the corresponding permutation
    test outputs a p-value equal to alpha.

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
    p_start = pval(x1, x2, partitions, delta=start, pooled=pooled, alternative=alternative)
    p_end = pval(x1, x2, partitions, delta=end, pooled=pooled, alternative=alternative)
    # print("p_start =", p_start, ", p_end=", p_end)
    assert (p_start - alpha) * (p_end - alpha) <= 0

    i = 0
    p = p_new = delta = None
    percent_change = lambda old, new : 100 * abs(new - old) / old

    while True:
        # print("iteration", i)
        delta = (start + end) / 2
        # print("delta =", delta, " in [", start, ",", end, "]")
        p_new = pval(x1, x2, partitions, delta=delta, pooled=pooled, alternative=alternative)
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


# TODO documentation
def pval(x1, x2, partitions, delta=0, pooled=True, alternative="two-sided"):
    """ Returns the proportion of permutations with a test statistic
    "as or more extreme" (i.e., based on the alternative) than the observed
    test statistic.

    Parameters
    ----------
    x1 : nd.array
        Data for group 1
    x2 : nd.array
        Data for group 2
    """
    n1, n2 = len(x1), len(x2)
    #print(n1, n2)
    # In general, due to this shift, the observed test statistic is not constant,
    # even though the observed data (prior to the shift) remains the same.
    x1 = x1 - delta

    combined = np.append(x1, x2)
    x1s = combined[partitions[:,:n1]]
    x2s = combined[partitions[:,n1:]]
    ts = ttest_ind(x1s, x2s, n1, n2, pooled)
    t_obs = ttest_ind(x1, x2, n1, n2, pooled)

    if alternative == "smaller":
        subset = np.where(ts <= t_obs)
    elif alternative == "larger":
        subset = np.where(ts >= t_obs)
    else:
        subset = np.where((ts <= -abs(t_obs)) | (ts >= abs(t_obs)))

    p = len(subset[0]) / len(partitions)
    return p


# TODO documentation
def ttest_ind(x1s, x2s, n1, n2, pooled=True):
    #n1, n2 = x1s.shape[-1], x2s.shape[-1]
    #print("n1 =", n1, "n2 =", n2)
    sum1 = np.sum(x1s, axis=-1)
    sum2 = np.sum(x2s, axis=-1)
    #print("sums", sum1, sum2)

    mean1 = sum1 / n1
    mean2 = sum2 / n2
    #print("means", mean1, mean2)

    var1 = np.var(x1s, ddof=1, axis=-1)
    var2 = np.var(x2s, ddof=1, axis=-1)
    #print("sample variances", var1, var2)

    # http://www.stat.yale.edu/Courses/1997-98/101/meancomp.htm
    if pooled:
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
        denom = np.sqrt(pooled_var * (1/n1 + 1/n2))
    else:
        denom = np.sqrt(var1/n1 + var2/n2)

    t = (mean1 - mean2) / denom
    return t
