import numpy as np
from scipy.stats import ttest_ind
import statsmodels.stats.api as sms


def permInterval(x1, x2, partitions, delta_true, pooled=True, alpha=0.05, alternative="two-sided"):
    wide_lo, wide_hi = tconf(x1, x2, pooled=pooled, alpha=0.01)
    narrow_lo, narrow_hi = tconf(x1, x2, pooled=pooled, alpha=0.1)
    lo = search(x1, x2, partitions, wide_lo, narrow_lo, alpha=alpha, pooled=pooled, alternative=alternative)
    hi = search(x1, x2, partitions, narrow_hi, wide_hi, alpha=alpha, pooled=pooled, alternative=alternative)
    return lo <= delta_true <= hi


def search(x1, x2, partitions, start, end, margin=0.005, threshold=1,
           alpha=0.05, pooled=True, alternative="two-sided"):
    """Returns the difference in means for which the corresponding permutation
    test outputs a p-value equal to alpha.

    This method performs a binary search to locate the difference in means,
    using `start` and `end` as initial guesses, and converges when either
    of the following occurs:
    (1) The corresponding p-values between consecutive iterations changes by
        at most `threshold` percent.
    (2) The corresponding p-value is within `margin` of `alpha`.

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
        The difference in the two population means corresponding to
        a p-value (approximately) equal to alpha.
    """

    p_start = pval(x1, x2, partitions, delta=start, pooled=pooled, alternative=alternative)
    p_end = pval(x1, x2, partitions, delta=end, pooled=pooled, alternative=alternative)
    # print("p_start =", p_start, ", p_end=", p_end)

    # Check that the p-values corresponding to `start` and `end` are on
    # opposite sides of `alpha`. Otherwise, the binary search will not converge.
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
            break  # (1) percent change is below threshold

        if p_new > alpha + margin:
            start = delta
        elif p_new < alpha - margin:
            end = delta
        else:
            break  # (2) p-value is within margin of error

        p = p_new
        i += 1

    return delta


def pval(x1, x2, partitions, delta=0, pooled=True, alternative="two-sided"):
    """ Returns the proportion of permutations with a test statistic
    "as or more extreme" (based on the alternative) than the observed
    test statistic.

    Parameters
    ----------
    x1 : nd.array
        Data for group 1
    x2 : nd.array
        Data for group 2
    """
    x1 = x1 - delta
    t_obs = ttest_ind(x1, x2, axis=-1, equal_var=pooled, alternative=alternative).statistic

    combined = np.append(x1, x2)
    n1 = len(x1)
    x1s = combined[partitions[:,:n1]]
    x2s = combined[partitions[:,n1:]]
    ts = ttest_ind(x1s, x2s, axis=-1, equal_var=pooled, alternative=alternative).statistic

    if alternative == "less":
        subset = np.where(ts <= t_obs)
    elif alternative == "greater":
        subset = np.where(ts >= t_obs)
    else:
        subset = np.where((ts <= -abs(t_obs)) | (ts >= abs(t_obs)))

    return len(subset[0]) / len(partitions)


# https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.CompareMeans.tconfint_diff.html
def tconf(x1, x2, alpha=0.05, pooled=True, alternative="two-sided"):
    cm = sms.CompareMeans(sms.DescrStatsW(x1), sms.DescrStatsW(x2))
    return cm.tconfint_diff(alpha, alternative, usevar="pooled" if pooled else "unequal")
