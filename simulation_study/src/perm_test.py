import numpy as np
import statsmodels.stats.api as sms


def find_intervals(batch_size):
    intervals = []
    n_captured = 0

    for _ in range(batch_size):
        x1 = np.random.gamma(gamma1[0], gamma1[1], n1)
        x2 = np.random.gamma(gamma2[0], gamma2[1], n2)

        # TODO accommodate different alternatives (i.e., lesser, greater, unequal)
        # lesser  - alpha = 0.95
        # greater - alpha = 0.05
        # unequal - alpha_left = 0.025, alpha_right = 0.975
        t99 = tconfint(0.001, pooled, x1, x2)
        t90 = tconfint(0.20, pooled, x1, x2)

        try:
            lower = search(x1, x2, partitions, t99[0], t90[0])
            upper = search(x1, x2, partitions, t90[1], t99[1])
        except AssertionError:
            continue

        intervals.append((lower, upper))
        n_captured += (lower <= delta_true) * (delta_true <= upper)

    return intervals, n_captured


# TODO write custom version for performance purposes
def tconfint(alpha, x1, x2, pooled=True, alternative="two-sided"):
    cm = sms.CompareMeans(sms.DescrStatsW(x1), sms.DescrStatsW(x2))
    return cm.tconfint_diff(alpha, alternative, usevar="pooled" if pooled else "unequal")


# TODO finish documentation
def pval(x1, x2, partitions, delta=0, pooled=True, alternative="unequal"):
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
    x1 = x1 - delta

    combined = np.append(x1, x2)
    x1s = combined[partitions[:,:n1]]
    x2s = combined[partitions[:,n1:]]
    ts = ttest_ind(x1s, x2s, n1, n2, pooled)
    t_obs = ttest_ind(x1, x2, n1, n2, pooled)
    #print("t_obs =", t_obs)

    if alternative == "less":
        subset = np.where(ts <= t_obs)
    elif alternative == "greater":
        subset = np.where(ts >= t_obs)
    else:
        subset = np.where((ts <= -abs(t_obs)) | (ts >= abs(t_obs)))

    p = len(subset[0]) / len(partitions)
    return p


def ttest_ind(x1s, x2s, n1, n2, pooled=True):
    #n1, n2 = x1s.shape[-1], x2s.shape[-1]
    #print("n1 =", n1, "n2 =", n2)
    sum1 = np.sum(x1s, axis=-1)
    sum2 = np.sum(x2s, axis=-1)
    #print("sums", sum1, sum2)

    mean1 = sum1 / n1
    mean2 = sum2 / n2
    #print("means", mean1, mean2)

    var1 = np.var(x1s, ddof=1)
    var2 = np.var(x2s, ddof=1)
    #print("sample variances", var1, var2)

    # http://www.stat.yale.edu/Courses/1997-98/101/meancomp.htm
    if pooled:
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
        denom = np.sqrt(pooled_var * (1/n1 + 1/n2))
    else:
        denom = np.sqrt(var1/n1 + var2/n2)

    t = (mean1 - mean2) / denom
    return t
