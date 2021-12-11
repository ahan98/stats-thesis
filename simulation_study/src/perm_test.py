import scipy.stats as stats
import numpy as np

"""
Returns the proportion of permutations with a test statistic
as or more extreme than the observed test statistic.

H_0: x1.bar - x2.bar = delta
"""
def pval(x1, x2, partitions, delta=0, pooled=True, alternative="unequal"):
    x1 = x1 - delta
    t_obs = ttest_ind(x1, x2, pooled)
    combined = np.append(x1, x2)

    count = n_combos = 0
    for idxs1, idxs2 in partitions:
        group1, group2 = combined[idxs1], combined[idxs2]
        t = ttest_ind(group1, group2, pooled)

        if alternative == "less":
            count += (t <= t_obs)
        elif alternative == "greater":
            count += (t >= t_obs)
        else:
            count += (t <= -abs(t_obs)) + (t >= abs(t_obs))

        n_combos += 1

    p = count / n_combos
    return delta, p


# TODO implement unpooled variance
def ttest_ind(x1, x2, pooled=True):
    n1, n2 = len(x1), len(x2)
    ss1, ss2 = np.sum(x1**2), np.sum(x2**2)
    mean1, mean2 = np.mean(x1), np.mean(x2)

    sample_var = lambda ss, mean, n: (ss - n*mean**2)/(n-1)
    var1, var2 = sample_var(ss1, mean1, n1), sample_var(ss2, mean2, n2)
    pooled_var = (n1 - 1)*var1 + (n2 - 1)*var2
    pooled_var /= n1 + n2 - 2.0

    t = (mean1 - mean2) / np.sqrt(pooled_var * (1.0/n1 + 1.0/n2))
    return t
