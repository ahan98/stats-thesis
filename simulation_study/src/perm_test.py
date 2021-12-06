import scipy.stats as stats
import numpy as np


"""
Returns the proportion of permutations with a test statistic
as or more extreme than the observed test statistic.

H_0: x1.bar - x2.bar = delta
"""
def pval(x1, x2, partitions, delta=0, pooled=True, alternative="unequal"):
    t_obs = stats.ttest_ind(x1, x2, equal_var = pooled).statistic
    combined = np.append(x1-delta, x2)

    count = n_combos = 0
    for idxs1, idxs2 in partitions:
        group1, group2 = combined[idxs1], combined[idxs2]

        # check for a particular delta
        # how/if these counts change after changing the test statistic
        # e.g., via standardizing (x1.bar - x2.bar) / sqrt(s1^2/n1 + s2^2/n2)
        t = stats.ttest_ind(group1, group2, equal_var=pooled).statistic

        if alternative == "less":
            count += (t <= t_obs)
        elif alternative == "greater":
            count += (t >= t_obs)
        else:
            count += (t <= -abs(t_obs)) + (t >= abs(t_obs))

        n_combos += 1

    p = count / n_combos
    return delta, p
