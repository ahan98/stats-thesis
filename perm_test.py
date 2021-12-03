import scipy.stats as stats
import numpy as np
from itertools import combinations


"""
Returns the proportion of permutations with a test statistic
as or more extreme than the observed test statistic.

H_0: x1.bar - x2.bar = delta
"""
def pval(x1, x2, delta=0, pooled=True, alternative="unequal"):
    x1 = x1 - delta
    t_obs = stats.ttest_ind(x1, x2, equal_var = pooled).statistic

    count = n_combos = 0
    n1, n2 = len(x1), len(x2)
    for combo in combinations(range(n1+n2), n1):
        group1 = []
        used = [False for _ in range(n1+n2)]

        for i in combo:
            used[i] = True
            group1.append(x1[i] if i < n1 else x2[i-n1])

        group2 = []
        for i in range(n1 + n2):
            if not used[i]:
                group2.append(x1[i] if i < n1 else x2[i-n1])

        # check for a particular delta
        # how/if these counts change after changing the test statistic
        # e.g., via standardizing (x1.bar - x2.bar) / sqrt(s1^2/n1 + s2^2/n2)
        t = stats.ttest_ind(group1, group2, equal_var=pooled)

        if alternative == "less":
            count += (t <= t_obs)
        elif alternative == "greater":
            count += (t >= t_obs)
        else:
            count += (t <= -abs(t_obs)) + (t >= abs(t_obs))

        n_combos += 1

    p = count / n_combos
    return delta, p
