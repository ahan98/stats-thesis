import numpy as np

# TODO implement unpooled
def ttest_ind_vectorized(x1s, x2s, n1, n2, pooled=True):
    #n1, n2 = x1s.shape[-1], x2s.shape[-1]
    #print("n1 =", n1, "n2 =", n2)
    sum1 = np.sum(x1s, axis=-1)
    sum2 = np.sum(x2s, axis=-1)
    #print("sums", sum1, sum2)

    mean1 = sum1 / n1
    mean2 = sum2 / n2
    #print("means", mean1, mean2)

    sample_var = lambda x, mean, n: (np.sum(x**2, axis=-1) - n*mean**2) / (n-1)
    var1 = sample_var(x1s, mean1, n1)
    var2 = sample_var(x2s, mean2, n2)
    #print("sample variances", var1, var2)

    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
    #print("pooled var", pooled_var)
    denom = np.sqrt(pooled_var * (1/n1 + 1/n2))

    t = (mean1 - mean2) / denom
    return t


"""
Returns the proportion of permutations with a test statistic
as or more extreme than the observed test statistic.

H_0: x1.bar - x2.bar = delta
"""
def pval_vectorized(x1, x2, partitions, delta=0, pooled=True, alternative="unequal"):
    n1, n2 = len(x1), len(x2)
    #print(n1, n2)
    x1 = x1 - delta

    combined = np.append(x1, x2)
    x1s = combined[partitions[:,:n1]]
    x2s = combined[partitions[:,n1:]]
    ts = ttest_ind_vectorized(x1s, x2s, n1, n2, pooled)

    subset = None
    t_obs = ttest_ind_vectorized(x1, x2, n1, n2, pooled)
    #print("t_obs =", t_obs)
    if alternative == "less":
        subset = np.where(ts <= t_obs)
    elif alternative == "greater":
        subset = np.where(ts >= t_obs)
    else:
        subset = np.where((ts <= -abs(t_obs)) | (ts >= abs(t_obs)))

    p = len(subset[0]) / len(partitions)
    return p
