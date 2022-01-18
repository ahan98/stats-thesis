using Statistics, HypothesisTests, Distributions
using Distributed


function permInterval(x1, x2, parts1, parts2, delta_true; pooled=true, alpha=0.05, alternative="two-sided")
    """Returns true (false) if permutation test confidence interval does (not) include difference in
    population means.

    Parameters
    ----------
    x1 : Vector{Float64}
        Data for group 1
    x2 : Vector{Float64}
        Data for group 2
    partitions : Tuple{Matrix{Int64}, Matrix{Int64}}
        The i-th rows of x1[partitions[1]] and x2[partitions[2]] denote the i-th arrangement of
        the original (n1+n2) observations into two groups of size n1 and n2.
    delta_true : Float64
        Difference in population means
    pooled : Bool
        Assume pooled or unpooled variances
    alpha : Float64
        Significance level
    alternative : String
        Type of alternative hypothesis ("two-sided", "smaller", "larger")

    Returns
    -------
    Bool
        True (false) if permutation test confidence interval does (not) include difference in population means.
    """

    # provide estimates of permutation test CI using t-test CIs
    wide_lo, wide_hi = tconf(x1, x2, alpha=0.01, pooled=pooled)
    narrow_lo, narrow_hi = tconf(x1, x2, alpha=0.1, pooled=pooled)
    # use binary search to find approximate permutation test confidence interval
    lo = search(x1, x2, parts1, parts2, wide_lo, narrow_lo, pooled=pooled, alpha=alpha, alternative=alternative)
    hi = search(x1, x2, parts1, parts2, narrow_hi, wide_hi, pooled=pooled, alpha=alpha, alternative=alternative)
    println("(", lo, ", ", hi, ")")
    return lo <= delta_true <= hi
end


function search(x1, x2, parts1, parts2, start, stop; pooled=true, alternative="two-sided",
                margin=0.005, threshold = 5.0, alpha=0.05)
    """Returns the difference in means for which the corresponding permutation
    test has a p-value approximately equal to alpha.

    This method returns the value for delta such that
    pval(x1, x2, partitions; pooled=true, alternative="two-sided", delta=delta) ~= alpha

    This method performs a binary search for delta in [start, stop], converging if one
    of the following occurs:
    (1) The percent change between the last two p-values is at most `threshold`.
    (2) The newest p-value is within `margin` of `alpha`.

    Parameters
    ----------
    x1 : Vector{Float64}
        Data for group 1
    x2 : Vector{Float64}
        Data for group 2
    partitions : Tuple{Matrix{Int64}, Matrix{Int64}}
        Each row of the first/second matrix contains the indexes of the original n1+n2 elements
        denoting each arrangement of the first/second group.
    start : Float64
        Initial lower estimate of delta
    stop : Float64
        Initial upper estimate of delta
    pooled : Bool
        Assume pooled or unpooled variances
    alternative : String
        Type of alternative hypothesis ("two-sided", "smaller", "larger")
    margin : Float64
        search() terminates if the newest p-value is within `margin` of `alpha`
    threshold : Float64
        search() terminates if the newest p-value is within `threshold` percent of the last p-value
    alpha : Float64
        Significance level

    Returns
    -------
    float
        The difference in the two population means corresponding to
        a p-value (approximately) equal to alpha.
    """
    p_start = pval(x1, x2, parts1, parts2, pooled=pooled, alternative=alternative, delta=start)
    p_end = pval(x1, x2, parts1, parts2, pooled=pooled, alternative=alternative, delta=stop)
    # p-values corresponding to `start` and `stop` must be on opposite sides of `alpha`
    @assert (p_start - alpha) * (p_end - alpha) <= 0

    p = p_new = delta = nothing
    percent_change = (old, new) -> 100 * abs(new-old) / old

    # i = 0
    while true
        # i += 1
        # println("iteration ", i)
        delta = (start + stop) / 2
        p_new = pval(x1, x2, parts1, parts2, pooled=pooled, alternative=alternative, delta=delta)

        if !isnothing(p) && percent_change(p, p_new) <= threshold
            break  # (1) percent change in p-value is below `threshold`
        end

        if p_new > alpha + margin
            start = delta
        elseif p_new < alpha + margin
            stop = delta
        else
            break  # (2) p-value is within `margin` of `alpha`
        end

        p = p_new
    end

    return delta
end


function pval(x1, x2, parts1, parts2; alternative="two-sided", delta=0)
    """Returns the permutation test p-value, i.e., the proportion of permutations
    (of the original n1+n2 observations into two groups of size n1 and n2) which have
    a test statistic as or more extreme than the observed test statistic.

    Parameters
    ----------
    x1 : Vector{Int64}
        Data for group 1
    x2 : Vector{Int64}
        Data for group 2
    partitions : Tuple{Matrix{Int64}, Matrix{Int64}}
        Each row of the first/second matrix contains the indexes of the original n1+n2 elements
        denoting each arrangement of the first/second group.
    pooled : Bool
        Assume pooled or unpooled variances
    alternative : String
        Type of alternative hypothesis ("two-sided", "larger", "smaller")
    delta : Float64
        Amount to shift original data in group 1 under the null hypothesis

    Returns
    -------
    Float64
        Proportion of all possible pairs of samples which have a test statistic as
        or more extreme than the test statistic for the original pair.
    """

    x1 = x1 .- delta               # shift group 1 under null hypothesis
    combined = vcat(x1, x2)        # join original pair into single vector
    x1s = combined[parts1]  # get all combinations of pairs from original pair
    x2s = combined[parts2]

    t_obs = ttest_ind(x1, x2, pooled)  # test statistic for observed data
    ts = ttest_ind(x1s, x2s, pooled)   # test statistic for all possible pairs of samples

    if alternative == "smaller"
        n_extreme = count(ts .<= t_obs)
    elseif alternative == "larger"
        n_extreme = count(ts .>= t_obs)
    else
        n_extreme = count(@. (ts <= -abs(t_obs)) | (ts >= abs(t_obs)))
    end

    return n_extreme / size(parts1)[1]  # proportion of pairs w/ extreme test statistic
end


function ttest_ind(x1s, x2s, pooled)
    """Returns the t test statistic for each pair of samples.

    Each pair of samples has n1/n2 observations in the first/second group.
    Let S denote the number of pairs.

    Parameters
    ----------
    x1s : Matrix{Float64}
        Data for group 1 (size S x n1)
    x2s : Matrix{Float64}
        Data for group 2 (size S x n2)
    pooled : Bool
        Assume pooled or unpooled variances

    Returns
    -------
    Matrix{Float64}
        Sx1 matrix where the i-th entry denotes the t test statistic for the i-th pair.
    """
    d = ndims(x1s)  # if 1D vector, compute by column, if 2D matrix, compute by row

    mean1 = mean(x1s, dims=d)
    var1 = var(x1s, mean=mean1, dims=d)

    mean2 = mean(x2s, dims=d)
    var2 = var(x2s, mean=mean2, dims=d)

    n1, n2 = size(x1s)[d], size(x2s)[d]  # number of observations per group

    if pooled
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
        denom = sqrt.(pooled_var * (1/n1 + 1/n2))
    else
        denom = sqrt.(var1/n1 + var2/n2)
    end

    return (mean1-mean2)./denom
end


# TODO consider merging with ttest_ind()
function tconf(x1, x2; pooled=true, alpha=0.05)
    """Returns (1-alpha)% t-test confidence interval for the difference in means.

    Reference: http://www.stat.yale.edu/Courses/1997-98/101/meancomp.htm

    Parameters
    ----------
    x1 : Vector{Int64}
        Data for group 1
    x2 : Vector{Int64}
        Data for group 2
    pooled : Bool
        Assume pooled (equal) or unpooled (unequal) variances
    alpha : Float64
        Significance level

    Returns
    -------
    Tuple{Float64, Float64}
        (1-alpha)% t-test confidence interval for the difference in means
    """
    n1, n2 = length(x1), length(x2)
    if pooled
        dof = n1 + n2 - 1
        var1 = var2 = ((n1-1)*var(x1) + (n2-2)*var(x2)) / (n1+n2-2)
    else
        dof = min(n1, n2) - 1
        var1, var2 = var(x1), var(x2)
    end

    t = TDist(dof)
    tcrit = quantile(t, 1-(alpha/2))
    margin = tcrit * sqrt(var1/n1 + var2/n2)
    diff = mean(x1) - mean(x2)
    return (diff - margin, diff + margin)
end
