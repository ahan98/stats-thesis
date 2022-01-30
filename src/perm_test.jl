using Statistics, HypothesisTests, Distributions
using Distributed


function permInterval(x1, x2, parts1, parts2, delta_true, wide, narrow; pooled=true, alpha=0.05, alternative="two-sided")
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

    # println(wide_lo, " ", wide_hi)
    # println(narrow_lo, " ", narrow_hi)
    # use binary search to find approximate permutation test confidence interval
    lo = search(x1, x2, parts1, parts2, wide[1], narrow[1], pooled=pooled, alpha=alpha, alternative=alternative)
    hi = search(x1, x2, parts1, parts2, narrow[2], wide[1], pooled=pooled, alpha=alpha, alternative=alternative)
    # println("(", lo, ", ", hi, ")")
    return lo <= delta_true <= hi
end


function search(x1, x2, parts1, parts2, start, stop; pooled=true, alternative="two-sided",
                margin=0.005, threshold=1.0, alpha=0.05)
    p_start = pval(x1, x2, parts1, parts2, pooled=pooled, alternative=alternative, delta=start)
    p_end = pval(x1, x2, parts1, parts2, pooled=pooled, alternative=alternative, delta=stop)
    # println("p_start = ", p_start, ", p_end = ", p_end)
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


function pval(x1, x2, parts1, parts2; pooled=false, alternative="two-sided", delta=0)
    x1_shifted = x1 .- delta             # shift group 1 under null hypothesis
    t_obs = ttest_ind(x1_shifted, x2, pooled)  # test statistic for observed data
    # println(t_obs)

    combined = vcat(x1_shifted, x2)  # join original pair into single vector
    x1s = combined[parts1]   # get all combinations of pairs from original pair
    x2s = combined[parts2]
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