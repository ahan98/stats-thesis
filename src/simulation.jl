module Simulation

export coverage
export Alternative, less, greater, twoSided

@enum Alternative less greater twoSided


function coverage(xs, ys, wide, narrow, delta_true, args)
    results = permInterval.(eachrow(xs), eachrow(ys), wide, narrow, delta_true, args)
    results = hcat(results...)
    coverage = sum(results[1,:]) / S
    avg_CI_width = mean(results[2,:])
    return coverage, avg_CI_width
end


function permInterval(x, y, wide, narrow, delta_true, args)
    """
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
        True if permutation confidence interval includes difference in population means.
    Float32
        Width of permutation confidence interval
    """
    wide_lo, wide_hi = wide
    narrow_lo, narrow_hi = narrow
    lo = search(x, y, wide_lo, narrow_lo, args.px, args.py, args.pooled, args.alt_lo, isLowerBound=true)
    hi = search(x, y, narrow_hi, wide_hi, args.px, args.py, args.pooled, args.alt_hi, isLowerBound=false)
    return [(lo <= delta_true <= hi), hi - lo]
end


function search(x, y, start, stop, px, py, pooled, alternative;
                isLowerBound=true, margin=0.005, threshold=1.0)
    p_start = pval(x, y, start, px, py, pooled, alternative)
    p_end   = pval(x, y, stop,  px, py, pooled, alternative)

    # p-values corresponding to `start` and `stop` must be on opposite sides of `alpha`
    @assert (p_start - alpha) * (p_end - alpha) <= 0

    p = p_new = delta = nothing
    percent_change = (old, new) -> 100 * abs(new-old) / old

    while true
        delta = (start + stop) / 2
        p_new = pval(x, y, delta, px, py, pooled, alternative)

        if !isnothing(p) && percent_change(p, p_new) <= threshold
            break  # (1) percent change in p-value is below `threshold`
        end

        compare = (alpha - p_new) - isLowerBound * 2 * (alpha - p_new)
        if margin < compare
            stop = delta
        elseif margin < -compare
            start = delta
        else
            break  # (2) p-value is within `margin` of `alpha`
        end

        p = p_new
    end

    return delta
end


function pval(x, y, delta, px, py, pooled, alternative)
    """
    Parameters
    ----------
    x : Vector{Real}
        Data for group 1

    y : Vector{Real}
        Data for group 2

    pooled : Bool
        Assume equal/unequal variances for the two groups

    alternative : String
        Type of alternative hypothesis

    delta : Real
        Null hypothesis difference in means

    Returns
    -------
    Float64
        Proportion of pairs among all sample combinations which have
        a test statistic as or more extreme than the original pair (x, y)
    """
    x_shift = x .- delta
    t_obs = t(x_shift, y, pooled)  # test statistic for observed data
    ts = testStatDistr(x_shift, y, px, py, pooled)

    if alternative == less
        n_extreme = count(ts .<= t_obs)
    elseif alternative == greater
        n_extreme = count(ts .>= t_obs)
    elseif alternative == twoSided
        n_extreme = count(@. (ts <= -abs(t_obs)) | (ts >= abs(t_obs)))
    else
        error("Undefined alternative: $alternative")
    end

    return dtype(n_extreme / size(px, 1))  # proportion of pairs w/ extreme test statistic
end


function testStatDistr(x, y, px, py, pooled)
    combined = vcat(x, y)      # join original pair into single vector
    xs = combined[px]          # get all combinations of pairs from original pair
    ys = combined[py]
    return t(xs, ys, pooled)   # test statistic for all possible pairs of samples
end

end
