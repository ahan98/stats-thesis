include("t.jl")

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
        True if permutation interval includes difference in population means.
    Float32
        Width of permutation interval
    """
    wide_lo, wide_hi = wide
    narrow_lo, narrow_hi = narrow
    lo = hi = undef
    @sync begin
        # search for lower and upper bounds in parallel
        @async lo = search(x, y, wide_lo, narrow_lo,
                           args.permuter, args.pooled, args.alt_lo, args.alpha, isLowerBound=true)
        @async hi = search(x, y, narrow_hi, wide_hi,
                           args.permuter, args.pooled, args.alt_hi, args.alpha, isLowerBound=false)
    end

    return (lo <= delta_true <= hi), (hi - lo)
end


function search(x, y, start, stop, permuter, pooled, alternative, alpha;
                isLowerBound=true, margin=0.005, threshold=1.0)
    p_start = pval(x, y, start, permuter, pooled, alternative)
    p_end   = pval(x, y, stop,  permuter, pooled, alternative)

    # p-values corresponding to `start` and `stop` must be on opposite sides of `alpha`
    @assert (p_start - alpha) * (p_end - alpha) <= 0

    p = p_new = delta = nothing
    percent_change = (old, new) -> 100 * abs(new-old) / old

    while true
        # @show start, stop
        delta = (start + stop) / 2
        p_new = pval(x, y, delta, permuter, pooled, alternative)

        if !isnothing(p) && percent_change(p, p_new) <= threshold
            # println("condition 1")
            break  # (1) percent change in p-value is below `threshold`
        end

        # if p_new < alpha - margin      # p-value is too small
        #     if isLowerBound
        #         start = delta          # go right
        #     else
        #         println("go left")
        #         stop = delta           # go left
        #     end
        # elseif p_new > alpha + margin  # p-value is too big
        #     if isLowerBound
        #         stop = delta           # go left
        #     else
        #         println("go right")
        #         start = delta          # go right
        #     end
        # else
        #     println("condition 2")
        #     break
        # end

        compare = (alpha - p_new) - isLowerBound * 2 * (alpha - p_new)
        if margin < compare
            stop = delta  # go left
        elseif margin < -compare
            start = delta # go right
        else
            # println("condition 2")
            break  # (2) p-value is within `margin` of `alpha`
        end

        p = p_new
    end

    return delta
end


function pval(x, y, delta, permuter, pooled, alternative, dtype=Float32)
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
    # @show t_obs

    # @show size(x_shift)
    # @show size(y)
    # @show size(px)
    # @show size(py)
    # @show pooled
    px, py, mc_size = permuter
    if mc_size > 0
        p_mc = hcat([shuffle(1:nx+ny) for _ in 1:mc_size]...)
        px = p_mc[1:nx, :]
        py = p_mc[nx+1:end, :]
    end
    ts = testStatDistr(x_shift, y, px, py, pooled)
    # @show size(ts)

    if alternative == smaller
        n_extreme = count(ts .<= t_obs)
    elseif alternative == greater
        n_extreme = count(ts .>= t_obs)
    elseif alternative == twoSided
        n_extreme = count(@. (ts <= -abs(t_obs)) | (ts >= abs(t_obs)))
    else
        error("Undefined alternative: $alternative")
    end

    return dtype(n_extreme / size(px, 2))  # proportion of pairs w/ extreme test statistic
end


function testStatDistr(x, y, px, py, pooled)
    combined = vcat(x, y)      # join original pair into single vector
    @inbounds xs = @view combined[px]          # get all combinations of pairs from original pair
    @inbounds ys = @view combined[py]
    return t(xs, ys, pooled)   # test statistic for all possible pairs of samples
end
