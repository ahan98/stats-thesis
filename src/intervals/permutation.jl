include("t.jl")
using StatsBase

@enum Alternative smaller greater twoSided

function permInterval(x, y, delta::Real, perms, pooled, alpha, alt_lo, alt_hi, margin=0.005)
    lo, hi = permInterval(x, y, perms, pooled, alpha, alt_lo, alt_hi, margin)
    #@show lo, hi
    return lo <= delta <= hi, hi - lo
end

function permInterval(x, y, perms, pooled, alpha, alt_lo, alt_hi, margin)
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

    wide, narrow = t_estimates(x, y, pooled)

    wide_lo, wide_hi = wide
    narrow_lo, narrow_hi = narrow

    lo = hi = nothing
    @sync begin
        # search for lower and upper bounds in parallel
        @async lo = search(x, y, wide_lo, narrow_lo,
                           perms, pooled, alt_lo, alpha, isLowerBound=true, margin=margin)
        @async hi = search(x, y, narrow_hi, wide_hi,
                           perms, pooled, alt_hi, alpha, isLowerBound=false, margin=margin)
    end

    return lo, hi
end


function search(x, y, start, stop, perms, pooled, alternative, alpha;
                isLowerBound=true, margin=0.005)
    p_start = pval(x, y, start, perms, pooled, alternative)
    p_end   = pval(x, y, stop,  perms, pooled, alternative)
    # p-values corresponding to `start` and `stop` must be on opposite sides of `alpha`
    @assert (p_start - alpha) * (p_end - alpha) <= 0

    p = p_new = delta = nothing
    percent_change = (old, new) -> 100 * abs(new-old) / old
    i = 0
    while true
        # @show start, stop
        delta = (start + stop) / 2
        p_new = pval(x, y, delta, perms, pooled, alternative)

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
        i += 1
        p = p_new
    end
    #@show i

    return delta
end


function pval(x, y, delta, perms, pooled, alternative, dtype=Float32)
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

    # @show size(x_shift)
    # @show size(y)
    # @show size(px)
    # @show size(py)
    # @show pooled
    if perms == nothing
        nx, ny = length(x), length(y)
        px, py = partition(nx, ny, 10_000)
    else
        px, py = perms
    end
    ts = testStatDistr(x_shift, y, px, py, pooled)
    # @show size(ts)

    if alternative == smaller
        n_extreme = count(ts .<= t_obs)
    elseif alternative == greater
        n_extreme = count(ts .>= t_obs)
        # @show "g", n_extreme
    elseif alternative == twoSided
        n_extreme = count(@. abs(ts) >= abs(t_obs))
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
