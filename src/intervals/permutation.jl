include("t.jl")
using StatsBase

@enum Alternative smaller greater twoSided
struct P
    mean_og   # unshifted group mean
    var_og    # unshifted group variance
    nshift    # number of items to be shifted
    shift_sum # original sum of items to be shifted
    n         # group size
end

function permInterval(x, y, wide, narrow, delta::Real, pooled, alpha, alt_lo, alt_hi, margin=0.005)
    lo, hi = permInterval(x, y, wide, narrow, pooled, alpha, alt_lo, alt_hi, margin)
#     @show lo, hi
    return lo <= delta <= hi, hi - lo
end

function permInterval(x, y, wide, narrow, pooled::Bool, alpha, alt_lo, alt_hi, margin)
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

    lo = hi = nothing
    @sync begin
        # search for lower and upper bounds in parallel
        @async lo = search(x, y, wide_lo, narrow_lo,
                           pooled, alt_lo, alpha, isLowerBound=true, margin=margin)
        @async hi = search(x, y, narrow_hi, wide_hi,
                           pooled, alt_hi, alpha, isLowerBound=false, margin=margin)
    end

    return lo, hi
end


function search(x, y, start, stop, pooled, alternative, alpha;
                isLowerBound=true, margin=0.005)
    p_start = pval(x, y, start, pooled, alternative)
    p_end   = pval(x, y, stop,  pooled, alternative)
    # p-values corresponding to `start` and `stop` must be on opposite sides of `alpha`
    @assert (p_start - alpha) * (p_end - alpha) <= 0

    p = p_new = delta = nothing
    percent_change = (old, new) -> 100 * abs(new-old) / old
    i = 0
    while true
        # @show start, stop
        delta = (start + stop) / 2
        p_new = pval(x, y, delta, pooled, alternative)

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


function pval(x::P, y::P, d, pooled, alternative, dtype=Float32)
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
    x_var, x_mean = var_shift(x, d)
    y_var, y_mean = var_shift(y, d)
    
    ts = @. x_mean - y_mean

#     diff = @. x_mean - y_mean
#     if pooled
#         var_pool = @. ((x.n - 1) * x_var + (y.n - 1) * y_var) / (x.n + y.n - 2)
#         denom = var_pool .* sqrt(1 / x.n + 1 / y.n)
#     else
#         denom = @. sqrt(x_var / x.n + y_var / y.n)
#     end
    
#     ts = @. diff / denom
    # @show ts[1:5]
    
    if alternative == smaller
        n_extreme = count(ts .<= ts[1])
    elseif alternative == greater
        n_extreme = count(ts .>= ts[1])
        # @show "g", n_extreme
    elseif alternative == twoSided
        n_extreme = count(@. abs(ts) >= abs(ts[1]))
    else
        error("Undefined alternative: $alternative")
    end

    return n_extreme / length(ts)  # proportion of pairs w/ extreme test statistic
end


# function pval(x::P, y::P, d)
#     x_var, x_mean = var_shift(x, d)
#     y_var, y_mean = var_shift(y, d)
    
#     diff = @. x_mean - y_mean
#     denom = @. sqrt(x_var / x.n + y_var / y.n)
    
#     ts = @. diff / denom

#     return count(@. abs(ts) >= abs(ts[1])) / length(ts)
# end

function var_shift(a::P, d)
    temp = a.nshift * d
    n = a.n
    mean_ = a.mean_og .- temp/n
    var_ = @. a.var_og + (temp^2/n + temp*d - 2*d*a.shift_sum + 2*temp*(a.mean_og - temp/n))/(n-1)
    return var_, mean_
end


function testStatDistr(x, y, px, py, pooled)
    combined = vcat(x, y)      # join original pair into single vector
    @inbounds xs = @view combined[px]          # get all combinations of pairs from original pair
    @inbounds ys = @view combined[py]
    return t(xs, ys, pooled)   # test statistic for all possible pairs of samples
end
