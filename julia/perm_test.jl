using Statistics, HypothesisTests, Distributions
include("partition.jl")


function permInterval(data, n1, n2; pooled=true, alpha=0.05, alternative="two-sided")
    x1 = data[1:n1]
    x2 = data[n1+1:n1+n2]
    parts = partition(n1, n2)
    wide_lo, wide_hi = tconf(x1, x2, alpha=0.01, pooled=pooled)
    narrow_lo, narrow_hi = tconf(x1, x2, alpha=0.1, pooled=pooled)
    lo = search(x1, x2, parts, wide_lo, narrow_lo, pooled=pooled, alpha=alpha, alternative=alternative)
    hi = search(x1, x2, parts, narrow_hi, wide_hi, pooled=pooled, alpha=alpha, alternative=alternative)
    return (lo, hi)
end


# TODO documentation
function search(x1, x2, partitions, start, stop; pooled=true, alternative="two-sided",
                margin=0.005, threshold = 5, alpha=0.05)
    """Returns delta s.t. pval(delta) ~= alpha
    """

    p_start = pval(x1, x2, partitions, pooled=pooled, alternative=alternative, delta=start)
    p_end = pval(x1, x2, partitions, pooled=pooled, alternative=alternative, delta=stop)
    # Check that the p-values corresponding to `start` and `stop` are on
    # opposite sides of `alpha`. Otherwise, the binary search will not converge.
    @assert (p_start - alpha) * (p_end - alpha) <= 0

    p = p_new = delta = nothing
    percent_change = (old, new) -> 100 * abs(new-old) / old

    # i = 0
    while true
        # i += 1
        # println("iteration ", i)
        delta = (start + stop) / 2
        p_new = pval(x1, x2, partitions, pooled=pooled, alternative=alternative, delta=delta)

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


function pval(x1, x2, partitions; pooled=true, alternative="two-sided", delta=0)
    """Returns the proportion of permutations with a test statistic
    as or more extreme than the observed test statistic (based on the alternative).

    Parameters
    ----------
    x1 : Vector{Int}
        Data for group 1
    x2 : Vector{Int}
        Data for group 2
    partitions : Matrix{Vector{Int64}}
        The i-th row is a pair of vectors denoting the indexes of the elements
        (from the pair of original samples concatenated together) in each
        possible pair of samples
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
    x1s = combined[partitions[1]]  # get all combinations of pairs from original pair
    x2s = combined[partitions[2]]

    t_obs = ttest_ind(x1, x2, pooled)  # test statistic for observed data
    ts = ttest_ind(x1s, x2s, pooled)   # test statistic for all possible pairs of samples

    if alternative == "smaller"
        n_extreme = count(ts .<= t_obs)
    elseif alternative == "larger"
        n_extreme = count(ts .>= t_obs)
    else
        n_extreme = count(@. (ts <= -abs(t_obs)) || (ts >= abs(t_obs)))
    end

    return n_extreme / size(partitions[1])[1]  # proportion of pairs w/ extreme test statistic
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


# TODO documentation
# TODO this is similar to ttest_ind() so organize code?
function tconf(x1, x2; pooled=true, alpha=0.05)
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
