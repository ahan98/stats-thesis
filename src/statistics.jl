using Statistics, Distributions

function t(xs, ys, pooled)
    """
    Parameters
    ----------
    xs : AbstractArray{Real, N}
        Data for group 1
        If N == 1, then size(xs) = (nx,)
        If N == 2, then size(xs) = (S, nx)
    
    ys : AbstractArray{Real, N}
        Data for group 2
        If N == 1, then size(ys) = (ny,)
        If N == 2, then size(ys) = (S, ny)
    
    pooled : Bool
        Assume equal/unequal variances for the two groups

    Returns
    -------
    S-element Vector{Float64}
        t test statistic for each pair
    """
    d = ndims(xs)  # if 1D vector, compute by column, if 2D matrix, compute by row

    meanx = _mean(xs, d)
    varx  = _var(xs, d, meanx)
    
    meany = _mean(ys, d)
    vary = _var(ys, d, meany)

    nx, ny = size(xs, d), size(ys, d)  # number of observations per group

    if pooled
        pooled_var = ((nx-1)*varx + (ny-1)*vary) / (nx+ny-2)
        denom = sqrt.(pooled_var * (1/nx + 1/ny))
    else
        denom = sqrt.(varx/nx + vary/ny)
    end

    return (meanx-meany)./denom
end

function tconf(x1, x2; pooled=true, alpha=0.05)
   d1, d2 = ndims(x1), ndims(x2)
    n1, n2 = size(x1, d1), size(x2, d2)
    
    if pooled
        dof = n1 + n2 - 2
        var1 = var2 = ((n1-1).*var(x1, dims=d1) .+ (n2-2).*var(x2, dims=d2)) ./ dof
        #var1, var2 = var(x1, dims=d1), var(x2, dims=d2)
    else
        dof = min(n1, n2) - 1
        var1, var2 = var(x1, dims=d1), var(x2, dims=d2)
    end

    t = TDist(dof)
    tcrit = quantile(t, 1-(alpha/2))
    margin = @. tcrit * sqrt(var1/n1 + var2/n2)
    diff = mean(x1, dims=d1) .- mean(x2, dims=d2)
    return vcat(zip(diff .- margin, diff .+ margin)...)
end

function _mean(x, d)
    d = ndims(x)
    return sum(x, dims=d) ./ size(x, d)
end


function _var(x, d, means)
    nx = size(x, d)
    ss = sum(x.^2, dims=d)
    return @. (ss - (nx * means.^2)) / (nx-1)
end


function _var(x, d)
    means = _mean(x ,d)
    return _var(x, d, means)
end
