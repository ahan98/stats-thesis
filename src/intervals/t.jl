using Statistics, Distributions

function tconf(x, y, delta, alpha, pooled)
    lo, hi = tconf(x, y, alpha, pooled)
    return lo <= delta <= hi, hi - lo
end

function tconf(x, y, alpha::Real, pooled::Bool)
    nx, ny = length(x), length(y)

    if pooled
        dof = nx + ny - 2
        varx = vary = ((nx-1)*var(x) + (ny-1)*var(y)) / dof
    else
        # https://online.stat.psu.edu/stat415/lesson/3/3.2
        varx, vary = var(x), var(y)
        a, b = varx / nx, vary / ny
        dof = (a + b)^2 / (a^2 / (nx - 1) + b^2 / (ny - 1))
    end

    t_dof = TDist(dof)
    tcrit = quantile(t_dof, alpha/2)
    margin = tcrit * sqrt(varx/nx + vary/ny)
    diff = mean(x) - mean(y)
    return diff - abs(margin), diff + abs(margin)
end

function t_estimates(x, y, pooled)
    # Compute t confidence intervals for each of the B*S pairs
    wide   = tconf(x, y, 0.000001, pooled)
    narrow = tconf(x, y, 0.5, pooled)
    return wide, narrow
end

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
    meanx = mean(xs, dims=1)
    varx  = var(xs, dims=1)

    meany = mean(ys, dims=1)
    vary  = var(ys, dims=1)

    nx, ny = size(xs, 1), size(ys, 1)  # number of observations per group

    if pooled
        pooled_var = ((nx-1)*varx + (ny-1)*vary) / (nx+ny-2)
        denom = sqrt.(pooled_var * (1/nx + 1/ny))
    else
        denom = sqrt.(varx/nx + vary/ny)
    end

    return (meanx-meany)./denom
end
