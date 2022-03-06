module TestStatistics

export t, tconf

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


function tconf(x, y; pooled=true, alpha=0.05, dtype=Float32)
    # dx, dy = ndims(x), ndims(y)
    dx = dy = 1
    nx, ny = size(x, dx), size(y, dy)

    if pooled
        dof = [nx + ny - 2]
        varx = vary = ((nx-1).*var(x, dims=dx) .+ (ny-1).*var(y, dims=dy)) ./ dof
    else
        # https://online.stat.psu.edu/stat415/lesson/3/3.2
        varx, vary = var(x, dims=dx), var(y, dims=dy)
        a, b = varx ./ nx, vary ./ ny
        dof = @. (a + b)^2 / (a^2 / (nx - 1) + b^2 / (ny - 1))
        # println(dof)
    end

    t = map(TDist, dof)
    tcrit = map(quantile, t, 1-(alpha/2))
    margin = @. tcrit * sqrt(varx/nx + vary/ny)
    diff = mean(x, dims=dx) .- mean(y, dims=dy)
    return vcat(zip(dtype.(diff .- margin), dtype.(diff .+ margin))...)
end

end
