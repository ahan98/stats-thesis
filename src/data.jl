using Distributions
using Random

include("statistics.jl")

function generateData(B, S, nx, ny, distrTypeX, paramsX, distrTypeY, paramsY, dtype=Float32, seed=123)
    """
    Parameters
    ----------
    B - number of batches (# average coverage probabilities per boxplot)
    S - size of batch     (# confidence intervals per average coverage probability)
    nx - size of group 1 samples
    ny - size of group 2 samples
    distrTypeX - distribution type for group 1
    distrTypeY - distribution type for group 2
    paramsX - list of ranges for each parameter of distrTypeX
    paramsY - list of ranges for each parameter of distrTypeY
    """

    # generate B distributions for each group
    distrX = distrTypeX.(paramsX...)
    distrY = distrTypeY.(paramsY...)
    deltas = @. dtype(mean(distrX) - mean(distrY))
    # @show distrX
    # @show distrY
    # @show deltas

    # draw S groups of n observations from each of the B distributions
    Random.seed!(seed)
    x = rand.(distrX, S * nx)  # B vectors of length S * nx
    y = rand.(distrY, S * ny)

    # reshape into single vector of length B*S*n and convert to `dtype`
    x = dtype.(vcat(x...))
    y = dtype.(vcat(y...))

    # reshape to 3D batches
    x = reshape(x, (nx, S, B))
    y = reshape(y, (ny, S, B))

    return x, y, deltas, distrX, distrY
end


function t_estimates(x, y, pooled)
    # Compute t confidence intervals for each of the B*S pairs
    wide   = tconf(x, y, alpha=0.00001, pooled=pooled)
    narrow = tconf(x, y, alpha=0.4, pooled=pooled)
    # @show size(wide)

    _, nsamples, nbatches = size(x)
    wide   = reshape(wide, nsamples, nbatches)
    narrow = reshape(narrow, nsamples, nbatches)

    return wide, narrow
end
