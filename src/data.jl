using Distributions

include("statistics.jl")
using .TestStatistics

function generateData(B, S, nx, ny, pooled, distrTypeX, paramsX, distrTypeY, paramsY, dtype=Float32)
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
    x = rand.(distrX, S * nx)  # B vectors of length S * nx
    y = rand.(distrY, S * ny)

    # reshape into single vector of length B*S*n and convert to `dtype`
    x = dtype.(vcat(x...))
    y = dtype.(vcat(y...))
    # @show size(x)

    # reshape to 3D batches
    x = reshape(x, (nx, S, B))
    y = reshape(y, (ny, S, B))

    # Compute t confidence intervals for each of the (B x S x n) pairs
    wide   = tconf(x, y, alpha=0.00001, pooled=pooled)
    narrow = tconf(x, y, alpha=0.4, pooled=pooled)
    # @show size(wide)

    wide   = reshape(wide, S, B)
    narrow = reshape(narrow, S, B)

    return x, y, wide, narrow, deltas, distrX, distrY
end
