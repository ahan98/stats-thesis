module Data

export generateData

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
    x = rand.(distrX, S * nx)  # size (B * S * nx,)
    y = rand.(distrY, S * ny)

    # flatten into (B, S * n)
    x = hcat(x...)'
    y = hcat(y...)'

    # reshape to 3D batch
    x = dtype.(reshape(x, (B, S, nx)))
    y = dtype.(reshape(y, (B, S, ny)))

    # Compute t confidence intervals for each of the (B x S x n) pairs
    wide   = tconf(x, y, alpha=0.00001, pooled=pooled)
    narrow = tconf(x, y, alpha=0.4, pooled=pooled)

    wide   = reshape(wide, B, S)
    narrow = reshape(narrow, B, S)

    return x, y, wide, narrow, deltas, distrX, distrY
end

end
