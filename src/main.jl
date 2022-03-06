using Random
using Distributions
using DataFrames, CSV
using FLoops

# compile local files
include("statistics.jl")
include("simulation.jl")
include("data.jl")
include("util.jl")

using .TestStatistics
using .Simulation
using .Data

"""
CONFIG
"""
alpha = 0.05

# data
B  = 10    # num. coverage probabilities per boxplot
S  = 1000   # num. samples per coverage probability
nx = 6      # size of group 1
ny = 6      # size of group 2
px, py = partition(nx, ny)
dtype = Float32

# distribution settings
Random.seed!(123)

distrTypeX = Normal
muX = dtype.(rand(Uniform(0, 1), B))
sdX = 2 * muX
paramsX = (muX, sdX)

distrTypeY = Normal
paramsY = paramsX

# Random.seed!(123)
x, y, wide, narrow, deltas, distrX, distrY = generateData(B, S, nx, ny, true,
                                                          distrTypeX, paramsX,
                                                          distrTypeY, paramsY)
@show x[1:5]
