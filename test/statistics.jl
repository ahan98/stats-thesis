using CUDA, BenchmarkTools, Test, Random
include("../src/kernels/statistics.jl")
include("../src/perm_test.jl")
include("../src/utils.jl")

@testset "t test statistic" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)
    x = CUDA.rand(Float64, 12)
    y = CUDA.rand(Float64, 8)

    @testset "pooled=false" begin
        cpu = PermTest.ttest_ind(x, y, false)
        gpu = PermTestCUDA.t(x', y', false)
        @test isapprox(cpu, gpu)
    end
end

# @testset "Permutation test p-value" begin
#     nx, ny = 12, 8
#     px, py = partition(nx, ny)
#     x = CUDA.rand(Float64, 12)
#     y = CUDA.rand(Float64, 8)
# 
#     cpu = PermTest.pval(x, y, px, py)
#     gpu = PermCUDA.pval(x, y, px, py)
#     @test isapprox(cpu, gpu)
# end
