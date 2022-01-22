using CUDA, BenchmarkTools, Test, Random, Statistics
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

    # @testset "pooled=true" begin
    #     # TODO
    # end
end

@testset "variance & mean" begin
    x = CUDA.rand(Float64, 1000, 500)
    target = (var(x, dims=2), mean(x, dims=2))
    v, m = PermTestCUDA.var(x)
    @test isapprox(v, target[1])
    @test isapprox(m, target[2])
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
