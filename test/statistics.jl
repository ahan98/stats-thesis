using CUDA, BenchmarkTools, Test, Random, Statistics
include("../src/kernels/statistics.jl")
include("../src/perm_test.jl")
include("../src/utils.jl")

DATA_TYPE = Float32

@testset "Permutation test (1-α)% CI" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)

    Random.seed!(1)
    x = CuArray(rand(DATA_TYPE, nx))
    y = CuArray(rand(DATA_TYPE, ny))

    wide_lo, wide_hi = PermTest.tconf(x, y, alpha=0.001)
    narrow_lo, narrow_hi = PermTest.tconf(x, y, alpha=0.2)

    # println("lo: ", wide_lo, ", hi: ", narrow_lo)
    # println(PermTestCUDA.search(x, y, px, py, wide_lo, narrow_lo, pooled=false))
    @test isapprox(PermTest.search(x, y, px, py, wide_lo, narrow_lo, pooled=false),
                   PermTestCUDA.search(x, y, px, py, wide_lo, narrow_lo, pooled=false))
end

@testset "Permutation test p-value" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)

    x = CUDA.rand(DATA_TYPE, nx)
    y = CUDA.rand(DATA_TYPE, ny)

    x_copy = copy(x)
    y_copy = copy(y)

    p1 = PermTestCUDA.pval(x, y, px, py)
    @test all(x .== x_copy)
    @test all(y .== y_copy)

    # @test isapprox(PermTest.pval(x, y, px, py),
    #                PermTestCUDA.pval(x, y, px, py))
end

@testset "t test statistic" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)
    x = CUDA.rand(DATA_TYPE, nx)
    y = CUDA.rand(DATA_TYPE, ny)

    x_copy = copy(x)
    y_copy = copy(y)

    @testset "pooled=false" begin
        cpu = PermTest.ttest_ind(x, y, false)
        gpu = PermTestCUDA.t(x', y', false)
        @test isapprox(cpu, gpu)
    end

    @test all(x .== x_copy)
    @test all(y .== y_copy)

    # @testset "pooled=true" begin
    #     # TODO
    # end
end

@testset "Variance & mean" begin
    x = CUDA.rand(DATA_TYPE, 1000, 500)
    x_copy = copy(x)

    target = (var(x, dims=2), mean(x, dims=2))
    v, m = PermTestCUDA.var(x)
    @test isapprox(v, target[1])
    @test isapprox(m, target[2])

    @test all(x .== x_copy)
end
