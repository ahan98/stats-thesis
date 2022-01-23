using CUDA, BenchmarkTools, Test, Random, Statistics
include("../src/kernels/statistics.jl")
include("../src/perm_test.jl")
include("../src/utils.jl")

DATA_TYPE = Float32

@testset "Permutation test (1-α)% CI" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)

    Random.seed!(1)
    x = CuArray(randn(DATA_TYPE, nx))  # standard normal data with mean 0
    y = CuArray(randn(DATA_TYPE, ny))

    wide = PermTest.tconf(x, y, alpha=0.001)
    narrow = PermTest.tconf(x, y, alpha=0.2)
    # println("lo: ", wide_lo, ", hi: ", narrow_lo)

    @testset "search()" begin
        @test isapprox(PermTest.search(x, y, px, py, wide[1], narrow[1], pooled=false),
                       PermTestCUDA.search(x, y, px, py, wide[1], narrow[1], pooled=false))
    end
    
    @testset "permInterval()" begin
        @test PermTest.permInterval(x, y, px, py, 0) == PermTestCUDA.permInterval(x, y, px, py, 0, wide, narrow)
    end
end

@testset "Permutation test p-value" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)

    x = CUDA.rand(DATA_TYPE, nx)
    y = CUDA.rand(DATA_TYPE, ny)

    x_copy = copy(x)
    y_copy = copy(y)

    @test all(x .== x_copy)
    @test all(y .== y_copy)
    @test isapprox(PermTest.pval(x, y, px, py),
                   PermTestCUDA.pval(x, y, px, py))
end

@testset "t test statistic" begin
    nx, ny = 12, 8
    px, py = Utils.partition(nx, ny)
    x = CUDA.rand(DATA_TYPE, nx)
    y = CUDA.rand(DATA_TYPE, ny)

    x_copy = copy(x)
    y_copy = copy(y)

    @testset "pooled=false" begin
        @test isapprox(PermTest.ttest_ind(x, y, false),
                       PermTestCUDA.t(x', y', false))
        @test all(x .== x_copy)
        @test all(y .== y_copy)
    end

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

