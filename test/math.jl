using CUDA, Test
include("../src/kernels/math.jl")

@testset "In-place array-scalar" begin
    x = CUDA.rand(Float64, 256*4)
    val = rand(Float64)
    
    target = copy(x)
    target .-= val
    target ./= val
    target .*= val
    target .+= val
    target = sqrt.(target)
    
    @cuda threads=256 blocks=4 sub!(x, val)
    @cuda threads=256 blocks=4 div!(x, val)
    @cuda threads=256 blocks=4 mul!(x, val)
    @cuda threads=256 blocks=4 add!(x, val)
    @cuda threads=256 blocks=4 sqrt!(x)
    
    @test all(target .== x)
end

@testset "In-place array-array" begin
    x = CUDA.rand(Float64, 256*4)
    y = CUDA.rand(Float64, 256*4)
    
    target = copy(x)
    target .-= y
    target ./= y
    target .*= y
    target .+= y

    @cuda threads=256 blocks=4 sub_arr!(x, y)
    @cuda threads=256 blocks=4 div_arr!(x, y)
    @cuda threads=256 blocks=4 mul_arr!(x, y)
    @cuda threads=256 blocks=4 add_arr!(x, y)
    
    @test all(target .== x)
end
