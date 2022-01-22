module PermTestFast
export pval, t

using CUDA
include("utils.jl")

function pval(x, y, px, py; pooled=false, alternative="two-sided", delta=0)
    x_shift = x .- delta
    t_obs = t(x_shift', y', pooled)  # test statistic for observed data

    combined = vcat(x_shift, y)  # join original pair into single vector
    @inbounds xs = combined[px]   # get all combinations of pairs from original pair
    @inbounds ys = combined[py]
    ts = t(xs, ys, pooled)   # test statistic for all possible pairs of samples

    if alternative == "smaller"
        n_extreme = count(ts .<= t_obs)
    elseif alternative == "larger"
        n_extreme = count(ts .>= t_obs)
    else
        n_extreme = count(@. (ts <= -abs(t_obs)) | (ts >= abs(t_obs)))
    end

    return n_extreme / size(px, 1)  # proportion of pairs w/ extreme test statistic
end

function t(x, y, pooled)
    varx, meanx = var_gpu(x)
    vary, meany = var_gpu(y)
    nx, ny = size(x, 2), size(y, 2)
    T, B = set_thread_block(size(x,1))
    
    if pooled
        # TODO
    end
    
    return @. (meanx - meany) / sqrt(varx/nx + vary/ny)
end


""" Variance """

function var_gpu(x)
    nrow, ncol = size(x)
    T, B = set_thread_block(nrow)
    ss = CUDA.zeros(Float64, size(x)[1])
    @cuda threads=T blocks=B sumsq!(x, ncol, ss)

    means = CUDA.zeros(Float64, nrow)
    @cuda threads=T blocks=B row_mean!(x, ncol, means)

    vars = CUDA.zeros(Float64, nrow)
    @cuda threads=T blocks=B _var!(ncol, ss, means, vars)
    return vars, means
end

function _var!(n, ss, means, out)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds out[i] = (ss[i] - (n * means[i]^2)) / (n-1)
    return
end

""" Mean """

function row_mean!(x, ncol, out)
    """out = sum(x, dims=2)"""
    row_idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    for i = 1:ncol
        @inbounds out[row_idx] += x[row_idx, i]
    end
    @inbounds out[row_idx] /= ncol
    return
end

""" Misc. """

function sumsq!(x, ncol, out)
    """out = sum(x, dims=2)"""
    row_idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    for i = 1:ncol
        @inbounds out[row_idx] += x[row_idx, i]^2
    end
    return
end

end  # module
