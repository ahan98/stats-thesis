module PermTestCUDA

export pval, t, var

using CUDA
include("../utils.jl")
include("math.jl")

function pval(x, y, px, py; pooled=false, alternative="two-sided", delta=0)
    T, B = Utils.set_thread_block(size(x,1))
    @cuda threads=T blocks=B sub!(x, delta)
    
    t_obs = t(x', y', pooled)  # test statistic for observed data

    combined = vcat(x, y)  # join original pair into single vector
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
    varx, meanx = var(x)
    vary, meany = var(y)
    nx, ny = size(x, 2), size(y, 2)
    T, B = Utils.set_thread_block(size(x,1))
    
    if pooled
        # TODO
    end

    # @. (meanx - meany) / sqrt(varx/nx + vary/ny)
    @cuda threads=T blocks=B div!(varx, nx)
    @cuda threads=T blocks=B div!(vary, ny)
    @cuda threads=T blocks=B add_arr!(varx, vary)
    @cuda threads=T blocks=B sqrt!(varx)
    @cuda threads=T blocks=B sub_arr!(meanx, meany)
    @cuda threads=T blocks=B div_arr!(meanx, varx)
    return meanx
end


function var(x)
    nsamples, sample_size = size(x)
    T, B = Utils.set_thread_block(nsamples)

    # (sum(x.^2, dims=2) .- (sample_size .* means.^2)) / (sample_size - 1)
    means = mean(x)
    temp = copy(means)
    @cuda threads=T blocks=B square!(temp)
    @cuda threads=T blocks=B mul!(temp, sample_size)

    ss = _row_sum_sq(x)
    @cuda threads=T blocks=B sub_arr!(ss, temp)
    @cuda threads=T blocks=B div!(ss, sample_size - 1)  

    return ss, means
end


function mean(x)
    """out = sum(x, dims=2)"""
    T, B = Utils.set_thread_block(size(x, 1))
    out = CUDA.zeros(size(x,1))
    @cuda threads=T blocks=B row_sum!(out, x)       # sum across each row of x, and store in out
    @cuda threads=T blocks=B div!(out, size(x, 2))  # divide each row sum by row length
    return out
end


function _row_sum_sq(x)
    out = CUDA.zeros(size(x,1))
    temp = copy(x)

    T, B = Utils.set_thread_block(length(x))
    @cuda threads=T blocks=B square!(temp)        # square each element

    T, B = Utils.set_thread_block(size(x, 1))
    @cuda threads=T blocks=B row_sum!(out, temp)  # sum across each row

    return out
end

end
