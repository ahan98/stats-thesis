using CUDA
include("../utils.jl")

""" array-scalar """

function add!(out, x, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] + val
    end
    return
end

function sub!(out, x, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] - val
    end
    return
end

function mul!(out, x, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] * val
    end
    return
end

function div!(out, x, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] / val
    end
    return
end

""" array-scalar (in-place) """

function add!(out, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] += val
    end
    return
end

function sub!(out, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] -= val
    end
    return
end

function mul!(out, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] *= val
    end
    return
end

function div!(out, val)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] /= val
    end
    return
end

""" array-array """

function add_arr!(out, x, y)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] + y[i]
    end
    return
end

function sub_arr!(out, x, y)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] - y[i]
    end
    return
end

function mul_arr!(out, x, y)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] * y[i]
    end
    return
end

function div_arr!(out, x, y)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = x[i] / y[i]
    end
    return
end

""" array-array (in-place) """

function add_arr!(out, x)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] += x[i]
    end
    return
end

function sub_arr!(out, x)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] -= x[i]
    end
    return
end

function mul_arr!(out, x)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] *= x[i]
    end
    return
end

function div_arr!(out, x)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] /= x[i]
    end
    return
end

""" 2-D Matrix """

function row_sum!(out, x)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for r = tidx:stride:size(x,1)
        for c = 1:size(x,2)
            @inbounds out[r] += x[r,c]
        end
    end
    return
end

""" Element-wise (in-place) """

function sqrt!(out)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = sqrt(out[i])
    end
    return
end

function square!(out)
    """ x .+= val """
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # thread index
    stride = blockDim().x * gridDim().x                       # num. threads per block
    for i = tidx:stride:length(out)
        @inbounds out[i] = out[i]^2
    end
    return
end

