using CUDA

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

