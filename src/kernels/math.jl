using CUDA

""" array-scalar """

function add!(x, val)
    """ x .+= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] += val
    return
end

function add!(out, x, val)
    """ x .+= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] + val
    return
end

function sub!(x, val)
    """ x .-= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] -= val
    return
end

function sub!(out, x, val)
    """ x .+= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] - val
    return
end

function mul!(x, val)
    """ x .*= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] *= val
    return
end

function mul!(out, x, val)
    """ x .*= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] * val
    return
end

function div!(x, val)
    """ x ./= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] /= val
    return
end

function div!(out, x, val)
    """ x ./= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] / val
    return
end

""" In-place array-array functions """

function add_arr!(x, y)
    """ x .+= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] += y[i]
    return
end

function add_arr!(out, x, y)
    """ x .+= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] + y[i]
    return
end

function sub_arr!(x, y)
    """ x .-= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] -= y[i]
    return
end

function sub_arr!(out, x, y)
    """ x .-= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] - y[i]
    return
end

function mul_arr!(x, y)
    """ x .*= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] *= y[i]
    return
end

function mul_arr!(out, x, y)
    """ x .*= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] * y[i]
    return
end

function div_arr!(x, y)
    """ x ./= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] /= y[i]
    return
end

function div_arr!(out, x, y)
    """ x ./= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds out[i] = x[i] / y[i]
    return
end


""" In-place array functions """

function sqrt!(x)
    """ x .*= val """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds x[i] = sqrt(x[i])
    return
end

