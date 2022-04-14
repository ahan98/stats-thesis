using StatsBase

# TODO add seed
function bootstrap(x, y, args, nx=length(x), ny=length(y))
    alpha, pooled, nsamples = args
    sample_stats = zeros(nsamples)
    for i in 1:nsamples
        x_bs = StatsBase.sample(x, nx)
        y_bs = StatsBase.sample(y, ny)
        @inbounds sample_stats[i] = t(x_bs, y_bs, pooled)[1]
    end
    sort!(sample_stats)
    quantileSize = ceil(Int, nsamples * alpha)
    return sample_stats[quantileSize], sample_stats[end-quantileSize+1]
end
