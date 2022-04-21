using StatsBase
using FLoops

# TODO add seed
function bootstrap(x, y, delta, alpha, pooled=false; nsamples=10_000)
    lo, hi = bootstrap(x, y, alpha, pooled, nsamples)
    return lo <= delta <= hi, hi - lo
end

function bootstrap(x, y, alpha, pooled, nsamples, nx=length(x), ny=length(y))
    sample_stats = zeros(nsamples)
    T = Threads.nthreads()
    @floop ThreadedEx(basesize=div(nsamples, T)) for i in 1:nsamples
        x_bs = StatsBase.sample(x, nx)
        y_bs = StatsBase.sample(y, ny)
        @inbounds sample_stats[i] = mean(x_bs) - mean(y_bs)#t(x_bs, y_bs, pooled)[1]
    end
    sort!(sample_stats)
    tail_size = ceil(Int, nsamples * alpha / 2)
    return sample_stats[tail_size + 1], sample_stats[end-tail_size]
end
