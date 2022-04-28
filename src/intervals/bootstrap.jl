using StatsBase
using FLoops

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
    # sort!(sample_stats)
    tail_size = ceil(Int, nsamples * alpha / 2)
    # return sample_stats[tail_size + 1], sample_stats[end-tail_size]
    lower = findKthLargest(nums, tail_size + 1)
    upper = findKthLargest(nums, nsamples - tail_size)
    return lower, upper
end


function findKthLargest(nums, k)
    lo, hi = 1, length(nums)

    while true
        pivot_idx = rand(lo:hi)
        pivot_idx = partition(pivot_idx, lo, hi)

        if pivot_idx == k
            return nums[k]
        end

        if pivot_idx < k
            lo = pivot_idx + 1
        else
            hi = pivot_idx - 1
        end
    end
end


function partition(pivot_idx, lo, hi)
    pivot_val = nums[pivot_idx]

    nums[hi], nums[pivot_idx] = nums[pivot_idx], nums[hi]

    n_smaller = lo
    for i = lo:hi-1
        if nums[i] < pivot_val
            nums[i], nums[n_smaller] = nums[n_smaller], nums[i]
            n_smaller += 1
        end
    end

    nums[n_smaller], nums[hi] = nums[hi], nums[n_smaller]
    return n_smaller
end


# nums = [4, 3, 3, 3, 6, 3, 2, 1, 4, 5]
# @show k = 4
# @show findKthLargest(nums, k)
# @show sort(nums)[k]

