using StatsBase
using FLoops

function bootstrap(x, y, delta, alpha; nsamples=2000)
    lo, hi = bootstrap(x, y, alpha, nsamples)
    return lo <= delta <= hi, hi - lo
end

function bootstrap(x, y, alpha, nsamples)
    stats = zeros(nsamples)
    T = Threads.nthreads()
    nx, ny = length(x), length(y)
    @floop ThreadedEx(basesize=div(nsamples, T)) for i in 1:nsamples
        @views xx = sample(x, nx, replace=true)
        @views yy = sample(y, ny, replace=true)
        @inbounds stats[i] = mean(xx) - mean(yy)
    end
    
    tail_size = ceil(Int, nsamples * alpha / 2)
    
    # sort!(sample_stats)
    # lower = sample_stats[tail_size + 1]
    # upper = sample_stats[end-tail_size]
    
    lower = findKthLargest(stats, tail_size + 1)
    upper = findKthLargest(stats, nsamples - tail_size)
    return lower, upper
end


function findKthLargest(nums, k)
    lo, hi = 1, length(nums)

    while true
        pivot_idx = rand(lo:hi)
        pivot_idx = partition(nums, pivot_idx, lo, hi)

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


function partition(nums, pivot_idx, lo, hi)
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

