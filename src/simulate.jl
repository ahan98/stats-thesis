using FLoops
include("permutation.jl")
include("bootstrap.jl")
include("t.jl")

@enum Alternative smaller greater twoSided

function simulate_perm(x, y, deltas, distrX, distrY; alpha=0.05, mc_size=0, save_csv=true)
    """
    Run all settings of permutation interval and save results as .csv
    """

    px, py = (mc_size == 0) ? partition(nx, ny) : (nothing, nothing)
    permuter = (px, py, mc_size)

    for isTwoSided in [true, false]
        alpha_temp = alpha
        alt_lo = alt_hi = twoSided

        if !isTwoSided
            alpha_temp = alpha / 2
            alt_lo = greater
            alt_hi = smaller
        end

        for pooled in [true, false]
            # TODO remove Args struct since you abstracted the simulation code elsewhere
            args = Args(permuter, pooled, alpha_temp, alt_lo, alt_hi)
            results = simulate(x, y, deltas, distrX, distrY,
                               permInterval, args,
                               save_csv=save_csv)
        end
    end
end

function simulate_bootstrap(x, y, deltas, distrX, distrY;
                            alpha=0.05, nsamples=10_000, save_csv=true)
    """
    Run all settings for bootstrap confidence intervals.
    """
    for pooled in [true, false]
        results = simulate(x, y, deltas, distrX, distrY,
                           bootstrap, alpha, pooled, nsamples,
                           save_csv=save_csv)
    end
end

function simulate_t(x, y, deltas, distrX, distrY; alpha=0.05, save_csv=true)
    for pooled in [true, false]
        # TODO either change tconf to handle only 1-D arrays (so "unvectorize" it),
        #      OR, change code here to custom handle and not use simulate()
        results = simulate(x, y, deltas, distrX, distrY,
                           tconf, alpha, pooled,
                           save_csv=save_csv)
    end
end

function simulate(x, y, deltas, distrX, distrY, method, method_args...; save_csv=true)
    # simulates one boxplot
    # method_args are all the method-specific arguments you need to pass,
    # FOR EACH pair of samples (x[:,j,i], y[:,j,i])

    nsamples, nbatches = size(x, 2), size(x, 3)

    results = Vector{Any}(undef, nbatches)
    @floop ThreadedEx() for i = 1:nbatches
        @floop ThreadedEx() for j = 1:nsamples
            @inbounds lo, hi = method(x[:,j,i], y[:,j,i], method_args)
            @reduce() do (coverage = 0; lo), (width = 0.0f0; hi)
                coverage += (lo <= deltas[i] <= hi)
                width += (hi - lo)
            end
        end
        results[i] = (coverage / nsamples, width / nsamples)
    end

    # # TODO extend to other interval methods
    # if save_csv
    #     save(results, distrX, distrY, pooled, args.alpha, isTwoSided)
    # end

    return results
end
