using FLoops
include("util.jl")
include("data.jl")
include("simulation.jl")

function main(nbatches, nsamples, nx, ny, distrTypeX, paramsX, distrTypeY, paramsY;
              mc_size=0, dtype=Float32, seed=123, save_csv=true)
    """
    Run all settings of simulation and save results as .csv
    """

    x, y, deltas, distrX, distrY = generateData(nbatches, nsamples, nx, ny,
                                                distrTypeX, paramsX, distrTypeY, paramsY,
                                                dtype, seed)

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
            args = Args(permuter, pooled, alpha_temp, alt_lo, alt_hi)

            results = Vector{Any}(undef, nbatches)

            @floop ThreadedEx() for i = 1:nbatches
                @floop ThreadedEx() for j = 1:nsamples
                    wide, narrow = t_estimates(x, y, pooled)
                    @inbounds a, b = permInterval(x[:,j,i], y[:,j,i], wide[j,i], narrow[j,i], deltas[i], args)
                    @reduce() do (coverage = 0; a), (width = 0.0f0; b)
                        coverage += a
                        width += b
                    end
                end
                results[i] = (coverage / nsamples, width / nsamples)
            end

            if save_csv
                save(results, distrX, distrY, pooled, args.alpha, isTwoSided)
            end
        end
    end
end

struct Args
    permuter::Tuple{Union{Matrix, Nothing}, Union{Matrix, Nothing}, Int}
    pooled::Bool
    alpha::Float32
    alt_lo::Alternative
    alt_hi::Alternative
end
