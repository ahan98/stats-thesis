using Combinatorics
using DataFrames, CSV
using StatsBase
using Distributions, Random


function random(D, n, dtype=Float32, digits=3)
    return dtype.(round.(rand(D, n), digits=digits))
end


function partition(n1, n2, mc=0)
    N = n1 + n2
    if mc > 0
        idxs = hcat([sample(1:N, N, replace=false) for _ in 1:mc]...)
        a = idxs[1:n1, :]
        b = idxs[n1+1:end, :]
    else
        a = hcat(combinations(1:N, n1)...)
        b = reverse(hcat(combinations(1:N, n2)...))
    end
    return a, b
end


function save(results, distrX, distrY, alpha, pooled=nothing, isTwoSided=nothing; prefix="", dir="./")
    # convert results to DataFrame
    probs  = [i for (i, _) in results]
    widths = [j for (_, j) in results]
    df = DataFrame(prob=probs, width=widths, distrX=distrX, distrY=distrY)

    # save DataFrame as .csv
    filename = prefix * (length(prefix) > 0 ? "_" : "")

    if !isnothing(isTwoSided)
        filename *= (isTwoSided ? "two" : "one") * "Sided_"
    end

    if !isnothing(pooled)
        filename *= (pooled ? "" : "un") * "pooled_"
    end
    
    filename *= string(alpha) * ".csv"

    @show dir * filename
    if !isdir(dir)
        mkpath(dir)
    end
    CSV.write(dir * filename, df)
end
