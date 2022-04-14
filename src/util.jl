using Combinatorics
using DataFrames, CSV

function partition(n1, n2)
    N = n1 + n2
    a = hcat(combinations(1:N, n1)...)
    b = hcat(combinations(1:N, n2)...)
    return a, reverse(b)
end


function save(results, distrX, distrY, alpha, pooled=nothing, isTwoSided=nothing; prefix="", parent_dir="")
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
        filename *= (pooled ? "" : "un") * "pooled_" * string(alpha) * ".csv"
    end

    if !isdir(parent_dir)
        mkdir(parent_dir)
    end
    CSV.write(parent_dir * filename, df)
end
