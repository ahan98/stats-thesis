using Combinatorics

function partition(n1, n2)
    N = n1 + n2
    a = collect(combinations(1:N, n1))
    b = reverse(collect(combinations(1:N, n2)))
    return [a b]
end
