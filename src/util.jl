using Combinatorics

function partition(n1, n2)
    N = n1 + n2
    a = hcat(combinations(1:N, n1)...)
    b = hcat(combinations(1:N, n2)...)
    return a, reverse(b)
end
