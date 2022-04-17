Exponential vs. Log-Normal

Compare two totally dissimilar distributions (only thing they have in common is support on positive reals).

```julia
Random.seed!(123)

distrTypeX = LogNormal{dtype}
X_mu = dtype.(round.(rand(B), digits=3))
X_sigma = dtype.(round.(rand(B) * 0.6, digits=3))
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Exponential{dtype}
Y_inv_scale = dtype.(round.(rand(B), digits = 3))
distrY = map(distrTypeY, Y_inv_scale)
```