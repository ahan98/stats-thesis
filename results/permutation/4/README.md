Compare two Normals which are identical except one has much larger spread.

```julia
distrTypeX = Normal{dtype}
X_mu = dtype.(round.(rand(B), digits=3))
X_sigma = dtype.(round.(rand(B), digits=3))
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Normal{dtype}
Y_mu = X_mu
Y_sigma = dtype.(rand(Uniform(2, 5), B)) .* X_sigma
distrY = map(distrTypeY, Y_mu, Y_sigma)
```