LogNormal vs. Gamma w/ equal mean and variance

```julia
distrTypeX = LogNormal{dtype}
X_mu = dtype.(round.(rand(B), digits=3))
X_sigma = dtype.(round.(rand(B) * 0.6, digits=3))
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Gamma{dtype}
Y_shape = @. 1 / (exp(X_sigma^2) - 1)
Y_scale = @. exp(X_mu + X_sigma^2 / 2) / Y_shape
distrY = map(distrTypeY, Y_shape, Y_scale)
```