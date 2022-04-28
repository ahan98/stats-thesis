Compares two identical Normals except for a shift in mean.

```julia
distrTypeX = Normal{dtype}
X_mu = dtype.(round.(rand(B), digits=3))
X_sigma = dtype.(round.(rand(B), digits=3))
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Normal{dtype}
Y_mu = dtype.(X_mu .+ ((round.(rand(B), digits=3) * 4) .- 2))
Y_sigma = X_sigma
distrY = map(distrTypeY, Y_mu, Y_sigma)
```