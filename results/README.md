1. LogNormal vs. Gamma (not symmetric; not exchangeable; not Normal; equal mean/variance)

```julia
Random.seed!(123)

distrTypeX = LogNormal{dtype}
X_mu = random(Uniform(0, 1), B)
X_sigma = random(Uniform(0, 0.6), B)
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Gamma{dtype}
Y_shape = @. 1 / (exp(X_sigma^2) - 1)
Y_scale = @. exp(X_mu + X_sigma^2 / 2) / Y_shape
distrY = map(distrTypeY, Y_shape, Y_scale)
```

2. Normal vs. Normal (symmetric; Normal; not exchangeable; unequal variance)

```julia
Random.seed!(123)

distrTypeX = Normal{dtype}
X_mu = random(Uniform(-1, 1), B)
X_sigma = random(Uniform(0.5, 1), B)
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Normal{dtype}
Y_mu = X_mu
Y_sigma = random(Uniform(1.5, 2), B)
distrY = map(distrTypeY, Y_mu, Y_sigma)
```

3. Log-Normal vs Exponential (not symmetric; not Normal; unequal means; unequal variance)

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

4. Laplace vs. Laplace (symmetric; exchangeable; not Normal)

```julia
Random.seed!(123)

distrTypeX = Laplace{dtype}
X_mu = random(Uniform(0, 1), B)
X_sigma = random(Uniform(2, 4), B)
distrX = map(distrTypeX, X_mu, X_sigma)

distrTypeY = Laplace{dtype}
Y_mu = X_mu .+ random(Uniform(-5, 5), B)
Y_sigma = X_sigma
distrY = map(distrTypeY, Y_mu, Y_sigma)
```

