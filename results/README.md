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

2. Normal vs. Normal (symmetric; Normal; equal mean; unequal variance)

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

3. Gamma vs Gumbel (not symmetric; not Normal; unequal mean; unequal variance)

```julia
Random.seed!(123)

distrTypeX = Gamma{dtype}
X_shape = random(Uniform(2, 4), B)
X_scale = random(Uniform(0.5, 2), B)
distrX = map(distrTypeX, X_shape, X_scale)

distrTypeY = Gumbel{dtype}
Y_loc = random(Uniform(0, 1), B)
Y_scale = random(Uniform(2, 4), B)
distrY = map(distrTypeY, Y_loc, Y_scale)
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

