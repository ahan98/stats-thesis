Compares two identical Normals except for a shift in mean.

```julia
X ~ Normal(mu, 2 * mu)
Y ~ Normal(5 + mu, 2 * mu)
mu ~ Uniform(0, 1)
```