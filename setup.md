## Installing Julia

1. Copy stable release link for your OS (https://julialang.org/downloads/#current_stable_release)
2. wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.1-linux-x86_64.tar.gz
3. tar -xvzf julia-1.7.1-linux-x86_64.tar.gz
4. sudo mv julia-1.7.1/ /opt/
5. sudo ln -s /opt/julia-1.7.1/bin/julia /usr/local/bin/julia

## Packages

```
pkg> add Folds, Distributions, HypothesisTests, Combinatorics, CUDA, FLoops, IJulia, DataFrames, CSV, BenchmarkTools
```

## Multithreaded Kernel

```
using IJulia
installkernel("Julia (8 threads)", env=Dict("JULIA_NUM_THREADS"=>"8"))
```
