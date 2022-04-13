include("data.jl")

function main(nbatches, nsamples, nx, ny, distrTypeX, paramsX, distrTypeY, paramsY;
              mc_size=0, dtype=Float32, seed=123, save_csv=true)
    """
    Run all settings of simulation and save results as .csv
    """

    x, y, deltas, distrX, distrY = generateData(nbatches, nsamples, nx, ny,
                                                distrTypeX, paramsX, distrTypeY, paramsY,
                                                dtype, seed)
end
