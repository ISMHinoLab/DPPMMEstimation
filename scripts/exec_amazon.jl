using LinearAlgebra
using Distributions
using Random
using DataFrames
using CSV
using Query

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1) 
# load Amazon Baby Registry data
amazon_dir = joinpath("$(@__DIR__)", "..", "data", "AmazonBabyRegistry")
categories = ["apparel", "bath", "bedding", "carseats", "decor", "diaper",
              "feeding", "furniture", "gear", "gifts", "health", "media",
              "moms", "pottytrain", "safety", "strollers", "toys"]

df_result = DataFrame()
n_exp = 5 # number of experiments
for category in categories
    reg_name = "1_100_100_100_$(category)_regs.csv"
    txt_name = "1_100_100_100_$(category)_item_names.txt"

    samples = CSV.read(joinpath(amazon_dir, reg_name), DataFrame, header = 0) |>
        eachrow .|>
        Vector .|>
        skipmissing .|>
        collect

    outdir = joinpath("$(@__DIR__)", "..", "output", "amazon", category)
    mkpath(outdir)

    N = length(readlines(joinpath(amazon_dir, txt_name)))
    M = length(samples)

    Random.seed!(1234)
    results_amazon_wishart = map(1:n_exp) do i
        Linit = initializer(N, init = :wishart)
        outdir_i = joinpath(outdir, string(i))
        return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5))
    end
    df_result = vcat(df_result,
                     summarize_to_df(results_amazon_wishart,
                                     dict_cols = Dict(:category => Symbol(category))))
end

outdir = joinpath("$(@__DIR__)", "..", "output")
CSV.write(joinpath(outdir, "amazon_results.csv"), df_result)
