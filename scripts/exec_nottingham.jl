using LinearAlgebra
using Distributions
using Random
using MAT
using CSV

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)
# load Nottingham data
nottingham = matread("../data/Nottingham.mat")
train_samples = vcat([[findall(isone, traindata[j, :]) for j in 1:size(traindata, 1)]
                      for traindata in nottingham["traindata"]]...)

train_nottingham = reshape([[findall(isone, traindata[j, :]) for j in 1:size(traindata, 1)]
                            for traindata in nottingham["traindata"]], :)

n_exp = 5 # number of experiments

## WISHART initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "nottingham", "wishart")
results_nottingham_wishart = map(1:n_exp) do i
    N = 88
    n_tracks = 25

    samples = vcat(sample(train_nottingham, n_tracks, replace = false)...)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5))
end

Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "nottingham", "basic")
results_nottingham_basic = map(1:n_exp) do i
    N = 88
    n_tracks = 25

    samples = vcat(sample(train_nottingham, n_tracks, replace = false)...)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5))
end

dict_wishart = Dict(:init => :wishart)
dict_basic = Dict(:init => :basic)
df_nottingham_wishart = summarize_to_df(results_nottingham_wishart, dict_cols = dict_wishart)
df_nottingham_basic = summarize_to_df(results_nottingham_basic, dict_cols = dict_basic)
outdir = joinpath("$(@__DIR__)", "..", "output")
CSV.write(joinpath(outdir, "nottingham_results.csv"), vcat(df_nottingham_wishart, df_nottingham_basic))


results_nottingham_basic = [load(outdir * "/nottingham/basic/$(i)/results.jld2") for i in 1:5]
df_nottingham_basic = summarize_to_df(results_nottingham_basic, dict_cols = dict_basic, str_keys = true)
results_nottingham_wishart = [load(outdir * "/nottingham/wishart/$(i)/results.jld2") for i in 1:5]
df_nottingham_wishart = summarize_to_df(results_nottingham_wishart, dict_cols = dict_basic, str_keys = true)
