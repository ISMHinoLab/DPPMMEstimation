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

n_exp = 30 # number of experiments

## WISHART initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "nottingham", "wishart", "default")
results_nottingham_wishart_default = map(1:n_exp) do i
    N = 88
    n_tracks = 25

    samples = vcat(sample(train_nottingham, n_tracks, replace = false)...)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5))
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "nottingham", "wishart", "accelerated")
results_nottingham_wishart_accelerated = map(1:n_exp) do i
    N = 88
    n_tracks = 25

    samples = vcat(sample(train_nottingham, n_tracks, replace = false)...)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5),
                        η = 0.1, ρ = 1.3, accelerate_steps = 5)
end


## BASIC initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "nottingham", "basic", "default")
results_nottingham_basic_default = map(1:n_exp) do i
    N = 88
    n_tracks = 25

    samples = vcat(sample(train_nottingham, n_tracks, replace = false)...)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5))
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "nottingham", "basic", "accelerated")
results_nottingham_basic_accelerated = map(1:n_exp) do i
    N = 88
    n_tracks = 25

    samples = vcat(sample(train_nottingham, n_tracks, replace = false)...)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5),
                        η = 0.01, ρ = 1.3, accelerate_steps = 10)
end


dict_wishart = Dict(:init => :wishart)
dict_basic = Dict(:init => :basic)
dict_default = Dict(:params => :default)
dict_accelerated = Dict(:params => :accelerated)
df_nottingham_wishart_default = summarize_to_df(results_nottingham_wishart_default,
                                                dict_cols = merge(dict_wishart, dict_default))
df_nottingham_wishart_accelerated = summarize_to_df(results_nottingham_wishart_accelerated,
                                                    dict_cols = merge(dict_wishart, dict_accelerated))
df_nottingham_basic_default = summarize_to_df(results_nottingham_basic_default,
                                              dict_cols = merge(dict_basic, dict_default))
df_nottingham_basic_accelerated = summarize_to_df(results_nottingham_basic_accelerated,
                                                  dict_cols = merge(dict_basic, dict_accelerated))
outdir = joinpath("$(@__DIR__)", "..", "output")
CSV.write(joinpath(outdir, "nottingham_results.csv"),
          vcat(df_nottingham_wishart_default, df_nottingham_wishart_accelerated,
               df_nottingham_basic_default, df_nottingham_basic_accelerated))


#results_nottingham_basic = [load(outdir * "/nottingham/basic/$(i)/results.jld2") for i in 1:n_exp]
#df_nottingham_basic = summarize_to_df(results_nottingham_basic, dict_cols = dict_basic, str_keys = true)
#results_nottingham_wishart = [load(outdir * "/nottingham/wishart/$(i)/results.jld2") for i in 1:n_exp]
#df_nottingham_wishart = summarize_to_df(results_nottingham_wishart, dict_cols = dict_wishart, str_keys = true)
