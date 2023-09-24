using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using CSV

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)

n_exp = 30 # number of experiments

# toy1: N = 32, M = 2500
## WISHART initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy1", "wishart", "default")
results_toy1_wishart_default = map(1:n_exp) do i
    N = 32
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L)
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy1", "wishart", "accelerated")
results_toy1_wishart_accelerated = map(1:n_exp) do i
    N = 32
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L,
                        η = 0.1, ρ = 1.3, accelerate_steps = 5)
end

## BASIC initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy1", "basic", "default")
results_toy1_basic_default = map(1:n_exp) do i
    N = 32
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L)
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy1", "basic", "accelerated")
results_toy1_basic_accelerated = map(1:n_exp) do i
    N = 32
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L,
                        η = 0.01, ρ = 1.3, accelerate_steps = 10)
end


# toy2: N = 32, M = 2500
## WISHART initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy2", "wishart", "default")
results_toy2_wishart_default = map(1:n_exp) do i
    N = 32
    M = 10000

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L)
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy2", "wishart", "accelerated")
results_toy2_wishart_accelerated = map(1:n_exp) do i
    N = 32
    M = 10000

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L,
                        η = 0.1, ρ = 1.3, accelerate_steps = 5)
end

## BASIC initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy2", "basic", "default")
results_toy2_basic_default = map(1:n_exp) do i
    N = 32
    M = 10000

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L)
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy2", "basic", "accelerated")
results_toy2_basic_accelerated = map(1:n_exp) do i
    N = 32
    M = 10000

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L,
                        η = 0.01, ρ = 1.3, accelerate_steps = 10)
end


# toy3: N = 128, M = 2500
## WISHART initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy3", "wishart", "default")
results_toy3_wishart_default = map(1:n_exp) do i
    N = 128
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L)
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy3", "wishart", "accelerated")
results_toy3_wishart_accelerated = map(1:n_exp) do i
    N = 128
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :wishart)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L,
                        η = 0.1, ρ = 1.3, accelerate_steps = 5)
end

## BASIC initialization
### Default hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy3", "basic", "default")
results_toy3_basic_default = map(1:n_exp) do i
    N = 128
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L)
end

### Accelerated hyperparameters
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy3", "basic", "accelerated")
results_toy3_basic_accelerated = map(1:n_exp) do i
    N = 128
    M = 2500

    V = rand(Uniform(0, 10), (N, round(Int, N))) / N
    L = V * V'

    dpp = DeterminantalPointProcess(L)
    samples = rand(dpp, M)

    Linit = initializer(N, init = :basic)
    outdir_i = joinpath(outdir, string(i))
    return experimenter(Linit, samples, outdir = outdir_i, Ltruth = L,
                        η = 0.01, ρ = 1.3, accelerate_steps = 10)
end


dict_wishart = Dict(:init => :wishart)
dict_basic = Dict(:init => :basic)
dict_default = Dict(:params => :default)
dict_accelerated = Dict(:params => :accelerated)
df_toy1_wishart_default = summarize_to_df(results_toy1_wishart_default,
                                          dict_cols = merge(dict_wishart, dict_default))
df_toy1_wishart_accelerated = summarize_to_df(results_toy1_wishart_accelerated,
                                              dict_cols = merge(dict_wishart, dict_accelerated))
df_toy1_basic_default = summarize_to_df(results_toy1_wishart_default,
                                        dict_cols = merge(dict_basic, dict_default))
df_toy1_basic_accelerated = summarize_to_df(results_toy1_wishart_accelerated,
                                              dict_cols = merge(dict_basic, dict_accelerated))
df_toy2_wishart_default = summarize_to_df(results_toy2_wishart_default,
                                          dict_cols = merge(dict_wishart, dict_default))
df_toy2_wishart_accelerated = summarize_to_df(results_toy2_wishart_accelerated,
                                              dict_cols = merge(dict_wishart, dict_accelerated))
df_toy2_basic_default = summarize_to_df(results_toy2_wishart_default,
                                        dict_cols = merge(dict_basic, dict_default))
df_toy2_basic_accelerated = summarize_to_df(results_toy2_wishart_accelerated,
                                              dict_cols = merge(dict_basic, dict_accelerated))
df_toy3_wishart_default = summarize_to_df(results_toy3_wishart_default,
                                          dict_cols = merge(dict_wishart, dict_default))
df_toy3_wishart_accelerated = summarize_to_df(results_toy3_wishart_accelerated,
                                              dict_cols = merge(dict_wishart, dict_accelerated))
df_toy3_basic_default = summarize_to_df(results_toy3_wishart_default,
                                        dict_cols = merge(dict_basic, dict_default))
df_toy3_basic_accelerated = summarize_to_df(results_toy3_wishart_accelerated,
                                              dict_cols = merge(dict_basic, dict_accelerated))

df_toy1 = vcat(df_toy1_wishart_default, df_toy1_wishart_accelerated,
               df_toy1_basic_default, df_toy1_basic_accelerated)
df_toy1[:, :setting] .= 1
df_toy2 = vcat(df_toy2_wishart_default, df_toy2_wishart_accelerated,
               df_toy2_basic_default, df_toy2_basic_accelerated)
df_toy2[:, :setting] .= 2
df_toy3 = vcat(df_toy3_wishart_default, df_toy3_wishart_accelerated,
               df_toy3_basic_default, df_toy3_basic_accelerated)
df_toy3[:, :setting] .= 3
outdir = joinpath("$(@__DIR__)", "..", "output")
CSV.write(joinpath(outdir, "toy_results.csv"), vcat(df_toy1, df_toy2, df_toy3))


## load results
#results_toy1_wishart_default = [load(outdir * "/toy1/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
#results_toy1_wishart_accelerated = [load(outdir * "/toy1/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
#results_toy1_basic_default = [load(outdir * "/toy1/basic/default/$(i)/results.jld2") for i in 1:n_exp]
#results_toy1_basic_accelerated = [load(outdir * "/toy1/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]
#results_toy2_wishart_default = [load(outdir * "/toy2/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
#results_toy2_wishart_accelerated = [load(outdir * "/toy2/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
#results_toy2_basic_default = [load(outdir * "/toy2/basic/default/$(i)/results.jld2") for i in 1:n_exp]
#results_toy2_basic_accelerated = [load(outdir * "/toy2/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]
#results_toy3_wishart_default = [load(outdir * "/toy3/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
#results_toy3_wishart_accelerated = [load(outdir * "/toy3/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
#results_toy3_basic_default = [load(outdir * "/toy3/basic/default/$(i)/results.jld2") for i in 1:n_exp]
#results_toy3_basic_accelerated = [load(outdir * "/toy3/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]
#
#df_toy1_wishart_default = summarize_to_df(results_toy1_wishart_default, str_keys = true,
#                                          dict_cols = merge(dict_wishart, dict_default))
#df_toy1_wishart_accelerated = summarize_to_df(results_toy1_wishart_accelerated, str_keys = true,
#                                              dict_cols = merge(dict_wishart, dict_accelerated))
#df_toy1_basic_default = summarize_to_df(results_toy1_basic_default, str_keys = true,
#                                        dict_cols = merge(dict_basic, dict_default))
#df_toy1_basic_accelerated = summarize_to_df(results_toy1_basic_accelerated, str_keys = true,
#                                              dict_cols = merge(dict_basic, dict_accelerated))
#df_toy2_wishart_default = summarize_to_df(results_toy2_wishart_default, str_keys = true,
#                                          dict_cols = merge(dict_wishart, dict_default))
#df_toy2_wishart_accelerated = summarize_to_df(results_toy2_wishart_accelerated, str_keys = true,
#                                              dict_cols = merge(dict_wishart, dict_accelerated))
#df_toy2_basic_default = summarize_to_df(results_toy2_basic_default, str_keys = true,
#                                        dict_cols = merge(dict_basic, dict_default))
#df_toy2_basic_accelerated = summarize_to_df(results_toy2_basic_accelerated, str_keys = true,
#                                              dict_cols = merge(dict_basic, dict_accelerated))
#df_toy3_wishart_default = summarize_to_df(results_toy3_wishart_default, str_keys = true,
#                                          dict_cols = merge(dict_wishart, dict_default))
#df_toy3_wishart_accelerated = summarize_to_df(results_toy3_wishart_accelerated, str_keys = true,
#                                              dict_cols = merge(dict_wishart, dict_accelerated))
#df_toy3_basic_default = summarize_to_df(results_toy3_basic_default, str_keys = true,
#                                        dict_cols = merge(dict_basic, dict_default))
#df_toy3_basic_accelerated = summarize_to_df(results_toy3_basic_accelerated, str_keys = true,
#                                              dict_cols = merge(dict_basic, dict_accelerated))



#=
# check other behaviors
dpp_fp = results_toy1_wishart[1][:fp]
lfdpp_grad = results_toy1_wishart[1][:grad]
dpp_mm = results_toy1_wishart[1][:mm]

# check convergent result
loglik_diffs(loglik_trace) = log10.(abs.(loglik_trace[2:end] - loglik_trace[1:end-1])) - log10.(abs.(loglik_trace[1:end-1]))
p = plot(loglik_diffs(dpp_fp.loglik_trace),
         ylabel = "log10 relative change", xlabel = "#iter.", legend = :topright, dpi = 200,
         label = "fixed-point", margin = 5Plots.mm, lw = 2)
plot!(p, loglik_diffs(lfdpp_grad.loglik_trace), lw = 2, label = "ADAM")
plot!(p, loglik_diffs(dpp_mm.loglik_trace), lw = 2, label = "MM")
#Plots.savefig(p, joinpath(outdir, "log10_lldiffs.pdf"))

# check minorizing functions
Random.seed!(1234)
N = size(dpp_fp.dpp.L, 1)
M = length(dpp_fp.samples)
X = (V -> V * V')(randn(N, N)) / N
i = 5
δs = range(-0.01, 0.01, length = 50)
L_tests = [dpp_fp.dpp_trace[i].L + δ * X for δ in δs]
ll = [compute_loglik(DPP(L), dpp_fp.samples) / M for L in L_tests]
ll_fp = [compute_minorizer_fp(L, dpp_fp.dpp_trace[i].L, dpp_fp.samples) for L in L_tests]
ll_mm = [compute_minorizer_mm(L, dpp_fp.dpp_trace[i].L, dpp_fp.samples) for L in L_tests]
p1 = plot(δs, [ll ll_fp ll_mm],
          ylabel = "objective value", xlabel = "δ", legend = :topright, dpi = 200,
          label = ["f(L) (objective)" "h(L|Lt) (fixed-point)" "g(L|Lt) (MM)"],
          margin = 5Plots.mm, lw = 2, size = (480, 360))

δs = range(-0.05, 1.0, length = 50)
L_tests = [dpp_fp.dpp_trace[i].L + δ * X for δ in δs]
ll = [compute_loglik(DPP(L), dpp_fp.samples) / M for L in L_tests]
ll_fp = [compute_minorizer_fp(L, dpp_fp.dpp_trace[i].L, dpp_fp.samples) for L in L_tests]
ll_mm = [compute_minorizer_mm(L, dpp_fp.dpp_trace[i].L, dpp_fp.samples) for L in L_tests]
p2 = plot(δs, [ll ll_fp ll_mm] ./ M,
          ylabel = "objective value", xlabel = "δ", legend = :topright, dpi = 200,
          label = ["f(L) (objective)" "h(L|Lt) (fixed-point)" "g(L|Lt) (MM)"],
          margin = 5Plots.mm, lw = 2, size = (480, 360))
p = plot(p1, p2, size = (960, 360))
#Plots.savefig(p, joinpath(outdir, "minorizer_behaviors.pdf"))
Plots.savefig(p1, joinpath(outdir, "minorizer_behaviors_neighbor.pdf"))
Plots.savefig(p2, joinpath(outdir, "minorizer_behaviors_nonneighbor.pdf"))
=#
