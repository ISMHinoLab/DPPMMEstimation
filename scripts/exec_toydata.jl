using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using CSV

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)

n_exp = 5 # number of experiments

# toy1: N = 32, M = 2500
## WISHART initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy1", "wishart")
results_toy1_wishart = map(1:n_exp) do i
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

## BASIC initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy1", "basic")
results_toy1_basic = map(1:n_exp) do i
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

# toy2: N = 32, M = 2500
## WISHART initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy2", "wishart")
results_toy2_wishart = map(1:n_exp) do i
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

## BASIC initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy2", "basic")
results_toy2_basic = map(1:n_exp) do i
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

# toy3: N = 128, M = 2500
## WISHART initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy3", "wishart")
results_toy3_wishart = map(1:n_exp) do i
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

## BASIC initialization
Random.seed!(1234)
outdir = joinpath("$(@__DIR__)", "..", "output", "toy3", "basic")
results_toy3_basic = map(1:n_exp) do i
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


dict_wishart = Dict(:init => :wishart)
dict_basic = Dict(:init => :basic)
df_toy1_wishart = summarize_to_df(results_toy1_wishart, dict_cols = dict_wishart)
df_toy1_basic = summarize_to_df(results_toy1_basic, dict_cols = dict_basic)
df_toy2_wishart = summarize_to_df(results_toy2_wishart, dict_cols = dict_wishart)
df_toy2_basic = summarize_to_df(results_toy2_basic, dict_cols = dict_basic)
df_toy3_wishart = summarize_to_df(results_toy3_wishart, dict_cols = dict_wishart)
df_toy3_basic = summarize_to_df(results_toy3_basic, dict_cols = dict_basic)

df_toy1 = vcat(df_toy1_wishart, df_toy1_basic)
df_toy1[:, :setting] .= 1
df_toy2 = vcat(df_toy2_wishart, df_toy2_basic)
df_toy2[:, :setting] .= 2
df_toy3 = vcat(df_toy3_wishart, df_toy3_basic)
df_toy3[:, :setting] .= 3
outdir = joinpath("$(@__DIR__)", "..", "output")
CSV.write(joinpath(outdir, "toy_results.csv"), vcat(df_toy1, df_toy2, df_toy3))

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
X = (V -> V * V')(randn(N, N)) / N
i = 5
δs = range(-0.01, 0.01, length = 50)
L_tests = [dpp_fp.dpp_trace[i].L + δ * X for δ in δs]
ll = [compute_loglik(DPP(L), samples) / M for L in L_tests]
ll_fp = [compute_minorizer_fp(L, dpp_fp.dpp_trace[i].L, samples) for L in L_tests]
ll_mm = [compute_minorizer_mm(L, dpp_fp.dpp_trace[i].L, samples) for L in L_tests]
p1 = plot(δs, [ll ll_fp ll_mm],
          ylabel = "objective value", xlabel = "δ", legend = :topright, dpi = 200,
          label = ["f(L) (objective)" "h(L|Lt) (fixed-point)" "g(L|Lt) (MM)"],
          margin = 5Plots.mm, lw = 2, size = (360, 480))

δs = range(-0.05, 1.0, length = 50)
L_tests = [dpp_fp.dpp_trace[i].L + δ * X for δ in δs]
ll = [compute_loglik(DPP(L), samples) / M for L in L_tests]
ll_fp = [compute_minorizer_fp(L, dpp_fp.dpp_trace[i].L, samples) for L in L_tests]
ll_mm = [compute_minorizer_mm(L, dpp_fp.dpp_trace[i].L, samples) for L in L_tests]
p2 = plot(δs, [ll ll_fp ll_mm] ./ M,
          ylabel = "objective value", xlabel = "δ", legend = :topright, dpi = 200,
          label = ["f(L) (objective)" "h(L|Lt) (fixed-point)" "g(L|Lt) (MM)"],
          margin = 5Plots.mm, lw = 2, size = (360, 480))
p = plot(p1, p2, size = (800, 600))
#Plots.savefig(p, joinpath(outdir, "minorizer_behaviors.pdf"))
=#
