using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using CSV

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)

Random.seed!(1234)
N = 64
M = 2500

V = rand(Uniform(0, 10), (N, round(Int, N))) / N
L = V * V'

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)

Linit = initializer(N, init = :wishart)
dpp_mm = mle_mm(DPP(Linit), samples, accelerate = false)
_dpp_mm = mle_mm(DPP(Linit), samples, accelerate = true, ϵ_μ = 0.1)

plot(dpp_mm.cputime_trace, dpp_mm.loglik_trace ./ M, label = "MM (original)", ylabel = "mean log-lik", xlabel = "CPU time[s]")
plot!(_dpp_mm.cputime_trace, _dpp_mm.loglik_trace ./ M, label = "MM (accelerated)", legend = :bottomright)

Plots.heatmap(Linit)
Plots.heatmap(dpp_mm.dpp.L)
Plots.heatmap(_dpp_mm.dpp.L)

