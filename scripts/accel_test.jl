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
dpp_mm = mle_mm(DPP(Linit), samples, accelerate_steps = 0, show_progress = true, tol = 1e-4) # no acceleration
dpp_amm = mle_mm(DPP(Linit), samples, accelerate_steps = 5, show_progress = true, tol = 1e-4) # accelerate initial 10 steps

plot(dpp_mm.cputime_trace, dpp_mm.loglik_trace ./ M, label = "MM (original)", ylabel = "mean log-lik", xlabel = "CPU time[s]")
plot!(dpp_amm.cputime_trace, dpp_amm.loglik_trace ./ M, label = "MM (accelerated)", legend = :bottomright)

plot(Plots.heatmap(Linit), Plots.heatmap(dpp_mm.dpp.L), Plots.heatmap(dpp_amm.dpp.L), size = (1200, 500), layout = (1, 3))

#p1 = plot(_dpp_mm.μ_trace, title = "δ_μ = 0.1", xlabel = "#iter.", ylabel = "μ(t)", legend = :none, ylims = (-1, 0))
#Plots.hline!(p1, [0, -0.9], linestyle = :dot, color = :black)
#p2 = plot(__dpp_mm.μ_trace, title = "δ_μ = 0.2", xlabel = "#iter.", ylabel = "μ(t)", legend = :none, ylims = (-1, 0))
#Plots.hline!(p2, [0, -0.8], linestyle = :dot, color = :black)
#p3 = plot(log.(-_dpp_mm.loglik_trace), xlabel = "#iter.", ylabel = "log(-1 × log-lik)", label = "MM (accelerated)", legend = :topright)
#plot!(p3, log.(-dpp_mm.loglik_trace), xlabel = "#iter.", ylabel = "log(-1 × log-lik)", label = "MM (original)")
#p4 = plot(log.(-__dpp_mm.loglik_trace), xlabel = "#iter.", ylabel = "log(-1 × log-lik)", label = "MM (accelerated)", legend = :topright)
#plot!(p4, log.(-dpp_mm.loglik_trace), xlabel = "#iter.", ylabel = "log(-1 × log-lik)", label = "MM (original)")
#plot(p1, p2, p3, p4, size = (800, 600))
#Plots.savefig("acceleration_behavior.pdf")


# Amazon dataset
amazon_dir = joinpath("$(@__DIR__)", "..", "data", "AmazonBabyRegistry")
category = "feeding"

reg_name = "1_100_100_100_$(category)_regs.csv"
txt_name = "1_100_100_100_$(category)_item_names.txt"

samples = CSV.read(joinpath(amazon_dir, reg_name), DataFrame, header = 0) |>
    eachrow .|>
    Vector .|>
    skipmissing .|>
    collect

N = length(readlines(joinpath(amazon_dir, txt_name)))
M = length(samples)

Random.seed!(1234)
Linit = initializer(N, init = :wishart)
dpp_mm_amazon = mle_mm(DPP(Linit), samples, accelerate_steps = 0, show_progress = true, tol = 1e-4)
dpp_amm_amazon = mle_mm(DPP(Linit), samples, accelerate_steps = 5, show_progress = true, tol = 1e-4)
plot(dpp_mm_amazon.cputime_trace, dpp_mm_amazon.loglik_trace ./ M, label = "MM (original)", ylabel = "mean log-lik", xlabel = "CPU time[s]")
plot!(dpp_amm_amazon.cputime_trace, dpp_amm_amazon.loglik_trace ./ M, label = "MM (accelerated)", legend = :bottomright)

