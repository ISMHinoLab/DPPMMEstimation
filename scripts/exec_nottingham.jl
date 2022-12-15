using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using MAT
using Plots

include("$(@__DIR__)/dpp_utils.jl")


# load Nottingham data
nottingham = matread("../data/Nottingham.mat")
train_samples = vcat([[findall(isone, traindata[j, :]) for j in 1:size(traindata, 1)]
                      for traindata in nottingham["traindata"]]...)

Random.seed!(1234)
N = 88
M = 5000

samples = train_samples[1:M]

# initializing L
Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
eig_init = eigen(Linit)
Vinit = eig_init.vectors * Diagonal(sqrt.(eig_init.values))

n_iter = 50
# fixed-point method
dpp_fp = mle(DPP(Linit), samples, ρ = 1.0, n_iter = 50)

# gradient ascent
lfdpp_grad = mle_grad(LFDPP(Vinit), samples, η = 1e-8, ϵ = 1e-8, n_iter = 50)

# MM algorithm
dpp_mm = mle_mm(DPP(Linit), samples, n_iter = 50)

loglik_truth = compute_loglik(DPP(L), samples)
loglik_min = minimum(hcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
loglik_max = maximum(hcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
p = plot([dpp_fp.cputime_trace, lfdpp_grad.cputime_trace, dpp_mm.cputime_trace],
          [dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace],
          ylabel = "log-likelihood", xlabel = "CPU time", legend = :bottomright, dpi = 200,
          ylims = (loglik_min, loglik_max),
          label = ["fixed-point" "gradient" "MM"], margin = 5Plots.mm, lw = 2)
hline!(p, [loglik_truth], label = "true param.", lw = 2)
