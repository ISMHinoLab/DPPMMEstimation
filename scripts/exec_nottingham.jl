using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using MAT
using Plots

include("$(@__DIR__)/dpp_utils.jl")

LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)

# load Nottingham data
nottingham = matread("../data/Nottingham.mat")
train_samples = vcat([[findall(isone, traindata[j, :]) for j in 1:size(traindata, 1)]
                      for traindata in nottingham["traindata"]]...)

Random.seed!(1234)
N = 88
M = 15000

samples = sample(train_samples, M, replace = false)

# initializing L
Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
eig_init = eigen(Linit)
Vinit = eig_init.vectors * Diagonal(sqrt.(eig_init.values))

max_iter = 1000
tol = 1e-5
# fixed-point method
dpp_fp = mle(DPP(Linit), samples, ρ = 1.0, max_iter = max_iter, tol = tol);

# gradient ascent
lfdpp_grad = mle_grad(LFDPP(Vinit), samples, η = 1e-8, ϵ = 1e-8, max_iter = max_iter, tol = tol);

# MM algorithm
dpp_mm = mle_mm(DPP(Linit), samples, max_iter = max_iter, tol = tol);


# check learning curves
loglik_min = minimum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
loglik_max = maximum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
p = plot(dpp_fp.cputime_trace, dpp_fp.loglik_trace,
         ylabel = "log-likelihood", xlabel = "CPU time", legend = :bottomright, dpi = 200,
         ylims = (loglik_min, loglik_max),
         label = "fixed-point", margin = 5Plots.mm, lw = 2)
plot!(p, lfdpp_grad.cputime_trace, lfdpp_grad.loglik_trace, lw = 2, label = "gradient")
plot!(p, dpp_mm.cputime_trace, dpp_mm.loglik_trace, lw = 2, label = "MM")


# check convergence
p = plot(log10.(abs.(dpp_fp.loglik_trace[2:end] - dpp_fp.loglik_trace[1:(end-1)])) - log10.(abs.(dpp_fp.loglik_trace[1:(end-1)])),
         ylabel = "log10 growth", xlabel = "# iter.", legend = :topright, dpi = 200,
         label = "fixed-point", margin = 5Plots.mm, lw = 2)
plot!(p, log10.(abs.(lfdpp_grad.loglik_trace[2:end] - lfdpp_grad.loglik_trace[1:(end-1)])) - log10.(abs.(lfdpp_grad.loglik_trace[1:(end-1)])),
      label = "gradient", lw = 2)
plot!(p, log10.(abs.(dpp_mm.loglik_trace[2:end] - dpp_mm.loglik_trace[1:(end-1)])) - log10.(abs.(dpp_mm.loglik_trace[1:(end-1)])),
      label = "MM", lw = 2)

