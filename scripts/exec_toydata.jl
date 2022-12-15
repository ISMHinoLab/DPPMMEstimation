using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using Plots

include("$(@__DIR__)/dpp_utils.jl")


Random.seed!(1234)
#N = 128
#M = 5000
N = 64
M = 2500

#L = rand(Wishart(10N, diagm(1:N) / 10)) / 10N
#L = rand(Wishart(N, diagm(10.0 .^ range(-3, 1, length = N)))) / N
#L = rand(Wishart(N, diagm(1:N) / N)) / N
V = rand(Uniform(0, 10), (N, round(Int, N))) / N; L = V * V'

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)

# initializing L
#Minit = [mean(in.(n, samples) .* in.(m, samples)) for n in 1:N, m in 1:N]
#Kinit = [n == m ? Minit[n, n] : max(Minit[n, n] * Minit[m, m] - Minit[n, m], 0) for n in 1:N, m in 1:N]
#Linit = Kinit * inv(I - Kinit)
#Linit = rand(Wishart(10N, diagm(10 * init_diag))) / 10N
#init = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
#Linit = rand(Wishart(10N, diagm(5 * ones(N)))) / 10N
Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
#Vinit = rand(Uniform(0, √2), (N, N)) / N; Linit = Vinit * Vinit'
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


# check minorizing functions
X = (V -> V * V')(randn(N, N)) / N
i = 5
L_tests = [dpp_fp.dpp_trace[i].L + ϵ * X for ϵ in range(-0.01, 0.01, length = 50)]
ll = [compute_loglik(DPP(L), samples) / M for L in L_tests]
ll_fp = [compute_minorizer_fp(L, dpp_fp.dpp_trace[i].L, samples) for L in L_tests]
ll_mm = [compute_minorizer_mm(L, dpp_fp.dpp_trace[i].L, samples) for L in L_tests]
plot([ll ll_fp ll_mm])
