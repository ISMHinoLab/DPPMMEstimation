using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using Plots
using Printf

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

max_iter = 1000
tol = 1e-4
# fixed-point method
dpp_fp = mle(DPP(Linit), samples, ρ = 1.0, max_iter = max_iter, tol = tol);

# gradient ascent
lfdpp_grad = mle_grad(LFDPP(Vinit), samples, η = 1e-9, ϵ = 1e-8, max_iter = max_iter, tol = tol);

# MM algorithm
dpp_mm = mle_mm(DPP(Linit), samples, max_iter = max_iter, tol = tol);

# check learning curves
outdir = joinpath("$(@__DIR__)", "..", "output")
loglik_truth = compute_loglik(DPP(L), samples)
loglik_min = minimum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
loglik_max = maximum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
p = plot(dpp_fp.cputime_trace, dpp_fp.loglik_trace,
         ylabel = "log-likelihood", xlabel = "CPU time (s)", legend = :bottomright, dpi = 200,
         ylims = (loglik_min, loglik_max),
         label = "fixed-point", margin = 5Plots.mm, lw = 2)
plot!(p, lfdpp_grad.cputime_trace, lfdpp_grad.loglik_trace, lw = 2, label = "ADAM")
plot!(p, dpp_mm.cputime_trace, dpp_mm.loglik_trace, lw = 2, label = "MM")
Plots.hline!(p, [loglik_truth], label = "true param.", lw = 2)
Plots.savefig(p, joinpath(outdir, @sprintf("toy_curves_%.0e.pdf", tol)))

p = plot(dpp_fp.cputime_trace, dpp_fp.loglik_trace,
         ylabel = "log-likelihood", xlabel = "CPU time (s)", legend = :bottomright, dpi = 200,
         xlims = (0, 1.5), ylims = (loglik_min, loglik_max),
         label = "fixed-point", margin = 5Plots.mm, lw = 2)
plot!(p, lfdpp_grad.cputime_trace, lfdpp_grad.loglik_trace, lw = 2, label = "ADAM")
plot!(p, dpp_mm.cputime_trace, dpp_mm.loglik_trace, lw = 2, label = "MM")
Plots.hline!(p, [loglik_truth], label = "true param.", lw = 2)
Plots.savefig(p, joinpath(outdir, @sprintf("toy_curves_%.0e_scaled.pdf", tol)))


# check convergent result
loglik_diffs(loglik_trace) = log10.(abs.(loglik_trace[2:end] - loglik_trace[1:end-1])) - log10.(abs.(loglik_trace[1:end-1]))
p = plot(loglik_diffs(dpp_fp.loglik_trace),
         ylabel = "log10 relative change", xlabel = "#iter.", legend = :topright, dpi = 200,
         label = "fixed-point", margin = 5Plots.mm, lw = 2)
plot!(p, loglik_diffs(lfdpp_grad.loglik_trace), lw = 2, label = "ADAM")
plot!(p, loglik_diffs(dpp_mm.loglik_trace), lw = 2, label = "MM")
Plots.savefig(p, joinpath(outdir, @sprintf("toy_log10_lldiffs_%.0e.pdf", tol)))



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
p2 = plot(δs, [ll ll_fp ll_mm],
          ylabel = "objective value", xlabel = "δ", legend = :topright, dpi = 200,
          label = ["f(L) (objective)" "h(L|Lt) (fixed-point)" "g(L|Lt) (MM)"],
          margin = 5Plots.mm, lw = 2, size = (360, 480))
p = plot(p1, p2, size = (800, 600))
Plots.savefig(p, joinpath(outdir, "minorizer_behaviors.pdf"))
