using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using ProgressMeter
using Plots

using BenchmarkTools


mutable struct DPP
    L :: Matrix{Float64}
    N :: Int64

    function DPP(L)
        N = size(L, 1)
        new(L, N)
    end
end

struct DPPResult
    samples :: Vector{Vector{Int64}}
    dpp :: DPP
    loglik :: Float64
    dpp_trace :: Vector{DPP}
    loglik_trace :: Vector{Float64}
    cputime_trace :: Vector{Float64}
    n_iter :: Int64

    function DPPResult(dpp_trace, samples)
        loglik_trace = map(dpp -> compute_loglik(dpp, samples), dpp_trace)
        new(samples, dpp_trace[end], loglik_trace[end], dpp_trace, loglik_trace, zeros(), length(dpp_trace))
    end

    function DPPResult(dpp_trace, samples, cputime_trace)
        loglik_trace = map(dpp -> compute_loglik(dpp, samples), dpp_trace)
        new(samples, dpp_trace[end], loglik_trace[end], dpp_trace, loglik_trace, cputime_trace, length(dpp_trace))
    end
end

mutable struct LFDPP
    V :: Matrix{Float64}
    N :: Int64
    K :: Int64

    function LFDPP(V)
        N, K = size(V)
        new(V, N, K)
    end
end

struct LFDPPResult
    samples :: Vector{Vector{Int64}}
    lfdpp :: LFDPP
    loglik :: Float64
    lfdpp_trace :: Vector{LFDPP}
    loglik_trace :: Vector{Float64}
    cputime_trace :: Vector{Float64}
    n_iter :: Int64

    function LFDPPResult(lfdpp_trace, samples)
        loglik_trace = map(lfdpp -> compute_loglik(lfdpp, samples), lfdpp_trace)
        new(samples, lfdpp_trace[end], loglik_trace[end], lfdpp_trace, loglik_trace, zeros(), length(lfdpp_trace))
    end

    function LFDPPResult(lfdpp_trace, samples, cputime_trace)
        loglik_trace = map(lfdpp -> compute_loglik(lfdpp, samples), lfdpp_trace)
        new(samples, lfdpp_trace[end], loglik_trace[end], lfdpp_trace, loglik_trace, cputime_trace, length(lfdpp_trace))
    end
end


function compute_loglik(dpp :: DPP, samples)
    logZ = logdet(dpp.L + I)
    try
        return sum([logdet(dpp.L[sample, sample]) for sample in samples]) - length(samples) * logZ
    catch
        return -Inf
    end
end

function compute_loglik(lfdpp :: LFDPP, samples)
    logZ = logdet(lfdpp.V' * lfdpp.V + I)
    try
        return sum([logdet(lfdpp.V[sample, :] * lfdpp.V[sample, :]') for sample in samples]) - length(samples) * logZ
    catch
        return -Inf
    end
end

function update_V(V, samples, ρ = 1.0)
    # update rule of the proposed method
    M = length(samples)
    U_samples = [I(N)[sample, :] for sample in samples]

    term1 = -mean([U_samples[m]' * inv(V[samples[m], :] * V[samples[m], :]') * U_samples[m] for m in 1:M])
    term2 = I - V * inv(I + V' * V) * V'
    return ((term1 + term2 + ρ * I) / ρ) \ V
end

function update_L(L, samples, ρ = 1.0)
    # update rule of the fixed-point method
    M = length(samples)
    U_samples = [I(N)[sample, :] for sample in samples]

    term1 = mean([U_samples[m]' * inv(L[samples[m], samples[m]]) * U_samples[m] for m in 1:M])
    term2 = -inv(L + I)
    Δ = term1 + term2
    return L + ρ * L * Δ * L
end

function grad_V(V, samples)
    # gradient of the mean log-likelihood by V
    M = length(samples)
    U_samples = [I(N)[sample, :] for sample in samples]

    term1 = mean([U_samples[m]' * inv(V[samples[m], :] * V[samples[m], :]') * U_samples[m] for m in 1:M])
    term2 = -(I - V * inv(I + V' * V) * V')
    return 2 * (term1 + term2) * V
end

function mle(lfdpp :: LFDPP, samples; n_iter = 100, ρ = 1.0, show_progress = true)
    # MLE for a low-rank factorized DPP by the proposed method
    prog = Progress(n_iter - 1, enabled = show_progress)

    lfdpp_trace = Vector{LFDPP}(undef, n_iter)
    lfdpp_trace[1] = lfdpp
    cputime_trace = zeros(n_iter)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            lfdpp_trace[i] = LFDPP(update_V(lfdpp_trace[i - 1].V, samples, ρ))
        end
        next!(prog)
    end
    return LFDPPResult(lfdpp_trace, samples, cumsum(cputime_trace))
end

function mle_grad(lfdpp :: LFDPP, samples; n_iter = 100, η = 1.0, show_progress = true)
    # MLE for a low-rank factorized DPP by gradient ascent
    prog = Progress(n_iter - 1, enabled = show_progress)

    lfdpp_trace = Vector{LFDPP}(undef, n_iter)
    lfdpp_trace[1] = lfdpp
    cputime_trace = zeros(n_iter)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            gradV = grad_V(lfdpp_trace[i - 1].V, samples)
            Vnext = lfdpp_trace[i - 1].V + η * gradV
            lfdpp_trace[i] = LFDPP(Vnext)
        end

        next!(prog)
    end
    return LFDPPResult(lfdpp_trace, samples, cumsum(cputime_trace))
end

function mle(dpp :: DPP, samples; n_iter = 100, ρ = 1.0, show_progress = true)
    # MLE for a full-rank DPP by the fixed-point method (Mariet & Sra, 2015)
    prog = Progress(n_iter - 1, enabled = show_progress)

    dpp_trace = Vector{DPP}(undef, n_iter)
    dpp_trace[1] = dpp
    cputime_trace = zeros(n_iter)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            dpp_trace[i] = DPP(update_L(dpp_trace[i - 1].L, samples, ρ))
        end
        next!(prog)
    end
    return DPPResult(dpp_trace, samples, cumsum(cputime_trace))
end

function logdet_divergence(X, Y)
    return logdet(Y) - logdet(X) + tr(inv(Y) * X) - size(Y, 1)
end


Random.seed!(1234)
N = 64
M = 1000

#L = rand(Wishart(10N, diagm(1:N) / 10)) / 10N
#L = rand(Wishart(N, diagm(10.0 .^ range(-3, 1, length = N)))) / N
#L = rand(Wishart(N, diagm(1:N) / N)) / N
#V = rand(Uniform(0, 10), (N, round(Int, N/2))) / N; L = V * V'
V = rand(Uniform(0, 10), (N, round(Int, N))) / N; L = V * V'

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)
K = min(maximum(length.(samples)) + 1, N)

#emp_probs = [mean(in.(n, samples)) for n in 1:N]
#init_diag = emp_probs ./ (1 .- emp_probs)
#Linit = rand(Wishart(10N, diagm(10 * init_diag))) / 10N
#Linit = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
Linit = rand(Wishart(10N, diagm(5 * ones(N)))) / 10N
#Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
#Vinit = rand(Uniform(0, √2), (N, round(Int, N/2))) / N; Linit = Vinit * Vinit' * 1e2
eig_init = eigen(Linit)
Vinit = eig_init.vectors[:, end - K + 1:end] * Diagonal(sqrt.(eig_init.values[end - K + 1:end]))


lfdpp_res1 = mle(LFDPP(Vinit), samples, ρ = 1.0)
lfdpp_res2 = mle(LFDPP(Vinit), samples, ρ = 2.0)
lfdpp_res5 = mle(LFDPP(Vinit), samples, ρ = 5.0)

loglik_truth = sum([logpdf(DeterminantalPointProcess(L), sample) for sample in samples])
cmin = minimum(hcat(L, Vinit * Vinit', lfdpp_res1.lfdpp.V * lfdpp_res1.lfdpp.V', lfdpp_res2.lfdpp.V * lfdpp_res2.lfdpp.V', lfdpp_res5.lfdpp.V * lfdpp_res5.lfdpp.V'))
cmax = maximum(hcat(L, Vinit * Vinit', lfdpp_res1.lfdpp.V * lfdpp_res1.lfdpp.V', lfdpp_res2.lfdpp.V * lfdpp_res2.lfdpp.V', lfdpp_res5.lfdpp.V * lfdpp_res5.lfdpp.V'))

p1 = plot([lfdpp_res1.loglik_trace, lfdpp_res2.loglik_trace, lfdpp_res5.loglik_trace],
          ylabel = "log-likelihood", xlabel = "iter.", legend = :bottomright, dpi = 200,
          label = ["rho = 1" "rho = 2" "rho = 5"], margin = 5Plots.mm, lw = 2)
hline!(p1, [loglik_truth], label = "true param.", lw = 2)
p2 = heatmap(L, clims = (cmin, cmax), title = "truth", dpi = 200)
p3 = heatmap(Vinit * Vinit', clims = (cmin, cmax), title = "init.", dpi = 200)
p4 = heatmap(lfdpp_res1.lfdpp.V * lfdpp_res1.lfdpp.V', clims = (cmin, cmax), title = "est. (rho = 1)", dpi = 200)
p = plot(p1, p2, p3, p4, size = (1200, 800))
savefig(p, "result.pdf")


# 勾配法との比較
lfdpp_grad1 = mle_grad(LFDPP(Vinit), samples, η = 0.5)
lfdpp_grad2 = mle_grad(LFDPP(Vinit), samples, η = 0.25)
lfdpp_grad5 = mle_grad(LFDPP(Vinit), samples, η = 0.1)
loglik_min = minimum(hcat(lfdpp_res1.loglik_trace, lfdpp_grad1.loglik_trace,
                          lfdpp_res5.loglik_trace, lfdpp_grad5.loglik_trace))
loglik_max = maximum(hcat(lfdpp_res1.loglik_trace, lfdpp_grad1.loglik_trace,
                          lfdpp_res5.loglik_trace, lfdpp_grad5.loglik_trace))
p = plot([lfdpp_res1.loglik_trace, lfdpp_grad1.loglik_trace],
         ylabel = "log-likelihood", xlabel = "#iter.", legend = :bottomright, dpi = 200,
         ylims = (loglik_min, loglik_max),
         label = ["proposed" "gradient ascent"], margin = 5Plots.mm, lw = 2)
hline!(p, [loglik_truth], label = "true param.", lw = 2)
savefig(p, "result_rho1.pdf")
p = plot([lfdpp_res5.loglik_trace, lfdpp_grad5.loglik_trace],
         ylabel = "log-likelihood", xlabel = "#iter.", legend = :bottomright, dpi = 200,
         ylims = (loglik_min, loglik_max),
         label = ["proposed" "gradient ascent"], margin = 5Plots.mm, lw = 2)
hline!(p, [loglik_truth], label = "true param.", lw = 2)
savefig(p, "result_rho5.pdf")

plot(lfdpp_res1.cputime_trace, lfdpp_res1.loglik_trace)
plot!(lfdpp_grad1.cputime_trace, lfdpp_grad1.loglik_trace)
plot(lfdpp_res2.cputime_trace, lfdpp_res2.loglik_trace)
plot!(lfdpp_grad2.cputime_trace, lfdpp_grad2.loglik_trace)
plot(lfdpp_res5.cputime_trace, lfdpp_res5.loglik_trace)
plot!(lfdpp_grad5.cputime_trace, lfdpp_grad5.loglik_trace)


# fixed-point method
dpp_res1 = mle(DPP(Linit), samples, ρ = 1.0)
p = plot([lfdpp_res1.cputime_trace, lfdpp_grad1.cputime_trace, dpp_res1.cputime_trace],
          [lfdpp_res1.loglik_trace, lfdpp_grad1.loglik_trace, dpp_res1.loglik_trace],
          ylabel = "log-likelihood", xlabel = "CPU time", legend = :bottomright, dpi = 200,
          ylims = (loglik_min, loglik_max),
          label = ["proposed" "gradient ascent" "fixed-point"], margin = 5Plots.mm, lw = 2)
hline!(p, [loglik_truth], label = "true param.", lw = 2)
