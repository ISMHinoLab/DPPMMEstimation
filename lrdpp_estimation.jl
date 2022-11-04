using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using ProgressMeter
using Plots

using BenchmarkTools


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
    n_iter :: Int64

    function LFDPPResult(lfdpp_trace, samples)
        loglik_trace = map(lfdpp -> compute_loglik(lfdpp, samples), lfdpp_trace)
        new(samples, lfdpp_trace[end], loglik_trace[end], lfdpp_trace, loglik_trace, length(lfdpp_trace))
    end
end


function compute_loglik(lfdpp :: LFDPP, samples)
    logZ = logdet(lfdpp.V' * lfdpp.V + I)
    return sum([logdet(lfdpp.V[sample, :] * lfdpp.V[sample, :]') for sample in samples]) - length(samples) * logZ
    #try
    #    return sum([logdet(lfdpp.V[sample, :] * lfdpp.V[sample, :]') for sample in samples]) - length(samples) * logZ
    #catch
    #    return -Inf
    #end
end

function update_V(V, samples, ρ = 1.0)
    # update rule of the proposed method
    M = length(samples)
    U_samples = [I(N)[sample, :] for sample in samples]

    term1 = -mean([U_samples[m]' * inv(V[samples[m], :] * V[samples[m], :]') * U_samples[m] for m in 1:M])
    term2 = I - V * inv(I + V' * V) * V'
    return ((term1 + term2 + ρ * I) / ρ) \ V
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

    for i in 2:n_iter
        lfdpp_trace[i] = LFDPP(update_V(lfdpp_trace[i - 1].V, samples, ρ))

        next!(prog)
    end
    return LFDPPResult(lfdpp_trace, samples)
end

function mle_grad(lfdpp :: LFDPP, samples; n_iter = 100, η = 1.0, show_progress = true)
    # MLE for a low-rank factorized DPP by gradient ascent
    prog = Progress(n_iter - 1, enabled = show_progress)

    logliks = zeros(n_iter)
    logliks[1] = compute_loglik(lfdpp, samples)

    for i in 2:n_iter
        gradV = grad_V(lfdpp.V, samples)
        Vnext = lfdpp.V + η * gradV
        lfdpp_next = LFDPP(Vnext)
        logliks[i] = compute_loglik(lfdpp_next, samples)
        lfdpp = lfdpp_next

        next!(prog)
    end
    return lfdpp, logliks
end



function logdet_divergence(X, Y)
    return logdet(Y) - logdet(X) + tr(inv(Y) * X) - size(Y, 1)
end


Random.seed!(1234)
N = 64
M = 1000

#L = rand(Wishart(10N, diagm(1:N) / 10)) / 10N
#L = rand(Wishart(N, diagm(10.0 .^ range(-3, 1, length = N)))) / N
L = rand(Wishart(N, diagm(1:N) / N)) / N
#V = rand(Uniform(0, √2), (N, round(Int, N/2))) / N; L = V * V'

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)
K = min(maximum(length.(samples)) + 1, N)

#Linit = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
Linit = rand(Wishart(10N, diagm(5 * ones(N)))) / 10N
#Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
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
lfdpp_grad1, logp_grad1 = mle_grad(LFDPP(Vinit), samples, η = 0.5)
lfdpp_grad5, logp_grad5 = mle_grad(LFDPP(Vinit), samples, η = 0.1)

plot([logp1, logp_grad1])
plot([logp5, logp_grad5])
