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

    logliks = zeros(n_iter)
    logliks[1] = compute_loglik(lfdpp, samples)

    for i in 2:n_iter
        lfdpp_next = LFDPP(update_V(lfdpp.V, samples, ρ))
        logliks[i] = compute_loglik(lfdpp_next, samples)
        lfdpp = lfdpp_next

        next!(prog)
    end
    return lfdpp, logliks
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


lfdpp1, logp1 = mle(LFDPP(Vinit), samples, ρ = 1.0)
lfdpp2, logp2 = mle(LFDPP(Vinit), samples, ρ = 2.0)
lfdpp5, logp5 = mle(LFDPP(Vinit), samples, ρ = 5.0)

loglik_truth = sum([logpdf(DeterminantalPointProcess(L), sample) for sample in samples])
cmin = minimum(hcat(L, Vinit * Vinit', lfdpp1.V * lfdpp1.V', lfdpp2.V * lfdpp2.V', lfdpp5.V * lfdpp5.V'))
cmax = maximum(hcat(L, Vinit * Vinit', lfdpp1.V * lfdpp1.V', lfdpp2.V * lfdpp2.V', lfdpp5.V * lfdpp5.V'))

p1 = plot([logp1, logp2, logp5], ylabel = "log-likelihood", xlabel = "iter.", legend = :bottomright, dpi = 200,
          label = ["rho = 1" "rho = 2" "rho = 5"], margin = 5Plots.mm, lw = 2)
hline!(p1, [loglik_truth], label = "true param.", lw = 2)
p2 = heatmap(L, clims = (cmin, cmax), title = "truth", dpi = 200)
p3 = heatmap(Vinit * Vinit', clims = (cmin, cmax), title = "init.", dpi = 200)
p4 = heatmap(lfdpp1.V * lfdpp1.V', clims = (cmin, cmax), title = "est. (rho = 1)", dpi = 200)
p = plot(p1, p2, p3, p4, size = (1200, 800))
savefig(p, "result.pdf")


# 勾配法との比較
lfdpp_grad1, logp_grad1 = mle_grad(LFDPP(Vinit), samples, η = 0.5)
lfdpp_grad5, logp_grad5 = mle_grad(LFDPP(Vinit), samples, η = 0.1)

plot([logp1, logp_grad1])
plot([logp5, logp_grad5])
