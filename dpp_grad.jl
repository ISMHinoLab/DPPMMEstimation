using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using ProgressMeter
using Plots
using FiniteDifferences


function compute_loglik(V, samples)
    try
        return mean([logpdf(DeterminantalPointProcess(V * V'), sample) for sample in samples])
    catch
        return -Inf
    end
end

function update_V(Vₜ, samples, U_samples, M, ρₜ = 1.0)
    Lₜ = Vₜ * Vₜ'
    term1 = -mean([U_samples[m]' * inv(Lₜ[samples[m], samples[m]]) * U_samples[m] for m in 1:M])
    term2 = inv(Lₜ + I)
    return ρₜ * (term1 + term2 + ρₜ * I) \ Vₜ
end

function grad_L(L, samples, U_samples)
    U_samples = [I(N)[sample, :] for sample in samples]
    term1 = mean([U_samples[m]' * inv(L[samples[m], samples[m]]) * U_samples[m] for m in 1:M])
    term2 = -inv(L + I)
    return term1 + term2
end

function logdet_divergence(X, Y)
    return logdet(Y) - logdet(X) + tr(inv(Y) * X) - size(Y, 1)
end


Random.seed!(1234)
N = 20
M = 100

#L = rand(Wishart(10N, diagm(1:N) / 10)) / 10N
#L = rand(Wishart(N, diagm(10.0 .^ range(-3, 1, length = N)))) / N
L = rand(Wishart(N, diagm(1:N) / N)) / N
#V = rand(Uniform(0, √2), (N, round(Int, N/2))) / N; L = V * V'

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)
K = min(maximum(length.(samples)) + 1, N)

Linit = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
#Linit = rand(Wishart(10N, diagm(5 * ones(N)))) / 10N
eig_init = eigen(Linit)
Vinit = eig_init.vectors[:, end - K + 1:end] * Diagonal(sqrt.(eig_init.values[end - K + 1:end]))

U_samples = [I(N)[sample, :] for sample in samples]

G = grad_L(Vinit * Vinit', samples, U_samples)

function ll(vecV; samples = samples, N = N, K = K)
    V = reshape(vecV, N, K)
    compute_loglik(V, samples)
end

compute_loglik(Vinit, samples)
ll(vec(Vinit))

gnumv = grad(central_fdm(5, 1), ll, vec(Vinit))
GnumV = reshape(gnumv[1], N, K)
plot(heatmap(G * Vinit), heatmap(GnumV))
