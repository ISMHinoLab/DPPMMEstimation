using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using ProgressMeter
using Plots


function compute_loglik(V, samples)
    try
        return sum([logpdf(DeterminantalPointProcess(V * V'), sample) for sample in samples])
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


Random.seed!(1234)
N = 50
M = 1000

L = rand(Wishart(10N, diagm(1:N) / 10)) / 10N

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)

K = maximum(length.(samples)) + 1


Linit = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
eig_init = eigen(Linit)
Vinit = eig_init.vectors[:, end - K + 1:end] * Diagonal(sqrt.(eig_init.values[end - K + 1:end]))

U_samples = [I(N)[sample, :] for sample in samples]


Vest = Vinit
n_iter = 100
logliks = Vector{Float64}(undef, n_iter)
logliks[1] = compute_loglik(Vest, samples)
@showprogress for i in 2:n_iter
    Vest = update_V(Vest, samples, U_samples, M, 1.0)
    logliks[i] = compute_loglik(Vest, samples)
end


cmin = minimum(hcat(L, Vinit * Vinit', Vest * Vest'))
cmax = maximum(hcat(L, Vinit * Vinit', Vest * Vest'))

p1 = plot(logliks, ylabel = "log-likelihood", xlabel = "iter.", legend = :none, dpi = 200)
p2 = heatmap(L, clims = (cmin, cmax), title = "truth", dpi = 200)
p3 = heatmap(Vinit * Vinit', clims = (cmin, cmax), title = "init.", dpi = 200)
p4 = heatmap(Vest * Vest', clims = (cmin, cmax), title = "est.", dpi = 200)

savefig(plot(p1, p2, p3, p4, size = (1200, 800)), "result.png")
