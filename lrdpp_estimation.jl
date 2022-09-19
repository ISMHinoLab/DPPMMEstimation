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

Linit = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
#Linit = rand(Wishart(10N, diagm(5 * ones(N)))) / 10N
eig_init = eigen(Linit)
Vinit = eig_init.vectors[:, end - K + 1:end] * Diagonal(sqrt.(eig_init.values[end - K + 1:end]))

U_samples = [I(N)[sample, :] for sample in samples]


Vest = Vinit
n_iter = 100
logliks = zeros(n_iter)
logliks[1] = compute_loglik(Vest, samples)
lddiv_numer = zeros(n_iter - 1)
lddiv_denom = zeros(n_iter - 1)
lddiv_numer_prox = zeros(n_iter - 1)
sval_traj = zeros(K, n_iter - 1)

ρ = 1.0
@showprogress for i in 2:n_iter
    try
        Vnext = update_V(Vest, samples, U_samples, M, ρ)
        logliks[i] = compute_loglik(Vnext, samples)
        lddiv_numer[i - 1] = mean([logdet_divergence((Vnext * Vnext')[sample, sample], (Vest * Vest')[sample, sample]) for sample in samples])
        lddiv_denom[i - 1] = logdet_divergence(Vnext * Vnext' + I, Vest * Vest' + I)
        lddiv_numer_prox[i - 1] = lddiv_numer[i - 1] - ρ * norm(Vnext - Vest, 2)
        sval_traj[:, i - 1] = svdvals(Vnext)
        Vest = Vnext
    catch
        break
    end
end


loglik_truth = sum([logpdf(DeterminantalPointProcess(L), sample) for sample in samples])
cmin = minimum(hcat(L, Vinit * Vinit', Vest * Vest'))
cmax = maximum(hcat(L, Vinit * Vinit', Vest * Vest'))

p1 = plot(logliks, ylabel = "log-likelihood", xlabel = "iter.", legend = :none, dpi = 200)
hline!(p1, [loglik_truth])
p2 = heatmap(L, clims = (cmin, cmax), title = "truth", dpi = 200)
p3 = heatmap(Vinit * Vinit', clims = (cmin, cmax), title = "init.", dpi = 200)
p4 = heatmap(Vest * Vest', clims = (cmin, cmax), title = "est.", dpi = 200)
p = plot(p1, p2, p3, p4, size = (1200, 800))
savefig(p, "result.png")

p1 = plot(hcat(lddiv_numer, lddiv_denom, lddiv_numer_prox), label = ["numer" "denom" "numer+prox"], title = "LogDet Div.")
p2 = plot(sval_traj', legend = :none, title = "trajectory of svdvals")
plot(p1, p2)

