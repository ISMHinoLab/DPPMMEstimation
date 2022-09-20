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
    return ((term1 + term2 + ρₜ * I) / ρₜ) \ Vₜ
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

K = min(maximum(length.(samples)), N)

#Linit = rand(Wishart(10N, diagm(10 * ones(N)))) / 10N
Linit = rand(Wishart(10N, diagm(5 * ones(N)))) / 10N
#Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
eig_init = eigen(Linit)
Vinit = eig_init.vectors[:, end - K + 1:end] * Diagonal(sqrt.(eig_init.values[end - K + 1:end]))

U_samples = [I(N)[sample, :] for sample in samples]


ρs = [5.0, 2.0, 1.0]
n_iter = 100
logliks = zeros(n_iter, length(ρs))
logliks[1, :] .= compute_loglik(Vinit, samples)
lddiv_numer = zeros(n_iter - 1, length(ρs))
lddiv_denom = zeros(n_iter - 1, length(ρs))
lddiv_numer_prox = zeros(n_iter - 1, length(ρs))
sval_traj = zeros(K, n_iter - 1)

Vest = Vinit
for (j, ρ) in enumerate(ρs)
    Vest = Vinit
    @showprogress for i in 2:n_iter
        try
            Vnext = update_V(Vest, samples, U_samples, M, ρ)
            logliks[i, j] = compute_loglik(Vnext, samples)
            lddiv_numer[i - 1, j] = mean([logdet_divergence((Vnext * Vnext')[sample, sample], (Vest * Vest')[sample, sample]) for sample in samples])
            lddiv_denom[i - 1, j] = logdet_divergence(Vnext * Vnext' + I, Vest * Vest' + I)
            lddiv_numer_prox[i - 1, j] = lddiv_numer[i - 1, j] - ρ * norm(Vnext - Vest, 2)
            if (ρ == 1.0)
                sval_traj[:, i - 1] = svdvals(Vnext)
            end
            Vest = Vnext
        catch
            break
        end
    end
end


loglik_truth = sum([logpdf(DeterminantalPointProcess(L), sample) for sample in samples])
cmin = minimum(hcat(L, Vinit * Vinit', Vest * Vest'))
cmax = maximum(hcat(L, Vinit * Vinit', Vest * Vest'))

p1 = plot(logliks, ylabel = "log-likelihood", xlabel = "iter.", legend = :bottomright, dpi = 200,
          label = ["rho = 5" "rho = 2" "rho = 1"], margin = 5Plots.mm, lw = 2)
hline!(p1, [loglik_truth], label = "true param.", lw = 2)
p2 = heatmap(L, clims = (cmin, cmax), title = "truth", dpi = 200)
p3 = heatmap(Vinit * Vinit', clims = (cmin, cmax), title = "init.", dpi = 200)
p4 = heatmap(Vest * Vest', clims = (cmin, cmax), title = "est. (rho = 1)", dpi = 200)
p = plot(p1, p2, p3, p4, size = (1200, 800))
savefig(p, "result.pdf")

plots_lddiv = [plot(hcat(lddiv_numer[:, j], lddiv_denom[:, j], lddiv_numer_prox[:, j]),
                    label = ["numer" "denom" "numer+prox"], title = "LogDet Div. (rho = $ρ)",
                    dpi = 200, legend = :bottomright)
               for (j, ρ) in enumerate(ρs)]
savefig.(plots_lddiv, ["result_rho$(ρ).pdf" for ρ in ρs])
