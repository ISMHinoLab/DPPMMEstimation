using LinearAlgebra
using SparseArrays
using MatrixEquations
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

function update_L(L, samples, ρ = 1.0)
    # update rule of the fixed-point method
    M = length(samples)
    U_samples = [sparse(I(N)[sample, :]) for sample in samples]

    term1 = Symmetric(mean([U_samples[m]' * inv(L[samples[m], samples[m]]) * U_samples[m] for m in 1:M]))
    term2 = -Symmetric(inv(L + I))
    Δ = term1 + term2
    return L + ρ * L * Δ * L
end

function update_L_mm(L, samples)
    # update rule of the MM algorithm.

    M = length(samples)
    U_samples = [sparse(I(N)[sample, :]) for sample in samples]

    Q = Symmetric(L * mean([U_samples[m]' * inv(L[samples[m], samples[m]]) * U_samples[m] for m in 1:M]) * L)
    G = Symmetric(inv(L + I))
    A = zeros(size(L))
    return arec(A, G, Q)[1]
end

function grad_V(V, samples)
    # gradient of the mean log-likelihood by V
    M = length(samples)
    U_samples = [I(N)[sample, :] for sample in samples]

    term1 = mean([U_samples[m]' * inv(V[samples[m], :] * V[samples[m], :]') * U_samples[m] for m in 1:M])
    term2 = -(I - V * inv(I + V' * V) * V')
    return 2 * (term1 + term2) * V
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

function mle_grad(lfdpp :: LFDPP, samples; n_iter = 100, show_progress = true,
                  η = 1e-8, ϵ = 1e-8, α = 0.9, β = 0.999)
    # MLE for a low-rank factorized DPP by Adam-adjusted gradient ascent
    # η: learning rate
    # ϵ: fudge factor for AdaGrad
    # α: mixing parameter of Momentum
    # β: mixing parameter of RMSProp

    prog = Progress(n_iter - 1, enabled = show_progress)

    lfdpp_trace = Vector{LFDPP}(undef, n_iter)
    lfdpp_trace[1] = lfdpp
    cputime_trace = zeros(n_iter)

    historical_grad = zeros(lfdpp.N, lfdpp.K)
    historical_velocity = zeros(lfdpp.N, lfdpp.K)
    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            gradV = -grad_V(lfdpp_trace[i - 1].V, samples)

            hisotrical_grad = β * historical_grad + (1 - β) * gradV .^ 2
            historical_velocity = α * historical_velocity + (1 - α) * gradV
            adj_hgrad = historical_grad ./ (1 - β ^ (i - 1))
            adj_hvelocity = historical_velocity ./ (1 - α ^ (i - 1))

            adj_grad = adj_hvelocity ./ (.√(adj_hgrad) .+ ϵ)
            Vnext = lfdpp_trace[i - 1].V - η * adj_grad
            lfdpp_trace[i] = LFDPP(Vnext)
        end

        next!(prog)
    end
    return LFDPPResult(lfdpp_trace, samples, cumsum(cputime_trace))
end

function mle_mm(dpp :: DPP, samples; n_iter = 100, show_progress = true)
    # MLE for a full-rank DPP by the MM algorithm
    prog = Progress(n_iter - 1, enabled = show_progress)

    dpp_trace = Vector{DPP}(undef, n_iter)
    dpp_trace[1] = dpp
    cputime_trace = zeros(n_iter)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            dpp_trace[i] = DPP(update_L_mm(dpp_trace[i - 1].L, samples))
        end
        next!(prog)
    end
    return DPPResult(dpp_trace, samples, cumsum(cputime_trace))
end


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
