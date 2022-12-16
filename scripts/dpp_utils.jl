using LinearAlgebra
using SparseArrays
using MatrixEquations
using DeterminantalPointProcesses
using ProgressMeter
using Plots


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


function solve_arec(A :: AbstractMatrix, G :: AbstractMatrix, Q :: AbstractMatrix)
    H = [A -G; -Q -A']
    S = schur(H)
    select = real(S.values) .< 0
    ordschur!(S, select)
    m, n = size(S.Z)
    @views X = S.Z[Int(m/2 + 1):m, 1:Int(n/2)] * inv(S.Z[1:Int(m/2), 1:Int(n/2)])
    return (X + X') / 2
end

function compute_loglik(dpp :: DPP, samples)
    try
        logZ = logdet(dpp.L + I)
        return sum([logdet(dpp.L[sample, sample]) for sample in samples]) - length(samples) * logZ
    catch
        return -Inf
    end
end

function compute_loglik(lfdpp :: LFDPP, samples)
    try
        logZ = logdet(lfdpp.V' * lfdpp.V + I)
        return sum([logdet(lfdpp.V[sample, :] * lfdpp.V[sample, :]') for sample in samples]) - length(samples) * logZ
    catch
        return -Inf
    end
end

function compute_minorizer_fp(L, Lt, samples)
    # compute value of the minorizer of the fixed-point algorithm
    M = length(samples)
    U_samples = [sparse(I(N)[sample, :]) for sample in samples]

    term1 = mean([logdet(Lt[samples[m], samples[m]]) -
                  tr(inv(L) * Lt * U_samples[m]' * inv(Lt[samples[m], samples[m]]) * U_samples[m] * Lt) +
                  length(samples[m]) for m in 1:M])
    term2 = -logdet(L) + logdet(Lt) - logdet(Lt + I) - tr(inv(Lt + I) * (inv(L) * Lt - I))
    return term1 + term2
end

function compute_minorizer_mm(L, Lt, samples)
    # compute value of the proposed minorizer
    M = length(samples)
    U_samples = [sparse(I(N)[sample, :]) for sample in samples]

    term1 = mean([logdet(Lt[samples[m], samples[m]]) -
                  tr(inv(L) * Lt * U_samples[m]' * inv(Lt[samples[m], samples[m]]) * U_samples[m] * Lt) +
                  length(samples[m]) for m in 1:M])
    term2 = -logdet(Lt + I) - tr(inv(Lt + I) * (L - Lt))
    return term1 + term2
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

function update_L_mm(L, samples; ϵ = 1e-10)
    # update rule of the MM algorithm.

    M = length(samples)
    U_samples = [sparse(I(N)[sample, :]) for sample in samples]

    Q = Symmetric(L * mean([U_samples[m]' * inv(L[samples[m], samples[m]]) * U_samples[m] for m in 1:M]) * L)
    G = Symmetric(inv(L + I))
    A = zeros(size(L))
    return arec(A, G, Q + ϵ * I)[1]
    #return solve_arec(A, G, Q + ϵ * I)
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

function mle_mm(dpp :: DPP, samples; n_iter = 100, show_progress = true, ϵ = 1e-10)
    # MLE for a full-rank DPP by the MM algorithm
    prog = Progress(n_iter - 1, enabled = show_progress)

    dpp_trace = Vector{DPP}(undef, n_iter)
    dpp_trace[1] = dpp
    cputime_trace = zeros(n_iter)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            dpp_trace[i] = DPP(update_L_mm(dpp_trace[i - 1].L, samples, ϵ = ϵ))
        end
        next!(prog)
    end
    return DPPResult(dpp_trace, samples, cumsum(cputime_trace))
end
