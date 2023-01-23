using Base.Filesystem
using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using DataFrames
using CSV
using Query
using Plots
using Printf

include("$(@__DIR__)/dpp_utils.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1) 
# load Amazon Baby Registry data
amazon_dir = joinpath("$(@__DIR__)", "..", "data", "AmazonBabyRegistry")
categories = ["apparel", "bath", "bedding", "carseats", "decor", "diaper",
              "feeding", "furniture", "gear", "gifts", "health", "media",
              "moms", "pottytrain", "safety", "strollers", "toys"]

#category = categories[1]
for category in categories
    reg_name = "1_100_100_100_$(category)_regs.csv"
    txt_name = "1_100_100_100_$(category)_item_names.txt"

    samples = CSV.read(joinpath(amazon_dir, reg_name), DataFrame, header = 0) |>
        eachrow .|>
        Vector .|>
        skipmissing .|>
        collect

    Random.seed!(1234)
    N = length(readlines(joinpath(amazon_dir, txt_name)))
    M = length(samples)

    # initializing L
    Linit = rand(Wishart(10N, diagm(ones(N)))) / 10N
    #Vinit = rand(Uniform(0, √2), (N, N)) / N; Linit = Vinit * Vinit'


    eig_init = eigen(Linit)
    evals = ifelse.(eig_init.values .< 0.0, 0.0, eig_init.values)
    Vinit = eig_init.vectors * Diagonal(sqrt.(evals))

    max_iter = 1000
    tol = 1e-4
    # fixed-point method
    dpp_fp = mle(DPP(Linit), samples, ρ = 1.0, max_iter = max_iter, tol = tol);

    # gradient ascent
    lfdpp_grad = mle_grad(LFDPP(Vinit), samples, η = 1e-9, ϵ = 1e-8, max_iter = max_iter, tol = tol);

    # MM algorithm
    dpp_mm = mle_mm(DPP(Linit), samples, max_iter = max_iter, tol = tol);


    # check learning curves
    outdir = joinpath("$(@__DIR__)", "..", "output", "amazon")
    mkpath(outdir)
    loglik_min = minimum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
    loglik_max = maximum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace))
    p = plot(dpp_fp.cputime_trace, dpp_fp.loglik_trace,
             ylabel = "log-likelihood", xlabel = "CPU time (s)", legend = :bottomright, dpi = 200,
             ylims = (loglik_min, loglik_max),
             label = "fixed-point", margin = 5Plots.mm, lw = 2)
    plot!(p, lfdpp_grad.cputime_trace, lfdpp_grad.loglik_trace, lw = 2, label = "ADAM")
    plot!(p, dpp_mm.cputime_trace, dpp_mm.loglik_trace, lw = 2, label = "MM")
    Plots.savefig(p, joinpath(outdir, @sprintf("amazon_%s_curves_wishart_%.0e.pdf", category, tol)))

    p = plot(dpp_fp.cputime_trace, dpp_fp.loglik_trace,
             ylabel = "log-likelihood", xlabel = "CPU time (s)", legend = :bottomright, dpi = 200,
             xlims = (0, 20), ylims = (loglik_min, loglik_max),
             label = "fixed-point", margin = 5Plots.mm, lw = 2)
    plot!(p, lfdpp_grad.cputime_trace, lfdpp_grad.loglik_trace, lw = 2, label = "ADAM")
    plot!(p, dpp_mm.cputime_trace, dpp_mm.loglik_trace, lw = 2, label = "MM")
    Plots.savefig(p, joinpath(outdir, @sprintf("amazon_%s_curves_wishart_%.0e_scaled.pdf", category, tol)))


    # check convergent result
    loglik_diffs(loglik_trace) = log10.(abs.(loglik_trace[2:end] - loglik_trace[1:end-1])) - log10.(abs.(loglik_trace[1:end-1]))
    p = plot(loglik_diffs(dpp_fp.loglik_trace),
             ylabel = "log10 relative change", xlabel = "#iter.", legend = :topright, dpi = 200,
             label = "fixed-point", margin = 5Plots.mm, lw = 2)
    plot!(p, loglik_diffs(lfdpp_grad.loglik_trace), lw = 2, label = "ADAM")
    plot!(p, loglik_diffs(dpp_mm.loglik_trace), lw = 2, label = "MM")
    Plots.savefig(p, joinpath(outdir, @sprintf("amazon_%s_log10_lldiffs_wishart_%.0e.pdf", category, tol)))
end

