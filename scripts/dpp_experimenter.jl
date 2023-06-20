using Base.Filesystem
using LinearAlgebra
using Distributions
using Random
using DataFrames
using Plots
using JLD2

include("$(@__DIR__)/dpp_utils.jl")

function vN_div(X, Y)
    # von Neumann divergence
    return tr(X * log(X)) - tr(X * log(Y)) - tr(X) + tr(Y)
end

function initializer(N; seed = nothing, init = :wishart)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    if (init == :wishart || init == :Wishart || init == :WISHART)
        #Linit = rand(Wishart(5N, diagm(ones(N)))) / 5N
        Linit = rand(Wishart(N, diagm(ones(N)))) / N
    elseif (init == :basic || init == :Basic || init == :BASIC)
        Vinit = rand(Uniform(0, √2), (N, N)) / N
        Linit = Vinit * Vinit'
    else
        throw(ArgumentError(":wishart and :basic are only allowed for the keyword argument \"init\"."))
    end
    return Linit
end

function experimenter(
        L, samples;
        max_iter = 1000, tol = 1e-4, # stopping criterion
        show_progress = true, # show progresses
        ρ = 1.0, # parameter for the fixed-point algorithm
        η = 1e-1, ϵ = 1e-7, α = 0.9, β = 0.999, # parameters for ADAM
        ϵ_mm = 1e-10, # machine epsilon for MM
        outdir = joinpath("$(@__DIR__)", "..", "output"), # path for an output directory
        save_figures = true, # save figures or not
        save_objects = true, # save .jld2 objects or not
        Ltruth = nothing, # ground truth of the parameter; if passed, a reference line will be plotted and the cosine similarities will be calculated
        fontscale = 1.0
    )

    # initial value for V s.t. L = VV'
    eig_init = eigen(L)
    V = eig_init.vectors * Diagonal(sqrt.(eig_init.values))

    # fixed-point
    dpp_fp = mle(DPP(L), samples, ρ = ρ, max_iter = max_iter, tol = tol, show_progress = show_progress)
    # gradient ascent
    lfdpp_grad = mle_grad(LFDPP(V), samples,
                          η = η, ϵ = ϵ, α = α, β = β,
                          max_iter = max_iter, tol = tol,
                          show_progress = show_progress)
    # MM algorithm
    dpp_mm = mle_mm(DPP(L), samples, ϵ = ϵ_mm, max_iter = max_iter, tol = tol, show_progress = show_progress)

    results = Dict(:fp => dpp_fp, :grad => lfdpp_grad, :mm => dpp_mm)
    if !isnothing(Ltruth)
        results[:fp_vn] = vN_div(dpp_fp.dpp.L, Ltruth)
        results[:grad_vn] = vN_div((V -> V * V')(lfdpp_grad.lfdpp.V), Ltruth)
        results[:mm_vn] = vN_div(dpp_mm.dpp.L, Ltruth)
    end

    if save_figures
        mkpath(outdir)
        M = length(samples)
        loglik_min = minimum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace)) / M
        loglik_max = maximum(vcat(dpp_fp.loglik_trace, lfdpp_grad.loglik_trace, dpp_mm.loglik_trace)) / M

        scalefontsizes(fontscale)
        p = plot(dpp_fp.cputime_trace, dpp_fp.loglik_trace / M,
                 ylabel = "mean log-likelihood", xlabel = "CPU time (s)", legend = :bottomright, dpi = 200,
                 ylims = (loglik_min, loglik_max),
                 label = "fixed-point", margin = 5Plots.mm, lw = 2)
        plot!(p, lfdpp_grad.cputime_trace, lfdpp_grad.loglik_trace / M, lw = 2, label = "ADAM")
        plot!(p, dpp_mm.cputime_trace, dpp_mm.loglik_trace / M, lw = 2, label = "MM")
        if !isnothing(Ltruth)
            loglik_truth = compute_loglik(DPP(Ltruth), samples)
            Plots.hline!(p, [loglik_truth / M], label = "true param.", lw = 2, ls = :dash, lc = :black)
        end
        Plots.savefig(p, joinpath(outdir, "curves.pdf"))
        scalefontsizes()
    end
    if save_objects
        save(joinpath(outdir, "results.jld2"), results)
    end
    return results
end

function summarize_to_df(results; dict_cols = nothing, str_keys = false)
    methods = [:fp, :grad, :mm]
    if str_keys
        methods = string.(methods)
    end
    df = vcat([DataFrame(loglik = [result[method].loglik for result in results],
                         mean_loglik = [result[method].loglik / length(result[method].samples) for result in results],
                         N = [size(string(method) == "grad" ? result[method].lfdpp.V : result[method].dpp.L, 1)
                                  for result in results],
                         M = [length(result[method].samples) for result in results],
                         n_iter = [result[method].n_iter for result in results],
                         cputime = [result[method].cputime_trace[end] for result in results],
                         mean_itertime = [result[method].cputime_trace[end] / result[method].n_iter for result in results],
                         method = method) for method in methods]...)

    div_methods = [:fp_vn, :grad_vn, :mm_vn]
    if all([div in keys(result) for result in results, div in div_methods])
        # if von Neumann divergences are stored, add a column
        vec_col = reshape([result[div] for result in results, div in div_methods], :)
        df = hcat(df, DataFrame(vn = vec_col))
    end

    if !isnothing(dict_cols)
        for (k, v) in dict_cols
            df[:, k] .= v
        end
    end
    return df
end
