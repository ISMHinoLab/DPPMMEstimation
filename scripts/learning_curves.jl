using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using CSV

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")


function curves_plot(res_default, res_accelerated; loglik_truth = nothing, colorscheme = :default, fscale = 1.75)
    M = length(res_default["fp"].samples)
    get_curves(x) = (x.cputime_trace, -(x.loglik_trace / M))

    fp_cpu_def, fp_obj_def = get_curves(res_default["fp"])
    grad_cpu_def, grad_obj_def = get_curves(res_default["grad"])
    mm_cpu_def, mm_obj_def = get_curves(res_default["mm"])
    fp_cpu_acc, fp_obj_acc = get_curves(res_accelerated["fp"])
    grad_cpu_acc, grad_obj_acc = get_curves(res_accelerated["grad"])
    mm_cpu_acc, mm_obj_acc = get_curves(res_accelerated["mm"])

    colors = repeat(palette(colorscheme)[1:3], 2)

    scalefontsizes(fscale)
    p = plot(fp_cpu_def, fp_obj_def,
             color = colors[1],
             yscale = :log10, minorgrid = true,
             ylabel = "negative mean log-likelihood", xlabel = "CPU time (s)", legend = :topright, dpi = 200,
             #ylims = (loglik_min, loglik_max),
             label = "FP (default)", margin = 5Plots.mm, lw = 2,
             size = (900, 600),
             labelfontsize = round(Int, 11 * fscale),
             tickfontsize = round(Int, 8 * fscale))
    plot!(p, grad_cpu_def, grad_obj_def, color = colors[2], lw = 2, label = "Adam (default)")
    plot!(p, mm_cpu_def, mm_obj_def, color = colors[3], lw = 2, label = "MM (default)")

    plot!(p, fp_cpu_acc, fp_obj_acc, color = colors[4], lw = 2, ls = :dashdot, label = "FP (accelerated)")
    plot!(p, grad_cpu_acc, grad_obj_acc, color = colors[5], lw = 2, ls = :dashdot, label = "Adam (accelerated)")
    plot!(p, mm_cpu_acc, mm_obj_acc, color = colors[6], lw = 2, ls = :dashdot, label = "MM (accelerated)")

    if !isnothing(loglik_truth)
        Plots.hline!(p, [-loglik_truth / M], label = "true param.", lw = 2, ls = :dash, lc = :black)
    end
    scalefontsizes()
    return p
end


LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)

## get true parameters
# toy1: N = 32, M = 2500
Random.seed!(1234)
N = 32
M = 2500
V = rand(Uniform(0, 10), (N, round(Int, N))) / N
L1 = V * V'

# toy2: N = 32, M = 2500
Random.seed!(1234)
N = 32
M = 10000
V = rand(Uniform(0, 10), (N, round(Int, N))) / N
L2 = V * V'

# toy3: N = 128, M = 2500
Random.seed!(1234)
N = 128
M = 2500
V = rand(Uniform(0, 10), (N, round(Int, N))) / N
L3 = V * V'


## load results
n_exp = 30
outdir = joinpath("$(@__DIR__)", "..", "output")
results_toy1_wishart_default = [load(outdir * "/toy1/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
results_toy1_wishart_accelerated = [load(outdir * "/toy1/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
results_toy1_basic_default = [load(outdir * "/toy1/basic/default/$(i)/results.jld2") for i in 1:n_exp]
results_toy1_basic_accelerated = [load(outdir * "/toy1/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]
results_toy2_wishart_default = [load(outdir * "/toy2/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
results_toy2_wishart_accelerated = [load(outdir * "/toy2/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
results_toy2_basic_default = [load(outdir * "/toy2/basic/default/$(i)/results.jld2") for i in 1:n_exp]
results_toy2_basic_accelerated = [load(outdir * "/toy2/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]
results_toy3_wishart_default = [load(outdir * "/toy3/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
results_toy3_wishart_accelerated = [load(outdir * "/toy3/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
results_toy3_basic_default = [load(outdir * "/toy3/basic/default/$(i)/results.jld2") for i in 1:n_exp]
results_toy3_basic_accelerated = [load(outdir * "/toy3/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]

l = compute_loglik(DPP(L1), results_toy1_wishart_default[1]["fp"].samples)
p = curves_plot(results_toy1_wishart_default[1], results_toy1_wishart_accelerated[1], loglik_truth = l)
Plots.savefig(p, joinpath(outdir, "curves_toy1_wishart.pdf"))

l = compute_loglik(DPP(L2), results_toy2_wishart_default[1]["fp"].samples)
p = curves_plot(results_toy2_wishart_default[1], results_toy2_wishart_accelerated[1], loglik_truth = l)
Plots.savefig(p, joinpath(outdir, "curves_toy2_wishart.pdf"))

l = compute_loglik(DPP(L3), results_toy3_wishart_default[1]["fp"].samples)
p = curves_plot(results_toy3_wishart_default[1], results_toy3_wishart_accelerated[1], loglik_truth = l)
Plots.savefig(p, joinpath(outdir, "curves_toy3_wishart.pdf"))

l = compute_loglik(DPP(L1), results_toy1_basic_default[1]["fp"].samples)
p = curves_plot(results_toy1_basic_default[1], results_toy1_basic_accelerated[1], loglik_truth = l)
Plots.savefig(p, joinpath(outdir, "curves_toy1_basic.pdf"))

l = compute_loglik(DPP(L2), results_toy2_basic_default[1]["fp"].samples)
p = curves_plot(results_toy2_basic_default[1], results_toy2_basic_accelerated[1], loglik_truth = l)
Plots.savefig(p, joinpath(outdir, "curves_toy2_basic.pdf"))

l = compute_loglik(DPP(L3), results_toy3_basic_default[1]["fp"].samples)
p = curves_plot(results_toy3_basic_default[1], results_toy3_basic_accelerated[1], loglik_truth = l)
Plots.savefig(p, joinpath(outdir, "curves_toy3_basic.pdf"))


# Nottingham
n_exp = 30
results_nottingham_wishart_default = [load(outdir * "/nottingham/wishart/default/$(i)/results.jld2") for i in 1:n_exp]
results_nottingham_wishart_accelerated = [load(outdir * "/nottingham/wishart/accelerated/$(i)/results.jld2") for i in 1:n_exp]
results_nottingham_basic_default = [load(outdir * "/nottingham/basic/default/$(i)/results.jld2") for i in 1:n_exp]
results_nottingham_basic_accelerated = [load(outdir * "/nottingham/basic/accelerated/$(i)/results.jld2") for i in 1:n_exp]

p = curves_plot(results_nottingham_wishart_default[1], results_nottingham_wishart_accelerated[1])
Plots.savefig(p, joinpath(outdir, "curves_nottingham_wishart.pdf"))

p = curves_plot(results_nottingham_basic_default[1], results_nottingham_basic_accelerated[1])
Plots.savefig(p, joinpath(outdir, "curves_nottingham_basic.pdf"))
