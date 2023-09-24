using StatsBase
using DataFrames
using DataFramesMeta
using CSV
using StatsPlots


outdir = joinpath("$(@__DIR__)", "..", "output")
df_toy = CSV.read(joinpath(outdir, "toy_results.csv"), DataFrame)
df_nottingham = CSV.read(joinpath(outdir, "nottingham_results.csv"), DataFrame)
df_amazon = CSV.read(joinpath(outdir, "amazon_results.csv"), DataFrame)

df_toy_aggr = @chain df_toy begin
    groupby([:setting, :init, :method, :params])
    @combine begin
        :mean_mloglik = mean(:mean_loglik)
        :std_mloglik = std(:mean_loglik)
        :mean_cputime = mean(:cputime)
        :std_cputime = std(:cputime)
        :mean_vn = mean(filter(!isnothing, tryparse.(Float64, :vn)))
        :std_vn = std(filter(!isnothing, tryparse.(Float64, :vn)))
        :mean_mitertime = mean(:mean_itertime)
    end
end

df_nottingham_aggr = @chain df_nottingham begin
    groupby([:init, :method, :params])
    @combine begin
        :mean_mloglik = mean(:mean_loglik)
        :std_mloglik = std(:mean_loglik)
        :mean_cputime = mean(:cputime)
        :std_cputime = std(:cputime)
        :mean_mitertime = mean(:mean_itertime)
        :mean_M = mean(:M)
    end
end

df_amazon_aggr = @chain df_amazon begin
    groupby([:category, :method, :params])
    @combine begin
        :mean_mloglik = mean(:mean_loglik)
        :std_mloglik = std(:mean_loglik)
        :mean_cputime = mean(:cputime)
        :std_cputime = std(:cputime)
        :mean_mitertime = mean(:mean_itertime)
        :mean_N = mean(:N)
        :mean_M = mean(:M)
    end
    @subset(:mean_N .> 25)
end

# visualize results on the toy data
df_toy_wishart = @orderby(@subset(df_toy_aggr, :init .== "wishart"), invperm(sortperm(:params, rev = true)))
df_toy_basic = @orderby(@subset(df_toy_aggr, :init .== "basic"), invperm(sortperm(:params, rev = true)))
fcolors = reshape(repeat(palette(:default)[1:3], 2), (1, :))
lcolors = reshape(vcat(repeat(palette(:default)[1:3]), repeat([:black], 3)), (1, :))
labels = ["FP (default)" "Adam (default)" "MM (default)" "FP (accelerated)" "Adam (accelerated)" "MM (accelerated)"]
# keyword arguments for plotting
plot_kws = Dict(:group => vcat(repeat(1:3, 3), repeat(4:6, 3)),
                :xticks => (1:3, ["N=32\nM=2500", "N=32\nM=10000", "N=128\nM=2500"]),
                :fillcolor => fcolors, :linecolor => lcolors,
                :fillalpha => reshape(vcat(repeat([0.1], 3), repeat([1.0], 3)), (1, :)),
                :linewidth => 2.5, :linecolor => lcolors,
                :markerstrokewidth => 2.5, :markerstrokecolor => lcolors,
                :label => labels, :dpi => 200,
                :tickfontsize => floor(8 * fontscale),
                :labelfontsize => floor(11 * fontscale),
                :margin => 5Plots.mm, :size => (960, 540))

p_toy_wishart_runtime = @with df_toy_wishart begin
    scalefontsizes(fontscale)
    p = groupedbar(:mean_cputime, yerror = :std_cputime,
                   ylabel = "CPU time [s]", legend = Symbol("topleft");
                   plot_kws...)
    scalefontsizes()
    p
end

p_toy_basic_runtime = @with df_toy_basic begin
    scalefontsizes(fontscale)
    p = groupedbar(:mean_cputime, yerror = :std_cputime,
                   ylabel = "CPU time [s]", legend = Symbol("topleft");
                   plot_kws...)
    scalefontsizes()
    p
end

p_toy_wishart_loglik = @with df_toy_wishart begin
    scalefontsizes(fontscale)
    p = groupedbar(-:mean_mloglik, yerror = :std_mloglik, yscale = Symbol("log10"),
                   ylabel = "negative mean log-likelihood", legend = Symbol("bottomleft");
                   plot_kws...)
    scalefontsizes()
    p
end

p_toy_basic_loglik = @with df_toy_basic begin
    scalefontsizes(fontscale)
    p = groupedbar(-:mean_mloglik, yerror = :std_mloglik, yscale = Symbol("log10"),
                   ylabel = "negative mean log-likelihood", legend = Symbol("bottomleft");
                   plot_kws...)
    scalefontsizes()
    p
end

p_toy_wishart_vn = @with df_toy_wishart begin
    fontscale = 1.5
    scalefontsizes(fontscale)
    p = groupedbar(:mean_vn, yerror = :std_vn,
                   ylabel = "von-Neumann divergence", legend = Symbol("bottomleft");
                   plot_kws...)
    scalefontsizes()
    p
end

p_toy_basic_vn = @with df_toy_basic begin
    fontscale = 1.5
    scalefontsizes(fontscale)
    p = groupedbar(:mean_vn, yerror = :std_vn,
                   ylabel = "von-Neumann divergence", legend = Symbol("bottomleft");
                   plot_kws...)
    scalefontsizes()
    p
end

Plots.savefig(p_toy_wishart_runtime, joinpath(outdir, "cputimes_toy_wishart.pdf"))
Plots.savefig(p_toy_basic_runtime, joinpath(outdir, "cputimes_toy_basic.pdf"))
Plots.savefig(p_toy_wishart_loglik, joinpath(outdir, "logliks_toy_wishart.pdf"))
Plots.savefig(p_toy_basic_loglik, joinpath(outdir, "logliks_toy_basic.pdf"))
Plots.savefig(p_toy_wishart_vn, joinpath(outdir, "vndivs_toy_wishart.pdf"))
Plots.savefig(p_toy_basic_vn, joinpath(outdir, "vndivs_toy_basic.pdf"))


# visualize results on Nottingham data
df_nottingham_ = @orderby(df_nottingham_aggr, invperm(sortperm(:params, rev = true)))

# keyword arguments for plotting
plot_kws = Dict(:group => vcat(repeat(1:3, 2), repeat(4:6, 2)),
                :xticks => (1:2, ["WISHART", "BASIC"]),
                :fillcolor => fcolors, :linecolor => lcolors,
                :fillalpha => reshape(vcat(repeat([0.1], 3), repeat([1.0], 3)), (1, :)),
                :linewidth => 2.5, :linecolor => lcolors,
                :markerstrokewidth => 2.5, :markerstrokecolor => lcolors,
                :label => labels, :dpi => 200,
                :tickfontsize => floor(8 * fontscale),
                :labelfontsize => floor(11 * fontscale),
                :margin => 5Plots.mm, :size => (960, 540))

p_nottingham_runtime = @with df_nottingham_ begin
    scalefontsizes(fontscale)
    p = groupedbar(:mean_cputime, yerror = :std_cputime,
                   ylabel = "CPU time [s]", legend = Symbol("topleft");
                   plot_kws...)
    scalefontsizes()
    p
end

p_nottingham_loglik = @with df_nottingham_ begin
    scalefontsizes(fontscale)
    p = groupedbar(-:mean_mloglik, yerror = :std_mloglik,
                   ylabel = "negative mean log-likelihood", legend = Symbol("bottomleft");
                   plot_kws...)
    scalefontsizes()
    p
end


p_amazon_default = @with @subset(df_amazon_aggr, :params .== "default") begin
    scalefontsizes(fontscale)
    p = groupedbar(:mean_cputime, group = :method, yerror = :std_cputime,
                   xticks = (1:13, unique(:category)),
                   xrotation = 60, xlabel = "category",
                   ylabel = "CPU time [s]", dpi = 200,
                   tickfontsize = floor(8 * fontscale),
                   labelfontsize = floor(11 * fontscale),
                   margin = 10Plots.mm, size = (960, 540),
                   title = "default hyperparameters")
    scalefontsizes()
    p
end

p_amazon_accelerated = @with @subset(df_amazon_aggr, :params .== "accelerated") begin
    scalefontsizes(fontscale)
    p = groupedbar(:mean_cputime, group = :method, yerror = :std_cputime,
                   xticks = (1:13, unique(:category)),
                   xrotation = 60, xlabel = "Category",
                   ylabel = "CPU time [s]", dpi = 200,
                   tickfontsize = floor(8 * fontscale),
                   labelfontsize = floor(11 * fontscale),
                   margin = 10Plots.mm, size = (960, 540),
                   title = "accelerated hyperparameters")
    scalefontsizes()
    p
end

Plots.savefig(p_amazon_default, joinpath(outdir, "cputimes_amazon_default.pdf"))
Plots.savefig(p_amazon_accelerated, joinpath(outdir, "cputimes_amazon_accelerated.pdf"))
