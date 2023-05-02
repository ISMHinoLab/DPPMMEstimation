using StatsBase
using DataFrames
using DataFramesMeta
using CSV


outdir = joinpath("$(@__DIR__)", "..", "output")
df_toy = CSV.read(joinpath(outdir, "toy_results.csv"), DataFrame)
df_nottingham = CSV.read(joinpath(outdir, "nottingham_results.csv"), DataFrame)
df_amazon = CSV.read(joinpath(outdir, "amazon_results.csv"), DataFrame)

df_toy_aggr = @chain df_toy begin
    groupby([:setting, :init, :method])
    @combine begin
        :mean_mloglik = mean(:mean_loglik)
        :std_mloglik = std(:mean_loglik)
        :mean_cputime = mean(:cputime)
        :std_cputime = std(:cputime)
        :mean_mitertime = mean(:mean_itertime)
    end
    #groupby([:setting, :method])
    #@combine begin
    #    :mean_mitertime = mean(:mean_mitertime) * 1000.0
    #end
end

df_nottingham_aggr = @chain df_nottingham begin
    groupby([:init, :method])
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
    groupby([:category, :method])
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


p_amazon = @with df_amazon_aggr begin
    fontscale = 1.5
    scalefontsizes(fontscale)
    p = groupedbar(:mean_cputime, group = :method, yerror = :std_cputime,
                   xticks = (1:13, unique(:category)),
                   xrotation = 60, xlabel = "Category",
                   ylabel = "CPU time [s]", dpi = 200,
                   tickfontsize = floor(8 * fontscale),
                   labelfontsize = floor(11 * fontscale),
                   margin = 10Plots.mm, size = (960, 540))
    scalefontsizes()
    p
end
savefig(p_amazon, joinpath(outdir, "cputimes_amazon.pdf"))
