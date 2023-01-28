using StatsBase
using DataFrames
using DataFramesMeta
using CSV


outdir = joinpath("$(@__DIR__)", "..", "output")
df_toy = CSV.read(joinpath(outdir, "toy_results.csv"), DataFrame)
df_nottingham = CSV.read(joinpath(outdir, "nottingham_results.csv"), DataFrame)
df_amazon = CSV.read(joinpath(outdir, "amazon_results.csv"), DataFrame)

@chain df_toy begin
    groupby([:setting, :init, :method])
    @combine begin
        :mean_mloglik = mean(:mean_loglik)
        :std_mloglik = std(:mean_loglik)
        :mean_cputime = mean(:cputime)
        :std_cputime = std(:cputime)
        :mean_mitertime = mean(:mean_itertime)
    end
end

@chain df_nottingham begin
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

@chain df_amazon begin
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
end

