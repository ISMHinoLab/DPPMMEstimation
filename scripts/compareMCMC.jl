using LinearAlgebra
using Distributions
using DeterminantalPointProcesses
using Random
using DelimitedFiles

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)

Random.seed!(1234)
N = 8
M = 1000

V = rand(Uniform(-10, 10), (N, 2N)) / 2N
L = V * V'

dpp = DeterminantalPointProcess(L)
samples = rand(dpp, M)

## WISHART initialization
Linit_wishart = initializer(N, init = :wishart)
dpp_mm_wishart = mle_mm(DPP(Linit_wishart), samples)

## BASIC initialization
Linit_basic = initializer(N, init = :basic)
dpp_mm_basic = mle_mm(DPP(Linit_basic), samples)


outdir = "$(@__DIR__)/../output/MCMC"
mkpath(outdir)
writedlm(joinpath(outdir, "samples.csv"), samples, ",")
writedlm(joinpath(outdir, "L_truth.csv"), L, ",")
writedlm(joinpath(outdir, "Linit_wishart.csv"), Linit_wishart, ",")
writedlm(joinpath(outdir, "Linit_basic.csv"), Linit_basic, ",")
writedlm(joinpath(outdir, "L_mm_wishart.csv"), dpp_mm_wishart.dpp.L, ",")
writedlm(joinpath(outdir, "L_mm_basic.csv"), dpp_mm_basic.dpp.L, ",")
