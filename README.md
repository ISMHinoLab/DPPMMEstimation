# DPPMMEstimation
Estimating parameters of determinantal point processes using an MM algorithm.
Provides examples of the paper submitted to UAI'23:  
"Anonymous author(s), Minorization-Maximization for Learning Determinantal Point Processes"

# Working Directory
Please set your working directory at `scripts/`.

# Recommended Environments
All the program are implemented in Julia.  

|Language|Recommended ver.|
|:------|:---------------|
|`Julia`  |≥ 1.8.0|

|Library|Recommended ver.|
|:------|:---------------|
|`Random`  ||
|`LinearAlgebra`  ||
|`SparseArrays`  ||
|`Plots`  |≥ 1.36.1|
|`UnicodePlots`  |≥ 3.3.1|
|`StatsBase`  |≥ 0.33.21|
|`Distributions`  |≥ 0.25.76|
|`DataFrames`  |≥ 1.4.2|
|`DataFramesMeta`  |≥ 0.13.0|
|`Query`  |≥ 1.0.0|
|`JLD2`  |≥ 0.4.30|
|`DeterminantalPointProcesses`  |≥ 0.2.2|
|`MatrixEquations`  |≥ 2.2.2|
|`CSV`  |≥ 0.10.7|
|`MAT`  |≥ 0.10.3|


# Codes for the Example

|File|Description|
|:---|:----------|
|`scripts/exec_toydata.jl`|Example on the toy data|
|`scripts/exec_nottingham.jl`|Example on the Nottingham dataset|
|`scripts/exec_amazon.jl`|Example on the Amazon Baby Registry Dataset|
|`scripts/aggr_results.jl`|Aggregate an experimental result into a `DataFrame`|


# References
- Anonymous author(s), "Minorization-Maximization for Learning Determinantal Point Processes," submitted to _UAI2023_.  
- Mariet, Z., and Sra, S. "Fixed-point Algorithms for Learning Determinantal Point Processes," _ICML2015_.  
- Gillenwater, J. A., Kulesza, A., Fox, E., and Taskar, B. "Expectation-Maximization for Learning Determinantal Point Processes," _NeurIPS2014_.
