library(rstan)
library(reshape2)
library(dplyr)
library(ggplot2)

source("./ggplot_theme_Publication-2.R")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

outdir <- file.path(getwd(), "../output/MCMC")

L.truth <- as.matrix(read.csv(file.path(outdir, "L_truth.csv"), header = FALSE))
Linit.wishart <- as.matrix(read.csv(file.path(outdir, "Linit_wishart.csv"), header = FALSE))
Linit.basic <- as.matrix(read.csv(file.path(outdir, "Linit_basic.csv"), header = FALSE))
L.mm.wishart <- as.matrix(read.csv(file.path(outdir, "L_mm_wishart.csv"), header = FALSE))
L.mm.basic <- as.matrix(read.csv(file.path(outdir, "L_mm_basic.csv"), header = FALSE))
samples <- lapply(
    readLines(file.path(outdir, "samples.csv")),
    function (x) {as.numeric(unlist(strsplit(x, ",")))}
)

N <- dim(L.truth)[1]
M <- length(samples)

init.f <- function() {
    return (list(L = rWishart(1, N, diag(N)) / N))
}

samples.flat <- unlist(samples)
samples.length <- sapply(samples, length)

stan.data <- list(
    N = N,
    M = M,
    samples_flat = samples.flat,
    samples_length = samples.length,
    L_truth = L.truth
)

#fit <- stan(file = "compareMCMC.stan", data = stan.data, init = init.f, seed = 1234)
fit <- stan(
    file = "compareMCMC.stan",
    data = stan.data,
    #init = lapply(1:4, function(i) list(V = diag(N))),
    seed = 1234,
    chains = 15
)
ext.fit <- extract(fit)

N.mc <- dim(ext.fit$L)[1]

df.ext <- melt(ext.fit$L)
names(df.ext)[2:3] <- c("row", "col")
df.ext <- df.ext %>%
    filter(row >= col) %>%
    mutate(rowcol = paste(row, col, sep = "-"))
df.ext$rowcol <- factor(df.ext$rowcol)

df.L <- data.frame(row = numeric(), col = numeric(),
                   truth = numeric(), mm_wishart = numeric(), mm_basic = numeric())
for (i in 1:N) {
    for (j in 1:i) {
        df.L <- bind_rows(
            df.L,
            data.frame(
                row = i,
                col = j,
                rowcol = paste(i, j, sep = "-"),
                truth = L.truth[i, j],
                mm_wishart = L.mm.wishart[i, j],
                mm_basic = L.mm.basic[i, j]
            )
        )
    }
}
df.L <- melt(df.L, id.vars = c("row", "col", "rowcol"))

#p <- ggplot(df.ext, aes(x = value, color = rowcol))
p <- ggplot(df.ext, aes(x = value)) + facet_grid(row ~ col) + geom_density()
p <- p + geom_vline(data = df.L, aes(xintercept = value, color = variable), linewidth = 1.0)
p <- p + theme_Publication(base_size = 20)
p <- p + labs(color = "Estimator")

ggsave("dpp_posteriors.pdf", p, width = 2000, height = 1500, units = "px", dpi = 100)

