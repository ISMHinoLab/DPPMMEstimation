data {
    int<lower=0> N; // number of items
    int<lower=0> M; // number of samples
    int<lower=0, upper=N> samples_length[M]; // sizes of sampled subsets
    int<lower=0, upper=N> samples_flat[sum(samples_length)]; // samples (flattened)
    cov_matrix[N] L_truth;
}

//parameters {
//    cov_matrix[N] L;
//}

parameters {
    cholesky_factor_corr[N] V;
    vector<lower=0>[N] sigma_vec;
}

transformed parameters {
    cov_matrix[N] L;
    L = multiply_lower_tri_self_transpose(
        diag_pre_multiply(sigma_vec, V)
    );
}

model {
    V ~ lkj_corr_cholesky(1.001);
    for (n in 1:N) {
        sigma_vec[n] ~ cauchy(0, 100);
    }
    int i = 1;
    target += - M * log_determinant(L + diag_matrix(rep_vector(1.0, N)));
    for (l in samples_length) {
        int idxs[l];
        idxs = samples_flat[i:(i+l-1)];
        target += log_determinant(L[idxs, idxs]);
        i += l;
    }
}

generated quantities {
    real l2dist;
    l2dist = sum(columns_dot_self(L - L_truth));
}
