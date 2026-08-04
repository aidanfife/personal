# =============================================================================
# A toy demonstration of JOMO (Joint Modelling Multiple Imputation)
# -----------------------------------------------------------------------------
# Goal: show how multilevel joint-model imputation works, how the native jomo
#       code is structured, and how it behaves under MCAR / MAR / MNAR.
#
# The script is deliberately small and heavily annotated. Read it top to
# bottom. Each numbered section maps to one idea. Run it with:
#
#     Rscript jomo_demo.R
#
# or step through it interactively in RStudio.
#
# Required packages: jomo, lme4
# Optional packages: coda (Geweke diagnostic), mitml (alternative pooling)
#
# Outputs:
#   jomo_diagnostics_trace_MAR.png
#   jomo_diagnostics_acf_MAR.png
#   jomo_imputed_vs_true_MAR.png
#
# IMPORTANT SCOPE NOTE:
# This is a MULTILEVEL teaching example, not yet a complete NHANES survey
# implementation. The cluster variable is analogous to a PSU, but jomo's
# `clus` argument does not, by itself, incorporate survey weights or strata.
# =============================================================================

required_packages <- c("jomo", "lme4")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install the required package(s) before running this script: ",
    paste(missing_packages, collapse = ", "),
    "\nRun: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))",
    call. = FALSE
  )
}

library(coda)
library(mitml)

suppressPackageStartupMessages({
  library(jomo)
  library(lme4)
})

set.seed(42)  # same simulated population every time

# Teaching values. These are not recommendations to copy blindly into the
# final NHANES analysis; Section 6 shows how diagnostics inform the choices.
N_IMPUTATIONS <- 5L
DIAGNOSTIC_ITERATIONS <- 5000L
N_BURN <- 1000L
N_BETWEEN <- 1000L

# Set to TRUE after the main continuous example works. Section 11 then adds an
# incomplete binary variable so jomo uses its latent-normal mixed-data model.
RUN_CATEGORICAL_EXTENSION <- FALSE


# =============================================================================
# 0.  WHY SIMULATE A CLUSTERED DATASET?
# -----------------------------------------------------------------------------
# Simulation gives us two things a real incomplete dataset cannot:
#
#   1. the TRUE values that will later be hidden, and
#   2. the TRUE coefficient used to generate the outcome.
#
# We also create genuine clustering. Individuals within the same cluster share
# random intercepts, so their values are correlated. This is the key feature
# that a multilevel imputation model should preserve.
#
# The round trip is:
#   complete clustered data -> create missingness -> impute with jomo ->
#   analyze each completed dataset -> pool -> compare with the known truth.
# =============================================================================


# =============================================================================
# 1.  BUILD A COMPLETE TWO-LEVEL TOY DATASET
# -----------------------------------------------------------------------------
# 40 clusters x 15 people = 600 observations.
#
# Level 2: cluster_id
# Level 1: individual rows within each cluster
#
# The data-generating model is:
#
#   x_ij = 0.5*z_ij + b_xj + e_xij
#
#   y_ij = 2 + 0.8*x_ij + 1.5*z_ij + b_yj + e_yij
#
# where b_xj and b_yj are cluster-specific random intercepts. Our substantive
# analysis question is the fixed effect of x on y. Its TRUE value is 0.8.
#
# `z` and `y` remain fully observed. We create missingness only in `x`, just as
# in the attached MICE example. Although y is complete, it belongs in jomo's
# joint response matrix Y so its association with x helps impute missing x.
# =============================================================================

n_clusters <- 40L
people_per_cluster <- 15L
n <- n_clusters * people_per_cluster

cluster_number <- rep(seq_len(n_clusters), each = people_per_cluster)
cluster_id <- factor(cluster_number)

# Separate cluster effects for x and for the residual part of y.
cluster_effect_x <- rnorm(n_clusters, mean = 0, sd = 3)
cluster_effect_y <- rnorm(n_clusters, mean = 0, sd = 4)

z <- rnorm(n, mean = 50, sd = 10)
x <- 0.5 * z + cluster_effect_x[cluster_number] + rnorm(n, 0, 5)
y <- 2 + 0.8 * x + 1.5 * z +
  cluster_effect_y[cluster_number] + rnorm(n, 0, 3)

# An extra binary variable is generated now but ignored in the main example.
# It is used only if RUN_CATEGORICAL_EXTENSION is changed to TRUE in Section 11.
p_smoke <- plogis(-0.5 + 0.03 * (z - 50) + 0.04 * (x - mean(x)))
smoke <- factor(
  rbinom(n, size = 1, prob = p_smoke),
  levels = c(0, 1),
  labels = c("no", "yes")
)

full <- data.frame(
  cluster_id = cluster_id,
  x = x,
  y = y,
  z = z,
  smoke = smoke
)

TRUE_BETA_X <- 0.8

cat("\n=== Section 1: the complete clustered toy dataset ===\n")
print(head(full, 4))
cat(sprintf("\nRows: %d | clusters: %d | rows per cluster: %d\n",
            nrow(full), nlevels(full$cluster_id), people_per_cluster))


# =============================================================================
# 2.  IMPOSE MISSINGNESS ON x
# -----------------------------------------------------------------------------
# The probability that x is missing is:
#
#   MCAR: unrelated to every variable (a constant-probability coin flip).
#
#   MAR : related to fully OBSERVED y and z. This is a stronger teaching
#         example than making missingness depend only on z: selection on the
#         outcome can bias the complete-case regression, while jomo can use
#         y and z to recover the missing x distribution.
#
#   MNAR: related to x itself. Once x vanishes, the variable driving its
#         missingness is unobserved. Ordinary MAR-based jomo cannot generally
#         repair that without an explicit MNAR sensitivity model.
#
# Each scenario targets approximately 30% missingness.
# =============================================================================

# Convert an arbitrary score into logistic probabilities whose mean is exactly
# `target` before the random missingness draws are made.
probabilities_with_target_mean <- function(score, target = 0.30) {
  score <- as.numeric(scale(score))
  intercept <- uniroot(
    function(a) mean(plogis(a + score)) - target,
    interval = c(-20, 20)
  )$root
  plogis(intercept + score)
}

make_x_missing <- function(data, probability, seed) {
  stopifnot(length(probability) == nrow(data))
  set.seed(seed)
  out <- data
  remove_x <- runif(nrow(data)) < probability
  out$x[remove_x] <- NA_real_
  out
}

# MCAR: constant missingness probability.
p_mcar <- rep(0.30, n)

# MAR: depends on observed y and z. y receives most of the weight because
# outcome-dependent selection makes the complete-case problem easy to see.
mar_score <- 0.85 * as.numeric(scale(y)) + 0.15 * as.numeric(scale(z))
p_mar <- probabilities_with_target_mean(mar_score, target = 0.30)

# MNAR: depends on the value of x that may become unobserved.
p_mnar <- probabilities_with_target_mean(x, target = 0.30)

dat_mcar <- make_x_missing(full, p_mcar, seed = 101)
dat_mar <- make_x_missing(full, p_mar, seed = 102)
dat_mnar <- make_x_missing(full, p_mnar, seed = 103)

cat("\n=== Section 2: how much x is missing in each scenario ===\n")
cat(sprintf("MCAR: %d/%d (%.1f%%) missing\n",
            sum(is.na(dat_mcar$x)), n, 100 * mean(is.na(dat_mcar$x))))
cat(sprintf("MAR : %d/%d (%.1f%%) missing\n",
            sum(is.na(dat_mar$x)), n, 100 * mean(is.na(dat_mar$x))))
cat(sprintf("MNAR: %d/%d (%.1f%%) missing\n",
            sum(is.na(dat_mnar$x)), n, 100 * mean(is.na(dat_mnar$x))))

# A compact missingness-pattern table without requiring the mice package.
cat("\nMissingness pattern for the MAR data:\n")
print(
  as.data.frame(
    table(
      y_observed = !is.na(dat_mar$y),
      x_observed = !is.na(dat_mar$x),
      z_observed = !is.na(dat_mar$z)
    )
  )
)


# =============================================================================
# 3.  DEFINE THE SUBSTANTIVE ANALYSIS AND THE BENCHMARKS
# -----------------------------------------------------------------------------
# The substantive model is a random-intercept linear mixed model:
#
#   y ~ x + z + (1 | cluster_id)
#
# For every missingness scenario we compare:
#
#   (A) COMPLETE DATA : model fitted before any values were removed.
#   (B) COMPLETE CASE : rows with missing x are silently discarded.
#   (C) JOMO          : fit the model in each imputation and pool the results.
#
# The complete-data estimate is not exactly 0.8 because a finite simulated
# sample contains sampling noise. It is the fairest practical benchmark for
# judging what the missing-data methods recovered.
# =============================================================================

fit_analysis_model <- function(data) {
  lme4::lmer(
    y ~ x + z + (1 | cluster_id),
    data = data,
    REML = FALSE,
    na.action = na.omit,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )
}

extract_x_estimate <- function(model) {
  coefficient_table <- summary(model)$coefficients
  c(
    beta_x = unname(coefficient_table["x", "Estimate"]),
    se = unname(coefficient_table["x", "Std. Error"])
  )
}

fit_slope <- function(data) {
  extract_x_estimate(fit_analysis_model(data))
}

truth <- fit_slope(full)

cat("\n=== Section 3: complete-data benchmark ===\n")
cat(sprintf("Generating beta_x: %.3f\n", TRUE_BETA_X))
cat(sprintf("Complete-data estimate: %.3f (SE %.3f)\n",
            truth["beta_x"], truth["se"]))


# =============================================================================
# 4.  TRANSLATE THE DATA INTO JOMO'S INPUTS
# -----------------------------------------------------------------------------
# Native jomo uses matrices/data frames rather than one formula:
#
#   Y    : joint outcomes. Put incomplete variables here. Fully observed y is
#          also placed here so its covariance with x helps impute x.
#
#   X    : fully observed fixed-effect predictors of every joint outcome.
#          Missing values are NOT allowed. An intercept is explicit.
#
#   Z    : predictors attached to random effects. We omit it, so jomo uses its
#          default column of 1s: a random intercept for every joint outcome.
#
#   clus : cluster membership. Supplying this makes the umbrella jomo()
#          function select a two-level imputation engine.
#
# Our joint imputation model is therefore:
#
#   (y_ij, x_ij)' = B'(1, z_ij) + u_j + e_ij
#
# with multivariate-normal level-2 random effects u_j and level-1 residuals
# e_ij. jomo estimates both covariance matrices.
# =============================================================================

make_jomo_inputs <- function(data, include_smoke = FALSE) {
  if (include_smoke) {
    Y <- data.frame(y = data$y, x = data$x, smoke = data$smoke)
  } else {
    Y <- data.frame(y = data$y, x = data$x)
  }

  X <- data.frame(
    intercept = rep(1, nrow(data)),
    z = data$z
  )

  clus <- data.frame(cluster_id = data$cluster_id)

  stopifnot(
    !anyNA(X),
    !anyNA(clus),
    is.numeric(Y$y),
    is.numeric(Y$x)
  )

  list(Y = Y, X = X, clus = clus)
}

inputs_mar <- make_jomo_inputs(dat_mar)

cat("\n=== Section 4: jomo input dimensions for MAR ===\n")
cat(sprintf("Y: %d rows x %d joint outcomes\n",
            nrow(inputs_mar$Y), ncol(inputs_mar$Y)))
cat(sprintf("X: %d rows x %d fixed-effect predictors\n",
            nrow(inputs_mar$X), ncol(inputs_mar$X)))
cat(sprintf("Clusters: %d\n", nlevels(factor(inputs_mar$clus[, 1]))))


# =============================================================================
# 5.  DRY RUN: CHECK THAT JOMO IS FITTING THE INTENDED MODEL
# -----------------------------------------------------------------------------
# The package authors recommend a very short .MCMCchain run before spending
# time on the real sampler. It answers structural questions:
#
#   - Did jomo recognize the data as clustered?
#   - Did it recognize two continuous joint outcomes?
#   - Are the fixed-effect and covariance arrays the expected dimensions?
#
# Two iterations are NOT enough for convergence. This is only a software/model
# specification check.
# =============================================================================

set.seed(500)
dry_run <- jomo::jomo.MCMCchain(
  Y = inputs_mar$Y,
  X = inputs_mar$X,
  clus = inputs_mar$clus,
  nburn = 2L,
  output = 0
)

cat("\n=== Section 5: dry-run output objects ===\n")
dry_dimensions <- lapply(
  dry_run[c("collectbeta", "collectomega", "collectu", "collectcovu")],
  dim
)
print(dry_dimensions)
cat("Expected: beta, level-1 covariance, random effects, and level-2 covariance.\n")


# =============================================================================
# 6.  RUN A LONGER MCMC CHAIN AND CHECK CONVERGENCE
# -----------------------------------------------------------------------------
# WHAT JOMO DOES UNDER THE HOOD:
#   - Unlike MICE, it does not start with a separate regression for each
#     incomplete variable. It posits ONE joint multivariate model.
#   - A Gibbs/data-augmentation sampler repeatedly draws model parameters,
#     covariance matrices, random effects, and missing values from their
#     conditional distributions under that joint model.
#   - The sampler must reach its stationary distribution before we save an
#     imputed dataset.
#
# The .MCMCchain function saves parameter draws instead of returning the final
# multiple imputations. We inspect three representative chains:
#
#   beta(z -> x)           : a fixed-effect parameter
#   residual cov(y, x)     : a level-1 covariance parameter
#   cluster cov(y, x)      : a level-2 covariance parameter
#
# Good signs: no drift, stable variation, reasonable mixing, and an ACF that
# falls toward zero. Inspect several parameters in a real analysis, especially
# covariance parameters, which often mix more slowly than fixed effects.
# =============================================================================

cat(sprintf("\n=== Section 6: running %d diagnostic MCMC iterations (MAR) ===\n",
            DIAGNOSTIC_ITERATIONS))

set.seed(501)
diagnostic_mar <- jomo::jomo.MCMCchain(
  Y = inputs_mar$Y,
  X = inputs_mar$X,
  clus = inputs_mar$clus,
  nburn = DIAGNOSTIC_ITERATIONS,
  output = 0
)

diagnostic_traces <- list(
  `fixed effect: z -> x` = as.numeric(
    diagnostic_mar$collectbeta["z", "x", ]
  ),
  `level-1 covariance: y,x` = as.numeric(
    diagnostic_mar$collectomega["y", "x", ]
  ),
  `level-2 covariance: y,x` = as.numeric(
    diagnostic_mar$collectcovu["y*Z1", "x*Z1", ]
  )
)

png("jomo_diagnostics_trace_MAR.png", width = 1000, height = 900)
par(mfrow = c(3, 1), mar = c(3.5, 4.2, 2.5, 1))
for (trace_name in names(diagnostic_traces)) {
  plot(
    diagnostic_traces[[trace_name]],
    type = "l",
    xlab = "MCMC iteration",
    ylab = "Parameter draw",
    main = trace_name
  )
}
dev.off()

png("jomo_diagnostics_acf_MAR.png", width = 1000, height = 900)
par(mfrow = c(3, 1), mar = c(3.5, 4.2, 2.5, 1))
for (trace_name in names(diagnostic_traces)) {
  acf(
    diagnostic_traces[[trace_name]],
    lag.max = 100,
    main = paste("ACF -", trace_name)
  )
}
dev.off()

# Show the remaining autocorrelation at the proposed distance between saved
# imputations. Values near zero suggest that N_BETWEEN is amply spaced for the
# inspected parameters. Always judge this together with the plots.
acf_at_lag <- function(x, lag) {
  as.numeric(acf(x, plot = FALSE, lag.max = lag)$acf[lag + 1L])
}

spacing_acf <- vapply(
  diagnostic_traces,
  acf_at_lag,
  numeric(1),
  lag = N_BETWEEN
)

cat(sprintf("\nAutocorrelation at proposed nbetween = %d:\n", N_BETWEEN))
print(round(spacing_acf, 3))

# Geweke is optional and complements rather than replaces trace/ACF review.
# A rough screening rule is |z| < 1.96, but passing it does not prove that the
# whole high-dimensional sampler has converged.
if (requireNamespace("coda", quietly = TRUE)) {
  geweke_z <- vapply(
    diagnostic_traces,
    function(x) coda::geweke.diag(coda::mcmc(x))$z,
    numeric(1)
  )
  cat("\nOptional Geweke z-scores:\n")
  print(round(geweke_z, 3))
  cat("Use |z| < 1.96 only as a rough screen; inspect the plots too.\n")
} else {
  cat("\nOptional Geweke diagnostic skipped: package 'coda' is not installed.\n")
}

cat("\nSaved jomo_diagnostics_trace_MAR.png and jomo_diagnostics_acf_MAR.png\n")
cat(sprintf(
  "Demo settings used below: nburn = %d, nbetween = %d, nimp = %d.\n",
  N_BURN, N_BETWEEN, N_IMPUTATIONS
))
cat("Change these only after reviewing your own chain diagnostics.\n")


# =============================================================================
# 7.  GENERATE MULTIPLE IMPUTATIONS WITH JOMO
# -----------------------------------------------------------------------------
# `nburn`    = MCMC updates before the first saved completed dataset.
# `nbetween` = MCMC updates between successive completed datasets.
# `nimp`     = number of completed datasets retained.
#
# WHY nimp = 5 INSTEAD OF 1:
# A single completed dataset hides uncertainty about the missing values.
# Multiple datasets are allowed to disagree. Rubin's rules later combine:
#
#   - within-imputation uncertainty from each fitted model, and
#   - between-imputation uncertainty from disagreement across imputations.
#
# Five is useful for a quick teaching example, not necessarily enough for the
# final NHANES project. The final number should reflect missing information and
# the desired Monte Carlo precision.
# =============================================================================

run_jomo <- function(data, seed) {
  inputs <- make_jomo_inputs(data)
  set.seed(seed)

  jomo::jomo(
    Y = inputs$Y,
    X = inputs$X,
    clus = inputs$clus,
    nburn = N_BURN,
    nbetween = N_BETWEEN,
    nimp = N_IMPUTATIONS,
    meth = "common",  # common level-1 covariance matrix across clusters
    output = 0
  )
}

cat("\n=== Section 7: creating the final imputations ===\n")
imp_mcar <- run_jomo(dat_mcar, seed = 701)
imp_mar <- run_jomo(dat_mar, seed = 702)
imp_mnar <- run_jomo(dat_mnar, seed = 703)

cat("Created MCAR, MAR, and MNAR jomo objects.\n")

# Native jomo returns LONG data:
#   Imputation == 0 : original incomplete data
#   Imputation == 1,...,5 : completed datasets
cat("\nImputation labels in the MAR object:\n")
print(table(imp_mar$Imputation))


# =============================================================================
# 8.  INSPECT THE COMPLETED DATASETS
# -----------------------------------------------------------------------------
# We split jomo's stacked output into a list, restore the analysis-friendly
# cluster_id name, and then verify four basic properties:
#
#   1. all missing x values were filled,
#   2. originally observed x values were not changed,
#   3. imputations differ from one another, and
#   4. the cluster structure is still present.
# =============================================================================

jomo_completed_list <- function(jomo_output) {
  imputation_numbers <- sort(unique(
    jomo_output$Imputation[jomo_output$Imputation > 0]
  ))

  completed <- lapply(imputation_numbers, function(k) {
    out <- jomo_output[jomo_output$Imputation == k, , drop = FALSE]
    out <- out[order(out$id), , drop = FALSE]
    out$cluster_id <- factor(out$clus)
    rownames(out) <- NULL
    out
  })

  names(completed) <- paste0("imputation_", imputation_numbers)
  completed
}

completed_mcar <- jomo_completed_list(imp_mcar)
completed_mar <- jomo_completed_list(imp_mar)
completed_mnar <- jomo_completed_list(imp_mnar)

mar_missing_rows <- which(is.na(dat_mar$x))
mar_observed_rows <- which(!is.na(dat_mar$x))

imputed_x_mar <- vapply(
  completed_mar,
  function(data) data$x[mar_missing_rows],
  numeric(length(mar_missing_rows))
)

cat("\n=== Section 8: first missing x cells across five MAR imputations ===\n")
print(round(head(imputed_x_mar, 6), 2))

all_x_filled <- all(vapply(
  completed_mar,
  function(data) !anyNA(data$x),
  logical(1)
))

largest_change_to_observed_x <- max(vapply(
  completed_mar,
  function(data) max(abs(
    data$x[mar_observed_rows] - dat_mar$x[mar_observed_rows]
  )),
  numeric(1)
))

cat(sprintf("\nAll MAR x values filled: %s\n", all_x_filled))
cat(sprintf("Largest change to an originally observed x: %.12f\n",
            largest_change_to_observed_x))

# A simple intraclass correlation (ICC) check for x. The five imputed ICCs do
# not need to equal the complete-data ICC exactly, but they should retain the
# clear within-cluster dependence instead of treating every row as independent.
icc_x <- function(data) {
  model <- lme4::lmer(
    x ~ 1 + (1 | cluster_id),
    data = data,
    REML = TRUE,
    na.action = na.omit,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )
  variance_table <- as.data.frame(lme4::VarCorr(model))
  between <- variance_table$vcov[variance_table$grp == "cluster_id"][1]
  within <- variance_table$vcov[variance_table$grp == "Residual"][1]
  between / (between + within)
}

complete_icc <- icc_x(full)
imputed_iccs <- vapply(completed_mar, icc_x, numeric(1))

cat(sprintf("\nComplete-data ICC for x: %.3f\n", complete_icc))
cat("ICC for x in each MAR imputation:\n")
print(round(imputed_iccs, 3))

# Compare both observed and truly missing x values with the pooled imputed
# draws. Under MAR, the observed and imputed distributions need not be
# identical: the missing rows were systematically selected using y and z.
observed_x_mar <- dat_mar$x[mar_observed_rows]
true_missing_x_mar <- full$x[mar_missing_rows]
pooled_imputed_x_mar <- as.numeric(imputed_x_mar)

png("jomo_imputed_vs_true_MAR.png", width = 1100, height = 500)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(
  density(observed_x_mar),
  lwd = 2,
  col = "steelblue",
  main = "Observed vs imputed x (MAR)",
  xlab = "x"
)
lines(density(pooled_imputed_x_mar), lwd = 2, col = "firebrick")
legend(
  "topright",
  legend = c("Observed x", "Imputed x draws"),
  col = c("steelblue", "firebrick"),
  lwd = 2,
  bty = "n"
)

plot(
  density(true_missing_x_mar),
  lwd = 2,
  col = "darkgreen",
  main = "Hidden truth vs imputed x (MAR)",
  xlab = "x"
)
lines(density(pooled_imputed_x_mar), lwd = 2, col = "firebrick")
legend(
  "topright",
  legend = c("True hidden x", "Imputed x draws"),
  col = c("darkgreen", "firebrick"),
  lwd = 2,
  bty = "n"
)

dev.off()
cat("\nSaved jomo_imputed_vs_true_MAR.png\n")


# =============================================================================
# 9.  FIT EACH IMPUTED DATASET AND POOL WITH RUBIN'S RULES
# -----------------------------------------------------------------------------
# The substantive analysis model is fitted separately to all five completed
# datasets. Nothing is pooled until all five model fits exist.
#
# For the x coefficient:
#
#   Qbar = average of the five coefficient estimates
#   Ubar = average of the five estimated variances (SE^2)
#   B    = variance of the five coefficient estimates
#   T    = Ubar + (1 + 1/m)*B
#
# The total SE is sqrt(T). The B term is the extra uncertainty caused by not
# knowing the missing x values. A single imputation would omit that term.
#
# This helper implements the standard large-sample Rubin pooling formulas so
# the arithmetic is visible. In the full project we can use mitml or another
# tested pooling layer, especially for multi-parameter tests and small-sample
# degrees-of-freedom adjustments.
# =============================================================================

fit_all_imputations <- function(completed_list) {
  lapply(completed_list, fit_analysis_model)
}

pool_x_with_rubin <- function(fitted_models) {
  m <- length(fitted_models)

  estimates <- vapply(
    fitted_models,
    function(model) unname(lme4::fixef(model)["x"]),
    numeric(1)
  )

  variances <- vapply(
    fitted_models,
    function(model) unname(as.matrix(stats::vcov(model))["x", "x"]),
    numeric(1)
  )

  q_bar <- mean(estimates)
  u_bar <- mean(variances)
  b <- stats::var(estimates)
  total_variance <- u_bar + (1 + 1 / m) * b
  pooled_se <- sqrt(total_variance)

  if (b <= .Machine$double.eps) {
    relative_increase <- 0
    df <- Inf
    fmi <- 0
  } else {
    relative_increase <- ((1 + 1 / m) * b) / u_bar
    df <- (m - 1) * (1 + 1 / relative_increase)^2
    fmi <- (relative_increase + 2 / (df + 3)) /
      (relative_increase + 1)
  }

  critical_value <- stats::qt(0.975, df = df)

  list(
    estimates = estimates,
    variances = variances,
    summary = c(
      beta_x = q_bar,
      se = pooled_se,
      df = df,
      fmi = fmi,
      ci_low = q_bar - critical_value * pooled_se,
      ci_high = q_bar + critical_value * pooled_se,
      within_variance = u_bar,
      between_variance = b,
      total_variance = total_variance
    )
  )
}

fits_mcar <- fit_all_imputations(completed_mcar)
fits_mar <- fit_all_imputations(completed_mar)
fits_mnar <- fit_all_imputations(completed_mnar)

pool_mcar <- pool_x_with_rubin(fits_mcar)
pool_mar <- pool_x_with_rubin(fits_mar)
pool_mnar <- pool_x_with_rubin(fits_mnar)

# The same native jomo output can instead be handed to mitml. This is the
# package-supported route to use later when you want more general pooled tests:
#
# if (requireNamespace("mitml", quietly = TRUE)) {
#   imp_mar_mitml <- mitml::jomo2mitml.list(imp_mar)
#   fits_mar_mitml <- with(
#     imp_mar_mitml,
#     lme4::lmer(y ~ x + z + (1 | clus), REML = FALSE)
#   )
#   print(mitml::testEstimates(fits_mar_mitml))
# }

cat("\n=== Section 9: the five MAR estimates before pooling ===\n")
print(round(pool_mar$estimates, 4))

cat("\nMAR Rubin components:\n")
cat(sprintf("Average estimate (Qbar)      : %.4f\n", pool_mar$summary["beta_x"]))
cat(sprintf("Within-imputation variance  : %.6f\n",
            pool_mar$summary["within_variance"]))
cat(sprintf("Between-imputation variance : %.6f\n",
            pool_mar$summary["between_variance"]))
cat(sprintf("Total variance              : %.6f\n",
            pool_mar$summary["total_variance"]))
cat(sprintf("Pooled SE                   : %.4f\n", pool_mar$summary["se"]))
cat(sprintf("Fraction missing information: %.3f\n", pool_mar$summary["fmi"]))


# =============================================================================
# 10. COMPARE COMPLETE DATA, COMPLETE CASES, AND JOMO
# -----------------------------------------------------------------------------
# What we expect across repeated simulations:
#
#   MCAR: complete cases and jomo are both approximately unbiased. Jomo can
#         recover information and therefore may improve efficiency.
#
#   MAR : complete-case analysis can be biased here because whether x is
#         observed depends on y. Jomo conditions on the observed y and z and
#         should move the estimate back toward the complete-data result.
#
#   MNAR: ordinary jomo and complete-case analysis can both be biased. The
#         missingness still depends on the unseen x after conditioning on the
#         imputation variables. A plausible-looking SE cannot repair a wrong
#         missingness assumption.
#
# One random toy sample will not reproduce the exact theoretical ordering every
# time. Judge the pattern, not whether every printed decimal is perfect.
# =============================================================================

normal_ci <- function(estimate, se) {
  estimate + c(-1, 1) * stats::qnorm(0.975) * se
}

comparison_rows <- function(scenario, incomplete_data, pooled_object) {
  cc <- fit_slope(incomplete_data)
  mi <- pooled_object$summary

  truth_ci <- normal_ci(truth["beta_x"], truth["se"])
  cc_ci <- normal_ci(cc["beta_x"], cc["se"])

  rbind(
    data.frame(
      scenario = scenario,
      method = "complete data",
      beta_x = unname(truth["beta_x"]),
      se = unname(truth["se"]),
      df = NA_real_,
      fmi = NA_real_,
      ci_low = truth_ci[1],
      ci_high = truth_ci[2]
    ),
    data.frame(
      scenario = scenario,
      method = "complete case",
      beta_x = unname(cc["beta_x"]),
      se = unname(cc["se"]),
      df = NA_real_,
      fmi = NA_real_,
      ci_low = cc_ci[1],
      ci_high = cc_ci[2]
    ),
    data.frame(
      scenario = scenario,
      method = "jomo pooled",
      beta_x = unname(mi["beta_x"]),
      se = unname(mi["se"]),
      df = unname(mi["df"]),
      fmi = unname(mi["fmi"]),
      ci_low = unname(mi["ci_low"]),
      ci_high = unname(mi["ci_high"])
    )
  )
}

results <- rbind(
  comparison_rows("MCAR", dat_mcar, pool_mcar),
  comparison_rows("MAR", dat_mar, pool_mar),
  comparison_rows("MNAR", dat_mnar, pool_mnar)
)

results$bias_from_generating_beta <- results$beta_x - TRUE_BETA_X
results$difference_from_complete_data <- results$beta_x - truth["beta_x"]
rownames(results) <- NULL

cat("\n=== Section 10: THE PAYOFF TABLE (generating beta_x = 0.8) ===\n")
print(format(results, digits = 3, nsmall = 3), row.names = FALSE)

cat("\nHow to read the table:\n")
cat(" * Compare beta_x with both 0.800 and the finite-sample complete-data row.\n")
cat(" * MCAR: CC and jomo should both be approximately unbiased.\n")
cat(" * MAR : jomo should usually recover more of the outcome-dependent loss.\n")
cat(" * MNAR: neither ordinary MAR-based approach is guaranteed to recover truth.\n")
cat(" * FMI quantifies how much pooled uncertainty is attributable to missingness.\n")


# =============================================================================
# 11. OPTIONAL: ADD AN INCOMPLETE BINARY VARIABLE
# -----------------------------------------------------------------------------
# Change RUN_CATEGORICAL_EXTENSION near the top to TRUE after the continuous
# example is understood. jomo will see numeric y/x plus factor smoke and select
# a mixed-data engine (jomo1ranmix). It represents binary smoke with an
# underlying latent normal variable during MCMC, then converts imputed latent
# draws back to the observed categories.
#
# This block demonstrates data-type handling only. If smoke were part of the
# substantive analysis, that analysis would also need to be fitted and pooled.
# =============================================================================

if (RUN_CATEGORICAL_EXTENSION) {
  mixed_data <- dat_mar
  set.seed(1101)
  remove_smoke <- runif(nrow(mixed_data)) < 0.20
  mixed_data$smoke[remove_smoke] <- NA

  mixed_inputs <- make_jomo_inputs(mixed_data, include_smoke = TRUE)

  stopifnot(is.factor(mixed_inputs$Y$smoke))

  cat("\n=== Section 11: optional continuous + binary imputation ===\n")
  cat(sprintf("Missing smoke values: %d\n", sum(is.na(mixed_data$smoke))))
  cat("jomo should report that it selected jomo1ranmix.\n")

  set.seed(1102)
  imp_mixed <- jomo::jomo(
    Y = mixed_inputs$Y,
    X = mixed_inputs$X,
    clus = mixed_inputs$clus,
    nburn = N_BURN,
    nbetween = N_BETWEEN,
    nimp = N_IMPUTATIONS,
    meth = "common",
    output = 1
  )

  cat("\nSmoke distribution across completed datasets:\n")
  print(prop.table(table(
    imp_mixed$smoke[imp_mixed$Imputation > 0],
    useNA = "ifany"
  )))
}


# =============================================================================
# 12. WHAT THIS DOES -- AND DOES NOT -- ESTABLISH FOR NHANES
# -----------------------------------------------------------------------------
# This example establishes the mechanics needed for the next project step:
#
#   - create Y / X / clus inputs,
#   - preserve a two-level dependence structure,
#   - run a dry check,
#   - diagnose MCMC convergence,
#   - choose nburn / nbetween,
#   - create multiple completed datasets,
#   - fit and pool the substantive model.
#
# It does NOT establish that setting clus = PSU fully incorporates the NHANES
# sample design. For the real project we still need to decide, justify, and
# test how to handle:
#
#   - a verified unique PSU identifier within the combined cycles,
#   - strata,
#   - examination/interview/subsample weights,
#   - survey-weighted substantive analyses after imputation, and
#   - whether design variables enter as fixed predictors, grouping variables,
#     or through some other survey-aware imputation strategy.
#
# Ordinary jomo assumes MAR given the imputation-model information. MCAR was a
# convenient first scenario, not an assumption required by the method. MNAR
# requires explicit sensitivity analysis rather than a standard jomo run.
# =============================================================================

cat("\nDone. Re-read Sections 4, 6, 7, 9, 10, and 12 for the teach-back.\n")
cat("Core sentence: jomo fits one multilevel joint distribution, uses MCMC to\n")
cat("draw missing values and parameters, and Rubin's rules carry imputation\n")
cat("uncertainty into the final pooled estimate.\n")
