# =============================================================================
# JOMO: SIMPLE MULTILEVEL MULTIPLE IMPUTATION EXAMPLE
# -----------------------------------------------------------------------------
# This version is designed to be run line-by-line, like the SMC-FCS example.
# It uses one MAR scenario and contains no user-written functions.
#
# The purpose is not just to obtain an answer. Each section isolates one stage
# of the multiple-imputation workflow so a new user can see the distinction
# between:
#   1. creating an imputation model,
#   2. checking the MCMC algorithm,
#   3. drawing several completed datasets,
#   4. fitting the substantive analysis model, and
#   5. pooling the analysis results with Rubin's rules.
#
# Run one numbered section at a time and inspect the objects it creates. The
# comments explain both what each line does and what a sensible result should
# look like.
#
# Required packages: jomo, lme4, mitml, coda
# =============================================================================

library(jomo)
library(lme4)
library(mitml)
library(coda)

set.seed(42)

# When the script is run with source(), sys.frame(1)$ofile contains the path to
# jomo_demo.R. This anchors the output folder beside the script rather than in
# whatever parent folder happens to be the current working directory.
if (sys.nframe() > 0 && !is.null(sys.frame(1)$ofile)) {
  project_directory <- dirname(normalizePath(sys.frame(1)$ofile))
} else {
  # Fallback when commands are run individually in the console.
  project_directory <- getwd()
}

output_directory <- file.path(project_directory, "jomo_example_output")
dir.create(output_directory, showWarnings = FALSE)

cat("\nJOMO example output folder:\n")
cat(normalizePath(output_directory), "\n")


# =============================================================================
# 1. BUILD A COMPLETE CLUSTERED DATASET
# -----------------------------------------------------------------------------
# There are 40 clusters with 25 people in each cluster, giving 1,000 rows.
# Think of a cluster as a school, clinic, neighborhood, or survey PSU, and each
# row as a person within that cluster. The people are the level-1 units and the
# clusters are the level-2 units.
#
# Both x and y contain cluster-specific random effects. Everyone in a cluster
# shares the same cluster effect, which makes observations within that cluster
# more alike than observations from different clusters. This is the dependence
# structure that motivates multilevel rather than single-level imputation.
#
# The final analysis model is:
#
#   y ~ x + z + (1 | cluster_id)
#
# The data-generating equations are approximately:
#
#   x_ij = 0.5*z_ij + cluster effect for x + individual error
#
#   y_ij = 2 + 0.8*x_ij + 1.5*z_ij
#          + cluster effect for y + individual error
#
# The subscript i represents a person and j represents a cluster. The true
# conditional coefficient of x in the y model is 0.8. Because the dataset is a
# random finite sample, the complete-data fitted estimate will be near 0.8 but
# will not normally equal it exactly.
# =============================================================================

n_clusters <- 40
people_per_cluster <- 25
n <- n_clusters * people_per_cluster

cluster_number <- rep(1:n_clusters, each = people_per_cluster)
cluster_id <- factor(cluster_number)

cluster_effect_x <- rnorm(n_clusters, mean = 0, sd = 3)
cluster_effect_y <- rnorm(n_clusters, mean = 0, sd = 4)

z <- rnorm(n, mean = 50, sd = 10)

x <- 0.5 * z +
  cluster_effect_x[cluster_number] +
  rnorm(n, mean = 0, sd = 5)

y <- 2 + 0.8 * x + 1.5 * z +
  cluster_effect_y[cluster_number] +
  rnorm(n, mean = 0, sd = 3)

# smoke is used only in the optional categorical section at the end.
p_smoke <- plogis(-0.5 + 0.03 * (z - 50) + 0.04 * (x - mean(x)))
smoke <- factor(
  rbinom(n, size = 1, prob = p_smoke),
  levels = c(0, 1),
  labels = c("no", "yes")
)

full <- data.frame(cluster_id, x, y, z, smoke)

cat("\n=== Section 1: complete clustered dataset ===\n")
print(head(full))
cat("\nRows:", nrow(full),
    "| clusters:", nlevels(full$cluster_id),
    "| observations per cluster:", people_per_cluster, "\n")

# The complete-data analysis is our best benchmark for this particular sample.
# The number 0.8 is the population-generating parameter, while complete_beta_x
# is what we would have estimated if no x values had been deleted. Later, JOMO
# should recover the complete-data result as closely as possible while also
# accounting for uncertainty about the missing values.
true_beta_x <- 0.8

complete_model <- lmer(
  y ~ x + z + (1 | cluster_id),
  data = full,
  REML = FALSE
)

complete_beta_x <- fixef(complete_model)["x"]
complete_se_x <- sqrt(vcov(complete_model)["x", "x"])

cat("\nGenerating beta_x:", true_beta_x, "\n")
cat("Complete-data estimate:", round(complete_beta_x, 3), "\n")


# =============================================================================
# 2. MAKE x MISSING UNDER MAR
# -----------------------------------------------------------------------------
# We deliberately hide some x values so that the original values remain known
# to us for evaluation. In real data, the missing values would be unknown.
#
# The probability that x is missing depends on y and z. Both variables remain
# fully observed. Therefore, once we condition on y and z, the missingness does
# not depend on the unseen value of x; this is a Missing At Random (MAR)
# mechanism.
#
# y receives most of the weight in mar_score. Consequently, missingness is
# related to the outcome, which can make complete-case analysis less reliable.
# JOMO is given both y and z and can use that information during imputation.
#
# scale() standardizes y and z before combining them so that their different
# units do not determine their influence. plogis() converts the score into a
# probability between zero and one. The intercept of -1 produces approximately
# 30% missingness, although the realized percentage varies randomly.
# =============================================================================

mar_score <- 0.85 * as.numeric(scale(y)) +
  0.15 * as.numeric(scale(z))

p_x_missing <- plogis(-1 + mar_score)
remove_x <- runif(n) < p_x_missing

dat_mar <- full
dat_mar$x[remove_x] <- NA

cat("\nMissing x:", sum(is.na(dat_mar$x)), "of", n, "rows\n")
cat("Percent missing:", round(100 * mean(is.na(dat_mar$x)), 1), "%\n")


# =============================================================================
# 3. WHAT JOMO IS DOING
# -----------------------------------------------------------------------------
# JOMO stands for Joint Modelling Multiple Imputation. The word "joint" is the
# key distinction. MICE generally cycles through variables and specifies a
# separate conditional regression for each incomplete variable. JOMO instead
# assumes that the variables in Y arise together from one multivariate model.
#
# Here, y and x are modeled jointly. Although y is complete, its covariance
# with x helps JOMO predict the missing x values. In other words, y does not
# need to be missing to be useful inside Y.
#
# Supplying cluster membership tells JOMO to estimate both within-cluster and
# between-cluster variation. The joint model can be summarized as:
#
#   (y_ij, x_ij)' = fixed effects of z + cluster random effects + residuals
#
# JOMO estimates covariance matrices for both the cluster random effects and
# the individual residuals. Those covariance matrices describe how y and x
# move together within clusters and between clusters.
#
# Estimation uses Markov chain Monte Carlo (MCMC). Conceptually, each iteration
# cycles through draws of:
#   - the missing x values given the current parameters and observed data;
#   - fixed-effect coefficients;
#   - cluster-specific random effects;
#   - within-cluster and between-cluster covariance matrices.
#
# Repeating these steps creates a Markov chain whose stable distribution is the
# posterior distribution under the joint model. Final imputations are draws
# from that distribution rather than single deterministic predictions.
#
# JOMO only creates the completed datasets. It does not by itself produce the
# final substantive coefficient of x. We obtain that coefficient later by
# fitting the analysis model to every completed dataset and pooling the fits.
# =============================================================================


# =============================================================================
# 4. CREATE THE JOMO INPUTS
# -----------------------------------------------------------------------------
# Native JOMO does not use one analysis-style formula. Instead, we explicitly
# construct the pieces of the joint imputation model.
#
# Y = variables modeled jointly. Missing values are allowed in Y. x belongs
#     here because it is incomplete. y also belongs here because its covariance
#     with x supplies predictive information about the missing x values.
#
# X = fully observed fixed-effect predictors for the joint outcomes. We include
#     an explicit intercept and z. Missing values are not allowed in X because
#     JOMO conditions on these predictors rather than imputing them in this
#     model. x is not placed in X because it is itself an incomplete joint
#     outcome.
#
# clus = one cluster identifier per row. Supplying clus activates JOMO's
#        two-level model and allows different clusters to have different random
#        intercepts.
#
# JOMO also has an argument called Z for the random-effect design matrix. We
# omit it here, so JOMO uses a column of ones: a random intercept for y and a
# random intercept for x. A more advanced model could supply Z to request
# random slopes.
# =============================================================================

Y <- data.frame(
  y = dat_mar$y,
  x = dat_mar$x
)

X <- data.frame(
  intercept = rep(1, n),
  z = dat_mar$z
)

clus <- data.frame(
  cluster_id = dat_mar$cluster_id
)

cat("\n=== Section 4: JOMO input dimensions ===\n")
cat("Y:", nrow(Y), "rows x", ncol(Y), "joint outcomes\n")
cat("X:", nrow(X), "rows x", ncol(X), "fixed-effect columns\n")
cat("Clusters:", length(unique(clus$cluster_id)), "\n")

# Expected dimensions are 1,000 x 2 for both Y and X, with 40 clusters. When
# the sampler runs, JOMO should report jomo1rancon: "ran" indicates random
# effects and "con" indicates that all joint outcomes are continuous.


# =============================================================================
# 5. CHECK MCMC CONVERGENCE
# -----------------------------------------------------------------------------
# Before using draws as imputations, we need evidence that the Markov chain has
# reached a stable region and is mixing adequately. If the chain is still
# drifting from its starting values, the resulting imputations may depend too
# heavily on initialization rather than the intended posterior distribution.
#
# jomo.MCMCchain() runs the same underlying sampler used for imputation but
# returns its parameter history for diagnostic inspection. It does not create
# the final five completed datasets used in the analysis.
#
# JOMO estimates many parameters, so a full analysis should inspect several of
# them. For a manageable teaching example, we select three representative
# chains:
#   - beta_trace: a fixed effect describing the relationship between z and x;
#   - level1_covariance_trace: association between y and x among individuals
#     after accounting for fixed effects and cluster random effects;
#   - level2_covariance_trace: association between the cluster random effects
#     for y and x.
#
# Covariance parameters are included because they can mix more slowly than
# ordinary regression coefficients and are central to multilevel imputation.
# =============================================================================

set.seed(500)

diagnostic_chain <- jomo.MCMCchain(
  Y = Y,
  X = X,
  clus = clus,
  nburn = 5000,
  output = 0
)

beta_trace <- as.numeric(
  diagnostic_chain$collectbeta["z", "x", ]
)

level1_covariance_trace <- as.numeric(
  diagnostic_chain$collectomega["y", "x", ]
)

level2_covariance_trace <- as.numeric(
  diagnostic_chain$collectcovu["y*Z1", "x*Z1", ]
)

# A trace plot displays the sampled parameter value at every iteration. A
# reassuring chain resembles a stable, irregular band: it moves around its
# typical value without a continuing upward trend, downward trend, or long
# periods stuck at one value. The band does not need to be perfectly smooth.
trace_plot_file <- file.path(
  output_directory,
  "jomo_trace_plots_MAR.png"
)

png(trace_plot_file, width = 1200, height = 900, res = 120)
par(mfrow = c(3, 1))
plot(beta_trace, type = "l", main = "Fixed effect: z -> x")
plot(level1_covariance_trace, type = "l", main = "Level-1 covariance: y, x")
plot(level2_covariance_trace, type = "l", main = "Level-2 covariance: y, x")
dev.off()

# An autocorrelation function (ACF) plot measures how strongly a draw is related
# to earlier draws. High autocorrelation at many lags means that the chain is
# moving slowly and contains less independent information. We want the bars to
# decline toward zero. This helps choose nbetween, the number of MCMC updates
# separating two saved completed datasets.
acf_plot_file <- file.path(
  output_directory,
  "jomo_acf_plots_MAR.png"
)

png(acf_plot_file, width = 1200, height = 900, res = 120)
par(mfrow = c(3, 1))
acf(beta_trace, lag.max = 100, main = "ACF: fixed effect")
acf(level1_covariance_trace, lag.max = 100, main = "ACF: level 1")
acf(level2_covariance_trace, lag.max = 100, main = "ACF: level 2")
dev.off()

cat("\n=== Section 5: saved MCMC diagnostics ===\n")
cat("Trace plots:", normalizePath(trace_plot_file), "\n")
cat("ACF plots  :", normalizePath(acf_plot_file), "\n")
cat("Trace file exists:", file.exists(trace_plot_file), "\n")
cat("ACF file exists  :", file.exists(acf_plot_file), "\n")

# The Geweke diagnostic compares the mean of an early portion of a chain with
# the mean of a later portion and reports a standardized z-score. A value within
# roughly -1.96 to 1.96 is a useful screen for stability. It is not proof of
# convergence: it examines only one chain at a time and can miss problems that
# are visible in the trace or ACF plots. The numerical result and plots should
# therefore be interpreted together.
geweke_results <- c(
  fixed_effect = geweke.diag(mcmc(beta_trace))$z,
  level1_covariance = geweke.diag(mcmc(level1_covariance_trace))$z,
  level2_covariance = geweke.diag(mcmc(level2_covariance_trace))$z
)

cat("\nGeweke z-scores:\n")
print(round(geweke_results, 3))


# =============================================================================
# 6. CREATE FIVE IMPUTED DATASETS
# -----------------------------------------------------------------------------
# This is the actual multiple-imputation run. Its three most important tuning
# arguments have different jobs:
#
# nburn = MCMC iterations completed before the first imputation is retained.
#         These early iterations allow the sampler to move away from its
#         starting values. They are not treated as completed datasets.
#
# nbetween = MCMC iterations separating successive saved imputations. Spacing
#            reduces dependence between the completed datasets. The ACF plots
#            provide information for this choice.
#
# nimp = number of completed datasets retained. Each dataset contains a
#        different plausible draw for every missing value. Five is convenient
#        for learning, but a final study would usually use more to reduce Monte
#        Carlo error in the pooled estimates and FMI.
#
# meth = "common" assumes that the level-1 residual covariance matrix is shared
# across clusters. This is a simpler and more stable choice for the toy example.
# It does not mean that every cluster has the same random intercept.
# =============================================================================

set.seed(600)

imputed_long <- jomo(
  Y = Y,
  X = X,
  clus = clus,
  nburn = 1000,
  nbetween = 1000,
  nimp = 5,
  meth = "common",
  output = 0
)

# JOMO returns a stacked, or long-format, object. Imputation 0 reproduces the
# original incomplete data, while Imputations 1-5 contain completed versions.
# With 1,000 original observations, each label should therefore appear 1,000
# times. The completed datasets agree on observed values but may disagree on
# imputed values.
cat("\n=== Section 6: rows in each JOMO dataset ===\n")
print(table(imputed_long$Imputation))

# jomo2mitml.list() converts that stacked output into a more analysis-friendly
# list, analogous to the imputation-list object used in the SMC-FCS example.
# Each list element is one ordinary completed data frame.
imputed_sets <- jomo2mitml.list(imputed_long)

imp1 <- imputed_sets[[1]]
imp2 <- imputed_sets[[2]]
imp3 <- imputed_sets[[3]]
imp4 <- imputed_sets[[4]]
imp5 <- imputed_sets[[5]]


# =============================================================================
# 7. INSPECT, ANALYZE, AND POOL
# -----------------------------------------------------------------------------
# Multiple imputation is not finished when JOMO returns five datasets. We must:
#   1. verify that the completed data behave as intended;
#   2. fit the same substantive model in every dataset; and
#   3. combine those model results rather than choosing one completed dataset.
#
# Two basic integrity checks come first. Every originally missing x should be
# filled, and every originally observed x should remain exactly unchanged.
# JOMO is meant to replace NAs, not revise known measurements.
# =============================================================================

missing_rows <- which(is.na(dat_mar$x))
observed_rows <- which(!is.na(dat_mar$x))

# Each row below corresponds to one originally missing x cell, and each column
# is its value in a different imputation. The columns should not be identical.
# Their disagreement is intentional: it represents uncertainty about what the
# missing value could have been given y, z, and the cluster structure.
x_imputations <- cbind(
  imputation_1 = imp1$x[missing_rows],
  imputation_2 = imp2$x[missing_rows],
  imputation_3 = imp3$x[missing_rows],
  imputation_4 = imp4$x[missing_rows],
  imputation_5 = imp5$x[missing_rows]
)

cat("\n=== Section 7: first missing x values across imputations ===\n")
print(head(round(x_imputations, 2)))

# Expected results are TRUE and 0. A nonzero change to observed x would indicate
# that the extraction or row alignment needs investigation. A tiny value caused
# only by floating-point representation could be harmless, but JOMO normally
# retains observed values exactly.
all_x_filled <- all(!is.na(x_imputations))

largest_change_to_observed_x <- max(abs(c(
  imp1$x[observed_rows] - dat_mar$x[observed_rows],
  imp2$x[observed_rows] - dat_mar$x[observed_rows],
  imp3$x[observed_rows] - dat_mar$x[observed_rows],
  imp4$x[observed_rows] - dat_mar$x[observed_rows],
  imp5$x[observed_rows] - dat_mar$x[observed_rows]
)))

cat("\nAll missing x values filled:", all_x_filled, "\n")
cat("Largest change to an observed x:",
    format(largest_change_to_observed_x, scientific = FALSE), "\n")

# We now fit the substantive model (the model answering the scientific
# question) to all five datasets. It must be identical across imputations.
# Changing the formula from one dataset to another would make pooling
# meaningless. JOMO names the cluster variable clus in its completed output,
# so the random-intercept term uses (1 | clus) here.
models <- with(
  imputed_sets,
  lmer(y ~ x + z + (1 | clus), REML = FALSE)
)

# This distinction is important: JOMO's direct output is imputed data, not the
# final beta_x. The five lmer models estimate beta_x. Rubin's rules then combine
# those five estimates into the result attributed to the JOMO workflow.
# testEstimates() performs package-based pooling for all fixed effects and
# reports pooled estimates, standard errors, degrees of freedom, relative
# increases in variance, and fractions of missing information.
pooled_models <- testEstimates(models)

cat("\n=== Package-pooled model results ===\n")
print(pooled_models)

# We also extract beta_x and its estimated sampling variance from every model so
# the pooling calculation is visible rather than hidden inside testEstimates().
beta_x_each <- c(
  fixef(models[[1]])["x"],
  fixef(models[[2]])["x"],
  fixef(models[[3]])["x"],
  fixef(models[[4]])["x"],
  fixef(models[[5]])["x"]
)

variance_x_each <- c(
  vcov(models[[1]])["x", "x"],
  vcov(models[[2]])["x", "x"],
  vcov(models[[3]])["x", "x"],
  vcov(models[[4]])["x", "x"],
  vcov(models[[5]])["x", "x"]
)

cat("\nThe five x estimates before pooling:\n")
print(round(beta_x_each, 4))

# Rubin's rules use four central quantities:
#
# Qbar = average of the five beta_x estimates.
# Ubar = average variance within the five fitted models.
# B    = variance of beta_x across the five imputations.
# T    = Ubar + (1 + 1/m)*B, the total pooled variance.
#
# Ubar represents ordinary complete-data sampling uncertainty. B represents
# additional uncertainty caused by not knowing the missing x values. The pooled
# standard error is sqrt(T), so a single imputation would generally understate
# uncertainty by omitting the between-imputation component.
m <- length(beta_x_each)
Qbar <- mean(beta_x_each)
Ubar <- mean(variance_x_each)
B <- var(beta_x_each)
total_variance <- Ubar + (1 + 1 / m) * B
jomo_pooled_se <- sqrt(total_variance)

if (B > .Machine$double.eps) {
  relative_increase <- ((1 + 1 / m) * B) / Ubar
  jomo_df <- (m - 1) * (1 + 1 / relative_increase)^2
  jomo_fmi <- (relative_increase + 2 / (jomo_df + 3)) /
    (relative_increase + 1)
  critical_value <- qt(0.975, df = jomo_df)
} else {
  jomo_df <- Inf
  jomo_fmi <- 0
  critical_value <- qnorm(0.975)
}

jomo_ci <- Qbar + c(-1, 1) * critical_value * jomo_pooled_se

cat("\nExplicit JOMO result for the x coefficient:\n")
cat("JOMO pooled beta_x:", round(Qbar, 4), "\n")
cat("Pooled SE         :", round(jomo_pooled_se, 4), "\n")
cat("95% CI            :", round(jomo_ci[1], 4), "to",
    round(jomo_ci[2], 4), "\n")
cat("Degrees of freedom:", round(jomo_df, 2), "\n")
cat("FMI               :", round(jomo_fmi, 3), "\n")

# Finally, compare three analyses:
#   - complete data: the benchmark before x was hidden;
#   - complete case: uses only rows where x remained observed;
#   - JOMO pooled: uses all rows after multiple imputation and carries the
#     imputation uncertainty into its standard error.
#
# The generating value 0.8 is the population target. The complete-data estimate
# is the fairest benchmark for judging missing-data recovery in this particular
# realized sample.
complete_cases <- dat_mar[!is.na(dat_mar$x), ]

complete_case_model <- lmer(
  y ~ x + z + (1 | cluster_id),
  data = complete_cases,
  REML = FALSE
)

complete_case_beta_x <- unname(fixef(complete_case_model)["x"])
complete_case_se_x <- sqrt(vcov(complete_case_model)["x", "x"])

complete_ci <- unname(complete_beta_x) +
  c(-1, 1) * qnorm(0.975) * complete_se_x
complete_case_ci <- complete_case_beta_x +
  c(-1, 1) * qnorm(0.975) * complete_case_se_x

results_table <- data.frame(
  method = c("complete data", "complete case", "JOMO pooled"),
  beta_x = c(unname(complete_beta_x), complete_case_beta_x, Qbar),
  se = c(complete_se_x, complete_case_se_x, jomo_pooled_se),
  df = c(NA, NA, jomo_df),
  fmi = c(NA, NA, jomo_fmi),
  ci_low = c(complete_ci[1], complete_case_ci[1], jomo_ci[1]),
  ci_high = c(complete_ci[2], complete_case_ci[2], jomo_ci[2]),
  bias_from_0.8 = c(
    unname(complete_beta_x) - true_beta_x,
    complete_case_beta_x - true_beta_x,
    Qbar - true_beta_x
  ),
  difference_from_complete = c(
    0,
    complete_case_beta_x - unname(complete_beta_x),
    Qbar - unname(complete_beta_x)
  )
)

display_results <- results_table
display_results[, -1] <- round(display_results[, -1], 4)

cat("\n=== FINAL COMPARISON: generating beta_x = 0.8 ===\n")
print(display_results, row.names = FALSE)

results_file <- file.path(output_directory, "jomo_x_coefficient_results.csv")
write.csv(results_table, results_file, row.names = FALSE)
cat("\nSaved coefficient results:", normalizePath(results_file), "\n")

# What we want to see:
#   - pooled x is reasonably close to the complete-data estimate and 0.8;
#   - its confidence interval includes the complete-data estimate and ideally 0.8;
#   - the five imputations differ, so between-imputation uncertainty is retained.
#
# FMI is the fraction of inferential uncertainty attributable to missing
# information. It is not simply the percentage of x values that are missing.
# One toy dataset demonstrates the process; repeated simulations would be
# required to estimate bias, efficiency, and confidence-interval coverage.


# =============================================================================
# 8. OPTIONAL: ADD AN INCOMPLETE BINARY VARIABLE
# -----------------------------------------------------------------------------
# This section shows why JOMO is useful when a multilevel dataset mixes variable
# types. We retain incomplete continuous x and additionally remove about 20% of
# the binary smoke values. y, x, and smoke are placed in the same Y object so
# JOMO can preserve their joint relationships while imputing both incomplete
# variables.
#
# JOMO should now select jomo1ranmix: "mix" indicates a mixture of continuous
# and categorical outcomes. It cannot model a factor with an ordinary normal
# distribution directly. Instead, it assumes an unobserved continuous latent
# variable underlying smoke. Values on one side of a threshold correspond to
# "no" and values on the other side correspond to "yes." JOMO samples on that
# latent scale and then converts draws back into the observed categories.
#
# If output is changed to 1, smoke.1 coefficients are on the latent-normal
# scale, not a logistic-regression scale. Its level-1 variance is fixed at 1
# for identification because the scale of an unobserved latent variable cannot
# otherwise be uniquely determined. That fixed value is expected rather than a
# warning sign.
#
# The goal is not to guess every hidden person's smoking status correctly. A
# valid multiple-imputation procedure should reproduce the distribution and
# relationships of smoking while expressing uncertainty through differences
# among completed datasets.
# =============================================================================

mixed_data <- dat_mar
smoke_truth <- mixed_data$smoke

set.seed(800)
remove_smoke <- runif(n) < 0.20
mixed_data$smoke[remove_smoke] <- NA

Y_mixed <- data.frame(
  y = mixed_data$y,
  x = mixed_data$x,
  smoke = mixed_data$smoke
)

set.seed(801)

imputed_mixed_long <- jomo(
  Y = Y_mixed,
  X = X,
  clus = clus,
  nburn = 1000,
  nbetween = 1000,
  nimp = 5,
  meth = "common",
  output = 0
)

mixed_sets <- jomo2mitml.list(imputed_mixed_long)

mixed1 <- mixed_sets[[1]]
mixed2 <- mixed_sets[[2]]
mixed3 <- mixed_sets[[3]]
mixed4 <- mixed_sets[[4]]
mixed5 <- mixed_sets[[5]]

observed_smoke_rows <- which(!is.na(mixed_data$smoke))

# Both checks should be TRUE.
all_smoke_filled <-
  !anyNA(mixed1$smoke) && !anyNA(mixed2$smoke) &&
  !anyNA(mixed3$smoke) && !anyNA(mixed4$smoke) &&
  !anyNA(mixed5$smoke)

observed_smoke_unchanged <-
  all(as.character(mixed1$smoke[observed_smoke_rows]) ==
        as.character(smoke_truth[observed_smoke_rows])) &&
  all(as.character(mixed2$smoke[observed_smoke_rows]) ==
        as.character(smoke_truth[observed_smoke_rows])) &&
  all(as.character(mixed3$smoke[observed_smoke_rows]) ==
        as.character(smoke_truth[observed_smoke_rows])) &&
  all(as.character(mixed4$smoke[observed_smoke_rows]) ==
        as.character(smoke_truth[observed_smoke_rows])) &&
  all(as.character(mixed5$smoke[observed_smoke_rows]) ==
        as.character(smoke_truth[observed_smoke_rows]))

cat("\n=== Section 8: categorical-imputation checks ===\n")
cat("All missing smoke values filled:", all_smoke_filled, "\n")
cat("Observed smoke values unchanged:", observed_smoke_unchanged, "\n")

# Because this is simulated data, smoke_truth retains the values that were
# deliberately hidden. We compare the true complete distribution with each
# completed dataset. In a real NHANES analysis, the hidden truth would not be
# available; comparisons would instead use observed distributions, scientific
# plausibility, relationships with other variables, and sensitivity checks.
smoke_levels <- levels(smoke_truth)

smoke_distribution_comparison <- rbind(
  complete_truth = prop.table(table(
    factor(smoke_truth, levels = smoke_levels)
  )),
  imputation_1 = prop.table(table(
    factor(mixed1$smoke, levels = smoke_levels)
  )),
  imputation_2 = prop.table(table(
    factor(mixed2$smoke, levels = smoke_levels)
  )),
  imputation_3 = prop.table(table(
    factor(mixed3$smoke, levels = smoke_levels)
  )),
  imputation_4 = prop.table(table(
    factor(mixed4$smoke, levels = smoke_levels)
  )),
  imputation_5 = prop.table(table(
    factor(mixed5$smoke, levels = smoke_levels)
  ))
)

smoke_distribution_comparison <- rbind(
  smoke_distribution_comparison,
  mean_imputation = colMeans(smoke_distribution_comparison[2:6, ])
)

cat("\nSmoking distribution: truth versus completed datasets\n")
print(round(smoke_distribution_comparison, 3))

# The full-sample distribution is dominated by the 80% of smoke values that
# were never missing. Therefore, we also focus only on the deliberately hidden
# rows. Their imputed "yes" proportions should fluctuate around the true hidden
# proportion rather than reproduce it exactly in every completed dataset.
hidden_smoke_recovery <- c(
  complete_truth = mean(smoke_truth[remove_smoke] == "yes"),
  imputation_1 = mean(mixed1$smoke[remove_smoke] == "yes"),
  imputation_2 = mean(mixed2$smoke[remove_smoke] == "yes"),
  imputation_3 = mean(mixed3$smoke[remove_smoke] == "yes"),
  imputation_4 = mean(mixed4$smoke[remove_smoke] == "yes"),
  imputation_5 = mean(mixed5$smoke[remove_smoke] == "yes")
)

cat("\nProportion 'yes' among deliberately hidden rows\n")
print(round(hidden_smoke_recovery, 3))

smoke_results_file <- file.path(
  output_directory,
  "jomo_smoke_distribution_results.csv"
)
write.csv(
  smoke_distribution_comparison,
  smoke_results_file,
  row.names = TRUE
)
cat("Saved smoking results:", normalizePath(smoke_results_file), "\n")

# What we want to see:
#   - JOMO selects jomo1ranmix;
#   - all_smoke_filled and observed_smoke_unchanged are TRUE;
#   - the mean completed distribution is close to complete_truth;
#   - hidden-row proportions vary around the hidden truth rather than matching
#     it exactly in every imputation.
# Multiple imputation aims to recover distributions, relationships, and
# uncertainty—not every person's exact hidden category.


# =============================================================================
# 9. NHANES BOUNDARY
# -----------------------------------------------------------------------------
# This example teaches multilevel JOMO mechanics, but it is not yet a complete
# NHANES implementation. A JOMO cluster can resemble an NHANES PSU because it
# represents dependence among people sampled from the same group. However,
# simply setting clus equal to PSU does not automatically incorporate:
#   - NHANES sampling weights;
#   - sampling strata;
#   - the nesting and uniqueness of PSU identifiers across survey cycles;
#   - survey-weighted estimation after imputation; or
#   - structural missingness caused by eligibility and questionnaire routing.
#
# Those features require explicit decisions about the imputation model and the
# final survey analysis. This toy example establishes the JOMO workflow that
# will support those later decisions; it does not settle them.
# =============================================================================

cat("\nDone. JOMO fits one multilevel joint distribution, uses MCMC to draw\n")
cat("plausible missing values, and pools analyses with Rubin's rules.\n")