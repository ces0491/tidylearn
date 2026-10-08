# ---- Compute advisor (tl_compute_advisor and helpers) ----

# Shared fixtures: synthetic gpu_check objects so tests don't depend on
# the host machine having (or not having) a real GPU.
fake_gpu_off <- structure(
  list(
    any_gpu = FALSE,
    cuda = list(
      driver_present = FALSE,
      device_count   = 0L,
      device_names   = character(0),
      driver_version = NA_character_
    ),
    backends = list(
      xgboost    = list(installed = FALSE, gpu_likely_works = FALSE),
      tensorflow = list(installed = FALSE, gpu_likely_works = FALSE),
      keras      = list(installed = FALSE, gpu_likely_works = FALSE)
    ),
    messages = character(0)
  ),
  class = "tidylearn_gpu_check"
)

fake_gpu_xgb <- structure(
  list(
    any_gpu = TRUE,
    cuda = list(
      driver_present = TRUE,
      device_count   = 1L,
      device_names   = "Tesla T4",
      driver_version = "525.85.12"
    ),
    backends = list(
      xgboost    = list(installed = TRUE,  gpu_likely_works = TRUE),
      tensorflow = list(installed = FALSE, gpu_likely_works = FALSE),
      keras      = list(installed = FALSE, gpu_likely_works = FALSE)
    ),
    messages = character(0)
  ),
  class = "tidylearn_gpu_check"
)

test_that(".character dispatch returns expected structure", {
  result <- tl_compute_advisor(
    "linear", iris, Species ~ .,
    gpu_check = fake_gpu_off
  )
  expect_s3_class(result, "tidylearn_compute_advice")
  expect_named(
    result,
    c("problem", "local_cpu", "local_gpu", "cloud",
      "recommendation", "reasoning")
  )
  expect_equal(result$problem$method, "linear")
  expect_equal(result$problem$rows, nrow(iris))
})

test_that(".character dispatch validates method name", {
  expect_error(
    tl_compute_advisor("nonexistent_method", iris, Species ~ .,
                       gpu_check = fake_gpu_off),
    "not supported by the advisor"
  )
})

test_that(".character dispatch rejects empty method name", {
  expect_error(
    tl_compute_advisor("", iris, Species ~ ., gpu_check = fake_gpu_off),
    "single non-empty method name"
  )
})

test_that(".character dispatch validates data type", {
  expect_error(
    tl_compute_advisor("linear", "not_a_df", Species ~ .,
                       gpu_check = fake_gpu_off),
    "'data' must be a data frame"
  )
})

test_that(".character dispatch validates hyperparams type", {
  expect_error(
    tl_compute_advisor("linear", iris, Species ~ .,
                       hyperparams = "bad",
                       gpu_check = fake_gpu_off),
    "must be a list"
  )
})

test_that(".character dispatch validates gpu_check class", {
  expect_error(
    tl_compute_advisor("linear", iris, Species ~ .,
                       gpu_check = list()),
    "tidylearn_gpu_check object"
  )
})

test_that(".tidylearn_supervised dispatch introspects method + formula", {
  fake_model <- structure(
    list(
      spec = list(
        method  = "linear",
        formula = Sepal.Length ~ Sepal.Width
      ),
      fit  = NULL,
      data = iris
    ),
    class = c("tidylearn_linear", "tidylearn_supervised", "tidylearn_model")
  )
  result <- tl_compute_advisor(fake_model, gpu_check = fake_gpu_off)
  expect_s3_class(result, "tidylearn_compute_advice")
  expect_equal(result$problem$method, "linear")
  expect_equal(result$problem$cols, 1L)
})

test_that(".tidylearn_supervised dispatch accepts new data argument", {
  fake_model <- structure(
    list(
      spec = list(
        method  = "linear",
        formula = Sepal.Length ~ Sepal.Width
      ),
      fit  = NULL,
      data = iris
    ),
    class = c("tidylearn_linear", "tidylearn_supervised", "tidylearn_model")
  )
  new_data <- iris[1:50, ]
  result <- tl_compute_advisor(
    fake_model, data = new_data, gpu_check = fake_gpu_off
  )
  expect_equal(result$problem$rows, 50L)
})

test_that(".default dispatch errors helpfully", {
  expect_error(
    tl_compute_advisor(42),
    "expects either a method name"
  )
  expect_error(
    tl_compute_advisor(list()),
    "expects either a method name"
  )
})

test_that("small problem recommends CPU", {
  result <- tl_compute_advisor(
    "linear", iris, Species ~ .,
    gpu_check = fake_gpu_xgb
  )
  expect_equal(result$recommendation, "cpu")
  expect_match(result$reasoning, "Cloud cold-start|run it locally")
})

test_that("non-GPU method reports local_gpu as not applicable", {
  result <- tl_compute_advisor(
    "linear", iris, Species ~ .,
    gpu_check = fake_gpu_xgb
  )
  expect_false(result$local_gpu$available)
  expect_match(result$local_gpu$notes, "no upstream GPU path")
})

test_that("xgboost with GPU detected reports GPU available", {
  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    gpu_check = fake_gpu_xgb
  )
  expect_true(result$local_gpu$available)
})

test_that("xgboost without GPU reports GPU not available", {
  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    gpu_check = fake_gpu_off
  )
  expect_false(result$local_gpu$available)
  expect_match(result$local_gpu$notes, "no GPU-capable backend")
})

test_that("cloud tier is reported as not configured in this slice", {
  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    gpu_check = fake_gpu_xgb
  )
  expect_false(result$cloud$configured)
  expect_match(result$cloud$notes, "not yet configured")
})

# The real CPU estimate divides by parallel::detectCores(), so on a
# many-core machine a "long job" drops under the 60s threshold and the
# recommendation flips. Pin it instead of allocating a large matrix and
# hoping the host has few enough cores. Every estimator (GPU, cloud)
# derives from this one, so mocking it makes the whole advisor
# deterministic.
local_pinned_cpu <- function(seconds = 1200, peak_ram_mb = 400,
                             env = parent.frame()) {
  testthat::local_mocked_bindings(
    tl_estimate_local_cpu_internal = function(method, n_rows, n_cols, hyp) {
      list(
        est_seconds     = seconds,
        est_peak_ram_mb = peak_ram_mb,
        cores_used      = 4L,
        feasible        = peak_ram_mb < 16384,
        notes           = character(0)
      )
    },
    .env = env
  )
}

test_that("long xgboost problem with GPU recommends gpu", {
  local_pinned_cpu()

  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    hyperparams = list(nrounds = 5000),
    gpu_check = fake_gpu_xgb
  )

  expect_equal(result$recommendation, "gpu")
  expect_match(result$reasoning, "faster than CPU")
})

test_that("xgboost long-running but no GPU does not recommend gpu", {
  local_pinned_cpu()

  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    hyperparams = list(nrounds = 5000),
    gpu_check = fake_gpu_off
  )

  expect_false(result$recommendation == "gpu")
  expect_false(result$local_gpu$available)
})

test_that("a fast GPU is not disqualified for finishing quickly", {
  # A GPU estimate under 5s used to fall through to "no meaningfully
  # faster tier available" even at a 15x speedup
  cpu <- list(est_seconds = 70, est_peak_ram_mb = 100, feasible = TRUE)
  gpu <- list(est_seconds = 4.7, available = TRUE, feasible = TRUE)
  cloud <- list(
    est_seconds = 200, est_cost_usd = 0.05, tier_label = "T4",
    ram_needed_gb = 1, configured = FALSE
  )

  rec <- tl_recommend_internal(cpu, gpu, cloud)

  expect_equal(rec$recommendation, "gpu")
})

test_that("sub-60s jobs still stay local even with a GPU available", {
  cpu <- list(est_seconds = 30, est_peak_ram_mb = 100, feasible = TRUE)
  gpu <- list(est_seconds = 2, available = TRUE, feasible = TRUE)
  cloud <- list(
    est_seconds = 60, est_cost_usd = 0.05, tier_label = "T4",
    ram_needed_gb = 1, configured = FALSE
  )

  rec <- tl_recommend_internal(cpu, gpu, cloud)

  expect_equal(rec$recommendation, "cpu")
  expect_match(rec$reasoning, "Cloud cold-start")
})

test_that("a GPU that is barely faster does not win", {
  cpu <- list(est_seconds = 120, est_peak_ram_mb = 100, feasible = TRUE)
  gpu <- list(est_seconds = 60, available = TRUE, feasible = TRUE)
  cloud <- list(
    est_seconds = 300, est_cost_usd = 0.05, tier_label = "T4",
    ram_needed_gb = 1, configured = FALSE
  )

  rec <- tl_recommend_internal(cpu, gpu, cloud)

  expect_false(rec$recommendation == "gpu")
})

test_that("tl_effective_p_internal handles NULL formula", {
  expect_equal(tl_effective_p_internal(iris, NULL), 4L)
})

test_that("tl_effective_p_internal handles dot formula", {
  expect_equal(tl_effective_p_internal(iris, Species ~ .), 4L)
})

test_that("tl_effective_p_internal counts explicit predictors", {
  expect_equal(
    tl_effective_p_internal(iris, Species ~ Sepal.Length + Sepal.Width),
    2L
  )
})

test_that("the predictor count follows the formula's terms", {
  # all.vars() on the right-hand side read y ~ . - id as two predictors
  # ("." and id), and y ~ 1 as every column but one
  wide <- as.data.frame(matrix(0, nrow = 3, ncol = 200))
  wide$id <- 1:3
  wide$y <- 0

  expect_equal(tl_effective_p_internal(wide, y ~ . - id), 200L)
  expect_equal(tl_effective_p_internal(wide, y ~ .), 201L)
  expect_equal(tl_effective_p_internal(wide, y ~ V1 * V2), 3L)
  expect_equal(tl_effective_p_internal(wide, y ~ 1), 1L)
  expect_equal(tl_effective_p_internal(wide, "y ~ . - id"), 200L)

  fake_model <- structure(
    list(
      spec = list(method = "linear", formula = y ~ . - id),
      fit  = NULL,
      data = wide
    ),
    class = c("tidylearn_linear", "tidylearn_supervised", "tidylearn_model")
  )
  advice <- tl_compute_advisor(fake_model, gpu_check = fake_gpu_off)
  expect_equal(advice$problem$cols, 200L)
})

# A data frame reporting `rows` rows without allocating them. The advisor
# reads only nrow() and the column names.
fake_frame <- function(rows, predictors) {
  cols <- stats::setNames(
    rep(list(numeric(0)), predictors + 1L),
    c("y", paste0("x", seq_len(predictors)))
  )
  structure(cols, row.names = c(NA_integer_, -as.integer(rows)),
            class = "data.frame")
}

test_that("the advisor sizes inputs past the integer limit", {
  # nrow() times the predictor count is an integer product: 1e7 rows by
  # 250 predictors overflowed to NA, and the advisor -- with it every
  # compute = "auto" fit that large -- failed with "missing value where
  # TRUE/FALSE needed"
  big <- fake_frame(1e7, 250)

  expect_no_warning(
    advice <- tl_compute_advisor("xgboost", big, y ~ .,
                                 gpu_check = fake_gpu_off)
  )
  size_mb <- 1e7 * 250 * 8 / 1e6
  expect_equal(advice$problem$rows, 1e7)
  expect_equal(advice$problem$cols, 250L)
  expect_equal(advice$problem$est_size_mb, size_mb)
  expect_equal(advice$local_cpu$est_peak_ram_mb,
               size_mb * .tl_method_profiles$xgboost$ram_mult)
  expect_false(advice$local_cpu$feasible)
  expect_equal(advice$recommendation, "cloud")
})

test_that("the advice writes large sizes out in full", {
  # format() writes a round number in scientific notation when that is
  # shorter, so a 100,000 MB peak printed as "1e+05" in the table, the
  # reasoning and the notes
  advice <- tl_compute_advisor("xgboost", fake_frame(1e9, 1250), y ~ .,
                               gpu_check = fake_gpu_off)
  output <- capture.output(print(advice))

  expect_false(any(grepl("[0-9]e[+]", output)))
  expect_true(any(grepl("(~10,000,000.0 MB)", output, fixed = TRUE)))
  expect_true(any(grepl("peak RAM ~40,000,000 MB", output, fixed = TRUE)))
  expect_match(advice$reasoning[1], "~40,000,000 MB", fixed = TRUE)
  expect_match(advice$local_cpu$notes, "~40,000,000 MB", fixed = TRUE)
})

test_that("the advisor's defaults are the fit functions' own", {
  # The advisor assumed 10 epochs of 128 units for "deep" and a hidden
  # layer of 10 for "nn". tl_fit_deep() trains 30 epochs through layers of
  # 32 and 16 and has no units argument; tl_fit_nn() uses a layer of 5.
  deep <- formals(tl_fit_deep)
  layers <- eval(deep$hidden_layers)
  weights <- sum(c(10, layers) * c(layers, 1))
  expect_equal(
    tl_method_complexity_internal("deep", 1000, 10, list()),
    1000 * eval(deep$epochs) * weights
  )

  nn <- formals(tl_fit_nn)
  expect_equal(
    tl_method_complexity_internal("nn", 1000, 10, list()),
    1000 * 10 * eval(nn$size) * eval(nn$maxit)
  )

  expect_equal(
    tl_method_complexity_internal("xgboost", 1000, 10, list()),
    1000 * 10 * eval(formals(tl_fit_xgboost)$nrounds)
  )

  # A layout given as hyperparameters is what gets sized: 10 inputs into
  # one layer of 64, then one output
  expect_equal(
    tl_method_complexity_internal(
      "deep", 1000, 10, list(epochs = 5, hidden_layers = 64)
    ),
    1000 * 5 * (10 * 64 + 64 * 1)
  )
})

test_that("a hyperparameter that is not a positive number is refused", {
  # nrounds = NA made every estimate NA, and the advisor failed with
  # "attempt to select less than one element"; a negative value gave a
  # negative runtime
  expect_error(
    tl_compute_advisor("xgboost", iris, Species ~ .,
                       hyperparams = list(nrounds = NA_real_),
                       gpu_check = fake_gpu_off),
    "Hyperparameter 'nrounds' must be a single positive number"
  )
  expect_error(
    tl_compute_advisor("xgboost", iris, Species ~ .,
                       hyperparams = list(nrounds = -100),
                       gpu_check = fake_gpu_off),
    "Hyperparameter 'nrounds' must be a single positive number"
  )
  expect_error(
    tl_compute_advisor("deep", iris, Species ~ .,
                       hyperparams = list(hidden_layers = c(32, NA)),
                       gpu_check = fake_gpu_off),
    "Hyperparameter 'hidden_layers' must be positive numbers"
  )

  # Valid values still size the fit, and names the method does not use
  # are not checked
  advice <- tl_compute_advisor("xgboost", iris, Species ~ .,
                               hyperparams = list(nrounds = 50, eta = NA),
                               gpu_check = fake_gpu_off)
  expect_s3_class(advice, "tidylearn_compute_advice")
  advice <- tl_compute_advisor("linear", iris, Species ~ .,
                               hyperparams = list(nrounds = NA),
                               gpu_check = fake_gpu_off)
  expect_s3_class(advice, "tidylearn_compute_advice")
})

test_that("tl_method_complexity_internal respects hyperparams", {
  base <- tl_method_complexity_internal("xgboost", 1000, 10, list())
  more <- tl_method_complexity_internal(
    "xgboost", 1000, 10, list(nrounds = 1000)
  )
  expect_gt(more, base)
})

test_that("tl_method_complexity_internal scales with rows and cols", {
  small <- tl_method_complexity_internal("linear", 100,  5, list())
  big   <- tl_method_complexity_internal("linear", 1000, 5, list())
  expect_gt(big, small)
})

test_that("tl_method_complexity_internal has a sane fallback", {
  result <- tl_method_complexity_internal(
    "nonexistent", 100, 5, list()
  )
  expect_equal(result, 500)
})

test_that("print.tidylearn_compute_advice runs without error", {
  result <- tl_compute_advisor(
    "linear", iris, Species ~ .,
    gpu_check = fake_gpu_off
  )
  expect_output(print(result), "tidylearn compute advice")
  expect_output(print(result), "Recommendation:")
})

test_that("print.tidylearn_compute_advice renders cloud cost + tier", {
  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    gpu_check = fake_gpu_xgb
  )
  output <- capture.output(print(result))
  expect_true(any(grepl("Cloud:", output)))
  expect_true(any(grepl("not configured", output)))
  # Tier label should appear (xgboost is GPU-eligible -> picks a GPU tier)
  expect_true(any(grepl("T4|A10G|A100", output)))
})

# ---- Cloud reframe: applies to all methods, RAM-bound triggers cloud ----

test_that("cloud estimate now applies to CPU-only methods too", {
  result <- tl_compute_advisor(
    "linear", iris, Species ~ .,
    gpu_check = fake_gpu_off
  )
  expect_true(is_finite_num(result$cloud$est_seconds))
  expect_true(is_finite_num(result$cloud$est_cost_usd))
  expect_false(result$cloud$has_gpu)
  expect_match(result$cloud$tier_label, "cpu-")
})

test_that("cloud estimate picks a GPU tier for GPU-eligible methods", {
  result <- tl_compute_advisor(
    "xgboost", iris, Species ~ .,
    gpu_check = fake_gpu_xgb
  )
  expect_true(result$cloud$has_gpu)
  expect_true(grepl("T4|A10G|A100", result$cloud$tier_label))
})

test_that("RAM-infeasible workload triggers cloud regardless of GPU", {
  # Build synthetic per-tier estimates that simulate a RAM-bound
  # local-CPU job and a viable cloud estimate. This is the unit-level
  # test of the new recommendation contract -- the integration is
  # exercised separately via tl_compute_advisor on real (small) data.
  ram_bound_cpu <- list(
    est_seconds     = 600,
    est_peak_ram_mb = 32000,
    cores_used      = 4L,
    feasible        = FALSE,
    notes           = "exceeds heuristic"
  )
  no_local_gpu <- list(
    est_seconds = NA_real_,
    available   = FALSE,
    feasible    = FALSE,
    notes       = "no GPU path"
  )
  cloud_est <- list(
    est_seconds    = 100,
    est_cost_usd   = 0.30,
    upload_seconds = 5,
    tier_name      = "cpu-large",
    tier_label     = "cpu-large (8 CPU / 64 GB)",
    ram_needed_gb  = 48,
    has_gpu        = FALSE,
    configured     = FALSE,
    notes          = "not configured"
  )

  # CPU-only method (linear) -> cloud should be recommended even
  # without any local GPU and without cloud being "configured".
  result <- tl_recommend_internal(ram_bound_cpu, no_local_gpu, cloud_est)
  expect_equal(result$recommendation, "cloud")
  expect_true(any(grepl("Local CPU infeasible", result$reasoning)))
  expect_true(any(grepl("cpu-large", result$reasoning)))
})

test_that("RAM-infeasible workload returns infeasible when no cloud estimate", {
  ram_bound_cpu <- list(
    est_seconds     = 600,
    est_peak_ram_mb = 32000,
    cores_used      = 4L,
    feasible        = FALSE,
    notes           = "exceeds heuristic"
  )
  no_local_gpu <- list(
    est_seconds = NA_real_,
    available   = FALSE,
    feasible    = FALSE,
    notes       = ""
  )
  no_cloud <- list(
    est_seconds   = NA_real_,
    est_cost_usd  = NA_real_,
    tier_name     = NA_character_,
    tier_label    = NA_character_,
    ram_needed_gb = 48,
    has_gpu       = FALSE,
    configured    = FALSE,
    notes         = ""
  )
  result <- tl_recommend_internal(ram_bound_cpu, no_local_gpu, no_cloud)
  expect_equal(result$recommendation, "infeasible")
})

test_that("tl_pick_modal_tier picks the cheapest viable GPU tier", {
  # Small RAM need + GPU-eligible method -> T4 (cheapest GPU tier)
  pick <- tl_pick_modal_tier_internal(
    method_has_gpu_path = TRUE,
    cpu_seconds         = 100,
    method_speedup      = 5,
    ram_needed_gb       = 4
  )
  expect_equal(pick$name, "t4")
  expect_true(pick$has_gpu)
  expect_true(pick$fits_ram_need)
})

test_that("tl_pick_modal_tier picks larger GPU tier when RAM needed", {
  # 60 GB RAM need + GPU -> needs A100 (T4 has only 16 GB, A10G 24 GB)
  pick <- tl_pick_modal_tier_internal(
    method_has_gpu_path = TRUE,
    cpu_seconds         = 100,
    method_speedup      = 5,
    ram_needed_gb       = 60
  )
  expect_equal(pick$name, "a100-40gb")
  expect_true(pick$fits_ram_need)
})

test_that("tl_pick_modal_tier picks CPU tier for CPU-only methods", {
  pick <- tl_pick_modal_tier_internal(
    method_has_gpu_path = FALSE,
    cpu_seconds         = 100,
    method_speedup      = 1,
    ram_needed_gb       = 4
  )
  expect_false(pick$has_gpu)
  expect_match(pick$name, "cpu-")
})

test_that("tl_pick_modal_tier falls back when no tier has enough RAM", {
  # 500 GB exceeds all tiers — picks biggest with fits_ram_need = FALSE
  pick <- tl_pick_modal_tier_internal(
    method_has_gpu_path = TRUE,
    cpu_seconds         = 100,
    method_speedup      = 5,
    ram_needed_gb       = 500
  )
  expect_equal(pick$name, "a100-80gb")
  expect_false(pick$fits_ram_need)
})

test_that("recommendation no longer gated on cloud$configured", {
  # Big RAM-bound CPU-only problem: advisor should recommend cloud
  # even though configured = FALSE. The new contract: advisor advises
  # optimally; caller decides whether it can act on the recommendation.
  result <- tl_compute_advisor(
    "linear",
    data = data.frame(y = numeric(5), x1 = numeric(5)),
    formula = y ~ x1,
    gpu_check = fake_gpu_off
  )
  # iris-sized problem is small enough that cpu is the right answer.
  # Just confirm we no longer gate on configured.
  expect_false(result$cloud$configured)
  # And that cloud estimate is populated (not NA) for this CPU-only fit.
  expect_true(is_finite_num(result$cloud$est_seconds))
})
