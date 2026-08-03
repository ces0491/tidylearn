# ---- Compute routing (tl_resolve_compute + tl_model compute arg) ----

# Shared fixtures matching test-compute-advisor.R conventions.
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
      keras      = list(installed = FALSE, gpu_likely_works = FALSE),
      torch      = list(installed = FALSE, gpu_likely_works = FALSE)
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
      keras      = list(installed = FALSE, gpu_likely_works = FALSE),
      torch      = list(installed = FALSE, gpu_likely_works = FALSE)
    ),
    messages = character(0)
  ),
  class = "tidylearn_gpu_check"
)

# ---- tl_resolve_compute ----

test_that("tl_resolve_compute rejects invalid compute values", {
  expect_error(
    tl_resolve_compute("xgboost", iris, Species ~ ., compute = "bogus"),
    "'compute' must be one of"
  )
  expect_error(
    tl_resolve_compute("xgboost", iris, Species ~ ., compute = NA),
    "'compute' must be one of"
  )
  expect_error(
    tl_resolve_compute("xgboost", iris, Species ~ ., compute = c("cpu", "gpu")),
    "'compute' must be one of"
  )
})

test_that("tl_resolve_compute('cloud') errors with a clear message", {
  expect_error(
    tl_resolve_compute("xgboost", iris, Species ~ ., compute = "cloud"),
    "Cloud compute is not yet wired up"
  )
})

test_that("tl_resolve_compute('cpu') returns 'cpu' without any detection", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) stop("should not be called"),
    .package = "tidylearn"
  )
  expect_equal(
    tl_resolve_compute("xgboost", iris, Species ~ ., compute = "cpu"),
    "cpu"
  )
})

test_that("tl_resolve_compute('auto') for CPU-only method skips detection", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) stop("should not be called"),
    .package = "tidylearn"
  )
  expect_equal(
    tl_resolve_compute("linear", iris, Species ~ ., compute = "auto"),
    "cpu"
  )
  expect_equal(
    tl_resolve_compute("forest", iris, Species ~ ., compute = "auto"),
    "cpu"
  )
})

test_that("tl_resolve_compute('gpu') for CPU-only method warns and CPUs", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_off,
    .package = "tidylearn"
  )
  expect_warning(
    result <- tl_resolve_compute(
      "linear", iris, Species ~ ., compute = "gpu"
    ),
    "no upstream GPU path"
  )
  expect_equal(result, "cpu")
})

test_that("tl_resolve_compute('gpu') with no GPU detected falls back", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_off,
    .package = "tidylearn"
  )
  expect_warning(
    result <- tl_resolve_compute(
      "xgboost", iris, Species ~ ., compute = "gpu"
    ),
    "no GPU-capable backend was"
  )
  expect_equal(result, "cpu")
})

test_that("tl_resolve_compute('gpu') with GPU detected returns 'gpu'", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_xgb,
    .package = "tidylearn"
  )
  expect_equal(
    tl_resolve_compute("xgboost", iris, Species ~ ., compute = "gpu"),
    "gpu"
  )
})

test_that("tl_resolve_compute('auto') consults advisor for GPU-capable", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_xgb,
    .package = "tidylearn"
  )
  # Small problem -> advisor recommends cpu -> resolve to cpu
  expect_message(
    result <- tl_resolve_compute(
      "xgboost", iris, Species ~ ., compute = "auto"
    ),
    "chose 'cpu'"
  )
  expect_equal(result, "cpu")
})

test_that("tl_resolve_compute('auto') picks gpu for long GPU-capable job", {
  # Pin the CPU estimate rather than allocating a large matrix: the real
  # estimate divides by detectCores(), so the expectation would flip on a
  # machine with enough cores
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_xgb,
    tl_estimate_local_cpu_internal = function(method, n_rows, n_cols, hyp) {
      list(
        est_seconds     = 1200,
        est_peak_ram_mb = 400,
        cores_used      = 4L,
        feasible        = TRUE,
        notes           = character(0)
      )
    },
    .package = "tidylearn"
  )

  expect_message(
    result <- tl_resolve_compute(
      "xgboost", iris, Species ~ ., compute = "auto",
      hyperparams = list(nrounds = 5000)
    ),
    "chose 'gpu'"
  )
  expect_equal(result, "gpu")
})

# ---- tl_model() integration ----

test_that("tl_model() defaults to compute = 'cpu' (no behaviour change)", {
  model <- tl_model(iris, Species ~ ., method = "tree")
  expect_equal(model$spec$compute, "cpu")
})

test_that("tl_model() forwards hyperparameters to the compute advisor", {
  # Without forwarding, "auto" sizes a default job (nrounds = 100) and can
  # be out by the ratio of requested to default
  seen <- NULL
  testthat::local_mocked_bindings(
    tl_resolve_compute = function(method, data, formula, compute = "cpu",
                                  hyperparams = list()) {
      seen <<- hyperparams
      "cpu"
    },
    .package = "tidylearn"
  )

  invisible(tl_model(
    iris, Species ~ ., method = "tree",
    compute = "auto", cp = 0.05
  ))

  expect_equal(seen$cp, 0.05)
})

test_that("tl_model(compute = 'cpu') sets spec$compute and proceeds", {
  model <- tl_model(
    iris, Species ~ ., method = "tree", compute = "cpu"
  )
  expect_equal(model$spec$compute, "cpu")
  expect_s3_class(model, "tidylearn_supervised")
})

test_that("tl_model(compute = 'gpu') for CPU-only method warns + CPUs", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_off,
    .package = "tidylearn"
  )
  expect_warning(
    model <- tl_model(
      iris, Species ~ ., method = "tree", compute = "gpu"
    ),
    "no upstream GPU path"
  )
  expect_equal(model$spec$compute, "cpu")
})

test_that("tl_model(compute = 'cloud') errors", {
  expect_error(
    tl_model(iris, Species ~ ., method = "tree", compute = "cloud"),
    "Cloud compute is not yet wired up"
  )
})

test_that("tl_model(compute = 'gpu') on unsupervised warns + CPUs", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) fake_gpu_off,
    .package = "tidylearn"
  )
  expect_warning(
    model <- tl_model(iris, method = "pca", compute = "gpu"),
    "no upstream GPU path"
  )
  expect_equal(model$spec$compute, "cpu")
})

test_that("tl_model(compute = 'auto') on unsupervised does not warn", {
  expect_no_warning(
    tl_model(iris, method = "pca", compute = "auto")
  )
})

test_that("tl_model(compute = 'cloud') on unsupervised errors", {
  expect_error(
    tl_model(iris, method = "pca", compute = "cloud"),
    "Cloud compute is not yet wired up"
  )
})

test_that("tl_model() records spec$compute on unsupervised models", {
  model <- tl_model(iris, method = "pca", compute = "cpu")
  expect_equal(model$spec$compute, "cpu")
  expect_s3_class(model, "tidylearn_unsupervised")
})

test_that("tl_model(compute = 'auto') falls through for CPU-only methods", {
  testthat::local_mocked_bindings(
    tl_check_gpu = function(...) stop("should not be called for CPU-only"),
    .package = "tidylearn"
  )
  model <- tl_model(
    iris, Species ~ ., method = "tree", compute = "auto"
  )
  expect_equal(model$spec$compute, "cpu")
})
