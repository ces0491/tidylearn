# ---- Tuning functions ----

test_that("tl_default_param_grid returns named list for tree", {
  grid <- tl_default_param_grid("tree", size = "small")

  expect_type(grid, "list")
  expect_true("cp" %in% names(grid))
  expect_true("minsplit" %in% names(grid))
})

test_that("tl_default_param_grid returns named list for forest", {
  grid <- tl_default_param_grid("forest", size = "medium")

  expect_type(grid, "list")
  expect_true("mtry" %in% names(grid))
  expect_true("ntree" %in% names(grid))
})

test_that("tl_default_param_grid returns named list for svm", {
  grid <- tl_default_param_grid("svm", size = "small")

  expect_type(grid, "list")
  expect_true("kernel" %in% names(grid))
  expect_true("cost" %in% names(grid))
})

test_that("tl_default_param_grid handles all supported methods", {
  methods <- c("tree", "forest", "boost", "svm", "nn",
               "ridge", "lasso", "elastic_net", "deep")

  for (method in methods) {
    grid <- tl_default_param_grid(method, size = "small")
    expect_type(grid, "list")
    expect_true(length(grid) > 0)
  }
})

test_that("tl_default_param_grid respects size parameter", {
  small <- tl_default_param_grid("forest", size = "small")
  medium <- tl_default_param_grid("forest", size = "medium")
  large <- tl_default_param_grid("forest", size = "large")

  # Larger grids should have more parameter values
  small_combos <- prod(sapply(small, length))
  medium_combos <- prod(sapply(medium, length))
  large_combos <- prod(sapply(large, length))

  expect_true(small_combos <= medium_combos)
  expect_true(medium_combos <= large_combos)
})

test_that("tl_default_param_grid warns for unknown method", {
  expect_warning(
    grid <- tl_default_param_grid("nonexistent"),
    "Unknown method"
  )
  expect_equal(length(grid), 0)
})

test_that("tl_default_param_grid elastic_net includes alpha", {
  grid <- tl_default_param_grid("elastic_net", size = "small")

  expect_true("alpha" %in% names(grid))
  expect_true(all(grid$alpha > 0 & grid$alpha < 1))
})

test_that("tl_default_param_grid ridge has no alpha in grid", {
  grid <- tl_default_param_grid("ridge", size = "medium")

  # Ridge should only have lambda, not alpha
  expect_true("lambda" %in% names(grid))
  expect_false("alpha" %in% names(grid))
})

test_that("tl_tune_grid works with tree method", {
  skip_if_not_installed("rpart")
  skip_if_not_installed("rsample")

  set.seed(42)
  param_grid <- list(cp = c(0.01, 0.1))

  result <- suppressMessages(
    tl_tune_grid(
      iris, Species ~ ., method = "tree",
      param_grid = param_grid, folds = 2,
      verbose = FALSE
    )
  )

  expect_s3_class(result, "tidylearn_model")
  expect_true(!is.null(attr(result, "tuning_results")))

  tuning <- attr(result, "tuning_results")
  expect_true("best_params" %in% names(tuning))
  expect_true("results" %in% names(tuning))
  expect_equal(nrow(tuning$results), 2)  # 2 param combos
})

test_that("tl_tune_grid validates param_grid input", {
  expect_error(
    tl_tune_grid(mtcars, mpg ~ wt, method = "linear",
                 param_grid = "not a list"),
    "param_grid must be a named list"
  )
})

test_that("tl_tune_random works with tree method", {
  skip_if_not_installed("rpart")
  skip_if_not_installed("rsample")

  set.seed(42)
  param_space <- list(cp = c(0.001, 0.1))

  result <- suppressMessages(
    tl_tune_random(
      iris, Species ~ ., method = "tree",
      param_space = param_space, n_iter = 2,
      folds = 2, verbose = FALSE, seed = 42
    )
  )

  expect_s3_class(result, "tidylearn_model")
  expect_true(!is.null(attr(result, "tuning_results")))

  tuning <- attr(result, "tuning_results")
  expect_equal(nrow(tuning$results), 2)  # 2 iterations
})

test_that("tl_tune_random validates param_space input", {
  expect_error(
    tl_tune_random(mtcars, mpg ~ wt, method = "linear",
                   param_space = "not a list"),
    "param_space must be a named list"
  )
})

test_that("tl_plot_tuning_results returns ggplot", {
  skip_if_not_installed("rpart")
  skip_if_not_installed("rsample")

  set.seed(42)
  param_grid <- list(
    cp = c(0.001, 0.01, 0.1),
    minsplit = c(5, 20)
  )

  model <- suppressMessages(
    tl_tune_grid(
      iris, Species ~ ., method = "tree",
      param_grid = param_grid, folds = 2,
      verbose = FALSE
    )
  )

  # Scatter plot
  p <- tl_plot_tuning_results(model, plot_type = "scatter")
  expect_s3_class(p, "ggplot")

  # Grid plot
  p2 <- tl_plot_tuning_results(model, plot_type = "grid")
  expect_s3_class(p2, "ggplot")
})

test_that("tl_plot_tuning_results errors without tuning results", {
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")

  expect_error(
    tl_plot_tuning_results(model),
    "tuning results"
  )
})

test_that("tl_tune_grid handles model fitting failures gracefully", {
  skip_if_not_installed("rsample")

  # Use a dataset where some parameter combos might fail
  param_grid <- list(cp = c(0.01, 0.5))

  # Should complete without error even if some folds perform poorly
  expect_no_error(
    suppressMessages(suppressWarnings(
      tl_tune_grid(
        iris, Species ~ ., method = "tree",
        param_grid = param_grid, folds = 2,
        verbose = FALSE
      )
    ))
  )
})
