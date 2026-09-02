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

# ---- Tuning search: metric direction and best-parameter extraction ----

tuning_fixture <- function(seed = 11, n = 60) {
  set.seed(seed)
  data.frame(
    x1 = c(stats::rnorm(n, -1), stats::rnorm(n, 1)),
    x2 = stats::rnorm(2 * n),
    y = factor(rep(c("a", "b"), each = n))
  )
}

test_that("tl_tune_grid accepts an explicit metric without maximize", {
  data <- tuning_fixture()

  # `maximize` used to be set only inside the `is.null(metric)` branch, so
  # naming a metric left it NULL and `if (maximize)` errored
  for (metric in c("accuracy", "f1", "auc")) {
    model <- tl_tune_grid(
      data, y ~ x1 + x2, method = "tree",
      param_grid = list(cp = c(0.01, 0.1)),
      metric = metric, folds = 3, verbose = FALSE
    )
    tuning <- attr(model, "tuning_results")

    expect_s3_class(model, "tidylearn_model")
    expect_true(tuning$maximize)
    expect_false(is.na(tuning$best_metric))
  }
})

test_that("tuning direction follows the metric, not the task", {
  expect_false(tl_metric_maximize("rmse"))
  expect_false(tl_metric_maximize("mse"))
  expect_false(tl_metric_maximize("mae"))
  expect_false(tl_metric_maximize("mape"))
  expect_true(tl_metric_maximize("accuracy"))
  expect_true(tl_metric_maximize("f1"))
  expect_true(tl_metric_maximize("rsq"))

  # A regression task scored on rsq must maximise, not minimise
  model <- tl_tune_grid(
    mtcars, mpg ~ wt + hp, method = "tree",
    param_grid = list(cp = c(0.01, 0.1)),
    metric = "rsq", folds = 3, verbose = FALSE
  )
  expect_true(attr(model, "tuning_results")$maximize)

  model <- tl_tune_grid(
    mtcars, mpg ~ wt + hp, method = "tree",
    param_grid = list(cp = c(0.01, 0.1)),
    metric = "rmse", folds = 3, verbose = FALSE
  )
  expect_false(attr(model, "tuning_results")$maximize)
})

test_that("an explicit maximize argument is still honoured", {
  data <- tuning_fixture()
  model <- tl_tune_grid(
    data, y ~ x1 + x2, method = "tree",
    param_grid = list(cp = c(0.01, 0.1)),
    metric = "accuracy", maximize = FALSE, folds = 3, verbose = FALSE
  )
  expect_false(attr(model, "tuning_results")$maximize)
})

test_that("tuning a single parameter keeps its name", {
  data <- tuning_fixture()

  # Indexing one column without drop = FALSE collapses the row to a bare
  # value, so the winning setting reached tl_model() positionally
  model <- tl_tune_grid(
    data, y ~ x1 + x2, method = "tree",
    param_grid = list(cp = c(0.5, 0.001)),
    metric = "accuracy", folds = 3, verbose = FALSE
  )
  best <- attr(model, "tuning_results")$best_params

  expect_named(best, "cp")
  # and the chosen value must actually reach the fitted model
  expect_equal(model$fit$control$cp, best$cp)
})

test_that("tuning several parameters keeps all names", {
  data <- tuning_fixture()
  model <- tl_tune_grid(
    data, y ~ x1 + x2, method = "tree",
    param_grid = list(cp = c(0.01, 0.1), minsplit = c(5, 20)),
    metric = "accuracy", folds = 3, verbose = FALSE
  )
  expect_setequal(
    names(attr(model, "tuning_results")$best_params),
    c("cp", "minsplit")
  )
})

test_that("tl_tune_random accepts an explicit metric and keeps names", {
  data <- tuning_fixture()
  model <- tl_tune_random(
    data, y ~ x1 + x2, method = "tree",
    param_space = list(cp = c(0.001, 0.3)),
    n_iter = 3, metric = "f1", folds = 3, verbose = FALSE, seed = 1
  )
  tuning <- attr(model, "tuning_results")

  expect_true(tuning$maximize)
  expect_named(tuning$best_params, "cp")
})

test_that("tl_plot_tuning_results handles every documented plot type", {
  data <- tuning_fixture()
  model <- tl_tune_grid(
    data, y ~ x1 + x2, method = "tree",
    param_grid = list(cp = c(0.01, 0.1), minsplit = c(5, 20)),
    metric = "accuracy", folds = 3, verbose = FALSE
  )

  for (plot_type in c("scatter", "grid", "parallel", "importance")) {
    expect_s3_class(
      suppressWarnings(tl_plot_tuning_results(model, plot_type = plot_type)),
      "ggplot"
    )
  }
})

test_that("tl_plot_tuning_results scores categorical parameters", {
  data <- tuning_fixture()
  model <- tl_tune_grid(
    data, y ~ x1 + x2, method = "svm",
    param_grid = list(kernel = c("linear", "radial"), cost = c(1, 10)),
    metric = "accuracy", folds = 3, verbose = FALSE
  )

  # The ANOVA branch used the .data pronoun, which aov() cannot evaluate
  plot <- suppressWarnings(
    tl_plot_tuning_results(model, plot_type = "importance")
  )
  expect_s3_class(plot, "ggplot")
  expect_setequal(plot$data$parameter, c("kernel", "cost"))
})

test_that("grid plot falls back to scatter when there are too many levels", {
  data <- tuning_fixture()
  model <- tl_tune_grid(
    data, y ~ x1 + x2, method = "tree",
    param_grid = list(cp = seq(0.001, 0.3, length.out = 25),
                      minsplit = c(5, 20)),
    metric = "accuracy", folds = 2, verbose = FALSE
  )

  # The fallback discarded its recursive result, leaving `p` undefined
  expect_warning(
    plot <- tl_plot_tuning_results(model, plot_type = "grid"),
    "too many unique values"
  )
  expect_s3_class(plot, "ggplot")
})

test_that("tl_compare_cv returns per-fold and summary tables", {
  data <- tuning_fixture()
  models <- list(
    tree = tl_model(data, y ~ x1 + x2, method = "tree"),
    logistic = tl_model(data, y ~ x1 + x2, method = "logistic")
  )

  set.seed(21)
  result <- tl_compare_cv(data, models, folds = 3)

  expect_named(result, c("fold_metrics", "summary"))
  expect_gt(nrow(result$fold_metrics), 0)
  expect_setequal(result$summary$model, c("tree", "logistic"))
  # Requires tl_evaluate() to honour the requested metric set
  expect_setequal(
    unique(result$summary$metric),
    c("accuracy", "precision", "recall", "f1", "auc")
  )
  expect_false(any(is.na(result$summary$mean_value)))
})

# ---- parameter spaces that cannot be sampled -------------------------

test_that("tl_tune_random refuses a range that runs backwards", {
  # runif(1, 0.1, 0.001) is NaN, and R only warns. Every iteration
  # therefore drew NaN, models were fitted with cp = NaN, and
  # best_params came back as NaN -- with nothing failing anywhere.
  set.seed(1)
  n <- 60
  d <- data.frame(x1 = stats::rnorm(n), x2 = stats::rnorm(n))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(n, sd = 0.3)

  expect_error(
    tl_tune_random(d, y ~ x1 + x2, "tree", list(cp = c(0.1, 0.001)),
                   n_iter = 3, folds = 3),
    "but a range is"
  )
  expect_error(
    tl_tune_random(d, y ~ x1 + x2, "tree", list(cp = c(0.01, 0.01)),
                   n_iter = 3, folds = 3),
    "but a range is"
  )
  expect_error(
    tl_tune_random(d, y ~ x1 + x2, "tree", list(cp = c(0.1, 0.001, "log")),
                   n_iter = 3, folds = 3),
    "but a range is"
  )
  # A log-uniform draw needs log(min), so a non-positive bound is no good
  expect_error(
    tl_tune_random(d, y ~ x1 + x2, "tree", list(cp = c(0, 0.1, "log")),
                   n_iter = 3, folds = 3),
    "both bounds must be positive"
  )

  # The forms that were always valid still are
  expect_s3_class(
    suppressWarnings(suppressMessages(
      tl_tune_random(d, y ~ x1 + x2, "tree", list(cp = c(0.001, 0.1)),
                     n_iter = 2, folds = 3, seed = 1)
    )),
    "tidylearn_tree"
  )
})

test_that("a discrete set need not be whole numbers", {
  # Only whole numbers reached the discrete branch, so the natural way to
  # write candidate cp values -- which are never integers -- was rejected
  # as an "Unsupported parameter space definition", while tl_tune_grid()
  # accepted the same vector.
  set.seed(1)
  n <- 60
  d <- data.frame(x1 = stats::rnorm(n), x2 = stats::rnorm(n))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(n, sd = 0.3)

  candidates <- c(0.001, 0.01, 0.1)
  tuned <- suppressWarnings(suppressMessages(
    tl_tune_random(d, y ~ x1 + x2, "tree", list(cp = candidates),
                   n_iter = 8, folds = 3, seed = 7)
  ))
  drawn <- attr(tuned, "tuning_results")$results$cp
  expect_length(drawn, 8L)
  expect_true(all(drawn %in% candidates))
})

test_that("a metric the task cannot produce is named, not a length error", {
  set.seed(1)
  n <- 60
  d <- data.frame(x1 = stats::rnorm(n), x2 = stats::rnorm(n))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(n, sd = 0.3)

  # Both of these failed with "replacement has length zero"
  for (bad in c("accuracy", "not_a_metric")) {
    msg <- tryCatch(
      suppressWarnings(suppressMessages(
        tl_tune_grid(d, y ~ x1 + x2, "tree", list(cp = c(0.001, 0.01)),
                     folds = 3, metric = bad)
      )),
      error = function(e) conditionMessage(e)
    )
    expect_match(msg, "was not produced for this task", info = bad)
    # The message has to say what it could have used instead
    expect_match(msg, "rmse", info = bad)
    expect_false(grepl("replacement has length zero", msg), info = bad)
  }
})

test_that("tl_tune_xgboost runs, and takes nrounds without colliding", {
  skip_if_not_installed("xgboost")

  # This function had no test at all, and two defects between it and any
  # result. It hardcoded nrounds = 1000 while forwarding `...` to the same
  # xgb.cv() call, so passing the one argument an xgboost tuner obviously
  # takes gave "formal argument \"nrounds\" matched by multiple actual
  # arguments" -- and on the default path, where nothing collided, it
  # still died on "attempt to select less than one element in get1index".
  grid <- list(max_depth = c(2, 3), eta = 0.3)

  tuned <- suppressWarnings(suppressMessages(
    tl_tune_xgboost(iris, Species ~ .,
      is_classification = TRUE, param_grid = grid,
      cv_folds = 3, nrounds = 20, verbose = FALSE
    )
  ))
  expect_s3_class(tuned, "tidylearn_model")

  results <- attr(tuned, "tuning_results")
  expect_length(results$results, nrow(expand.grid(grid)))

  # The iteration has to be a real index into the evaluation log, not the
  # NULL that xgboost >= 3.0 returns from the pre-3.0 location
  expect_true(is.finite(results$best_iteration))
  expect_gte(results$best_iteration, 1)
  expect_true(is.finite(results$best_score))
  expect_true(results$best_params$max_depth %in% grid$max_depth)

  # And the default nrounds path, which never collided and failed anyway
  expect_s3_class(
    suppressWarnings(suppressMessages(
      tl_tune_xgboost(iris, Species ~ .,
        is_classification = TRUE,
        param_grid = list(max_depth = 2, eta = 0.3),
        cv_folds = 3, verbose = FALSE
      )
    )),
    "tidylearn_model"
  )
})

test_that("tl_tune_xgboost scores every task's own metric", {
  skip_if_not_installed("xgboost")

  # The score is read from a column named after the eval_metric, and the
  # metric differs by task -- mlogloss, logloss, rmse. The test above
  # covers mlogloss, so only the other two are run here: xgboost is the
  # heaviest thing in the suite and a third run would buy nothing.
  binary <- iris[iris$Species != "setosa", ]
  binary$Species <- droplevels(binary$Species)

  cases <- list(
    binary = list(data = binary, formula = Species ~ ., classify = TRUE),
    regression = list(data = mtcars, formula = mpg ~ ., classify = FALSE)
  )

  for (name in names(cases)) {
    case <- cases[[name]]
    tuned <- suppressWarnings(suppressMessages(
      tl_tune_xgboost(case$data, case$formula,
        is_classification = case$classify,
        param_grid = list(max_depth = 2, eta = 0.3),
        cv_folds = 3, nrounds = 20, verbose = FALSE
      )
    ))
    results <- attr(tuned, "tuning_results")

    # A NULL score is the failure mode: it collapses which.min() to
    # integer(0) rather than producing a wrong number
    expect_true(is.finite(results$best_score), info = name)
    expect_true(is.finite(results$best_iteration), info = name)
    expect_gte(results$best_iteration, 1)
    expect_lte(results$best_iteration, 20)
  }
})

test_that("tl_xgb_best_iteration reads either xgboost layout", {
  log <- data.frame(
    iter = 1:20,
    test_mlogloss_mean = seq(1, 0.2, length.out = 20)
  )

  # Where xgboost >= 3.0 puts it
  expect_equal(
    tl_xgb_best_iteration(list(
      early_stop = list(best_iteration = 14L), evaluation_log = log
    )),
    14L
  )

  # Where xgboost < 3.0 put it
  expect_equal(
    tl_xgb_best_iteration(list(best_iteration = 7L, evaluation_log = log)),
    7L
  )

  # Neither, because early stopping was off and the run went the distance
  expect_equal(tl_xgb_best_iteration(list(evaluation_log = log)), 20L)
})

test_that("tl_tune_xgboost takes a grid of a single parameter", {
  skip_if_not_installed("xgboost")

  # expand.grid() of one parameter is a single-column data frame, and
  # `[i, ]` on one of those drops to a bare vector with the column name
  # gone. as.list() then produced an unnamed list and xgboost refused the
  # whole fit: "parameter names cannot be empty strings". Every earlier
  # test named two parameters, so nothing caught it. tl_tune_grid() and
  # tl_tune_random() had the same slip fixed for 0.4.0.
  tuned <- suppressWarnings(suppressMessages(
    tl_tune_xgboost(iris, Species ~ .,
      is_classification = TRUE,
      param_grid = list(max_depth = c(2, 4)),
      cv_folds = 3, nrounds = 20, verbose = FALSE
    )
  ))
  expect_s3_class(tuned, "tidylearn_model")

  # The name has to survive, or the value reaches xgboost positionally
  results <- attr(tuned, "tuning_results")
  expect_true("max_depth" %in% names(results$best_params))
  expect_true(results$best_params$max_depth %in% c(2, 4))
})
