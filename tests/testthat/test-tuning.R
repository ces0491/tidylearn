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
               "ridge", "lasso", "elastic_net", "deep", "xgboost")

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

test_that("a grid or space must name every element, once", {
  # An unnamed candidate vector reached tl_model() positionally, where
  # rpart.control() discarded it: list(c(0.01, 0.1)) fitted the same tree
  # twice and returned best_params with an element named "<dbl>"
  unnamed <- list(c(0.01, 0.1))
  partly <- list(cp = c(0.01, 0.1), c(5, 10))
  for (grid in list(unnamed, partly)) {
    expect_error(
      tl_tune_grid(mtcars, mpg ~ wt, method = "tree", param_grid = grid,
                   folds = 2, verbose = FALSE),
      "param_grid must be a named list: element [12] has no name"
    )
    expect_error(
      tl_tune_random(mtcars, mpg ~ wt, method = "tree", param_space = grid,
                     n_iter = 2, folds = 2, verbose = FALSE, seed = 1),
      "param_space must be a named list: element [12] has no name"
    )
  }
  expect_error(
    tl_tune_grid(mtcars, mpg ~ wt, method = "tree",
                 param_grid = list(cp = 0.01, cp = 0.1), folds = 2,
                 verbose = FALSE),
    "param_grid names must be unique.*repeated: cp"
  )

  # A data frame of candidates is named by its columns, and still runs
  set.seed(1)
  tuned <- tl_tune_grid(mtcars, mpg ~ wt, method = "tree",
                        param_grid = data.frame(cp = c(0.01, 0.1)),
                        folds = 2, verbose = FALSE)
  expect_named(attr(tuned, "tuning_results")$best_params, "cp")
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

test_that("the tuners take a metric's direction from the pipelines' rule", {
  # The tuners kept their own list of error metrics and read every other
  # name, unknown ones included, as higher-is-better
  known <- c(tl_known_metrics(TRUE), tl_known_metrics(FALSE))
  for (metric in known) {
    expect_identical(tl_metric_maximize(metric),
                     tl_metric_higher_better(metric), info = metric)
  }
  testthat::local_mocked_bindings(
    tl_metric_higher_better = function(metrics) rep(FALSE, length(metrics))
  )
  expect_false(tl_metric_maximize("accuracy"))
  testthat::local_mocked_bindings(
    tl_metric_higher_better = function(metrics) rep(NA, length(metrics))
  )
  expect_error(tl_metric_maximize("mystery"),
               "Metric \"mystery\" has no direction to optimise")
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

test_that("tuning plots draw list-valued parameters", {
  # crossing() keeps a list candidate, such as rpart's parms, as a list
  # column, and every plot type failed on it: "unimplemented type 'list'"
  # for parallel and importance, a scale error for scatter and grid
  set.seed(1)
  tuned <- tl_tune_grid(
    iris, Species ~ ., method = "tree",
    param_grid = list(cp = c(0.01, 0.1),
                      parms = list(list(split = "gini"),
                                   list(split = "information"))),
    folds = 2, verbose = FALSE
  )
  for (plot_type in c("scatter", "grid", "parallel", "importance")) {
    plot <- tl_plot_tuning_results(tuned, plot_type = plot_type)
    expect_no_error(ggplot2::ggplot_build(plot))
  }

  # Each cell is labelled the way the verbose messages describe it
  plot <- tl_plot_tuning_results(tuned, plot_type = "scatter")
  expect_setequal(
    as.character(plot$data$parms),
    c("list(split = \"gini\")", "list(split = \"information\")")
  )

  # Every set is its own line. Lines were grouped by rank, so sets whose
  # scores tied -- all four here -- were joined into one
  plot <- tl_plot_tuning_results(tuned, plot_type = "parallel")
  is_line <- vapply(plot$layers, function(layer) {
    inherits(layer$geom, "GeomLine")
  }, logical(1))
  lines <- ggplot2::layer_data(plot, which(is_line))
  n_sets <- nrow(attr(tuned, "tuning_results")$results)
  expect_length(unique(lines$group), n_sets)

  # A vector-valued candidate, as in the deep grid's hidden_layers, failed
  # the default scatter. The importance of a categorical parameter is its
  # eta squared from a one-way ANOVA of the score.
  combinations <- list(
    list(hidden_layers = 10, activation = "relu"),
    list(hidden_layers = c(10, 5), activation = "relu"),
    list(hidden_layers = 10, activation = "tanh"),
    list(hidden_layers = c(10, 5), activation = "tanh")
  )
  scores <- c(0.3, 0.5, 0.4, 0.6)
  results <- tl_tune_results_frame(
    lapply(scores, function(s) list(mean_metric = s, n_folds_ok = 2L)),
    combinations, c("hidden_layers", "activation")
  )
  expect_type(results$hidden_layers, "list")
  deep <- structure(list(), tuning_results = list(
    results = results, metric = "accuracy", maximize = TRUE
  ))
  for (plot_type in c("scatter", "grid", "parallel")) {
    expect_no_error(ggplot2::ggplot_build(
      tl_plot_tuning_results(deep, plot_type = plot_type)
    ))
  }
  importance <- tl_plot_tuning_results(deep, plot_type = "importance")$data
  # Group means 0.35 and 0.55 around 0.45: 0.04 of a total 0.05
  expect_equal(importance$importance[importance$parameter == "hidden_layers"],
               0.8)
  expect_equal(importance$importance[importance$parameter == "activation"],
               0.2)
})

test_that("a categorical parameter with one observed value has importance 0", {
  # aov() needs two levels, so a kernel fixed at "radial" -- or one whose
  # other value failed every fold -- stopped the plot with "contrasts can
  # be applied only to factors with 2 or more levels"
  tuned <- tl_tune_random(
    iris, Species ~ ., method = "svm",
    param_space = list(kernel = "radial", cost = c(0.1, 10)),
    n_iter = 3, folds = 2, verbose = FALSE, seed = 2
  )
  importance <- tl_plot_tuning_results(tuned, plot_type = "importance")$data
  expect_identical(importance$importance[importance$parameter == "kernel"], 0)

  results <- attr(tuned, "tuning_results")$results
  expected <- abs(stats::cor(results$cost, results$mean_metric))
  expect_equal(importance$importance[importance$parameter == "cost"], expected)

  set.seed(3)
  failing <- suppressWarnings(tl_tune_grid(
    iris[, 1:4], Sepal.Length ~ ., method = "svm",
    param_grid = list(kernel = c("linear", "nope"), cost = c(1, 2)),
    folds = 2, verbose = FALSE
  ))
  importance <- tl_plot_tuning_results(failing, plot_type = "importance")$data
  expect_identical(importance$importance[importance$parameter == "kernel"], 0)
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
    "a range with equal ends"
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

test_that("the tuners name a bad metric, direction, fold count or n_iter", {
  # metric = c("rmse", "mae") failed with "the condition has length > 1",
  # n_iter = 0 reported that every parameter set failed in every fold, and a
  # fold count rsample refused came back as an error about `v`
  grid <- list(cp = c(0.01, 0.1))
  tune <- function(...) {
    tl_tune_grid(mtcars, mpg ~ wt, method = "tree", param_grid = grid,
                 verbose = FALSE, ...)
  }
  search <- function(...) {
    tl_tune_random(mtcars, mpg ~ wt, method = "tree", param_space = grid,
                   verbose = FALSE, seed = 1, ...)
  }

  expect_error(tune(folds = 2, metric = c("rmse", "mae")),
               "'metric' must be a single metric name")
  expect_error(search(folds = 2, metric = c("rmse", "mae")),
               "'metric' must be a single metric name")
  expect_error(tune(folds = 2, maximize = c(TRUE, FALSE)),
               "'maximize' must be TRUE, FALSE or NULL")
  expect_error(search(folds = 2, n_iter = 0),
               "'n_iter' must be a single whole number of at least 1; got 0")
  expect_error(search(folds = 2, n_iter = 2.5),
               "'n_iter' must be a single whole number of at least 1; got 2.5")
  for (folds in list(2.5, 1, c(2, 3), 40)) {
    expect_error(tune(folds = folds),
                 "'folds' must be a whole number between 2 and nrow\\(data\\)",
                 info = deparse(folds))
    expect_error(search(folds = folds),
                 "'folds' must be a whole number between 2 and nrow\\(data\\)",
                 info = deparse(folds))
  }
  expect_error(
    tl_tune_grid(mtcars, mpg ~ wt, method = "forrest", param_grid = grid,
                 folds = 2, verbose = FALSE),
    "'method' must be one of the supervised methods"
  )

  # An unknown name is refused before anything is fitted
  fitted <- 0
  testthat::local_mocked_bindings(tl_model = function(...) {
    fitted <<- fitted + 1
    stop("tl_model() should not be reached")
  })
  expect_error(tune(folds = 2, metric = "acuracy"),
               "Metric \"acuracy\" was not produced for this task")
  expect_identical(fitted, 0)
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

test_that("tl_compare_cv refits a model with the arguments it was built with", {
  shallow <- tl_model(mtcars, mpg ~ ., method = "tree", cp = 0.5)
  deep <- tl_model(mtcars, mpg ~ ., method = "tree",
                   cp = 0.0001, minsplit = 2)

  set.seed(1)
  cv <- tl_compare_cv(mtcars, list(shallow = shallow, deep = deep),
                      folds = 3, metrics = "rmse")
  rmse <- split(cv$fold_metrics$value, cv$fold_metrics$model)
  expect_false(isTRUE(all.equal(rmse$shallow, rmse$deep)))

  # An argument passed to tl_compare_cv() overrides the recorded one, so
  # both models become the shallow tree again
  set.seed(1)
  cv_override <- tl_compare_cv(mtcars, list(shallow = shallow, deep = deep),
                               folds = 3, metrics = "rmse",
                               cp = 0.5, minsplit = 20)
  rmse <- split(cv_override$fold_metrics$value,
                cv_override$fold_metrics$model)
  expect_equal(rmse$shallow, rmse$deep)
})

test_that("tl_compare_cv refuses a per-row argument it cannot split", {
  w <- rep(1, nrow(mtcars))
  weighted <- tl_model(mtcars, mpg ~ wt, method = "linear", weights = w)
  plain <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_error(
    tl_compare_cv(mtcars, list(weighted = weighted, plain = plain),
                  folds = 3),
    "cannot re-split 'weights'"
  )
})

test_that("the tuners refuse a per-row argument they cannot split", {
  # Every fold received the whole weight vector, so each fit failed with
  # "variable lengths differ" and the search ended in "Every parameter set
  # failed in every fold"
  w <- rep(c(1, 2), length.out = nrow(mtcars))
  expect_error(
    tl_tune_grid(mtcars, mpg ~ wt + hp, method = "tree",
                 param_grid = list(cp = c(0.01, 0.1)), folds = 3,
                 verbose = FALSE, weights = w),
    "tl_tune_grid\\(\\) cannot re-split 'weights' across folds"
  )
  expect_error(
    tl_tune_random(mtcars, mpg ~ wt + hp, method = "tree",
                   param_space = list(cp = c(0.01, 0.1)), n_iter = 2,
                   folds = 3, verbose = FALSE, seed = 1, subset = 1:20),
    "tl_tune_random\\(\\) cannot re-split 'subset' across folds"
  )

  # Other fitting arguments still reach every fold and the final fit
  set.seed(1)
  tuned <- tl_tune_grid(mtcars, mpg ~ wt + hp, method = "tree",
                        param_grid = list(cp = c(0.01, 0.1)), folds = 3,
                        verbose = FALSE, maxdepth = 1)
  expect_identical(tuned$fit$control$maxdepth, 1)
})

test_that("forward and both selection expand a dot formula", {
  # step() expanded `.` against the start model's `1`, so there were no
  # terms to add and every call returned mpg ~ 1
  for (direction in c("forward", "both")) {
    model <- tl_step_selection(mtcars, mpg ~ ., direction = direction)
    expect_gt(length(attr(terms(model$spec$formula), "term.labels")), 0)
  }

  # `- wt` is honoured as well, rather than wt re-entering the scope
  model <- tl_step_selection(mtcars, mpg ~ . - wt, direction = "forward")
  expect_false("wt" %in% attr(terms(model$spec$formula), "term.labels"))

  # Backward selection on an explicit formula is unchanged
  model <- tl_step_selection(mtcars, mpg ~ wt + hp + qsec + drat,
                             direction = "backward")
  reference <- step(lm(mpg ~ wt + hp + qsec + drat, data = mtcars),
                    trace = FALSE)
  expect_equal(coef(model$fit), coef(reference))
})

test_that("forward and both selection keep offsets and the intercept setting", {
  # The starting model was update(formula, . ~ 1), which dropped offset()
  # terms and put back an intercept the caller had removed with - 1
  for (direction in c("forward", "both")) {
    model <- tl_step_selection(mtcars, mpg ~ wt + hp + offset(qsec),
                               direction = direction)
    reference <- step(
      lm(mpg ~ 1 + offset(qsec), mtcars),
      scope = list(lower = ~ 1 + offset(qsec),
                   upper = ~ wt + hp + offset(qsec)),
      direction = direction, trace = 0
    )
    expect_equal(coef(model$fit), coef(reference), info = direction)
    expect_equal(
      unname(stats::model.offset(stats::model.frame(model$fit))),
      mtcars$qsec, info = direction
    )

    model <- tl_step_selection(mtcars, mpg ~ wt + hp - 1,
                               direction = direction)
    expect_identical(attr(stats::terms(model$fit), "intercept"), 0L,
                     info = direction)
    reference <- step(
      lm(mpg ~ 0, mtcars),
      scope = list(lower = ~ 0, upper = ~ wt + hp - 1),
      direction = direction, trace = 0
    )
    expect_equal(coef(model$fit), coef(reference), info = direction)
  }
})

test_that("tl_compare_cv refuses foldid, and overrides a list argument whole", {
  foldid <- rep(1:4, length.out = nrow(mtcars))
  lasso <- tl_model(mtcars, mpg ~ ., method = "lasso", foldid = foldid)
  linear <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_error(
    tl_compare_cv(mtcars, list(lasso = lasso, linear = linear), folds = 3),
    "cannot re-split 'foldid'"
  )

  # With parms replaced rather than merged, the information-split tree and
  # the default tree receive identical parms and score identically
  info <- tl_model(iris, Species ~ ., method = "tree",
                   parms = list(split = "information"))
  gini <- tl_model(iris, Species ~ ., method = "tree")
  set.seed(1)
  cv <- tl_compare_cv(iris, list(info = info, gini = gini), folds = 3,
                      metrics = "accuracy",
                      parms = list(prior = c(0.2, 0.3, 0.5)))
  accuracy <- split(cv$fold_metrics$value, cv$fold_metrics$model)
  expect_equal(accuracy$info, accuracy$gini)
})

# ---- task, fold coverage, sampling and default grids -----------------

collect_warnings <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = warnings)
}

test_that("tuning logistic on a numeric 0/1 response scores accuracy", {
  mt <- mtcars
  mt$am01 <- mt$am

  # The tuners chose the default metric from is.factor(y), so a 0/1 response
  # got "rmse", which tl_model() -- treating logistic as classification --
  # never produces
  grid <- suppressWarnings(tl_tune_grid(
    mt, am01 ~ wt + hp, method = "logistic",
    param_grid = list(maxit = c(25, 50)), folds = 2, verbose = FALSE
  ))
  tuning <- attr(grid, "tuning_results")
  expect_identical(tuning$metric, "accuracy")
  expect_true(tuning$maximize)
  expect_true(is.finite(tuning$best_metric))

  random <- suppressWarnings(tl_tune_random(
    mt, am01 ~ wt + hp, method = "logistic",
    param_space = list(maxit = c(25, 50)), n_iter = 2, folds = 2,
    verbose = FALSE, seed = 1
  ))
  expect_identical(attr(random, "tuning_results")$metric, "accuracy")

  # A numeric response under any other method is still regression
  tree <- tl_tune_grid(
    mtcars, mpg ~ wt + hp, method = "tree",
    param_grid = list(cp = c(0.01, 0.1)), folds = 2, verbose = FALSE
  )
  expect_identical(attr(tree, "tuning_results")$metric, "rmse")
})

test_that("the response note is given once per search, not once per fold", {
  # tl_model() notes a numeric response with few values once per fit, and
  # every fold refit repeated it: a 2-set, 3-fold search printed it 7 times
  note <- "Note: Response 'cyl' has 3 unique numeric values"
  messages_of <- function(expr) {
    seen <- character()
    withCallingHandlers(expr, message = function(m) {
      seen <<- c(seen, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
    seen
  }

  set.seed(1)
  seen <- messages_of(tl_tune_grid(
    mtcars, cyl ~ wt + hp, method = "tree",
    param_grid = list(cp = c(0.01, 0.1)), folds = 3, verbose = TRUE
  ))
  # Once, from the final fit on all the rows
  expect_identical(sum(grepl(note, seen, fixed = TRUE)), 1L)
  # The search's own progress messages are still given
  expect_true(any(grepl("Parameter set 2 of 2", seen, fixed = TRUE)))

  seen <- messages_of(tl_tune_random(
    mtcars, cyl ~ wt + hp, method = "tree",
    param_space = list(cp = c(0.01, 0.1)), n_iter = 2, folds = 3,
    verbose = FALSE, seed = 1
  ))
  expect_identical(sum(grepl(note, seen, fixed = TRUE)), 1L)

  # tl_compare_cv() refits models the caller has already built, and the
  # note was given when they were
  shallow <- suppressMessages(tl_model(mtcars, cyl ~ wt, method = "tree"))
  deep <- suppressMessages(tl_model(mtcars, cyl ~ wt + hp, method = "tree"))
  set.seed(1)
  seen <- messages_of(tl_compare_cv(
    mtcars, list(shallow = shallow, deep = deep), folds = 3, metrics = "rmse"
  ))
  expect_identical(sum(grepl(note, seen, fixed = TRUE)), 0L)
})

test_that("a parameter set that failed a fold cannot win", {
  results <- data.frame(
    mean_metric = c(3.0, 2.0, 2.5),
    n_folds_ok = c(3L, 2L, 3L)
  )

  # Set 2 has the lowest error, but on two of the three folds
  expect_identical(
    tl_tune_select_best(results, maximize = FALSE, folds = 3,
                        labels = c("a", "b", "c")),
    3L
  )
  # Complete sets are still compared on their scores
  expect_identical(
    tl_tune_select_best(results, maximize = TRUE, folds = 3,
                        labels = c("a", "b", "c")),
    1L
  )
})

test_that("tuning results record how many folds each set completed", {
  skip_if_not_installed("gbm")

  # gbm refuses nTrain * bag.fraction <= 2 * n.minobsinnode + 1, which
  # fails the 15-row training fold for 3.3 and passes the 16-row one
  set.seed(1)
  run <- collect_warnings(tl_tune_grid(
    mtcars[1:31, ], mpg ~ wt + hp, method = "boost",
    param_grid = list(n.minobsinnode = c(1, 3.3), n.trees = 50,
                      bag.fraction = 0.5),
    folds = 2, verbose = FALSE
  ))
  tuning <- attr(run$value, "tuning_results")

  expect_true(any(grepl("parameters: n.minobsinnode=3.3", run$warnings)))
  expect_identical(tuning$results$n_folds_ok, c(2L, 1L))
  expect_identical(tuning$best_params$n.minobsinnode, 1)
})

test_that("when no set completes every fold the most complete one is used", {
  skip_if_not_installed("gbm")

  set.seed(1)
  run <- collect_warnings(tl_tune_grid(
    mtcars[1:31, ], mpg ~ wt + hp, method = "boost",
    param_grid = list(n.minobsinnode = c(3.3, 3.4), n.trees = 50,
                      bag.fraction = 0.5),
    folds = 2, verbose = FALSE
  ))
  tuning <- attr(run$value, "tuning_results")

  expect_identical(tuning$results$n_folds_ok, c(1L, 1L))
  expect_true(any(grepl("No parameter set completed all 2 folds",
                        run$warnings)))
  expect_true(is.finite(tuning$best_metric))

  # The same fallback in the random tuner
  run <- collect_warnings(tl_tune_random(
    mtcars[1:31, ], mpg ~ wt + hp, method = "boost",
    param_space = list(n.minobsinnode = 3.3, n.trees = 50,
                       bag.fraction = 0.5),
    n_iter = 1, folds = 2, verbose = FALSE, seed = 1
  ))
  expect_true(any(grepl("No parameter set completed all 2 folds",
                        run$warnings)))
})

test_that("a metric undefined on a fold is not reported as a failed fit", {
  # precision is undefined on a fold where nothing is predicted positive.
  # No fit failed, yet the warning said no set "completed all 10 folds" and
  # pointed at errors from failed fits
  d <- mtcars
  d$am <- factor(d$am, labels = c("auto", "manual"))
  set.seed(1)
  splits <- rsample::vfold_cv(d, v = 10)$splits
  predicts_positive <- vapply(splits, function(split) {
    fit <- rpart::rpart(am ~ wt + hp, data = rsample::analysis(split),
                        method = "class")
    any(predict(fit, rsample::assessment(split), type = "class") == "manual")
  }, logical(1))
  expect_false(all(predicts_positive))

  set.seed(1)
  run <- collect_warnings(tl_tune_grid(
    d, am ~ wt + hp, method = "tree", param_grid = list(cp = c(0.01, 0.1)),
    folds = 10, metric = "precision", verbose = FALSE
  ))
  tuning <- attr(run$value, "tuning_results")
  expect_identical(tuning$results$n_folds_ok[1], sum(predicts_positive))
  expect_false(any(grepl("Error fitting model", run$warnings)))
  fallback <- grep("No parameter set", run$warnings, value = TRUE)
  expect_length(fallback, 1)
  expect_match(fallback, "No parameter set was scored on all 10 folds")
  expect_match(fallback, "No fit failed: \"precision\" could not be computed")
  expect_false(grepl("failed fit", fallback))

  # Both causes at once are both named
  results <- data.frame(mean_metric = c(0.5, 0.6), n_folds_ok = c(2L, 2L))
  expect_warning(
    tl_tune_select_best(results, TRUE, 3, c("a", "b"), n_failed = c(1L, 0L),
                        metric = "precision"),
    "On some folds the fit failed and on others \"precision\" could not"
  )

  # With nothing scored, the stop says which of the two happened
  results <- data.frame(mean_metric = c(NA_real_, NA_real_),
                        n_folds_ok = c(0L, 0L))
  expect_error(
    tl_tune_select_best(results, TRUE, 3, c("a", "b"), n_failed = c(0L, 0L),
                        metric = "precision"),
    "every fit succeeded, but \"precision\" could not be computed on any fold"
  )
  expect_error(
    tl_tune_select_best(results, TRUE, 3, c("a", "b"), n_failed = c(3L, 1L),
                        metric = "precision"),
    "some fits failed, and on the other folds \"precision\" could not"
  )
  expect_error(
    tl_tune_select_best(results, TRUE, 3, c("a", "b"), n_failed = c(3L, 3L),
                        metric = "precision"),
    "Every parameter set failed in every fold"
  )
})

test_that("a fold where auc is undefined is left out of the tuners' scores", {
  # A fold holding one class stopped the search with ROCR's "Number of
  # classes is not equal to 2"
  d <- mtcars
  d$am <- factor(d$am, labels = c("auto", "manual"))
  set.seed(1)
  one_class <- vapply(rsample::vfold_cv(d, v = 10)$splits, function(split) {
    length(unique(rsample::assessment(split)$am)) == 1L
  }, logical(1))
  expect_true(any(one_class))

  for (tuner in c("grid", "random")) {
    set.seed(1)
    run <- collect_warnings(
      if (tuner == "grid") {
        tl_tune_grid(d, am ~ wt + hp, method = "tree",
                     param_grid = list(cp = c(0.01, 0.1)), folds = 10,
                     metric = "auc", verbose = FALSE)
      } else {
        tl_tune_random(d, am ~ wt + hp, method = "tree",
                       param_space = list(cp = c(0.01, 0.1)), n_iter = 2,
                       folds = 10, metric = "auc", verbose = FALSE)
      }
    )
    tuning <- attr(run$value, "tuning_results")
    expect_identical(tuning$results$n_folds_ok, rep(sum(!one_class), 2),
                     info = tuner)
    expect_true(any(grepl("No fit failed: \"auc\" could not be computed",
                          run$warnings)), info = tuner)
  }
})

# Every predictor missing on the rows of one assessment fold. vfold_cv()
# assigns rows from the seed and the row count alone, so the fold's rows can
# be found first and blanked.
blank_one_fold <- function(seed = 1, folds = 4) {
  data <- mtcars[, c("mpg", "wt", "hp")]
  set.seed(seed)
  rows <- rsample::complement(rsample::vfold_cv(data, v = folds)$splits[[2]])
  data$wt[rows] <- NA
  data$hp[rows] <- NA
  data
}

test_that("a fold with no row to score is left out of the tuners' scores", {
  # tl_evaluate() refuses a fold on which no row can be scored, and the
  # refusal stopped the whole search
  d <- blank_one_fold()
  for (tuner in c("grid", "random")) {
    set.seed(1)
    run <- collect_warnings(
      if (tuner == "grid") {
        tl_tune_grid(d, mpg ~ wt + hp, method = "svm",
                     param_grid = list(cost = c(0.5, 1)), folds = 4,
                     verbose = FALSE)
      } else {
        tl_tune_random(d, mpg ~ wt + hp, method = "svm",
                       param_space = list(cost = c(0.5, 1)), n_iter = 2,
                       folds = 4, verbose = FALSE)
      }
    )
    tuning <- attr(run$value, "tuning_results")
    expect_identical(tuning$results$n_folds_ok, c(3L, 3L), info = tuner)
    expect_true(any(grepl(
      "Fold 2 is left out of the score for cost=.*none of its 8 rows can be",
      run$warnings
    )), info = tuner)
    expect_true(any(grepl("No fit failed: \"rmse\" could not be computed",
                          run$warnings)), info = tuner)
  }

  # The score is the mean over the other three folds, as e1071 fits them
  set.seed(1)
  splits <- rsample::vfold_cv(d, v = 4)$splits
  rmse <- vapply(splits[-2], function(split) {
    fit <- e1071::svm(mpg ~ wt + hp, data = rsample::analysis(split),
                      type = "eps-regression", kernel = "radial", cost = 0.5)
    test <- rsample::assessment(split)
    sqrt(mean((predict(fit, test) - test$mpg)^2))
  }, numeric(1))
  set.seed(1)
  tuned <- suppressWarnings(tl_tune_grid(
    d, mpg ~ wt + hp, method = "svm", param_grid = list(cost = 0.5),
    folds = 4, verbose = FALSE
  ))
  expect_equal(attr(tuned, "tuning_results")$results$mean_metric, mean(rmse))
})

test_that("tl_tune_random keeps a single value as given", {
  set.seed(1)

  # sample(20, 1) draws from 1:20, and a logical was drawn from
  # c(TRUE, FALSE) whatever value was supplied
  expect_true(all(replicate(50, tl_draw_param(20)) == 20))
  expect_true(all(replicate(50, tl_draw_param(0.05)) == 0.05))
  expect_true(all(replicate(50, tl_draw_param(TRUE))))
  expect_true(all(replicate(50, tl_draw_param(c(TRUE, TRUE)))))
  expect_identical(tl_draw_param("gini"), "gini")

  # The multi-value forms are read as before
  expect_setequal(replicate(50, tl_draw_param(c(TRUE, FALSE))),
                  c(TRUE, FALSE))
  expect_setequal(replicate(50, tl_draw_param(c("a", "b"))), c("a", "b"))
  ints <- replicate(100, tl_draw_param(c(10, 20)))
  expect_true(all(ints %in% 10:20))
  expect_gt(length(unique(ints)), 2)
  cont <- replicate(50, tl_draw_param(c(0.01, 0.1)))
  expect_true(all(cont >= 0.01 & cont <= 0.1))
  expect_gt(length(unique(cont)), 2)

  model <- tl_tune_random(
    iris, Species ~ ., method = "tree",
    param_space = list(minsplit = 20, cp = c(0.01, 0.1)),
    n_iter = 4, folds = 2, verbose = FALSE, seed = 1
  )
  expect_true(all(attr(model, "tuning_results")$results$minsplit == 20))
})

test_that("tl_default_param_grid has no grid for logistic regression", {
  # The ridge lambda grid it used to return is not a glm() argument, so
  # every fit failed
  expect_warning(
    grid <- tl_default_param_grid("logistic"),
    "glm\\(\\) has no hyperparameter to tune"
  )
  expect_identical(grid, list())

  # The methods with a grid still return one, without a warning
  for (method in c("tree", "ridge", "lasso", "elastic_net")) {
    expect_no_warning(grid <- tl_default_param_grid(method))
    expect_gt(length(grid), 0)
  }
})

test_that("the default forest grid asks only for what randomForest reads", {
  large <- tl_default_param_grid("forest", size = "large")

  # sampsize is a row count, and the grid held fractions of one
  expect_false("sampsize" %in% names(large))
  expect_true(all(large$mtry >= 1))
})

test_that("a forest mtry above the predictor count is capped", {
  skip_if_not_installed("randomForest")

  # iris has four predictors. randomForest resets mtry = 6 to 4 with a
  # warning in every fold, and the results reported 6 as if it had been used
  run <- collect_warnings(tl_tune_grid(
    iris, Species ~ ., method = "forest",
    param_grid = list(mtry = c(2, 6), ntree = 20), folds = 2, verbose = FALSE
  ))
  results <- attr(run$value, "tuning_results")$results
  expect_true(any(grepl("mtry = 6 exceeds the 4 predictors", run$warnings)))
  expect_false(any(grepl("invalid mtry", run$warnings)))
  expect_setequal(results$mtry, c(2, 4))

  # A grid that fits is left alone
  run <- collect_warnings(tl_tune_grid(
    iris, Species ~ ., method = "forest",
    param_grid = list(mtry = c(2, 4), ntree = 20), folds = 2, verbose = FALSE
  ))
  expect_length(run$warnings, 0)
  expect_setequal(attr(run$value, "tuning_results")$results$mtry, c(2, 4))

  # The random tuner caps each draw
  run <- collect_warnings(tl_tune_random(
    iris, Species ~ ., method = "forest",
    param_space = list(mtry = 9, ntree = 20), n_iter = 1, folds = 2,
    verbose = FALSE, seed = 1
  ))
  expect_true(any(grepl("mtry = 9 exceeds the 4 predictors", run$warnings)))
  expect_false(any(grepl("invalid mtry", run$warnings)))
  expect_equal(attr(run$value, "tuning_results")$best_params$mtry, 4)
})

test_that("the mtry cap counts predictors as randomForest does", {
  skip_if_not_installed("randomForest")

  # mpg ~ . - id has three predictors, but its model frame keeps the id
  # column, so four were counted: mtry = 4 went uncapped, randomForest reset
  # it to 3 in every fold, and best_params reported 4 for a fit using 3
  d <- mtcars[, c("mpg", "wt", "hp", "qsec")]
  d$id <- seq_len(nrow(d))
  set.seed(1)
  run <- collect_warnings(tl_tune_grid(
    d, mpg ~ . - id, method = "forest",
    param_grid = list(mtry = c(3, 4), ntree = 20), folds = 3, verbose = FALSE
  ))
  expect_true(any(grepl("mtry = 4 exceeds the 3 predictors", run$warnings)))
  expect_false(any(grepl("invalid mtry", run$warnings)))
  tuning <- attr(run$value, "tuning_results")
  expect_identical(tuning$results$mtry, 3)
  expect_equal(tuning$best_params$mtry, run$value$fit$mtry)

  # randomForest's own count, read off the importance table it fits
  for (formula in list(mpg ~ . - id, mpg ~ wt * hp, mpg ~ log(wt) + wt)) {
    set.seed(1)
    forest <- randomForest::randomForest(formula, data = d, ntree = 5)
    capped <- suppressWarnings(
      tl_tune_cap_mtry(list(list(mtry = 99)), "forest", formula, d)
    )
    expect_equal(capped[[1]]$mtry, nrow(forest$importance),
                 info = deparse(formula))
  }
})

test_that("list-valued grid cells reach the model unwrapped", {
  grid <- do.call(tidyr::crossing, list(
    hidden_layers = list(c(10), c(10, 5)), dropout = c(0, 0.2)
  ))

  # crossing() stores a vector-valued candidate as a list column, so the
  # row handed to tl_model() carried list(c(10, 5)) rather than c(10, 5)
  expect_identical(
    tl_tune_grid_row(grid, 3),
    list(hidden_layers = c(10, 5), dropout = 0)
  )
  # A list-valued argument is unwrapped once, not flattened
  parms <- do.call(tidyr::crossing, list(parms = list(list(split = "gini"))))
  expect_identical(
    tl_tune_grid_row(parms, 1),
    list(parms = list(split = "gini"))
  )

  # Through both tuners, recording what tl_model() receives and fitting a
  # tree in its place so that no deep model is built
  real_model <- tl_model
  seen <- list()
  testthat::local_mocked_bindings(
    tl_model = function(data, formula, method, ..., hidden_layers) {
      seen[[length(seen) + 1]] <<- hidden_layers
      real_model(data, formula, method = "tree")
    }
  )
  is_pair <- function(x) identical(x, c(10, 5))

  model <- tl_tune_grid(
    iris, Species ~ ., method = "deep",
    param_grid = list(hidden_layers = list(c(10), c(10, 5))),
    folds = 2, verbose = FALSE
  )
  expect_true(all(vapply(seen, is.numeric, logical(1))))
  expect_true(any(vapply(seen, is_pair, logical(1))))
  tuning <- attr(model, "tuning_results")
  expect_true(is.numeric(tuning$best_params$hidden_layers))
  expect_identical(tuning$results$hidden_layers, list(10, c(10, 5)))

  seen <- list()
  model <- tl_tune_random(
    iris, Species ~ ., method = "deep",
    param_space = list(hidden_layers = list(c(10, 5))),
    n_iter = 1, folds = 2, verbose = FALSE, seed = 1
  )
  expect_true(all(vapply(seen, is_pair, logical(1))))
  expect_identical(
    attr(model, "tuning_results")$best_params$hidden_layers, c(10, 5)
  )
})

test_that("verbose tuning describes character and vector parameters", {
  # round(unlist(params), 4) failed on a character parameter, so verbose
  # random search over an svm kernel stopped before fitting anything
  messages <- testthat::capture_messages(
    tl_tune_random(
      iris, Species ~ ., method = "svm",
      param_space = list(kernel = c("linear", "radial")),
      n_iter = 1, folds = 2, verbose = TRUE, seed = 1
    )
  )
  expect_true(any(grepl("Iteration 1 of 1: kernel=(linear|radial)",
                        messages)))
  expect_identical(
    tl_tune_format_params(list(hidden_layers = c(10, 5), cp = 0.012345)),
    "hidden_layers=c(10, 5), cp=0.01235"
  )
})

test_that("a search where every fit fails says so", {
  # With nothing scored, best_params came back empty and the final fit
  # failed with "argument is of length zero"
  for (tuner in c("grid", "random")) {
    msg <- tryCatch(
      suppressWarnings(
        if (tuner == "grid") {
          tl_tune_grid(
            mtcars, mpg ~ wt, method = "svm",
            param_grid = list(kernel = c("nope1", "nope2")),
            folds = 2, verbose = FALSE
          )
        } else {
          tl_tune_random(
            mtcars, mpg ~ wt, method = "svm",
            param_space = list(kernel = c("nope1", "nope2")),
            n_iter = 2, folds = 2, verbose = FALSE, seed = 1
          )
        }
      ),
      error = function(e) conditionMessage(e)
    )
    expect_match(msg, "Every parameter set failed in every fold", info = tuner)
  }

  # One set scored on a single fold is enough to go on with
  results <- data.frame(mean_metric = c(NA, 2), n_folds_ok = c(0L, 1L))
  expect_warning(
    best <- tl_tune_select_best(results, FALSE, 2, c("a", "b")),
    "No parameter set completed all 2 folds. Using b"
  )
  expect_identical(best, 2L)
})

test_that("forward selection keeps a transformed response", {
  model <- tl_step_selection(mtcars, log(mpg) ~ wt + hp + qsec,
                             direction = "forward")
  expect_identical(deparse(model$spec$formula[[2]]), "log(mpg)")
})

test_that("tl_compare_cv keeps every model under its own name", {
  m_wt <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m_wt_hp <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(
    tl_compare_cv(mtcars, list(a = m_wt, a = m_wt_hp), folds = 3,
                  metrics = "rmse"),
    "unique"
  )
  set.seed(1)
  cv <- tl_compare_cv(mtcars, list(a = m_wt, m_wt_hp), folds = 3,
                      metrics = "rmse")
  expect_setequal(cv$summary$model, c("a", "Model_2"))
})

test_that("tl_compare_cv names a bad model list, fold count or metric", {
  m <- tl_model(mtcars, mpg ~ wt, method = "linear")
  # list() failed with "argument is not interpretable as logical", and a
  # bare model was read as the list of its own components
  expect_error(tl_compare_cv(mtcars, list(), folds = 3),
               "'models' must be a non-empty list of tidylearn models")
  expect_error(tl_compare_cv(mtcars, m, folds = 3), "wrap it in list\\(\\)")
  km <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  expect_error(tl_compare_cv(iris, list(km = km), folds = 3),
               "compares supervised models; 'km' is an unsupervised model")
  expect_error(tl_compare_cv(mtcars, list(m = m), folds = 2.5),
               "'folds' must be a whole number between 2 and nrow\\(data\\)")

  # A misspelt metric was dropped without a word, and a metric of the other
  # task returned an empty summary
  tree <- tl_model(iris, Species ~ ., method = "tree")
  expect_error(
    tl_compare_cv(iris, list(tree = tree), folds = 3,
                  metrics = c("accuracy", "acuracy")),
    "Unknown classification metric\\(s\\) in 'metrics': \"acuracy\""
  )
  expect_error(
    tl_compare_cv(iris, list(tree = tree), folds = 3, metrics = "rmse"),
    "Unknown classification metric\\(s\\) in 'metrics': \"rmse\""
  )
  expect_error(
    tl_compare_cv(iris, list(tree = tree), folds = 3, metrics = character()),
    "'metrics' is empty. Name at least one of: accuracy"
  )
  set.seed(1)
  cv <- tl_compare_cv(iris, list(tree = tree), folds = 3,
                      metrics = c("accuracy", "f1"))
  expect_setequal(cv$summary$metric, c("accuracy", "f1"))
})

test_that("tl_compare_cv refuses a model a refit would not reproduce", {
  # A refit from formula, method and arguments scored a semi-supervised
  # model as a tree trained on every training-fold label, and failed on the
  # is_anomaly column an anomaly-aware model is fitted with
  set.seed(1)
  semi <- suppressWarnings(tl_semisupervised(
    iris, Species ~ ., labeled_indices = c(1:5, 51:55, 101:105)
  ))
  expect_error(
    tl_compare_cv(iris, list(semi = semi), folds = 3, metrics = "accuracy"),
    "cannot refit 'semi': tl_semisupervised\\(\\)"
  )
  flag <- tl_anomaly_aware(iris, Species ~ ., response = "Species",
                           action = "flag")
  expect_error(
    tl_compare_cv(iris, list(flag = flag), folds = 3, metrics = "accuracy"),
    "cannot refit 'flag': tl_anomaly_aware\\(\\)"
  )

  # tl_auto_ml() marks a model fitted on features it engineered, PCA scores
  # or cluster labels, with $feature_transform
  engineered <- tl_model(iris, Species ~ ., method = "tree")
  engineered$feature_transform <- list(kind = "cluster")
  plain <- tl_model(iris, Species ~ ., method = "tree")
  expect_error(
    tl_compare_cv(iris, list(plain = plain, engineered = engineered),
                  folds = 3, metrics = "accuracy"),
    "cannot refit 'engineered': tl_auto_ml\\(\\)"
  )

  # A model tl_model() built is refitted as before
  set.seed(1)
  cv <- tl_compare_cv(iris, list(plain = plain), folds = 3,
                      metrics = "accuracy")
  expect_identical(nrow(cv$fold_metrics), 3L)
})

test_that("a fold on which auc is undefined is left out of tl_compare_cv()", {
  # A fold holding one class stopped the whole comparison with ROCR's
  # "Number of classes is not equal to 2"
  d <- mtcars
  d$am <- factor(d$am, labels = c("auto", "manual"))
  m1 <- tl_model(d, am ~ wt, method = "tree")
  m2 <- tl_model(d, am ~ wt + hp, method = "tree")
  set.seed(2)
  one_class <- vapply(rsample::vfold_cv(d, v = 5)$splits, function(split) {
    length(unique(rsample::assessment(split)$am)) == 1L
  }, logical(1))
  expect_true(any(one_class))

  set.seed(2)
  cv <- suppressWarnings(tl_compare_cv(d, list(a = m1, b = m2), folds = 5))
  auc <- cv$fold_metrics[cv$fold_metrics$metric == "auc", ]
  for (name in c("a", "b")) {
    values <- auc$value[auc$model == name]
    expect_identical(is.na(values), one_class, info = name)
    in_summary <- cv$summary$model == name & cv$summary$metric == "auc"
    expect_equal(cv$summary$mean_value[in_summary],
                 mean(values[!one_class]), info = name)
  }
})

test_that("a fold with no row to score is left out of tl_compare_cv()", {
  # tl_evaluate() refuses a fold on which no row can be scored, and the
  # refusal stopped the whole comparison
  d <- blank_one_fold()
  simple <- tl_model(d, mpg ~ wt, method = "linear")
  full <- tl_model(d, mpg ~ wt + hp, method = "linear")
  set.seed(1)
  run <- collect_warnings(tl_compare_cv(
    d, list(simple = simple, full = full), folds = 4,
    metrics = c("rmse", "mae")
  ))
  folds <- run$value$fold_metrics
  expect_true(all(is.na(folds$value[folds$fold == 2])))
  expect_false(anyNA(folds$value[folds$fold != 2]))
  expect_setequal(folds$metric[folds$fold == 2], c("rmse", "mae"))
  for (name in c("simple", "full")) {
    expect_true(any(grepl(
      paste0("Fold 2 is left out of the summary for '", name, "'"),
      run$warnings
    )), info = name)
  }

  # The summary is over the other three folds, as lm() fits them
  set.seed(1)
  splits <- rsample::vfold_cv(d, v = 4)$splits
  rmse <- vapply(splits[-2], function(split) {
    fit <- lm(mpg ~ wt, rsample::analysis(split))
    test <- rsample::assessment(split)
    sqrt(mean((predict(fit, test) - test$mpg)^2))
  }, numeric(1))
  summary <- run$value$summary
  expect_equal(
    summary$mean_value[summary$model == "simple" & summary$metric == "rmse"],
    mean(rmse)
  )

  # and the paired test uses the three folds both models were scored on
  test <- tl_test_model_difference(run$value, baseline_model = "simple",
                                   metric = "rmse")
  values <- split(folds$value[folds$metric == "rmse"],
                  folds$model[folds$metric == "rmse"])
  expect_equal(
    test$p_value,
    stats::t.test(values$full[-2], values$simple[-2], paired = TRUE)$p.value
  )
})

test_that("a metric undefined on every fold summarises as NA", {
  # min() and max() over no values returned Inf and -Inf, each with a
  # warning, and mean() returned NaN
  rare <- data.frame(x = seq_len(30),
                     y = factor(rep(c("neg", "pos"), c(26, 4))))
  # minsplit above the row count leaves the root alone, so "neg" is
  # predicted everywhere and precision has no predicted positives
  stump <- tl_model(rare, y ~ x, method = "tree", minsplit = 100)
  set.seed(1)
  splits <- rsample::vfold_cv(rare, v = 3)$splits
  set.seed(1)
  run <- collect_warnings(tl_compare_cv(
    rare, list(stump = stump), folds = 3,
    metrics = c("accuracy", "precision")
  ))
  expect_false(any(grepl("no non-missing arguments", run$warnings)))
  summary <- run$value$summary
  precision <- summary[summary$metric == "precision", ]
  for (column in c("mean_value", "sd_value", "min_value", "max_value")) {
    expect_identical(precision[[column]], NA_real_, info = column)
  }
  accuracy <- vapply(splits, function(split) {
    mean(rsample::assessment(split)$y == "neg")
  }, numeric(1))
  expect_equal(summary$mean_value[summary$metric == "accuracy"],
               mean(accuracy))
})

test_that("tl_test_model_difference reads \"wilcox.test\" as \"wilcox\"", {
  # The diagnostics vignette wrote "wilcox.test", which match.arg() refused
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  set.seed(1)
  cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 6,
                      metrics = "rmse")
  alias <- tl_test_model_difference(cv, baseline_model = "simple",
                                    metric = "rmse", test = "wilcox.test")
  expect_identical(
    alias,
    tl_test_model_difference(cv, baseline_model = "simple", metric = "rmse",
                             test = "wilcox")
  )
  rmse <- split(cv$fold_metrics$value, cv$fold_metrics$model)
  direct <- stats::wilcox.test(rmse$full, rmse$simple, paired = TRUE)
  expect_equal(alias$p_value, direct$p.value)
  expect_error(tl_test_model_difference(cv, test = "wilcoxon"),
               "'test' must be \"t.test\" or \"wilcox\"")
})

test_that("tl_test_model_difference compares the folds both models scored", {
  # The mean difference was taken over each model's own scored folds, so a
  # fold one model could not be scored on shifted it away from the paired
  # test beside it
  a <- c(0.5, 0.6, NA, 0.8)
  b <- c(0.4, 0.7, 0.9, 0.6)
  cv <- list(fold_metrics = data.frame(
    metric = "precision", value = c(a, b), fold = rep(1:4, 2),
    model = rep(c("a", "b"), each = 4)
  ))
  result <- tl_test_model_difference(cv, baseline_model = "a",
                                     metric = "precision")
  both <- !is.na(a) & !is.na(b)
  expect_equal(result$mean_diff, mean(b[both] - a[both]))
  expect_equal(result$p_value,
               stats::t.test(b[both], a[both], paired = TRUE)$p.value)

  # Fewer than two scored pairs cannot be tested; that comparison is NA
  # rather than an error that discards every other metric's result
  cv$fold_metrics$value[cv$fold_metrics$model == "a"] <- c(NA, NA, NA, 0.8)
  cv$fold_metrics <- rbind(cv$fold_metrics, data.frame(
    metric = "accuracy", value = c(0.7, 0.8, 0.75, 0.9, 0.8, 0.85, 0.8, 0.95),
    fold = rep(1:4, 2), model = rep(c("a", "b"), each = 4)
  ))
  expect_warning(
    result <- tl_test_model_difference(cv, baseline_model = "a"),
    "1 fold scored for both"
  )
  expect_true(is.na(result$p_value[result$metric == "precision"]))
  expect_false(is.na(result$p_value[result$metric == "accuracy"]))
})

test_that("BIC selection penalises by the rows the model used", {
  set.seed(4)
  n <- 60
  d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  d$y <- d$x1 + 0.3 * d$x3 + rnorm(n)
  d$y[sample(n, 40)] <- NA
  model <- tl_step_selection(d, y ~ x1 + x2 + x3, direction = "backward",
                             criterion = "BIC")
  reference <- step(lm(y ~ x1 + x2 + x3, d), k = log(20), trace = 0)
  expect_setequal(attr(terms(model$spec$formula), "term.labels"),
                  attr(terms(formula(reference)), "term.labels"))
})

test_that("stepwise selection fits every candidate on the same rows", {
  # Each candidate was fitted on the rows it could use, so a variable with
  # missing values changed the row count as it entered or left, and step()
  # stopped with "number of rows in use has changed"
  complete <- stats::na.omit(airquality)
  upper <- formula(lm(Ozone ~ ., complete))
  by_name <- function(x) x[order(names(x))]
  for (direction in c("forward", "backward", "both")) {
    expect_message(
      model <- tl_step_selection(airquality, Ozone ~ ., direction = direction,
                                 criterion = "BIC"),
      "111 of 153 rows"
    )
    start <- if (direction == "backward") {
      lm(Ozone ~ ., complete)
    } else {
      lm(Ozone ~ 1, complete)
    }
    scope <- if (direction == "backward") {
      upper
    } else {
      list(lower = ~ 1, upper = upper)
    }
    reference <- step(start, scope = scope, direction = direction,
                      k = log(nrow(complete)), trace = 0)
    expect_equal(by_name(coef(model$fit)), by_name(coef(reference)),
                 info = direction)
    # The rows left out are recorded where lm() records its own, so the
    # fit still lines up with the data the caller passed
    expect_identical(nrow(model$data), nrow(airquality))
    expect_identical(sort(tl_fitted_rows(model)),
                     which(stats::complete.cases(airquality)))
  }

  # Backward selection dropping the one column with missing values
  set.seed(1)
  d <- mtcars[, c("mpg", "wt", "hp", "qsec")]
  d$noise <- stats::rnorm(nrow(d))
  d$noise[1:3] <- NA
  expect_message(
    model <- tl_step_selection(d, mpg ~ ., direction = "backward"),
    "29 of 32 rows"
  )
  reference <- step(lm(mpg ~ ., stats::na.omit(d)), trace = 0)
  expect_equal(coef(model$fit), coef(reference))

  # A variable from the caller's frame stays aligned with the data's rows
  d$hp[4] <- NA
  select_with_local <- function() {
    trend <- seq_len(nrow(d))
    tl_step_selection(d, mpg ~ wt + hp + trend, direction = "forward")
  }
  expect_message(model <- select_with_local(), "31 of 32 rows")
  trend <- seq_len(nrow(d))
  rows <- -4
  reference <- step(
    lm(mpg ~ 1, d[rows, ]),
    scope = list(lower = ~ 1, upper = ~ wt + hp + trend[rows]),
    direction = "forward", trace = 0
  )
  expect_equal(unname(coef(model$fit)), unname(coef(reference)))

  # Complete data is left as it is, without a message, and so are rows
  # missing only the response, which every candidate drops alike
  expect_no_message(
    tl_step_selection(mtcars, mpg ~ wt + hp, direction = "backward")
  )
  d <- mtcars
  d$mpg[1:3] <- NA
  expect_no_message(
    model <- tl_step_selection(d, mpg ~ wt + hp, direction = "forward")
  )
  expect_identical(stats::nobs(model$fit), 29L)
})

test_that("a generated model name does not collide with a chosen one", {
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ hp, method = "linear")
  set.seed(1)
  cv <- tl_compare_cv(mtcars, list(m1, Model_1 = m2), folds = 3,
                      metrics = "rmse")
  # The unnamed first model is Model_1 by position, which the caller gave to
  # the second, so it is numbered on
  expect_setequal(cv$summary$model, c("Model_1.1", "Model_1"))
})

test_that("a range whose ends match is the single value", {
  draws <- replicate(20, tl_draw_param(c(20, 20)))
  expect_true(all(draws == 20))
})

test_that("forward selection sees a variable from the caller's frame", {
  select_with_local <- function() {
    noise <- seq_len(nrow(mtcars))
    tl_step_selection(mtcars, mpg ~ wt + hp + noise, direction = "forward")
  }
  expect_s3_class(select_with_local(), "tidylearn_model")
})

test_that("linear and polynomial get grids that say what they are", {
  expect_warning(grid <- tl_default_param_grid("linear"), "lm\\(\\)")
  expect_identical(grid, list())
  expect_no_warning(poly <- tl_default_param_grid("polynomial"))
  expect_named(poly, "degree")
  model <- tl_model(mtcars, mpg ~ wt, method = "polynomial",
                    degree = max(poly$degree))
  expect_s3_class(model, "tidylearn_model")
})

test_that("tl_default_param_grid gives xgboost a grid tl_model() can fit", {
  # xgboost is a supported method, and the grid warned "Unknown method"
  for (size in c("small", "medium", "large")) {
    expect_no_warning(grid <- tl_default_param_grid("xgboost", size = size))
    expect_gt(length(grid), 0)
    expect_true(all(names(grid) %in% names(formals(tl_fit_xgboost))),
                info = size)
  }

  # The values tl_tune_xgboost() searches by default, plus nrounds, which
  # that function chooses by early stopping and tl_tune_grid() has to tune
  large <- tl_default_param_grid("xgboost", size = "large")
  expect_identical(
    large[setdiff(names(large), "nrounds")],
    list(max_depth = c(3, 6, 9), eta = c(0.01, 0.1, 0.3),
         subsample = c(0.7, 1.0), colsample_bytree = c(0.7, 1.0),
         min_child_weight = c(1, 3, 5), gamma = c(0, 0.1, 0.2))
  )
  for (size in c("small", "medium")) {
    grid <- tl_default_param_grid("xgboost", size = size)
    for (param in setdiff(names(grid), "nrounds")) {
      expect_true(all(grid[[param]] %in% large[[param]]),
                  info = paste(size, param))
    }
  }

  skip_if_not_installed("xgboost")
  set.seed(1)
  tuned <- tl_tune_grid(mtcars, mpg ~ ., method = "xgboost",
                        param_grid = tl_default_param_grid("xgboost", "small"),
                        folds = 2, verbose = FALSE)
  results <- attr(tuned, "tuning_results")$results
  expect_true(all(results$n_folds_ok == 2))
  expect_identical(tuned$spec$args$nrounds,
                   attr(tuned, "tuning_results")$best_params$nrounds)
})

test_that("is_classification fits the grid to the task", {
  # The argument was documented and never read
  classify <- tl_default_param_grid("svm", "medium", is_classification = TRUE)
  regress <- tl_default_param_grid("svm", "medium", is_classification = FALSE)
  # epsilon is the width of the regression SVM's insensitive band; a
  # classifier has no use for it
  expect_false("epsilon" %in% names(classify))
  expect_true("epsilon" %in% names(regress))
  expect_identical(regress[names(classify)], classify)
  expect_identical(tl_default_param_grid("svm", "medium"), classify)
  fit <- tl_model(mtcars, mpg ~ wt + hp, method = "svm",
                  epsilon = regress$epsilon[1])
  expect_identical(fit$fit$epsilon, regress$epsilon[1])

  # randomForest's default nodesize is 1 for classification and 5 for
  # regression, and each grid is built around its own
  forest_classify <- tl_default_param_grid("forest", "large", TRUE)
  forest_regress <- tl_default_param_grid("forest", "large", FALSE)
  expect_identical(min(forest_classify$nodesize), 1)
  expect_true(5 %in% forest_regress$nodesize)
  expect_gt(min(forest_regress$nodesize), 1)
  expect_identical(forest_regress[c("mtry", "ntree")],
                   forest_classify[c("mtry", "ntree")])

  expect_error(tl_default_param_grid("svm", is_classification = NA),
               "'is_classification' must be TRUE or FALSE")
  expect_error(tl_default_param_grid(c("tree", "forest")),
               "'method' must be a single method name")
})

test_that("a one-parameter search says why the default plot is unavailable", {
  set.seed(1)
  tuned <- tl_tune_grid(mtcars, mpg ~ wt + hp, method = "tree",
                        param_grid = list(cp = c(0.01, 0.1)), folds = 2,
                        verbose = FALSE)
  expect_error(tl_plot_tuning_results(tuned), "needs two tuned parameters")
  expect_s3_class(tl_plot_tuning_results(tuned, plot_type = "parallel"),
                  "ggplot")
})

test_that("a forest mtry below 1 is raised to 1", {
  set.seed(1)
  expect_warning(
    tuned <- tl_tune_grid(mtcars, mpg ~ wt + hp, method = "forest",
                          param_grid = list(mtry = c(0, 2), ntree = 20),
                          folds = 2, verbose = FALSE),
    "below 1"
  )
  expect_setequal(attr(tuned, "tuning_results")$results$mtry, c(1, 2))
})

test_that("an empty candidate vector is named", {
  expect_error(
    tl_tune_grid(mtcars, mpg ~ wt, method = "tree",
                 param_grid = list(cp = numeric(0)), folds = 2,
                 verbose = FALSE),
    "no candidate values for: cp"
  )
})

test_that("random search names an empty or degenerate parameter space", {
  expect_error(
    tl_tune_random(mtcars, mpg ~ wt, method = "tree",
                   param_space = list(cp = numeric(0)), n_iter = 2,
                   folds = 2, verbose = FALSE),
    "no candidate values for: cp"
  )
  expect_error(
    tl_tune_random(mtcars, mpg ~ wt, method = "tree",
                   param_space = list(cp = c(0.05, 0.05)), n_iter = 2,
                   folds = 2, verbose = FALSE),
    "equal ends"
  )
  # A list is a set of candidates, even one that looks like a log spec
  expect_no_error(tl_check_param_space(list(size = list(10, 1, "log"))))
  expect_error(tl_check_param_space(list(cp = c(0.2, 0.001))),
               "Write it as c\\(0.001, 0.2\\)\\.")
})

test_that("a per-row argument is recorded by name, not copied", {
  # The fit already holds the weights; keeping their values in the spec as
  # well doubled them for nothing, since a fold cannot use them
  w <- rep(c(1, 2), length.out = nrow(mtcars))
  model <- tl_model(mtcars, mpg ~ wt, method = "linear", weights = w,
                    x = TRUE)
  expect_false("weights" %in% names(model$spec$args))
  expect_identical(model$spec$per_row_args, "weights")
  # other arguments are still kept whole
  expect_true(isTRUE(model$spec$args$x))

  plain <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_identical(plain$spec$per_row_args, character(0))

  # tl_compare_cv() still refuses the recorded one, and one passed to it
  expect_error(tl_compare_cv(mtcars, list(a = model, b = plain), folds = 3),
               "cannot re-split 'weights'")
  expect_error(tl_compare_cv(mtcars, list(a = plain, b = plain), folds = 3,
                             weights = w),
               "cannot re-split 'weights'")
})
