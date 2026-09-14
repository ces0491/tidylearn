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
