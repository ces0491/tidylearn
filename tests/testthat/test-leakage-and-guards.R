# Regression tests for defects that produced a plausible wrong answer
# rather than an error, and for the guards added alongside them.

# ---- preprocessing is learned inside the resampling ------------------

test_that("pipeline folds learn their own preprocessing statistics", {
  # The bug: centres, scales and medians were learned from the whole
  # dataset and only then split, so every assessment row helped define
  # the transformation it was scored under.
  set.seed(42)
  data <- data.frame(x1 = stats::rnorm(80), x2 = stats::rnorm(80))
  data$y <- 2 * data$x1 + stats::rnorm(80)

  captured <- list()
  local_learn <- tl_learn_preprocessing

  # Each fold's statistics must come from its own analysis rows, so they
  # cannot all equal the full-data statistics.
  full_stats <- local_learn(
    data, y ~ x1 + x2,
    list(impute_missing = TRUE, standardize = TRUE)
  )

  folds <- split(seq_len(80), rep(1:4, length.out = 80))
  for (i in seq_along(folds)) {
    captured[[i]] <- local_learn(
      data[-folds[[i]], ], y ~ x1 + x2,
      list(impute_missing = TRUE, standardize = TRUE)
    )
  }

  fold_centres <- vapply(captured, function(s) s$center$x1, numeric(1))
  expect_false(all(fold_centres == full_stats$center$x1))
})

test_that("preprocessing never imputes the response", {
  data <- data.frame(x = c(1, 2, NA, 4, 5), y = c(1, NA, 3, 4, 5))

  stats_learned <- tl_learn_preprocessing(
    data, y ~ x,
    list(impute_missing = TRUE, standardize = FALSE)
  )

  # A median is recorded for the predictor but not for the outcome:
  # imputing y would fabricate both a training target and a piece of
  # evaluation ground truth.
  expect_true(!is.null(stats_learned$medians$x))
  expect_null(stats_learned$medians$y)

  applied <- tl_apply_preprocessing(
    data, list(impute_missing = TRUE, standardize = FALSE), stats_learned
  )
  expect_false(anyNA(applied$x))
  expect_true(is.na(applied$y[2]))
})

test_that("a pipeline still runs end to end and predicts", {
  set.seed(7)
  data <- data.frame(x1 = stats::rnorm(60), x2 = stats::rnorm(60))
  data$y <- 1.5 * data$x1 + stats::rnorm(60)

  pipe <- tl_pipeline(
    data, y ~ x1 + x2,
    preprocessing = list(
      impute_missing = TRUE, standardize = TRUE, dummy_encode = FALSE
    ),
    models = list(linear = list(method = "linear")),
    evaluation = list(
      metrics = "rmse", validation = "cv", cv_folds = 3,
      best_metric = "rmse"
    )
  )
  pipe <- tl_run_pipeline(pipe, verbose = FALSE)

  expect_false(is.null(pipe$results$best_model))
  expect_equal(nrow(tl_predict_pipeline(pipe, data[1:5, ])), 5)
})

# ---- leaderboard ranking ---------------------------------------------

test_that("mape ranks ascending, so the lowest error wins", {
  # mape was absent from the ascending-sort list, so it fell into the
  # descending branch and create_leaderboard() returned the model with
  # the HIGHEST error as the best one.
  results <- list(
    good = tibble::tibble(metric = "mape", value = 5),
    bad  = tibble::tibble(metric = "mape", value = 40)
  )

  leaderboard <- create_leaderboard(results, "mape", "regression")
  expect_equal(leaderboard$model[1], "good")

  # An accuracy-style metric still ranks the other way
  score_results <- list(
    good = tibble::tibble(metric = "accuracy", value = 0.9),
    bad  = tibble::tibble(metric = "accuracy", value = 0.4)
  )
  expect_equal(
    create_leaderboard(score_results, "accuracy", "classification")$model[1],
    "good"
  )
})

test_that("an unrankable metric is refused instead of guessed", {
  results <- list(a = tibble::tibble(metric = "mystery", value = 1))

  expect_error(
    create_leaderboard(results, "mystery", "regression"),
    "higher or lower is better"
  )
})

# ---- splitting --------------------------------------------------------

test_that("tl_split never returns an empty train or test set", {
  # floor(n * prop) could hit zero, and data[-integer(0), ] then selects
  # nothing -- so both halves came back empty and every row vanished.
  for (n in 2:12) {
    for (prop in c(0.1, 0.2, 0.5, 0.8, 0.95)) {
      split <- tl_split(iris[seq_len(n), ], prop = prop)

      expect_gt(nrow(split$train), 0)
      expect_gt(nrow(split$test), 0)
      expect_equal(nrow(split$train) + nrow(split$test), n)
    }
  }
})

# ---- clustering and distance -----------------------------------------

test_that("tidy_dbscan treats a dist input as a dissimilarity", {
  distances <- stats::dist(iris[, 1:4])

  ours <- tidy_dbscan(distances, eps = 0.5, minPts = 5)
  theirs <- dbscan::dbscan(distances, eps = 0.5, minPts = 5)

  expect_equal(ours$clusters$cluster, as.integer(theirs$cluster))
  expect_gt(ours$n_clusters, 0)
})

test_that("tidy_dbscan reports core points", {
  result <- tidy_dbscan(iris[, 1:4], eps = 0.5, minPts = 5)

  # Every non-noise cluster must contain at least one core point --
  # reading a non-existent "core" attribute made them all FALSE.
  expect_true(any(result$clusters$is_core))
  expect_true(all(result$summary$n_core > 0))
})

test_that("kmeans metrics survive non-default algorithms", {
  # kmeans() returns ifault = NULL for Lloyd/Forgy/MacQueen, and a
  # logical(0) recycled the whole metrics tibble to zero rows.
  for (algorithm in c("Hartigan-Wong", "Lloyd", "MacQueen")) {
    result <- suppressWarnings(
      tidy_kmeans(iris[, 1:4], k = 3, algorithm = algorithm)
    )

    expect_equal(nrow(result$metrics), 1L, info = algorithm)
    expect_false(is.na(result$metrics$tot_withinss), info = algorithm)
  }
})

test_that("tidy_gower matches named weights to their variables", {
  data <- data.frame(
    x = c(0, 1, 0, 1),
    color = factor(c("a", "a", "b", "b"))
  )

  by_name <- tidy_gower(data, weights = c(color = 100, x = 1))
  by_position <- tidy_gower(data, weights = c(x = 1, color = 100))

  # Naming the weights must determine which variable each applies to
  expect_equal(as.matrix(by_name), as.matrix(by_position))

  expect_error(
    tidy_gower(data, weights = c(color = 100)),
    "no entry for"
  )
})

# ---- out-of-sample transforms ----------------------------------------

test_that("PCA and kmeans predict by variable name, not column order", {
  pca <- tl_model(iris[, 1:4], method = "pca")
  clusters <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  in_order <- predict(pca, new_data = iris[1:10, 1:4])
  reordered <- predict(pca, new_data = iris[1:10, c(4, 3, 2, 1)])
  expect_equal(in_order$PC1, reordered$PC1)

  expect_equal(
    predict(clusters, new_data = iris[, 1:4])$cluster,
    predict(clusters, new_data = iris[, c(4, 3, 2, 1)])$cluster
  )

  expect_error(
    predict(pca, new_data = iris[, 1:3]),
    "new_data is missing: Petal.Width"
  )
})

# ---- diagnostics ------------------------------------------------------

test_that("the linearity check can actually fail", {
  # cor(fitted, residuals) is identically zero for any OLS fit with an
  # intercept, so the old check reported SATISFIED for every model.
  curved <- data.frame(x = 1:100)
  curved$y <- curved$x^2

  result <- suppressWarnings(tl_check_assumptions(
    tl_model(curved, y ~ x, method = "linear"), verbose = FALSE
  ))
  expect_false(isTRUE(result$linearity$check))

  set.seed(5)
  straight <- data.frame(x = stats::rnorm(200))
  straight$y <- 3 * straight$x + stats::rnorm(200)

  result_ok <- suppressWarnings(tl_check_assumptions(
    tl_model(straight, y ~ x, method = "linear"), verbose = FALSE
  ))
  expect_true(isTRUE(result_ok$linearity$check))
})

test_that("penalised fits are refused rather than half-diagnosed", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")

  expect_error(tl_check_assumptions(model), "penalised regression")
  expect_error(tl_influence_measures(model), "penalised regression")
})

# ---- credentials ------------------------------------------------------

test_that("connection strings are redacted before they can be printed", {
  expect_equal(
    tl_redact_db_url("postgres://alice:s3cret@db.host/prod"),
    "postgres://alice:***@db.host/prod"
  )
  expect_equal(
    tl_redact_db_url("mysql://bob:pw@host:3306/db"),
    "mysql://bob:***@host:3306/db"
  )

  # Non-URL sources pass through untouched
  expect_equal(tl_redact_db_url("localhost"), "localhost")
  expect_equal(tl_redact_db_url("/tmp/data.csv"), "/tmp/data.csv")

  # And the parse-failure path must not echo the password back
  expect_error(
    tl_parse_db_url("mysql://alice:s3cret@/db"),
    "\\*\\*\\*"
  )
  expect_false(
    grepl(
      "s3cret",
      tryCatch(
        tl_parse_db_url("mysql://alice:s3cret@/db"),
        error = function(e) conditionMessage(e)
      )
    )
  )
})

# ---- RNG hygiene -----------------------------------------------------

# set.seed() rewrites the session stream. A function taking a `seed` for
# its own reproducibility was also deciding what every later sample() or
# rnorm() in the caller's script returned, so two scripts differing only
# in whether they passed `seed` diverged everywhere downstream.

test_that("a seeded helper leaves the caller's random stream alone", {
  seeded_calls <- list(
    tl_split = function() tl_split(iris, prop = 0.7, seed = 1),
    tl_tune_random = function() {
      suppressWarnings(suppressMessages(tl_tune_random(
        iris[1:40, ], Sepal.Length ~ ., method = "tree",
        param_space = list(cp = c(0.001, 0.1)), n_iter = 2, seed = 1
      )))
    }
  )

  for (nm in names(seeded_calls)) {
    set.seed(999)
    before <- get(".Random.seed", envir = globalenv())
    invisible(seeded_calls[[nm]]())
    after <- get(".Random.seed", envir = globalenv())
    expect_identical(before, after, info = nm)
  }
})

test_that("the caller's next draw does not depend on the seed we were given", {
  set.seed(7)
  invisible(tl_split(iris, prop = 0.7, seed = 1))
  with_seed_one <- runif(1)

  set.seed(7)
  invisible(tl_split(iris, prop = 0.7, seed = 2))
  with_seed_two <- runif(1)

  expect_equal(with_seed_one, with_seed_two)
})

test_that("preserving the stream did not break the seed's own job", {
  expect_equal(
    tl_split(iris, prop = 0.7, seed = 42)$train,
    tl_split(iris, prop = 0.7, seed = 42)$train
  )
  expect_false(isTRUE(all.equal(
    tl_split(iris, prop = 0.7, seed = 42)$train,
    tl_split(iris, prop = 0.7, seed = 43)$train
  )))
})
