make_regression_pipeline <- function(data = mtcars) {
  tl_pipeline(
    data, mpg ~ wt + hp,
    models = list(linear = list(method = "linear")),
    evaluation = list(
      metrics = "rmse", validation = "cv",
      cv_folds = 3, best_metric = "rmse"
    )
  )
}

test_that("tl_run_pipeline records preprocessing statistics on the raw scale", {
  set.seed(201)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)
  stats_learned <- res$results$preprocessing_stats

  expect_equal(stats_learned$center$wt, mean(mtcars$wt))
  expect_equal(stats_learned$scale$wt, stats::sd(mtcars$wt))
  expect_equal(stats_learned$medians$wt, stats::median(mtcars$wt))

  # The response is never standardized
  expect_null(stats_learned$center$mpg)

  # The stored training data is standardized, so deriving the centre and
  # scale from it would give 0 and 1
  expect_equal(mean(res$results$processed_data$wt), 0)
  expect_equal(stats::sd(res$results$processed_data$wt), 1)
})

test_that("tl_predict_pipeline standardizes new data with training stats", {
  set.seed(202)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)

  preds <- tl_predict_pipeline(res, new_data = mtcars)
  in_sample <- predict(
    res$results$best_model,
    new_data = res$results$processed_data
  )$.pred

  expect_equal(preds$.pred, in_sample)

  # Predictions must land on the response scale, not hundreds of units away
  expect_true(all(abs(preds$.pred - mtcars$mpg) < 10))
})

test_that("tl_predict_pipeline imputes with the raw training median", {
  set.seed(203)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)

  new_data <- mtcars[1:5, ]
  new_data$wt[1] <- NA

  preds <- tl_predict_pipeline(res, new_data = new_data)

  # Substituting the median by hand must give the same answer
  manual <- mtcars[1:5, ]
  manual$wt[1] <- stats::median(mtcars$wt)
  expect_equal(preds$.pred, tl_predict_pipeline(res, new_data = manual)$.pred)
  expect_false(any(is.na(preds$.pred)))
})

test_that("tl_predict_pipeline rejects pipelines without recorded stats", {
  set.seed(204)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)
  res$results$preprocessing_stats <- NULL

  expect_error(
    tl_predict_pipeline(res, new_data = mtcars[1:5, ]),
    "did not record preprocessing statistics"
  )
})

test_that("tl_run_pipeline requires a named models list", {
  pipe <- tl_pipeline(mtcars, mpg ~ wt + hp, models = "linear")

  expect_error(tl_run_pipeline(pipe, verbose = FALSE), "named list")
})

test_that("tl_run_pipeline selects a best model using the default metrics", {
  skip_if_not_installed("rpart")

  binary_iris <- droplevels(subset(iris, Species != "virginica"))

  # Default evaluation asks for f1, which requires tl_evaluate to honour
  # the metrics argument
  pipe <- tl_pipeline(
    binary_iris, Species ~ .,
    models = list(tree = list(method = "tree"))
  )
  pipe$evaluation$cv_folds <- 3

  set.seed(205)
  res <- tl_run_pipeline(pipe, verbose = FALSE)

  expect_equal(res$results$best_model_name, "tree")
  expect_false(is.na(res$results$metric_values[["tree"]]))
})

test_that("tl_run_pipeline standardizes constant columns safely", {
  data <- mtcars[, c("mpg", "wt", "hp")]
  data$constant <- 1

  pipe <- tl_pipeline(
    data, mpg ~ wt + hp + constant,
    models = list(linear = list(method = "linear")),
    evaluation = list(
      metrics = "rmse", validation = "cv",
      cv_folds = 3, best_metric = "rmse"
    )
  )

  set.seed(206)
  res <- tl_run_pipeline(pipe, verbose = FALSE)

  expect_equal(res$results$preprocessing_stats$scale$constant, 1)
  expect_false(any(is.na(res$results$processed_data$constant)))
})
