make_binary_data <- function(seed = 101, n = 40) {
  set.seed(seed)
  data.frame(
    x = c(stats::rnorm(n, -1), stats::rnorm(n, 1)),
    y = factor(rep(c("a", "b"), each = n))
  )
}

test_that("tl_evaluate scores classification on class labels", {
  d <- make_binary_data()
  model <- tl_model(d, y ~ x, method = "logistic")

  ev <- tl_evaluate(model)
  manual <- mean(predict(model, new_data = d, type = "class")$.pred == d$y)

  expect_equal(ev$metric, "accuracy")
  expect_equal(ev$value, manual)
  # The default predict type for logistic returns probabilities; comparing
  # those to class labels scores every model at zero
  expect_gt(ev$value, 0.5)
})

test_that("tl_evaluate honours the metrics argument for classification", {
  d <- make_binary_data()
  model <- tl_model(d, y ~ x, method = "logistic")

  requested <- c("accuracy", "precision", "recall", "f1", "auc")
  ev <- tl_evaluate(model, metrics = requested)

  expect_setequal(ev$metric, requested)
  expect_false(any(is.na(ev$value)))
})

test_that("tl_evaluate honours the metrics argument for regression", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_equal(tl_evaluate(model)$metric, c("rmse", "mae", "rsq"))

  requested <- c("rmse", "mse", "mae", "mape", "rsq")
  ev <- tl_evaluate(model, metrics = requested)
  expect_setequal(ev$metric, requested)

  value_of <- function(m) ev$value[ev$metric == m]
  expect_equal(value_of("rmse"), sqrt(value_of("mse")))
  expect_equal(
    value_of("rsq"),
    summary(stats::lm(mpg ~ wt + hp, data = mtcars))$r.squared
  )
})

test_that("tl_evaluate errors when the response is missing from new data", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_error(
    tl_evaluate(model, new_data = mtcars[, c("wt", "hp")]),
    "not found in the evaluation data"
  )
})

test_that("tl_cv forwards metrics to tl_evaluate", {
  set.seed(102)
  cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3,
              metrics = c("rmse", "mape"))

  expect_setequal(cv$summary$metric, c("rmse", "mape"))
  expect_false(any(is.na(cv$summary$mean)))
})

test_that("tl_calc_regression_metrics handles degenerate inputs", {
  # Constant actuals leave no variance to explain
  res <- tl_calc_regression_metrics(rep(2, 5), rep(2, 5), metrics = "rsq")
  expect_true(is.na(res$value))

  # Zero actuals are excluded from MAPE rather than producing Inf
  res <- tl_calc_regression_metrics(c(0, 2, 4), c(0, 3, 5), metrics = "mape")
  expect_equal(res$value, mean(c(0.5, 0.25)) * 100)

  # Missing values are dropped pairwise
  res <- tl_calc_regression_metrics(c(1, NA, 3), c(1, 5, 3), metrics = "mae")
  expect_equal(res$value, 0)
})
