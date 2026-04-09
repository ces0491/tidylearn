# ---- Diagnostics functions ----

# -- Influence measures --

test_that("tl_influence_measures returns data frame for linear model", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  inf <- tl_influence_measures(model)

  expect_s3_class(inf, "data.frame")
  expect_equal(nrow(inf), nrow(mtcars))
  expect_true("cooks_distance" %in% names(inf))
  expect_true("leverage" %in% names(inf))
  expect_true("dffits" %in% names(inf))
  expect_true("is_influential" %in% names(inf))
})

test_that("tl_influence_measures respects custom thresholds", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  # Very strict threshold should flag more observations
  strict <- tl_influence_measures(model, threshold_cook = 0.001)
  # Very lenient threshold should flag fewer
  lenient <- tl_influence_measures(model, threshold_cook = 10)

  expect_gte(
    sum(strict$is_cook_influential),
    sum(lenient$is_cook_influential)
  )
})

test_that("tl_influence_measures errors for unsupported methods", {
  skip_if_not_installed("randomForest")

  model <- tl_model(iris, Species ~ ., method = "forest")

  expect_error(tl_influence_measures(model), "linear-based")
})

test_that("tl_influence_measures includes dfbetas columns", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  inf <- tl_influence_measures(model)

  # Should have dfbetas for intercept, wt, hp
  dfbetas_cols <- grep("^dfbetas_", names(inf), value = TRUE)
  expect_true(length(dfbetas_cols) >= 3)
})

test_that("tl_influence_measures stores thresholds as attributes", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  inf <- tl_influence_measures(model)

  expect_true(!is.null(attr(inf, "threshold_cook")))
  expect_true(!is.null(attr(inf, "threshold_leverage")))
  expect_true(!is.null(attr(inf, "threshold_dffits")))
})

# -- Influence plotting --

test_that("tl_plot_influence returns ggplot for cook type", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  p <- tl_plot_influence(model, plot_type = "cook")

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_influence returns ggplot for leverage type", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  p <- tl_plot_influence(model, plot_type = "leverage")

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_influence returns ggplot for index type", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  p <- tl_plot_influence(model, plot_type = "index")

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_influence errors for invalid plot type", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_error(tl_plot_influence(model, plot_type = "invalid"), "Invalid")
})

# -- Assumption checking --

test_that("tl_check_assumptions returns list for linear model", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  assumptions <- tl_check_assumptions(model, test = FALSE, verbose = FALSE)

  expect_type(assumptions, "list")
  expect_true("linearity" %in% names(assumptions))
  expect_true("normality" %in% names(assumptions))
  expect_true("homoscedasticity" %in% names(assumptions))
  expect_true("multicollinearity" %in% names(assumptions))
  expect_true("outliers" %in% names(assumptions))
  expect_true("overall" %in% names(assumptions))
})

test_that("tl_check_assumptions each check has standard structure", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  assumptions <- tl_check_assumptions(model, test = FALSE, verbose = FALSE)

  # Each assumption (except overall) should have assumption, check,
  # details, recommendation
  for (name in setdiff(names(assumptions), "overall")) {
    check <- assumptions[[name]]
    expect_true("assumption" %in% names(check),
                label = paste(name, "has assumption field"))
    expect_true("details" %in% names(check),
                label = paste(name, "has details field"))
    expect_true("recommendation" %in% names(check),
                label = paste(name, "has recommendation field"))
  }
})

test_that("tl_check_assumptions overall has correct counts", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  assumptions <- tl_check_assumptions(model, test = FALSE, verbose = FALSE)

  overall <- assumptions$overall
  expect_true("n_checked" %in% names(overall))
  expect_true("n_violated" %in% names(overall))
  expect_true("n_satisfied" %in% names(overall))
  expect_equal(overall$n_checked, overall$n_violated + overall$n_satisfied)
})

test_that("tl_check_assumptions errors for unsupported methods", {
  skip_if_not_installed("randomForest")

  model <- tl_model(iris, Species ~ ., method = "forest")

  expect_error(tl_check_assumptions(model), "linear-based")
})

test_that("tl_check_assumptions works with statistical tests when available", {
  skip_if_not_installed("car")
  skip_if_not_installed("lmtest")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  assumptions <- tl_check_assumptions(model, test = TRUE, verbose = FALSE)

  # Should include Durbin-Watson and Breusch-Pagan results
  expect_true(grepl("Durbin-Watson", assumptions$independence$details))
  expect_true(grepl("Breusch-Pagan", assumptions$homoscedasticity$details))
})

test_that("tl_check_assumptions works for polynomial model", {
  model <- tl_model(mtcars, mpg ~ wt, method = "polynomial", degree = 2)
  assumptions <- tl_check_assumptions(model, test = FALSE, verbose = FALSE)

  expect_type(assumptions, "list")
  expect_true("linearity" %in% names(assumptions))
})

# -- Outlier detection --

test_that("tl_detect_outliers works with IQR method", {
  result <- tl_detect_outliers(iris, variables = c("Sepal.Length"),
                               method = "iqr", plot = FALSE)

  expect_type(result, "list")
  expect_true("outlier_flags" %in% names(result))
  expect_true("outlier_counts" %in% names(result))
  expect_true("outlier_indices" %in% names(result))
  expect_equal(result$method, "iqr")
})

test_that("tl_detect_outliers works with z-score method", {
  result <- tl_detect_outliers(iris, variables = c("Sepal.Length"),
                               method = "z-score", plot = FALSE)

  expect_type(result, "list")
  expect_equal(result$method, "z-score")
})

test_that("tl_detect_outliers works with cook method", {
  result <- tl_detect_outliers(
    mtcars,
    variables = c("mpg", "wt", "hp"),
    method = "cook", plot = FALSE
  )

  expect_type(result, "list")
  expect_equal(result$method, "cook")
})

test_that("tl_detect_outliers works with mahalanobis method", {
  result <- tl_detect_outliers(
    iris,
    variables = c("Sepal.Length", "Sepal.Width"),
    method = "mahalanobis", plot = FALSE
  )

  expect_type(result, "list")
  expect_equal(result$method, "mahalanobis")
})

test_that("tl_detect_outliers auto-selects numeric variables", {
  result <- tl_detect_outliers(iris, method = "iqr", plot = FALSE)

  expect_type(result, "list")
  # Should have used all 4 numeric columns
  expect_equal(ncol(result$outlier_flags), 4)
})

test_that("tl_detect_outliers errors for non-numeric variable", {
  expect_error(
    tl_detect_outliers(iris, variables = "Species", plot = FALSE),
    "not numeric"
  )
})

test_that("tl_detect_outliers errors for nonexistent variable", {
  expect_error(
    tl_detect_outliers(iris, variables = "nonexistent", plot = FALSE),
    "not found"
  )
})

test_that("tl_detect_outliers respects threshold parameter", {
  strict <- tl_detect_outliers(iris, method = "iqr",
                               threshold = 0.5, plot = FALSE)
  lenient <- tl_detect_outliers(iris, method = "iqr",
                                threshold = 3.0, plot = FALSE)

  # Stricter threshold should find more outliers
  expect_gte(strict$outlier_counts$total, lenient$outlier_counts$total)
})

test_that("tl_detect_outliers creates plot when requested", {
  result <- tl_detect_outliers(iris, variables = c("Sepal.Length"),
                               method = "iqr", plot = TRUE)

  expect_true(!is.null(result$plot))
  expect_s3_class(result$plot, "ggplot")
})

test_that("tl_detect_outliers returns NULL plot when plot = FALSE", {
  result <- tl_detect_outliers(iris, method = "iqr", plot = FALSE)

  expect_null(result$plot)
})

test_that("tl_detect_outliers errors for invalid method", {
  expect_error(
    tl_detect_outliers(iris, method = "invalid", plot = FALSE),
    "Invalid method"
  )
})

test_that("tl_detect_outliers cook method requires 2+ variables", {
  expect_error(
    tl_detect_outliers(iris, variables = "Sepal.Length",
                       method = "cook", plot = FALSE),
    "at least 2"
  )
})

test_that("tl_detect_outliers mahalanobis requires 2+ variables", {
  expect_error(
    tl_detect_outliers(iris, variables = "Sepal.Length",
                       method = "mahalanobis", plot = FALSE),
    "at least 2"
  )
})

# -- Diagnostic dashboard --

test_that("tl_diagnostic_dashboard errors without gridExtra", {
  skip_if(requireNamespace("gridExtra", quietly = TRUE),
          "gridExtra is installed, cannot test missing-package path")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_diagnostic_dashboard(model), "gridExtra")
})

# -- Classification auto-detection fix --

test_that("numeric response with few unique values is treated as regression", {
  # Create data with numeric response having <= 10 unique values
  set.seed(42)
  data <- data.frame(
    y = sample(1:5, 50, replace = TRUE),
    x1 = rnorm(50),
    x2 = rnorm(50)
  )

  # Should be treated as regression (not classification) since y is numeric
  expect_message(
    model <- tl_model(data, y ~ x1 + x2, method = "linear"),
    "unique numeric values"
  )
  expect_false(model$spec$is_classification)
})

test_that("factor response is treated as classification regardless of levels", {
  data <- data.frame(
    y = factor(rep(c("A", "B"), 25)),
    x1 = rnorm(50)
  )

  model <- tl_model(data, y ~ x1, method = "logistic")
  expect_true(model$spec$is_classification)
})
