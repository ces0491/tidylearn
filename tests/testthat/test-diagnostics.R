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

test_that("influence measures cover a rank-deficient fit", {
  # dfbetas() has no column for an aliased coefficient, so looping over
  # coef() failed with "subscript out of bounds"
  d <- transform(mtcars, wt2 = 2 * wt)
  model <- tl_model(d, mpg ~ wt + wt2 + hp, method = "linear")
  inf <- tl_influence_measures(model)

  expect_setequal(grep("^dfbetas_", names(inf), value = TRUE),
                  c("dfbetas__Intercept_", "dfbetas_wt", "dfbetas_hp"))
  expect_equal(inf$dfbetas_hp, stats::dfbetas(model$fit)[, "hp"],
               ignore_attr = TRUE)
  # The thresholds count the three coefficients estimated, not four
  expect_equal(attr(inf, "threshold_leverage"), 2 * 3 / 32)
  expect_equal(attr(inf, "threshold_dffits"), 2 * sqrt(3 / 32))
})

test_that("diagnostics of an na.exclude fit match the na.omit fit", {
  # na.exclude pads every residual and influence measure back to all 32
  # rows while the fit used 30: "arguments imply differing number of rows:
  # 30, 32"
  d <- mtcars
  d$wt[c(3, 7)] <- NA
  excluded <- tl_model(d, mpg ~ wt + hp, method = "linear",
                       na.action = stats::na.exclude)
  omitted <- tl_model(d, mpg ~ wt + hp, method = "linear")

  inf <- tl_influence_measures(excluded)
  expect_identical(inf$observation, setdiff(1:32, c(3L, 7L)))
  expect_equal(inf, tl_influence_measures(omitted))
  expect_equal(inf$cooks_distance, stats::cooks.distance(omitted$fit),
               ignore_attr = TRUE)

  expect_equal(tl_check_assumptions(excluded, verbose = FALSE),
               tl_check_assumptions(omitted, verbose = FALSE))
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

test_that("a factor's GVIF is read on the VIF scale, not as its Df", {
  skip_if_not_installed("car")
  # car::vif() returns a GVIF table for a multi-df term, and max() over it
  # took the Df column: carb's 5 df were "Maximum VIF: 5", flagged as
  # multicollinearity, while its GVIF is 1.6
  d <- transform(mtcars, carb = factor(carb))
  model <- tl_model(d, mpg ~ wt + carb, method = "linear")
  result <- tl_check_assumptions(model, verbose = FALSE)$multicollinearity

  gvif <- car::vif(model$fit)
  adjusted <- gvif[, "GVIF"]^(1 / gvif[, "Df"])
  expect_true(result$check)
  expect_identical(result$details,
                   paste("Maximum VIF:", round(max(adjusted), 4)))

  # A collinear pair is still flagged when a factor is in the model
  d$wt_band <- cut(d$wt, breaks = c(0, 2.5, 3.2, 3.6, 6))
  collinear <- tl_model(d, mpg ~ wt + wt_band + hp, method = "linear")
  result <- tl_check_assumptions(collinear, verbose = FALSE)$multicollinearity
  gvif <- car::vif(collinear$fit)
  adjusted <- gvif[, "GVIF"]^(1 / gvif[, "Df"])
  expect_false(result$check)
  expect_identical(result$details,
                   paste("Maximum VIF:", round(max(adjusted), 4)))
})

test_that("the multicollinearity check is kept when VIF cannot run", {
  # The fallback assigned inside the error handler, which changed a copy:
  # mpg ~ wt came back with no multicollinearity element at all
  single <- tl_check_assumptions(tl_model(mtcars, mpg ~ wt, method = "linear"),
                                 verbose = FALSE)
  expect_identical(single$multicollinearity$details,
                   "Model has only one predictor")
  expect_true(single$multicollinearity$check)

  # car::vif() refuses aliased coefficients; the correlation fallback
  # finds the exact copy
  d <- transform(mtcars, wt2 = 2 * wt)
  aliased <- tl_model(d, mpg ~ wt + wt2 + hp, method = "linear")
  result <- tl_check_assumptions(aliased, verbose = FALSE)$multicollinearity
  expect_false(result$check)
  expect_identical(result$details, "Maximum correlation between predictors: 1")
})

test_that("a logistic model is not held to the OLS assumptions", {
  skip_if_not_installed("lmtest")
  # Shapiro-Wilk on the deviance residuals gave p = 0 with advice to
  # transform, and bptest() tested a linear probability model
  am_data <- transform(mtcars, am = factor(am))
  model <- tl_model(am_data, am ~ wt + hp, method = "logistic")

  for (test in c(TRUE, FALSE)) {
    result <- tl_check_assumptions(model, test = test, verbose = FALSE)
    expect_null(result$normality$check)
    expect_null(result$homoscedasticity$check)
    expect_match(result$normality$details,
                 "Not an assumption of logistic regression", fixed = TRUE)
    expect_match(result$homoscedasticity$details,
                 "Not an assumption of logistic regression", fixed = TRUE)
  }

  # Linearity, independence, multicollinearity and influence still count
  result <- tl_check_assumptions(model, verbose = FALSE)
  expect_identical(result$overall$n_checked, 4L)

  linear <- tl_check_assumptions(
    tl_model(mtcars, mpg ~ wt + hp, method = "linear"), verbose = FALSE
  )
  expect_match(linear$normality$details, "Shapiro-Wilk", fixed = TRUE)
  expect_match(linear$homoscedasticity$details, "Breusch-Pagan", fixed = TRUE)
})

test_that("an undecided check is reported as unknown, not violated", {
  # Two fitted values leave the linearity test nothing to fit. Its NA
  # stopped the verbose summary with "missing value where TRUE/FALSE
  # needed" and gave "NA assumption(s) appear to be violated" without it
  dd <- data.frame(y = rep(c(1, 3), each = 10) + rep(c(-1, 1), 10),
                   x = rep(0:1, each = 10))
  model <- suppressMessages(tl_model(dd, y ~ x, method = "linear"))

  messages <- testthat::capture_messages(
    result <- tl_check_assumptions(model)
  )
  expect_true(any(grepl("Linearity: UNKNOWN", messages, fixed = TRUE)))
  expect_true(is.na(result$linearity$check))
  expect_false(anyNA(unlist(result$overall[c("n_checked", "n_violated",
                                             "n_satisfied")])))
  expect_identical(result$overall$n_checked,
                   result$overall$n_violated + result$overall$n_satisfied)
  expect_no_match(result$overall$status, "NA", fixed = TRUE)

  # len ~ supp has two fitted values too, but floating-point noise made
  # them four distinct numbers, so the test ran and reported p = 1
  supp <- tl_model(ToothGrowth, len ~ supp, method = "linear")
  expect_true(is.na(
    tl_check_assumptions(supp, verbose = FALSE)$linearity$check
  ))
})

test_that("the Durbin-Watson statistic is the one lmtest reports", {
  skip_if_not_installed("lmtest")
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  dw <- unname(lmtest::dwtest(model$fit)$statistic)
  result <- tl_check_assumptions(model, verbose = FALSE)
  expect_identical(result$independence$details,
                   paste("Durbin-Watson statistic:", round(dw, 4)))
  expect_false(result$independence$check)
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

test_that("Cook's distance flags influential rows despite a missing value", {
  # lm() dropped the incomplete row, and its 31 distances were recycled
  # into a 32-row matrix, shifting every column: nine rows were flagged
  d <- mtcars
  d$wt[5] <- NA
  result <- tl_detect_outliers(d, c("mpg", "wt", "hp"), method = "cook",
                               plot = FALSE)

  cooks <- stats::cooks.distance(stats::lm(mpg ~ wt + hp, data = d))
  expect_identical(rownames(d)[result$outlier_indices],
                   names(cooks)[cooks > 4 / 32])
  expect_true(all(is.na(result$outlier_flags[5, ])))
  expect_identical(result$outlier_counts$total, 4L)

  plotted <- tl_detect_outliers(d, c("mpg", "wt", "hp"), method = "cook")
  expect_false(5 %in% plotted$plot$data$observation)
})

test_that("outlier counts survive a missing value", {
  # any() over a row holding an NA flag is NA, so the total was NA while
  # outlier_indices listed rows
  # Ozone above Q3 + 1.5 IQR in rows 62 and 117, Wind in 9, 18 and 48
  iqr <- tl_detect_outliers(airquality, method = "iqr", plot = FALSE)
  expect_identical(iqr$outlier_counts$total, 5L)
  expect_identical(iqr$outlier_indices, c(9L, 18L, 48L, 62L, 117L))

  z <- tl_detect_outliers(airquality, method = "z-score", plot = FALSE)
  expect_identical(z$outlier_counts$total, length(z$outlier_indices))

  mahal <- tl_detect_outliers(airquality, c("Ozone", "Solar.R", "Wind"),
                              method = "mahalanobis", plot = FALSE)
  expect_identical(mahal$outlier_counts$total, 3L)
})

test_that("Cook's distance accepts a non-syntactic column name", {
  # The formula was pasted together as "mpg ~ car weight + hp", which does
  # not parse: "unexpected symbol"
  renamed <- mtcars[, c("mpg", "wt", "hp")]
  names(renamed)[2] <- "car weight"
  result <- tl_detect_outliers(renamed, c("mpg", "car weight", "hp"),
                               method = "cook", plot = FALSE)

  cooks <- stats::cooks.distance(stats::lm(mpg ~ wt + hp, data = mtcars))
  expect_identical(result$outlier_indices, unname(which(cooks > 4 / 32)))
  expect_identical(colnames(result$outlier_flags),
                   c("mpg", "car weight", "hp"))
})

test_that("per-variable outlier flags stay a matrix for a single row", {
  # sapply() simplified one row's flags to a vector, and combining them
  # failed with "dim(X) must have a positive length"
  for (method in c("iqr", "z-score")) {
    result <- tl_detect_outliers(mtcars[1, ], c("mpg", "wt"), method = method,
                                 plot = FALSE)
    expect_identical(dim(result$outlier_flags), c(1L, 2L), info = method)
    expect_identical(colnames(result$outlier_flags), c("mpg", "wt"))
    expect_identical(result$outlier_counts$total, 0L, info = method)
  }
})

# -- Diagnostic dashboard --

test_that("tl_diagnostic_dashboard errors without gridExtra", {
  skip_if(requireNamespace("gridExtra", quietly = TRUE),
          "gridExtra is installed, cannot test missing-package path")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_diagnostic_dashboard(model), "gridExtra")
})

test_that("tl_diagnostic_dashboard refuses a fit with no residuals to draw", {
  skip_if_not_installed("gridExtra")
  # A tree reached rstandard() and failed with "no applicable method for
  # 'rstandard' applied to an object of class \"rpart\""
  tree <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")
  expect_error(
    tl_diagnostic_dashboard(tree),
    "The diagnostic dashboard is only available for linear-based models",
    fixed = TRUE
  )

  linear <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_s3_class(tl_diagnostic_dashboard(linear), "gtable")
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

# ---- diagnostics when the fit dropped an incomplete row --------------

test_that("diagnostics survive a missing predictor value", {
  # lm() drops incomplete cases, so residuals(), fitted() and every
  # influence measure came back one observation shorter than model$data.
  # Combining them failed with "arguments imply differing number of rows:
  # 60, 59", which describes nothing the caller did.
  set.seed(1)
  n <- 60
  d <- data.frame(x = stats::rnorm(n))
  d$y <- 2 * d$x + stats::rnorm(n, sd = 0.3)
  d$x[3] <- NA

  model <- suppressWarnings(tl_model(d, y ~ x, method = "linear"))

  expect_no_error(
    suppressWarnings(tl_check_assumptions(model, verbose = FALSE))
  )
  influence <- suppressWarnings(tl_influence_measures(model))
  expect_s3_class(influence, "data.frame")
  expect_equal(nrow(influence), 59L)
})

test_that("influence numbers observations by their row in the data", {
  # With 1:n the labels silently shifted: every observation after a
  # dropped row was attributed to its neighbour.
  set.seed(1)
  n <- 60
  d <- data.frame(x = stats::rnorm(n))
  d$y <- 2 * d$x + stats::rnorm(n, sd = 0.3)
  d$x[3] <- NA

  model <- suppressWarnings(tl_model(d, y ~ x, method = "linear"))
  influence <- suppressWarnings(tl_influence_measures(model))

  # Row 3 was dropped, so it must be absent -- and the numbering must
  # still reach 60 rather than stopping at 59
  expect_false(3 %in% influence$observation)
  expect_equal(max(influence$observation), 60L)
  expect_equal(head(influence$observation, 4), c(1L, 2L, 4L, 5L))
})

test_that("complete data is unaffected by the alignment", {
  set.seed(1)
  n <- 40
  d <- data.frame(x = stats::rnorm(n))
  d$y <- 2 * d$x + stats::rnorm(n, sd = 0.3)

  influence <- tl_influence_measures(tl_model(d, y ~ x, method = "linear"))
  expect_equal(nrow(influence), n)
  expect_equal(influence$observation, seq_len(n))
})
