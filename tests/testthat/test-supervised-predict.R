make_factor_data <- function(seed = 501, n = 120) {
  set.seed(seed)
  data <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    g = factor(sample(c("a", "b", "c"), n, replace = TRUE))
  )
  data$y <- 2 * data$x1 - 1.5 * data$x2 + as.numeric(data$g) +
    stats::rnorm(n, sd = 0.4)
  data$cls <- factor(
    ifelse(data$y > stats::median(data$y), "hi", "lo")
  )
  data
}

test_that("glmnet prediction handles factor predictors", {
  data <- make_factor_data()
  model <- tl_model(data, y ~ x1 + x2 + g, method = "lasso")

  # Training used treatment contrasts with the intercept dropped; building
  # the prediction design as "~ predictors - 1" one-hot encodes the first
  # factor and yields one column too many
  preds <- predict(model, new_data = data)$.pred
  expect_length(preds, nrow(data))
  expect_gt(stats::cor(preds, data$y), 0.9)

  # Must agree with glmnet driven directly through the same design
  model_frame <- stats::model.frame(y ~ x1 + x2 + g, data = data)
  x_direct <- stats::model.matrix(
    stats::terms(model_frame), model_frame
  )[, -1, drop = FALSE]
  direct <- as.vector(stats::predict(
    model$fit, newx = x_direct,
    s = attr(model$fit, "lambda_min"), type = "response"
  ))
  expect_equal(preds, direct)
})

test_that("glmnet prediction survives a subset missing a factor level", {
  data <- make_factor_data()
  model <- tl_model(data, y ~ x1 + x2 + g, method = "lasso")

  subset_data <- data[data$g == "a", ][1:5, ]
  expect_length(predict(model, new_data = subset_data)$.pred, 5)
})

test_that("glmnet classification honours the prediction type", {
  data <- make_factor_data()
  model <- tl_model(data, cls ~ x1 + x2 + g, method = "ridge")
  levels_cls <- levels(data$cls)

  classes <- predict(model, new_data = data, type = "class")$.pred
  expect_true(is.factor(classes))
  expect_identical(levels(classes), levels_cls)
  expect_gt(mean(classes == data$cls), 0.7)

  probs <- predict(model, new_data = data, type = "prob")
  expect_setequal(names(probs), levels_cls)
  expect_true(all(abs(rowSums(probs) - 1) < 1e-10))

  # The two views must agree
  expect_equal(
    probs[[levels_cls[2]]] > 0.5,
    classes == levels_cls[2]
  )

  expect_error(
    predict(model, new_data = data, type = "nope"),
    "Invalid prediction type"
  )
})

test_that("glmnet multiclass prediction returns per-class probabilities", {
  data <- make_factor_data()
  set.seed(502)
  data$grp <- factor(sample(c("p", "q", "r"), nrow(data), replace = TRUE))

  model <- tl_model(data, grp ~ x1 + x2, method = "lasso")

  probs <- predict(model, new_data = data, type = "prob")
  expect_setequal(names(probs), c("p", "q", "r"))
  expect_true(all(abs(rowSums(probs) - 1) < 1e-8))

  expect_true(is.factor(predict(model, new_data = data, type = "class")$.pred))
})

test_that("tl_evaluate works end to end for regularized classification", {
  data <- make_factor_data()
  model <- tl_model(data, cls ~ x1 + x2 + g, method = "ridge")

  # Requires both class labels and probabilities to come back correctly
  result <- tl_evaluate(model, metrics = c("accuracy", "auc"))

  expect_setequal(result$metric, c("accuracy", "auc"))
  expect_false(any(is.na(result$value)))
})

test_that("boost fits binary classification", {
  skip_if_not_installed("gbm")

  binary_iris <- droplevels(subset(iris, Species != "virginica"))
  levels_iris <- levels(binary_iris$Species)

  # gbm's bernoulli distribution requires a numeric 0/1 response; passing
  # the factor straight through fails at fit time
  model <- tl_model(binary_iris, Species ~ ., method = "boost", n.trees = 50)
  expect_s3_class(model, "tidylearn_model")

  classes <- predict(model, new_data = binary_iris, type = "class")$.pred
  expect_gt(mean(classes == binary_iris$Species), 0.9)

  probs <- predict(model, new_data = binary_iris, type = "prob")
  expect_setequal(names(probs), levels_iris)

  # The positive class must be the second level, matching the encoding
  expect_equal(
    probs[[levels_iris[2]]] > 0.5,
    classes == levels_iris[2]
  )
})

test_that("boost regression is unaffected", {
  skip_if_not_installed("gbm")

  data <- make_factor_data()
  model <- tl_model(data, y ~ x1 + x2, method = "boost", n.trees = 50)

  preds <- predict(model, new_data = data)$.pred
  expect_type(preds, "double")
  expect_gt(stats::cor(preds, data$y), 0.8)
})
