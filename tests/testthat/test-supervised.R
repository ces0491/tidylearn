test_that("linear regression models work", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_s3_class(model, "tidylearn_linear")
  expect_false(model$spec$is_classification)

  # Predictions should be numeric
  preds <- predict(model)
  expect_type(preds$.pred, "double")
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("logistic regression models work for classification", {
  model <- tl_model(iris, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_logistic")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("tree models work for classification", {
  skip_if_not_installed("rpart")

  model <- tl_model(iris, Species ~ Sepal.Length + Sepal.Width, method = "tree")

  expect_s3_class(model, "tidylearn_tree")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("tree models work for regression", {
  skip_if_not_installed("rpart")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")

  expect_s3_class(model, "tidylearn_tree")
  expect_false(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_type(preds$.pred, "double")
})

test_that("random forest models work for classification", {
  skip_if_not_installed("randomForest")

  model <- tl_model(iris, Species ~ ., method = "forest")

  expect_s3_class(model, "tidylearn_forest")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("random forest models work for regression", {
  skip_if_not_installed("randomForest")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "forest")

  expect_s3_class(model, "tidylearn_forest")
  expect_false(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_type(preds$.pred, "double")
})

test_that("ridge regression works", {
  skip_if_not_installed("glmnet")

  model <- tl_model(mtcars, mpg ~ ., method = "ridge")

  expect_s3_class(model, "tidylearn_ridge")

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("lasso regression works", {
  skip_if_not_installed("glmnet")

  model <- tl_model(mtcars, mpg ~ ., method = "lasso")

  expect_s3_class(model, "tidylearn_lasso")

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("elastic net works", {
  skip_if_not_installed("glmnet")

  model <- tl_model(mtcars, mpg ~ ., method = "elastic_net", alpha = 0.5)

  expect_s3_class(model, "tidylearn_elastic_net")

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("polynomial regression works", {
  model <- tl_model(mtcars, mpg ~ wt, method = "polynomial", degree = 2)

  expect_s3_class(model, "tidylearn_polynomial")
  expect_false(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_type(preds$.pred, "double")
})

test_that("supervised models handle new data correctly", {
  # Split data
  split <- tl_split(iris, prop = 0.7, seed = 123)

  # Train on training set
  model <- tl_model(split$train, Species ~ ., method = "logistic")

  # Predict on test set
  preds <- predict(model, new_data = split$test)

  expect_equal(nrow(preds), nrow(split$test))
})

test_that("supervised models work with formula variations", {
  # Formula with interaction
  model1 <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")
  expect_s3_class(model1, "tidylearn_linear")

  # Formula with all variables
  model2 <- tl_model(iris, Species ~ ., method = "logistic")
  expect_s3_class(model2, "tidylearn_logistic")

  # Formula with subset of variables
  model3 <- tl_model(iris, Species ~ Sepal.Length + Petal.Length, method = "logistic")
  expect_s3_class(model3, "tidylearn_logistic")
})
