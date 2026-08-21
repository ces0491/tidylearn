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
  model3 <- tl_model(iris, Species ~ Sepal.Length + Petal.Length,
                     method = "logistic")
  expect_s3_class(model3, "tidylearn_logistic")
})

# Neural networks. Nothing in the suite fitted one before, which is how a
# two-class fit came to be broken for as long as it was.

test_that("neural networks fit two-class problems", {
  skip_if_not_installed("nnet")

  # nnet.formula() supplies entropy = TRUE itself for a two-level factor.
  # Naming it again here reached nnet.default() twice and stopped with
  # "formal argument 'entropy' matched by multiple actual arguments" -- for
  # every binary classification, on any data.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  model <- tl_model(iris_binary, Species ~ ., method = "nn", trace = FALSE)

  expect_s3_class(model, "tidylearn_nn")
  expect_true(model$spec$is_classification)
  expect_equal(nrow(predict(model)), nrow(iris_binary))
})

test_that("the error criterion follows the number of classes", {
  skip_if_not_installed("nnet")

  # Left to nnet: cross-entropy on two levels, softmax on three or more.
  # Multiclass tolerated the duplicate argument only because
  # nnet.default() sets entropy <- FALSE whenever softmax is on, so this
  # also pins that the multiclass fit is unchanged by the repair.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  binary <- tl_model(iris_binary, Species ~ ., method = "nn", trace = FALSE)
  expect_true(binary$fit$entropy)
  expect_false(binary$fit$softmax)

  set.seed(1)
  multiclass <- tl_model(iris, Species ~ ., method = "nn", trace = FALSE)
  expect_false(multiclass$fit$entropy)
  expect_true(multiclass$fit$softmax)
})

test_that("two-class neural network prediction covers every type", {
  skip_if_not_installed("nnet")

  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)
  levels_expected <- levels(iris_binary$Species)

  set.seed(1)
  model <- tl_model(iris_binary, Species ~ ., method = "nn", trace = FALSE)

  labels <- predict(model, type = "class")
  expect_true(all(labels$.pred %in% levels_expected))

  probs <- predict(model, type = "prob")
  expect_named(probs, levels_expected)
  expect_equal(rowSums(probs), rep(1, nrow(iris_binary)), tolerance = 1e-6)

  expect_equal(nrow(predict(model, new_data = iris_binary[1, ])), 1)

  metrics <- tl_evaluate(model, metrics = "accuracy")
  expect_gt(metrics$value[metrics$metric == "accuracy"], 0.8)
})

test_that("neural network arguments still reach nnet", {
  skip_if_not_installed("nnet")

  # The repair removed an argument from the call; the pass-through that
  # shares that `...` has to keep working.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  model <- tl_model(
    iris_binary, Species ~ .,
    method = "nn", size = 3, decay = 0.1, maxit = 50, trace = FALSE
  )

  expect_equal(model$fit$n[2], 3)
  expect_equal(model$fit$decay, 0.1)
})

test_that("neural network tuning runs for every response type", {
  skip_if_not_installed("nnet")

  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  tuned <- tl_tune_nn(
    iris_binary, Species ~ .,
    is_classification = TRUE,
    sizes = c(2, 3), decays = c(0, 0.1), folds = 2
  )

  expect_true(tuned$best_size %in% c(2, 3))
  expect_true(tuned$best_decay %in% c(0, 0.1))
  expect_equal(nrow(tuned$tuning_results), 4)
})
