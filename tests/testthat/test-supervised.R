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
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model <- tl_model(binary_iris, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_logistic")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(binary_iris))
})

test_that("logistic regression refuses a response it cannot model", {
  # glm(binomial) takes a three-level factor without complaint and fits
  # the first level against the rest. The failure used to surface at
  # predict() or tl_evaluate(), three calls after the mistake.
  expect_error(
    tl_model(iris, Species ~ ., method = "logistic"),
    "binary only.*3 levels"
  )

  # The message has to name a way forward, not just refuse
  err <- tryCatch(
    tl_model(iris, Species ~ ., method = "logistic"),
    error = function(e) conditionMessage(e)
  )
  expect_match(err, "forest")
  expect_match(err, "xgboost")

  # One level is a different problem and says so
  one_class <- droplevels(subset(iris, Species == "setosa"))
  expect_error(
    tl_model(one_class, Species ~ ., method = "logistic"),
    "only one"
  )
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
  model <- tl_model(split$train, Species ~ ., method = "forest")

  # Predict on test set
  preds <- predict(model, new_data = split$test)

  expect_equal(nrow(preds), nrow(split$test))
})

test_that("supervised models work with formula variations", {
  # Formula with interaction
  model1 <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")
  expect_s3_class(model1, "tidylearn_linear")

  # Formula with all variables
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model2 <- tl_model(binary_iris, Species ~ ., method = "logistic")
  expect_s3_class(model2, "tidylearn_logistic")

  # Formula with subset of variables
  model3 <- tl_model(binary_iris, Species ~ Sepal.Length + Petal.Length,
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

# ---- deep learning: the learning rate has to reach the optimizer -----

test_that("tl_fit_deep(learning_rate=) sets the optimizer's learning rate", {
  skip_on_cran()
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  usable <- tryCatch({
    keras::keras_model_sequential()
    TRUE
  }, error = function(e) FALSE)
  skip_if_not(usable, "No TensorFlow backend available")

  # tl_tune_deep() passed optimizer = optimizer_adam(learning_rate = lr)
  # into tl_fit_deep(), which has no such formal, so it landed in ... and
  # went to keras::fit(). The model is already compiled by then, and
  # compile() is what sets the optimizer -- so the whole learning_rates
  # grid searched over a value that never changed anything.
  set.seed(1)
  d <- data.frame(x1 = stats::rnorm(60), x2 = stats::rnorm(60))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(60, sd = 0.3)

  rate_of <- function(lr) {
    fit <- suppressWarnings(suppressMessages(
      tl_fit_deep(d, y ~ x1 + x2, is_classification = FALSE,
                  hidden_layers = c(4), epochs = 1, verbose = 0,
                  learning_rate = lr)
    ))
    as.numeric(keras::k_get_value(fit$model$optimizer$learning_rate))
  }

  expect_equal(rate_of(0.5), 0.5, tolerance = 1e-6)
  expect_equal(rate_of(0.01), 0.01, tolerance = 1e-6)

  # Left alone, keras keeps its own default rather than being forced
  expect_true(is.finite(rate_of(NULL)))
})
