test_that("tl_model creates supervised models correctly", {
  # Test with classification
  model <- tl_model(iris, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_model")
  expect_s3_class(model, "tidylearn_supervised")
  expect_s3_class(model, "tidylearn_logistic")
  expect_equal(model$spec$paradigm, "supervised")
  expect_equal(model$spec$method, "logistic")
  expect_true(model$spec$is_classification)
  expect_equal(model$spec$response_var, "Species")
})

test_that("tl_model creates unsupervised models correctly", {
  # Test PCA
  model <- tl_model(iris[, 1:4], method = "pca")

  expect_s3_class(model, "tidylearn_model")
  expect_s3_class(model, "tidylearn_unsupervised")
  expect_s3_class(model, "tidylearn_pca")
  expect_equal(model$spec$paradigm, "unsupervised")
  expect_equal(model$spec$method, "pca")
})

test_that("tl_model validates inputs", {
  # Invalid data type
  expect_error(
    tl_model("not_a_dataframe", method = "linear"),
    "data.*must be a data frame"
  )

  # Unknown method
  expect_error(
    tl_model(iris, Species ~ ., method = "unknown_method"),
    "Unknown method"
  )
})

test_that("tl_model determines task type correctly", {
  # Classification with factor
  model_factor <- tl_model(iris, Species ~ ., method = "logistic")
  expect_true(model_factor$spec$is_classification)

  # Regression with numeric
  model_numeric <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_false(model_numeric$spec$is_classification)
})

test_that("predict.tidylearn_model works for supervised models", {
  model <- tl_model(iris, Species ~ ., method = "logistic")

  # Predict on training data
  pred_train <- predict(model)
  expect_s3_class(pred_train, "tbl_df")
  expect_equal(nrow(pred_train), nrow(iris))

  # Predict on new data
  pred_new <- predict(model, new_data = iris[1:10, ])
  expect_equal(nrow(pred_new), 10)
})

test_that("predict.tidylearn_model works for unsupervised models", {
  model <- tl_model(iris[, 1:4], method = "pca")

  # Transform data
  transformed <- predict(model)
  expect_s3_class(transformed, "tbl_df")
})

test_that("print.tidylearn_model displays correctly", {
  model <- tl_model(iris, Species ~ ., method = "forest")

  # Should print without error
  expect_output(print(model), "tidylearn Model")
  expect_output(print(model), "Paradigm: supervised")
  expect_output(print(model), "Method: forest")
  expect_output(print(model), "Task: Classification")
})

test_that("tl_version returns package version", {
  version <- tl_version()
  expect_s3_class(version, "package_version")
})
