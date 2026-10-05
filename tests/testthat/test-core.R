test_that("tl_model creates supervised models correctly", {
  # Test with classification. Logistic is binary only, so this needs a
  # two-level response -- three levels is an error at fit time.
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model <- tl_model(binary_iris, Species ~ ., method = "logistic")

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
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model_factor <- tl_model(binary_iris, Species ~ ., method = "logistic")
  expect_true(model_factor$spec$is_classification)

  # Regression with numeric
  model_numeric <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_false(model_numeric$spec$is_classification)
})

test_that("predict.tidylearn_model works for supervised models", {
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model <- tl_model(binary_iris, Species ~ ., method = "logistic")

  # Predict on training data
  pred_train <- predict(model)
  expect_s3_class(pred_train, "tbl_df")
  expect_equal(nrow(pred_train), nrow(binary_iris))

  # Predict on new data
  pred_new <- predict(model, new_data = binary_iris[1:10, ])
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

test_that("tl_align_classes() reads observed classes against the model's", {
  # Declared but unused levels drop out, and the model's order wins
  observed <- factor(c("virginica", "versicolor"),
                     levels = c("virginica", "setosa", "versicolor"))
  aligned <- tidylearn:::tl_align_classes(observed, c("versicolor", "virginica"))
  expect_identical(levels(aligned$actuals), c("versicolor", "virginica"))
  expect_identical(as.character(aligned$actuals), c("virginica", "versicolor"))
  expect_identical(aligned$keep, c(TRUE, TRUE))

  # A 0/1 numeric response reads against the character levels the spec holds
  aligned <- tidylearn:::tl_align_classes(c(0, 1, 1), c("0", "1"))
  expect_identical(as.character(aligned$actuals), c("0", "1", "1"))

  # Missing values are not scored, without a warning of their own
  expect_no_warning(
    aligned <- tidylearn:::tl_align_classes(c("a", NA), c("a", "b"))
  )
  expect_identical(aligned$keep, c(TRUE, FALSE))

  # A class the model never saw is left out, and the warning names it
  expect_warning(
    aligned <- tidylearn:::tl_align_classes(c("a", "c", "c"), c("a", "b")),
    "2 row\\(s\\) belong to a class the model was not trained on \\(c\\)"
  )
  expect_identical(aligned$keep, c(TRUE, FALSE, FALSE))
})

test_that("magrittr's %>% stays exported for existing user code", {
  # The package itself pipes with |>. The re-export is kept so code that
  # used %>% after library(tidylearn) alone does not break.
  expect_true("%>%" %in% getNamespaceExports("tidylearn"))
  expect_identical(tidylearn::`%>%`, magrittr::`%>%`)

  # nolint start: pipe_consistency_linter.
  pred <- tl_model(mtcars, mpg ~ wt, method = "linear") %>%
    predict(new_data = mtcars[1:3, ])
  # nolint end
  expect_equal(nrow(pred), 3)
})
