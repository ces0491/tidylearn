test_that("tl_table dispatches correctly for supervised models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table dispatches correctly for unsupervised models", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "pca")
  tbl <- tl_table(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table rejects non-tidylearn objects", {
  skip_if_not_installed("gt")

  expect_error(tl_table(lm(mpg ~ wt, data = mtcars)), "tidylearn_model")
})

test_that("tl_table_metrics returns gt_tbl", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table_metrics(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_coefficients works for linear models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table_coefficients(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_coefficients works for regularised models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
  tbl <- tl_table_coefficients(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_coefficients errors for unsupported methods", {
  skip_if_not_installed("gt")

  model <- tl_model(iris, Species ~ ., method = "forest")
  expect_error(tl_table_coefficients(model), "importance")
})

test_that("tl_table_confusion works for classification", {
  skip_if_not_installed("gt")

  model <- tl_model(iris, Species ~ ., method = "forest")
  tbl <- tl_table_confusion(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_confusion errors for regression", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_table_confusion(model), "classification")
})

test_that("tl_table_importance works for tree-based models", {
  skip_if_not_installed("gt")

  model <- tl_model(iris, Species ~ ., method = "forest")
  tbl <- tl_table_importance(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_variance works for PCA", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "pca")
  tbl <- tl_table_variance(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_variance errors for non-PCA models", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  expect_error(tl_table_variance(model), "PCA")
})

test_that("tl_table_loadings works for PCA", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "pca")
  tbl <- tl_table_loadings(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_clusters works for kmeans", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  tbl <- tl_table_clusters(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_comparison requires at least 2 models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_table_comparison(model), "at least 2")
})

test_that("tl_table_comparison works with multiple models", {
  skip_if_not_installed("gt")

  m1 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
  tbl <- tl_table_comparison(m1, m2, names = c("Linear", "Lasso"))
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table errors for unknown type", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_table(model, type = "nonexistent"), "Unknown table type")
})
