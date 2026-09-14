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

test_that("two models of one method get separate comparison columns", {
  skip_if_not_installed("gt")

  # Both default to "linear (reg)", which pivoted into list cells
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "linear")
  data <- tl_table_comparison(m1, m2)[["_data"]]

  expect_equal(ncol(data), 3)
  expect_true(all(vapply(data[-1], is.numeric, logical(1))))

  expect_error(tl_table_comparison(m1, m2, names = "a"), "must match")
  expect_error(tl_table_comparison(m1, m2, names = c("a", "a")), "unique")
})

test_that("comparison names must not be missing", {
  skip_if_not_installed("gt")
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ hp, method = "linear")
  expect_error(tl_table_comparison(m1, m2, names = c("a", NA)),
               "no missing values")
})

test_that("the confusion table says when rows are missing the response", {
  skip_if_not_installed("gt")
  model <- tl_model(iris, Species ~ ., method = "forest", ntree = 50)
  d <- iris
  d$Species[1:5] <- NA
  expect_warning(tl_table_confusion(model, new_data = d), "5 row")
})

test_that("forest importance works without permutation importance", {
  model <- tl_model(iris, Species ~ ., method = "forest", ntree = 50,
                    importance = FALSE)
  imp <- tl_extract_importance(model)
  expect_setequal(imp$feature, names(iris)[1:4])
  expect_equal(max(imp$importance), 100)
  reg <- tl_model(mtcars, mpg ~ ., method = "forest", ntree = 50,
                  importance = FALSE)
  expect_equal(max(tl_extract_importance(reg)$importance), 100)
})

test_that("the dbscan cluster table does not count noise as a cluster", {
  skip_if_not_installed("gt")
  model <- tl_model(iris[, 1:4], method = "dbscan", eps = 0.4, minPts = 5)
  subtitle <- tl_table_clusters(model)[["_heading"]]$subtitle
  expect_match(subtitle, paste(model$fit$n_clusters, "clusters"))
  expect_match(subtitle, "noise")
})

test_that("cluster tables average around a missing value", {
  skip_if_not_installed("gt")
  with_na <- iris[, 1:4]
  with_na[5, 1] <- NA
  tbl <- suppressWarnings(tl_table_clusters(tl_model(with_na,
                                                     method = "hclust")))
  expect_false(anyNA(tbl[["_data"]]$Sepal.Length))
})

test_that("a long formula gives one source note", {
  model <- tl_model(mtcars, mpg ~ cyl + disp + hp + drat + wt + qsec + vs +
                      am + gear + carb + I(wt^2) + I(hp^2), method = "linear")
  expect_length(tl_model_info(model), 1)
})

test_that("tl_table_importance supports xgboost, as documented", {
  skip_if_not_installed("xgboost")
  skip_if_not_installed("gt")
  model <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "xgboost",
                    nrounds = 10)
  data <- tl_table_importance(model)[["_data"]]
  expect_true(all(data$feature %in% c("wt", "hp", "qsec")))
  expect_equal(max(data$importance), 100)
})
