test_that("PCA models work", {
  model <- tl_model(iris[, 1:4], method = "pca")

  expect_s3_class(model, "tidylearn_pca")
  expect_equal(model$spec$paradigm, "unsupervised")

  # Check PCA components
  expect_true("scores" %in% names(model$fit))
  expect_true("loadings" %in% names(model$fit))
  expect_true("variance_explained" %in% names(model$fit))

  # Transform data
  transformed <- predict(model)
  expect_s3_class(transformed, "tbl_df")
  expect_true(all(grepl("PC", names(transformed)) | names(transformed) == ".obs_id"))
})

test_that("K-means clustering works", {
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  expect_s3_class(model, "tidylearn_kmeans")

  # Check cluster assignments
  expect_true("clusters" %in% names(model$fit))
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(iris))
  expect_true("cluster" %in% names(clusters))

  # Clusters should be 1 to k
  expect_true(all(clusters$cluster %in% 1:3))
})

test_that("PAM (K-medoids) clustering works", {
  skip_if_not_installed("cluster")

  model <- tl_model(iris[, 1:4], method = "pam", k = 3)

  expect_s3_class(model, "tidylearn_pam")

  # Check cluster assignments
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(iris))
  expect_true(all(clusters$cluster %in% 1:3))
})

test_that("CLARA clustering works", {
  skip_if_not_installed("cluster")

  # Create larger dataset for CLARA
  large_data <- iris[rep(1:nrow(iris), 10), 1:4]

  model <- tl_model(large_data, method = "clara", k = 3, samples = 5)

  expect_s3_class(model, "tidylearn_clara")

  # Check cluster assignments
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(large_data))
})

test_that("Hierarchical clustering works", {
  model <- tl_model(iris[, 1:4], method = "hclust")

  expect_s3_class(model, "tidylearn_hclust")

  # Check dendrogram exists
  expect_true("model" %in% names(model$fit))
  expect_s3_class(model$fit$model, "hclust")
})

test_that("DBSCAN clustering works", {
  skip_if_not_installed("dbscan")

  model <- tl_model(iris[, 1:4], method = "dbscan", eps = 0.5, minPts = 5)

  expect_s3_class(model, "tidylearn_dbscan")

  # Check cluster assignments (including noise points as 0)
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(iris))
  expect_true("cluster" %in% names(clusters))
})

test_that("MDS works", {
  model <- tl_model(iris[, 1:4], method = "mds", k = 2)

  expect_s3_class(model, "tidylearn_mds")

  # Check MDS points
  expect_true("points" %in% names(model$fit))
  points <- model$fit$points
  expect_equal(nrow(points), nrow(iris))
})

test_that("clustering models predict on new data", {
  # Train clustering model
  model <- tl_model(iris[1:100, 1:4], method = "kmeans", k = 3)

  # Predict on new data
  new_data <- iris[101:150, 1:4]
  predictions <- predict(model, new_data = new_data)

  expect_equal(nrow(predictions), nrow(new_data))
  expect_true("cluster" %in% names(predictions))
})

test_that("PCA retains specified number of components", {
  model <- tl_model(iris[, 1:4], method = "pca")

  # Default should retain all components
  transformed <- predict(model)
  pc_cols <- sum(grepl("^PC", names(transformed)))
  expect_equal(pc_cols, 4)
})

test_that("unsupervised methods handle different data sizes", {
  # Small dataset
  small_data <- iris[1:20, 1:4]
  model_small <- tl_model(small_data, method = "kmeans", k = 2)
  expect_s3_class(model_small, "tidylearn_kmeans")

  # Large dataset
  large_data <- iris[rep(1:nrow(iris), 5), 1:4]
  model_large <- tl_model(large_data, method = "kmeans", k = 3)
  expect_s3_class(model_large, "tidylearn_kmeans")
})

test_that("clustering validates k parameter", {
  # k should be reasonable - expect an error for invalid k
  expect_error(
    tl_model(iris[, 1:4], method = "kmeans", k = nrow(iris) + 1)
  )

  # Valid k should work
  expect_s3_class(
    tl_model(iris[, 1:4], method = "kmeans", k = 3),
    "tidylearn_kmeans"
  )
})

test_that("unsupervised methods work with formula", {
  # PCA with formula
  model <- tl_model(iris, ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, method = "pca")

  expect_s3_class(model, "tidylearn_pca")

  # Clustering with formula
  model2 <- tl_model(iris, ~ Sepal.Length + Sepal.Width, method = "kmeans", k = 3)
  expect_s3_class(model2, "tidylearn_kmeans")
})
