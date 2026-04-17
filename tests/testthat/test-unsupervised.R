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
  expect_true(
    all(grepl("PC", names(transformed)) |
          names(transformed) == ".obs_id")
  )
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
  large_data <- iris[rep(seq_len(nrow(iris)), 10), 1:4]

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
  large_data <- iris[rep(seq_len(nrow(iris)), 5), 1:4]
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

test_that("tidy_gower returns a dist object with correct metadata", {
  d <- tidy_gower(iris[1:10, 1:4])

  expect_s3_class(d, "dist")
  expect_equal(attr(d, "Size"), 10L)
  expect_equal(attr(d, "method"), "gower")
})

test_that("tidy_gower distances are non-negative and symmetric", {
  d <- tidy_gower(iris[1:20, 1:4])
  m <- as.matrix(d)

  expect_true(all(m >= 0))
  expect_equal(m, t(m))               # symmetric
  expect_true(all(diag(m) == 0))       # self-distance is 0
})

test_that("tidy_gower gives distance 0 for identical rows", {
  dup <- iris[c(1, 1, 2), 1:4]
  d <- as.matrix(tidy_gower(dup))

  expect_equal(d[1, 2], 0)   # row 1 vs its duplicate
  expect_gt(d[1, 3], 0)      # genuinely different rows are non-zero
})

test_that("tidy_gower numeric-only distances are correct", {
  # Two observations: x in {0, 1}.  range = 1, so d = |0-1|/1 = 1.
  df <- data.frame(x = c(0, 1))
  d  <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 1)

  # Three observations: x in {0, 0.5, 1}.
  # d(1,2) = 0.5, d(1,3) = 1, d(2,3) = 0.5
  df3 <- data.frame(x = c(0, 0.5, 1))
  d3  <- as.matrix(tidy_gower(df3))

  expect_equal(d3[1, 2], 0.5)
  expect_equal(d3[1, 3], 1.0)
  expect_equal(d3[2, 3], 0.5)
})

test_that("tidy_gower categorical distances are correct", {
  # Same category → 0; different category → 1
  df <- data.frame(color = factor(c("red", "red", "blue")))
  d  <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 0)   # red vs red
  expect_equal(d[1, 3], 1)   # red vs blue
  expect_equal(d[2, 3], 1)   # red vs blue
})

test_that("tidy_gower ordered distances are correct", {
  # Ranks: low=1, medium=2, high=3; range = 2
  # d(low, medium) = 1/2, d(low, high) = 2/2 = 1, d(medium, high) = 1/2
  df <- data.frame(
    rating = ordered(c("low", "medium", "high"),
                     levels = c("low", "medium", "high"))
  )
  d <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 0.5)
  expect_equal(d[1, 3], 1.0)
  expect_equal(d[2, 3], 0.5)
})

test_that("tidy_gower handles mixed numeric and categorical columns", {
  # x in {0, 1, 0.5}, color in {red, red, blue}; equal weights → average of two d_k
  # d(1,2): d_x = 1, d_color = 0  → 0.5
  # d(1,3): d_x = 0.5, d_color = 1  → 0.75
  # d(2,3): d_x = 0.5, d_color = 1  → 0.75
  df <- data.frame(
    x     = c(0, 1, 0.5),
    color = factor(c("red", "red", "blue"))
  )
  d <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 0.50)
  expect_equal(d[1, 3], 0.75)
  expect_equal(d[2, 3], 0.75)
})

test_that("tidy_gower skips NA values when computing distances", {
  # x: {0, 1}, y: {1, NA}.  For pair (1,2) only x is valid.
  # d = |0-1| / (1-0) = 1
  df <- data.frame(x = c(0, 1), y = c(1, NA_real_))
  d  <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 1)   # y skipped; only x contributes
})

test_that("tidy_gower respects custom weights", {
  # Upweight x so the numeric dimension dominates
  df <- data.frame(
    x     = c(0, 1),
    color = factor(c("red", "blue"))
  )
  d_equal    <- as.matrix(tidy_gower(df))
  d_weighted <- as.matrix(tidy_gower(df, weights = c(x = 10, color = 1)))

  # Need ≥3 rows so the column range is wider than the pair (1,2) difference.
  # With only 2 rows, range == difference, so d_x = 1 always — weights can't
  # move a result that is already at its maximum.
  #
  # x: range = 1, d(1,2)_x = |0 - 0.1| / 1 = 0.1
  # color: d(1,2)_color = 1  (red vs blue)
  # equal weights: (0.1 + 1) / 2 = 0.55
  # heavy color (1, 100): (1*0.1 + 100*1) / 101 ≈ 0.991
  df2 <- data.frame(
    x     = c(0, 0.1, 1),
    color = factor(c("red", "blue", "red"))
  )
  d_eq2 <- as.matrix(tidy_gower(df2))
  d_wt2 <- as.matrix(tidy_gower(df2, weights = c(x = 1, color = 100)))

  expect_gt(d_wt2[1, 2], d_eq2[1, 2])  # heavy color weight → larger distance
})

test_that("tidy_gower constant columns contribute zero to numerator but not denominator", {
  # A constant column (range = 0) has d_k = 0 for every pair, so it adds
  # nothing to the numerator.  However, both observations are non-NA, so the
  # column IS counted in valid_vars (the denominator).  The net effect is that
  # the overall distance is exactly halved compared to y alone.
  #
  # pair (1,2): d_x = 0, d_y = |0-1|/1 = 1  → (0 + 1) / 2 = 0.5
  # y alone:    d_y = 1/1 = 1
  df   <- data.frame(x = c(5, 5, 5), y = c(0, 1, 0.5))
  d    <- as.matrix(tidy_gower(df))
  d_y  <- as.matrix(tidy_gower(data.frame(y = c(0, 1, 0.5))))

  expect_equal(d, d_y / 2)
})


test_that("tidy_dist dispatches to tidy_gower for method = 'gower'", {
  df <- data.frame(
    x     = c(1, 2, 3),
    color = factor(c("a", "b", "a"))
  )
  d1 <- tidy_dist(df, method = "gower")
  d2 <- tidy_gower(df)

  expect_equal(as.vector(d1), as.vector(d2))
  expect_equal(attr(d1, "method"), "gower")
})

test_that("unsupervised methods work with formula", {
  # PCA with formula
  model <- tl_model(
    iris,
    ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    method = "pca"
  )

  expect_s3_class(model, "tidylearn_pca")

  # Clustering with formula
  model2 <- tl_model(iris, ~ Sepal.Length + Sepal.Width,
                     method = "kmeans", k = 3)
  expect_s3_class(model2, "tidylearn_kmeans")
})
