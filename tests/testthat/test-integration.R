test_that("tl_reduce_dimensions works with PCA", {
  result <- tl_reduce_dimensions(iris, response = "Species", method = "pca", n_components = 3)

  expect_type(result, "list")
  expect_true("data" %in% names(result))
  expect_true("reduction_model" %in% names(result))

  # Check transformed data has PC columns
  expect_true(any(grepl("PC", names(result$data))))

  # Response should be preserved
  expect_true("Species" %in% names(result$data))
  expect_equal(result$data$Species, iris$Species)

  # Should have requested number of components
  pc_cols <- sum(grepl("^PC\\d+$", names(result$data)))
  expect_equal(pc_cols, 3)
})

test_that("tl_reduce_dimensions works without response", {
  result <- tl_reduce_dimensions(iris[, 1:4], method = "pca", n_components = 2)

  expect_type(result, "list")
  expect_true("data" %in% names(result))

  # Should have PC columns
  pc_cols <- sum(grepl("^PC", names(result$data)))
  expect_gte(pc_cols, 2)
})

test_that("tl_add_cluster_features adds cluster columns", {
  data_with_clusters <- tl_add_cluster_features(iris, response = "Species",
                                                method = "kmeans", k = 3)

  # Should have cluster column
  expect_true(any(grepl("cluster_", names(data_with_clusters))))

  # Original columns should be preserved
  expect_true(all(names(iris) %in% names(data_with_clusters)))

  # Cluster column should be a factor
  cluster_col <- grep("cluster_", names(data_with_clusters), value = TRUE)
  expect_s3_class(data_with_clusters[[cluster_col]], "factor")
})

test_that("tl_add_cluster_features works with different clustering methods", {
  # K-means
  data_kmeans <- tl_add_cluster_features(iris, response = "Species",
                                         method = "kmeans", k = 3)
  expect_true("cluster_kmeans" %in% names(data_kmeans))

  # PAM
  skip_if_not_installed("cluster")
  data_pam <- tl_add_cluster_features(iris, response = "Species",
                                     method = "pam", k = 3)
  expect_true("cluster_pam" %in% names(data_pam))
})

test_that("tl_semisupervised performs label propagation", {
  # Use only 10% of labels
  set.seed(123)
  labeled_idx <- sample(nrow(iris), size = 15)

  model <- tl_semisupervised(iris, Species ~ .,
                            labeled_indices = labeled_idx,
                            cluster_method = "kmeans",
                            supervised_method = "logistic")

  expect_s3_class(model, "tidylearn_semisupervised")
  expect_s3_class(model, "tidylearn_supervised")

  # Should have semisupervised info
  expect_true("semisupervised_info" %in% names(model))
  expect_equal(model$semisupervised_info$labeled_indices, labeled_idx)

  # Can predict
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("tl_anomaly_aware detects and handles outliers", {
  skip_if_not_installed("dbscan")

  # Flag anomalies
  model_flag <- tl_anomaly_aware(iris, Species ~ .,
                                response = "Species",
                                anomaly_method = "dbscan",
                                action = "flag",
                                supervised_method = "logistic")

  expect_s3_class(model_flag, "tidylearn_anomaly_aware")
  expect_true("anomaly_info" %in% names(model_flag))
  expect_equal(model_flag$anomaly_info$action, "flag")

  # Remove anomalies
  model_remove <- tl_anomaly_aware(iris, Species ~ .,
                                  response = "Species",
                                  anomaly_method = "dbscan",
                                  action = "remove",
                                  supervised_method = "logistic")

  expect_s3_class(model_remove, "tidylearn_anomaly_aware")
  expect_true("anomalies_removed" %in% names(model_remove))
})

test_that("tl_stratified_models creates cluster-specific models", {
  models <- tl_stratified_models(mtcars, mpg ~ .,
                                cluster_method = "kmeans",
                                k = 3,
                                supervised_method = "linear")

  expect_s3_class(models, "tidylearn_stratified")
  expect_true("cluster_model" %in% names(models))
  expect_true("supervised_models" %in% names(models))

  # Should have one model per cluster
  expect_gte(length(models$supervised_models), 1)
  expect_lte(length(models$supervised_models), 3)
})

test_that("predict.tidylearn_stratified assigns to clusters and predicts", {
  models <- tl_stratified_models(mtcars, mpg ~ .,
                                cluster_method = "kmeans",
                                k = 2,
                                supervised_method = "linear")

  # Predict on training data
  preds <- predict(models)
  expect_equal(nrow(preds), nrow(mtcars))
  expect_true(".pred" %in% names(preds))
  expect_true(".cluster" %in% names(preds))

  # Predict on new data
  preds_new <- predict(models, new_data = mtcars[1:10, ])
  expect_equal(nrow(preds_new), 10)
})

test_that("integration functions validate inputs", {
  # Invalid response variable
  expect_error(
    tl_reduce_dimensions(iris, response = "InvalidColumn", method = "pca"),
    "Response variable.*not found"
  )

  expect_error(
    tl_add_cluster_features(iris, response = "InvalidColumn", method = "kmeans", k = 3),
    "Response variable.*not found"
  )
})

test_that("reduced data can be used for supervised learning", {
  # Reduce dimensions
  reduced <- tl_reduce_dimensions(iris, response = "Species",
                                 method = "pca", n_components = 3)

  # Train model on reduced data
  model <- tl_model(reduced$data, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_logistic")

  # Can predict
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("cluster features improve model", {
  # This is more of an integration test to ensure the workflow works
  data_clustered <- tl_add_cluster_features(iris, response = "Species",
                                           method = "kmeans", k = 3)

  # Train model with cluster features
  model <- tl_model(data_clustered, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_logistic")

  # Can predict
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(data_clustered))
})
