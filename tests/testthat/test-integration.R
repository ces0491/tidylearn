test_that("tl_reduce_dimensions works with PCA", {
  result <- tl_reduce_dimensions(iris, response = "Species",
                                 method = "pca", n_components = 3)

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
                             supervised_method = "forest")

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
                                 supervised_method = "forest")

  expect_s3_class(model_flag, "tidylearn_anomaly_aware")
  expect_true("anomaly_info" %in% names(model_flag))
  expect_equal(model_flag$anomaly_info$action, "flag")

  # Remove anomalies
  model_remove <- tl_anomaly_aware(iris, Species ~ .,
                                   response = "Species",
                                   anomaly_method = "dbscan",
                                   action = "remove",
                                   supervised_method = "forest")

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
    tl_add_cluster_features(iris, response = "InvalidColumn",
                            method = "kmeans", k = 3),
    "Response variable.*not found"
  )
})

test_that("reduced data can be used for supervised learning", {
  # Reduce dimensions
  reduced <- tl_reduce_dimensions(iris, response = "Species",
                                  method = "pca", n_components = 3)

  # Train model on reduced data. Species has three levels, so this needs
  # a method that handles more than two classes.
  model <- tl_model(reduced$data, Species ~ ., method = "forest")

  expect_s3_class(model, "tidylearn_forest")

  # Can predict
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("cluster features improve model", {
  # This is more of an integration test to ensure the workflow works
  data_clustered <- tl_add_cluster_features(iris,
                                            response = "Species",
                                            method = "kmeans", k = 3)

  # Train model with cluster features
  model <- tl_model(data_clustered, Species ~ ., method = "forest")

  expect_s3_class(model, "tidylearn_forest")

  # Can predict
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(data_clustered))
})

test_that("tl_semisupervised refuses a response it cannot vote on", {
  expect_error(
    tl_semisupervised(mtcars, mpg ~ ., labeled_indices = 1:10),
    "categorical response"
  )

  # A character response is categorical and is still accepted
  chr <- transform(iris, Species = as.character(Species))
  set.seed(123)
  model <- tl_semisupervised(chr, Species ~ .,
                             labeled_indices = c(1:5, 51:55, 101:105))
  expect_s3_class(model, "tidylearn_semisupervised")
  expect_equal(model$semisupervised_info$n_unlabelled_dropped, 0L)
})

test_that("rows whose cluster holds no label are left out, and counted", {
  # Six labels from two classes -> k = 2, and on iris one of those two
  # clusters holds none of them. Its rows became NA and disappeared at
  # fit time without a word.
  idx <- c(which(iris$Species == "versicolor")[1:3],
           which(iris$Species == "virginica")[1:3])
  set.seed(1)
  expect_warning(
    model <- tl_semisupervised(iris, Species ~ ., labeled_indices = idx),
    "no labelled observation"
  )

  dropped <- model$semisupervised_info$n_unlabelled_dropped
  expect_gt(dropped, 0)
  expect_false(anyNA(model$data$Species))
  expect_equal(nrow(model$data) + dropped, nrow(iris))
})

test_that("downweight reaches the fit, or is refused", {
  skip_if_not_installed("dbscan")

  # rpart.control() used to swallow the weights, so this was the
  # unweighted tree exactly
  unweighted <- tl_model(iris, Species ~ ., method = "tree")
  tree <- tl_anomaly_aware(iris, Species ~ ., response = "Species",
                           action = "downweight")
  expect_gt(tree$anomaly_info$n_anomalies, 0)
  expect_false(isTRUE(all.equal(tree$fit$frame, unweighted$fit$frame)))
  expect_identical(tree$fit$call$weights, as.name("weights"))

  # and lm() errored on weights arriving through ...
  d <- mtcars[, c("mpg", "wt", "hp")]
  lin <- tl_anomaly_aware(d, mpg ~ wt, response = "mpg",
                          action = "downweight",
                          supervised_method = "linear", eps = 15, minPts = 3)
  expect_gt(lin$anomaly_info$n_anomalies, 0)
  w <- ifelse(lin$anomaly_info$is_anomaly, 0.1, 1)
  expect_equal(unname(coef(lin$fit)),
               unname(coef(lm(mpg ~ wt, data = d, weights = w))))

  expect_error(
    tl_anomaly_aware(iris, Species ~ ., response = "Species",
                     action = "downweight", supervised_method = "svm"),
    "case weights"
  )
})

test_that("downweight accepts every method that applies case weights", {
  skip_if_not_installed("dbscan")
  d <- mtcars[, c("mpg", "wt", "hp")]
  for (method in c("ridge", "lasso", "elastic_net", "forest")) {
    model <- tl_anomaly_aware(d, mpg ~ wt + hp, response = "mpg",
                              action = "downweight",
                              supervised_method = method,
                              eps = 15, minPts = 3)
    expect_s3_class(model, "tidylearn_anomaly_aware")
  }
  for (method in c("svm", "boost")) {
    expect_error(
      tl_anomaly_aware(d, mpg ~ wt + hp, response = "mpg",
                       action = "downweight", supervised_method = method,
                       eps = 15, minPts = 3),
      "case weights"
    )
  }
})

test_that("downweighting a logistic fit raises no binomial weight warning", {
  skip_if_not_installed("dbscan")
  d <- droplevels(iris[iris$Species != "setosa", ])
  expect_no_warning(
    tl_anomaly_aware(d, Species ~ ., response = "Species",
                     action = "downweight", supervised_method = "logistic")
  )
})

test_that("tl_semisupervised takes a logical selector and a string formula", {
  selector <- seq_len(nrow(iris)) %in% c(1:5, 51:55, 101:105)
  set.seed(123)
  by_position <- tl_semisupervised(iris, Species ~ .,
                                   labeled_indices = which(selector))
  set.seed(123)
  by_logical <- tl_semisupervised(iris, "Species ~ .",
                                  labeled_indices = selector)
  expect_equal(by_logical$semisupervised_info$labeled_indices,
               which(selector))
  expect_equal(by_logical$semisupervised_info$n_unlabelled_dropped,
               by_position$semisupervised_info$n_unlabelled_dropped)
})

test_that("hclust takes k in the integration helpers", {
  out <- tl_add_cluster_features(iris, response = "Species",
                                 method = "hclust", k = 4)
  expect_equal(nlevels(out$cluster_hclust), 4)

  model <- tl_semisupervised(iris, Species ~ .,
                             labeled_indices = c(1:5, 51:55, 101:105),
                             cluster_method = "hclust")
  expect_s3_class(model, "tidylearn_semisupervised")
})

test_that("flag adds the indicator without rewriting the formula", {
  skip_if_not_installed("dbscan")
  model <- tl_anomaly_aware(iris, Species ~ . - Sepal.Width,
                            response = "Species", action = "flag")
  labels <- attr(terms(model$spec$formula, data = model$data), "term.labels")
  expect_false("Sepal.Width" %in% labels)
  expect_true("is_anomalyTRUE" %in% colnames(model.matrix(
    model$spec$formula, model$data
  )) || "is_anomaly" %in% labels)

  poly_model <- tl_anomaly_aware(mtcars, mpg ~ poly(wt, 2),
                                 response = "mpg", action = "flag",
                                 supervised_method = "linear")
  expect_match(deparse(poly_model$spec$formula), "poly(wt, 2)", fixed = TRUE)
})

test_that("stratified models predict on data without the response", {
  models <- tl_stratified_models(mtcars, mpg ~ ., k = 2,
                                 supervised_method = "linear")
  preds <- predict(models, new_data = mtcars[1:5, -1])
  expect_equal(nrow(preds), 5)
  expect_equal(preds$.pred, predict(models, new_data = mtcars[1:5, ])$.pred)
})

test_that("the integration helpers take a string formula", {
  models <- tl_stratified_models(mtcars, "mpg ~ .", k = 2,
                                 supervised_method = "linear")
  expect_s3_class(models, "tidylearn_stratified")
  skip_if_not_installed("dbscan")
  model <- tl_anomaly_aware(iris, "Species ~ .", response = "Species")
  expect_s3_class(model, "tidylearn_anomaly_aware")
})

test_that("stratified probability predictions keep their columns", {
  set.seed(1)
  d <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  d$y <- factor(ifelse(d$x1 + rnorm(100) > 0, "a", "b"))
  models <- tl_stratified_models(d, y ~ x1 + x2, k = 2)
  expect_no_warning(probs <- predict(models, type = "prob"))
  expect_true(all(c("a", "b", ".cluster") %in% names(probs)))
  expect_equal(nrow(probs), nrow(d))
  expect_equal(probs$a + probs$b, rep(1, nrow(d)))
})

test_that("tl_anomaly_aware names a bad action or method", {
  skip_if_not_installed("dbscan")
  expect_error(
    tl_anomaly_aware(iris, Species ~ ., response = "Species",
                     action = "nope"),
    "'action'"
  )
  expect_error(
    tl_anomaly_aware(iris, Species ~ ., response = "Species",
                     anomaly_method = "isolation_forest"),
    "'anomaly_method'"
  )
})

test_that("stratified predictions carry the training classes in order", {
  # Two well-separated groups, each holding only some of the classes, so
  # each cluster's model knows a different subset of A, B, C
  set.seed(1)
  n <- 60
  d <- data.frame(
    x1 = c(rnorm(n, 0), rnorm(n, 10)),
    x2 = c(rnorm(n, 0), rnorm(n, 10))
  )
  d$y <- factor(c(sample(c("A", "B"), n, TRUE), sample(c("B", "C"), n, TRUE)))
  models <- tl_stratified_models(d, y ~ x1 + x2, k = 2)

  forward <- predict(models, d)
  reversed <- predict(models, d[rev(seq_len(nrow(d))), ])
  expect_identical(levels(forward$.pred), c("A", "B", "C"))
  expect_identical(levels(reversed$.pred), c("A", "B", "C"))
  expect_identical(levels(predict(models, d[1, ])$.pred), c("A", "B", "C"))

  probs <- predict(models, d, type = "prob")
  expect_identical(names(probs)[1:3], c("A", "B", "C"))
  # a class a cluster never saw has probability 0, not NA
  expect_false(anyNA(probs[c("A", "B", "C")]))
  expect_equal(rowSums(probs[c("A", "B", "C")]), rep(1, nrow(d)))
  reversed_probs <- predict(models, d[rev(seq_len(nrow(d))), ], type = "prob")
  expect_identical(names(reversed_probs)[1:3], c("A", "B", "C"))
})
