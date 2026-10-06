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

test_that("tl_reduce_dimensions names an unknown method or component count", {
  # method = "kmeans" failed with "object 'transformed' not found", and
  # more components than the data has with "Elements PC5 and PC6 don't
  # exist"
  expect_error(
    tl_reduce_dimensions(iris, "Species", method = "kmeans"),
    "'method' must be \"pca\" or \"mds\"; got \"kmeans\""
  )
  expect_error(
    tl_reduce_dimensions(iris, "Species", n_components = 6),
    "'n_components' is 6, but the PCA has 4 components"
  )
  expect_error(
    tl_reduce_dimensions(iris, "Species", method = "mds", n_components = 3),
    "'n_components' is 3, but the MDS has 2 dimensions"
  )
  expect_error(
    tl_reduce_dimensions(iris, "Species", n_components = 0),
    "'n_components' must be a single whole number of at least 1"
  )

  # Every component the data has is still a valid request
  all_four <- tl_reduce_dimensions(iris, "Species", n_components = 4)
  expect_setequal(names(all_four$data), c(paste0("PC", 1:4), "Species"))
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

test_that("tl_semisupervised keeps the response's level order", {
  # as.factor() on the pseudo-labels sorted them alphabetically, which put
  # versicolor first and made virginica the positive class: .pred for the
  # first versicolor row was 0.001
  binary_iris <- iris[iris$Species != "setosa", ]
  binary_iris$Species <- factor(as.character(binary_iris$Species),
                                levels = c("virginica", "versicolor"))
  set.seed(1)
  model <- suppressWarnings(tl_semisupervised(
    binary_iris, Species ~ .,
    labeled_indices = c(1:5, 51:55), supervised_method = "logistic"
  ))
  expect_identical(model$spec$response_levels, c("virginica", "versicolor"))
  expect_identical(levels(model$data$Species), c("virginica", "versicolor"))

  # glm() on the same pseudo-labels, in the declared order, models the
  # probability of versicolor
  pseudo <- model$data
  pseudo$Species <- factor(as.character(pseudo$Species),
                           levels = c("virginica", "versicolor"))
  direct <- suppressWarnings(
    glm(Species ~ ., data = pseudo, family = binomial)
  )
  expect_equal(
    predict(model, binary_iris, type = "response")$.pred,
    predict(direct, binary_iris, type = "response")
  )
})

test_that("a cluster whose labelled rows have no label propagates nothing", {
  # The labelled virginica rows carry NA. Their cluster used to take the
  # first level, setosa, for all its rows -- a label nothing in the cluster
  # had -- and k counted NA as a third class
  ir <- iris
  ir$Species[101:105] <- NA
  labelled <- c(1:5, 51:55, 101:105)
  set.seed(2)
  model <- suppressWarnings(
    tl_semisupervised(ir, Species ~ ., labeled_indices = labelled)
  )
  info <- model$semisupervised_info

  # Two classes carry labels, so two clusters
  expect_equal(nrow(info$cluster_model$fit$model$centers), 2)
  expect_false(anyNA(info$label_mapping$cluster_label))

  # Every training label comes from a labelled row in the same cluster
  clusters <- info$cluster_model$fit$clusters$cluster
  kept <- as.integer(rownames(model$data))
  carried <- split(as.character(ir$Species[labelled]), clusters[labelled])
  from_own_cluster <- mapply(
    function(label, cluster) label %in% carried[[as.character(cluster)]],
    as.character(model$data$Species), clusters[kept]
  )
  expect_true(all(from_own_cluster))
  expect_equal(nrow(model$data) + info$n_unlabelled_dropped, nrow(ir))
})

test_that("a character response with missing labels propagates", {
  # The setosa cluster's only labels are NA. table() of them is empty, and
  # summarize() failed with "Can't combine NULL and non NULL results"
  chr <- transform(iris, Species = as.character(Species))
  chr$Species[1:5] <- NA
  set.seed(2)
  expect_warning(
    model <- tl_semisupervised(chr, Species ~ .,
                               labeled_indices = c(1:5, 51:55, 101:105)),
    "5 labelled rows whose own label is missing"
  )
  expect_s3_class(model, "tidylearn_semisupervised")
  expect_false(anyNA(model$data$Species))
  # The setosa rows have no labelled class to take, so none is trained on
  expect_false(any(as.integer(rownames(model$data)) %in% 1:50))
})

test_that("tl_semisupervised needs two labelled classes", {
  expect_error(
    tl_semisupervised(iris, Species ~ ., labeled_indices = 1:10),
    "needs labelled rows from at least two classes; found 1 \\(setosa\\)"
  )
  no_labels <- iris
  no_labels$Species[1:10] <- NA
  expect_error(
    tl_semisupervised(no_labels, Species ~ ., labeled_indices = 1:10),
    "found 0"
  )
})

test_that("supervised and clustering settings reach their own stage", {
  labelled <- c(1:5, 51:55, 101:105)

  # cp = 0.001 went to k-means too, which failed with "unused argument"
  set.seed(3)
  semi <- tl_semisupervised(iris, Species ~ ., labeled_indices = labelled,
                            cp = 0.001)
  expect_equal(semi$fit$control$cp, 0.001)

  # and k-means settings had no way in that left the supervised model alone
  set.seed(3)
  semi_lloyd <- tl_semisupervised(
    iris, Species ~ ., labeled_indices = labelled,
    cluster_args = list(algorithm = "Lloyd", nstart = 1)
  )
  # kmeans() sets ifault for Hartigan-Wong only, so a recorded convergence
  # flag of NA is the mark of Lloyd
  expect_true(is.na(
    semi_lloyd$semisupervised_info$cluster_model$fit$metrics$converged
  ))

  set.seed(3)
  strat <- tl_stratified_models(mtcars, mpg ~ wt + hp, k = 2, cp = 0.001,
                                cluster_args = list(algorithm = "Lloyd"))
  expect_true(all(vapply(strat$supervised_models,
                         function(m) m$fit$control$cp, numeric(1)) == 0.001))
  expect_true(is.na(strat$cluster_model$fit$metrics$converged))

  expect_error(
    tl_semisupervised(iris, Species ~ ., labeled_indices = labelled,
                      cluster_args = list(k = 4)),
    "sets k to the number of labelled classes"
  )
  expect_error(
    tl_stratified_models(mtcars, mpg ~ wt, cluster_args = list(k = 4)),
    "Pass k as the k argument"
  )
  expect_error(
    tl_stratified_models(mtcars, mpg ~ wt, cluster_args = list(5)),
    "'cluster_args' must be a named list"
  )
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
  direct_tree <- rpart::rpart(
    Species ~ ., data = iris, method = "class",
    weights = ifelse(tree$anomaly_info$is_anomaly, 0.1, 1)
  )
  expect_equal(tree$fit$frame, direct_tree$frame)

  # and lm() errored on weights arriving through ...
  d <- mtcars[, c("mpg", "wt", "hp")]
  lin <- tl_anomaly_aware(d, mpg ~ wt + hp, response = "mpg",
                          action = "downweight",
                          supervised_method = "linear", eps = 15, minPts = 3)
  expect_gt(lin$anomaly_info$n_anomalies, 0)
  w <- ifelse(lin$anomaly_info$is_anomaly, 0.1, 1)
  expect_equal(unname(coef(lin$fit)),
               unname(coef(lm(mpg ~ wt + hp, data = d, weights = w))))

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
  for (method in c("svm", "deep")) {
    expect_error(
      tl_anomaly_aware(d, mpg ~ wt + hp, response = "mpg",
                       action = "downweight", supervised_method = method,
                       eps = 15, minPts = 3),
      "action = \"downweight\" needs a method that takes case weights",
      info = method
    )
  }
})

test_that("downweight reaches boost, nn and xgboost fits", {
  skip_if_not_installed("dbscan")

  # gbm, nnet and xgboost take case weights, and downweight refused all
  # three. Each fit has to match its package called with the same weights.
  d <- iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length")]
  f <- Sepal.Length ~ Sepal.Width + Petal.Length
  downweighted <- function(method) {
    set.seed(11)
    tl_anomaly_aware(d, f, response = "Sepal.Length", action = "downweight",
                     supervised_method = method, eps = 0.3, minPts = 5)
  }

  boost <- downweighted("boost")
  expect_gt(boost$anomaly_info$n_anomalies, 0)
  w <- ifelse(boost$anomaly_info$is_anomaly, 0.1, 1)
  set.seed(11)
  direct_gbm <- gbm::gbm(f, data = d, distribution = "gaussian",
                         n.trees = 100, interaction.depth = 3,
                         shrinkage = 0.1, n.minobsinnode = 10, cv.folds = 0,
                         verbose = FALSE, weights = w)
  expect_equal(boost$fit$fit, direct_gbm$fit)

  nn <- downweighted("nn")
  set.seed(11)
  direct_nnet <- nnet::nnet(f, data = d, size = 5, decay = 0, maxit = 100,
                            trace = FALSE, linout = TRUE, weights = w)
  expect_equal(nn$fit$wts, direct_nnet$wts)

  skip_if_not_installed("xgboost")
  xgb <- downweighted("xgboost")
  x <- as.matrix(d[, c("Sepal.Width", "Petal.Length")])
  direct_xgb <- xgboost::xgb.train(
    params = list(objective = "reg:squarederror", eval_metric = "rmse",
                  max_depth = 6, eta = 0.3, subsample = 1,
                  colsample_bytree = 1, min_child_weight = 1, gamma = 0,
                  alpha = 0, lambda = 1),
    data = xgboost::xgb.DMatrix(x, label = d$Sepal.Length, weight = w),
    nrounds = 100, verbose = 0
  )
  expect_equal(unname(predict(xgb, d)$.pred), predict(direct_xgb, x),
               tolerance = 1e-6)
  # and the weights moved the fit away from the unweighted one
  unweighted <- tl_model(d, f, method = "xgboost")
  expect_false(isTRUE(all.equal(predict(xgb, d)$.pred,
                                predict(unweighted, d)$.pred)))
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

test_that("tl_anomaly_aware stops when DBSCAN marks every row", {
  skip_if_not_installed("dbscan")

  # At the default eps on unscaled mtcars all 32 cars are noise. remove
  # then failed in lm() with "0 (non-NA) cases", downweight returned the
  # unweighted fit without a word, and flag gave an NA coefficient.
  for (action in c("remove", "flag", "downweight")) {
    expect_error(
      tl_anomaly_aware(mtcars, mpg ~ wt + hp, response = "mpg",
                       action = action, supervised_method = "linear"),
      "DBSCAN marked all 32 rows as noise.*eps = 0.5.*minPts = 5",
      info = action
    )
  }

  # A neighbourhood wide enough for these units finds some normal rows
  model <- tl_anomaly_aware(mtcars, mpg ~ wt + hp, response = "mpg",
                            action = "remove", supervised_method = "linear",
                            eps = 40, minPts = 3)
  expect_lt(model$anomaly_info$n_anomalies, nrow(mtcars))
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

test_that("a cluster holding one class predicts that class", {
  # k-means puts all 50 setosa rows in a cluster of their own, and the tree
  # refused it as a one-class response, so the whole call failed
  set.seed(1)
  models <- tl_stratified_models(iris, Species ~ ., k = 3)
  expect_equal(unname(models$single_class_clusters), "setosa")
  expect_length(models$supervised_models, 2)

  setosa <- iris$Species == "setosa"
  preds <- predict(models)
  expect_identical(levels(preds$.pred), levels(iris$Species))
  expect_true(all(preds$.pred[setosa] == "setosa"))
  expect_false(anyNA(preds$.pred))

  probs <- predict(models, iris, type = "prob")
  expect_equal(probs$setosa[setosa], rep(1, sum(setosa)))
  expect_equal(probs$versicolor[setosa], rep(0, sum(setosa)))
  expect_equal(rowSums(probs[levels(iris$Species)]), rep(1, nrow(iris)))

  # type can also be given by position, as the other clusters' models take it
  expect_equal(predict(models, iris, "prob"), probs)
})

test_that("stratified logistic fits warn about the conversion once", {
  # Each cluster's logistic fit on a 0/1 numeric response warned that it
  # converts the response to a factor, once per cluster
  set.seed(214)
  binary <- data.frame(x1 = c(stats::rnorm(40), stats::rnorm(40, 6)),
                       x2 = stats::rnorm(80))
  binary$y <- as.integer(binary$x2 + stats::rnorm(80) > 0)

  seen <- 0
  set.seed(215)
  withCallingHandlers(
    models <- tl_stratified_models(binary, y ~ x1 + x2, k = 2,
                                   supervised_method = "logistic"),
    tidylearn_response_conversion = function(w) {
      seen <<- seen + 1
      invokeRestart("muffleWarning")
    }
  )
  expect_length(models$supervised_models, 2)
  expect_equal(seen, 1)
})

test_that("tl_stratified_models cuts hclust at k and predicts training rows", {
  # hclust failed with "unused argument (k = 2)", and pam and clara fits
  # could not predict even the rows they were fitted on
  hc <- tl_stratified_models(mtcars, mpg ~ wt + hp,
                             cluster_method = "hclust", k = 2)
  tree <- stats::hclust(stats::dist(mtcars[, c("wt", "hp")]),
                        method = "average")
  expect_equal(hc$clusters, unname(stats::cutree(tree, k = 2)))
  preds <- predict(hc)
  expect_equal(preds$.cluster, hc$clusters)
  expect_false(anyNA(preds$.pred))

  skip_if_not_installed("cluster")
  for (method in c("pam", "clara")) {
    fitted <- tl_stratified_models(mtcars, mpg ~ wt + hp,
                                   cluster_method = method, k = 2)
    preds <- predict(fitted)
    expect_equal(preds$.cluster, fitted$clusters, info = method)
    expect_false(anyNA(preds$.pred), info = method)
    # New rows still need a method that can assign them
    expect_error(predict(fitted, new_data = mtcars[1:3, ]),
                 "does not support out-of-sample prediction", info = method)
  }
})

test_that("the cluster-based helpers refuse a method they cannot use", {
  # dbscan takes no k and failed with "unused argument (k = 2)"
  expect_error(
    tl_stratified_models(mtcars, mpg ~ wt + hp, cluster_method = "dbscan"),
    paste0("'cluster_method' must be one of \"kmeans\", \"pam\", ",
           "\"clara\", \"hclust\"; got \"dbscan\".*chooses its own ",
           "number of clusters")
  )
  expect_error(
    tl_semisupervised(iris, Species ~ ., labeled_indices = c(1:5, 51:55),
                      cluster_method = "pca"),
    "'cluster_method' must be one of"
  )
})

test_that("clustering and anomaly detection use the formula's predictors", {
  # A column the formula excludes still took part in the clusters and the
  # noise points: with a row id in iris, which is sorted by class, the id
  # was one of the k-means columns.
  ir <- iris
  ir$id <- seq_len(nrow(ir))
  measurements <- names(iris)[1:4]

  set.seed(5)
  semi <- tl_semisupervised(ir, Species ~ . - id,
                            labeled_indices = c(1:5, 51:55, 101:105))
  expect_setequal(
    colnames(semi$semisupervised_info$cluster_model$fit$model$centers),
    measurements
  )

  set.seed(5)
  strat <- tl_stratified_models(ir, Species ~ . - id, k = 2)
  expect_setequal(colnames(strat$cluster_model$fit$model$centers),
                  measurements)

  skip_if_not_installed("dbscan")
  flagged <- tl_anomaly_aware(ir, Species ~ . - id, response = "Species")
  direct <- dbscan::dbscan(ir[, measurements], eps = 0.5, minPts = 5)
  expect_equal(flagged$anomaly_info$is_anomaly, direct$cluster == 0)
})
