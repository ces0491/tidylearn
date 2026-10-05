test_that("tl_auto_ml runs with basic settings", {
  skip_on_cran()
  skip_if_not_installed("randomForest")

  # Run with short time budget for testing
  result <- tl_auto_ml(iris, Species ~ .,
                       use_reduction = TRUE,
                       use_clustering = TRUE,
                       time_budget = 10)

  expect_type(result, "list")
  expect_true("models" %in% names(result))
  expect_true("results" %in% names(result))
  expect_true("best_model" %in% names(result))

  # Should have trained at least one model
  expect_gte(length(result$models), 1)
})

test_that("tl_auto_ml detects task type automatically", {
  skip_on_cran()

  # Classification task
  result_class <- tl_auto_ml(iris, Species ~ .,
                             task = "auto",
                             time_budget = 5)

  expect_type(result_class, "list")

  # Regression task
  result_reg <- tl_auto_ml(mtcars, mpg ~ wt + hp,
                           task = "auto",
                           time_budget = 5)

  expect_type(result_reg, "list")
})

test_that("tl_auto_ml respects time budget", {
  skip_on_cran()

  # Very short time budget
  start_time <- Sys.time()
  result <- tl_auto_ml(iris, Species ~ .,
                       time_budget = 5,
                       use_reduction = FALSE,
                       use_clustering = FALSE)
  end_time <- Sys.time()

  # Should complete within reasonable time (with some buffer)
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
  expect_lt(elapsed, 20)  # Should finish well before 20 seconds
})

test_that("tl_auto_ml can disable reduction and clustering", {
  skip_on_cran()

  result <- tl_auto_ml(iris, Species ~ .,
                       use_reduction = FALSE,
                       use_clustering = FALSE,
                       time_budget = 10)

  expect_type(result, "list")

  # Should still have baseline models
  expect_gte(length(result$models), 1)

  # Model names should not include "pca_" or "clustered_"
  model_names <- names(result$models)
  expect_false(any(grepl("^pca_", model_names)))
  expect_false(any(grepl("^clustered_", model_names)))
})

test_that("tl_auto_ml handles small datasets", {
  skip_on_cran()

  # Small dataset -- sampled across all three classes, since a
  # single-class response is not a classification problem
  small_data <- iris[c(1:10, 51:60, 101:110), ]

  result <- tl_auto_ml(small_data, Species ~ .,
                       time_budget = 5,
                       cv_folds = 3)

  expect_type(result, "list")
  expect_gte(length(result$models), 1)
})

test_that("tl_auto_ml rejects a single-class response", {
  skip_on_cran()

  expect_error(
    tl_auto_ml(iris[1:30, ], Species ~ ., time_budget = 5),
    "at least two observed classes"
  )
})

test_that("tl_auto_ml checks task and metric before fitting anything", {
  # Each of these used to be found out only after every candidate was
  # fitted: "sensitivity" could not be ranked, "roc_auc" scored every
  # model NA and returned the first, and task = "Classification" fitted
  # the regression methods
  expect_error(
    tl_auto_ml(iris, Species ~ ., task = "Classification"),
    "'task' must be one of \"auto\", \"classification\", \"regression\""
  )
  expect_error(
    tl_auto_ml(iris, Species ~ ., metric = "roc_auc"),
    "'metric' must be one of the classification metrics"
  )
  expect_error(
    tl_auto_ml(iris, Species ~ ., metric = "rmse"),
    "'metric' must be one of the classification metrics"
  )
  expect_error(
    tl_auto_ml(mtcars, mpg ~ wt, metric = c("rmse", "mae")),
    "'metric' must be one of the regression metrics"
  )

  # A task the response contradicts scored every candidate NA
  expect_error(
    tl_auto_ml(mtcars, mpg ~ wt, task = "classification"),
    "'mpg' is numeric, which tl_model\\(\\) fits as a regression"
  )
  expect_error(
    tl_auto_ml(iris, Species ~ ., task = "regression"),
    "'Species' is categorical, which tl_model\\(\\) fits as a classification"
  )

  # Refused before the run starts, so nothing is fitted or narrated
  expect_no_message(
    try(tl_auto_ml(iris, Species ~ ., metric = "roc_auc"), silent = TRUE)
  )
})

test_that("the leaderboard ranks every computable metric its own way", {
  # sensitivity and specificity were missing from the leaderboard's lists,
  # so ranking by them was refused after every model was fitted
  for (metric in c(tl_known_metrics(TRUE), tl_known_metrics(FALSE))) {
    results <- list(
      small = tibble::tibble(metric = metric, value = 0.2),
      large = tibble::tibble(metric = metric, value = 0.8)
    )
    expected <- if (tl_metric_higher_better(metric)) "large" else "small"
    expect_equal(create_leaderboard(results, metric, "any")$model[1],
                 expected, info = metric)
  }
})

test_that("AutoML's PCA features use the formula's predictors only", {
  # Species ~ . - id put the row id into the rotation. iris is sorted by
  # class, so the id tracks Species and carried it into the components.
  ir <- iris
  ir$id <- seq_len(nrow(ir))
  measurements <- names(iris)[1:4]

  predictors <- tl_formula_predictors(Species ~ . - id, ir)
  expect_setequal(predictors, measurements)

  pca <- tl_automl_pca_variant(ir, Species ~ . - id, "Species", predictors)
  rotation <- pca$reduction_model$fit$model$rotation
  expect_setequal(rownames(rotation), measurements)
  expect_equal(pca$n_components, 2)
  expect_equal(pca$formula, Species ~ PC1 + PC2, ignore_formula_env = TRUE)
  expect_setequal(names(pca$data), c("PC1", "PC2", "Species"))

  # The fold refit sees the same columns
  fold <- pca$transform(ir[1:120, ])
  expect_setequal(names(fold$apply(ir[121:150, ])), c("PC1", "PC2", "Species"))

  # An explicit right-hand side was rotated over all four measurements
  two <- tl_automl_pca_variant(
    iris, Species ~ Sepal.Length + Sepal.Width, "Species",
    c("Sepal.Length", "Sepal.Width")
  )
  expect_setequal(rownames(two$reduction_model$fit$model$rotation),
                  c("Sepal.Length", "Sepal.Width"))
  expect_equal(two$formula, Species ~ PC1, ignore_formula_env = TRUE)
})

test_that("AutoML's cluster features are fitted and used as the formula says", {
  # The centres were fitted on every column but the response, id included,
  # and an explicit formula left cluster_kmeans out of the clustered
  # models, so their predictions equalled the baselines'
  ir <- iris
  ir$id <- seq_len(nrow(ir))
  measurements <- names(iris)[1:4]

  set.seed(1)
  clustered <- tl_automl_cluster_variant(
    ir, Species ~ . - id, "Species", measurements, k = 3
  )
  expect_setequal(colnames(clustered$cluster_model$fit$model$centers),
                  measurements)
  expect_true("cluster_kmeans" %in% all.vars(clustered$formula))
  expect_false("id" %in% all.vars(clustered$formula))
  expect_s3_class(clustered$data$cluster_kmeans, "factor")

  fold <- clustered$transform(ir[1:120, ])
  expect_equal(fold$formula, clustered$formula)
  expect_s3_class(fold$apply(ir[121:150, ])$cluster_kmeans, "factor")

  set.seed(1)
  regression <- tl_automl_cluster_variant(
    mtcars, mpg ~ wt + hp, "mpg", c("wt", "hp"), k = 3
  )
  model <- tl_model(regression$data, regression$formula, method = "linear")
  expect_true(any(grepl("^cluster_kmeans", names(coef(model$fit)))))
})

test_that("extract_metric_score reads both evaluation result shapes", {
  # tl_evaluate() shape
  evaluated <- tibble::tibble(
    metric = c("accuracy", "f1"), value = c(0.9, 0.8)
  )
  expect_equal(extract_metric_score(evaluated, "accuracy"), 0.9)
  expect_equal(extract_metric_score(evaluated, "f1"), 0.8)
  expect_true(is.na(extract_metric_score(evaluated, "auc")))

  # tl_cv() shape
  cross_validated <- list(
    folds = list(),
    summary = tibble::tibble(
      metric = c("accuracy", "f1"),
      mean = c(0.7, 0.6),
      sd = c(0.1, 0.1)
    )
  )
  expect_equal(extract_metric_score(cross_validated, "accuracy"), 0.7)
  expect_true(is.na(extract_metric_score(cross_validated, "auc")))

  expect_true(is.na(extract_metric_score(NULL, "accuracy")))
})

test_that("tl_auto_ml leaderboard is scored and ranked", {
  skip_on_cran()

  binary_iris <- droplevels(subset(iris, Species != "virginica"))

  set.seed(301)
  result <- suppressWarnings(
    tl_auto_ml(binary_iris, Species ~ .,
               use_reduction = FALSE,
               use_clustering = FALSE,
               time_budget = 20,
               cv_folds = 3)
  )

  expect_true(all(c("model", "score", "evaluation") %in%
                    names(result$leaderboard)))
  expect_false(all(is.na(result$leaderboard$score)))
  expect_true(all(result$leaderboard$evaluation %in% c("cv", "train")))

  # Accuracy ranks highest-first, and the reported best model is the winner
  scored <- result$leaderboard$score[!is.na(result$leaderboard$score)]
  expect_false(is.unsorted(rev(scored)))
  expect_equal(
    result$leaderboard$model[1],
    result$leaderboard$model[which.max(result$leaderboard$score)]
  )
})

test_that("tl_auto_ml ranks regression models by ascending rmse", {
  skip_on_cran()

  set.seed(302)
  result <- suppressWarnings(
    tl_auto_ml(mtcars, mpg ~ wt + hp,
               use_reduction = FALSE,
               use_clustering = FALSE,
               time_budget = 20,
               cv_folds = 3)
  )

  expect_equal(result$metric, "rmse")
  scored <- result$leaderboard$score[!is.na(result$leaderboard$score)]
  expect_gt(length(scored), 0)
  expect_false(is.unsorted(scored))
})

test_that("tl_auto_ml returns best model", {
  skip_on_cran()

  result <- tl_auto_ml(iris, Species ~ .,
                       time_budget = 10)

  expect_true("best_model" %in% names(result))
  expect_s3_class(result$best_model, "tidylearn_model")

  # Best model should be one of the trained models
  expect_true(!is.null(result$best_model))
})

test_that("tl_auto_ml works with regression tasks", {
  skip_on_cran()

  result <- tl_auto_ml(mtcars, mpg ~ .,
                       task = "regression",
                       time_budget = 10)

  expect_type(result, "list")
  expect_gte(length(result$models), 1)
})

test_that("tl_auto_ml handles errors gracefully", {
  skip_on_cran()

  # Should not crash even if some models fail
  result <- tl_auto_ml(iris, Species ~ .,
                       time_budget = 5)

  expect_type(result, "list")
  # Should have trained at least one successful model
  expect_gte(length(result$models), 1)
})

test_that("every AutoML candidate can predict on raw new data", {
  skip_on_cran()

  split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)
  result <- suppressMessages(
    tl_auto_ml(split$train, Species ~ ., time_budget = 30, cv_folds = 3)
  )

  # The pca_ and clustered_ variants are fitted on columns that exist only
  # inside the search. Without the transform they carry, predicting on raw
  # new data fails with "object 'PC1' not found".
  expect_true(any(grepl("^pca_", names(result$models))))
  expect_true(any(grepl("^clustered_", names(result$models))))

  for (model_name in names(result$models)) {
    preds <- expect_no_error(
      predict(result$models[[model_name]], new_data = split$test)
    )
    expect_equal(nrow(preds), nrow(split$test), info = model_name)
    expect_true(all(preds$.pred %in% levels(iris$Species)), info = model_name)
  }
})

test_that("a column the formula excludes reaches no AutoML candidate", {
  skip_on_cran()

  # iris is sorted by class, so a row id tracks Species. `- id` kept it out
  # of the baselines but not out of the PCA rotation or the k-means
  # centres, so the pca_ and clustered_ candidates were scored partly on it.
  ir <- iris
  ir$id <- seq_len(nrow(ir))
  set.seed(22)
  result <- suppressMessages(
    tl_auto_ml(ir, Species ~ . - id, time_budget = 200, cv_folds = 3)
  )
  measurements <- names(iris)[1:4]

  pca_models <- result$models[grepl("^pca_", names(result$models))]
  clustered <- result$models[grepl("^clustered_", names(result$models))]
  expect_gt(length(pca_models), 0)
  expect_gt(length(clustered), 0)

  for (model in pca_models) {
    rotation <- model$feature_transform$reduction_model$fit$model$rotation
    expect_setequal(rownames(rotation), measurements)
  }
  for (model in clustered) {
    centers <- model$feature_transform$cluster_model$fit$model$centers
    expect_setequal(colnames(centers), measurements)
    expect_true("cluster_kmeans" %in% all.vars(model$spec$formula))
    expect_false("id" %in% all.vars(model$spec$formula))
  }

  # So these variants predict rows that carry no id at all
  for (model in c(pca_models, clustered)) {
    preds <- predict(model, new_data = iris[c(1, 51, 101), 1:4])
    expect_equal(nrow(preds), 3)
  }
})

test_that("tl_auto_ml ranks by sensitivity, highest first", {
  skip_on_cran()

  binary_iris <- droplevels(subset(iris, Species != "virginica"))
  set.seed(303)
  result <- suppressWarnings(suppressMessages(
    tl_auto_ml(binary_iris, Species ~ ., metric = "sensitivity",
               use_reduction = FALSE, use_clustering = FALSE,
               time_budget = 5, cv_folds = 2)
  ))

  expect_equal(result$metric, "sensitivity")
  scored <- result$leaderboard$score[!is.na(result$leaderboard$score)]
  expect_gt(length(scored), 0)
  expect_false(is.unsorted(rev(scored)))
})

test_that("the AutoML best model predicts whatever variant won", {
  skip_on_cran()

  split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)
  result <- suppressMessages(
    tl_auto_ml(split$train, Species ~ ., time_budget = 30, cv_folds = 3)
  )

  preds <- predict(result$best_model, new_data = split$test)
  expect_equal(nrow(preds), nrow(split$test))
  expect_false(anyNA(preds$.pred))
})

test_that("tl_explore keeps max_components components", {
  # max_components was recorded in the summary and never applied, so the
  # PCA kept all four components
  eda <- suppressMessages(tl_explore(iris, "Species", max_components = 2))
  direct <- stats::prcomp(iris[, 1:4], center = TRUE, scale. = TRUE)

  rotation <- eda$pca$fit$model$rotation
  expect_equal(ncol(rotation), 2)
  expect_equal(unname(rotation), unname(direct$rotation[, 1:2]))
  expect_equal(setdiff(names(eda$pca$fit$scores), ".obs_id"),
               c("PC1", "PC2"))
  expect_equal(nrow(eda$pca$fit$variance_explained), 2)
  expect_equal(eda$summary$n_components, 2)

  # More components than the data has keeps what there is
  wide <- suppressMessages(tl_explore(iris, "Species", max_components = 10))
  expect_equal(wide$summary$n_components, 4)
})

test_that("tl_explore refuses a k_range or max_components it cannot use", {
  # k = 1 has no silhouette, and failed with "incorrect number of
  # dimensions"
  expect_error(
    tl_explore(iris, "Species", k_range = 1:3),
    "'k_range' must hold whole numbers from 2 to 149; got 1, 2, 3"
  )
  expect_error(
    tl_explore(iris, "Species", max_components = 0),
    "'max_components' must be a single whole number of at least 1"
  )

  eda <- suppressMessages(tl_explore(iris, "Species", k_range = 2:3))
  expect_true(eda$summary$best_k %in% 2:3)
})

test_that("tl_transfer_learning takes a string and an explicit formula", {
  # A string formula gave "Response variable 'NA' not found", and an
  # explicit right-hand side failed because the tree was fitted on PCA
  # scores under a formula naming the raw columns
  from_string <- suppressMessages(tl_transfer_learning(iris, "Species ~ ."))
  expect_s3_class(from_string, "tidylearn_transfer")

  two <- suppressMessages(
    tl_transfer_learning(iris, Species ~ Sepal.Length + Petal.Length)
  )
  rotation <- two$pretrain_model$fit$model$rotation
  expect_setequal(rownames(rotation), c("Sepal.Length", "Petal.Length"))

  # Predictions are the supervised model's on prcomp() scores of the same
  # two columns
  rows <- iris[c(1, 51, 101, 150), ]
  direct <- stats::prcomp(iris[, c("Sepal.Length", "Petal.Length")],
                          center = TRUE, scale. = TRUE)
  scores <- as.data.frame(predict(direct, rows))
  expect_equal(
    unname(predict(two, rows)$.pred),
    unname(predict(two$supervised_model, new_data = scores)$.pred)
  )
})

test_that("tl_transfer_learning refuses a pre-training it cannot apply", {
  # "autoencoder" was documented and failed with "Unknown method", and an
  # MDS fit could not project the rows it was asked to predict
  for (method in c("autoencoder", "mds")) {
    expect_error(
      tl_transfer_learning(iris, Species ~ ., pretrain_method = method),
      "'pretrain_method' must be \"pca\"",
      info = method
    )
  }
})

test_that("a model with no feature transform is untouched by predict", {
  model <- tl_model(iris, Species ~ ., method = "tree")
  expect_null(model$feature_transform)
  expect_equal(nrow(predict(model, new_data = iris[1:10, ])), 10)
})

test_that("AutoML models predict a single row", {
  skip_on_cran()

  # The pca_ and clustered_ variants replay a transformation before
  # dispatching, so they route through the unsupervised predict paths where
  # a one-row frame is easiest to get wrong.
  split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)
  result <- suppressMessages(
    tl_auto_ml(split$train, Species ~ ., time_budget = 30, cv_folds = 3)
  )

  for (model_name in names(result$models)) {
    preds <- predict(result$models[[model_name]], new_data = split$test[1, ])
    expect_equal(nrow(preds), 1, info = model_name)
  }
})
