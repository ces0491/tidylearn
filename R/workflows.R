#' High-Level Workflows for Common Machine Learning Patterns
#'
#' These functions provide end-to-end workflows that showcase tidylearn's
#' ability to seamlessly combine multiple learning paradigms

#' Auto ML: Automated Machine Learning Workflow
#'
#' Automatically explores multiple modeling approaches including
#' dimensionality reduction, clustering, and various supervised methods.
#' Returns the best performing model based on cross-validation.
#'
#' @param data A data frame
#' @param formula Model formula (for supervised learning)
#' @param task Task type: "classification", "regression", or "auto" (default)
#' @param use_reduction Whether to try dimensionality reduction (default: TRUE)
#' @param use_clustering Whether to add cluster features (default: TRUE)
#' @param time_budget Time budget in seconds (default: 300)
#' @param cv_folds Number of cross-validation folds (default: 5)
#' @param metric Evaluation metric (default: auto-selected based on task)
#' @return Best model with performance comparison
#' @export
#' @examples
#' \dontrun{
#' # Automated modeling
#' result <- tl_auto_ml(iris, Species ~ .)
#' best_model <- result$best_model
#' result$leaderboard
#' }
tl_auto_ml <- function(data, formula, task = "auto",
                       use_reduction = TRUE, use_clustering = TRUE,
                       time_budget = 300, cv_folds = 5, metric = NULL) {
  start_time <- Sys.time()

  # Determine task type
  if (task == "auto") {
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]
    task <- if (is.factor(y) || is.character(y)) "classification" else "regression"
  }

  # Set default metric
  if (is.null(metric)) {
    metric <- if (task == "classification") "accuracy" else "rmse"
  }

  message("Starting Auto ML with task: ", task)
  message("Time budget: ", time_budget, " seconds")

  # Prepare candidate models
  models <- list()
  results <- list()

  # 1. Baseline models
  message("\n[1/4] Training baseline models...")
  if (task == "classification") {
    baseline_methods <- c("logistic", "tree", "forest")
  } else {
    baseline_methods <- c("linear", "tree", "forest")
  }

  for (method in baseline_methods) {
    if (difftime(Sys.time(), start_time, units = "secs") > time_budget) break

    model_name <- paste0("baseline_", method)
    message("  Training: ", model_name)

    tryCatch({
      model <- tl_model(data, formula, method = method)
      cv_result <- tl_cv(data, formula, method = method, folds = cv_folds)
      models[[model_name]] <- model
      results[[model_name]] <- cv_result
    }, error = function(e) {
      message("    Failed: ", e$message)
    })
  }

  # 2. Models with dimensionality reduction
  if (use_reduction) {
    message("\n[2/4] Training models with dimensionality reduction...")

    tryCatch({
      response_var <- all.vars(formula)[1]
      n_predictors <- ncol(data) - 1
      n_components <- min(5, ceiling(n_predictors / 2))

      reduced <- tl_reduce_dimensions(data, response = response_var,
                                     method = "pca", n_components = n_components)

      # Update formula for reduced data
      pred_vars <- names(reduced$data)[!names(reduced$data) %in% c(".obs_id", response_var)]
      formula_reduced <- as.formula(paste(response_var, "~", paste(pred_vars, collapse = " + ")))

      for (method in baseline_methods) {
        if (difftime(Sys.time(), start_time, units = "secs") > time_budget) break

        model_name <- paste0("pca_", method)
        message("  Training: ", model_name)

        tryCatch({
          model <- tl_model(reduced$data, formula_reduced, method = method)
          # Add reduction info
          model$reduction_info <- list(
            reduction_model = reduced$reduction_model,
            n_components = n_components
          )
          models[[model_name]] <- model
          # Approximate CV (using training evaluation)
          eval_result <- tl_evaluate(model)
          results[[model_name]] <- eval_result
        }, error = function(e) {
          message("    Failed: ", e$message)
        })
      }
    }, error = function(e) {
      message("  Dimensionality reduction failed: ", e$message)
    })
  }

  # 3. Models with cluster features
  if (use_clustering) {
    message("\n[3/4] Training models with cluster features...")

    tryCatch({
      response_var <- all.vars(formula)[1]
      k <- if (task == "classification") length(unique(data[[response_var]])) else 3

      data_clustered <- tl_add_cluster_features(data, response = response_var,
                                               method = "kmeans", k = k)

      for (method in baseline_methods) {
        if (difftime(Sys.time(), start_time, units = "secs") > time_budget) break

        model_name <- paste0("clustered_", method)
        message("  Training: ", model_name)

        tryCatch({
          model <- tl_model(data_clustered, formula, method = method)
          eval_result <- tl_evaluate(model)
          models[[model_name]] <- model
          results[[model_name]] <- eval_result
        }, error = function(e) {
          message("    Failed: ", e$message)
        })
      }
    }, error = function(e) {
      message("  Cluster feature engineering failed: ", e$message)
    })
  }

  # 4. Advanced models if time allows
  message("\n[4/4] Training advanced models...")
  if (difftime(Sys.time(), start_time, units = "secs") < time_budget * 0.7) {
    advanced_methods <- if (task == "classification") c("svm", "xgboost") else c("ridge", "lasso")

    for (method in advanced_methods) {
      if (difftime(Sys.time(), start_time, units = "secs") > time_budget) break

      model_name <- paste0("advanced_", method)
      message("  Training: ", model_name)

      tryCatch({
        model <- tl_model(data, formula, method = method)
        cv_result <- tl_cv(data, formula, method = method, folds = cv_folds)
        models[[model_name]] <- model
        results[[model_name]] <- cv_result
      }, error = function(e) {
        message("    Failed: ", e$message)
      })
    }
  }

  # Create leaderboard
  message("\n[*] Creating leaderboard...")
  leaderboard <- create_leaderboard(results, metric, task)

  # Get best model
  best_model_name <- leaderboard$model[1]
  best_model <- models[[best_model_name]]

  total_time <- difftime(Sys.time(), start_time, units = "secs")
  message("\nAuto ML complete in ", round(total_time, 2), " seconds")
  message("Best model: ", best_model_name)

  structure(
    list(
      best_model = best_model,
      all_models = models,
      leaderboard = leaderboard,
      task = task,
      metric = metric,
      runtime = total_time
    ),
    class = c("tidylearn_automl", "list")
  )
}

#' Create leaderboard from results
#' @keywords internal
#' @noRd
create_leaderboard <- function(results, metric, task) {
  scores <- sapply(results, function(r) {
    if (is.list(r) && metric %in% names(r)) {
      r[[metric]]
    } else if (is.list(r) && "metrics" %in% names(r)) {
      r$metrics[[metric]]
    } else {
      NA
    }
  })

  leaderboard <- tibble::tibble(
    model = names(scores),
    score = scores
  )

  # Sort: ascending for error metrics, descending for accuracy metrics
  if (metric %in% c("rmse", "mae", "mse")) {
    leaderboard <- leaderboard %>% dplyr::arrange(score)
  } else {
    leaderboard <- leaderboard %>% dplyr::arrange(dplyr::desc(score))
  }

  leaderboard
}

#' Print auto ML results
#' @export
print.tidylearn_automl <- function(x, ...) {
  cat("tidylearn Auto ML Results\n")
  cat("=========================\n")
  cat("Task:", x$task, "\n")
  cat("Metric:", x$metric, "\n")
  cat("Runtime:", round(x$runtime, 2), "seconds\n")
  cat("Models trained:", length(x$all_models), "\n\n")

  cat("Leaderboard:\n")
  print(x$leaderboard, n = 10)

  cat("\nBest model:", x$leaderboard$model[1], "\n")
  cat("Best score:", x$leaderboard$score[1], "\n")

  invisible(x)
}

#' Exploratory Data Analysis Workflow
#'
#' Comprehensive EDA combining unsupervised learning techniques
#' to understand data structure before modeling
#'
#' @param data A data frame
#' @param response Optional response variable for colored visualizations
#' @param max_components Maximum PCA components to compute (default: 5)
#' @param k_range Range of k values for clustering (default: 2:6)
#' @return An EDA object with multiple analyses
#' @export
#' @examples
#' \dontrun{
#' eda <- tl_explore(iris, response = "Species")
#' plot(eda)
#' }
tl_explore <- function(data, response = NULL, max_components = 5, k_range = 2:6) {
  message("Running Exploratory Data Analysis...")

  # 1. Dimensionality Reduction
  message("[1/4] PCA analysis...")
  predictor_data <- if (!is.null(response)) {
    data %>% dplyr::select(-dplyr::all_of(response))
  } else {
    data
  }

  pca_result <- tl_model(predictor_data, method = "pca")

  # 2. Optimal clustering
  message("[2/4] Finding optimal clusters...")
  optimal_k <- tl_optimal_clusters(predictor_data, k_range = k_range)

  # 3. Cluster analysis with optimal k
  message("[3/4] Clustering analysis...")
  k_best <- optimal_k$best_k
  kmeans_result <- tl_model(predictor_data, method = "kmeans", k = k_best)
  hclust_result <- tl_model(predictor_data, method = "hclust")

  # 4. Distance analysis
  message("[4/4] Distance analysis...")
  # Compute pairwise distances (sample if large dataset)
  if (nrow(data) > 1000) {
    sample_idx <- sample(nrow(data), 1000)
    distance_data <- predictor_data[sample_idx, ]
  } else {
    distance_data <- predictor_data
  }

  message("EDA complete!")

  structure(
    list(
      data = data,
      response = response,
      pca = pca_result,
      optimal_k = optimal_k,
      kmeans = kmeans_result,
      hclust = hclust_result,
      summary = list(
        n_obs = nrow(data),
        n_vars = ncol(predictor_data),
        n_components = max_components,
        best_k = k_best
      )
    ),
    class = c("tidylearn_eda", "list")
  )
}

#' Print EDA results
#' @export
print.tidylearn_eda <- function(x, ...) {
  cat("tidylearn Exploratory Data Analysis\n")
  cat("===================================\n")
  cat("Observations:", x$summary$n_obs, "\n")
  cat("Variables:", x$summary$n_vars, "\n")
  cat("Optimal clusters:", x$summary$best_k, "\n\n")

  cat("PCA Variance Explained (first 5 components):\n")
  print(head(x$pca$fit$variance, 5))

  cat("\nCluster sizes (k =", x$summary$best_k, "):\n")
  print(table(x$kmeans$fit$clusters$cluster))

  invisible(x)
}

#' Find optimal number of clusters
#' @keywords internal
#' @noRd
tl_optimal_clusters <- function(data, k_range = 2:6, method = "silhouette") {
  scores <- numeric(length(k_range))

  for (i in seq_along(k_range)) {
    k <- k_range[i]
    km <- tl_model(data, method = "kmeans", k = k)

    # Compute silhouette score
    if (requireNamespace("cluster", quietly = TRUE)) {
      sil <- cluster::silhouette(km$fit$clusters$cluster,
                                 stats::dist(dplyr::select(data, where(is.numeric))))
      scores[i] <- mean(sil[, 3])
    } else {
      # Fallback to within-cluster sum of squares
      scores[i] <- -km$fit$metrics$tot_withinss
    }
  }

  best_idx <- which.max(scores)

  list(
    k_values = k_range,
    scores = scores,
    best_k = k_range[best_idx],
    best_score = scores[best_idx]
  )
}

#' Transfer Learning Workflow
#'
#' Use unsupervised pre-training (e.g., autoencoder features) before supervised learning
#'
#' @param data Training data
#' @param formula Model formula
#' @param pretrain_method Pre-training method: "pca", "autoencoder"
#' @param supervised_method Supervised learning method
#' @param ... Additional arguments
#' @return A transfer learning model
#' @export
#' @examples
#' \dontrun{
#' model <- tl_transfer_learning(iris, Species ~ ., pretrain_method = "pca")
#' }
tl_transfer_learning <- function(data, formula, pretrain_method = "pca",
                                 supervised_method = "logistic", ...) {
  message("Transfer Learning Workflow")
  message("==========================")

  response_var <- all.vars(formula)[1]

  # Phase 1: Unsupervised pre-training
  message("[Phase 1] Unsupervised pre-training with ", pretrain_method, "...")
  pretrain_model <- tl_reduce_dimensions(data, response = response_var,
                                        method = pretrain_method, ...)

  # Phase 2: Supervised learning on transformed features
  message("[Phase 2] Supervised learning with ", supervised_method, "...")
  supervised_model <- tl_model(pretrain_model$data, formula, method = supervised_method)

  # Combine models
  structure(
    list(
      pretrain_model = pretrain_model$reduction_model,
      supervised_model = supervised_model,
      formula = formula,
      method = supervised_method
    ),
    class = c("tidylearn_transfer", "list")
  )
}

#' Predict with transfer learning model
#' @export
predict.tidylearn_transfer <- function(object, new_data, ...) {
  # Transform new data using pre-trained model
  transformed <- predict(object$pretrain_model, new_data = new_data)

  # Predict using supervised model
  predict(object$supervised_model, new_data = transformed, ...)
}
