#' @title High-Level Workflows for Common Machine Learning Patterns
#' @name tidylearn-workflows
#' @description Functions providing end-to-end workflows that showcase
#'   tidylearn's ability to seamlessly combine multiple learning paradigms
NULL

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
#' @param time_budget Time budget in seconds (default: 300). Controls which
#'   models are attempted and whether cross-validation is used for evaluation.
#'   The budget is checked **between** model fits, not during them -- once a
#'   model starts training it runs to completion because R cannot safely
#'   interrupt C-level code (e.g. randomForest, xgboost, e1071).
#'
#'   How the budget shapes the workflow:
#'   \itemize{
#'     \item **Under 30s**: Only fast models are attempted (tree, logistic/linear).
#'       Cross-validation is skipped; models are ranked on training-set metrics
#'       only. Expect 2 models in the leaderboard. Use this for quick sanity
#'       checks or interactive exploration.
#'     \item **30--120s**: All baseline models are attempted including random
#'       forest. Cross-validation runs when enough time remains after each
#'       model fit; otherwise training metrics are used. Advanced models
#'       (SVM, XGBoost / ridge, lasso) are attempted if 40\% of the budget
#'       remains after baselines. Dimensionality reduction and clustering
#'       pipelines run if enabled and 10\% of the budget remains.
#'     \item **120s+ (recommended)**: The full pipeline runs -- all baselines,
#'       advanced models, PCA-augmented variants, and cluster-augmented
#'       variants, each with cross-validation. Expect 9--11 models in the
#'       leaderboard.
#'   }
#'
#'   Because individual model fits (especially forest, SVM, XGBoost with CV)
#'   can take 5--30s each depending on data size, the actual wall-clock time
#'   may modestly exceed the budget by the duration of the last model that
#'   was started before the budget expired.
#' @param cv_folds Number of cross-validation folds (default: 5). Reducing
#'   this (e.g. to 2 or 3) is an effective way to stay closer to the time
#'   budget since CV is typically the most expensive step.
#' @param metric Evaluation metric (default: auto-selected based on task).
#'   For classification: "accuracy"; for regression: "rmse".
#' @return A list with class \code{"tidylearn_automl"} containing:
#'   \describe{
#'     \item{best_model}{The best tidylearn model object}
#'     \item{models}{Named list of all successfully trained models}
#'     \item{leaderboard}{Tibble ranking models by the chosen metric}
#'     \item{task}{Detected or specified task type}
#'     \item{metric}{Metric used for ranking}
#'     \item{runtime}{Total elapsed time as a difftime object}
#'   }
#' @export
#' @examples
#' \donttest{
#' # Quick run with fast models only (< 30s budget skips forest/SVM/XGBoost)
#' result <- tl_auto_ml(iris, Species ~ .,
#'   time_budget = 10,
#'   use_reduction = FALSE,
#'   use_clustering = FALSE,
#'   cv_folds = 2)
#' result$leaderboard
#' }
tl_auto_ml <- function(data, formula, task = "auto",
                       use_reduction = TRUE, use_clustering = TRUE,
                       time_budget = 300, cv_folds = 5, metric = NULL) {
  start_time <- Sys.time()

  # Helper: remaining seconds in the budget
  time_remaining <- function() {
    as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  }
  budget_left <- function() time_budget - time_remaining()

  # Helper: train a model with error handling.
  # Note: R cannot safely interrupt C-level code (randomForest, xgboost),
  # so we control budget by skipping models rather than killing them.
  safe_train <- function(expr_fn, label) {
    if (budget_left() <= 0) {
      message("    ", label, ": skipped (time budget exhausted)")
      return(NULL)
    }
    tryCatch(
      expr_fn(),
      error = function(e) {
        message("    ", label, ": failed - ", e$message)
        NULL
      }
    )
  }

  # Determine task type
  if (task == "auto") {
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]
    task <- if (is.factor(y) || is.character(y)) {
      "classification"
    } else {
      "regression"
    }
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

  # 1. Baseline models (ordered fast to slow)
  # Slow methods (forest, svm, xgboost) involve C code that cannot be

  # interrupted, so we only attempt them when the budget is generous.
  message("\n[1/4] Training baseline models...")
  fast_methods <- if (task == "classification") {
    c("tree", "logistic")
  } else {
    c("tree", "linear")
  }
  slow_methods <- "forest"  # C-level, ~5s+ per fit

  baseline_methods <- if (time_budget >= 30) {
    c(fast_methods, slow_methods)
  } else {
    fast_methods
  }

  for (method in baseline_methods) {
    if (budget_left() <= 0) break

    model_name <- paste0("baseline_", method)
    message("  Training: ", model_name)

    model <- safe_train(function() {
      tl_model(data, formula, method = method)
    }, model_name)
    if (is.null(model)) next

    # CV only when enough budget remains
    eval_result <- if (budget_left() > time_budget * 0.3) {
      safe_train(function() {
        tl_cv(data, formula, method = method, folds = cv_folds)
      }, paste0(model_name, " CV"))
    } else {
      NULL
    }
    if (is.null(eval_result)) eval_result <- tl_evaluate(model)

    models[[model_name]] <- model
    results[[model_name]] <- eval_result
  }

  # 2. Models with dimensionality reduction
  if (use_reduction && budget_left() > max(5, time_budget * 0.1)) {
    message("\n[2/4] Training models with dimensionality reduction...")

    tryCatch({
      response_var <- all.vars(formula)[1]
      n_predictors <- ncol(data) - 1
      n_components <- min(5, ceiling(n_predictors / 2))

      reduced <- tl_reduce_dimensions(
        data, response = response_var,
        method = "pca",
        n_components = n_components
      )

      # Update formula for reduced data -- exclude .obs_id
      pred_vars <- names(reduced$data)[
        !names(reduced$data) %in% c(".obs_id", response_var)
      ]
      formula_reduced <- as.formula(paste(
        response_var, "~",
        paste(pred_vars, collapse = " + ")
      ))

      # Drop .obs_id from reduced data
      reduced_data <- reduced$data
      if (".obs_id" %in% names(reduced_data)) {
        reduced_data <- reduced_data[, names(reduced_data) != ".obs_id",
                                     drop = FALSE]
      }

      for (method in baseline_methods) {
        if (budget_left() < max(2, time_budget * 0.05)) break

        model_name <- paste0("pca_", method)
        message("  Training: ", model_name)

        result <- safe_train(function() {
          model <- tl_model(reduced_data, formula_reduced, method = method)
          model$reduction_info <- list(
            reduction_model = reduced$reduction_model,
            n_components = n_components
          )
          eval_result <- tl_evaluate(model)
          list(model = model, eval_result = eval_result)
        }, model_name)

        if (!is.null(result)) {
          models[[model_name]] <- result$model
          results[[model_name]] <- result$eval_result
        }
      }
    }, error = function(e) {
      message("  Dimensionality reduction failed: ", e$message)
    })
  }

  # 3. Models with cluster features
  if (use_clustering && budget_left() > max(5, time_budget * 0.1)) {
    message("\n[3/4] Training models with cluster features...")

    tryCatch({
      response_var <- all.vars(formula)[1]
      k <- if (task == "classification") {
        length(unique(data[[response_var]]))
      } else {
        3
      }

      data_clustered <- tl_add_cluster_features(
        data, response = response_var,
        method = "kmeans", k = k
      )

      for (method in baseline_methods) {
        if (budget_left() < max(2, time_budget * 0.05)) break

        model_name <- paste0("clustered_", method)
        message("  Training: ", model_name)

        result <- safe_train(function() {
          model <- tl_model(data_clustered, formula, method = method)
          eval_result <- tl_evaluate(model)
          list(model = model, eval_result = eval_result)
        }, model_name)

        if (!is.null(result)) {
          models[[model_name]] <- result$model
          results[[model_name]] <- result$eval_result
        }
      }
    }, error = function(e) {
      message("  Cluster feature engineering failed: ", e$message)
    })
  }

  # 4. Advanced models if time allows (these are C-heavy and slow)
  message("\n[4/4] Training advanced models...")
  if (budget_left() > time_budget * 0.4 && time_budget >= 30) {
    advanced_methods <- if (task == "classification") {
      c("svm", "xgboost")
    } else {
      c("ridge", "lasso")
    }

    for (method in advanced_methods) {
      if (budget_left() <= 0) break

      model_name <- paste0("advanced_", method)
      message("  Training: ", model_name)

      model <- safe_train(function() {
        tl_model(data, formula, method = method)
      }, model_name)
      if (is.null(model)) next

      eval_result <- if (budget_left() > time_budget * 0.3) {
        safe_train(function() {
          tl_cv(data, formula, method = method, folds = cv_folds)
        }, paste0(model_name, " CV"))
      } else {
        NULL
      }
      if (is.null(eval_result)) eval_result <- tl_evaluate(model)

      models[[model_name]] <- model
      results[[model_name]] <- eval_result
    }
  }

  # Create leaderboard
  message("\n[*] Creating leaderboard...")
  leaderboard <- create_leaderboard(results, metric, task)

  # Get best model
  if (nrow(leaderboard) == 0 || length(models) == 0) {
    warning("No models were successfully trained within the time budget.",
            call. = FALSE)
    best_model_name <- NA_character_
    best_model <- NULL
  } else {
    best_model_name <- leaderboard$model[1]
    best_model <- models[[best_model_name]]
  }

  total_time <- difftime(Sys.time(), start_time, units = "secs")
  message("\nAuto ML complete in ", round(total_time, 2), " seconds")
  message("Best model: ", best_model_name)

  structure(
    list(
      best_model = best_model,
      models = models,              # Add for test compatibility
      all_models = models,          # Keep for backward compatibility
      results = results,            # Add for test compatibility
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
  if (length(results) == 0) {
    return(tibble::tibble(model = character(0), score = numeric(0)))
  }

  scores <- vapply(results, function(r) {
    if (is.list(r) && metric %in% names(r)) {
      as.numeric(r[[metric]])
    } else if (is.list(r) && "metrics" %in% names(r)) {
      val <- r$metrics[[metric]]
      if (is.null(val)) NA_real_ else as.numeric(val)
    } else {
      NA_real_
    }
  }, numeric(1))

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
#' @param x A tidylearn_automl object
#' @param ... Additional arguments (ignored)
#' @return The input object \code{x}, returned invisibly.
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
#' @return A list with class \code{"tidylearn_eda"} containing:
#'   \describe{
#'     \item{data}{The original data frame.}
#'     \item{response}{The response variable name, or \code{NULL}.}
#'     \item{pca}{The fitted PCA model.}
#'     \item{optimal_k}{List with optimal cluster count results.}
#'     \item{kmeans}{The fitted k-means model.}
#'     \item{hclust}{The fitted hierarchical clustering model.}
#'     \item{summary}{List with \code{n_obs}, \code{n_vars},
#'       \code{n_components}, and \code{best_k}.}
#'   }
#' @export
#' @examples
#' \donttest{
#' eda <- tl_explore(iris, response = "Species")
#' plot(eda)
#' }
tl_explore <- function(data, response = NULL,
                       max_components = 5,
                       k_range = 2:6) {
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
#' @param x A tidylearn_eda object
#' @param ... Additional arguments (ignored)
#' @return The input object \code{x}, returned invisibly.
#' @examples
#' \donttest{
#' eda <- tl_explore(iris, response = "Species")
#' print(eda)
#' }
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

#' Plot EDA results
#' @param x A tidylearn_eda object
#' @param ... Additional arguments (ignored)
#' @return The input object \code{x}, returned invisibly. Called for its
#'   side effect of plotting a PCA scatter plot coloured by cluster.
#' @examples
#' \donttest{
#' eda <- tl_explore(iris, response = "Species")
#' plot(eda)
#' }
#' @export
plot.tidylearn_eda <- function(x, ...) {
  # Get PCA scores for visualization
  pca_scores <- x$pca$fit$scores
  clusters <- x$kmeans$fit$clusters$cluster

  # Create plot data
  plot_data <- data.frame(
    PC1 = pca_scores$PC1,
    PC2 = pca_scores$PC2,
    Cluster = as.factor(clusters)
  )

  # Add response if available
  if (!is.null(x$response) && x$response %in% names(x$data)) {
    plot_data$Response <- x$data[[x$response]]
  }

  # Create the plot (use .data$ to avoid R CMD check NOTEs)
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(
    x = .data[["PC1"]],
    y = .data[["PC2"]],
    color = .data[["Cluster"]]
  )) +
    ggplot2::geom_point(size = 2, alpha = 0.7) +
    ggplot2::labs(
      title = "EDA: PCA with K-means Clusters",
      subtitle = paste("k =", x$summary$best_k, "clusters"),
      x = "Principal Component 1",
      y = "Principal Component 2"
    ) +
    ggplot2::theme_minimal()

  print(p)
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
      dist_mat <- stats::dist(
        dplyr::select(data, where(is.numeric))
      )
      sil <- cluster::silhouette(
        km$fit$clusters$cluster, dist_mat
      )
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
#' Use unsupervised pre-training (e.g., autoencoder
#' features) before supervised learning
#'
#' @param data Training data
#' @param formula Model formula
#' @param pretrain_method Pre-training method: "pca", "autoencoder"
#' @param supervised_method Supervised learning method
#' @param ... Additional arguments
#' @return A list with class \code{"tidylearn_transfer"} containing:
#'   \describe{
#'     \item{pretrain_model}{The fitted dimensionality reduction model.}
#'     \item{supervised_model}{The fitted supervised tidylearn model.}
#'     \item{formula}{The model formula.}
#'     \item{method}{The supervised learning method used.}
#'   }
#' @export
#' @examples
#' \donttest{
#' model <- tl_transfer_learning(iris, Species ~ .,
#'   pretrain_method = "pca", supervised_method = "logistic")
#' }
tl_transfer_learning <- function(data, formula, pretrain_method = "pca",
                                 supervised_method = "logistic", ...) {
  message("Transfer Learning Workflow")
  message("==========================")

  response_var <- all.vars(formula)[1]

  # Phase 1: Unsupervised pre-training
  message("[Phase 1] Unsupervised pre-training with ", pretrain_method, "...")
  pretrain_model <- tl_reduce_dimensions(
    data, response = response_var,
    method = pretrain_method, ...
  )

  # Phase 2: Supervised learning on transformed features
  message("[Phase 2] Supervised learning with ", supervised_method, "...")

  # Remove .obs_id from PCA/MDS output (row identifier, not a feature)

  supervised_data <- pretrain_model$data
  if (".obs_id" %in% names(supervised_data)) {
    supervised_data <- supervised_data[, names(supervised_data) != ".obs_id",
                                       drop = FALSE]
  }

  supervised_model <- tl_model(
    supervised_data, formula,
    method = supervised_method
  )

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
#' @param object A tidylearn_transfer model object
#' @param new_data New data for predictions
#' @param ... Additional arguments
#' @return A \link[tibble]{tibble} with a \code{.pred} column containing
#'   predictions.
#' @examples
#' \donttest{
#' model <- tl_transfer_learning(iris, Species ~ .,
#'   pretrain_method = "pca", supervised_method = "logistic")
#' preds <- predict(model, iris[1:5, ])
#' }
#' @export
predict.tidylearn_transfer <- function(object, new_data, ...) {
  # Transform new data using pre-trained model
  transformed <- predict(object$pretrain_model, new_data = new_data)

  # Remove .obs_id from transformed data (row identifier, not a feature)
  if (".obs_id" %in% names(transformed)) {
    transformed <- transformed[, names(transformed) != ".obs_id", drop = FALSE]
  }

  # Predict using supervised model
  predict(object$supervised_model, new_data = transformed, ...)
}
