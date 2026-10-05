#' @title High-Level Workflows for Common Machine Learning Patterns
#' @name tidylearn-workflows
#' @description Functions providing end-to-end workflows that showcase
#'   tidylearn's ability to seamlessly combine multiple learning paradigms
NULL

#' The full set of cluster labels a fitted clustering model can emit
#'
#' @param cluster_model A fitted tidylearn clustering model
#' @return A character vector of cluster labels
#' @keywords internal
#' @noRd
tl_cluster_levels <- function(cluster_model) {
  centers <- cluster_model$fit$model$centers
  if (!is.null(centers)) {
    return(as.character(seq_len(nrow(centers))))
  }

  as.character(sort(unique(cluster_model$fit$clusters$cluster)))
}

#' Settle tl_auto_ml()'s task against its response
#'
#' @param task The \code{task} argument
#' @param y The response column
#' @param response_var Its name, for the message
#' @return \code{"classification"} or \code{"regression"}
#' @keywords internal
#' @noRd
tl_automl_task <- function(task, y, response_var) {
  tasks <- c("auto", "classification", "regression")
  if (!is.character(task) || length(task) != 1L || !task %in% tasks) {
    stop(
      "'task' must be one of ", paste0("\"", tasks, "\"", collapse = ", "),
      "; got ", tl_describe_value(task), ".",
      call. = FALSE
    )
  }

  # tl_model() takes the task from the response whatever `task` says, so a
  # task the response contradicts had every candidate scored on metrics it
  # could not produce, and the leaderboard came back all NA
  observed <- if (is.factor(y) || is.character(y)) {
    "classification"
  } else {
    "regression"
  }
  if (task != "auto" && task != observed) {
    stop(
      "task = \"", task, "\", but '", response_var, "' is ",
      if (observed == "classification") "categorical" else "numeric",
      ", which tl_model() fits as a ", observed, ". Use task = \"",
      observed, "\" or \"auto\"",
      if (task == "classification") {
        paste0(", or convert '", response_var,
               "' with factor() if its values are classes")
      } else {
        ""
      },
      ".",
      call. = FALSE
    )
  }

  observed
}

#' Settle tl_auto_ml()'s ranking metric
#'
#' @param metric The \code{metric} argument, or NULL for the task's default
#' @param task \code{"classification"} or \code{"regression"}
#' @return A single metric name that tl_evaluate() computes for the task
#' @keywords internal
#' @noRd
tl_automl_metric <- function(metric, task) {
  is_classification <- task == "classification"
  if (is.null(metric)) {
    return(if (is_classification) "accuracy" else "rmse")
  }

  known <- tl_known_metrics(is_classification)
  if (!is.character(metric) || length(metric) != 1L || !metric %in% known) {
    stop(
      "'metric' must be one of the ", task, " metrics: ",
      paste(known, collapse = ", "), "; got ", tl_describe_value(metric),
      ".",
      call. = FALSE
    )
  }

  metric
}

#' The numeric columns among a formula's predictors
#'
#' PCA and k-means take numeric columns only, and drop the rest themselves.
#' Selecting them here keeps the fold refits and the stored model on the
#' same columns.
#'
#' @param data The training data
#' @param predictor_vars The formula's predictors
#' @param what What they are for, for the message
#' @return A character vector of column names
#' @keywords internal
#' @noRd
tl_numeric_predictors <- function(data, predictor_vars, what) {
  numeric_vars <- predictor_vars[
    vapply(data[predictor_vars], is.numeric, logical(1))
  ]
  if (length(numeric_vars) == 0) {
    stop("the formula names no numeric predictors to ", what, ".",
         call. = FALSE)
  }
  numeric_vars
}

#' Build AutoML's PCA-augmented candidates
#'
#' The rotation is fitted on the formula's predictors only. Fitted on every
#' column but the response, it took in columns the formula leaves out, and
#' a row id on data sorted by class carries the class into the components.
#'
#' @param data The training data
#' @param formula The caller's formula; its left-hand side is kept
#' @param response_var The response column
#' @param predictor_vars The formula's predictors
#' @return A list: \code{data} (component scores and the response),
#'   \code{formula}, \code{reduction_model}, \code{n_components}, and
#'   \code{transform}, which refits the rotation inside a \code{tl_cv()}
#'   fold
#' @keywords internal
#' @noRd
tl_automl_pca_variant <- function(data, formula, response_var,
                                  predictor_vars) {
  numeric_vars <- tl_numeric_predictors(data, predictor_vars, "reduce")
  n_components <- min(5, ceiling(length(numeric_vars) / 2))
  pc_cols <- paste0("PC", seq_len(n_components))

  reduced <- tl_reduce_dimensions(
    data[, c(numeric_vars, response_var), drop = FALSE],
    response = response_var,
    method = "pca",
    n_components = n_components
  )
  formula_reduced <- stats::reformulate(pc_cols, response = formula[[2]])

  # Refit the rotation inside every fold. Scoring against a rotation
  # derived from the assessment rows themselves would put these
  # variants on the leaderboard with an advantage the baselines
  # never get.
  transform <- function(train_rows) {
    fold_reduction <- tl_model(
      train_rows[, numeric_vars, drop = FALSE],
      method = "pca"
    )

    list(
      formula = formula_reduced,
      apply = function(rows) {
        scores <- predict(
          fold_reduction,
          new_data = rows[, numeric_vars, drop = FALSE]
        )
        out <- scores[, pc_cols, drop = FALSE]
        out[[response_var]] <- rows[[response_var]]
        out
      }
    )
  }

  list(
    data = reduced$data,
    formula = formula_reduced,
    reduction_model = reduced$reduction_model,
    n_components = n_components,
    transform = transform
  )
}

#' Build AutoML's cluster-augmented candidates
#'
#' k-means is fitted on the formula's predictors only, for the reason given
#' in \code{tl_automl_pca_variant()}, and the assignment is added to the
#' formula's own terms. Fitted under the caller's formula, the candidates
#' saw the cluster column only through a \code{.}: an explicit right-hand
#' side left it out, and the clustered models predicted exactly what the
#' baselines did.
#'
#' @param data The training data
#' @param formula The caller's formula
#' @param response_var The response column
#' @param predictor_vars The formula's predictors
#' @param k The number of clusters
#' @return A list: \code{data} (the training data plus the cluster column),
#'   \code{formula}, \code{cluster_model}, \code{column}, \code{levels},
#'   and \code{transform}, which refits the centres inside a \code{tl_cv()}
#'   fold
#' @keywords internal
#' @noRd
tl_automl_cluster_variant <- function(data, formula, response_var,
                                      predictor_vars, k) {
  numeric_vars <- tl_numeric_predictors(data, predictor_vars, "cluster")
  column <- "cluster_kmeans"

  clustered <- tl_add_cluster_features(
    data[, c(numeric_vars, response_var), drop = FALSE],
    response = response_var,
    method = "kmeans", k = k
  )
  data_clustered <- data
  data_clustered[[column]] <- clustered[[column]]

  # update() cannot expand `.` itself, so the formula is expanded against
  # the data first -- the data without the cluster column, which would
  # otherwise be counted twice
  expanded <- stats::formula(
    stats::terms(formula, data = data, simplify = TRUE)
  )
  formula_clustered <- stats::update(expanded, . ~ . + cluster_kmeans)

  # Refit the centroids inside every fold, for the same reason as the PCA
  # variants
  transform <- function(train_rows) {
    fold_clusters <- tl_model(
      train_rows[, numeric_vars, drop = FALSE],
      method = "kmeans", k = k
    )
    fold_levels <- tl_cluster_levels(fold_clusters)

    list(
      formula = formula_clustered,
      apply = function(rows) {
        assignments <- predict(
          fold_clusters,
          new_data = rows[, numeric_vars, drop = FALSE]
        )
        rows[[column]] <- factor(assignments$cluster, levels = fold_levels)
        rows
      }
    )
  }

  list(
    data = data_clustered,
    formula = formula_clustered,
    cluster_model = attr(clustered, "cluster_model"),
    column = column,
    levels = levels(data_clustered[[column]]),
    transform = transform
  )
}

#' Auto ML: Automated Machine Learning Workflow
#'
#' Automatically explores multiple modeling approaches including
#' dimensionality reduction, clustering, and various supervised methods.
#' Returns the best performing model, scored by cross-validation where the
#' time budget allows.
#'
#' The PCA and cluster variants are built from the formula's predictors
#' only, so a column the formula leaves out (\code{y ~ . - id}) reaches no
#' candidate. The cluster variants add the cluster assignment to the
#' formula's terms.
#'
#' @param data A data frame
#' @param formula Model formula (for supervised learning)
#' @param task Task type: "classification", "regression", or "auto"
#'   (default), which reads it off the response. An explicit task has to
#'   agree with the response, since \code{tl_model()} fits a factor or
#'   character response as a classification and a numeric one as a
#'   regression.
#' @param use_reduction Whether to try dimensionality reduction (default: TRUE)
#' @param use_clustering Whether to add cluster features (default: TRUE)
#' @param time_budget Time budget in seconds (default: 300). The budget is
#'   checked between model fits, not during them: once a model starts
#'   training it runs to completion, because R cannot safely interrupt
#'   C-level code (randomForest, xgboost, e1071). A run can therefore
#'   overshoot the budget by the length of the last fit it started.
#'
#'   The budget gates the workflow as follows:
#'   \itemize{
#'     \item Baseline models: a tree, with linear regression for a numeric
#'       response or logistic regression for a two-class one. A random
#'       forest is added when \code{time_budget} is 30 or more.
#'     \item PCA and cluster variants, when enabled: each phase starts only
#'       if more than \code{max(5, 0.1 * time_budget)} seconds remain, and
#'       fits one variant per baseline method while at least
#'       \code{max(2, 0.05 * time_budget)} seconds remain.
#'     \item Advanced models (SVM and XGBoost for classification, ridge and
#'       lasso for regression): only when \code{time_budget} is 30 or more
#'       and more than 40\% of it remains when the phase starts.
#'     \item Scoring: a model is cross-validated when more than 30\% of the
#'       budget remains at the moment it is scored, and scored on its own
#'       training data otherwise. The leaderboard's \code{evaluation} column
#'       records which.
#'   }
#'
#'   The example below, with \code{time_budget = 10} on the three-class
#'   \code{iris}, fits a single tree and cross-validates it.
#' @param cv_folds Number of cross-validation folds (default: 5). Reducing
#'   this (e.g. to 2 or 3) is an effective way to stay closer to the time
#'   budget since CV is typically the most expensive step.
#' @param metric Evaluation metric (default: "accuracy" for classification,
#'   "rmse" for regression). Classification takes "accuracy", "precision",
#'   "recall", "sensitivity", "specificity", "f1", "auc" or "pr_auc";
#'   regression takes "rmse", "mse", "mae", "mape" or "rsq". It is checked
#'   before any model is fitted.
#' @return A list with class \code{"tidylearn_automl"} containing:
#'   \describe{
#'     \item{best_model}{The best tidylearn model object}
#'     \item{models}{Named list of all successfully trained models}
#'     \item{leaderboard}{Tibble ranking models by the chosen metric, with
#'       columns \code{model}, \code{score} and \code{evaluation}. The
#'       \code{evaluation} column records how each score was obtained --
#'       \code{"cv"} for cross-validated, \code{"train"} for training-set
#'       metrics, which are optimistic. Scores of different kinds are not
#'       directly comparable; a mixed leaderboard means the budget ran
#'       short of cross-validating every model.}
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
  formula <- tl_as_formula(formula)

  # Settled before the clock starts. Checked at the leaderboard, an
  # unrankable metric cost every fit before it was refused, and an
  # unrecognised task fitted the regression methods to a factor.
  response_var <- all.vars(formula)[1]
  if (!response_var %in% names(data)) {
    stop(
      "The formula's response, '", response_var,
      "', is not a column of `data`.",
      call. = FALSE
    )
  }
  y <- data[[response_var]]
  task <- tl_automl_task(task, y, response_var)

  n_classes <- length(unique(stats::na.omit(y)))

  if (task == "classification" && n_classes < 2) {
    stop(
      "Classification requires at least two observed classes in '",
      response_var, "'; found ", n_classes, ".",
      call. = FALSE
    )
  }

  metric <- tl_automl_metric(metric, task)

  # The predictors the formula names. The PCA and cluster variants are
  # built from these alone, as the baselines are.
  predictor_vars <- tl_formula_predictors(formula, data)

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

  # Helper: score a fitted model. Cross-validation when the budget allows,
  # otherwise training-set metrics. Training metrics are optimistic, so the
  # kind is recorded and reported on the leaderboard -- without it, models
  # scored in-sample would outrank cross-validated ones by overfitting.
  evaluate_model <- function(model, eval_data, eval_formula, method, label,
                             transform = NULL) {
    if (budget_left() > time_budget * 0.3) {
      cv <- safe_train(function() {
        tl_cv(eval_data, eval_formula, method = method,
              folds = cv_folds, metrics = metric, transform = transform)
      }, paste0(label, " CV"))

      if (!is.null(cv)) {
        return(list(result = cv, kind = "cv"))
      }
    }

    # Not routed through safe_train: a model that fitted should not be
    # discarded just because the budget expired before scoring it
    train_result <- tryCatch(
      tl_evaluate(model, metrics = metric),
      error = function(e) {
        message("    ", label, ": evaluation failed - ", e$message)
        NULL
      }
    )

    if (is.null(train_result)) {
      return(NULL)
    }

    list(result = train_result, kind = "train")
  }

  message("Starting Auto ML with task: ", task)
  message("Time budget: ", time_budget, " seconds")

  # Prepare candidate models
  models <- list()
  results <- list()
  eval_kinds <- character()

  # 1. Baseline models (ordered fast to slow)
  # Slow methods (forest, svm, xgboost) involve C code that cannot be

  # interrupted, so we only attempt them when the budget is generous.
  message("\n[1/4] Training baseline models...")
  fast_methods <- if (task == "classification") {
    # Logistic regression is binary-only here, so it is skipped rather
    # than fitted to a multiclass response it cannot represent
    if (n_classes == 2) c("tree", "logistic") else "tree"
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

    scored <- evaluate_model(model, data, formula, method, model_name)
    if (is.null(scored)) next

    models[[model_name]] <- model
    results[[model_name]] <- scored$result
    eval_kinds[[model_name]] <- scored$kind
  }

  # 2. Models with dimensionality reduction
  if (use_reduction && budget_left() > max(5, time_budget * 0.1)) {
    message("\n[2/4] Training models with dimensionality reduction...")

    tryCatch({
      reduced <- tl_automl_pca_variant(
        data, formula, response_var, predictor_vars
      )

      for (method in baseline_methods) {
        if (budget_left() < max(2, time_budget * 0.05)) break

        model_name <- paste0("pca_", method)
        message("  Training: ", model_name)

        result <- safe_train(function() {
          model <- tl_model(reduced$data, reduced$formula, method = method)
          model$reduction_info <- list(
            reduction_model = reduced$reduction_model,
            n_components = reduced$n_components
          )
          # Carry the projection so predict() can apply it to raw new data
          model$feature_transform <- list(
            kind = "pca",
            reduction_model = reduced$reduction_model,
            response = response_var
          )
          scored <- evaluate_model(
            model, data, reduced$formula, method, model_name,
            transform = reduced$transform
          )
          list(model = model, scored = scored)
        }, model_name)

        if (!is.null(result) && !is.null(result$scored)) {
          models[[model_name]] <- result$model
          results[[model_name]] <- result$scored$result
          eval_kinds[[model_name]] <- result$scored$kind
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
      # One cluster per observed class. Counting unique() values counted a
      # missing response as a class of its own.
      k <- if (task == "classification") n_classes else 3

      clustered <- tl_automl_cluster_variant(
        data, formula, response_var, predictor_vars, k = k
      )

      for (method in baseline_methods) {
        if (budget_left() < max(2, time_budget * 0.05)) break

        model_name <- paste0("clustered_", method)
        message("  Training: ", model_name)

        result <- safe_train(function() {
          model <- tl_model(clustered$data, clustered$formula, method = method)
          # Carry the clustering so predict() can assign new rows to it
          model$feature_transform <- list(
            kind = "cluster",
            cluster_model = clustered$cluster_model,
            column = clustered$column,
            levels = clustered$levels,
            response = response_var
          )
          scored <- evaluate_model(
            model, data, clustered$formula, method, model_name,
            transform = clustered$transform
          )
          list(model = model, scored = scored)
        }, model_name)

        if (!is.null(result) && !is.null(result$scored)) {
          models[[model_name]] <- result$model
          results[[model_name]] <- result$scored$result
          eval_kinds[[model_name]] <- result$scored$kind
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

      scored <- evaluate_model(model, data, formula, method, model_name)
      if (is.null(scored)) next

      models[[model_name]] <- model
      results[[model_name]] <- scored$result
      eval_kinds[[model_name]] <- scored$kind
    }
  }

  # Create leaderboard
  message("\n[*] Creating leaderboard...")
  leaderboard <- create_leaderboard(results, metric, task, eval_kinds)

  # Get best model
  if (nrow(leaderboard) == 0 || length(models) == 0) {
    warning("No models were successfully trained within the time budget.",
            call. = FALSE)
    best_model_name <- NA_character_
    best_model <- NULL
  } else if (all(is.na(leaderboard$score))) {
    warning(
      "Metric '", metric, "' could not be computed for any model, so the ",
      "leaderboard is unranked. Returning the first model trained.",
      call. = FALSE
    )
    best_model_name <- leaderboard$model[1]
    best_model <- models[[best_model_name]]
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
      best_model_name = best_model_name,
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

#' Pull a single metric value out of an evaluation result
#'
#' Auto ML collects results from two sources with different shapes:
#' \code{\link{tl_cv}} returns \code{list(folds, summary)} where
#' \code{summary} has \code{metric}/\code{mean}/\code{sd} columns, while
#' \code{\link{tl_evaluate}} returns a tibble with \code{metric}/
#' \code{value} columns.
#'
#' @param result An evaluation result
#' @param metric Name of the metric to extract
#' @return The metric value, or \code{NA_real_} if it is not present
#' @keywords internal
#' @noRd
extract_metric_score <- function(result, metric) {
  if (is.null(result)) {
    return(NA_real_)
  }

  pick <- function(df, value_col) {
    hit <- which(df$metric == metric)
    if (length(hit) == 0) NA_real_ else as.numeric(df[[value_col]][hit[1]])
  }

  # tl_evaluate() output -- checked before the list branches below because
  # a tibble is also a list
  if (is.data.frame(result) &&
        all(c("metric", "value") %in% names(result))) {
    return(pick(result, "value"))
  }

  if (is.list(result) && !is.data.frame(result)) {
    # tl_cv() output
    if ("summary" %in% names(result)) {
      summary_df <- result$summary
      if (is.data.frame(summary_df) &&
            all(c("metric", "mean") %in% names(summary_df))) {
        return(pick(summary_df, "mean"))
      }
      return(NA_real_)
    }

    # Plain named-list shapes
    if (metric %in% names(result)) {
      return(as.numeric(result[[metric]])[1])
    }
    if ("metrics" %in% names(result)) {
      nested <- result$metrics
      if (is.data.frame(nested) &&
            all(c("metric", "value") %in% names(nested))) {
        return(pick(nested, "value"))
      }
      val <- nested[[metric]]
      return(if (is.null(val)) NA_real_ else as.numeric(val)[1])
    }
  }

  NA_real_
}

#' Create leaderboard from results
#' @keywords internal
#' @noRd
create_leaderboard <- function(results, metric, task, eval_kinds = NULL) {
  if (length(results) == 0) {
    return(tibble::tibble(
      model = character(0), score = numeric(0), evaluation = character(0)
    ))
  }

  scores <- vapply(results, function(r) {
    val <- extract_metric_score(r, metric)
    if (is.nan(val)) NA_real_ else val
  }, numeric(1))

  kinds <- if (is.null(eval_kinds)) {
    rep(NA_character_, length(scores))
  } else {
    unname(eval_kinds[names(scores)])
  }

  leaderboard <- tibble::tibble(
    model = names(scores),
    score = scores,
    evaluation = kinds
  )

  # Sort: ascending for error metrics, descending for accuracy metrics.
  # A guessed direction for a metric tidylearn does not know hands back
  # the worst model as the winner whenever the guess is wrong, so refuse
  # instead. tl_auto_ml() checks its metric up front; this guards direct
  # callers.
  higher_better <- tl_metric_higher_better(metric)

  if (is.na(higher_better)) {
    known <- c(tl_known_metrics(TRUE), tl_known_metrics(FALSE))
    stop(
      "Cannot rank models by '", metric, "': tidylearn does not know ",
      "whether higher or lower is better. Use one of: ",
      paste(sort(known), collapse = ", "), ".",
      call. = FALSE
    )
  }

  if (higher_better) {
    leaderboard <- leaderboard |> dplyr::arrange(dplyr::desc(.data$score))
  } else {
    leaderboard <- leaderboard |> dplyr::arrange(.data$score)
  }

  leaderboard
}

#' Print auto ML results
#' @param x A tidylearn_automl object
#' @param ... Additional arguments (ignored)
#' @return The input object \code{x}, returned invisibly.
#' @export
#' @examples
#' \donttest{
#' result <- tl_auto_ml(iris, Species ~ .,
#'   time_budget = 10,
#'   use_reduction = FALSE,
#'   use_clustering = FALSE,
#'   cv_folds = 2)
#'
#' # The leaderboard, the winner and the metric it was ranked on
#' print(result)
#' }
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
#' @param max_components Maximum number of PCA components to keep (default:
#'   5), or fewer if the data has fewer numeric columns
#' @param k_range Range of k values for clustering (default: 2:6). Each is
#'   a whole number from 2 to one less than the number of rows, the range a
#'   silhouette is defined over.
#' @return A list with class \code{"tidylearn_eda"} containing:
#'   \describe{
#'     \item{data}{The original data frame.}
#'     \item{response}{The response variable name, or \code{NULL}.}
#'     \item{pca}{The fitted PCA model, keeping the first
#'       \code{max_components} components, as
#'       \code{prcomp(rank. = max_components)} would.}
#'     \item{optimal_k}{List with optimal cluster count results.}
#'     \item{kmeans}{The fitted k-means model.}
#'     \item{hclust}{The fitted hierarchical clustering model.}
#'     \item{summary}{List with \code{n_obs}, \code{n_vars},
#'       \code{n_components} (the number kept), and \code{best_k}.}
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
  if (!is.numeric(max_components) || length(max_components) != 1L ||
        is.na(max_components) || max_components < 1 ||
        max_components != round(max_components)) {
    stop(
      "'max_components' must be a single whole number of at least 1; got ",
      tl_describe_value(max_components), ".",
      call. = FALSE
    )
  }

  # A silhouette compares each point's cluster with the nearest other one,
  # so it needs at least two clusters and one row more than clusters. k = 1
  # failed inside the scoring with "incorrect number of dimensions".
  max_k <- nrow(data) - 1
  if (!is.numeric(k_range) || length(k_range) == 0 || anyNA(k_range) ||
        any(k_range != round(k_range)) || any(k_range < 2) ||
        any(k_range > max_k)) {
    stop(
      "'k_range' must hold whole numbers from 2 to ", max_k, "; got ",
      if (length(k_range) == 0) "nothing" else paste(k_range, collapse = ", "),
      ". A silhouette needs at least two clusters, and fewer clusters ",
      "than rows.",
      call. = FALSE
    )
  }

  message("Running Exploratory Data Analysis...")

  # 1. Dimensionality Reduction
  message("[1/4] PCA analysis...")
  predictor_data <- if (!is.null(response)) {
    data |> dplyr::select(-dplyr::all_of(response))
  } else {
    data
  }

  pca_result <- tl_model(predictor_data, method = "pca")
  n_components <- min(
    as.integer(max_components), ncol(pca_result$fit$model$rotation)
  )
  pca_result <- tl_keep_components(pca_result, n_components)

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
        n_components = n_components,
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

  cat("PCA Variance Explained (first ", x$summary$n_components,
      " components):\n", sep = "")
  print(x$pca$fit$variance_explained)

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

  if (!all(c("PC1", "PC2") %in% names(pca_scores))) {
    stop(
      "plot() draws the first two principal components, but this analysis ",
      "kept ", x$summary$n_components, ". Run tl_explore() with ",
      "max_components of at least 2, on data with two or more numeric ",
      "columns.",
      call. = FALSE
    )
  }

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

#' Keep the leading components of a fitted PCA model
#'
#' What \code{prcomp(rank. = n)} returns, applied after the fit: the
#' rotation, the scores and the loadings keep their first \code{n}
#' components, and the standard deviations keep all of them, as prcomp()
#' does. The variance table keeps the first \code{n} rows, whose cumulative
#' share is still of the total variance.
#'
#' @param pca_model A \code{tl_model(method = "pca")} model
#' @param n The number of components to keep
#' @return \code{pca_model}, trimmed
#' @keywords internal
#' @noRd
tl_keep_components <- function(pca_model, n) {
  keep <- seq_len(n)
  pc_cols <- paste0("PC", keep)

  fit <- pca_model$fit
  fit$model$rotation <- fit$model$rotation[, keep, drop = FALSE]
  fit$model$x <- fit$model$x[, keep, drop = FALSE]
  fit$scores <- fit$scores[, c(".obs_id", pc_cols)]
  fit$loadings <- fit$loadings[, c("variable", pc_cols)]
  fit$variance_explained <- fit$variance_explained[keep, ]

  pca_model$fit <- fit
  pca_model$spec$n_components <- n
  pca_model
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
#' Use unsupervised pre-training before supervised learning: the
#' predictors the formula names are projected onto their principal
#' components, and the supervised model is fitted on the component scores.
#'
#' @param data Training data
#' @param formula Model formula, or a string that parses as one. The PCA is
#'   fitted on the numeric predictors it names.
#' @param pretrain_method Pre-training method. Only \code{"pca"} is
#'   available: \code{predict()} has to project new rows, and PCA is the
#'   reduction that can.
#' @param supervised_method Supervised learning method (default:
#'   \code{"tree"}, which handles both regression and classification with
#'   any number of classes). \code{"logistic"} is binary-only and errors
#'   on a response with more than two levels.
#' @param ... Additional arguments passed to
#'   \code{\link{tl_reduce_dimensions}}, such as \code{n_components}
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
#'   pretrain_method = "pca", supervised_method = "tree")
#' }
tl_transfer_learning <- function(data, formula, pretrain_method = "pca",
                                 supervised_method = "tree", ...) {
  formula <- tl_as_formula(formula)

  # "autoencoder" was documented and never implemented, and an MDS fit has
  # no projection for new rows, so predict() on it always failed
  if (!identical(pretrain_method, "pca")) {
    stop(
      "'pretrain_method' must be \"pca\", the one pre-training method that ",
      "can project the new rows predict() is given; got ",
      tl_describe_value(pretrain_method), ".",
      call. = FALSE
    )
  }

  message("Transfer Learning Workflow")
  message("==========================")

  response_var <- all.vars(formula)[1]
  predictor_vars <- tl_formula_predictors(formula, data)

  # Phase 1: Unsupervised pre-training, on the predictors the formula names
  message("[Phase 1] Unsupervised pre-training with ", pretrain_method, "...")
  pretrain_model <- tl_reduce_dimensions(
    data[, c(predictor_vars, response_var), drop = FALSE],
    response = response_var,
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

  # The component scores replace the columns the formula's right-hand side
  # names, so the model is fitted on the scores under the same response.
  # Fitting the caller's formula worked only for `.`: an explicit
  # right-hand side named columns the scores no longer had.
  supervised_formula <- stats::reformulate(
    setdiff(names(supervised_data), response_var),
    response = formula[[2]]
  )
  supervised_model <- tl_model(
    supervised_data, supervised_formula,
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
#'   pretrain_method = "pca", supervised_method = "tree")
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
