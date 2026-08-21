#' @title Model Pipeline Functions for tidylearn
#' @name tidylearn-pipeline
#' @description Functions for creating end-to-end model pipelines
#' @importFrom stats formula
#' @importFrom dplyr filter select mutate
NULL

#' Fill in and validate a pipeline evaluation specification
#'
#' @param evaluation A named list of evaluation settings, or NULL.
#' @param is_classification Whether the response is categorical.
#' @return The list, with every recognised setting present.
#' @keywords internal
#' @noRd
merge_evaluation_spec <- function(evaluation, is_classification) {
  defaults <- if (is_classification) {
    list(
      metrics = c("accuracy", "precision", "recall", "f1", "auc"),
      validation = "cv",
      cv_folds = 5,
      train_prop = 0.7,
      best_metric = "f1"
    )
  } else {
    list(
      metrics = c("rmse", "mae", "rsq", "mape"),
      validation = "cv",
      cv_folds = 5,
      train_prop = 0.7,
      best_metric = "rmse"
    )
  }

  if (is.null(evaluation)) {
    return(defaults)
  }

  if (!is.list(evaluation) ||
        is.null(names(evaluation)) ||
        any(names(evaluation) == "")) {
    stop(
      "'evaluation' must be a named list: ",
      paste(names(defaults), collapse = ", "), ".",
      call. = FALSE
    )
  }

  unknown <- setdiff(names(evaluation), names(defaults))
  if (length(unknown) > 0) {
    stop(
      "Unknown evaluation setting(s): ",
      paste(unknown, collapse = ", "),
      ". Available settings: ", paste(names(defaults), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  evaluation <- utils::modifyList(defaults, evaluation)

  if (!evaluation$validation %in% c("cv", "split")) {
    stop(
      "evaluation$validation must be \"cv\" or \"split\"; got \"",
      evaluation$validation, "\".",
      call. = FALSE
    )
  }

  if (!evaluation$best_metric %in% evaluation$metrics) {
    stop(
      "evaluation$best_metric (\"", evaluation$best_metric,
      "\") must be one of evaluation$metrics: ",
      paste(evaluation$metrics, collapse = ", "), ".",
      call. = FALSE
    )
  }

  evaluation
}

#' Fill in and validate a pipeline preprocessing specification
#'
#' @param preprocessing A named list of preprocessing switches, or NULL.
#' @return The list, with every recognised step present.
#' @keywords internal
#' @noRd
merge_preprocessing_spec <- function(preprocessing) {
  defaults <- list(
    impute_missing = TRUE,
    standardize = TRUE,
    dummy_encode = TRUE
  )

  if (is.null(preprocessing)) {
    return(defaults)
  }

  if (!is.list(preprocessing) ||
        is.null(names(preprocessing)) ||
        any(names(preprocessing) == "")) {
    stop(
      "'preprocessing' must be a named list of steps: ",
      paste(names(defaults), collapse = ", "), ".",
      call. = FALSE
    )
  }

  # An unrecognised name would otherwise be accepted and silently do
  # nothing, which reads as a step that ran
  unknown <- setdiff(names(preprocessing), names(defaults))
  if (length(unknown) > 0) {
    stop(
      "Unknown preprocessing step(s): ",
      paste(unknown, collapse = ", "),
      ". Available steps: ", paste(names(defaults), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  not_flag <- names(preprocessing)[
    !vapply(preprocessing, function(x) {
      is.logical(x) && length(x) == 1L && !is.na(x)
    }, logical(1))
  ]
  if (length(not_flag) > 0) {
    stop(
      "Preprocessing step(s) must be TRUE or FALSE: ",
      paste(not_flag, collapse = ", "), ".",
      call. = FALSE
    )
  }

  utils::modifyList(defaults, preprocessing)
}

#' Create a modeling pipeline
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the model
#' @param preprocessing A list of preprocessing steps
#' @param models A list of models to train
#' @param evaluation A list of evaluation criteria
#' @param ... Additional arguments
#' @return A \code{tidylearn_pipeline} object (S3 list) with components
#'   \code{$formula}, \code{$data}, \code{$preprocessing},
#'   \code{$models}, \code{$evaluation}, and \code{$results}
#'   (initially \code{NULL}; populated after \code{\link{tl_run_pipeline}}).
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(tree = list(method = "tree")))
#' print(pipe)
#' }
#' @export
tl_pipeline <- function(data, formula,
                        preprocessing = NULL,
                        models = NULL,
                        evaluation = NULL, ...) {
  # Fill in whichever preprocessing steps the caller left unnamed. A
  # partial list used to reach `if (preprocessing$standardize)` as NULL
  # and fail with "argument is of length zero" inside tl_run_pipeline(),
  # long after the mistake was made.
  preprocessing <- merge_preprocessing_spec(preprocessing)

  # Create default models if not provided
  if (is.null(models)) {
    # Determine if classification or regression
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]
    is_classification <- is.factor(y) || is.character(y)

    if (is_classification) {
      models <- list(
        logistic = list(method = "logistic"),
        tree = list(method = "tree"),
        forest = list(method = "forest", ntree = 500)
      )
    } else {
      models <- list(
        linear = list(method = "linear"),
        lasso = list(method = "lasso"),
        forest = list(method = "forest", ntree = 500)
      )
    }
  }

  # Fill in whichever evaluation settings the caller left unnamed, for
  # the same reason as the preprocessing spec above
  response_var <- all.vars(formula)[1]
  evaluation <- merge_evaluation_spec(
    evaluation,
    is_classification = is.factor(data[[response_var]]) ||
      is.character(data[[response_var]])
  )

  # Create pipeline object
  pipeline <- list(
    formula = formula,
    data = data,
    preprocessing = preprocessing,
    models = models,
    evaluation = evaluation,
    results = NULL
  )

  class(pipeline) <- "tidylearn_pipeline"

  pipeline
}

#' Run a tidylearn pipeline
#'
#' @param pipeline A tidylearn pipeline object
#' @param verbose Logical; whether to print progress
#' @return The input \code{tidylearn_pipeline} object with its
#'   \code{$results} component populated. Results include
#'   \code{$processed_data} (the training data after preprocessing),
#'   \code{$preprocessing_stats} (the medians, modes, centres and scales
#'   learned from the training data, replayed by
#'   \code{\link{tl_predict_pipeline}}), \code{$model_results} (a named
#'   list of per-model fits and metrics), \code{$best_model_name},
#'   \code{$best_model} (the winning \code{tidylearn_model}), and
#'   \code{$metric_values}.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(tree = list(method = "tree")),
#'   evaluation = list(metrics = "accuracy", validation = "cv",
#'     cv_folds = 2, best_metric = "accuracy"))
#' pipe <- tl_run_pipeline(pipe, verbose = FALSE)
#' }
#' @export
tl_run_pipeline <- function(pipeline, verbose = TRUE) {
  # Check if pipeline is valid
  if (!inherits(pipeline, "tidylearn_pipeline")) {
    stop("Input must be a tidylearn pipeline object", call. = FALSE)
  }

  # Extract components
  data <- pipeline$data
  formula <- pipeline$formula
  preprocessing <- pipeline$preprocessing
  models <- pipeline$models
  evaluation <- pipeline$evaluation

  # Without names the training loop below silently does nothing, leaving
  # an empty leaderboard and an unhelpful downstream error
  if (length(models) == 0 || is.null(names(models)) ||
        any(!nzchar(names(models)))) {
    stop(
      "`models` must be a named list of model configurations, e.g. ",
      "list(linear = list(method = \"linear\")).",
      call. = FALSE
    )
  }

  # Apply preprocessing
  if (verbose) {
    message("Applying preprocessing steps...")
  }

  processed_data <- data

  # Preprocessing statistics learned from the training data. These are
  # recorded on the scale each step sees, so tl_predict_pipeline() can
  # replay the same transformation on raw new data.
  preprocessing_stats <- list(
    medians = list(),
    modes = list(),
    center = list(),
    scale = list()
  )

  if (preprocessing$impute_missing) {
    if (verbose) {
      message("  - Imputing missing values")
    }

    # Simple imputation for numeric and categorical variables
    for (col in names(processed_data)) {
      if (is.numeric(processed_data[[col]])) {
        # Record the median even when this column is complete -- new data
        # may still have gaps here
        med <- median(processed_data[[col]], na.rm = TRUE)
        preprocessing_stats$medians[[col]] <- med

        na_idx <- is.na(processed_data[[col]])
        if (any(na_idx)) {
          processed_data[[col]][na_idx] <- med
        }
      } else if (is.factor(processed_data[[col]]) ||
                   is.character(processed_data[[col]])) {
        # Calculate mode
        if (is.factor(processed_data[[col]])) {
          tab <- table(processed_data[[col]])
        } else {
          tab <- table(processed_data[[col]], useNA = "no")
        }
        mode_val <- if (length(tab) > 0) {
          names(tab)[which.max(tab)]
        } else {
          NA_character_
        }
        preprocessing_stats$modes[[col]] <- mode_val

        na_idx <- is.na(processed_data[[col]])
        if (any(na_idx) && !is.na(mode_val)) {
          processed_data[[col]][na_idx] <- mode_val
        }
      }
    }
  }

  if (preprocessing$standardize) {
    if (verbose) {
      message("  - Standardizing numeric features")
    }

    # Identify numeric columns (excluding response)
    response_var <- all.vars(formula)[1]
    numeric_cols <- sapply(processed_data, is.numeric)
    numeric_cols[response_var] <- FALSE  # Don't standardize response

    # Standardize each numeric column, recording the centre and scale
    for (col in names(processed_data)[numeric_cols]) {
      col_mean <- mean(processed_data[[col]], na.rm = TRUE)
      col_sd <- stats::sd(processed_data[[col]], na.rm = TRUE)

      # A constant column would divide by zero; centre it only
      if (is.na(col_sd) || col_sd == 0) {
        col_sd <- 1
      }

      preprocessing_stats$center[[col]] <- col_mean
      preprocessing_stats$scale[[col]] <- col_sd

      processed_data[[col]] <- (processed_data[[col]] - col_mean) / col_sd
    }
  }

  if (preprocessing$dummy_encode) {
    if (verbose) {
      message("  - Creating dummy variables for categorical features")
    }

    # Let model.matrix handle this during model fitting
    # We don't modify the processed_data here
  }

  # Set up validation strategy
  if (evaluation$validation == "cv") {
    cv_folds <- evaluation$cv_folds

    if (verbose) {
      message("Setting up ", cv_folds, "-fold cross-validation")
    }

    # Create cross-validation splits
    cv_splits <- rsample::vfold_cv(processed_data, v = cv_folds)
  } else if (evaluation$validation == "split") {
    train_prop <- evaluation$train_prop
    if (is.null(train_prop)) train_prop <- 0.8

    if (verbose) {
      message("Setting up train/test split (", train_prop * 100, "% / ",
              (1 - train_prop) * 100, "%)")
    }

    # Create a single train/test split
    train_idx <- sample(
      nrow(processed_data),
      round(train_prop * nrow(processed_data))
    )
    train_data <- processed_data[train_idx, ]
    test_data <- processed_data[-train_idx, ]
  }

  # Train and evaluate models
  model_results <- list()

  for (model_name in names(models)) {
    if (verbose) {
      message("Training model: ", model_name)
    }

    # Extract model configuration
    model_config <- models[[model_name]]
    method <- model_config$method

    # Remove method from config to use rest as parameters
    model_params <- model_config[names(model_config) != "method"]

    if (evaluation$validation == "cv") {
      # Cross-validation approach
      cv_results <- list()

      for (i in 1:cv_folds) {
        if (verbose) {
          message("  - Fold ", i, "/", cv_folds)
        }

        # Get training and testing data for this fold
        train_fold <- rsample::analysis(cv_splits$splits[[i]])
        test_fold <- rsample::assessment(cv_splits$splits[[i]])

        # Fit model on training fold
        model_args <- c(
          list(
            data = train_fold,
            formula = formula,
            method = method
          ),
          model_params
        )

        fold_model <- do.call(tl_model, model_args)

        # Evaluate on test fold
        fold_metrics <- tl_evaluate(
          fold_model, test_fold,
          metrics = evaluation$metrics
        )

        # Store fold results
        cv_results[[i]] <- list(
          model = fold_model,
          metrics = fold_metrics
        )
      }

      # Calculate average metrics across folds
      all_metrics <- do.call(rbind, lapply(cv_results, function(x) x$metrics))

      avg_metrics <- all_metrics %>%
        dplyr::group_by(.data$metric) %>%
        dplyr::summarize(
          mean_value = mean(.data$value, na.rm = TRUE),
          sd_value = sd(.data$value, na.rm = TRUE)
        )

      # Train final model on all data
      final_model_args <- c(
        list(
          data = processed_data,
          formula = formula,
          method = method
        ),
        model_params
      )

      final_model <- do.call(tl_model, final_model_args)

      # Store results
      model_results[[model_name]] <- list(
        model = final_model,
        cv_results = cv_results,
        avg_metrics = avg_metrics
      )

      if (verbose) {
        for (i in seq_len(nrow(avg_metrics))) {
          metric <- avg_metrics$metric[i]
          mean_val <- avg_metrics$mean_value[i]
          sd_val <- avg_metrics$sd_value[i]

          message("    ", metric, ": ", round(mean_val, 4),
                  " (+/-", round(sd_val, 4), ")")
        }
      }
    } else if (evaluation$validation == "split") {
      # Train/test split approach
      # Fit model on training data
      model_args <- c(
        list(
          data = train_data,
          formula = formula,
          method = method
        ),
        model_params
      )

      split_model <- do.call(tl_model, model_args)

      # Evaluate on test data
      test_metrics <- tl_evaluate(
        split_model, test_data,
        metrics = evaluation$metrics
      )

      # Store results
      model_results[[model_name]] <- list(
        model = split_model,
        test_metrics = test_metrics
      )

      if (verbose) {
        for (i in seq_len(nrow(test_metrics))) {
          metric <- test_metrics$metric[i]
          value <- test_metrics$value[i]

          message("    ", metric, ": ", round(value, 4))
        }
      }
    }
  }

  # Select the best model
  best_metric <- evaluation$best_metric

  if (verbose) {
    message("Selecting best model based on ", best_metric)
  }

  # Extract metric values for each model
  metric_values <- sapply(
    names(model_results),
    function(model_name) {
      result <- model_results[[model_name]]

      if (evaluation$validation == "cv") {
        metric_row <- result$avg_metrics$metric ==
          best_metric
        if (any(metric_row)) {
          val <- result$avg_metrics$mean_value[metric_row]
          if (is.nan(val) || is.na(val)) NA_real_ else val
        } else {
          NA_real_
        }
      } else {
        metric_row <- result$test_metrics$metric ==
          best_metric
        if (any(metric_row)) {
          val <- result$test_metrics$value[metric_row]
          if (is.nan(val) || is.na(val)) NA_real_ else val
        } else {
          NA_real_
        }
      }
    }
  )

  # Determine if higher or lower is better for this metric
  metrics_higher_better <- c(
    "accuracy", "precision", "recall",
    "f1", "auc", "rsq"
  )
  is_higher_better <- best_metric %in%
    metrics_higher_better

  # Find best model (use na.rm-safe which.max/which.min)
  valid_values <- !is.na(metric_values)
  if (!any(valid_values)) {
    best_idx <- 1L
    warning(
      "Could not determine best model from metric '",
      best_metric, "' -- all values NA. Using first model.",
      call. = FALSE
    )
  } else if (is_higher_better) {
    best_idx <- which.max(metric_values)
  } else {
    best_idx <- which.min(metric_values)
  }

  best_model_name <- names(model_results)[best_idx]
  best_model <- model_results[[best_model_name]]$model

  if (verbose) {
    message("Best model: ", best_model_name, " with ", best_metric, " = ",
            round(metric_values[best_idx], 4))
  }

  # Update pipeline with results
  pipeline$results <- list(
    processed_data = processed_data,
    preprocessing_stats = preprocessing_stats,
    model_results = model_results,
    best_model_name = best_model_name,
    best_model = best_model,
    metric_values = metric_values
  )

  pipeline
}

#' Get the best model from a pipeline
#'
#' @param pipeline A tidylearn pipeline object with results
#' @return The best \code{tidylearn_model} object from the pipeline,
#'   selected by the metric specified in \code{evaluation$best_metric}.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(tree = list(method = "tree")),
#'   evaluation = list(metrics = "accuracy", validation = "cv",
#'     cv_folds = 2, best_metric = "accuracy"))
#' pipe <- tl_run_pipeline(pipe, verbose = FALSE)
#' best <- tl_get_best_model(pipe)
#' }
#' @export
tl_get_best_model <- function(pipeline) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop(
      "Pipeline has not been run yet. Use tl_run_pipeline() first.",
      call. = FALSE
    )
  }

  pipeline$results$best_model
}

#' Compare models from a pipeline
#'
#' @param pipeline A tidylearn pipeline object with results
#' @param metrics Character vector of metrics to compare
#'   (if NULL, uses all available)
#' @return A \code{\link[ggplot2]{ggplot}} object showing a faceted bar
#'   chart comparing metric values across models, with the best model
#'   highlighted.
#' @importFrom ggplot2 ggplot aes geom_col facet_wrap labs theme_minimal
#' @export
tl_compare_pipeline_models <- function(pipeline, metrics = NULL) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop(
      "Pipeline has not been run yet. Use tl_run_pipeline() first.",
      call. = FALSE
    )
  }

  # Extract model results
  model_results <- pipeline$results$model_results

  # Create data frame for plotting
  comparison_data <- NULL

  for (model_name in names(model_results)) {
    result <- model_results[[model_name]]

    if (!is.null(pipeline$evaluation) &&
          pipeline$evaluation$validation == "cv") {
      # Get from average metrics
      model_metrics <- result$avg_metrics

      # Filter metrics if specified
      if (!is.null(metrics)) {
        model_metrics <- model_metrics[model_metrics$metric %in% metrics, ]
      }

      # Add model name
      model_metrics$model <- model_name

      # Rename columns
      names(model_metrics)[names(model_metrics) == "mean_value"] <- "value"

      # Add to comparison data
      if (is.null(comparison_data)) {
        comparison_data <- model_metrics
      } else {
        comparison_data <- rbind(comparison_data, model_metrics)
      }
    } else {
      # Get from test metrics
      model_metrics <- result$test_metrics

      # Filter metrics if specified
      if (!is.null(metrics)) {
        model_metrics <- model_metrics[model_metrics$metric %in% metrics, ]
      }

      # Add model name
      model_metrics$model <- model_name

      # Add to comparison data
      if (is.null(comparison_data)) {
        comparison_data <- model_metrics
      } else {
        comparison_data <- rbind(comparison_data, model_metrics)
      }
    }
  }

  # Add highlight for best model
  comparison_data$is_best <-
    comparison_data$model ==
    pipeline$results$best_model_name

  # Determine which metrics are "higher is better"
  metrics_higher_better <- c(
    "accuracy", "precision", "recall",
    "f1", "auc", "rsq"
  )
  comparison_data$higher_better <-
    comparison_data$metric %in% metrics_higher_better

  # Create comparison plot
  p <- ggplot2::ggplot(
    comparison_data,
    ggplot2::aes(
      x = model,
      y = value,
      fill = is_best
    )
  ) +
    ggplot2::geom_col() +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::geom_text(
      ggplot2::aes(label = round(value, 3)),
      vjust = -0.5,
      size = 3
    ) +
    ggplot2::scale_fill_manual(values = c("steelblue", "darkred")) +
    ggplot2::labs(
      title = "Model Comparison",
      subtitle = "Best model highlighted in red",
      x = "Model",
      y = "Metric Value",
      fill = "Best Model"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )

  p
}

#' Make predictions using a pipeline
#'
#' @param pipeline A tidylearn pipeline object with results
#' @param new_data A data frame containing the new data
#' @param type Type of prediction (default: "response")
#' @param model_name Name of model to use (if NULL, uses the best model)
#' @param ... Additional arguments passed to predict
#' @return A \link[tibble]{tibble} with a \code{.pred} column containing
#'   predictions from the selected (or best) pipeline model, after
#'   applying the same preprocessing steps used during training.
#' @export
tl_predict_pipeline <- function(pipeline,
                                new_data,
                                type = "response",
                                model_name = NULL,
                                ...) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop(
      "Pipeline has not been run yet. Use tl_run_pipeline() first.",
      call. = FALSE
    )
  }

  # Determine which model to use
  if (is.null(model_name)) {
    # Use best model
    model <- pipeline$results$best_model
  } else {
    # Use specified model
    if (!model_name %in% names(pipeline$results$model_results)) {
      stop("Model not found in pipeline: ", model_name, call. = FALSE)
    }
    model <- pipeline$results$model_results[[model_name]]$model
  }

  # Apply same preprocessing steps to new data
  processed_new_data <- new_data

  if (!is.null(pipeline$preprocessing)) {
    stats_learned <- pipeline$results$preprocessing_stats

    needs_stats <- isTRUE(pipeline$preprocessing$impute_missing) ||
      isTRUE(pipeline$preprocessing$standardize)

    if (needs_stats && is.null(stats_learned)) {
      stop(
        "This pipeline was run by an older version of tidylearn that did ",
        "not record preprocessing statistics, so new data cannot be ",
        "transformed consistently. Re-run it with tl_run_pipeline().",
        call. = FALSE
      )
    }

    if (isTRUE(pipeline$preprocessing$impute_missing)) {
      # Impute with the statistics learned from the raw training data
      for (col in names(processed_new_data)) {
        na_idx <- is.na(processed_new_data[[col]])
        if (!any(na_idx)) next

        if (is.numeric(processed_new_data[[col]])) {
          med <- stats_learned$medians[[col]]
          if (is.null(med)) {
            med <- median(processed_new_data[[col]], na.rm = TRUE)
          }
          processed_new_data[[col]][na_idx] <- med
        } else if (is.factor(processed_new_data[[col]]) ||
                     is.character(processed_new_data[[col]])) {
          mode_val <- stats_learned$modes[[col]]
          if (is.null(mode_val) || is.na(mode_val)) next
          processed_new_data[[col]][na_idx] <- mode_val
        }
      }
    }

    if (isTRUE(pipeline$preprocessing$standardize)) {
      # Standardize exactly the columns that were standardized during
      # training, using the training centre and scale
      for (col in names(stats_learned$center)) {
        if (!col %in% names(processed_new_data)) next

        processed_new_data[[col]] <-
          (processed_new_data[[col]] - stats_learned$center[[col]]) /
          stats_learned$scale[[col]]
      }
    }
  }

  # Make predictions
  predict(model, processed_new_data, type = type, ...)
}

#' Save a pipeline to disk
#'
#' @param pipeline A tidylearn pipeline object
#' @param file Path to save the pipeline
#' @return Called for its side effect of saving to disk; returns
#'   \code{NULL} invisibly.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' tl_save_pipeline(pipe, tempfile(fileext = ".rds"))
#' }
#' @export
tl_save_pipeline <- function(pipeline, file) {
  # Validate input
  if (!inherits(pipeline, "tidylearn_pipeline")) {
    stop("Input must be a tidylearn pipeline object", call. = FALSE)
  }

  # Save as RDS
  saveRDS(pipeline, file = file)

  invisible(NULL)
}

#' Load a pipeline from disk
#'
#' @param file Path to the pipeline file
#' @return A \code{tidylearn_pipeline} object previously saved with
#'   \code{\link{tl_save_pipeline}}.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' f <- tempfile(fileext = ".rds")
#' tl_save_pipeline(pipe, f)
#' pipe2 <- tl_load_pipeline(f)
#' }
#' @export
tl_load_pipeline <- function(file) {
  # Load RDS
  pipeline <- readRDS(file)

  # Validate
  if (!inherits(pipeline, "tidylearn_pipeline")) {
    stop("Loaded object is not a tidylearn pipeline", call. = FALSE)
  }

  pipeline
}

#' Print a tidylearn pipeline
#'
#' @param x A tidylearn pipeline object
#' @param ... Additional arguments (not used)
#' @return The input pipeline object \code{x}, returned invisibly.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' print(pipe)
#' }
#' @export
print.tidylearn_pipeline <- function(x, ...) {
  # Extract pipeline components
  formula <- x$formula
  data_dims <- dim(x$data)
  preprocessing <- names(x$preprocessing)[sapply(x$preprocessing, isTRUE)]
  models <- names(x$models)

  # Print pipeline summary
  cat("Tidylearn Pipeline\n")
  cat("=================\n")
  cat("Formula:", deparse(formula), "\n")
  cat("Data:", data_dims[1], "observations,", data_dims[2], "variables\n")
  cat("Preprocessing:", paste(preprocessing, collapse = ", "), "\n")
  cat("Models:", paste(models, collapse = ", "), "\n")

  # Print evaluation if available
  if (!is.null(x$evaluation)) {
    validation <- x$evaluation$validation
    metrics <- paste(x$evaluation$metrics, collapse = ", ")

    cat("Evaluation: ", validation)
    if (validation == "cv") {
      cat(" (", x$evaluation$cv_folds, " folds)", sep = "")
    }
    cat("\n")
    cat("Metrics:", metrics, "\n")
    cat("Best metric:", x$evaluation$best_metric, "\n")
  }

  # Print results if available
  if (!is.null(x$results)) {
    cat("\nResults\n")
    cat("=======\n")
    cat("Best model:", x$results$best_model_name, "\n")

    # Print metric values
    cat("Performance:\n")
    for (i in seq_along(x$results$metric_values)) {
      m_name <- names(x$results$metric_values)[i]
      metric_value <- x$results$metric_values[i]
      is_best <- if (m_name == x$results$best_model_name) " (best)" else ""

      cat(
        "  ", m_name, ": ", x$evaluation$best_metric, " = ",
        round(metric_value, 4), is_best, "\n",
        sep = ""
      )
    }
  }

  invisible(x)
}

#' Summarize a tidylearn pipeline
#'
#' @param object A tidylearn pipeline object
#' @param ... Additional arguments (not used)
#' @return The input pipeline \code{object}, returned invisibly. Called
#'   for its side effect of printing detailed pipeline and model results.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' summary(pipe)
#' }
#' @export
summary.tidylearn_pipeline <- function(object, ...) {
  # If no results, just print the pipeline
  if (is.null(object$results)) {
    print(object)
    return(invisible(object))
  }

  # Print pipeline info
  print(object)

  # Print more detailed results
  cat("\nDetailed Results\n")
  cat("===============\n")

  for (model_name in names(object$results$model_results)) {
    cat("\nModel:", model_name, "\n")

    result <- object$results$model_results[[model_name]]

    if (!is.null(object$evaluation) && object$evaluation$validation == "cv") {
      # Print average metrics with standard deviation
      cat("Cross-validation metrics:\n")
      for (i in seq_len(nrow(result$avg_metrics))) {
        metric <- result$avg_metrics$metric[i]
        mean_val <- result$avg_metrics$mean_value[i]
        sd_val <- result$avg_metrics$sd_value[i]

        cat("  ", metric, ": ", round(mean_val, 4),
            " (+/-", round(sd_val, 4), ")", "\n", sep = "")
      }
    } else {
      # Print test metrics
      cat("Test metrics:\n")
      for (i in seq_len(nrow(result$test_metrics))) {
        metric <- result$test_metrics$metric[i]
        value <- result$test_metrics$value[i]

        cat("  ", metric, ": ", round(value, 4), "\n", sep = "")
      }
    }
  }

  # Summary of best model
  cat("\nBest Model Summary\n")
  cat("=================\n")
  print(summary(object$results$best_model))

  invisible(object)
}
