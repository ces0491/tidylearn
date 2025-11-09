#' @title Model Pipeline Functions for tidysl
#' @name tidysl-pipeline
#' @description Functions for creating end-to-end model pipelines
#' @importFrom stats formula
#' @importFrom dplyr %>% filter select mutate
NULL

#' Create a modeling pipeline
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the model
#' @param preprocessing A list of preprocessing steps
#' @param models A list of models to train
#' @param evaluation A list of evaluation criteria
#' @param ... Additional arguments
#' @return A tidysl pipeline object
#' @export
tl_pipeline <- function(data, formula, preprocessing = NULL, models = NULL, evaluation = NULL, ...) {
  # Create default preprocessing if not provided
  if (is.null(preprocessing)) {
    preprocessing <- list(
      impute_missing = TRUE,
      standardize = TRUE,
      dummy_encode = TRUE
    )
  }

  # Create default models if not provided
  if (is.null(models)) {
    # Determine if classification or regression
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]
    is_classification <- is.factor(y) || is.character(y) ||
      (is.numeric(y) && length(unique(y)) <= 10)

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

  # Create default evaluation if not provided
  if (is.null(evaluation)) {
    # Determine if classification or regression
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]
    is_classification <- is.factor(y) || is.character(y) ||
      (is.numeric(y) && length(unique(y)) <= 10)

    if (is_classification) {
      evaluation <- list(
        metrics = c("accuracy", "precision", "recall", "f1", "auc"),
        validation = "cv",
        cv_folds = 5,
        best_metric = "f1"
      )
    } else {
      evaluation <- list(
        metrics = c("rmse", "mae", "rsq", "mape"),
        validation = "cv",
        cv_folds = 5,
        best_metric = "rmse"
      )
    }
  }

  # Create pipeline object
  pipeline <- list(
    formula = formula,
    data = data,
    preprocessing = preprocessing,
    models = models,
    evaluation = evaluation,
    results = NULL
  )

  class(pipeline) <- "tidysl_pipeline"

  return(pipeline)
}

#' Run a tidysl pipeline
#'
#' @param pipeline A tidysl pipeline object
#' @param verbose Logical; whether to print progress
#' @return A tidysl pipeline with results
#' @export
tl_run_pipeline <- function(pipeline, verbose = TRUE) {
  # Check if pipeline is valid
  if (!inherits(pipeline, "tidysl_pipeline")) {
    stop("Input must be a tidysl pipeline object", call. = FALSE)
  }

  # Extract components
  data <- pipeline$data
  formula <- pipeline$formula
  preprocessing <- pipeline$preprocessing
  models <- pipeline$models
  evaluation <- pipeline$evaluation

  # Apply preprocessing
  if (verbose) {
    message("Applying preprocessing steps...")
  }

  processed_data <- data

  if (preprocessing$impute_missing) {
    if (verbose) {
      message("  - Imputing missing values")
    }

    # Simple imputation for numeric and categorical variables
    for (col in names(processed_data)) {
      if (is.numeric(processed_data[[col]])) {
        # Impute with median for numeric
        na_idx <- is.na(processed_data[[col]])
        if (any(na_idx)) {
          med <- median(processed_data[[col]], na.rm = TRUE)
          processed_data[[col]][na_idx] <- med
        }
      } else if (is.factor(processed_data[[col]]) || is.character(processed_data[[col]])) {
        # Impute with mode for categorical
        na_idx <- is.na(processed_data[[col]])
        if (any(na_idx)) {
          # Calculate mode
          if (is.factor(processed_data[[col]])) {
            tab <- table(processed_data[[col]])
          } else {
            tab <- table(processed_data[[col]], useNA = "no")
          }
          mode_val <- names(tab)[which.max(tab)]
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

    # Standardize each numeric column
    for (col in names(processed_data)[numeric_cols]) {
      processed_data[[col]] <- scale(processed_data[[col]])
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
    set.seed(123)  # For reproducibility
    train_idx <- sample(nrow(processed_data), round(train_prop * nrow(processed_data)))
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
        fold_metrics <- tl_evaluate(fold_model, test_fold, metrics = evaluation$metrics)

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
        for (i in 1:nrow(avg_metrics)) {
          metric <- avg_metrics$metric[i]
          mean_val <- avg_metrics$mean_value[i]
          sd_val <- avg_metrics$sd_value[i]

          message("    ", metric, ": ", round(mean_val, 4),
                  " (±", round(sd_val, 4), ")")
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
      test_metrics <- tl_evaluate(split_model, test_data, metrics = evaluation$metrics)

      # Store results
      model_results[[model_name]] <- list(
        model = split_model,
        test_metrics = test_metrics
      )

      if (verbose) {
        for (i in 1:nrow(test_metrics)) {
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
  metric_values <- sapply(names(model_results), function(model_name) {
    result <- model_results[[model_name]]

    if (evaluation$validation == "cv") {
      # Get from average metrics
      metric_row <- result$avg_metrics$metric == best_metric
      if (any(metric_row)) {
        return(result$avg_metrics$mean_value[metric_row])
      } else {
        return(NA)
      }
    } else {
      # Get from test metrics
      metric_row <- result$test_metrics$metric == best_metric
      if (any(metric_row)) {
        return(result$test_metrics$value[metric_row])
      } else {
        return(NA)
      }
    }
  })

  # Determine if higher or lower is better for this metric
  metrics_higher_better <- c("accuracy", "precision", "recall", "f1", "auc", "rsq")
  is_higher_better <- best_metric %in% metrics_higher_better

  # Find best model
  if (is_higher_better) {
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
    model_results = model_results,
    best_model_name = best_model_name,
    best_model = best_model,
    metric_values = metric_values
  )

  return(pipeline)
}

#' Get the best model from a pipeline
#'
#' @param pipeline A tidysl pipeline object with results
#' @return The best tidysl model
#' @export
tl_get_best_model <- function(pipeline) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop("Pipeline has not been run yet. Use tl_run_pipeline() first.", call. = FALSE)
  }

  return(pipeline$results$best_model)
}

#' Compare models from a pipeline
#'
#' @param pipeline A tidysl pipeline object with results
#' @param metrics Character vector of metrics to compare (if NULL, uses all available)
#' @return A comparison plot of model performance
#' @importFrom ggplot2 ggplot aes geom_col facet_wrap labs theme_minimal
#' @export
tl_compare_pipeline_models <- function(pipeline, metrics = NULL) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop("Pipeline has not been run yet. Use tl_run_pipeline() first.", call. = FALSE)
  }

  # Extract model results
  model_results <- pipeline$results$model_results

  # Create data frame for plotting
  comparison_data <- NULL

  for (model_name in names(model_results)) {
    result <- model_results[[model_name]]

    if (!is.null(pipeline$evaluation) && pipeline$evaluation$validation == "cv") {
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
  comparison_data$is_best <- comparison_data$model == pipeline$results$best_model_name

  # Determine which metrics are "higher is better"
  metrics_higher_better <- c("accuracy", "precision", "recall", "f1", "auc", "rsq")
  comparison_data$higher_better <- comparison_data$metric %in% metrics_higher_better

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

  return(p)
}

#' Make predictions using a pipeline
#'
#' @param pipeline A tidysl pipeline object with results
#' @param new_data A data frame containing the new data
#' @param type Type of prediction (default: "response")
#' @param model_name Name of model to use (if NULL, uses the best model)
#' @param ... Additional arguments passed to predict
#' @return Predictions
#' @export
tl_predict_pipeline <- function(pipeline, new_data, type = "response", model_name = NULL, ...) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop("Pipeline has not been run yet. Use tl_run_pipeline() first.", call. = FALSE)
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
    if (pipeline$preprocessing$impute_missing) {
      # Simple imputation for numeric and categorical variables
      for (col in names(processed_new_data)) {
        if (is.numeric(processed_new_data[[col]])) {
          # Impute with median for numeric
          na_idx <- is.na(processed_new_data[[col]])
          if (any(na_idx)) {
            # Use median from original processed data if available
            if (col %in% names(pipeline$results$processed_data)) {
              med <- median(pipeline$results$processed_data[[col]], na.rm = TRUE)
            } else {
              med <- median(processed_new_data[[col]], na.rm = TRUE)
            }
            processed_new_data[[col]][na_idx] <- med
          }
        } else if (is.factor(processed_new_data[[col]]) || is.character(processed_new_data[[col]])) {
          # Impute with mode for categorical
          na_idx <- is.na(processed_new_data[[col]])
          if (any(na_idx)) {
            # Use mode from original processed data if available
            if (col %in% names(pipeline$results$processed_data)) {
              if (is.factor(pipeline$results$processed_data[[col]])) {
                tab <- table(pipeline$results$processed_data[[col]])
              } else {
                tab <- table(pipeline$results$processed_data[[col]], useNA = "no")
              }
            } else {
              if (is.factor(processed_new_data[[col]])) {
                tab <- table(processed_new_data[[col]])
              } else {
                tab <- table(processed_new_data[[col]], useNA = "no")
              }
            }
            mode_val <- names(tab)[which.max(tab)]
            processed_new_data[[col]][na_idx] <- mode_val
          }
        }
      }
    }

    if (pipeline$preprocessing$standardize) {
      # Identify numeric columns (excluding response)
      response_var <- all.vars(pipeline$formula)[1]
      numeric_cols <- sapply(processed_new_data, is.numeric)
      if (response_var %in% names(numeric_cols)) {
        numeric_cols[response_var] <- FALSE  # Don't standardize response
      }

      # Standardize using means and sds from training data
      for (col in names(processed_new_data)[numeric_cols]) {
        if (col %in% names(pipeline$results$processed_data)) {
          # Use mean and sd from training data
          col_mean <- mean(pipeline$results$processed_data[[col]], na.rm = TRUE)
          col_sd <- sd(pipeline$results$processed_data[[col]], na.rm = TRUE)

          processed_new_data[[col]] <- (processed_new_data[[col]] - col_mean) / col_sd
        } else {
          # Just standardize with its own mean and sd
          processed_new_data[[col]] <- scale(processed_new_data[[col]])
        }
      }
    }
  }

  # Make predictions
  preds <- predict(model, processed_new_data, type = type, ...)

  return(preds)
}

#' Save a pipeline to disk
#'
#' @param pipeline A tidysl pipeline object
#' @param file Path to save the pipeline
#' @return Invisible NULL
#' @export
tl_save_pipeline <- function(pipeline, file) {
  # Validate input
  if (!inherits(pipeline, "tidysl_pipeline")) {
    stop("Input must be a tidysl pipeline object", call. = FALSE)
  }

  # Save as RDS
  saveRDS(pipeline, file = file)

  return(invisible(NULL))
}

#' Load a pipeline from disk
#'
#' @param file Path to the pipeline file
#' @return A tidysl pipeline object
#' @export
tl_load_pipeline <- function(file) {
  # Load RDS
  pipeline <- readRDS(file)

  # Validate
  if (!inherits(pipeline, "tidysl_pipeline")) {
    stop("Loaded object is not a tidysl pipeline", call. = FALSE)
  }

  return(pipeline)
}

#' Print a tidysl pipeline
#'
#' @param x A tidysl pipeline object
#' @param ... Additional arguments (not used)
#' @return Invisibly returns the pipeline
#' @export
print.tidysl_pipeline <- function(x, ...) {
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
      model_name <- names(x$results$metric_values)[i]
      metric_value <- x$results$metric_values[i]

      cat("  ", model_name, ": ", x$evaluation$best_metric, " = ",
          round(metric_value, 4),
          if (model_name == x$results$best_model_name) " (best)" else "",
          "\n", sep = "")
    }
  }

  return(invisible(x))
}

#' Summarize a tidysl pipeline
#'
#' @param object A tidysl pipeline object
#' @param ... Additional arguments (not used)
#' @return Invisibly returns the pipeline
#' @export
summary.tidysl_pipeline <- function(object, ...) {
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
      for (i in 1:nrow(result$avg_metrics)) {
        metric <- result$avg_metrics$metric[i]
        mean_val <- result$avg_metrics$mean_value[i]
        sd_val <- result$avg_metrics$sd_value[i]

        cat("  ", metric, ": ", round(mean_val, 4),
            " (±", round(sd_val, 4), ")", "\n", sep = "")
      }
    } else {
      # Print test metrics
      cat("Test metrics:\n")
      for (i in 1:nrow(result$test_metrics)) {
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

  return(invisible(object))
}
