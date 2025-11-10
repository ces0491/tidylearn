#' @title XGBoost Functions for tidysl
#' @name tidysl-xgboost
#' @description XGBoost-specific implementation for gradient boosting
#' @importFrom stats model.matrix as.formula
#' @importFrom dplyr %>% filter select mutate
NULL

#' Fit an XGBoost model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param nrounds Number of boosting rounds (default: 100)
#' @param max_depth Maximum depth of trees (default: 6)
#' @param eta Learning rate (default: 0.3)
#' @param subsample Subsample ratio of observations (default: 1)
#' @param colsample_bytree Subsample ratio of columns (default: 1)
#' @param min_child_weight Minimum sum of instance weight needed in a child (default: 1)
#' @param gamma Minimum loss reduction to make a further partition (default: 0)
#' @param alpha L1 regularization term (default: 0)
#' @param lambda L2 regularization term (default: 1)
#' @param early_stopping_rounds Early stopping rounds (default: NULL)
#' @param nthread Number of threads (default: max available)
#' @param verbose Verbose output (default: 0)
#' @param ... Additional arguments to pass to xgb.train()
#' @return A fitted XGBoost model
#' @keywords internal
tl_fit_xgboost <- function(data, formula, is_classification = FALSE,
                           nrounds = 100, max_depth = 6, eta = 0.3,
                           subsample = 1, colsample_bytree = 1,
                           min_child_weight = 1, gamma = 0,
                           alpha = 0, lambda = 1,
                           early_stopping_rounds = NULL,
                           nthread = NULL, verbose = 0, ...) {
  # Check if xgboost is installed
  tl_check_packages("xgboost")

  # Parse formula
  response_var <- all.vars(formula)[1]

  # Prepare data for XGBoost
  # Create model matrix for predictors (exclude intercept)
  X <- stats::model.matrix(formula, data = data)[, -1, drop = FALSE]

  # Extract response variable
  y <- data[[response_var]]

  # Prepare response variable based on problem type
  if (is_classification) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (length(levels(y)) == 2) {
      # Binary classification: convert to 0/1
      y_numeric <- as.integer(y) - 1
      objective <- "binary:logistic"
      eval_metric <- "logloss"
    } else {
      # Multiclass classification: convert to 0-based index
      y_numeric <- as.integer(y) - 1
      objective <- "multi:softprob"
      eval_metric <- "mlogloss"
      num_class <- length(levels(y))
    }
  } else {
    # Regression
    y_numeric <- y
    objective <- "reg:squarederror"
    eval_metric <- "rmse"
  }

  # Create DMatrix object
  dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y_numeric)

  # Set parameters
  params <- list(
    objective = objective,
    eval_metric = eval_metric,
    max_depth = max_depth,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    alpha = alpha,
    lambda = lambda
  )

  # Add num_class parameter for multiclass classification
  if (is_classification && length(levels(y)) > 2) {
    params$num_class <- num_class
  }

  # Add nthread parameter if specified
  if (!is.null(nthread)) {
    params$nthread <- nthread
  }

  # Train XGBoost model
  xgb_model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    early_stopping_rounds = early_stopping_rounds,
    verbose = verbose,
    ...
  )

  # Store additional information for later use
  attr(xgb_model, "feature_names") <- colnames(X)
  attr(xgb_model, "response_var") <- response_var
  attr(xgb_model, "is_classification") <- is_classification

  if (is_classification) {
    attr(xgb_model, "response_levels") <- levels(y)
  }

  return(xgb_model)
}

#' Predict using an XGBoost model
#'
#' @param model A tidysl XGBoost model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification), "class" (for classification)
#' @param ntreelimit Limit number of trees used for prediction (default: NULL, uses all trees)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_xgboost <- function(model, new_data, type = "response", ntreelimit = NULL, ...) {
  # Extract XGBoost model
  xgb_model <- model$fit

  # Extract metadata
  feature_names <- attr(xgb_model, "feature_names")
  is_classification <- model$spec$is_classification

  # Create model matrix for new data (exclude intercept)
  formula <- model$spec$formula
  X_new <- stats::model.matrix(formula, data = new_data)[, -1, drop = FALSE]

  # Check column names match
  if (!all(colnames(X_new) %in% feature_names)) {
    missing_cols <- setdiff(colnames(X_new), feature_names)
    warning("New data contains columns not in the training data: ",
            paste(missing_cols, collapse = ", "))
  }

  # Create DMatrix for prediction
  dtest <- xgboost::xgb.DMatrix(data = as.matrix(X_new[, feature_names, drop = FALSE]))

  # Make predictions
  if (is_classification) {
    if (type == "prob") {
      # Get class probabilities
      response_levels <- attr(xgb_model, "response_levels")
      n_classes <- length(response_levels)

      if (n_classes == 2) {
        # Binary classification
        prob <- predict(xgb_model, newdata = dtest, ntreelimit = ntreelimit)

        # Create data frame with probabilities for both classes
        prob_df <- data.frame(
          prob0 = 1 - prob,
          prob1 = prob
        )
        colnames(prob_df) <- response_levels

        return(prob_df)
      } else {
        # Multiclass classification
        probs <- predict(xgb_model, newdata = dtest, ntreelimit = ntreelimit,
                                  reshape = TRUE)
        colnames(probs) <- response_levels

        return(as.data.frame(probs))
      }
    } else if (type == "class" || type == "response") {
      # Get predicted classes
      response_levels <- attr(xgb_model, "response_levels")
      n_classes <- length(response_levels)

      if (n_classes == 2) {
        # Binary classification
        prob <- predict(xgb_model, newdata = dtest, ntreelimit = ntreelimit)
        pred_classes <- ifelse(prob > 0.5, response_levels[2], response_levels[1])
      } else {
        # Multiclass classification
        probs <- predict(xgb_model, newdata = dtest, ntreelimit = ntreelimit,
                                  reshape = TRUE)
        pred_idx <- max.col(probs)
        pred_classes <- response_levels[pred_idx]
      }

      # Convert to factor with original levels
      pred_classes <- factor(pred_classes, levels = response_levels)

      return(pred_classes)
    } else {
      stop("Invalid prediction type for XGBoost classification. Use 'prob', 'class', or 'response'.",
           call. = FALSE)
    }
  } else {
    # Regression predictions
    pred <- predict(xgb_model, newdata = dtest, ntreelimit = ntreelimit)
    return(pred)
  }
}

#' Plot feature importance for an XGBoost model
#'
#' @param model A tidysl XGBoost model object
#' @param top_n Number of top features to display (default: 10)
#' @param importance_type Type of importance: "gain", "cover", "frequency"
#' @param ... Additional arguments
#' @return A ggplot object
#' @export
tl_plot_xgboost_importance <- function(model, top_n = 10, importance_type = "gain", ...) {
  # Check if model is an XGBoost model
  if (!inherits(model, "tidysl_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  # Extract XGBoost model
  xgb_model <- model$fit

  # Calculate feature importance
  importance <- xgboost::xgb.importance(
    model = xgb_model,
    feature_names = attr(xgb_model, "feature_names"),
    ...
  )

  # Limit to top features
  if (nrow(importance) > top_n) {
    importance <- importance[1:top_n, ]
  }

  # Create importance plot
  p <- xgboost::xgb.plot.importance(importance, rel_to_first = TRUE, xlab = "Relative importance")

  return(p)
}

#' Plot XGBoost tree visualization
#'
#' @param model A tidysl XGBoost model object
#' @param tree_index Index of the tree to plot (default: 0, first tree)
#' @param ... Additional arguments
#' @return Tree visualization
#' @export
tl_plot_xgboost_tree <- function(model, tree_index = 0, ...) {
  # Check if model is an XGBoost model
  if (!inherits(model, "tidysl_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  # Extract XGBoost model
  xgb_model <- model$fit

  # Plot tree
  xgboost::xgb.plot.tree(model = xgb_model, tree_index = tree_index, ...)
}

#' Tune XGBoost hyperparameters
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param param_grid Named list of parameter values to try
#' @param cv_folds Number of cross-validation folds (default: 5)
#' @param early_stopping_rounds Early stopping rounds (default: 10)
#' @param verbose Logical indicating whether to print progress (default: TRUE)
#' @param ... Additional arguments
#' @return A list with the best model and tuning results
#' @export
tl_tune_xgboost <- function(data, formula, is_classification = FALSE,
                            param_grid = NULL, cv_folds = 5,
                            early_stopping_rounds = 10,
                            verbose = TRUE, ...) {
  # Check if xgboost is installed
  tl_check_packages("xgboost")

  # Set default parameter grid if not provided
  if (is.null(param_grid)) {
    param_grid <- list(
      max_depth = c(3, 6, 9),
      eta = c(0.01, 0.1, 0.3),
      subsample = c(0.7, 1.0),
      colsample_bytree = c(0.7, 1.0),
      min_child_weight = c(1, 3, 5),
      gamma = c(0, 0.1, 0.2)
    )
  }

  # Parse formula
  response_var <- all.vars(formula)[1]

  # Prepare data for XGBoost
  X <- stats::model.matrix(formula, data = data)[, -1, drop = FALSE]
  y <- data[[response_var]]

  # Prepare response variable based on problem type
  if (is_classification) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (length(levels(y)) == 2) {
      # Binary classification: convert to 0/1
      y_numeric <- as.integer(y) - 1
      objective <- "binary:logistic"
      eval_metric <- "logloss"
    } else {
      # Multiclass classification: convert to 0-based index
      y_numeric <- as.integer(y) - 1
      objective <- "multi:softprob"
      eval_metric <- "mlogloss"
      num_class <- length(levels(y))
    }
  } else {
    # Regression
    y_numeric <- y
    objective <- "reg:squarederror"
    eval_metric <- "rmse"
  }

  # Create DMatrix object
  dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y_numeric)

  # Create parameter grid
  param_df <- expand.grid(param_grid)

  if (verbose) {
    message("Tuning XGBoost with ", nrow(param_df), " parameter combinations")
    message("Cross-validation with ", cv_folds, " folds")
  }

  # Initialize storage for results
  results <- list()

  # Loop through parameter combinations
  for (i in 1:nrow(param_df)) {
    if (verbose) {
      message("Parameter set ", i, " of ", nrow(param_df), ": ",
              paste(names(param_df), param_df[i,], sep = "=", collapse = ", "))
    }

    # Extract parameters for this iteration
    params <- as.list(param_df[i,])

    # Set basic parameters
    params$objective <- objective
    params$eval_metric <- eval_metric

    # Add num_class parameter for multiclass classification
    if (is_classification && length(levels(y)) > 2) {
      params$num_class <- num_class
    }

    # Run cross-validation
    cv_result <- xgboost::xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 1000,  # Set high, will be limited by early stopping
      nfold = cv_folds,
      early_stopping_rounds = early_stopping_rounds,
      verbose = ifelse(verbose, 1, 0),
      ...
    )

    # Extract best iteration and performance
    best_iteration <- cv_result$best_iteration
    best_score <- cv_result$evaluation_log[best_iteration, ][[paste0("test_", eval_metric, "_mean")]]

    # Store results
    results[[i]] <- list(
      params = params,
      best_iteration = best_iteration,
      best_score = best_score,
      cv_result = cv_result
    )

    if (verbose) {
      message("  Best iteration: ", best_iteration,
              ", Best ", eval_metric, ": ", round(best_score, 6))
    }
  }

  # Find best parameter set
  best_scores <- sapply(results, function(x) x$best_score)

  # Determine if we should minimize or maximize the metric
  minimize <- eval_metric %in% c("error", "logloss", "mlogloss", "rmse", "mae")

  if (minimize) {
    best_idx <- which.min(best_scores)
  } else {
    best_idx <- which.max(best_scores)
  }

  # Extract best parameters and iteration
  best_params <- results[[best_idx]]$params
  best_iteration <- results[[best_idx]]$best_iteration

  if (verbose) {
    message("Best parameters found: ",
            paste(names(best_params), best_params, sep = "=", collapse = ", "))
    message("Best iteration: ", best_iteration)
    message("Best ", eval_metric, ": ", round(results[[best_idx]]$best_score, 6))
  }

  # Train final model with best parameters
  final_model <- xgboost::xgb.train(
    params = best_params,
    data = dtrain,
    nrounds = best_iteration,
    verbose = 0,
    ...
  )

  # Store additional information for later use
  attr(final_model, "feature_names") <- colnames(X)
  attr(final_model, "response_var") <- response_var
  attr(final_model, "is_classification") <- is_classification

  if (is_classification) {
    attr(final_model, "response_levels") <- levels(y)
  }

  # Create tidysl model wrapper
  model <- structure(
    list(
      spec = list(
        formula = formula,
        method = "xgboost",
        is_classification = is_classification,
        response_var = response_var
      ),
      fit = final_model,
      data = data
    ),
    class = c("tidysl_xgboost", "tidysl_model")
  )

  # Add tuning results to model
  attr(model, "tuning_results") <- list(
    param_grid = param_grid,
    results = results,
    best_params = best_params,
    best_iteration = best_iteration,
    best_score = results[[best_idx]]$best_score,
    minimize = minimize
  )

  return(model)
}

#' Generate SHAP values for XGBoost model interpretation
#'
#' @param model A tidysl XGBoost model object
#' @param data Data for SHAP value calculation (default: NULL, uses training data)
#' @param n_samples Number of samples to use (default: 100, NULL for all)
#' @param trees_idx Trees to include (default: NULL, uses all trees)
#' @return A data frame with SHAP values
#' @export
tl_xgboost_shap <- function(model, data = NULL, n_samples = 100, trees_idx = NULL) {
  # Check if model is an XGBoost model
  if (!inherits(model, "tidysl_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  # Check if xgboost is installed
  tl_check_packages("xgboost")

  # Use training data if data is not provided
  if (is.null(data)) {
    data <- model$data
  }

  # Extract model and feature names
  xgb_model <- model$fit
  feature_names <- attr(xgb_model, "feature_names")

  # Prepare data for XGBoost
  formula <- model$spec$formula
  X <- stats::model.matrix(formula, data = data)[, -1, drop = FALSE]

  # Sample data if needed
  if (!is.null(n_samples) && n_samples < nrow(X)) {
    sample_idx <- sample(nrow(X), n_samples)
    X <- X[sample_idx, , drop = FALSE]
    data <- data[sample_idx, ]
  }

  # Create DMatrix object
  dmatrix <- xgboost::xgb.DMatrix(data = as.matrix(X))

  # Calculate SHAP values
  shap_values <- predict(xgb_model, dmatrix, predcontrib = TRUE,
                                  approxcontrib = FALSE, trees_idx = trees_idx)

  # Remove BIAS column (last column) if present
  if (ncol(shap_values) == ncol(X) + 1) {
    bias <- shap_values[, ncol(shap_values)]
    shap_values <- shap_values[, -ncol(shap_values), drop = FALSE]
  } else {
    bias <- rep(0, nrow(shap_values))
  }

  # Convert to data frame and add feature names
  shap_df <- as.data.frame(shap_values)
  colnames(shap_df) <- feature_names

  # Add bias column
  shap_df$BIAS <- bias

  # Add row identifiers
  shap_df$row_id <- 1:nrow(shap_df)

  # Add original data for reference
  if (nrow(shap_df) == nrow(data)) {
    # Combine with original data for later reference
    for (col in names(data)) {
      if (!col %in% names(shap_df)) {  # Avoid duplicates
        shap_df[[col]] <- data[[col]]
      }
    }
  }

  return(shap_df)
}

#' Plot SHAP summary for XGBoost model
#'
#' @param model A tidysl XGBoost model object
#' @param data Data for SHAP value calculation (default: NULL, uses training data)
#' @param top_n Number of top features to display (default: 10)
#' @param n_samples Number of samples to use (default: 100, NULL for all)
#' @return A ggplot object with SHAP summary
#' @importFrom ggplot2 ggplot aes geom_point scale_color_gradient labs theme_minimal
#' @export
tl_plot_xgboost_shap_summary <- function(model, data = NULL, top_n = 10, n_samples = 100) {
  # Calculate SHAP values
  shap_df <- tl_xgboost_shap(model, data, n_samples)

  # Convert wide to long format for plotting
  feature_cols <- attr(model$fit, "feature_names")

  # Compute absolute mean SHAP value for each feature
  feature_importance <- sapply(feature_cols, function(feat) {
    mean(abs(shap_df[[feat]]))
  })

  # Sort features by importance
  sorted_features <- names(sort(feature_importance, decreasing = TRUE))

  # Limit to top features
  if (length(sorted_features) > top_n) {
    top_features <- sorted_features[1:top_n]
  } else {
    top_features <- sorted_features
  }

  # Prepare data for plotting
  plot_data <- NULL
  for (feat in top_features) {
    # Get feature values and SHAP values
    feat_vals <- if (feat %in% names(model$data)) model$data[[feat]] else NA
    shap_vals <- shap_df[[feat]]

    # Combine into data frame
    feat_data <- data.frame(
      feature = feat,
      feature_value = feat_vals,
      shap_value = shap_vals,
      abs_shap_value = abs(shap_vals)
    )

    # Add to plot data
    if (is.null(plot_data)) {
      plot_data <- feat_data
    } else {
      plot_data <- rbind(plot_data, feat_data)
    }
  }

  # Create plot
  if (requireNamespace("ggforce", quietly = TRUE)) {
    # Use violin plots with jittered points
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = reorder(feature, abs_shap_value), y = shap_value)) +
      ggforce::geom_sina(ggplot2::aes(color = feature_value), size = 2, alpha = 0.7) +
      ggplot2::scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = "SHAP Feature Importance",
        subtitle = "Features sorted by mean absolute SHAP value",
        x = NULL,
        y = "SHAP Value (impact on prediction)",
        color = "Feature Value"
      ) +
      ggplot2::theme_minimal()
  } else {
    # Fall back to basic jittered points
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = reorder(feature, abs_shap_value), y = shap_value)) +
      ggplot2::geom_jitter(ggplot2::aes(color = feature_value), width = 0.2, alpha = 0.7) +
      ggplot2::scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = "SHAP Feature Importance",
        subtitle = "Features sorted by mean absolute SHAP value",
        x = NULL,
        y = "SHAP Value (impact on prediction)",
        color = "Feature Value"
      ) +
      ggplot2::theme_minimal()
  }

  return(p)
}

#' Plot SHAP dependence for a specific feature
#'
#' @param model A tidysl XGBoost model object
#' @param feature Feature name to plot
#' @param interaction_feature Feature to use for coloring (default: NULL)
#' @param data Data for SHAP value calculation (default: NULL, uses training data)
#' @param n_samples Number of samples to use (default: 100, NULL for all)
#' @return A ggplot object with SHAP dependence plot
#' @importFrom ggplot2 ggplot aes geom_point geom_smooth scale_color_gradient labs theme_minimal
#' @export
tl_plot_xgboost_shap_dependence <- function(model, feature, interaction_feature = NULL,
                                            data = NULL, n_samples = 100) {
  # Check if feature exists
  feature_names <- attr(model$fit, "feature_names")
  if (!feature %in% feature_names) {
    stop("Feature not found in model: ", feature, call. = FALSE)
  }

  # Check interaction feature if provided
  if (!is.null(interaction_feature) && !interaction_feature %in% feature_names) {
    stop("Interaction feature not found in model: ", interaction_feature, call. = FALSE)
  }

  # Calculate SHAP values
  shap_df <- tl_xgboost_shap(model, data, n_samples)

  # Use original data if available
  if (is.null(data)) {
    data <- model$data
  }

  # Sample data if needed
  if (!is.null(n_samples) && n_samples < nrow(data)) {
    sample_idx <- sample(nrow(data), n_samples)
    data <- data[sample_idx, ]
  }

  # Get feature values and SHAP values
  feature_values <- data[[feature]]
  shap_values <- shap_df[[feature]]

  # Create plot data
  plot_data <- data.frame(
    feature_value = feature_values,
    shap_value = shap_values
  )

  # Add interaction feature if provided
  if (!is.null(interaction_feature)) {
    plot_data$interaction_value <- data[[interaction_feature]]
  }

  # Create plot
  if (is.null(interaction_feature)) {
    # Without interaction feature
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = feature_value, y = shap_value)) +
      ggplot2::geom_point(alpha = 0.7) +
      ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = TRUE, color = "red") +
      ggplot2::labs(
        title = paste("SHAP Dependence Plot for", feature),
        x = feature,
        y = "SHAP Value (impact on prediction)"
      ) +
      ggplot2::theme_minimal()
  } else {
    # With interaction feature
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = feature_value, y = shap_value, color = interaction_value)) +
      ggplot2::geom_point(alpha = 0.7) +
      ggplot2::scale_color_gradient2(low = "blue", mid = "white", high = "red") +
      ggplot2::labs(
        title = paste("SHAP Dependence Plot for", feature),
        subtitle = paste("Colored by", interaction_feature),
        x = feature,
        y = "SHAP Value (impact on prediction)",
        color = interaction_feature
      ) +
      ggplot2::theme_minimal()
  }

  return(p)
}
