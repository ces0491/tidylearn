#' @title Regularization Functions for tidysl
#' @name tidysl-regularization
#' @description Ridge, Lasso, and Elastic Net regularization functionality
#' @importFrom glmnet glmnet cv.glmnet predict.glmnet
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble
#' @importFrom dplyr %>%
NULL

#' Fit a Ridge regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso, between 0-1 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses cross-validation to select)
#' @param cv_folds Number of folds for cross-validation (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted Ridge regression model
#' @keywords internal
tl_fit_ridge <- function(data, formula, is_classification = FALSE,
                         alpha = 0, lambda = NULL, cv_folds = 5, ...) {
  return(tl_fit_regularized(data, formula, is_classification, alpha, lambda, cv_folds, ...))
}

#' Fit a Lasso regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso, between 0-1 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses cross-validation to select)
#' @param cv_folds Number of folds for cross-validation (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted Lasso regression model
#' @keywords internal
tl_fit_lasso <- function(data, formula, is_classification = FALSE,
                         alpha = 1, lambda = NULL, cv_folds = 5, ...) {
  return(tl_fit_regularized(data, formula, is_classification, alpha, lambda, cv_folds, ...))
}

#' Fit an Elastic Net regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param alpha Mixing parameter (default: 0.5 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses cross-validation to select)
#' @param cv_folds Number of folds for cross-validation (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted Elastic Net regression model
#' @keywords internal
tl_fit_elastic_net <- function(data, formula, is_classification = FALSE,
                               alpha = 0.5, lambda = NULL, cv_folds = 5, ...) {
  return(tl_fit_regularized(data, formula, is_classification, alpha, lambda, cv_folds, ...))
}

#' Fit a regularized regression model (Ridge, Lasso, or Elastic Net)
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso, between 0-1 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses cross-validation to select)
#' @param cv_folds Number of folds for cross-validation (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted regularized regression model
#' @keywords internal
tl_fit_regularized <- function(data, formula, is_classification = FALSE,
                               alpha = 0, lambda = NULL, cv_folds = 5, ...) {
  # Check if glmnet is installed
  tl_check_packages("glmnet")

  # Parse the formula
  response_var <- all.vars(formula)[1]

  # Extract response (y) and predictors (X) matrices
  y <- data[[response_var]]

  # Create model matrix for predictors (X)
  # Remove intercept as glmnet adds it by default
  X <- stats::model.matrix(formula, data = data)[, -1, drop = FALSE]

  # Determine the appropriate family based on problem type
  if (is_classification) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (length(levels(y)) == 2) {
      # Binary classification
      family <- "binomial"
    } else {
      # Multiclass classification
      family <- "multinomial"
    }
  } else {
    # Regression
    family <- "gaussian"
  }

  # Fit the model
  if (is.null(lambda)) {
    # Use cross-validation to select optimal lambda
    cv_model <- glmnet::cv.glmnet(
      x = X,
      y = y,
      alpha = alpha,
      family = family,
      nfolds = cv_folds,
      ...
    )

    # Extract optimal lambda
    lambda_min <- cv_model$lambda.min
    lambda_1se <- cv_model$lambda.1se

    # Fit final model with optimal lambda
    # By default, use lambda.1se for better generalization
    model <- glmnet::glmnet(
      x = X,
      y = y,
      alpha = alpha,
      family = family,
      lambda = cv_model$lambda,  # Keep all lambda values for plots
      ...
    )

    # Store cross-validation results
    attr(model, "cv_results") <- cv_model
    attr(model, "lambda_min") <- lambda_min
    attr(model, "lambda_1se") <- lambda_1se
  } else {
    # Fit model with user-specified lambda
    model <- glmnet::glmnet(
      x = X,
      y = y,
      alpha = alpha,
      family = family,
      lambda = lambda,
      ...
    )

    # Store lambda
    attr(model, "lambda_min") <- lambda
    attr(model, "lambda_1se") <- lambda
  }

  # Store formula, variables, and alpha for future reference
  attr(model, "formula") <- formula
  attr(model, "response_var") <- response_var
  attr(model, "alpha") <- alpha
  attr(model, "is_classification") <- is_classification

  return(model)
}

#' Predict using a regularized regression model
#'
#' @param model A tidysl regularized model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "class" (for classification), "prob" (for classification)
#' @param lambda Which lambda to use for prediction ("1se" or "min", default: "1se")
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_regularized <- function(model, new_data, type = "response",
                                   lambda = "1se", ...) {
  fit <- model$fit
  is_classification <- model$spec$is_classification
  formula <- model$spec$formula
  response_var <- model$spec$response_var

  # Extract lambda value to use
  if (lambda == "1se") {
    lambda_val <- attr(fit, "lambda_1se")
  } else if (lambda == "min") {
    lambda_val <- attr(fit, "lambda_min")
  } else if (is.numeric(lambda)) {
    lambda_val <- lambda
  } else {
    stop("Invalid lambda specification. Use '1se', 'min', or a numeric value.", call. = FALSE)
  }

  # Create model matrix for new data (excluding intercept)
  X_new <- stats::model.matrix(formula, data = new_data)[, -1, drop = FALSE]

  # Make predictions
  if (is_classification) {
    # Classification predictions
    if (type == "response" || type == "link") {
      # Linear predictor
      preds <- glmnet::predict.glmnet(fit, newx = X_new, s = lambda_val, type = "link", ...)
      return(as.vector(preds))
    } else if (type == "class") {
      # Predicted classes
      if ("family" %in% names(fit) && fit$family == "binomial") {
        # Binary classification
        probs <- glmnet::predict.glmnet(fit, newx = X_new, s = lambda_val, type = "response", ...)

        # Get class levels from training data
        class_levels <- levels(factor(model$data[[response_var]]))

        # Classify based on probability > 0.5
        pred_classes <- ifelse(as.vector(probs) > 0.5, class_levels[2], class_levels[1])
        pred_classes <- factor(pred_classes, levels = class_levels)

        return(pred_classes)
      } else {
        # Multiclass classification
        preds <- glmnet::predict.glmnet(fit, newx = X_new, s = lambda_val, type = "class", ...)
        return(as.vector(preds))
      }
    } else if (type == "prob") {
      # Predicted probabilities
      if ("family" %in% names(fit) && fit$family == "binomial") {
        # Binary classification
        pos_probs <- as.vector(glmnet::predict.glmnet(fit, newx = X_new, s = lambda_val, type = "response", ...))
        neg_probs <- 1 - pos_probs

        # Get class levels from training data
        class_levels <- levels(factor(model$data[[response_var]]))

        # Create a data frame with probabilities for each class
        prob_df <- tibble::tibble(
          !!class_levels[1] := neg_probs,
          !!class_levels[2] := pos_probs
        )

        return(prob_df)
      } else {
        # Multiclass classification
        probs <- glmnet::predict.glmnet(fit, newx = X_new, s = lambda_val, type = "response", ...)

        # Reshape to data frame
        prob_df <- as.data.frame(probs)
        names(prob_df) <- levels(factor(model$data[[response_var]]))

        return(tibble::as_tibble(prob_df))
      }
    } else {
      stop("Invalid prediction type for classification. Use 'response', 'class', or 'prob'.", call. = FALSE)
    }
  } else {
    # Regression predictions
    preds <- glmnet::predict.glmnet(fit, newx = X_new, s = lambda_val, type = "response", ...)
    return(as.vector(preds))
  }
}

#' Predict using a Ridge regression model
#'
#' @param model A tidysl Ridge model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_ridge <- function(model, new_data, type = "response", ...) {
  return(tl_predict_regularized(model, new_data, type, ...))
}

#' Predict using a Lasso regression model
#'
#' @param model A tidysl Lasso model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_lasso <- function(model, new_data, type = "response", ...) {
  return(tl_predict_regularized(model, new_data, type, ...))
}

#' Predict using an Elastic Net regression model
#'
#' @param model A tidysl Elastic Net model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_elastic_net <- function(model, new_data, type = "response", ...) {
  return(tl_predict_regularized(model, new_data, type, ...))
}

#' Plot regularization path for a regularized regression model
#'
#' @param model A tidysl regularized model object
#' @param label_n Number of top features to label (default: 5)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_line scale_x_log10 labs theme_minimal
#' @export
tl_plot_regularization_path <- function(model, label_n = 5, ...) {
  # Extract the glmnet model
  fit <- model$fit

  # Get coefficient matrix
  coef_matrix <- as.matrix(coef(fit))

  # Exclude intercept
  coef_matrix <- coef_matrix[-1, , drop = FALSE]

  # Create a data frame for plotting
  coef_df <- tibble::tibble(
    lambda = rep(fit$lambda, each = nrow(coef_matrix)),
    feature = rep(rownames(coef_matrix), times = length(fit$lambda)),
    coefficient = as.vector(coef_matrix)
  )

  # Identify the top features (by maximum absolute coefficient)
  top_features <- coef_df %>%
    dplyr::group_by(.data$feature) %>%
    dplyr::summarize(max_abs_coef = max(abs(.data$coefficient)), .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(.data$max_abs_coef)) %>%
    dplyr::slice_head(n = label_n) %>%
    dplyr::pull(.data$feature)

  # Mark top features for labeling
  coef_df <- coef_df %>%
    dplyr::mutate(is_top = .data$feature %in% top_features)

  # Get optimal lambda values
  lambda_min <- attr(fit, "lambda_min")
  lambda_1se <- attr(fit, "lambda_1se")

  # Create the plot
  p <- ggplot2::ggplot(coef_df, ggplot2::aes(x = lambda, y = coefficient, group = feature, color = is_top)) +
    ggplot2::geom_line(ggplot2::aes(alpha = is_top, size = is_top)) +
    ggplot2::geom_vline(xintercept = lambda_min, linetype = "dashed", color = "blue") +
    ggplot2::geom_vline(xintercept = lambda_1se, linetype = "dashed", color = "red") +
    ggplot2::scale_x_log10() +
    ggplot2::scale_alpha_manual(values = c(0.3, 1)) +
    ggplot2::scale_size_manual(values = c(0.5, 1.2)) +
    ggplot2::scale_color_manual(values = c("gray", "steelblue")) +
    ggplot2::labs(
      title = "Regularization Path",
      subtitle = paste0("Blue: lambda.min, Red: lambda.1se"),
      x = "Lambda (log scale)",
      y = "Coefficients"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none")

  # Add feature labels at the right end of the plot
  if (label_n > 0) {
    # Get the rightmost lambda value
    rightmost_lambda <- min(fit$lambda)

    # Get coefficients at the rightmost lambda for top features
    label_data <- coef_df %>%
      dplyr::filter(.data$is_top, .data$lambda == rightmost_lambda)

    # Add labels
    p <- p + ggplot2::geom_text(
      data = label_data,
      ggplot2::aes(label = feature, x = lambda * 1.1, y = coefficient),
      hjust = 0, vjust = 0.5, size = 3
    )
  }

  return(p)
}

#' Plot cross-validation results for a regularized regression model
#'
#' @param model A tidysl regularized model object
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_line geom_ribbon scale_x_log10 labs theme_minimal
#' @export
tl_plot_cv_results <- function(model, ...) {
  # Extract the glmnet model
  fit <- model$fit

  # Check if cross-validation results are available
  cv_results <- attr(fit, "cv_results")
  if (is.null(cv_results)) {
    stop("Cross-validation results not available. Model must be fitted with lambda = NULL.", call. = FALSE)
  }

  # Get lambda values and mean cross-validated error
  lambda <- cv_results$lambda
  cvm <- cv_results$cvm  # Mean cross-validated error
  cvsd <- cv_results$cvsd  # Standard error of cross-validated error

  # Create a data frame for plotting
  cv_df <- tibble::tibble(
    lambda = lambda,
    error = cvm,
    error_upper = cvm + cvsd,
    error_lower = cvm - cvsd
  )

  # Get optimal lambda values
  lambda_min <- cv_results$lambda.min
  lambda_1se <- cv_results$lambda.1se

  # Determine y-axis label based on model type
  if (model$spec$is_classification) {
    y_label <- "Binomial Deviance"
  } else {
    y_label <- "Mean Squared Error"
  }

  # Create the plot
  p <- ggplot2::ggplot(cv_df, ggplot2::aes(x = lambda, y = error)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = error_lower, ymax = error_upper), alpha = 0.2) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::geom_vline(xintercept = lambda_min, linetype = "dashed", color = "blue") +
    ggplot2::geom_vline(xintercept = lambda_1se, linetype = "dashed", color = "red") +
    ggplot2::scale_x_log10() +
    ggplot2::labs(
      title = "Cross-Validation Results",
      subtitle = paste0("Blue: lambda.min, Red: lambda.1se"),
      x = "Lambda (log scale)",
      y = y_label
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Plot variable importance for a regularized regression model
#'
#' @param model A tidysl regularized model object
#' @param lambda Which lambda to use ("1se" or "min", default: "1se")
#' @param top_n Number of top features to display (default: 20)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_col coord_flip labs theme_minimal
#' @export
tl_plot_importance_regularized <- function(model, lambda = "1se", top_n = 20, ...) {
  # Extract the glmnet model
  fit <- model$fit

  # Extract lambda value to use
  if (lambda == "1se") {
    lambda_val <- attr(fit, "lambda_1se")
  } else if (lambda == "min") {
    lambda_val <- attr(fit, "lambda_min")
  } else if (is.numeric(lambda)) {
    lambda_val <- lambda
  } else {
    stop("Invalid lambda specification. Use '1se', 'min', or a numeric value.", call. = FALSE)
  }

  # Get coefficients at selected lambda
  lambda_index <- which.min(abs(fit$lambda - lambda_val))
  coefs <- as.matrix(coef(fit, s = lambda_val))

  # Exclude intercept
  coefs <- coefs[-1, , drop = FALSE]

  # Create a data frame for plotting
  importance_df <- tibble::tibble(
    feature = rownames(coefs),
    importance = abs(as.vector(coefs))
  ) %>%
    dplyr::arrange(dplyr::desc(.data$importance)) %>%
    dplyr::filter(.data$importance > 0) %>%
    dplyr::slice_head(n = top_n)

  # Create the plot
  p <- ggplot2::ggplot(importance_df, ggplot2::aes(x = stats::reorder(feature, importance), y = importance)) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Feature Importance",
      subtitle = paste0("Based on coefficient magnitudes at lambda = ", round(lambda_val, 5)),
      x = NULL,
      y = "Absolute Coefficient Value"
    ) +
    ggplot2::theme_minimal()

  return(p)
}
