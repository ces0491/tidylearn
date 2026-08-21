#' @title Regularization Functions for tidylearn
#' @name tidylearn-regularization
#' @description Ridge, Lasso, and Elastic Net regularization
#'   functionality
#' @importFrom glmnet glmnet cv.glmnet predict.glmnet
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble
NULL

#' Fit a Ridge regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso,
#'   between 0-1 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses
#'   cross-validation to select)
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted Ridge regression model
#' @keywords internal
tl_fit_ridge <- function(data, formula,
                         is_classification = FALSE,
                         alpha = 0, lambda = NULL,
                         cv_folds = 5, ...) {
  tl_fit_regularized(
    data, formula, is_classification,
    alpha, lambda, cv_folds, ...
  )
}

#' Fit a Lasso regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso,
#'   between 0-1 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses
#'   cross-validation to select)
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted Lasso regression model
#' @keywords internal
tl_fit_lasso <- function(data, formula,
                         is_classification = FALSE,
                         alpha = 1, lambda = NULL,
                         cv_folds = 5, ...) {
  tl_fit_regularized(
    data, formula, is_classification,
    alpha, lambda, cv_folds, ...
  )
}

#' Fit an Elastic Net regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (default: 0.5 for
#'   Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses
#'   cross-validation to select)
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted Elastic Net regression model
#' @keywords internal
tl_fit_elastic_net <- function(data, formula,
                               is_classification = FALSE,
                               alpha = 0.5,
                               lambda = NULL,
                               cv_folds = 5, ...) {
  tl_fit_regularized(
    data, formula, is_classification,
    alpha, lambda, cv_folds, ...
  )
}

#' Fit a regularized regression model
#'
#' Fits Ridge, Lasso, or Elastic Net regularization.
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso,
#'   between 0-1 for Elastic Net)
#' @param lambda Regularization parameter (if NULL, uses
#'   cross-validation to select)
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet()
#' @return A fitted regularized regression model
#' @keywords internal
tl_fit_regularized <- function(data, formula,
                               is_classification = FALSE,
                               alpha = 0, lambda = NULL,
                               cv_folds = 5, ...) {
  # Check if glmnet is installed
  tl_check_packages("glmnet")

  # Parse the formula
  response_var <- all.vars(formula)[1]

  # Extract response (y) and predictors matrices
  y <- data[[response_var]]

  # Create model matrix for predictors
  # Remove intercept as glmnet adds it by default
  model_frame <- stats::model.frame(formula, data = data)
  design_terms <- stats::terms(model_frame)
  x_mat <- stats::model.matrix(
    design_terms, model_frame
  )[, -1, drop = FALSE]

  # Retain the terms and factor levels so prediction can rebuild an
  # identically-coded design matrix on new data
  design_xlevels <- stats::.getXlevels(design_terms, model_frame)

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
      x = x_mat,
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
    # Use lambda.1se for better generalization
    model <- glmnet::glmnet(
      x = x_mat,
      y = y,
      alpha = alpha,
      family = family,
      lambda = cv_model$lambda,
      ...
    )

    # Store cross-validation results
    attr(model, "cv_results") <- cv_model
    attr(model, "lambda_min") <- lambda_min
    attr(model, "lambda_1se") <- lambda_1se
  } else {
    # Fit model with user-specified lambda
    model <- glmnet::glmnet(
      x = x_mat,
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

  # Store formula, variables, and alpha
  attr(model, "formula") <- formula
  attr(model, "response_var") <- response_var
  attr(model, "alpha") <- alpha
  attr(model, "is_classification") <- is_classification

  # Design specification used at fit time
  attr(model, "tl_terms") <- design_terms
  attr(model, "tl_xlevels") <- design_xlevels
  attr(model, "tl_colnames") <- colnames(x_mat)
  if (is_classification) {
    attr(model, "response_levels") <- levels(y)
  }

  model
}

#' Plot regularization path for a regularized model
#'
#' @param model A tidylearn regularized model object
#' @param label_n Number of top features to label
#'   (default: 5)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "lasso")
#' tl_plot_regularization_path(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_line
#'   scale_x_log10 labs theme_minimal
#' @export
tl_plot_regularization_path <- function(model,
                                        label_n = 5,
                                        ...) {
  # Extract the glmnet model
  fit <- model$fit

  # Get coefficient matrix
  coef_matrix <- as.matrix(coef(fit))

  # Exclude intercept
  coef_matrix <- coef_matrix[-1, , drop = FALSE]

  # Create a data frame for plotting
  coef_df <- tibble::tibble(
    lambda = rep(
      fit$lambda, each = nrow(coef_matrix)
    ),
    feature = rep(
      rownames(coef_matrix),
      times = length(fit$lambda)
    ),
    coefficient = as.vector(coef_matrix)
  )

  # Identify the top features (by max absolute coef)
  top_features <- coef_df %>%
    dplyr::group_by(.data$feature) %>%
    dplyr::summarize(
      max_abs_coef = max(abs(.data$coefficient)),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(.data$max_abs_coef)) %>%
    dplyr::slice_head(n = label_n) %>%
    dplyr::pull(.data$feature)

  # Mark top features for labeling
  coef_df <- coef_df %>%
    dplyr::mutate(
      is_top = .data$feature %in% top_features
    )

  # Get optimal lambda values
  lambda_min <- attr(fit, "lambda_min")
  lambda_1se <- attr(fit, "lambda_1se")

  # Create the plot
  p <- ggplot2::ggplot(
    coef_df,
    ggplot2::aes(
      x = lambda,
      y = coefficient,
      group = feature,
      color = is_top
    )
  ) +
    ggplot2::geom_line(
      # `size` on a line is deprecated since ggplot2 3.4.0
      ggplot2::aes(alpha = is_top, linewidth = is_top)
    ) +
    ggplot2::geom_vline(
      xintercept = lambda_min,
      linetype = "dashed",
      color = "blue"
    ) +
    ggplot2::geom_vline(
      xintercept = lambda_1se,
      linetype = "dashed",
      color = "red"
    ) +
    ggplot2::scale_x_log10() +
    ggplot2::scale_alpha_manual(
      values = c(0.3, 1)
    ) +
    ggplot2::scale_linewidth_manual(
      values = c(0.5, 1.2)
    ) +
    ggplot2::scale_color_manual(
      values = c("gray", "steelblue")
    ) +
    ggplot2::labs(
      title = "Regularization Path",
      subtitle = paste0(
        "Blue: lambda.min, Red: lambda.1se"
      ),
      x = "Lambda (log scale)",
      y = "Coefficients"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none")

  # Add feature labels at the right end of the plot
  if (label_n > 0) {
    # Get the rightmost lambda value
    rightmost_lambda <- min(fit$lambda)

    # Get coefficients at rightmost lambda for top feats
    label_data <- coef_df %>%
      dplyr::filter(
        .data$is_top,
        .data$lambda == rightmost_lambda
      )

    # Add labels
    p <- p + ggplot2::geom_text(
      data = label_data,
      ggplot2::aes(
        label = feature,
        x = lambda * 1.1,
        y = coefficient
      ),
      hjust = 0, vjust = 0.5, size = 3
    )
  }

  p
}

#' Plot cross-validation results for a regularized model
#'
#' Shows the cross-validation error as a function of
#' lambda for ridge, lasso, or elastic net models fitted
#' with cv.glmnet.
#'
#' @param model A tidylearn regularized model object
#'   (ridge, lasso, or elastic_net)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "ridge")
#' tl_plot_regularization_cv(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_point geom_line
#'   geom_ribbon scale_x_log10 labs theme_minimal
#' @export
tl_plot_regularization_cv <- function(model, ...) {
  # Extract the glmnet model
  fit <- model$fit

  # Check if cross-validation results are available
  cv_results <- attr(fit, "cv_results")
  if (is.null(cv_results)) {
    stop(
      "Cross-validation results not available. ",
      "Model must be fitted with lambda = NULL.",
      call. = FALSE
    )
  }

  # Get lambda values and mean cross-validated error
  lambda <- cv_results$lambda
  cvm <- cv_results$cvm
  cvsd <- cv_results$cvsd

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
  p <- ggplot2::ggplot(
    cv_df,
    ggplot2::aes(x = lambda, y = error)
  ) +
    ggplot2::geom_ribbon(
      ggplot2::aes(
        ymin = error_lower,
        ymax = error_upper
      ),
      alpha = 0.2
    ) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::geom_vline(
      xintercept = lambda_min,
      linetype = "dashed",
      color = "blue"
    ) +
    ggplot2::geom_vline(
      xintercept = lambda_1se,
      linetype = "dashed",
      color = "red"
    ) +
    ggplot2::scale_x_log10() +
    ggplot2::labs(
      title = "Cross-Validation Results",
      subtitle = paste0(
        "Blue: lambda.min, Red: lambda.1se"
      ),
      x = "Lambda (log scale)",
      y = y_label
    ) +
    ggplot2::theme_minimal()

  p
}

#' Plot variable importance for a regularized model
#'
#' @param model A tidylearn regularized model object
#' @param lambda Which lambda to use
#'   ("1se" or "min", default: "1se")
#' @param top_n Number of top features to display
#'   (default: 20)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "lasso")
#' tl_plot_importance_regularized(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_col coord_flip
#'   labs theme_minimal
#' @export
tl_plot_importance_regularized <- function(model,
                                           lambda = "1se",
                                           top_n = 20,
                                           ...) {
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
    stop(
      "Invalid lambda specification. ",
      "Use '1se', 'min', or a numeric value.",
      call. = FALSE
    )
  }

  # Get coefficients at selected lambda
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
  p <- ggplot2::ggplot(
    importance_df,
    ggplot2::aes(
      x = stats::reorder(feature, importance),
      y = importance
    )
  ) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Feature Importance",
      subtitle = paste0(
        "Based on coefficient magnitudes at ",
        "lambda = ", round(lambda_val, 5)
      ),
      x = NULL,
      y = "Absolute Coefficient Value"
    ) +
    ggplot2::theme_minimal()

  p
}


#' Predict using a glmnet model
#' @keywords internal
#' @noRd
tl_predict_glmnet <- function(model, new_data,
                              type = "response",
                              ...) {
  fit <- model$fit
  is_classification <- model$spec$is_classification

  formula <- model$spec$formula
  response_var <- model$spec$response_var

  # Rebuild the design matrix exactly as the fit did. Deriving it from a
  # "~ predictors - 1" formula instead would one-hot encode the first
  # factor, giving one more column than the model was trained on.
  design_terms <- attr(fit, "tl_terms")
  design_xlevels <- attr(fit, "tl_xlevels")

  if (is.null(design_terms)) {
    # Model fitted by an earlier version: recover the design from the
    # training data, which the model object carries
    training_frame <- stats::model.frame(formula, data = model$data)
    design_terms <- stats::terms(training_frame)
    design_xlevels <- stats::.getXlevels(design_terms, training_frame)
  }

  predictor_terms <- stats::delete.response(design_terms)
  new_frame <- stats::model.frame(
    predictor_terms, new_data,
    xlev = design_xlevels,
    na.action = stats::na.pass
  )
  x_new <- stats::model.matrix(
    predictor_terms, new_frame
  )[, -1, drop = FALSE]

  # Get optimal lambda value from model
  lambda_val <- attr(fit, "lambda_min")
  if (is.null(lambda_val)) {
    # If no cv was used, use the first lambda
    lambda_val <- fit$lambda[1]
  }

  if (!is_classification) {
    return(as.vector(stats::predict(
      fit, newx = x_new, type = "response", s = lambda_val
    )))
  }

  class_levels <- attr(fit, "response_levels")
  if (is.null(class_levels)) {
    class_levels <- levels(factor(model$data[[response_var]]))
  }

  if (type == "prob") {
    probs <- stats::predict(
      fit, newx = x_new, type = "response", s = lambda_val
    )

    if (length(class_levels) == 2) {
      # glmnet returns the probability of the second level
      positive <- as.vector(probs)
      tibble::tibble(
        !!class_levels[1] := 1 - positive,
        !!class_levels[2] := positive
      )
    } else {
      # Multinomial: an [observation, class, lambda] array. drop = TRUE
      # would flatten a single-row prediction to a bare vector, so pin
      # the shape explicitly instead.
      prob_mat <- matrix(
        probs[, , 1], nrow = nrow(x_new), ncol = length(class_levels)
      )
      colnames(prob_mat) <- class_levels
      tibble::as_tibble(as.data.frame(prob_mat))
    }
  } else if (type == "class" || type == "response") {
    preds <- stats::predict(
      fit, newx = x_new, type = "class", s = lambda_val
    )
    factor(as.vector(preds), levels = class_levels)
  } else {
    stop(
      "Invalid prediction type for regularized classification. ",
      "Use 'prob', 'class', or 'response'.",
      call. = FALSE
    )
  }
}
