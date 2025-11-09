#' @title Regression Functions for tidysl
#' @name tidysl-regression
#' @description Linear and polynomial regression functionality
#' @importFrom stats lm predict poly model.matrix
#' @importFrom tibble tibble
#' @importFrom dplyr %>% bind_cols
NULL

#' Fit a linear regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param ... Additional arguments to pass to lm()
#' @return A fitted linear regression model
#' @keywords internal
tl_fit_linear <- function(data, formula, ...) {
  lm_model <- stats::lm(formula, data = data, ...)
  return(lm_model)
}

#' Predict using a linear regression model
#'
#' @param model A tidysl linear model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "confidence", "prediction"
#' @param level Confidence level for intervals (default: 0.95)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_linear <- function(model, new_data, type = "response", level = 0.95, ...) {
  if (type == "response") {
    preds <- stats::predict(model$fit, newdata = new_data, ...)
    return(preds)
  } else if (type == "confidence") {
    pred_obj <- stats::predict(model$fit, newdata = new_data, interval = "confidence", level = level, ...)
    return(as.data.frame(pred_obj))
  } else if (type == "prediction") {
    pred_obj <- stats::predict(model$fit, newdata = new_data, interval = "prediction", level = level, ...)
    return(as.data.frame(pred_obj))
  } else {
    stop("Invalid prediction type. Use 'response', 'confidence', or 'prediction'.", call. = FALSE)
  }
}

#' Fit a polynomial regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param degree Degree of the polynomial (default: 2)
#' @param ... Additional arguments to pass to lm()
#' @return A fitted polynomial regression model
#' @keywords internal
tl_fit_polynomial <- function(data, formula, degree = 2, ...) {
  # Parse the formula to get the response and predictor variables
  terms <- stats::terms(formula, data = data)
  response_var <- all.vars(formula)[1]
  predictor_vars <- attr(terms, "term.labels")

  # Create a new formula with polynomial terms
  poly_formula <- paste(response_var, "~")

  for (i in seq_along(predictor_vars)) {
    var <- predictor_vars[i]
    # Check if this is an interaction term or has special characters
    if (grepl(":", var) || grepl("\\*", var) || grepl("^I\\(", var)) {
      # Keep interaction terms as they are
      poly_formula <- paste(poly_formula, var,
                            ifelse(i < length(predictor_vars), "+", ""))
    } else {
      # Create polynomial term
      poly_formula <- paste(poly_formula, paste0("poly(", var, ", degree = ", degree, ", raw = TRUE)"),
                            ifelse(i < length(predictor_vars), "+", ""))
    }
  }

  # Fit the polynomial model
  poly_model <- stats::lm(as.formula(poly_formula), data = data, ...)

  # Store original formula and degree for future reference
  attr(poly_model, "original_formula") <- formula
  attr(poly_model, "poly_degree") <- degree

  return(poly_model)
}

#' Predict using a polynomial regression model
#'
#' @param model A tidysl polynomial model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "confidence", "prediction"
#' @param level Confidence level for intervals (default: 0.95)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_polynomial <- function(model, new_data, type = "response", level = 0.95, ...) {
  # Extract polynomial degree
  degree <- attr(model$fit, "poly_degree")
  if (is.null(degree)) degree <- 2  # Default if not stored

  # Call the underlying prediction function
  if (type == "response") {
    preds <- stats::predict(model$fit, newdata = new_data, ...)
    return(preds)
  } else if (type == "confidence") {
    pred_obj <- stats::predict(model$fit, newdata = new_data, interval = "confidence", level = level, ...)
    return(as.data.frame(pred_obj))
  } else if (type == "prediction") {
    pred_obj <- stats::predict(model$fit, newdata = new_data, interval = "prediction", level = level, ...)
    return(as.data.frame(pred_obj))
  } else {
    stop("Invalid prediction type. Use 'response', 'confidence', or 'prediction'.", call. = FALSE)
  }
}

#' Plot diagnostics for a regression model
#'
#' @param model A tidysl regression model object
#' @param which Which plots to create (1:4)
#' @param ... Additional arguments
#' @return A ggplot object (or list of ggplot objects)
#' @importFrom ggplot2 ggplot aes geom_point geom_smooth geom_hline geom_text
#' @importFrom ggplot2 labs theme_minimal scale_color_gradient stat_qq stat_qq_line
#' @keywords internal
tl_plot_diagnostics <- function(model, which = 1:4, ...) {
  # Get residuals and fitted values
  fitted_vals <- fitted(model$fit)
  residuals <- residuals(model$fit)
  std_residuals <- rstandard(model$fit)

  # Create data frame for plotting
  plot_data <- tibble::tibble(
    fitted = fitted_vals,
    residuals = residuals,
    std_residuals = std_residuals,
    abs_residuals = abs(residuals),
    sqrt_abs_residuals = sqrt(abs(residuals)),
    leverage = hatvalues(model$fit),
    cooks_distance = cooks.distance(model$fit)
  )

  plots <- list()

  # Plot 1: Residuals vs Fitted
  if (1 %in% which) {
    p1 <- ggplot2::ggplot(plot_data, ggplot2::aes(x = fitted, y = residuals)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      ggplot2::geom_smooth(se = FALSE, color = "blue", method = "loess") +
      ggplot2::labs(
        title = "Residuals vs Fitted",
        x = "Fitted values",
        y = "Residuals"
      ) +
      ggplot2::theme_minimal()

    plots[["residuals_vs_fitted"]] <- p1
  }

  # Plot 2: Normal Q-Q
  if (2 %in% which) {
    p2 <- ggplot2::ggplot(plot_data, ggplot2::aes(sample = std_residuals)) +
      ggplot2::stat_qq() +
      ggplot2::stat_qq_line(color = "red") +
      ggplot2::labs(
        title = "Normal Q-Q",
        x = "Theoretical Quantiles",
        y = "Standardized Residuals"
      ) +
      ggplot2::theme_minimal()

    plots[["qq"]] <- p2
  }

  # Plot 3: Scale-Location
  if (3 %in% which) {
    p3 <- ggplot2::ggplot(plot_data, ggplot2::aes(x = fitted, y = sqrt_abs_residuals)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_smooth(se = FALSE, color = "blue", method = "loess") +
      ggplot2::labs(
        title = "Scale-Location",
        x = "Fitted values",
        y = "√|Standardized residuals|"
      ) +
      ggplot2::theme_minimal()

    plots[["scale_location"]] <- p3
  }

  # Plot 4: Cook's distance / Leverage
  if (4 %in% which) {
    p4 <- ggplot2::ggplot(plot_data, ggplot2::aes(x = leverage, y = std_residuals)) +
      ggplot2::geom_point(ggplot2::aes(size = cooks_distance, color = cooks_distance), alpha = 0.6) +
      ggplot2::geom_hline(yintercept = c(-2, 0, 2), linetype = "dashed", color = "red") +
      ggplot2::scale_color_gradient(low = "blue", high = "red") +
      ggplot2::labs(
        title = "Residuals vs Leverage",
        x = "Leverage",
        y = "Standardized residuals",
        size = "Cook's distance",
        color = "Cook's distance"
      ) +
      ggplot2::theme_minimal()

    plots[["residuals_vs_leverage"]] <- p4
  }

  # Return a single plot or list of plots
  if (length(plots) == 1) {
    return(plots[[1]])
  } else {
    return(plots)
  }
}

#' Plot actual vs predicted values for a regression model
#'
#' @param model A tidysl regression model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_abline labs theme_minimal
#' @keywords internal
tl_plot_actual_predicted <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual and predicted values
  actuals <- new_data[[model$spec$response_var]]
  predictions <- predict(model, new_data)$prediction

  # Create data frame for plotting
  plot_data <- tibble::tibble(
    actual = actuals,
    predicted = predictions
  )

  # Calculate correlation
  corr <- round(cor(actuals, predictions), 3)
  r_squared <- round(cor(actuals, predictions)^2, 3)

  # Create the plot
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = actual, y = predicted)) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    ggplot2::labs(
      title = "Actual vs Predicted Values",
      subtitle = paste0("Correlation: ", corr, ", R²: ", r_squared),
      x = "Actual values",
      y = "Predicted values"
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Plot residuals for a regression model
#'
#' @param model A tidysl regression model object
#' @param type Type of residual plot: "fitted" (default), "histogram", "predicted"
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_hline geom_histogram labs theme_minimal
#' @keywords internal
tl_plot_residuals <- function(model, type = "fitted", ...) {
  # Get residuals and fitted values
  fitted_vals <- fitted(model$fit)
  residuals <- residuals(model$fit)

  # Create data frame for plotting
  plot_data <- tibble::tibble(
    fitted = fitted_vals,
    residuals = residuals
  )

  # Create the plot based on type
  if (type == "fitted") {
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = fitted, y = residuals)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      ggplot2::labs(
        title = "Residuals vs Fitted Values",
        x = "Fitted values",
        y = "Residuals"
      ) +
      ggplot2::theme_minimal()
  } else if (type == "histogram") {
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = residuals)) +
      ggplot2::geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
      ggplot2::labs(
        title = "Histogram of Residuals",
        x = "Residuals",
        y = "Count"
      ) +
      ggplot2::theme_minimal()
  } else if (type == "predicted") {
    # Get predictions
    predictions <- predict(model, model$data)$prediction

    # Add to plot data
    plot_data$predicted <- predictions

    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = predicted, y = residuals)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      ggplot2::labs(
        title = "Residuals vs Predicted Values",
        x = "Predicted values",
        y = "Residuals"
      ) +
      ggplot2::theme_minimal()
  } else {
    stop("Invalid plot type. Use 'fitted', 'histogram', or 'predicted'.", call. = FALSE)
  }

  return(p)
}

#' Create confidence and prediction interval plots
#'
#' @param model A tidysl regression model object
#' @param new_data Optional data frame for prediction (if NULL, uses training data)
#' @param level Confidence level (default: 0.95)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_ribbon labs theme_minimal
#' @export
tl_plot_intervals <- function(model, new_data = NULL, level = 0.95, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  if (model$spec$is_classification) {
    stop("Interval plots are only available for regression models", call. = FALSE)
  }

  # Get actual values
  x_var <- all.vars(model$spec$formula)[-1][1]  # First predictor variable
  y_var <- model$spec$response_var

  # Sort data by x variable for smooth curves
  sorted_data <- new_data[order(new_data[[x_var]]), ]

  # Calculate intervals
  intervals <- tl_prediction_intervals(model, sorted_data, level = level)

  # Create plot data
  plot_data <- tibble::tibble(
    x = sorted_data[[x_var]],
    y = sorted_data[[y_var]],
    pred = intervals$prediction,
    conf_lower = intervals$conf_lower,
    conf_upper = intervals$conf_upper,
    pred_lower = intervals$pred_lower,
    pred_upper = intervals$pred_upper
  )

  # Create the plot
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = x)) +
    # Prediction intervals (wider)
    ggplot2::geom_ribbon(ggplot2::aes(ymin = pred_lower, ymax = pred_upper),
                         fill = "lightblue", alpha = 0.3) +
    # Confidence intervals (narrower)
    ggplot2::geom_ribbon(ggplot2::aes(ymin = conf_lower, ymax = conf_upper),
                         fill = "steelblue", alpha = 0.5) +
    # Fitted line
    ggplot2::geom_line(ggplot2::aes(y = pred), color = "blue", size = 1) +
    # Actual points
    ggplot2::geom_point(ggplot2::aes(y = y), alpha = 0.6) +
    ggplot2::labs(
      title = paste0("Regression with ", level * 100, "% Confidence & Prediction Intervals"),
      subtitle = "Dark band: Confidence interval, Light band: Prediction interval",
      x = x_var,
      y = y_var
    ) +
    ggplot2::theme_minimal()

  return(p)
}
