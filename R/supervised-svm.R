#' @title Support Vector Machines for tidysl
#' @name tidysl-svm
#' @description SVM functionality for classification and regression
#' @importFrom e1071 svm tune
#' @importFrom stats predict
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr %>% mutate
NULL

#' Fit a support vector machine model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param kernel Kernel function ("linear", "polynomial", "radial", "sigmoid")
#' @param cost Cost parameter (default: 1)
#' @param gamma Gamma parameter for kernels (default: 1/ncol(data))
#' @param degree Degree for polynomial kernel (default: 3)
#' @param tune Logical indicating whether to tune hyperparameters (default: FALSE)
#' @param tune_folds Number of folds for cross-validation during tuning (default: 5)
#' @param ... Additional arguments to pass to svm()
#' @return A fitted SVM model
#' @keywords internal
tl_fit_svm <- function(data, formula, is_classification = FALSE,
                       kernel = "radial", cost = 1, gamma = NULL, degree = 3,
                       tune = FALSE, tune_folds = 5, ...) {
  # Check if e1071 is installed
  tl_check_packages("e1071")

  # Set default gamma if not provided
  if (is.null(gamma)) {
    gamma <- 1 / ncol(data)
  }

  # Determine SVM type based on problem type
  if (is_classification) {
    # Get response variable
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]

    # Check if binary or multiclass
    if (is.factor(y) && length(levels(y)) == 2) {
      # Binary classification
      svm_type <- "C-classification"
    } else {
      # Multiclass classification
      svm_type <- "C-classification"
    }
  } else {
    # Regression
    svm_type <- "eps-regression"
  }

  if (tune) {
    # Tune hyperparameters using cross-validation
    tune_result <- e1071::tune(
      svm,
      train.x = formula,
      data = data,
      type = svm_type,
      kernel = kernel,
      ranges = list(
        cost = c(0.1, 1, 10, 100),
        gamma = if (kernel != "linear") c(0.001, 0.01, 0.1, 1) else NULL,
        degree = if (kernel == "polynomial") c(2, 3, 4) else NULL
      ),
      tunecontrol = e1071::tune.control(cross = tune_folds)
    )

    # Extract best parameters
    best_params <- tune_result$best.parameters
    cost <- best_params$cost
    if (kernel != "linear") gamma <- best_params$gamma
    if (kernel == "polynomial") degree <- best_params$degree

    # Store tuning results for later reference
    tuning_results <- tune_result
  } else {
    tuning_results <- NULL
  }

  # Fit the SVM model
  svm_model <- e1071::svm(
    formula = formula,
    data = data,
    type = svm_type,
    kernel = kernel,
    cost = cost,
    gamma = gamma,
    degree = degree,
    probability = is_classification,  # Enable probability estimates for classification
    ...
  )

  # Store tuning results if available
  if (!is.null(tuning_results)) {
    attr(svm_model, "tuning_results") <- tuning_results
  }

  return(svm_model)
}

#' Predict using a support vector machine model
#'
#' @param model A tidysl SVM model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_svm <- function(model, new_data, type = "response", ...) {
  # Get the SVM model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  if (is_classification) {
    if (type == "prob") {
      # Check if probability model was enabled
      if (!fit$probability) {
        stop("Probability estimates not available. Refit the model with probability = TRUE.", call. = FALSE)
      }

      # Get class probabilities
      probs <- attr(
        predict(fit, newdata = new_data, probability = TRUE, ...),
        "probabilities"
      )

      # Convert to tibble with appropriate column names
      class_levels <- colnames(probs)
      prob_df <- as.data.frame(probs)
      names(prob_df) <- class_levels

      return(tibble::as_tibble(prob_df))
    } else if (type == "class" || type == "response") {
      # Get predicted classes
      preds <- predict(fit, newdata = new_data, ...)
      return(preds)
    } else {
      stop("Invalid prediction type for SVM classification. Use 'prob', 'class', or 'response'.", call. = FALSE)
    }
  } else {
    # Regression predictions
    preds <- predict(fit, newdata = new_data, ...)
    return(preds)
  }
}

#' Plot SVM decision boundary
#'
#' @param model A tidysl SVM model object
#' @param x_var Name of the x-axis variable
#' @param y_var Name of the y-axis variable
#' @param grid_size Number of points in each dimension for the grid (default: 100)
#' @param ... Additional arguments
#' @return A ggplot object with decision boundary
#' @importFrom ggplot2 ggplot aes geom_point geom_contour scale_fill_gradient2 labs theme_minimal
#' @export
tl_plot_svm_boundary <- function(model, x_var = NULL, y_var = NULL, grid_size = 100, ...) {
  if (model$spec$method != "svm") {
    stop("Decision boundary plot is only available for SVM models", call. = FALSE)
  }

  if (!model$spec$is_classification) {
    stop("Decision boundary plot is only available for classification models", call. = FALSE)
  }

  # Get original data
  data <- model$data
  formula <- model$spec$formula
  response_var <- all.vars(formula)[1]

  # If x_var and y_var are not specified, use the first two predictors
  if (is.null(x_var) || is.null(y_var)) {
    predictor_vars <- all.vars(formula)[-1]
    if (length(predictor_vars) < 2) {
      stop("At least two predictor variables are required for decision boundary plot", call. = FALSE)
    }
    x_var <- predictor_vars[1]
    y_var <- predictor_vars[2]
  }

  # Check if variables exist
  if (!x_var %in% names(data) || !y_var %in% names(data)) {
    stop("Variables not found in the model data", call. = FALSE)
  }

  # Create grid for prediction
  x_range <- range(data[[x_var]], na.rm = TRUE)
  y_range <- range(data[[y_var]], na.rm = TRUE)

  x_grid <- seq(x_range[1], x_range[2], length.out = grid_size)
  y_grid <- seq(y_range[1], y_range[2], length.out = grid_size)

  grid_data <- expand.grid(x = x_grid, y = y_grid)
  names(grid_data) <- c(x_var, y_var)

  # Add other predictors with mean values
  for (var in setdiff(names(data), c(response_var, x_var, y_var))) {
    if (is.factor(data[[var]]) || is.character(data[[var]])) {
      # For categorical variables, use most frequent value
      most_freq <- names(sort(table(data[[var]]), decreasing = TRUE)[1])
      grid_data[[var]] <- most_freq
    } else {
      # For continuous variables, use mean
      grid_data[[var]] <- mean(data[[var]], na.rm = TRUE)
    }
  }

  # Make predictions on the grid
  if (model$fit$type == "C-classification") {
    # For classification, get probabilities or decision values
    if (model$fit$probability) {
      # Use probabilities
      probs <- attr(
        predict(model$fit, newdata = grid_data, probability = TRUE),
        "probabilities"
      )

      # For binary classification, use probability of positive class
      if (ncol(probs) == 2) {
        pos_class <- colnames(probs)[2]
        grid_data$pred <- probs[, pos_class]
      } else {
        # For multiclass, use predicted class
        preds <- predict(model$fit, newdata = grid_data)
        grid_data$pred <- as.integer(preds)
      }
    } else {
      # Use decision values
      decision_values <- attr(
        predict(model$fit, newdata = grid_data, decision.values = TRUE),
        "decision.values"
      )

      # For binary classification
      if (is.vector(decision_values)) {
        grid_data$pred <- decision_values
      } else {
        # For multiclass, use predicted class
        preds <- predict(model$fit, newdata = grid_data)
        grid_data$pred <- as.integer(preds)
      }
    }
  } else {
    # For regression, use predicted values
    preds <- predict(model$fit, newdata = grid_data)
    grid_data$pred <- preds
  }

  # Create plot
  p <- ggplot2::ggplot() +
    # Add decision boundary contours
    ggplot2::geom_contour_filled(
      data = grid_data,
      ggplot2::aes(x = .data[[x_var]], y = .data[[y_var]], z = pred),
      alpha = 0.3
    ) +
    # Add decision boundary line (only for binary classification)
    ggplot2::geom_contour(
      data = grid_data,
      ggplot2::aes(x = .data[[x_var]], y = .data[[y_var]], z = pred),
      breaks = 0.5,
      color = "black",
      linewidth = 1
    ) +
    # Add original data points
    ggplot2::geom_point(
      data = data,
      ggplot2::aes(x = .data[[x_var]], y = .data[[y_var]], color = .data[[response_var]]),
      size = 3,
      alpha = 0.7
    ) +
    ggplot2::labs(
      title = "SVM Decision Boundary",
      x = x_var,
      y = y_var,
      color = response_var,
      fill = "Predicted Class"
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Plot SVM tuning results
#'
#' @param model A tidysl SVM model object
#' @param ... Additional arguments
#' @return A ggplot object with tuning results
#' @importFrom ggplot2 ggplot aes geom_tile scale_fill_gradient2 labs theme_minimal
#' @export
tl_plot_svm_tuning <- function(model, ...) {
  if (model$spec$method != "svm") {
    stop("Tuning plot is only available for SVM models", call. = FALSE)
  }

  # Check if tuning results are available
  tuning_results <- attr(model$fit, "tuning_results")
  if (is.null(tuning_results)) {
    stop("No tuning results available. Fit the model with tune = TRUE.", call. = FALSE)
  }

  # Extract performance data
  perf_data <- tuning_results$performances

  # Create appropriate plot based on parameters tuned
  if ("gamma" %in% names(perf_data) && "cost" %in% names(perf_data)) {
    # Plot gamma vs cost
    p <- ggplot2::ggplot(perf_data, ggplot2::aes(x = gamma, y = cost, fill = error)) +
      ggplot2::geom_tile() +
      ggplot2::scale_x_log10() +
      ggplot2::scale_y_log10() +
      ggplot2::scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = mean(perf_data$error)) +
      ggplot2::labs(
        title = "SVM Parameter Tuning",
        subtitle = paste0("Best parameters: gamma = ", tuning_results$best.parameters$gamma,
                          ", cost = ", tuning_results$best.parameters$cost),
        x = "Gamma (log scale)",
        y = "Cost (log scale)",
        fill = "Error"
      ) +
      ggplot2::theme_minimal()
  } else if ("cost" %in% names(perf_data)) {
    # Plot cost only
    p <- ggplot2::ggplot(perf_data, ggplot2::aes(x = cost, y = error)) +
      ggplot2::geom_line() +
      ggplot2::geom_point() +
      ggplot2::scale_x_log10() +
      ggplot2::labs(
        title = "SVM Parameter Tuning",
        subtitle = paste0("Best parameter: cost = ", tuning_results$best.parameters$cost),
        x = "Cost (log scale)",
        y = "Error"
      ) +
      ggplot2::theme_minimal()
  } else {
    # Generic plot of all parameters
    p <- ggplot2::ggplot(perf_data, ggplot2::aes(x = seq_len(nrow(perf_data)), y = error)) +
      ggplot2::geom_line() +
      ggplot2::geom_point() +
      ggplot2::labs(
        title = "SVM Parameter Tuning",
        subtitle = paste0("Best parameters: ", paste(names(tuning_results$best.parameters),
                                                     tuning_results$best.parameters,
                                                     sep = " = ", collapse = ", ")),
        x = "Parameter Combination",
        y = "Error"
      ) +
      ggplot2::theme_minimal()
  }

  return(p)
}
