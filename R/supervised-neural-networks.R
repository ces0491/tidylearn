#' @title Neural Networks for tidysl
#' @name tidysl-neural-networks
#' @description Neural network functionality for classification and regression
#' @importFrom nnet nnet
#' @importFrom stats predict
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr %>% mutate
NULL

#' Fit a neural network model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param size Number of units in the hidden layer (default: 5)
#' @param decay Weight decay parameter (default: 0)
#' @param maxit Maximum number of iterations (default: 100)
#' @param trace Logical; whether to print progress (default: FALSE)
#' @param ... Additional arguments to pass to nnet()
#' @return A fitted neural network model
#' @keywords internal
tl_fit_nn <- function(data, formula, is_classification = FALSE,
                      size = 5, decay = 0, maxit = 100, trace = FALSE, ...) {
  # Check if nnet is installed
  tl_check_packages("nnet")

  # Get response variable
  response_var <- all.vars(formula)[1]

  if (is_classification) {
    # For classification, ensure response is a factor
    if (!is.factor(data[[response_var]])) {
      data[[response_var]] <- factor(data[[response_var]])
    }

    # Fit classification neural network
    nn_model <- nnet::nnet(
      formula = formula,
      data = data,
      size = size,
      decay = decay,
      maxit = maxit,
      trace = trace,
      # For classification
      entropy = TRUE,  # Use cross-entropy error function
      ...
    )
  } else {
    # Fit regression neural network
    nn_model <- nnet::nnet(
      formula = formula,
      data = data,
      size = size,
      decay = decay,
      maxit = maxit,
      trace = trace,
      linout = TRUE,  # Linear output for regression
      ...
    )
  }

  return(nn_model)
}

#' Predict using a neural network model
#'
#' @param model A tidysl neural network model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification), "class" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_nn <- function(model, new_data, type = "response", ...) {
  # Get the neural network model
  fit <- model$fit
  is_classification <- model$spec$is_classification
  response_var <- model$spec$response_var

  if (is_classification) {
    # Get original response levels
    levels <- levels(factor(model$data[[response_var]]))

    if (type == "prob") {
      # Get class probabilities
      probs <- as.data.frame(predict(fit, newdata = new_data, type = "raw", ...))

      # For binary classification
      if (length(levels) == 2) {
        # nnet returns probability for the second class only
        if (ncol(probs) == 1) {
          # Create data frame with probabilities for both classes
          prob_df <- tibble::tibble(
            !!levels[1] := 1 - probs[[1]],
            !!levels[2] := probs[[1]]
          )
        } else {
          # Already has both classes
          names(probs) <- levels
          prob_df <- tibble::as_tibble(probs)
        }
      } else {
        # Multiclass - ensure column names match class levels
        names(probs) <- levels
        prob_df <- tibble::as_tibble(probs)
      }

      return(prob_df)
    } else if (type == "class") {
      # Get probabilities first
      probs <- as.data.frame(predict(fit, newdata = new_data, type = "raw", ...))

      # For binary classification
      if (length(levels) == 2) {
        if (ncol(probs) == 1) {
          # Classify based on probability > 0.5
          pred_classes <- ifelse(probs[[1]] > 0.5, levels[2], levels[1])
        } else {
          # Find class with highest probability
          pred_classes <- levels[apply(probs, 1, which.max)]
        }
      } else {
        # Multiclass - find class with highest probability
        pred_classes <- levels[apply(probs, 1, which.max)]
      }

      # Convert to factor with original levels
      pred_classes <- factor(pred_classes, levels = levels)

      return(pred_classes)
    } else if (type == "response") {
      # Same as "class" for classification
      return(tl_predict_nn(model, new_data, type = "class", ...))
    } else {
      stop("Invalid prediction type for neural networks. Use 'prob', 'class', or 'response'.", call. = FALSE)
    }
  } else {
    # Regression predictions
    preds <- as.vector(predict(fit, newdata = new_data, type = "raw", ...))
    return(preds)
  }
}

#' Plot neural network architecture
#'
#' @param model A tidysl neural network model object
#' @param ... Additional arguments
#' @return A ggplot object with neural network architecture
#' @importFrom ggplot2 ggplot aes geom_segment geom_point geom_text theme_void
#' @export
tl_plot_nn_architecture <- function(model, ...) {
  if (model$spec$method != "nn") {
    stop("Neural network architecture plot is only available for neural network models", call. = FALSE)
  }

  # Check if NeuralNetTools is installed
  if (!requireNamespace("NeuralNetTools", quietly = TRUE)) {
    message("Package 'NeuralNetTools' is required for neural network visualization. Please install it.")
    return(NULL)
  }

  # Plot using NeuralNetTools
  NeuralNetTools::plotnet(model$fit, ...)
}

#' Tune a neural network model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param sizes Vector of hidden layer sizes to try
#' @param decays Vector of weight decay parameters to try
#' @param folds Number of cross-validation folds (default: 5)
#' @param ... Additional arguments to pass to nnet()
#' @return A list with the best model and tuning results
#' @export
tl_tune_nn <- function(data, formula, is_classification = FALSE,
                       sizes = c(1, 2, 5, 10), decays = c(0, 0.001, 0.01, 0.1),
                       folds = 5, ...) {
  # Check if nnet is installed
  tl_check_packages("nnet")

  # Create cross-validation splits
  cv_splits <- rsample::vfold_cv(data, v = folds)

  # Initialize results
  tune_results <- expand.grid(
    size = sizes,
    decay = decays,
    error = NA
  )

  # Get response variable
  response_var <- all.vars(formula)[1]

  # Loop through parameter combinations
  for (i in 1:nrow(tune_results)) {
    size <- tune_results$size[i]
    decay <- tune_results$decay[i]

    # Cross-validation for this parameter combination
    cv_errors <- numeric(folds)

    for (j in 1:folds) {
      # Get training and testing data for this fold
      train_data <- rsample::analysis(cv_splits$splits[[j]])
      test_data <- rsample::assessment(cv_splits$splits[[j]])

      # Train neural network
      nn <- tl_fit_nn(
        data = train_data,
        formula = formula,
        is_classification = is_classification,
        size = size,
        decay = decay,
        maxit = 100,
        trace = FALSE,
        ...
      )

      # Make predictions
      preds <- predict(nn, newdata = test_data, type = "raw")

      # Calculate error
      if (is_classification) {
        # For classification, use accuracy
        if (is.factor(test_data[[response_var]])) {
          actuals <- test_data[[response_var]]
        } else {
          actuals <- factor(test_data[[response_var]])
        }

        # Convert predictions to class labels
        levels <- levels(actuals)

        if (length(levels) == 2) {
          if (is.vector(preds)) {
            # Binary classification with single output
            pred_classes <- ifelse(preds > 0.5, levels[2], levels[1])
          } else {
            # Binary classification with multiple outputs
            pred_classes <- levels[apply(preds, 1, which.max)]
          }
        } else {
          # Multiclass classification
          pred_classes <- levels[apply(preds, 1, which.max)]
        }

        pred_classes <- factor(pred_classes, levels = levels)

        # Calculate classification error (1 - accuracy)
        cv_errors[j] <- 1 - mean(pred_classes == actuals)
      } else {
        # For regression, use MSE
        actuals <- test_data[[response_var]]
        cv_errors[j] <- mean((preds - actuals)^2)
      }
    }

    # Store mean error for this parameter combination
    tune_results$error[i] <- mean(cv_errors)
  }

  # Find best parameters
  best_idx <- which.min(tune_results$error)
  best_size <- tune_results$size[best_idx]
  best_decay <- tune_results$decay[best_idx]

  # Train final model with best parameters
  best_model <- tl_fit_nn(
    data = data,
    formula = formula,
    is_classification = is_classification,
    size = best_size,
    decay = best_decay,
    maxit = 100,
    ...
  )

  # Return results
  return(list(
    model = best_model,
    best_size = best_size,
    best_decay = best_decay,
    tuning_results = tune_results
  ))
}

#' Plot neural network training history
#'
#' @param model A tidysl neural network model object
#' @param ... Additional arguments
#' @return A ggplot object with training history
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @export
tl_plot_nn_tuning <- function(model, ...) {
  if (!is.list(model) || !"tuning_results" %in% names(model)) {
    stop("This function requires the output from tl_tune_nn()", call. = FALSE)
  }

  # Extract tuning results
  tune_results <- model$tuning_results

  # Create data for heatmap
  heatmap_data <- tune_results %>%
    dplyr::mutate(size = factor(.data$size), decay = factor(.data$decay))

  # Create heatmap
  p <- ggplot2::ggplot(heatmap_data, ggplot2::aes(x = decay, y = size, fill = error)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = round(error, 4)), color = "white") +
    ggplot2::scale_fill_gradient(low = "blue", high = "red") +
    ggplot2::labs(
      title = "Neural Network Parameter Tuning",
      subtitle = paste0("Best parameters: size = ", model$best_size, ", decay = ", model$best_decay),
      x = "Weight Decay",
      y = "Hidden Layer Size",
      fill = "Error"
    ) +
    ggplot2::theme_minimal()

  return(p)
}
