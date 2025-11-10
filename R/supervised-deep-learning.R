#' @title Deep Learning for tidysl
#' @name tidysl-deep-learning
#' @description Deep learning functionality using Keras/TensorFlow
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr %>% mutate
NULL

#' Fit a deep learning model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param hidden_layers Vector of units in each hidden layer (default: c(32, 16))
#' @param activation Activation function for hidden layers (default: "relu")
#' @param dropout Dropout rate for regularization (default: 0.2)
#' @param epochs Number of training epochs (default: 30)
#' @param batch_size Batch size for training (default: 32)
#' @param validation_split Proportion of data for validation (default: 0.2)
#' @param verbose Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch) (default: 0)
#' @param ... Additional arguments
#' @return A fitted deep learning model
#' @keywords internal
tl_fit_deep <- function(data, formula, is_classification = FALSE,
                        hidden_layers = c(32, 16), activation = "relu", dropout = 0.2,
                        epochs = 30, batch_size = 32, validation_split = 0.2,
                        verbose = 0, ...) {
  # Check if keras is installed
  tl_check_packages(c("keras", "tensorflow"))

  # Parse the formula
  response_var <- all.vars(formula)[1]

  # Prepare data
  # Extract response (y)
  y <- data[[response_var]]

  # Create model matrix for predictors (X), excluding the intercept
  X <- stats::model.matrix(formula, data = data)[, -1, drop = FALSE]

  # Normalize features
  X_means <- colMeans(X)
  X_sds <- apply(X, 2, sd)
  X_scaled <- scale(X, center = X_means, scale = X_sds)

  # Prepare y based on problem type
  if (is_classification) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (length(levels(y)) == 2) {
      # Binary classification
      y_numeric <- as.integer(y) - 1  # Convert to 0/1
      output_units <- 1
      output_activation <- "sigmoid"
      loss <- "binary_crossentropy"
      metrics <- c("accuracy")
    } else {
      # Multiclass classification
      # One-hot encode the response
      y_onehot <- keras::to_categorical(as.integer(y) - 1)
      y_numeric <- y_onehot
      output_units <- length(levels(y))
      output_activation <- "softmax"
      loss <- "categorical_crossentropy"
      metrics <- c("accuracy")
    }
  } else {
    # Regression
    y_numeric <- y
    output_units <- 1
    output_activation <- "linear"
    loss <- "mse"
    metrics <- c("mae")
  }

  # Create sequential model
  model <- keras::keras_model_sequential()

  # Add input layer with appropriate shape
  model %>% keras::layer_dense(units = hidden_layers[1], activation = activation,
                               input_shape = ncol(X))

  # Add dropout for regularization
  if (dropout > 0) {
    model %>% keras::layer_dropout(rate = dropout)
  }

  # Add hidden layers
  for (i in 2:length(hidden_layers)) {
    model %>% keras::layer_dense(units = hidden_layers[i], activation = activation)

    if (dropout > 0) {
      model %>% keras::layer_dropout(rate = dropout)
    }
  }

  # Add output layer
  model %>% keras::layer_dense(units = output_units, activation = output_activation)

  # Compile the model
  model %>% keras::compile(
    optimizer = "adam",
    loss = loss,
    metrics = metrics
  )

  # Fit the model
  history <- model %>% keras::fit(
    x = X_scaled,
    y = y_numeric,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose,
    ...
  )

  # Store data for future predictions
  model_data <- list(
    model = model,
    history = history,
    X_means = X_means,
    X_sds = X_sds,
    formula = formula,
    is_classification = is_classification,
    levels = if (is_classification) levels(factor(y)) else NULL
  )

  return(model_data)
}

#' Predict using a deep learning model
#'
#' @param model A tidysl deep learning model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification), "class" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_deep <- function(model, new_data, type = "response", ...) {
  # Extract the deep learning model and associated data
  fit <- model$fit
  is_classification <- model$spec$is_classification
  formula <- model$spec$formula

  # Create model matrix for new data
  X_new <- stats::model.matrix(formula, data = new_data)[, -1, drop = FALSE]

  # Scale using the training data parameters
  X_new_scaled <- scale(X_new, center = fit$X_means, scale = fit$X_sds)

  # Make predictions
  raw_preds <- predict(fit$model, X_new_scaled)

  if (is_classification) {
    if (length(fit$levels) == 2) {
      # Binary classification
      if (type == "prob") {
        # Get probabilities
        prob_df <- tibble::tibble(
          !!fit$levels[1] := 1 - raw_preds[,1],
          !!fit$levels[2] := raw_preds[,1]
        )

        return(prob_df)
      } else if (type == "class" || type == "response") {
        # Get classes
        pred_classes <- ifelse(raw_preds[,1] > 0.5, fit$levels[2], fit$levels[1])
        pred_classes <- factor(pred_classes, levels = fit$levels)

        return(pred_classes)
      } else {
        stop("Invalid prediction type for deep learning classification. Use 'prob', 'class', or 'response'.", call. = FALSE)
      }
    } else {
      # Multiclass classification
      if (type == "prob") {
        # Convert to data frame with class probabilities
        prob_df <- as.data.frame(raw_preds)
        names(prob_df) <- fit$levels

        return(tibble::as_tibble(prob_df))
      } else if (type == "class" || type == "response") {
        # Get classes with highest probability
        pred_idx <- apply(raw_preds, 1, which.max)
        pred_classes <- fit$levels[pred_idx]
        pred_classes <- factor(pred_classes, levels = fit$levels)

        return(pred_classes)
      } else {
        stop("Invalid prediction type for deep learning classification. Use 'prob', 'class', or 'response'.", call. = FALSE)
      }
    }
  } else {
    # Regression predictions
    return(as.vector(raw_preds))
  }
}

#' Plot deep learning model training history
#'
#' @param model A tidysl deep learning model object
#' @param metrics Which metrics to plot (default: c("loss", "val_loss"))
#' @param ... Additional arguments
#' @return A ggplot object with training history
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @export
tl_plot_deep_history <- function(model, metrics = c("loss", "val_loss"), ...) {
  if (model$spec$method != "deep") {
    stop("Training history plot is only available for deep learning models", call. = FALSE)
  }

  # Extract training history
  history <- model$fit$history

  # Convert to data frame
  history_df <- tibble::tibble(
    epoch = seq_len(length(history$metrics$loss)),
    loss = history$metrics$loss
  )

  # Add validation metrics if available
  if (!is.null(history$metrics$val_loss)) {
    history_df$val_loss <- history$metrics$val_loss
  }

  # Add accuracy metrics if available
  if (!is.null(history$metrics$accuracy)) {
    history_df$accuracy <- history$metrics$accuracy
    if (!is.null(history$metrics$val_accuracy)) {
      history_df$val_accuracy <- history$metrics$val_accuracy
    }
  }

  # Add MAE metrics if available
  if (!is.null(history$metrics$mae)) {
    history_df$mae <- history$metrics$mae
    if (!is.null(history$metrics$val_mae)) {
      history_df$val_mae <- history$metrics$val_mae
    }
  }

  # Convert to long format for plotting
  history_long <- history_df %>%
    tidyr::pivot_longer(
      cols = -epoch,
      names_to = "metric",
      values_to = "value"
    ) %>%
    dplyr::filter(.data$metric %in% metrics)

  # Create the plot
  p <- ggplot2::ggplot(history_long, ggplot2::aes(x = epoch, y = value, color = metric)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Deep Learning Training History",
      x = "Epoch",
      y = "Value",
      color = "Metric"
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Plot deep learning model architecture
#'
#' @param model A tidysl deep learning model object
#' @param ... Additional arguments
#' @return A plot of the deep learning model architecture
#' @export
tl_plot_deep_architecture <- function(model, ...) {
  if (model$spec$method != "deep") {
    stop("Architecture plot is only available for deep learning models", call. = FALSE)
  }

  # Check if keras is installed
  tl_check_packages("keras")

  # Plot model architecture
  getFromNamespace("plot_model", "keras")(
    model$fit$model,
    show_shapes = TRUE,
    show_layer_names = TRUE,
    ...
  )
}

#' Tune a deep learning model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param hidden_layers_options List of vectors defining hidden layer configurations to try
#' @param learning_rates Learning rates to try (default: c(0.01, 0.001, 0.0001))
#' @param batch_sizes Batch sizes to try (default: c(16, 32, 64))
#' @param epochs Number of training epochs (default: 30)
#' @param validation_split Proportion of data for validation (default: 0.2)
#' @param ... Additional arguments
#' @return A list with the best model and tuning results
#' @export
tl_tune_deep <- function(data, formula, is_classification = FALSE,
                         hidden_layers_options = list(c(32), c(64, 32), c(128, 64, 32)),
                         learning_rates = c(0.01, 0.001, 0.0001),
                         batch_sizes = c(16, 32, 64),
                         epochs = 30, validation_split = 0.2, ...) {
  # Check if keras is installed
  tl_check_packages(c("keras", "tensorflow"))

  # Create grid of hyperparameters
  hyperparams <- expand.grid(
    hidden_layers_idx = seq_along(hidden_layers_options),
    learning_rate = learning_rates,
    batch_size = batch_sizes,
    val_loss = NA
  )

  # Train models with different hyperparameters
  for (i in 1:nrow(hyperparams)) {
    # Get current hyperparameters
    hidden_layers <- hidden_layers_options[[hyperparams$hidden_layers_idx[i]]]
    learning_rate <- hyperparams$learning_rate[i]
    batch_size <- hyperparams$batch_size[i]

    # Fit model with current hyperparameters
    model <- tryCatch({
      tl_fit_deep(
        data = data,
        formula = formula,
        is_classification = is_classification,
        hidden_layers = hidden_layers,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = validation_split,
        optimizer = keras::optimizer_adam(learning_rate = learning_rate),
        verbose = 0,
        ...
      )
    }, error = function(e) {
      message("Error fitting model with hyperparameters: ",
              "hidden_layers=", paste(hidden_layers, collapse=","),
              ", learning_rate=", learning_rate,
              ", batch_size=", batch_size)
      message("Error message: ", e$message)
      return(NULL)
    })

    # If model was successfully trained, store validation loss
    if (!is.null(model) && !is.null(model$history)) {
      val_losses <- model$history$metrics$val_loss
      hyperparams$val_loss[i] <- min(val_losses, na.rm = TRUE)
    }
  }

  # Find best hyperparameters (minimizing validation loss)
  best_idx <- which.min(hyperparams$val_loss)
  best_hidden_layers <- hidden_layers_options[[hyperparams$hidden_layers_idx[best_idx]]]
  best_learning_rate <- hyperparams$learning_rate[best_idx]
  best_batch_size <- hyperparams$batch_size[best_idx]

  # Train final model with best hyperparameters
  best_model <- tl_fit_deep(
    data = data,
    formula = formula,
    is_classification = is_classification,
    hidden_layers = best_hidden_layers,
    epochs = epochs,
    batch_size = best_batch_size,
    validation_split = validation_split,
    optimizer = keras::optimizer_adam(learning_rate = best_learning_rate),
    ...
  )

  # Return results
  return(list(
    model = best_model,
    best_hidden_layers = best_hidden_layers,
    best_learning_rate = best_learning_rate,
    best_batch_size = best_batch_size,
    tuning_results = hyperparams
  ))
}
