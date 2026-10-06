#' @title Deep Learning for tidylearn
#' @name tidylearn-deep-learning
#' @description Deep learning functionality using Keras/TensorFlow
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr mutate
NULL

#' Fit a deep learning model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param hidden_layers Vector of units in each hidden layer
#'   (default: c(32, 16))
#' @param activation Activation function for hidden layers
#'   (default: "relu")
#' @param dropout Dropout rate for regularization (default: 0.2)
#' @param epochs Number of training epochs (default: 30)
#' @param batch_size Batch size for training (default: 32)
#' @param validation_split Proportion of the rows held out to validate on,
#'   drawn at random (default: 0.2). Their row numbers are kept as
#'   \code{$validation_rows}. Pass \code{validation_data} through \code{...}
#'   to validate on data of your own instead.
#' @param learning_rate Optimizer learning rate. NULL (default) leaves
#'   keras's own adam default in place.
#' @param verbose Verbosity mode (0 = silent, 1 = progress bar,
#'   2 = one line per epoch) (default: 0)
#' @param ... Additional arguments to pass to keras's fit()
#' @param compute Compute tier. Either \code{"cpu"} (default) or
#'   \code{"gpu"}. GPU usage is handled automatically by the underlying
#'   tensorflow runtime when CUDA is configured; this argument is
#'   accepted for API consistency with the rest of \pkg{tidylearn} but
#'   does not itself change the keras model setup. The expectation is
#'   that the caller has already resolved the compute tier via
#'   \code{\link{tl_compute_advisor}} / \code{tl_resolve_compute}.
#' @return A fitted deep learning model
#' @keywords internal
tl_fit_deep <- function(data, formula,
                        is_classification = FALSE,
                        hidden_layers = c(32, 16),
                        activation = "relu",
                        dropout = 0.2,
                        epochs = 30, batch_size = 32,
                        validation_split = 0.2,
                        learning_rate = NULL,
                        verbose = 0, ...,
                        compute = "cpu") {
  # Check if keras is installed
  tl_check_packages(c("keras", "tensorflow"))
  dots <- list(...)

  # One model frame supplies both x and y, so a row dropped for a missing
  # value leaves both. model.matrix() applied na.omit by itself while the
  # response was read straight from data, so one missing predictor left y
  # a row longer than x, and keras trained on pairs shifted by one.
  frame <- stats::model.frame(formula, data = data, na.action = stats::na.omit)
  y <- unname(stats::model.response(frame))
  x_mat <- stats::model.matrix(attr(frame, "terms"), frame)
  x_mat <- x_mat[, colnames(x_mat) != "(Intercept)", drop = FALSE]

  # The rows of data the fit used, for reporting which were held out
  fitted_rows <- seq_len(nrow(data))
  omitted <- attr(frame, "na.action")
  if (!is.null(omitted)) {
    fitted_rows <- fitted_rows[-as.integer(omitted)]
  }

  # Normalize features. A constant column has sd 0, and scaling by it
  # turned the column into NaN, which keras carried into every
  # prediction. Scaled by 1 instead, it is 0 after centring and adds
  # nothing. A factor level no training row uses gives the same all-zero
  # column.
  x_means <- colMeans(x_mat)
  x_sds <- apply(x_mat, 2, stats::sd)
  x_sds[!is.finite(x_sds) | x_sds == 0] <- 1
  x_scaled <- scale(x_mat, center = x_means, scale = x_sds)

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
      # Multiclass classification. One-hot encode the response, with a
      # column for every class even if no complete row holds one.
      y_numeric <- keras::to_categorical(
        as.integer(y) - 1,
        num_classes = length(levels(y))
      )
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
  model |> keras::layer_dense(
    units = hidden_layers[1],
    activation = activation,
    input_shape = ncol(x_scaled)
  )

  # Add dropout for regularization
  if (dropout > 0) {
    model |> keras::layer_dropout(rate = dropout)
  }

  # Add the remaining hidden layers. seq_len() rather than 2:length():
  # with a single hidden layer, 2:1 counts backwards and adds a layer
  # with units = hidden_layers[1] followed by units = NA. The default
  # tuning grid includes single-layer candidates, so this was reachable.
  for (i in seq_len(length(hidden_layers) - 1L) + 1L) {
    model |> keras::layer_dense(
      units = hidden_layers[i],
      activation = activation
    )

    if (dropout > 0) {
      model |> keras::layer_dropout(rate = dropout)
    }
  }

  # Add output layer
  model |> keras::layer_dense(
    units = output_units,
    activation = output_activation
  )

  # Compile the model. The optimizer carries the learning rate, and
  # compile() is the only place it can be set -- passing an optimizer to
  # fit() does nothing, because the model has already been compiled.
  optimizer <- if (is.null(learning_rate)) {
    "adam"
  } else {
    keras::optimizer_adam(learning_rate = learning_rate)
  }

  model |> keras::compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
  )

  # keras's validation_split holds out the last rows as given, before any
  # shuffling. iris is sorted by species, so the default 0.2 held out
  # every virginica row: the model never trained on that class and was
  # validated on nothing else. Hold out a random set of rows instead,
  # unless the caller supplies validation data of their own.
  n <- nrow(x_scaled)
  held_out <- if (validation_split > 0 && is.null(dots$validation_data)) {
    sort(sample.int(n, floor(n * validation_split)))
  } else {
    integer(0)
  }
  train_rows <- setdiff(seq_len(n), held_out)
  rows_of <- function(v, idx) {
    if (is.matrix(v)) v[idx, , drop = FALSE] else v[idx]
  }

  fit_args <- list(
    x = x_scaled[train_rows, , drop = FALSE],
    y = rows_of(y_numeric, train_rows),
    epochs = epochs,
    batch_size = batch_size,
    verbose = verbose
  )
  if (length(held_out) > 0) {
    fit_args$validation_data <- list(
      x_scaled[held_out, , drop = FALSE],
      rows_of(y_numeric, held_out)
    )
  }

  # Fit the model
  history <- do.call(
    keras::fit,
    c(list(object = model), tl_override_args(fit_args, dots))
  )

  # Store data for future predictions
  model_data <- list(
    model = model,
    history = history,
    x_means = x_means,
    x_sds = x_sds,
    formula = formula,
    is_classification = is_classification,
    levels = if (is_classification) levels(y) else NULL,
    validation_rows = fitted_rows[held_out]
  )

  model_data
}

#' Predict using a deep learning model
#'
#' @param model A tidylearn deep learning model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default),
#'   "prob" (for classification), "class" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_deep <- function(model, new_data,
                            type = "response", ...) {
  # Extract the deep learning model and associated data
  fit <- model$fit
  is_classification <- model$spec$is_classification

  if (is_classification && !type %in% c("prob", "class", "response")) {
    stop(
      "Invalid prediction type for deep learning ",
      "classification. Use 'prob', 'class', ",
      "or 'response'.",
      call. = FALSE
    )
  }

  # The predictors alone, pinned to the training factor levels, with
  # incomplete rows kept in place. Built from the full formula, the design
  # matrix demanded the response column, which unlabelled data does not
  # have; dropped incomplete rows, so three rows in came back as two; and
  # took its factor levels from the new data, which changed the columns.
  x_new <- tl_predictor_matrix(
    model$spec$formula, new_data, xlev = model$spec$xlev
  )
  columns <- names(fit$x_means)
  missing_cols <- setdiff(columns, colnames(x_new))
  if (length(missing_cols) > 0) {
    stop(
      "New data is missing predictors used at fit time: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  x_new <- x_new[, columns, drop = FALSE]

  # Predict the complete rows, scaled with the training means and sds, and
  # leave NA in the others. keras prints a progress bar unless told not to.
  keep <- stats::complete.cases(x_new)
  n_outputs <- if (is_classification && length(fit$levels) > 2) {
    length(fit$levels)
  } else {
    1L
  }
  raw_preds <- matrix(NA_real_, nrow = nrow(x_new), ncol = n_outputs)
  if (any(keep)) {
    x_new_scaled <- scale(
      x_new[keep, , drop = FALSE],
      center = fit$x_means, scale = fit$x_sds
    )
    raw_preds[keep, ] <- predict(fit$model, x_new_scaled, verbose = 0)
  }

  if (!is_classification) {
    # Regression predictions
    return(as.vector(raw_preds))
  }

  probs <- tl_class_prob_matrix(raw_preds, fit$levels)
  if (type == "prob") {
    return(tibble::as_tibble(as.data.frame(probs)))
  }
  tl_class_from_probs(probs)
}

#' Plot deep learning model training history
#'
#' @param model A tidylearn deep learning model object
#' @param metrics Which metrics to plot
#'   (default: c("loss", "val_loss"))
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \dontrun{
#' if (requireNamespace("keras", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "deep", epochs = 5)
#'   tl_plot_deep_history(model)
#' }
#' }
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @export
tl_plot_deep_history <- function(model,
                                 metrics = c("loss",
                                             "val_loss"),
                                 ...) {
  if (model$spec$method != "deep") {
    stop(
      "Training history plot is only available ",
      "for deep learning models",
      call. = FALSE
    )
  }

  # Extract training history
  history <- model$fit$history

  # Convert to data frame
  history_df <- tibble::tibble(
    epoch = seq_along(history$metrics$loss),
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
      history_df$val_accuracy <-
        history$metrics$val_accuracy
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
  history_long <- history_df |>
    tidyr::pivot_longer(
      cols = -epoch,
      names_to = "metric",
      values_to = "value"
    ) |>
    dplyr::filter(.data$metric %in% metrics)

  # Create the plot
  p <- ggplot2::ggplot(
    history_long,
    ggplot2::aes(
      x = epoch, y = value, color = metric
    )
  ) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Deep Learning Training History",
      x = "Epoch",
      y = "Value",
      color = "Metric"
    ) +
    ggplot2::theme_minimal()

  p
}

#' Plot deep learning model architecture
#'
#' @param model A tidylearn deep learning model object
#' @param ... Additional arguments passed to the \code{plot()} method keras
#'   provides for its models, such as \code{to_file} or \code{dpi}.
#'   \code{show_shapes} and \code{show_layer_names} default to \code{TRUE}.
#' @return \code{NULL}, invisibly. Called for its side effect: keras draws
#'   the architecture diagram on the current graphics device, or writes it
#'   to \code{to_file}. keras renders it through the Python packages
#'   \code{pydot} and \code{graphviz}, and errors saying so when they are
#'   not installed.
#' @examples
#' \dontrun{
#' if (requireNamespace("keras", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "deep", epochs = 5)
#'   tl_plot_deep_architecture(model)
#' }
#' }
#' @export
tl_plot_deep_architecture <- function(model, ...) {
  if (model$spec$method != "deep") {
    stop(
      "Architecture plot is only available ",
      "for deep learning models",
      call. = FALSE
    )
  }

  # Check if keras is installed
  tl_check_packages("keras")

  # keras 2.x draws a model through the plot() method it registers for its
  # model class. It has no plot_model() function, so the lookup that was
  # here failed on every call with "object 'plot_model' not found".
  args <- tl_override_args(
    list(show_shapes = TRUE, show_layer_names = TRUE),
    list(...)
  )
  do.call(plot, c(list(model$fit$model), args))
}

#' Tune a deep learning model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem. \code{NULL} (default) reads it from the
#'   response, as \code{\link{tl_model}} does: a factor or character
#'   response is classification. \code{FALSE} with such a response is an
#'   error.
#' @param hidden_layers_options List of vectors defining hidden
#'   layer configurations to try
#' @param learning_rates Learning rates to try
#'   (default: c(0.01, 0.001, 0.0001))
#' @param batch_sizes Batch sizes to try
#'   (default: c(16, 32, 64))
#' @param epochs Number of training epochs (default: 30)
#' @param validation_split Proportion of the rows held out to score each
#'   configuration on, drawn at random (default: 0.2). Every configuration
#'   is scored on the same rows.
#' @param ... Additional arguments passed to keras's fit() for every
#'   configuration; \code{verbose} (default 0) replaces the value used
#'   otherwise.
#' @return A list with elements \code{model} (the best configuration refitted
#'   as a \code{tidylearn_model}, so \code{predict()} and the deep plots take
#'   it; the keras model is at \code{$model$fit$model}),
#'   \code{best_hidden_layers} (optimal layer configuration),
#'   \code{best_learning_rate}, \code{best_batch_size}, and
#'   \code{tuning_results} (a data frame of all hyperparameter combinations
#'   and their validation losses).
#' @examples
#' \dontrun{
#' if (requireNamespace("keras", quietly = TRUE)) {
#'   result <- tl_tune_deep(iris, Species ~ .,
#'     hidden_layers_options = list(c(10), c(10, 5)),
#'     learning_rates = c(0.01, 0.001), batch_sizes = c(32),
#'     epochs = 5)
#'   predict(result$model, iris[1:5, ])
#' }
#' }
#' @export
tl_tune_deep <- function(data, formula,
                         is_classification = NULL,
                         hidden_layers_options = list(
                           c(32), c(64, 32),
                           c(128, 64, 32)
                         ),
                         learning_rates = c(
                           0.01, 0.001, 0.0001
                         ),
                         batch_sizes = c(16, 32, 64),
                         epochs = 30,
                         validation_split = 0.2, ...) {
  # Check if keras is installed
  tl_check_packages(c("keras", "tensorflow"))

  formula <- tl_as_formula(formula)
  task <- tl_tuner_task(data, formula, is_classification)
  data <- task$data
  is_classification <- task$is_classification
  dots <- list(...)

  # tl_fit_deep() holds out a random set of rows to validate on. Drawing
  # it from one seed for every configuration scores them all on the same
  # rows, so their validation losses compare like with like.
  split_seed <- sample.int(.Machine$integer.max, 1L)
  with_split <- function(expr) {
    tl_local_seed(split_seed)
    expr
  }

  # Create grid of hyperparameters
  hyperparams <- expand.grid(
    hidden_layers_idx = seq_along(hidden_layers_options),
    learning_rate = learning_rates,
    batch_size = batch_sizes,
    val_loss = NA
  )

  # Train models with different hyperparameters
  for (i in seq_len(nrow(hyperparams))) {
    # Get current hyperparameters
    hl_idx <- hyperparams$hidden_layers_idx[i]
    hidden_layers <- hidden_layers_options[[hl_idx]]
    learning_rate <- hyperparams$learning_rate[i]
    batch_size <- hyperparams$batch_size[i]

    # Fit model with current hyperparameters. verbose = 0 is a default,
    # not a fixed value: alongside the caller's own it failed with
    # "formal argument matched by multiple actual arguments".
    model <- tryCatch({
      with_split(do.call(tl_fit_deep, c(
        list(
          data = data,
          formula = formula,
          is_classification = is_classification,
          hidden_layers = hidden_layers,
          epochs = epochs,
          batch_size = batch_size,
          validation_split = validation_split,
          learning_rate = learning_rate
        ),
        tl_override_args(list(verbose = 0), dots)
      )))
    }, error = function(e) {
      message(
        "Error fitting model with hyperparameters: ",
        "hidden_layers=",
        paste(hidden_layers, collapse = ","),
        ", learning_rate=", learning_rate,
        ", batch_size=", batch_size
      )
      message("Error message: ", e$message)
      NULL
    })

    # If model was successfully trained, store validation loss
    if (!is.null(model) && !is.null(model$history)) {
      val_losses <- model$history$metrics$val_loss
      hyperparams$val_loss[i] <- min(
        val_losses, na.rm = TRUE
      )
    }
  }

  # Find best hyperparameters (minimizing validation loss)
  # Every configuration can fail -- a bad argument forwarded through ...
  # reaches keras::fit() and each fit is caught individually, leaving
  # val_loss all NA. which.min() then returns integer(0) and the indexing
  # below failed with "attempt to select less than one element in
  # get1index", which says nothing about what went wrong.
  if (all(is.na(hyperparams$val_loss))) {
    stop(
      "No deep learning configuration could be fitted. The messages above ",
      "report why each one failed.",
      call. = FALSE
    )
  }

  best_idx <- which.min(hyperparams$val_loss)
  best_hl_idx <- hyperparams$hidden_layers_idx[best_idx]
  best_hidden_layers <-
    hidden_layers_options[[best_hl_idx]]
  best_learning_rate <-
    hyperparams$learning_rate[best_idx]
  best_batch_size <- hyperparams$batch_size[best_idx]

  # Refit the winner through tl_model(), on the same held-out rows. It was
  # the bare list tl_fit_deep() builds, and predict() on it failed with
  # "no applicable method for 'predict' applied to an object of class
  # \"list\"".
  best_model <- with_split(do.call(tl_model, c(
    list(
      data = data,
      formula = formula,
      method = "deep",
      hidden_layers = best_hidden_layers,
      epochs = epochs,
      batch_size = best_batch_size,
      validation_split = validation_split,
      learning_rate = best_learning_rate
    ),
    dots
  )))

  # Return results
  list(
    model = best_model,
    best_hidden_layers = best_hidden_layers,
    best_learning_rate = best_learning_rate,
    best_batch_size = best_batch_size,
    tuning_results = hyperparams
  )
}
