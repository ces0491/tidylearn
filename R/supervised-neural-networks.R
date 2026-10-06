#' @title Neural Networks for tidylearn
#' @name tidylearn-neural-networks
#' @description Neural network functionality for classification and regression
#' @importFrom nnet nnet
#' @importFrom stats predict
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr mutate
NULL

#' Fit a neural network model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param size Number of units in the hidden layer (default: 5)
#' @param decay Weight decay parameter (default: 0)
#' @param maxit Maximum number of iterations (default: 100)
#' @param trace Logical; whether to print progress (default: FALSE)
#' @param ... Additional arguments to pass to nnet(), including case
#'   \code{weights}. For regression, \code{linout} replaces the default
#'   \code{TRUE}. An offset is refused: nnet leaves it out of the fit.
#' @return A fitted neural network model
#' @keywords internal
tl_fit_nn <- function(data, formula, is_classification = FALSE,
                      size = 5, decay = 0, maxit = 100, trace = FALSE, ...) {
  # Check if nnet is installed
  tl_check_packages("nnet")
  tl_refuse_offset(formula, data, list(...), "nn", "nnet()")

  # Get response variable
  response_var <- all.vars(formula)[1]

  # For classification, ensure response is a factor. Only a bare column is
  # converted: a computed response such as cut(mpg, 3) reads its column,
  # and handed a factor, it failed.
  if (is_classification && is.name(formula[[2L]]) &&
        !is.factor(data[[response_var]])) {
    data[[response_var]] <- factor(data[[response_var]])
  }

  args <- list(
    formula = formula,
    data = data,
    size = size,
    decay = decay,
    maxit = maxit,
    trace = trace
  )

  # The error criterion of a classification fit is deliberately left to
  # nnet. nnet.formula() already chooses it from the response:
  # cross-entropy for a two-level factor, softmax for three or more. Naming
  # entropy = TRUE here reached nnet.default() through `...` alongside the
  # one nnet.formula() supplies itself, and a two-class fit died with
  # "formal argument 'entropy' matched by multiple actual arguments".
  # Multiclass survived only because nnet.default() sets entropy <- FALSE
  # whenever softmax is on, so the argument it collided with was never
  # there.
  if (!is_classification) {
    args$linout <- TRUE  # Linear output for regression
  }

  # nnet() evaluates weights and subset inside its own model frame, which a
  # value forwarded through ... cannot reach: weights failed with "..1 used
  # in an incorrect context". So the call is built from values, and the
  # caller's arguments replace the defaults above -- linout included --
  # rather than colliding with them.
  nn_model <- tl_fit_by_value(
    nnet::nnet, "nnet", tl_override_args(args, list(...))
  )

  # NeuralNetTools::plotnet() evaluates `mod_in$call$formula` for any net
  # with a single output unit -- every regression and two-class fit. A
  # call that recorded the symbol `formula` resolved it to stats::formula,
  # the function, and plotting failed with "cannot coerce type 'closure'
  # to vector of type 'character'". The stored call has to hold the
  # formula itself.
  nn_model$call$formula <- formula

  nn_model
}

#' Class probabilities as an n x k matrix
#'
#' A two-class network has one output unit, the probability of the second
#' level -- the positive class -- and the first level's is its complement.
#'
#' @param raw The network's output: a vector, or a matrix with one column
#'   per output unit
#' @param class_levels The classes, in level order
#' @return A numeric matrix with one named column per class
#' @keywords internal
#' @noRd
tl_class_prob_matrix <- function(raw, class_levels) {
  raw <- as.matrix(raw)
  if (ncol(raw) == 1L && length(class_levels) == 2L) {
    raw <- cbind(1 - raw[, 1], raw[, 1])
  }
  if (ncol(raw) != length(class_levels)) {
    stop(
      "The network returned ", ncol(raw), " output columns for ",
      length(class_levels), " classes.",
      call. = FALSE
    )
  }
  dimnames(raw) <- list(NULL, class_levels)
  raw
}

#' The most probable class in each row of a probability matrix
#'
#' A row whose probabilities are missing -- what predict.nnet() and the
#' deep path leave where a predictor is missing -- gets an NA class.
#' \code{apply(probs, 1, which.max)} returned \code{integer(0)} for such a
#' row, and indexing the levels with the list that made failed with
#' "invalid subscript type 'list'". \code{max.col()} returns NA instead.
#'
#' @param probs Matrix from \code{tl_class_prob_matrix()}
#' @return A factor with the matrix's column names as levels
#' @keywords internal
#' @noRd
tl_class_from_probs <- function(probs) {
  class_levels <- colnames(probs)
  idx <- max.col(probs, ties.method = "first")
  factor(class_levels[idx], levels = class_levels)
}

#' Settle a tuner's task from its response, as tl_model() does
#'
#' The tuners took \code{is_classification} on trust and defaulted it to
#' FALSE, so a factor response was tuned as a regression on its integer
#' codes: silently for xgboost, and failing with "NA/NaN argument" for
#' nnet. The response was not cleaned either, so a subset that still
#' declared a class it no longer held was tuned with that class as well.
#'
#' The response is the one the formula computes, so \code{factor(am) ~ .}
#' is a classification although \code{am} is numeric.
#'
#' @param data Training data
#' @param formula The model formula
#' @param is_classification The caller's flag: \code{NULL} to read the task
#'   from the response, or \code{TRUE} or \code{FALSE}
#' @return A list: \code{data}, whose classification response, when it is
#'   a column of its own, is a factor of the classes it holds; and
#'   \code{is_classification}
#' @keywords internal
#' @noRd
tl_tuner_task <- function(data, formula, is_classification) {
  lhs <- formula[[2L]]
  response_label <- if (is.name(lhs)) as.character(lhs) else deparse1(lhs)
  y <- tl_formula_response(formula, data)
  categorical <- is.factor(y) || is.character(y)

  if (is.null(is_classification)) {
    is_classification <- categorical
  } else if (!is.logical(is_classification) ||
               length(is_classification) != 1L ||
               is.na(is_classification)) {
    stop(
      "'is_classification' must be TRUE, FALSE or NULL; got ",
      tl_describe_value(is_classification), ".",
      call. = FALSE
    )
  }

  if (!is_classification && categorical) {
    stop(
      "is_classification = FALSE, but '", response_label, "' is a ",
      if (is.factor(y)) "factor" else "character vector",
      ". A regression would fit its classes' integer codes. Leave ",
      "is_classification out to classify it, or pass a numeric response.",
      call. = FALSE
    )
  }

  # A computed response is left to the formula: writing factor(am) back
  # over am would change what the formula computes
  if (is_classification && is.name(lhs)) {
    data[[response_label]] <- tl_normalise_response(y)
  }

  list(data = data, is_classification = is_classification)
}

#' Predict using a neural network model
#'
#' @param model A tidylearn neural network model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default),
#'   "prob" (for classification), "class" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_nn <- function(model, new_data, type = "response", ...) {
  # Get the neural network model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  if (!is_classification) {
    # Regression predictions
    return(as.vector(predict(fit, newdata = new_data, type = "raw", ...)))
  }

  if (!type %in% c("prob", "class", "response")) {
    stop(
      "Invalid prediction type for neural networks. ",
      "Use 'prob', 'class', or 'response'.",
      call. = FALSE
    )
  }

  # The classes of the response the formula computes. The raw column of
  # factor(mpg > 20) ~ . held 25 values, and a one-output network could
  # not be read as 25 classes.
  class_levels <- model$spec$response_levels %||%
    levels(tl_normalise_response(
      tl_formula_response(model$spec$formula, model$data)
    ))

  # predict.nnet() keeps a row with a missing predictor as a row of NA, so
  # the probabilities, and the classes read from them, stay aligned with
  # new_data. "response" is the class, as for the other classifiers.
  probs <- tl_class_prob_matrix(
    predict(fit, newdata = new_data, type = "raw", ...),
    class_levels
  )
  if (type == "prob") {
    return(tibble::as_tibble(as.data.frame(probs)))
  }
  tl_class_from_probs(probs)
}

#' Plot neural network architecture
#'
#' @param model A tidylearn neural network model object
#' @param ... Additional arguments
#' @return The return value of \code{\link[NeuralNetTools]{plotnet}}, called for
#'   its side effect of drawing the network diagram, or \code{NULL} if the
#'   \pkg{NeuralNetTools} package is not installed.
#' @examples
#' \donttest{
#' if (requireNamespace("NeuralNetTools", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "nn", size = 3)
#'   tl_plot_nn_architecture(model)
#' }
#' }
#' @importFrom ggplot2 ggplot aes geom_segment geom_point geom_text theme_void
#' @export
tl_plot_nn_architecture <- function(model, ...) {
  if (model$spec$method != "nn") {
    stop(
      "Neural network architecture plot is only ",
      "available for neural network models",
      call. = FALSE
    )
  }

  # Check if NeuralNetTools is installed
  if (!requireNamespace("NeuralNetTools", quietly = TRUE)) {
    message(
      "Package 'NeuralNetTools' is required for ",
      "neural network visualization. Please install it."
    )
    return(NULL)
  }

  # Plot using NeuralNetTools
  NeuralNetTools::plotnet(model$fit, ...)
}

#' Tune a neural network model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem. \code{NULL} (default) reads it from the
#'   response, as \code{\link{tl_model}} does: a factor or character
#'   response is classification. \code{FALSE} with such a response is an
#'   error.
#' @param sizes Vector of hidden layer sizes to try
#' @param decays Vector of weight decay parameters to try
#' @param folds Number of cross-validation folds (default: 5)
#' @param ... Additional arguments to pass to nnet(). \code{maxit}
#'   (default 100) and \code{trace} (default \code{FALSE}) replace the
#'   values used otherwise. Arguments with one value per row --
#'   \code{weights}, \code{subset}, \code{offset}, \code{foldid},
#'   \code{strata} -- are refused, since each fold fits a subset of the
#'   rows.
#' @return A list with elements \code{model} (the best fitted \code{nnet}
#'   model), \code{best_size} (optimal hidden-layer size), \code{best_decay}
#'   (optimal weight decay), and \code{tuning_results} (a data frame of all
#'   parameter combinations and their cross-validated errors: the
#'   misclassification rate for classification, the mean squared error for
#'   regression).
#' @export
#' @examples
#' \donttest{
#' tuned <- tl_tune_nn(iris, Species ~ .,
#'   is_classification = TRUE,
#'   sizes = c(2, 5), decays = c(0, 0.01), folds = 3)
#'
#' tuned$best_size
#' tuned$best_decay
#' tuned$tuning_results
#'
#' # The grid this searched, drawn as a heatmap
#' tl_plot_nn_tuning(tuned)
#' }
tl_tune_nn <- function(data, formula, is_classification = NULL,
                       sizes = c(1, 2, 5, 10), decays = c(0, 0.001, 0.01, 0.1),
                       folds = 5, ...) {
  # Check if nnet is installed
  tl_check_packages("nnet")

  formula <- tl_as_formula(formula)
  task <- tl_tuner_task(data, formula, is_classification)
  data <- task$data
  is_classification <- task$is_classification

  # A per-row argument went whole into every fold, a subset of the rows,
  # and nnet failed on "variable lengths differ"
  tl_check_per_row_args(names2(list(...)), "tl_tune_nn()")

  # maxit and trace are defaults, not fixed values: set alongside ..., the
  # caller's own failed with "formal argument matched by multiple actual
  # arguments"
  fit_args <- tl_override_args(list(maxit = 100, trace = FALSE), list(...))

  # Create cross-validation splits
  cv_splits <- rsample::vfold_cv(data, v = folds)

  # Initialize results
  tune_results <- expand.grid(
    size = sizes,
    decay = decays,
    error = NA
  )

  # Loop through parameter combinations
  for (i in seq_len(nrow(tune_results))) {
    # Cross-validation for this parameter combination
    cv_errors <- numeric(folds)

    for (j in seq_len(folds)) {
      # Get training and testing data for this fold
      train_data <- rsample::analysis(cv_splits$splits[[j]])
      test_data <- rsample::assessment(cv_splits$splits[[j]])

      # Train neural network
      nn <- do.call(tl_fit_nn, c(
        list(
          data = train_data,
          formula = formula,
          is_classification = is_classification,
          size = tune_results$size[i],
          decay = tune_results$decay[i]
        ),
        fit_args
      ))

      # Score the fold through the path predict() uses. Reading
      # predict.nnet()'s raw output here instead, a two-class network's
      # n x 1 matrix failed the is.vector() test, and which.max() over its
      # one column chose the first class for every row: each candidate
      # scored the share of the second class, and the first in the grid
      # always won.
      fold_model <- list(
        fit = nn,
        data = train_data,
        spec = list(
          is_classification = is_classification,
          formula = formula,
          response_levels = nn$lev
        )
      )
      preds <- tl_predict_nn(fold_model, test_data, type = "class")
      # The response the formula computes: scored against the raw column,
      # log(y) ~ x was judged by how far its log-scale predictions fell
      # from y itself
      actuals <- tl_formula_response(formula, test_data)

      if (is_classification) {
        # Classification error, over the rows the fold's model can score
        aligned <- tl_align_classes(actuals, nn$lev)
        scored <- aligned$keep & !is.na(preds)
        cv_errors[j] <- mean(preds[scored] != aligned$actuals[scored])
      } else {
        # For regression, use MSE
        cv_errors[j] <- mean((preds - actuals)^2, na.rm = TRUE)
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
  best_model <- do.call(tl_fit_nn, c(
    list(
      data = data,
      formula = formula,
      is_classification = is_classification,
      size = best_size,
      decay = best_decay
    ),
    fit_args
  ))

  # Return results
  list(
    model = best_model,
    best_size = best_size,
    best_decay = best_decay,
    tuning_results = tune_results
  )
}

#' Plot a neural network tuning grid
#'
#' Draws the size-by-decay grid as a heatmap of cross-validated error.
#'
#' @param model The list returned by \code{\link{tl_tune_nn}}, not a fitted
#'   model — the grid it draws lives in that list's
#'   \code{$tuning_results}. Anything without that element is refused.
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @export
#' @examples
#' \donttest{
#' tuned <- tl_tune_nn(iris, Species ~ .,
#'   is_classification = TRUE,
#'   sizes = c(2, 5), decays = c(0, 0.01), folds = 3)
#'
#' # The tuning result itself, not tuned$model
#' tl_plot_nn_tuning(tuned)
#' }
tl_plot_nn_tuning <- function(model, ...) {
  if (!is.list(model) || !"tuning_results" %in% names(model)) {
    stop("This function requires the output from tl_tune_nn()", call. = FALSE)
  }

  # Extract tuning results
  tune_results <- model$tuning_results

  # Create data for heatmap
  heatmap_data <- tune_results |>
    dplyr::mutate(size = factor(.data$size), decay = factor(.data$decay))

  # Create heatmap
  p <- ggplot2::ggplot(
    heatmap_data,
    ggplot2::aes(x = decay, y = size, fill = error)
  ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = round(error, 4)), color = "white") +
    ggplot2::scale_fill_gradient(low = "blue", high = "red") +
    ggplot2::labs(
      title = "Neural Network Parameter Tuning",
      subtitle = paste0(
        "Best parameters: size = ", model$best_size,
        ", decay = ", model$best_decay
      ),
      x = "Weight Decay",
      y = "Hidden Layer Size",
      fill = "Error"
    ) +
    ggplot2::theme_minimal()

  p
}
