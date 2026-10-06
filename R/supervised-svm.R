#' @title Support Vector Machines for tidylearn
#' @name tidylearn-svm
#' @description SVM functionality for classification and
#'   regression
#' @importFrom e1071 svm tune
#' @importFrom stats predict
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr mutate
NULL

#' Fit a support vector machine model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param kernel Kernel function
#'   ("linear", "polynomial", "radial", "sigmoid")
#' @param cost Cost parameter (default: 1)
#' @param gamma Gamma parameter for kernels. Left to \code{e1071::svm()}
#'   when \code{NULL}, which uses 1 divided by the number of columns in
#'   the design matrix.
#' @param degree Degree for polynomial kernel (default: 3)
#' @param tune Logical indicating whether to tune
#'   hyperparameters (default: FALSE)
#' @param tune_folds Number of folds for cross-validation
#'   during tuning (default: 5)
#' @param ... Additional arguments to pass to svm(). \code{type} and
#'   \code{probability} replace the defaults chosen from the task. Case
#'   \code{weights} are refused: e1071 has none. So is an offset, which
#'   e1071 leaves out of the fit.
#' @return A fitted SVM model
#' @keywords internal
tl_fit_svm <- function(data, formula,
                       is_classification = FALSE,
                       kernel = "radial",
                       cost = 1, gamma = NULL,
                       degree = 3, tune = FALSE,
                       tune_folds = 5, ...) {
  # Check if e1071 is installed
  tl_check_packages("e1071")
  dots <- list(...)
  tl_refuse_offset(formula, data, dots, "svm", "e1071::svm()")

  # svm.default() swallows an argument it does not recognise, so weights
  # were accepted and ignored: a weighted fit was identical to the
  # unweighted one, while the model recorded that weights had been used.
  if ("weights" %in% names2(dots)) {
    stop(
      "Method \"svm\" cannot use case weights: e1071::svm() has no case ",
      "weights, so they would be ignored. For per-class weights in ",
      "classification, pass class.weights; for case weights, use a method ",
      "that applies them, such as \"tree\", \"boost\", \"nn\" or \"xgboost\".",
      call. = FALSE
    )
  }

  # No default gamma. e1071's own is 1 / ncol(design matrix), which
  # accounts for the dummy columns a factor predictor expands into.
  # Deriving one here from ncol(data) - 1 counted every column in the
  # frame rather than the formula's predictors, so `mpg ~ wt + hp` on
  # mtcars got a kernel width of 1/10 instead of 1/2, with nothing said.
  # Below, gamma is passed on only when the caller or the tuner set one.

  # The SVM type follows the task unless the caller names one
  svm_type <- if (!is.null(dots$type)) {
    dots$type
  } else if (is_classification) {
    "C-classification"
  } else {
    "eps-regression"
  }

  if (tune) {
    # Tune hyperparameters using cross-validation
    tune_ranges <- list(
      cost = c(0.1, 1, 10, 100)
    )
    if (kernel != "linear") {
      tune_ranges$gamma <- c(0.001, 0.01, 0.1, 1)
    }
    if (kernel == "polynomial") {
      tune_ranges$degree <- c(2, 3, 4)
    }

    tune_result <- e1071::tune(
      svm,
      train.x = formula,
      data = data,
      type = svm_type,
      kernel = kernel,
      ranges = tune_ranges,
      tunecontrol = e1071::tune.control(
        cross = tune_folds
      )
    )

    # Extract best parameters
    best_params <- tune_result$best.parameters
    cost <- best_params$cost
    if (kernel != "linear") {
      gamma <- best_params$gamma
    }
    if (kernel == "polynomial") {
      degree <- best_params$degree
    }

    # Store tuning results for later reference
    tuning_results <- tune_result
  } else {
    tuning_results <- NULL
  }

  # Fit the SVM model. probability and type are defaults, not fixed
  # values: alongside the caller's own, they failed with "formal argument
  # matched by multiple actual arguments".
  args <- list(
    formula = formula,
    data = data,
    type = svm_type,
    kernel = kernel,
    cost = cost,
    degree = degree,
    probability = is_classification
  )
  if (!is.null(gamma)) {
    args$gamma <- gamma
  }

  svm_model <- tl_restore_call_data(
    do.call(e1071::svm, tl_override_args(args, dots))
  )

  # Store tuning results if available
  if (!is.null(tuning_results)) {
    attr(svm_model, "tuning_results") <- tuning_results
  }

  svm_model
}

#' Predict using a support vector machine model
#'
#' @param model A tidylearn SVM model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default),
#'   "prob" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_svm <- function(model, new_data,
                           type = "response", ...) {
  # Get the SVM model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  # The columns the fitted model reads. predict.svm() applies its
  # na.action, na.omit, to the whole of newdata before it selects them, so
  # a missing value in the response or in a column the formula never used
  # dropped the row as well, and the shorter result stopped lining up with
  # new_data: airquality's Ozone ~ Temp + Wind returned 111 predictions
  # for 153 rows. Handed these columns alone, it has nothing else to drop.
  predictors <- all.vars(stats::delete.response(fit$terms))
  missing_cols <- setdiff(predictors, names(new_data))
  if (length(missing_cols) > 0) {
    stop(
      "New data is missing predictors used at fit time: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  predictor_data <- new_data[, predictors, drop = FALSE]

  # Incomplete rows are dropped here and put back as NA afterwards, so
  # row i of the output still describes row i of new_data
  keep <- tl_complete_predictor_rows(model$spec$formula, predictor_data)
  predict_data <- predictor_data[keep, , drop = FALSE]

  if (is_classification) {
    if (type == "prob") {
      # e1071 records the probability flag as $compprob; $probability is
      # the argument name, not a slot on the fitted object
      if (!isTRUE(fit$compprob)) {
        stop(
          "Probability estimates not available. ",
          "Refit the model with probability = TRUE.",
          call. = FALSE
        )
      }

      # Get class probabilities. predict.svm() fails on zero rows, which
      # is what is left when every row misses a predictor.
      probs <- if (any(keep)) {
        attr(
          predict(
            fit, newdata = predict_data,
            probability = TRUE, ...
          ),
          "probabilities"
        )
      } else {
        matrix(numeric(0), nrow = 0, ncol = length(fit$levels),
               dimnames = list(NULL, fit$levels))
      }

      # e1071 orders the probability columns by its own internal class
      # ordering; align them with the response factor levels so every
      # method returns the same column order
      class_levels <- model$spec$response_levels %||% colnames(probs)
      if (setequal(class_levels, colnames(probs))) {
        probs <- probs[, class_levels, drop = FALSE]
      }

      probs <- tl_realign_prob_matrix(probs, keep)
      tibble::as_tibble(as.data.frame(probs))
    } else if (type == "class" || type == "response") {
      # Get predicted classes
      preds <- predict(fit, newdata = predict_data, ...)
      tl_realign_predictions(preds, keep)
    } else {
      stop(
        "Invalid prediction type for SVM ",
        "classification. Use 'prob', 'class', ",
        "or 'response'.",
        call. = FALSE
      )
    }
  } else {
    # Regression predictions
    preds <- predict(fit, newdata = predict_data, ...)
    tl_realign_predictions(preds, keep)
  }
}

#' Plot SVM decision boundary
#'
#' @param model A tidylearn SVM model object
#' @param x_var Name of the x-axis variable. Defaults to the first numeric
#'   predictor in the model's formula.
#' @param y_var Name of the y-axis variable. Defaults to the next numeric
#'   predictor in the model's formula.
#' @param grid_size Number of points in each dimension
#'   for the grid (default: 100)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object. The other predictors are
#'   held at their mean, or their most frequent level. A two-class model
#'   fitted with probabilities also gets the 0.5 probability contour.
#' @importFrom ggplot2 ggplot aes geom_point geom_contour
#'   scale_fill_gradient2 labs theme_minimal
#' @examples
#' \donttest{
#' if (requireNamespace("e1071", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "svm")
#'   tl_plot_svm_boundary(model,
#'     x_var = "Sepal.Length", y_var = "Sepal.Width")
#' }
#' }
#' @export
tl_plot_svm_boundary <- function(model,
                                 x_var = NULL,
                                 y_var = NULL,
                                 grid_size = 100,
                                 ...) {
  if (model$spec$method != "svm") {
    stop(
      "Decision boundary plot is only available ",
      "for SVM models",
      call. = FALSE
    )
  }

  if (!model$spec$is_classification) {
    stop(
      "Decision boundary plot is only available ",
      "for classification models",
      call. = FALSE
    )
  }

  # Get original data
  data <- model$data
  formula <- model$spec$formula
  # The legend names the response as the formula writes it
  response_label <- deparse1(formula[[2L]])

  # The predictors the model was fitted on. The axes used to default to
  # the first two numeric columns of the data, so Species ~ Petal.Length +
  # Petal.Width was drawn over the sepal columns the model never saw, with
  # one predicted class across the whole grid.
  predictor_vars <- intersect(get_formula_vars(formula, data), names(data))

  # Fill in whichever axis was not named, from the numeric predictors
  if (is.null(x_var) || is.null(y_var)) {
    numeric_vars <- predictor_vars[
      vapply(data[predictor_vars], is.numeric, logical(1))
    ]
    candidates <- setdiff(numeric_vars, c(x_var, y_var))
    if (is.null(x_var)) {
      x_var <- candidates[1]
      candidates <- candidates[-1]
    }
    if (is.null(y_var)) {
      y_var <- candidates[1]
    }
    if (is.na(x_var) || is.na(y_var)) {
      stop(
        "At least two numeric predictor variables are ",
        "required for decision boundary plot",
        call. = FALSE
      )
    }
  }

  # Check if variables exist
  if (!x_var %in% names(data) ||
        !y_var %in% names(data)) {
    stop(
      "Variables not found in the model data",
      call. = FALSE
    )
  }

  # Create grid for prediction
  x_range <- range(data[[x_var]], na.rm = TRUE)
  y_range <- range(data[[y_var]], na.rm = TRUE)

  x_grid <- seq(
    x_range[1], x_range[2], length.out = grid_size
  )
  y_grid <- seq(
    y_range[1], y_range[2], length.out = grid_size
  )

  grid_data <- expand.grid(x = x_grid, y = y_grid)
  names(grid_data) <- c(x_var, y_var)

  # Hold the other predictors at their mean, or their most frequent level
  other_vars <- setdiff(predictor_vars, c(x_var, y_var))
  for (var in other_vars) {
    if (is.factor(data[[var]]) ||
          is.character(data[[var]])) {
      # For categorical variables, use most frequent value
      most_freq <- names(
        sort(table(data[[var]]), decreasing = TRUE)[1]
      )
      grid_data[[var]] <- if (is.factor(data[[var]])) {
        factor(most_freq, levels = levels(data[[var]]))
      } else {
        most_freq
      }
    } else {
      # For continuous variables, use mean
      grid_data[[var]] <- mean(
        data[[var]], na.rm = TRUE
      )
    }
  }

  # Make predictions on the grid -- always produce predicted class labels
  preds <- predict(model$fit, newdata = grid_data)
  grid_data$pred_class <- as.character(preds)

  # For binary classification, also get numeric probability for contour
  # line. e1071 stores the type as a code -- 0 is C-classification, 1
  # nu-classification -- and the probability flag as $compprob. Comparing
  # $type with the string and reading $probability, which is an argument
  # name rather than a slot, made this FALSE for every model, so the
  # contour was never drawn.
  has_numeric_pred <- FALSE
  if (model$fit$type %in% c(0, 1) && isTRUE(model$fit$compprob)) {
    probs <- attr(
      predict(model$fit, newdata = grid_data, probability = TRUE),
      "probabilities"
    )
    if (!is.null(probs) && ncol(probs) == 2) {
      # The positive class, the second level, whatever order e1071 keeps
      positive <- model$spec$response_levels[2]
      if (is.null(positive) || !positive %in% colnames(probs)) {
        positive <- colnames(probs)[2]
      }
      grid_data$pred_prob <- unname(probs[, positive])
      has_numeric_pred <- TRUE
    }
  }

  # Create plot using geom_raster for decision regions
  p <- ggplot2::ggplot() +
    ggplot2::geom_raster(
      data = grid_data,
      ggplot2::aes(
        x = .data[[x_var]],
        y = .data[[y_var]],
        fill = .data[["pred_class"]]
      ),
      alpha = 0.3
    )

  # Add decision boundary contour line for binary classification
  if (has_numeric_pred) {
    p <- p + ggplot2::geom_contour(
      data = grid_data,
      ggplot2::aes(
        x = .data[[x_var]],
        y = .data[[y_var]],
        z = .data[["pred_prob"]]
      ),
      breaks = 0.5,
      color = "black",
      linewidth = 1
    )
  }

  # Add original data points, coloured by the response the formula
  # computes. Coloured by the raw column, factor(mpg > 20) ~ . drew 25
  # shades of mpg rather than its two classes.
  point_data <- data
  point_data$.observed <- tl_normalise_response(
    tl_formula_response(formula, data)
  )
  p <- p +
    ggplot2::geom_point(
      data = point_data,
      ggplot2::aes(
        x = .data[[x_var]],
        y = .data[[y_var]],
        color = .data[[".observed"]]
      ),
      size = 3,
      alpha = 0.7
    ) +
    ggplot2::labs(
      title = "SVM Decision Boundary",
      x = x_var,
      y = y_var,
      color = response_label,
      fill = "Predicted Class"
    ) +
    ggplot2::theme_minimal()

  p
}

#' Plot SVM tuning results
#'
#' @param model A tidylearn SVM model object
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @importFrom ggplot2 ggplot aes geom_tile
#'   scale_fill_gradient2 labs theme_minimal
#' @examples
#' \donttest{
#' if (requireNamespace("e1071", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "svm",
#'     kernel = "linear", tune = TRUE, tune_folds = 2)
#'   tl_plot_svm_tuning(model)
#' }
#' }
#' @export
tl_plot_svm_tuning <- function(model, ...) {
  if (model$spec$method != "svm") {
    stop(
      "Tuning plot is only available for SVM models",
      call. = FALSE
    )
  }

  # Check if tuning results are available
  tuning_results <- attr(model$fit, "tuning_results")
  if (is.null(tuning_results)) {
    stop(
      "No tuning results available. ",
      "Fit the model with tune = TRUE.",
      call. = FALSE
    )
  }

  # Extract performance data
  perf_data <- tuning_results$performances

  # Create appropriate plot based on parameters tuned
  if ("gamma" %in% names(perf_data) &&
        "cost" %in% names(perf_data)) {
    # Plot gamma vs cost
    p <- ggplot2::ggplot(
      perf_data,
      ggplot2::aes(
        x = gamma, y = cost, fill = error
      )
    ) +
      ggplot2::geom_tile() +
      ggplot2::scale_x_log10() +
      ggplot2::scale_y_log10() +
      ggplot2::scale_fill_gradient2(
        low = "blue", high = "red",
        mid = "white",
        midpoint = mean(perf_data$error)
      ) +
      ggplot2::labs(
        title = "SVM Parameter Tuning",
        subtitle = paste0(
          "Best parameters: gamma = ",
          tuning_results$best.parameters$gamma,
          ", cost = ",
          tuning_results$best.parameters$cost
        ),
        x = "Gamma (log scale)",
        y = "Cost (log scale)",
        fill = "Error"
      ) +
      ggplot2::theme_minimal()
  } else if ("cost" %in% names(perf_data)) {
    # Plot cost only
    p <- ggplot2::ggplot(
      perf_data,
      ggplot2::aes(x = cost, y = error)
    ) +
      ggplot2::geom_line() +
      ggplot2::geom_point() +
      ggplot2::scale_x_log10() +
      ggplot2::labs(
        title = "SVM Parameter Tuning",
        subtitle = paste0(
          "Best parameter: cost = ",
          tuning_results$best.parameters$cost
        ),
        x = "Cost (log scale)",
        y = "Error"
      ) +
      ggplot2::theme_minimal()
  } else {
    # Generic plot of all parameters
    p <- ggplot2::ggplot(
      perf_data,
      ggplot2::aes(
        x = seq_len(nrow(perf_data)), y = error
      )
    ) +
      ggplot2::geom_line() +
      ggplot2::geom_point() +
      ggplot2::labs(
        title = "SVM Parameter Tuning",
        subtitle = paste0(
          "Best parameters: ",
          paste(
            names(tuning_results$best.parameters),
            tuning_results$best.parameters,
            sep = " = ", collapse = ", "
          )
        ),
        x = "Parameter Combination",
        y = "Error"
      ) +
      ggplot2::theme_minimal()
  }

  p
}
