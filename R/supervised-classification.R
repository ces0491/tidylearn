#' @title Classification Functions for tidysl
#' @name tidysl-classification
#' @description Logistic regression and classification metrics functionality
#' @importFrom stats glm predict binomial
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr %>% mutate
#' @importFrom ROCR prediction performance
#' @importFrom ggplot2 ggplot aes geom_line geom_abline labs theme_minimal
NULL

#' Fit a logistic regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param ... Additional arguments to pass to glm()
#' @return A fitted logistic regression model
#' @keywords internal
tl_fit_logistic <- function(data, formula, ...) {
  # Ensure response variable is a factor
  response_var <- all.vars(formula)[1]
  if (!is.factor(data[[response_var]])) {
    warning("Converting response variable to factor for logistic regression", call. = FALSE)
    data[[response_var]] <- factor(data[[response_var]])
  }

  # Fit the logistic regression model
  glm_model <- stats::glm(formula, data = data, family = stats::binomial(), ...)

  return(glm_model)
}

#' Predict using a logistic regression model
#'
#' @param model A tidysl logistic model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "prob" (default), "class", "response"
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_logistic <- function(model, new_data, type = "prob", ...) {
  # Extract necessary information
  response_var <- model$spec$response_var

  # Get original response variable from training data
  original_response <- model$data[[response_var]]
  if (!is.factor(original_response)) {
    original_response <- factor(original_response)
  }
  class_levels <- levels(original_response)

  # Make predictions based on the type
  if (type == "response") {
    # Get the linear predictor
    preds <- stats::predict(model$fit, newdata = new_data, type = "response", ...)
    return(preds)
  } else if (type == "prob") {
    # Binary classification
    if (length(class_levels) == 2) {
      # Get probabilities for the positive class
      pos_probs <- stats::predict(model$fit, newdata = new_data, type = "response", ...)
      neg_probs <- 1 - pos_probs

      # Create a data frame with probabilities for each class
      prob_df <- tibble::tibble(
        !!class_levels[1] := neg_probs,
        !!class_levels[2] := pos_probs
      )

      return(prob_df)
    } else {
      # Multiclass classification (requires multinomial logistic regression)
      stop("Multiclass logistic regression not currently implemented", call. = FALSE)
    }
  } else if (type == "class") {
    # Binary classification
    if (length(class_levels) == 2) {
      # Get probabilities for the positive class
      pos_probs <- stats::predict(model$fit, newdata = new_data, type = "response", ...)

      # Classify based on probability > 0.5
      pred_classes <- ifelse(pos_probs > 0.5, class_levels[2], class_levels[1])
      pred_classes <- factor(pred_classes, levels = class_levels)

      return(pred_classes)
    } else {
      # Multiclass classification (requires multinomial logistic regression)
      stop("Multiclass logistic regression not currently implemented", call. = FALSE)
    }
  } else {
    stop("Invalid prediction type. Use 'prob', 'class', or 'response'.", call. = FALSE)
  }
}

#' Plot ROC curve for a classification model
#'
#' @param model A tidysl classification model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object with ROC curve
#' @importFrom ROCR prediction performance
#' @importFrom ggplot2 ggplot aes geom_line geom_abline labs theme_minimal
#' @keywords internal
tl_plot_roc <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual values
  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) {
    actuals <- factor(actuals)
  }

  # For binary classification
  if (length(levels(actuals)) == 2) {
    # Get probabilities
    probs <- predict(model, new_data, type = "prob")
    pos_class <- levels(actuals)[2]
    pos_probs <- probs[[pos_class]]

    # Convert actuals to binary (0/1)
    binary_actuals <- as.integer(actuals == pos_class)

    # Create ROC curve
    pred_obj <- ROCR::prediction(pos_probs, binary_actuals)
    perf <- ROCR::performance(pred_obj, "tpr", "fpr")

    # Calculate AUC
    auc <- unlist(ROCR::performance(pred_obj, "auc")@y.values)

    # Create data frame for plotting
    roc_data <- tibble::tibble(
      fpr = unlist(perf@x.values),
      tpr = unlist(perf@y.values)
    )

    # Create the plot
    p <- ggplot2::ggplot(roc_data, ggplot2::aes(x = fpr, y = tpr)) +
      ggplot2::geom_line(color = "blue", size = 1) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      ggplot2::labs(
        title = "ROC Curve",
        subtitle = paste0("AUC = ", round(auc, 3)),
        x = "False Positive Rate",
        y = "True Positive Rate"
      ) +
      ggplot2::coord_fixed() +
      ggplot2::theme_minimal()

    return(p)
  } else {
    # Multiclass ROC (one-vs-rest)
    stop("Multiclass ROC curves not currently implemented", call. = FALSE)
  }
}

#' Plot confusion matrix for a classification model
#'
#' @param model A tidysl classification model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object with confusion matrix
#' @importFrom ggplot2 ggplot aes geom_tile geom_text scale_fill_gradient
#' @keywords internal
tl_plot_confusion <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual values
  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) {
    actuals <- factor(actuals)
  }

  # Get predicted classes
  predicted <- predict(model, new_data, type = "class")$prediction

  # Create confusion matrix
  cm <- table(Actual = actuals, Predicted = predicted)

  # Convert to data frame for plotting
  cm_df <- as.data.frame(as.table(cm))

  # Calculate percentages
  cm_df$percentage <- cm_df$Freq / sum(cm_df$Freq) * 100

  # Create the plot
  p <- ggplot2::ggplot(cm_df, ggplot2::aes(x = Predicted, y = Actual)) +
    ggplot2::geom_tile(ggplot2::aes(fill = Freq), color = "white") +
    ggplot2::geom_text(ggplot2::aes(label = paste0(Freq, "\n(", round(percentage, 1), "%)")),
                       color = "black", size = 4) +
    ggplot2::scale_fill_gradient(low = "white", high = "steelblue") +
    ggplot2::labs(
      title = "Confusion Matrix",
      x = "Predicted Class",
      y = "Actual Class",
      fill = "Count"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(panel.grid = ggplot2::element_blank())

  return(p)
}

#' Plot precision-recall curve for a classification model
#'
#' @param model A tidysl classification model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object with precision-recall curve
#' @importFrom ROCR prediction performance
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @keywords internal
tl_plot_precision_recall <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual values
  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) {
    actuals <- factor(actuals)
  }

  # For binary classification
  if (length(levels(actuals)) == 2) {
    # Get probabilities
    probs <- predict(model, new_data, type = "prob")
    pos_class <- levels(actuals)[2]
    pos_probs <- probs[[pos_class]]

    # Convert actuals to binary (0/1)
    binary_actuals <- as.integer(actuals == pos_class)

    # Create precision-recall curve
    pred_obj <- ROCR::prediction(pos_probs, binary_actuals)
    perf <- ROCR::performance(pred_obj, "prec", "rec")

    # Calculate area under PR curve
    pr_auc <- tl_calculate_pr_auc(perf)

    # Create data frame for plotting
    pr_data <- tibble::tibble(
      recall = unlist(perf@x.values),
      precision = unlist(perf@y.values)
    )

    # Remove NA/NaN values
    pr_data <- pr_data[!is.na(pr_data$precision) & !is.na(pr_data$recall), ]

    # Create the plot
    p <- ggplot2::ggplot(pr_data, ggplot2::aes(x = recall, y = precision)) +
      ggplot2::geom_line(color = "blue", size = 1) +
      ggplot2::labs(
        title = "Precision-Recall Curve",
        subtitle = paste0("Area Under PR Curve = ", round(pr_auc, 3)),
        x = "Recall",
        y = "Precision"
      ) +
      ggplot2::ylim(0, 1) +
      ggplot2::xlim(0, 1) +
      ggplot2::theme_minimal()

    return(p)
  } else {
    # Multiclass precision-recall curves (not implemented)
    stop("Multiclass precision-recall curves not currently implemented", call. = FALSE)
  }
}

#' Plot calibration curve for a classification model
#'
#' @param model A tidysl classification model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param bins Number of bins for grouping predictions (default: 10)
#' @param ... Additional arguments
#' @return A ggplot object with calibration curve
#' @importFrom ggplot2 ggplot aes geom_point geom_line geom_abline labs theme_minimal
#' @keywords internal
tl_plot_calibration <- function(model, new_data = NULL, bins = 10, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual values
  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) {
    actuals <- factor(actuals)
  }

  # For binary classification
  if (length(levels(actuals)) == 2) {
    # Get probabilities
    probs <- predict(model, new_data, type = "prob")
    pos_class <- levels(actuals)[2]
    pos_probs <- probs[[pos_class]]

    # Convert actuals to binary (0/1)
    binary_actuals <- as.integer(actuals == pos_class)

    # Create bins of predictions
    bin_breaks <- seq(0, 1, length.out = bins + 1)
    bin_mids <- (bin_breaks[-1] + bin_breaks[-(bins + 1)]) / 2

    # Assign each prediction to a bin
    bins_idx <- cut(pos_probs, breaks = bin_breaks, labels = FALSE, include.lowest = TRUE)

    # Calculate mean predicted probability and fraction of positives for each bin
    calibration_data <- tibble::tibble(
      bin = bins_idx,
      prob = pos_probs,
      actual = binary_actuals
    ) %>%
      dplyr::group_by(.data$bin) %>%
      dplyr::summarize(
        mean_pred_prob = mean(.data$prob),
        frac_pos = mean(.data$actual),
        n = dplyr::n(),
        .groups = "drop"
      )

    # Create the plot
    p <- ggplot2::ggplot(calibration_data, ggplot2::aes(x = mean_pred_prob, y = frac_pos)) +
      ggplot2::geom_point(ggplot2::aes(size = n), alpha = 0.7) +
      ggplot2::geom_line(color = "blue") +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      ggplot2::labs(
        title = "Calibration Curve",
        subtitle = "Perfectly calibrated predictions should lie on the diagonal",
        x = "Mean Predicted Probability",
        y = "Fraction of Positives",
        size = "Count"
      ) +
      ggplot2::coord_fixed() +
      ggplot2::theme_minimal()

    return(p)
  } else {
    # Multiclass calibration (not implemented)
    stop("Multiclass calibration curves not currently implemented", call. = FALSE)
  }
}
