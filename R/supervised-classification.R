#' @title Classification Functions for tidylearn
#' @name tidylearn-classification
#' @description Logistic regression and classification metrics functionality
#' @importFrom stats glm predict binomial
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr mutate
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
  # tl_model() has already made the response a factor and dropped levels
  # that no row uses. Normalise again rather than assume it, because the
  # count below has to be of classes that are present: a subset of a
  # larger frame keeps every original level, and rejecting a two-class
  # response for declaring a third would refuse data glm() fits happily.
  #
  # The response is the one the formula computes. Read off the column named
  # first, I(mpg > 20) ~ wt was checked as raw mpg and refused for having
  # 25 levels. Only a bare column name is written back into `data`:
  # overwriting mpg with a factor would change what I(mpg > 20) computes.
  lhs <- formula[[2L]]
  response_label <- if (is.name(lhs)) as.character(lhs) else deparse1(lhs)
  computed <- tl_formula_response(formula, data)
  response <- tl_normalise_response(computed)
  if (is.name(lhs)) {
    data[[response_label]] <- response
  }

  # Reject a response glm(binomial) cannot model, here rather than at
  # predict() or tl_evaluate(). glm() takes a three-class factor without
  # complaint and fits the first class against the other two, so nothing
  # upstream of prediction says the model is meaningless.
  tl_check_binary_response(response, response_label)

  # glm() compares every row with the first declared level of a factor
  # response. A computed one is fitted as computed, so if no row has that
  # level, every row would count as the other class.
  if (is.factor(computed) &&
        !identical(levels(computed)[1], levels(response)[1])) {
    stop(
      "The response ", response_label, " declares '", levels(computed)[1],
      "' first, a class no row has,\nand glm() would compare every row ",
      "with it. Wrap the response in droplevels().",
      call. = FALSE
    )
  }

  # Fit the logistic regression model
  # By value, so weights = <a vector> reaches glm() as a vector
  glm_model <- tl_fit_by_value(
    stats::glm, "glm",
    list(formula = formula, data = data, family = stats::binomial(), ...)
  )

  # Store the family as a call update() and step() can re-evaluate, rather
  # than the family object printed in full
  glm_model$call$family <- quote(binomial())
  glm_model
}

#' Refuse a response that logistic regression cannot model
#'
#' @param y The response, as a factor whose levels are the classes present
#' @param response_var Its name, for the message
#' @return `TRUE`, invisibly, when the response has exactly two classes
#' @keywords internal
#' @noRd
tl_check_binary_response <- function(y, response_var) {
  class_levels <- levels(y)

  if (length(class_levels) == 2) {
    return(invisible(TRUE))
  }

  if (length(class_levels) < 2) {
    found <- if (length(class_levels) == 0) {
      "none"
    } else {
      paste0("only one ('", class_levels, "')")
    }
    stop(
      "Logistic regression needs a response with two levels, but '",
      response_var, "' has ", found,
      ". There is nothing to discriminate.",
      call. = FALSE
    )
  }

  stop(
    "Logistic regression is binary only, but '", response_var, "' has ",
    length(class_levels), " levels (",
    paste(class_levels, collapse = ", "), "). ",
    "glm(family = binomial) would model '", class_levels[1],
    "' against the rest without saying so, and the resulting model ",
    "cannot be predicted from or evaluated. ",
    "Refit with method = \"forest\", \"tree\", \"boost\", \"svm\", \"nn\", ",
    "\"ridge\", \"lasso\", \"elastic_net\" or \"xgboost\", all of which ",
    "handle more than two classes; or collapse '", response_var,
    "' to two levels first.",
    call. = FALSE
  )
}

#' Predict using a logistic regression model
#'
#' @param model A tidylearn logistic model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "prob" (default), "class", "response"
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_logistic <- function(model, new_data, type = "prob", ...) {
  # The model's classes. Read off the training column, I(mpg > 20) ~ wt
  # had one class per distinct mpg.
  class_levels <- tl_model_classes(model)

  # Make predictions based on the type
  if (type == "response") {
    # Get the linear predictor
    preds <- stats::predict(
      model$fit, newdata = new_data, type = "response", ...
    )
    preds
  } else if (type == "prob") {
    # Binary classification
    if (length(class_levels) == 2) {
      # Get probabilities for the positive class
      pos_probs <- stats::predict(
        model$fit, newdata = new_data,
        type = "response", ...
      )
      neg_probs <- 1 - pos_probs

      # Create a data frame with probabilities for each class
      prob_df <- tibble::tibble(
        !!class_levels[1] := neg_probs,
        !!class_levels[2] := pos_probs
      )

      prob_df
    } else {
      # Multiclass classification (requires multinomial logistic regression)
      stop(
        "Multiclass logistic regression not ",
        "currently implemented",
        call. = FALSE
      )
    }
  } else if (type == "class") {
    # Binary classification
    if (length(class_levels) == 2) {
      # Get probabilities for the positive class
      pos_probs <- stats::predict(
        model$fit, newdata = new_data,
        type = "response", ...
      )

      # Classify based on probability > 0.5
      pred_classes <- ifelse(pos_probs > 0.5, class_levels[2], class_levels[1])
      pred_classes <- factor(pred_classes, levels = class_levels)

      pred_classes
    } else {
      # Multiclass classification (requires multinomial logistic regression)
      stop(
        "Multiclass logistic regression not ",
        "currently implemented",
        call. = FALSE
      )
    }
  } else {
    stop(
      "Invalid prediction type. ",
      "Use 'prob', 'class', or 'response'.",
      call. = FALSE
    )
  }
}

#' The classes a classification chart scores against
#'
#' Read from the model rather than from the rows being scored: a test split
#' of \code{iris[iris$Species != "setosa", ]} still declares setosa, which
#' made a binary model look multiclass, and a test factor whose levels were
#' reordered moved the positive class.
#'
#' @param model A tidylearn model.
#' @param new_data The rows being scored.
#' @param chart What is being drawn, for messages (e.g. "ROC curve").
#' @return The model's classes; the second is the positive class.
#' @keywords internal
#' @noRd
tl_chart_classes <- function(model, new_data, chart) {
  if (!isTRUE(model$spec$is_classification)) {
    stop("The ", chart, " is only available for classification models.",
         call. = FALSE)
  }

  # Every column the response is computed from, not only the first
  lhs <- model$spec$formula[[2L]]
  needed <- setdiff(all.vars(lhs), names(new_data))
  if (length(needed) > 0) {
    stop("The ", chart, " compares predictions with the observed classes, ",
         "so\nnew_data needs the column(s) the response ", deparse1(lhs),
         " is made of: ", paste0("'", needed, "'", collapse = ", "), ".",
         call. = FALSE)
  }

  tl_model_classes(model)
}

#' Observed outcomes and positive-class probabilities for a binary chart
#'
#' Rows missing the response or the probability are left out with a
#' warning giving the count; ROCR stopped on them with "'predictions'
#' contains NA". Rows of a class the model never saw are left out by
#' \code{tl_align_classes()}, which warns for those itself.
#'
#' @param model A binary tidylearn classification model.
#' @param new_data The rows being scored.
#' @param model_levels The model's two classes.
#' @param chart What is being drawn, for messages.
#' @param both_classes Whether the chart needs rows of each class.
#' @return A list: \code{actual}, 1 for the positive (second) class and 0
#'   otherwise; \code{prob}, its predicted probability.
#' @keywords internal
#' @noRd
tl_binary_chart_scores <- function(model, new_data, model_levels, chart,
                                   both_classes = TRUE) {
  # As the formula computes it. The column alone is the raw mpg when the
  # response is I(mpg > 20).
  observed <- tl_observed_response(model, new_data)
  aligned <- tl_align_classes(observed, model_levels)
  pos_class <- model_levels[2]
  pos_probs <- predict(model, new_data, type = "prob")[[pos_class]]

  unseen <- !is.na(observed) & !aligned$keep
  incomplete <- (is.na(observed) | is.na(pos_probs)) & !unseen
  if (any(incomplete)) {
    warning(
      sum(incomplete), " row(s) with a missing response or predicted ",
      "probability are left out of the ", chart, ".",
      call. = FALSE
    )
  }

  keep <- aligned$keep & !is.na(pos_probs)
  actual <- as.integer(aligned$actuals[keep] == pos_class)

  if (length(actual) == 0L) {
    stop("No row scored here has both a response and a predicted ",
         "probability, so there is no ", chart, " to draw.", call. = FALSE)
  }
  # ROCR stops with "Number of classes is not equal to 2"
  present <- model_levels[sort(unique(actual)) + 1L]
  if (both_classes && length(present) < 2L) {
    stop(
      "The ", chart, " needs rows of both classes (",
      paste(model_levels, collapse = ", "), "), but the rows scored here ",
      "are all '", present, "'.",
      call. = FALSE
    )
  }

  list(actual = actual, prob = pos_probs[keep])
}

#' Plot ROC curve for a classification model
#'
#' @param model A tidylearn classification model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object with ROC curve
#' @importFrom ROCR prediction performance
#' @importFrom ggplot2 ggplot aes geom_line geom_abline labs theme_minimal
#' @keywords internal
tl_plot_roc <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  model_levels <- tl_chart_classes(model, new_data, "ROC curve")
  if (length(model_levels) != 2L) {
    # Multiclass ROC (one-vs-rest)
    stop("Multiclass ROC curves not currently implemented", call. = FALSE)
  }
  scores <- tl_binary_chart_scores(model, new_data, model_levels,
                                   "ROC curve")

  # Create ROC curve
  pred_obj <- ROCR::prediction(scores$prob, scores$actual)
  perf <- ROCR::performance(pred_obj, "tpr", "fpr")

  # Calculate AUC
  auc <- unlist(ROCR::performance(pred_obj, "auc")@y.values)

  # Create data frame for plotting
  roc_data <- tibble::tibble(
    fpr = unlist(perf@x.values),
    tpr = unlist(perf@y.values)
  )

  # Create the plot
  ggplot2::ggplot(roc_data, ggplot2::aes(x = fpr, y = tpr)) +
    ggplot2::geom_line(color = "blue", linewidth = 1) +
    ggplot2::geom_abline(
      intercept = 0, slope = 1,
      linetype = "dashed", color = "gray"
    ) +
    ggplot2::labs(
      title = "ROC Curve",
      subtitle = paste0("AUC = ", round(auc, 3)),
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    ggplot2::coord_fixed() +
    ggplot2::theme_minimal()
}

#' Plot confusion matrix for a classification model
#'
#' @param model A tidylearn classification model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object with confusion matrix
#' @importFrom ggplot2 ggplot aes geom_tile geom_text scale_fill_gradient
#' @keywords internal
tl_plot_confusion <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # The rows and columns are the model's classes, whatever levels the
  # scored data declares or in whatever order
  model_levels <- tl_chart_classes(model, new_data, "confusion matrix")
  observed <- tl_observed_response(model, new_data)
  aligned <- tl_align_classes(observed, model_levels)

  # Get predicted classes
  predicted <- factor(
    as.character(predict(model, new_data, type = "class")$.pred),
    levels = model_levels
  )

  # table() drops a row missing either value, so the counts summed to fewer
  # rows than were passed in, with nothing to say so. tl_align_classes()
  # warns for a class the model never saw.
  unseen <- !is.na(observed) & !aligned$keep
  incomplete <- (is.na(observed) | is.na(predicted)) & !unseen
  if (any(incomplete)) {
    warning(
      sum(incomplete), " row(s) with a missing response or prediction are ",
      "left out of the confusion matrix.",
      call. = FALSE
    )
  }
  keep <- aligned$keep & !is.na(predicted)

  # Create confusion matrix
  cm <- table(Actual = aligned$actuals[keep], Predicted = predicted[keep])

  # Convert to data frame for plotting
  cm_df <- as.data.frame(as.table(cm))

  # Calculate percentages
  cm_df$percentage <- cm_df$Freq / sum(cm_df$Freq) * 100

  # Create the plot
  p <- ggplot2::ggplot(cm_df, ggplot2::aes(x = Predicted, y = Actual)) +
    ggplot2::geom_tile(ggplot2::aes(fill = Freq), color = "white") +
    ggplot2::geom_text(
      ggplot2::aes(
        label = paste0(
          Freq, "\n(", round(percentage, 1), "%)"
        )
      ),
      color = "black", size = 4
    ) +
    ggplot2::scale_fill_gradient(low = "white", high = "steelblue") +
    ggplot2::labs(
      title = "Confusion Matrix",
      x = "Predicted Class",
      y = "Actual Class",
      fill = "Count"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(panel.grid = ggplot2::element_blank())

  p
}

#' Plot precision-recall curve for a classification model
#'
#' @param model A tidylearn classification model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object with precision-recall curve
#' @importFrom ROCR prediction performance
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @keywords internal
tl_plot_precision_recall <- function(model, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  model_levels <- tl_chart_classes(model, new_data, "precision-recall curve")
  if (length(model_levels) != 2L) {
    # Multiclass precision-recall curves (not implemented)
    stop(
      "Multiclass precision-recall curves not ",
      "currently implemented",
      call. = FALSE
    )
  }
  scores <- tl_binary_chart_scores(model, new_data, model_levels,
                                   "precision-recall curve")

  # Create precision-recall curve
  pred_obj <- ROCR::prediction(scores$prob, scores$actual)
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
  ggplot2::ggplot(pr_data, ggplot2::aes(x = recall, y = precision)) +
    ggplot2::geom_line(color = "blue", linewidth = 1) +
    ggplot2::labs(
      title = "Precision-Recall Curve",
      subtitle = paste0("Area Under PR Curve = ", round(pr_auc, 3)),
      x = "Recall",
      y = "Precision"
    ) +
    ggplot2::ylim(0, 1) +
    ggplot2::xlim(0, 1) +
    ggplot2::theme_minimal()
}

#' Plot calibration curve for a classification model
#'
#' @param model A tidylearn classification model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param bins Number of bins for grouping predictions (default: 10)
#' @param ... Additional arguments
#' @return A ggplot object with calibration curve
#' @importFrom ggplot2 ggplot aes geom_point geom_line
#'   geom_abline labs theme_minimal
#' @keywords internal
tl_plot_calibration <- function(model, new_data = NULL, bins = 10, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  model_levels <- tl_chart_classes(model, new_data, "calibration curve")
  if (length(model_levels) != 2L) {
    # Multiclass calibration (not implemented)
    stop(
      "Multiclass calibration curves not ",
      "currently implemented",
      call. = FALSE
    )
  }
  # One class among the rows still has a calibration to show: the fraction
  # of positives in each bin is then 0 or 1 throughout
  scores <- tl_binary_chart_scores(model, new_data, model_levels,
                                   "calibration curve",
                                   both_classes = FALSE)

  # Create bins of predictions
  bin_breaks <- seq(0, 1, length.out = bins + 1)

  # Assign each prediction to a bin
  bins_idx <- cut(
    scores$prob, breaks = bin_breaks,
    labels = FALSE, include.lowest = TRUE
  )

  # Calculate mean predicted probability and fraction of
  # positives for each bin
  calibration_data <- tibble::tibble(
    bin = bins_idx,
    prob = scores$prob,
    actual = scores$actual
  ) |>
    dplyr::group_by(.data$bin) |>
    dplyr::summarize(
      mean_pred_prob = mean(.data$prob),
      frac_pos = mean(.data$actual),
      n = dplyr::n(),
      .groups = "drop"
    )

  # Create the plot
  ggplot2::ggplot(
    calibration_data,
    ggplot2::aes(x = mean_pred_prob, y = frac_pos)
  ) +
    ggplot2::geom_point(ggplot2::aes(size = n), alpha = 0.7) +
    ggplot2::geom_line(color = "blue") +
    ggplot2::geom_abline(
      intercept = 0, slope = 1,
      linetype = "dashed", color = "gray"
    ) +
    ggplot2::labs(
      title = "Calibration Curve",
      subtitle = paste0(
        "Perfectly calibrated predictions ",
        "should lie on the diagonal"
      ),
      x = "Mean Predicted Probability",
      y = "Fraction of Positives",
      size = "Count"
    ) +
    ggplot2::coord_fixed() +
    ggplot2::theme_minimal()
}
