#' @title Metrics Functionality for tidylearn
#' @name tidylearn-metrics
#' @description Functions for calculating model evaluation metrics
#' @importFrom yardstick accuracy precision recall f_meas rmse rsq mae mape roc_auc pr_auc
#' @importFrom ROCR prediction performance
#' @importFrom dplyr tibble %>% mutate
NULL

#' Calculate classification metrics
#'
#' @param actuals Actual values (ground truth)
#' @param predicted Predicted class values
#' @param predicted_probs Predicted probabilities (for metrics like AUC)
#' @param metrics Character vector of metrics to compute
#' @param thresholds Optional vector of thresholds to evaluate for threshold-dependent metrics
#' @param ... Additional arguments
#' @return A tibble of evaluation metrics
#' @export
tl_calc_classification_metrics <- function(actuals, predicted, predicted_probs = NULL,
                                           metrics = c("accuracy", "precision", "recall", "f1", "auc"),
                                           thresholds = NULL, ...) {
  # Ensure actuals is a factor
  if (!is.factor(actuals)) {
    actuals <- as.factor(actuals)
  }

  # Ensure predicted is a factor with the same levels
  if (!is.factor(predicted)) {
    predicted <- factor(predicted, levels = levels(actuals))
  }

  # Create a results data frame
  results <- tibble::tibble(metric = character(), value = numeric())

  # Calculate basic classification metrics
  if ("accuracy" %in% metrics) {
    acc <- yardstick::accuracy_vec(actuals, predicted)
    results <- results %>% dplyr::add_row(metric = "accuracy", value = acc)
  }

  if ("precision" %in% metrics) {
    prec <- yardstick::precision_vec(actuals, predicted)
    results <- results %>% dplyr::add_row(metric = "precision", value = prec)
  }

  if ("recall" %in% metrics || "sensitivity" %in% metrics) {
    rec <- yardstick::recall_vec(actuals, predicted)
    results <- results %>% dplyr::add_row(metric = "recall", value = rec)
    if ("sensitivity" %in% metrics) {
      results <- results %>% dplyr::add_row(metric = "sensitivity", value = rec)
    }
  }

  if ("specificity" %in% metrics) {
    spec <- yardstick::specificity_vec(actuals, predicted)
    results <- results %>% dplyr::add_row(metric = "specificity", value = spec)
  }

  if ("f1" %in% metrics) {
    f1 <- yardstick::f_meas_vec(actuals, predicted, beta = 1)
    results <- results %>% dplyr::add_row(metric = "f1", value = f1)
  }

  # Calculate threshold-dependent metrics if probabilities are provided
  if (!is.null(predicted_probs) && (
    "auc" %in% metrics ||
    "pr_auc" %in% metrics ||
    !is.null(thresholds))) {

    # For binary classification
    if (ncol(predicted_probs) == 2) {
      # Use the probability of the positive class
      pos_class <- levels(actuals)[2]
      probs <- predicted_probs[[pos_class]]

      # AUC ROC
      if ("auc" %in% metrics) {
        binary_actuals <- as.integer(actuals == pos_class)
        pred_obj <- ROCR::prediction(probs, binary_actuals)
        auc <- unlist(ROCR::performance(pred_obj, "auc")@y.values)
        results <- results %>% dplyr::add_row(metric = "auc", value = auc)
      }

      # PR AUC
      if ("pr_auc" %in% metrics) {
        binary_actuals <- as.integer(actuals == pos_class)
        pred_obj <- ROCR::prediction(probs, binary_actuals)
        perf <- ROCR::performance(pred_obj, "prec", "rec")
        pr_auc <- tl_calculate_pr_auc(perf)
        results <- results %>% dplyr::add_row(metric = "pr_auc", value = pr_auc)
      }

      # Evaluate metrics at different thresholds
      if (!is.null(thresholds)) {
        threshold_metrics <- tl_evaluate_thresholds(
          actuals = actuals,
          probs = probs,
          thresholds = thresholds,
          pos_class = pos_class
        )
        results <- dplyr::bind_rows(results, threshold_metrics)
      }
    } else {
      # Multiclass AUC (one-vs-rest)
      if ("auc" %in% metrics) {
        # Calculate one-vs-rest AUC for each class
        class_aucs <- purrr::map_dbl(names(predicted_probs), function(class_name) {
          binary_actuals <- as.integer(actuals == class_name)
          pred_obj <- ROCR::prediction(predicted_probs[[class_name]], binary_actuals)
          unlist(ROCR::performance(pred_obj, "auc")@y.values)
        })

        # Average AUC across classes
        macro_auc <- mean(class_aucs)
        results <- results %>% dplyr::add_row(metric = "auc", value = macro_auc)

        # Add individual class AUCs
        for (i in seq_along(names(predicted_probs))) {
          class_name <- names(predicted_probs)[i]
          results <- results %>%
            dplyr::add_row(metric = paste0("auc_", class_name), value = class_aucs[i])
        }
      }
    }
  }

  return(results)
}

#' Calculate the area under the precision-recall curve
#'
#' @param perf A ROCR performance object
#' @return The area under the PR curve
#' @keywords internal
tl_calculate_pr_auc <- function(perf) {
  precision <- perf@y.values[[1]]
  recall <- perf@x.values[[1]]

  # Remove NA/NaN values
  valid <- !is.na(precision) & !is.na(recall)
  precision <- precision[valid]
  recall <- recall[valid]

  # Sort by recall
  ord <- order(recall)
  recall <- recall[ord]
  precision <- precision[ord]

  # Calculate AUC using trapezoidal rule
  auc <- 0
  for (i in 2:length(recall)) {
    auc <- auc + (recall[i] - recall[i-1]) * (precision[i] + precision[i-1]) / 2
  }

  return(auc)
}

#' Evaluate metrics at different thresholds
#'
#' @param actuals Actual values (ground truth)
#' @param probs Predicted probabilities
#' @param thresholds Vector of thresholds to evaluate
#' @param pos_class The positive class
#' @return A tibble of metrics at different thresholds
#' @keywords internal
tl_evaluate_thresholds <- function(actuals, probs, thresholds, pos_class) {
  # No need to convert actuals to binary here, we need the factor for the metrics

  threshold_results <- purrr::map_dfr(thresholds, function(threshold) {
    # Make predictions at this threshold
    pred_class <- factor(ifelse(probs >= threshold, pos_class, levels(actuals)[1]),
                         levels = levels(actuals))

    # Calculate metrics
    acc <- yardstick::accuracy_vec(actuals, pred_class)
    prec <- yardstick::precision_vec(actuals, pred_class)
    rec <- yardstick::recall_vec(actuals, pred_class)
    f1 <- yardstick::f_meas_vec(actuals, pred_class, beta = 1)

    # Calculate F2 and F0.5 scores
    f2 <- yardstick::f_meas_vec(actuals, pred_class, beta = 2)
    f0.5 <- yardstick::f_meas_vec(actuals, pred_class, beta = 0.5)

    # Return results for this threshold
    tibble::tibble(
      threshold = threshold,
      metric = c(
        paste0("accuracy_t", threshold),
        paste0("precision_t", threshold),
        paste0("recall_t", threshold),
        paste0("f1_t", threshold),
        paste0("f2_t", threshold),
        paste0("f0.5_t", threshold)
      ),
      value = c(acc, prec, rec, f1, f2, f0.5)
    )
  })

  return(threshold_results)
}
