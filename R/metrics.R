#' @title Metrics Functionality for tidylearn
#' @name tidylearn-metrics
#' @description Functions for calculating model evaluation metrics
#' @importFrom yardstick accuracy precision recall f_meas
#'   rmse rsq mae mape roc_auc pr_auc
#' @importFrom ROCR prediction performance
#' @importFrom dplyr tibble mutate
NULL

#' Calculate classification metrics
#'
#' @param actuals Actual values (ground truth)
#' @param predicted Predicted class values
#' @param predicted_probs Predicted probabilities (for metrics like AUC)
#' @param metrics Character vector of metrics to compute
#' @param thresholds Optional vector of thresholds to evaluate
#'   for threshold-dependent metrics
#' @param ... Additional arguments
#' @return A \link[tibble]{tibble} with columns \code{metric} (character)
#'   and \code{value} (numeric) containing the requested classification
#'   metrics. When \code{thresholds} are supplied, additional rows are
#'   appended with threshold-specific metric names.
#' @examples
#' \donttest{
#' model <- tl_model(iris, Species ~ ., method = "forest")
#' preds <- predict(model)
#' tl_calc_classification_metrics(iris$Species, preds$.pred)
#' }
#' @export
tl_calc_classification_metrics <- function(
    actuals, predicted,
    predicted_probs = NULL,
    metrics = c("accuracy", "precision",
                "recall", "f1", "auc"),
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

  # tidylearn treats the second factor level as the positive class -- see
  # the AUC branch below, tl_predict_classification(), and the lift/gain
  # plots. yardstick defaults to the first level, so binary metrics have
  # to say so explicitly or they describe the negative class.
  ev <- tl_event_level_args(actuals)

  # Calculate basic classification metrics
  if ("accuracy" %in% metrics) {
    acc <- yardstick::accuracy_vec(actuals, predicted)
    results <- results %>% dplyr::add_row(metric = "accuracy", value = acc)
  }

  if ("precision" %in% metrics) {
    prec <- do.call(
      yardstick::precision_vec, c(list(actuals, predicted), ev)
    )
    results <- results %>% dplyr::add_row(metric = "precision", value = prec)
  }

  if ("recall" %in% metrics || "sensitivity" %in% metrics) {
    rec <- do.call(yardstick::recall_vec, c(list(actuals, predicted), ev))
    results <- results %>% dplyr::add_row(metric = "recall", value = rec)
    if ("sensitivity" %in% metrics) {
      results <- results %>% dplyr::add_row(metric = "sensitivity", value = rec)
    }
  }

  if ("specificity" %in% metrics) {
    spec <- do.call(
      yardstick::specificity_vec, c(list(actuals, predicted), ev)
    )
    results <- results %>% dplyr::add_row(metric = "specificity", value = spec)
  }

  if ("f1" %in% metrics) {
    f1 <- do.call(
      yardstick::f_meas_vec, c(list(actuals, predicted, beta = 1), ev)
    )
    results <- results %>% dplyr::add_row(metric = "f1", value = f1)
  }

  # Calculate threshold-dependent metrics if probabilities are provided
  has_threshold_metrics <- !is.null(predicted_probs) &&
    ("auc" %in% metrics || "pr_auc" %in% metrics ||
       !is.null(thresholds))
  if (has_threshold_metrics) {

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
        class_aucs <- purrr::map_dbl(
          names(predicted_probs),
          function(class_name) {
            binary_actuals <- as.integer(
              actuals == class_name
            )
            pred_obj <- ROCR::prediction(
              predicted_probs[[class_name]],
              binary_actuals
            )
            unlist(
              ROCR::performance(pred_obj, "auc")@y.values
            )
          }
        )

        # Average AUC across classes
        macro_auc <- mean(class_aucs)
        results <- results %>% dplyr::add_row(metric = "auc", value = macro_auc)

        # Add individual class AUCs
        for (i in seq_along(names(predicted_probs))) {
          class_name <- names(predicted_probs)[i]
          results <- results %>%
            dplyr::add_row(
              metric = paste0("auc_", class_name),
              value = class_aucs[i]
            )
        }
      }
    }
  }

  results
}

#' Positive-class argument for yardstick binary metrics
#'
#' tidylearn's positive class is the second factor level. yardstick's
#' default is the first, and its \code{event_level} argument only applies
#' to the binary case -- passing it for a multiclass problem warns.
#'
#' @param actuals A factor of ground-truth values
#' @return A list to splice into a yardstick call: \code{event_level =
#'   "second"} for a two-level factor, empty otherwise
#' @keywords internal
tl_event_level_args <- function(actuals) {
  if (nlevels(actuals) == 2L) list(event_level = "second") else list()
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
    width <- recall[i] - recall[i - 1]
    height <- (precision[i] + precision[i - 1]) / 2
    auc <- auc + width * height
  }

  auc
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
  # No need to convert actuals to binary here,
  # we need the factor for the metrics

  threshold_results <- purrr::map_dfr(thresholds, function(threshold) {
    # Make predictions at this threshold
    pred_vals <- ifelse(
      probs >= threshold, pos_class,
      levels(actuals)[1]
    )
    pred_class <- factor(
      pred_vals, levels = levels(actuals)
    )

    # Score against the same positive class the threshold assigns above,
    # not yardstick's default first level
    ev <- tl_event_level_args(actuals)

    # Calculate metrics
    acc <- yardstick::accuracy_vec(actuals, pred_class)
    prec <- do.call(
      yardstick::precision_vec, c(list(actuals, pred_class), ev)
    )
    rec <- do.call(yardstick::recall_vec, c(list(actuals, pred_class), ev))
    f1 <- do.call(
      yardstick::f_meas_vec, c(list(actuals, pred_class, beta = 1), ev)
    )

    # Calculate F2 and F0.5 scores
    f2 <- do.call(
      yardstick::f_meas_vec, c(list(actuals, pred_class, beta = 2), ev)
    )
    f0_5 <- do.call(
      yardstick::f_meas_vec, c(list(actuals, pred_class, beta = 0.5), ev)
    )

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
      value = c(acc, prec, rec, f1, f2, f0_5)
    )
  })

  threshold_results
}


#' Calculate regression metrics
#'
#' @param actuals Actual values (ground truth)
#' @param predicted Predicted values
#' @param metrics Character vector of metrics to compute
#' @return A \link[tibble]{tibble} with columns \code{metric} and
#'   \code{value}
#' @keywords internal
#' @noRd
tl_calc_regression_metrics <- function(actuals, predicted,
                                       metrics = c("rmse", "mae", "rsq")) {
  actuals <- as.numeric(actuals)
  predicted <- as.numeric(predicted)

  # Drop pairs where either side is missing so every metric is computed
  # over the same observations
  complete <- !is.na(actuals) & !is.na(predicted)
  actuals <- actuals[complete]
  predicted <- predicted[complete]

  residuals <- predicted - actuals

  results <- tibble::tibble(metric = character(), value = numeric())

  if ("mse" %in% metrics) {
    results <- dplyr::add_row(
      results, metric = "mse", value = mean(residuals^2)
    )
  }

  if ("rmse" %in% metrics) {
    results <- dplyr::add_row(
      results, metric = "rmse", value = sqrt(mean(residuals^2))
    )
  }

  if ("mae" %in% metrics) {
    results <- dplyr::add_row(
      results, metric = "mae", value = mean(abs(residuals))
    )
  }

  if ("mape" %in% metrics) {
    nonzero <- actuals != 0
    mape <- if (any(nonzero)) {
      mean(abs(residuals[nonzero] / actuals[nonzero])) * 100
    } else {
      NA_real_
    }
    results <- dplyr::add_row(results, metric = "mape", value = mape)
  }

  if ("rsq" %in% metrics) {
    ss_res <- sum(residuals^2)
    ss_tot <- sum((actuals - mean(actuals))^2)
    rsq <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
    results <- dplyr::add_row(results, metric = "rsq", value = rsq)
  }

  results
}

#' Evaluate a tidylearn model
#' @param object A tidylearn model object
#' @param new_data Optional new data for evaluation
#'   (if NULL, uses training data)
#' @param metrics Character vector of metrics to compute. If \code{NULL}
#'   (the default), \code{"accuracy"} is used for classification models
#'   and \code{c("rmse", "mae", "rsq")} for regression models.
#'   Classification supports \code{"accuracy"}, \code{"precision"},
#'   \code{"recall"}, \code{"sensitivity"}, \code{"specificity"},
#'   \code{"f1"}, \code{"auc"} and \code{"pr_auc"}; regression supports
#'   \code{"rmse"}, \code{"mse"}, \code{"mae"}, \code{"mape"} and
#'   \code{"rsq"}.
#' @param ... Additional arguments passed to \code{predict()}
#' @return A \link[tibble]{tibble} with columns \code{metric} (character)
#'   and \code{value} (numeric), containing one row per requested metric.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_evaluate(model)
#' tl_evaluate(model, metrics = c("rmse", "mape"))
#' }
#' @export
tl_evaluate <- function(object, new_data = NULL, metrics = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- object$data
  }

  if (!inherits(object, "tidylearn_supervised")) {
    # Unsupervised evaluation (placeholder)
    return(tibble::tibble(metric = "completed", value = 1))
  }

  response_var <- object$spec$response_var
  if (!response_var %in% names(new_data)) {
    stop(
      "Response variable '", response_var,
      "' not found in the evaluation data.",
      call. = FALSE
    )
  }
  actuals <- new_data[[response_var]]

  # Ask for the prediction type each metric family needs. The default
  # predict type is method-dependent -- logistic returns probabilities
  # for type = "response" -- so classification must request "class".
  extract_pred <- function(type) {
    preds <- predict(object, new_data = new_data, type = type, ...)
    if (!".pred" %in% names(preds)) {
      stop(
        "predict() for method '", object$spec$method, "' did not return a ",
        "'.pred' column for type = '", type, "'.",
        call. = FALSE
      )
    }
    preds$.pred
  }

  if (object$spec$is_classification) {
    if (is.null(metrics)) metrics <- "accuracy"

    predicted <- extract_pred("class")

    predicted_probs <- NULL
    if (any(c("auc", "pr_auc") %in% metrics)) {
      predicted_probs <- predict(
        object, new_data = new_data, type = "prob", ...
      )
    }

    tl_calc_classification_metrics(
      actuals = actuals,
      predicted = predicted,
      predicted_probs = predicted_probs,
      metrics = metrics
    )
  } else {
    if (is.null(metrics)) metrics <- c("rmse", "mae", "rsq")

    predicted <- extract_pred("response")

    tl_calc_regression_metrics(
      actuals = actuals,
      predicted = predicted,
      metrics = metrics
    )
  }
}

#' Cross-validation for tidylearn models
#' @param data Data frame
#' @param formula Model formula
#' @param method Modeling method
#' @param folds Number of cross-validation folds
#' @param metrics Character vector of metrics to compute on each fold,
#'   passed to \code{\link{tl_evaluate}}. If \code{NULL} (the default),
#'   \code{tl_evaluate}'s per-task defaults are used.
#' @param transform Optional function for feature engineering that has to
#'   be refitted per fold. It is called with the training rows of each
#'   fold and must return a list with an \code{apply} function (applied
#'   to both the training and assessment rows) and, optionally, a
#'   \code{formula} to fit under. Use this for anything that learns
#'   parameters from the data -- PCA rotations, cluster centroids,
#'   target encodings -- since fitting those before the split inflates
#'   every fold's score.
#' @param ... Additional arguments passed to \code{\link{tl_model}}
#' @return A list with two elements:
#'   \describe{
#'     \item{\code{$folds}}{A list of per-fold evaluation
#'       \link[tibble]{tibble}s, each with \code{metric} and
#'       \code{value} columns.}
#'     \item{\code{$summary}}{A \link[tibble]{tibble} with columns
#'       \code{metric}, \code{mean}, and \code{sd} summarizing
#'       performance across folds.}
#'   }
#' @examples
#' \donttest{
#' cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3)
#' cv$summary
#' }
#' @export
tl_cv <- function(data, formula, method, folds = 5, metrics = NULL,
                  transform = NULL, ...) {
  n <- nrow(data)
  if (folds < 2 || folds > n) {
    stop("'folds' must be between 2 and nrow(data) (", n, "). Got: ", folds,
         call. = FALSE)
  }
  indices <- sample(1:n)

  # Assign every row to exactly one fold. Sizing the folds by
  # floor(n / folds) and slicing forward leaves the final n %% folds rows
  # in no test set at all, so spread the remainder instead.
  fold_id <- rep(seq_len(folds), length.out = n)

  cv_results <- list()
  test_sizes <- integer(folds)

  for (i in 1:folds) {
    # Create fold indices
    test_indices <- indices[fold_id == i]
    train_indices <- setdiff(1:n, test_indices)

    # Split data
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]

    fold_formula <- formula

    # Feature engineering that learns from data -- PCA rotations, cluster
    # centroids -- has to be refitted inside the fold. Fitting it once on
    # everything beforehand lets each assessment row shape the features
    # it is then scored on.
    if (!is.null(transform)) {
      fitted_transform <- transform(train_data)
      train_data <- fitted_transform$apply(train_data)
      test_data <- fitted_transform$apply(test_data)
      fold_formula <- fitted_transform$formula %||% formula
    }

    # Train model. tl_model() notes things about the response -- that a
    # numeric column with few distinct values is being treated as
    # regression, say -- which is worth saying once and not once per
    # fold. The caller asked for cross-validation, not for k fits.
    model <- suppressMessages(
      tl_model(train_data, fold_formula, method = method, ...)
    )

    # Evaluate
    eval_result <- tl_evaluate(model, new_data = test_data, metrics = metrics)

    cv_results[[i]] <- eval_result
    test_sizes[i] <- nrow(test_data)
  }

  # Combine results
  all_results <- dplyr::bind_rows(cv_results)

  # Calculate mean and sd for each metric
  summary_results <- all_results %>%
    dplyr::group_by(metric) %>%
    dplyr::summarize(
      mean = mean(value, na.rm = TRUE),
      sd = stats::sd(value, na.rm = TRUE),
      .groups = "drop"
    )

  # A metric that is NA in every fold reaches the summary as NaN, from
  # mean() over nothing. That is arithmetically right and reads as a
  # malfunction: rsq needs variation in the truth, so it is undefined
  # whenever a fold holds one observation -- which is exactly what
  # folds = nrow(data) asks for. Say so once rather than leave a bare NaN.
  undefined <- summary_results$metric[!is.finite(summary_results$mean)]
  if (length(undefined) > 0) {
    fold_sizes <- test_sizes
    message(
      "Note: ", paste(undefined, collapse = ", "),
      " could not be computed for any fold, so ",
      if (length(undefined) == 1) "it is" else "they are",
      " reported as NA. ",
      if (min(fold_sizes) <= 1) {
        paste0(
          "The smallest fold holds ", min(fold_sizes),
          if (min(fold_sizes) == 1) " observation" else " observations",
          ", and metrics needing variation within a fold -- rsq among ",
          "them -- are undefined there. Use fewer folds."
        )
      } else {
        "Check that the requested metrics suit this task."
      }
    )
  }

  list(
    folds = cv_results,
    summary = summary_results
  )
}
