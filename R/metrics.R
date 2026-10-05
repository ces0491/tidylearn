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
#' Scores predicted classes, and optionally class probabilities, against
#' observed classes.
#'
#' The classes, in order, are the levels of \code{predicted} when it is a
#' factor of two or more levels -- which is how \code{predict()} returns
#' them, in the model's order -- and otherwise the classes present in
#' \code{actuals}. Any other class found in \code{actuals} or
#' \code{predicted} follows them. For two classes the second is the
#' positive class. \code{\link{tl_evaluate}} instead leaves out rows of a
#' class the model was never trained on, since it knows the model's
#' classes.
#'
#' A row missing its observed class, its prediction or one of its
#' probabilities is dropped before anything is computed, so every metric
#' describes the same rows. With no row left -- \code{actuals} empty, or
#' every row incomplete -- there is nothing to score, and it is an error
#' of class \code{tidylearn_no_scored_rows}.
#'
#' @param actuals Observed classes: a factor, or a character, logical or
#'   numeric vector.
#' @param predicted Predicted classes, one for each element of
#'   \code{actuals}.
#' @param predicted_probs Class probabilities, needed for \code{"auc"},
#'   \code{"pr_auc"} and \code{thresholds}: a data frame (or matrix) with a
#'   row for each element of \code{actuals} and a column for each class,
#'   named by class -- the shape \code{predict(model, type = "prob")}
#'   returns. Without it, \code{"auc"} and \code{"pr_auc"} are left out of
#'   the result, with a warning when they were asked for by name.
#' @param metrics Character vector of metrics to compute, from
#'   \code{"accuracy"}, \code{"precision"}, \code{"recall"},
#'   \code{"sensitivity"}, \code{"specificity"}, \code{"f1"}, \code{"auc"}
#'   and \code{"pr_auc"}. For more than two classes, \code{"precision"},
#'   \code{"recall"}, \code{"specificity"} and \code{"f1"} are macro
#'   averages, and \code{"auc"} and \code{"pr_auc"} average the one-vs-rest
#'   areas of the classes.
#' @param thresholds Optional numeric vector of cut-offs on the positive
#'   class's probability, for binary classification. Each adds rows
#'   scoring the classes that cut-off assigns. Needs
#'   \code{predicted_probs}.
#' @param ... Not used.
#' @return A \link[tibble]{tibble} with columns \code{metric} (character)
#'   and \code{value} (numeric), one row per requested metric. For more
#'   than two classes, \code{"auc"} is followed by an \code{auc_<class>}
#'   row for each class.
#'
#'   \code{"auc"} and \code{"pr_auc"} need at least two classes among the
#'   scored rows, and are \code{NA}, with a warning, when there is only
#'   one. A class with no scored row has no one-vs-rest area: its
#'   \code{auc_<class>} row is \code{NA}, the averages cover the other
#'   classes, and a warning names it.
#'
#'   With \code{thresholds}, six rows per cut-off follow -- for a cut-off
#'   of 0.5, \code{accuracy_t0.5}, \code{precision_t0.5},
#'   \code{recall_t0.5}, \code{f1_t0.5}, \code{f2_t0.5} and
#'   \code{f0.5_t0.5} -- and the tibble gains a \code{threshold} column,
#'   \code{NA} on the other rows.
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
  tl_check_metric_names(metrics, is_classification = TRUE)
  if (length(predicted) != length(actuals)) {
    stop(
      "'actuals' and 'predicted' must be the same length; got ",
      length(actuals), " and ", length(predicted), ".",
      call. = FALSE
    )
  }

  # Read both sides against one set of classes. A test split of a
  # subsetted frame still declares the levels the subset removed, so taking
  # the classes from the observed factor made a binary model multiclass,
  # and a factor whose levels were merely reordered moved the positive
  # class.
  class_levels <- tl_scored_classes(actuals, predicted, predicted_probs)
  aligned <- tl_align_classes(actuals, class_levels)
  actuals <- aligned$actuals
  predicted <- factor(as.character(predicted), levels = class_levels)
  predicted_probs <- tl_check_class_probs(
    predicted_probs, class_levels, length(actuals)
  )

  # One mask for every metric, as the regression metrics use, so each
  # describes the same rows. Dropped metric by metric, a row with a
  # prediction but no probabilities would count towards accuracy and not
  # towards auc.
  complete <- aligned$keep & !is.na(predicted)
  if (!is.null(predicted_probs)) {
    complete <- complete & stats::complete.cases(predicted_probs)
  }
  # With nothing left, accuracy came back NaN without a message
  if (!any(complete)) {
    tl_stop_no_scored_rows(
      if (length(complete) == 0L) {
        "'actuals' is empty, so there is nothing to score."
      } else {
        paste0(
          "None of the ", length(complete), " observations can be scored: ",
          "each is missing its observed class, its prediction or a ",
          "probability."
        )
      },
      n_rows = length(complete)
    )
  }
  if (!is.null(predicted_probs)) {
    predicted_probs <- predicted_probs[complete, , drop = FALSE]
  }
  actuals <- actuals[complete]
  predicted <- predicted[complete]

  # Create a results data frame
  results <- tibble::tibble(metric = character(), value = numeric())

  # tidylearn treats the second factor level as the positive class -- see
  # tl_ranking_metrics(), tl_predict_logistic(), and the lift/gain plots.
  # yardstick defaults to the first level, so binary metrics have to say
  # so explicitly or they describe the negative class.
  ev <- tl_event_level_args(actuals)

  # Calculate basic classification metrics
  if ("accuracy" %in% metrics) {
    acc <- yardstick::accuracy_vec(actuals, predicted)
    results <- results |> dplyr::add_row(metric = "accuracy", value = acc)
  }

  if ("precision" %in% metrics) {
    prec <- do.call(
      yardstick::precision_vec, c(list(actuals, predicted), ev)
    )
    results <- results |> dplyr::add_row(metric = "precision", value = prec)
  }

  # Recall and sensitivity are one number under two names; each is
  # reported only when asked for
  if (any(c("recall", "sensitivity") %in% metrics)) {
    rec <- do.call(yardstick::recall_vec, c(list(actuals, predicted), ev))
    for (name in intersect(c("recall", "sensitivity"), metrics)) {
      results <- results |> dplyr::add_row(metric = name, value = rec)
    }
  }

  if ("specificity" %in% metrics) {
    spec <- do.call(
      yardstick::specificity_vec, c(list(actuals, predicted), ev)
    )
    results <- results |> dplyr::add_row(metric = "specificity", value = spec)
  }

  if ("f1" %in% metrics) {
    f1 <- do.call(
      yardstick::f_meas_vec, c(list(actuals, predicted, beta = 1), ev)
    )
    results <- results |> dplyr::add_row(metric = "f1", value = f1)
  }

  # Metrics that rank rows by probability
  ranked <- intersect(c("auc", "pr_auc"), metrics)
  if (length(ranked) > 0 && !is.null(predicted_probs)) {
    results <- dplyr::bind_rows(
      results, tl_ranking_metrics(actuals, predicted_probs, ranked)
    )
  } else if (length(ranked) > 0 && !missing(metrics)) {
    # The default metrics include auc, and leaving it out quietly is the
    # documented behaviour there. Asked for by name, it says so.
    warning(
      paste(ranked, collapse = " and "),
      if (length(ranked) == 1L) " needs" else " need",
      " 'predicted_probs' and ",
      if (length(ranked) == 1L) "is" else "are",
      " left out of the result.",
      call. = FALSE
    )
  }

  # Evaluate metrics at different thresholds
  if (!is.null(thresholds)) {
    if (is.null(predicted_probs)) {
      warning(
        "'thresholds' need 'predicted_probs', and are ignored without them.",
        call. = FALSE
      )
    } else if (length(class_levels) != 2L) {
      warning(
        "'thresholds' apply to binary classification only, and are ",
        "ignored for ", length(class_levels), " classes.",
        call. = FALSE
      )
    } else {
      threshold_metrics <- tl_evaluate_thresholds(
        actuals = actuals,
        probs = predicted_probs[[class_levels[2]]],
        thresholds = thresholds,
        pos_class = class_levels[2]
      )
      results <- dplyr::bind_rows(results, threshold_metrics)
    }
  }

  results
}

#' The classes observed values are scored against
#'
#' @param actuals Observed classes
#' @param predicted Predicted classes
#' @param predicted_probs Class probabilities, or NULL
#' @return A character vector of classes, the positive class second
#' @keywords internal
#' @noRd
tl_scored_classes <- function(actuals, predicted, predicted_probs) {
  observed <- levels(tl_normalise_response(actuals))

  # predict() returns classes as a factor in the model's levels, so those
  # are the model's classes, in its order, whatever the scored rows hold.
  # No model predicts a single class, so a one-level factor was built from
  # the predictions themselves and says nothing about the order.
  stated <- if (is.factor(predicted) && nlevels(predicted) >= 2L) {
    levels(predicted)
  } else if (!is.null(colnames(predicted_probs))) {
    colnames(predicted_probs)
  } else {
    observed
  }

  # A class the stated set lacks is still a class: leaving its rows out
  # would score a model as if it never met them
  union(
    stated,
    c(observed, sort(unique(as.character(predicted[!is.na(predicted)]))))
  )
}

#' Check class probabilities against the classes being scored
#'
#' @param predicted_probs Class probabilities, or NULL
#' @param class_levels The classes being scored
#' @param n_rows The number of observed values
#' @return The probability columns for \code{class_levels}, in that order,
#'   or NULL
#' @keywords internal
#' @noRd
tl_check_class_probs <- function(predicted_probs, class_levels, n_rows) {
  if (is.null(predicted_probs)) {
    return(NULL)
  }
  if (is.matrix(predicted_probs)) {
    predicted_probs <- as.data.frame(predicted_probs)
  }

  shape <- paste0(
    "'predicted_probs' must be a data frame with a probability column per ",
    "class, named by class -- the shape predict(type = \"prob\") returns."
  )
  if (!is.data.frame(predicted_probs)) {
    stop(
      shape, " Got ", tl_describe_value(predicted_probs), ".",
      call. = FALSE
    )
  }
  missing_cols <- setdiff(class_levels, names(predicted_probs))
  if (length(missing_cols) > 0) {
    stop(
      shape, " Missing: ", paste0("\"", missing_cols, "\"", collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (nrow(predicted_probs) != n_rows) {
    stop(
      "'predicted_probs' has ", nrow(predicted_probs), " rows for ",
      n_rows, " observed values; it needs one row per observation.",
      call. = FALSE
    )
  }

  predicted_probs[class_levels]
}

#' AUC and PR AUC on scored rows
#'
#' Both rank the rows by a class's probability, so each needs rows of that
#' class and of another. A cross-validation fold can hold a single class,
#' and a class can be missing from the rows scored. Neither area exists
#' then: ROCR stopped on it, aborting the resampling run it was part of,
#' and yardstick returns NA, NaN or 1 depending on which class is missing.
#' It is reported as NA, with a warning, so the resampling functions leave
#' that fold out.
#'
#' @param actuals Observed classes, complete, as a factor in the classes
#'   being scored
#' @param probs Probabilities, complete, one column per class named by
#'   class
#' @param wanted Which of "auc" and "pr_auc" to compute
#' @return A tibble with columns \code{metric} and \code{value}
#' @keywords internal
#' @noRd
tl_ranking_metrics <- function(actuals, probs, wanted) {
  class_levels <- levels(actuals)
  binary <- length(class_levels) == 2L
  present <- class_levels[class_levels %in% actuals]
  undefined <- length(present) < 2L
  named <- paste(wanted, collapse = " and ")

  # tl_calc_classification_metrics() refuses rows that hold no class at
  # all, so an undefined area here means exactly one class
  if (undefined) {
    one <- length(wanted) == 1L
    warning(
      named, if (one) " is" else " are", " undefined when the scored rows ",
      "hold a single class (\"", present, "\"), so ",
      if (one) "it is" else "they are", " NA.",
      call. = FALSE
    )
  }

  # For two classes the curve is the positive class's, the second level.
  # For more, each class is ranked against the rest and the areas are
  # averaged, as yardstick's macro estimator does.
  ranked_classes <- if (binary) class_levels[2] else class_levels
  absent <- setdiff(ranked_classes, present)
  if (!undefined && !binary && length(absent) > 0) {
    several <- length(absent) > 1L
    warning(
      "No scored row belongs to ",
      paste0("\"", absent, "\"", collapse = " or "), ", so ", named,
      if (length(wanted) == 1L) " leaves " else " leave ",
      if (several) "them" else "it", " out: ",
      if (several) {
        "their one-vs-rest values are"
      } else {
        "its one-vs-rest value is"
      },
      " NA and the macro average covers the other classes.",
      call. = FALSE
    )
  }

  one_vs_rest <- function(area) {
    vapply(ranked_classes, function(cls) {
      if (undefined || !cls %in% present) {
        return(NA_real_)
      }
      truth <- factor(actuals == cls, levels = c(FALSE, TRUE))
      area(truth, probs[[cls]], event_level = "second")
    }, numeric(1))
  }
  average <- function(values) {
    if (all(is.na(values))) NA_real_ else mean(values, na.rm = TRUE)
  }

  results <- tibble::tibble(metric = character(), value = numeric())
  if ("auc" %in% wanted) {
    class_aucs <- one_vs_rest(yardstick::roc_auc_vec)
    results <- results |>
      dplyr::add_row(metric = "auc", value = average(class_aucs))
    if (!binary) {
      results <- results |>
        dplyr::add_row(
          metric = paste0("auc_", class_levels),
          value = unname(class_aucs)
        )
    }
  }
  if ("pr_auc" %in% wanted) {
    class_pr_aucs <- one_vs_rest(yardstick::pr_auc_vec)
    results <- results |>
      dplyr::add_row(metric = "pr_auc", value = average(class_pr_aucs))
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
#' Reads the area off a curve ROCR built, and agrees with
#' \code{yardstick::pr_auc()}. ROCR's curve starts at recall 0, where
#' nothing is yet called positive and precision is 0/0. The area has to
#' start there too, at precision 1 as yardstick's does: integrating from
#' the first finite point lost everything before it, so a perfect ranking
#' of 5 positives in 20 scored 0.8, a tree on two iris species 0.07, and
#' constant scores left no area at all.
#'
#' @param perf A ROCR performance object of precision against recall
#' @return The area under the precision-recall curve, or \code{NA} when
#'   the curve has fewer than two points
#' @keywords internal
tl_calculate_pr_auc <- function(perf) {
  precision <- perf@y.values[[1]]
  recall <- perf@x.values[[1]]

  precision[recall %in% 0 & is.nan(precision)] <- 1

  # Remove NA/NaN values
  valid <- !is.na(precision) & !is.na(recall)
  precision <- precision[valid]
  recall <- recall[valid]

  if (length(recall) < 2L) {
    return(NA_real_)
  }

  # Sort by recall. order() keeps tied recalls in ROCR's order, which is
  # the order the curve passes through them.
  ord <- order(recall)
  recall <- recall[ord]
  precision <- precision[ord]

  # Calculate AUC using trapezoidal rule
  sum(
    diff(recall) *
      (utils::head(precision, -1) + utils::tail(precision, -1)) / 2
  )
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

#' Refuse a metric name tl_evaluate() does not compute
#'
#' A name outside \code{tl_known_metrics()} matched no branch, so it was
#' left out of the result: \code{metrics = "RMSE"} returned a 0-row tibble,
#' and \code{tl_cv()} a 0-row summary, without a message.
#'
#' @param metrics The requested metric names
#' @param is_classification Whether the task is classification
#' @return `TRUE`, invisibly, when every name is known
#' @keywords internal
#' @noRd
tl_check_metric_names <- function(metrics, is_classification) {
  known <- tl_known_metrics(is_classification)
  if (length(metrics) == 0) {
    stop(
      "'metrics' is empty. Name at least one of: ",
      paste(known, collapse = ", "), ".",
      call. = FALSE
    )
  }

  unknown <- setdiff(metrics, known)
  if (length(unknown) > 0) {
    stop(
      "Unknown ", if (is_classification) "classification" else "regression",
      " metric(s) in 'metrics': ",
      paste0("\"", unknown, "\"", collapse = ", "),
      ". Available: ", paste(known, collapse = ", "), ".",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

#' The metrics tl_evaluate() computes when none are named
#'
#' @param is_classification Whether the task is classification
#' @return A character vector of metric names
#' @keywords internal
#' @noRd
tl_default_metrics <- function(is_classification) {
  if (is_classification) "accuracy" else c("rmse", "mae", "rsq")
}

#' Signal that no row is left to score
#'
#' An error with a class of its own, so resampling code can leave a fold
#' with nothing to score out of its summary instead of stopping the run.
#'
#' @param message The message
#' @param n_rows The number of rows offered for scoring
#' @param reason Why none of them could be scored, for a caller composing
#'   its own message, or NULL
#' @keywords internal
#' @noRd
tl_stop_no_scored_rows <- function(message, n_rows, reason = NULL) {
  stop(structure(
    class = c("tidylearn_no_scored_rows", "error", "condition"),
    list(message = message, call = NULL, n_rows = n_rows, reason = reason)
  ))
}

#' Refuse to score when every row has been dropped
#'
#' A row without a response or a prediction is dropped before any metric is
#' computed, and so is a row of a class the model was not trained on. With
#' none left, the metrics came back NaN or NA without naming the cause.
#'
#' @param no_response,no_prediction,unseen Logical vectors with an element
#'   per scored row: the response is missing; the prediction, or one of its
#'   probabilities, is missing; the class is one the model was not trained
#'   on
#' @return `TRUE`, invisibly, when at least one row can be scored
#' @keywords internal
#' @noRd
tl_check_scored_rows <- function(no_response, no_prediction, unseen = FALSE) {
  # A predict method returning the wrong number of rows is reported where
  # the lengths are compared, not here
  if (length(no_prediction) != length(no_response) ||
        any(!(no_response | no_prediction | unseen))) {
    return(invisible(TRUE))
  }

  count <- function(flags, one, several, what) {
    n <- sum(flags)
    if (n == 0) NULL else paste0(n, if (n == 1) one else several, what)
  }
  reasons <- c(
    count(no_response, " is", " are", " missing the response"),
    count(
      unseen, " belongs", " belong", " to a class the model was not trained on"
    ),
    count(
      no_prediction, " has", " have",
      " no prediction, which happens wherever a predictor is missing"
    )
  )
  last <- length(reasons)
  reason <- if (last > 2L) {
    paste0(paste(reasons[-last], collapse = ", "), " and ", reasons[last])
  } else {
    paste(reasons, collapse = " and ")
  }

  n_rows <- length(no_response)
  tl_stop_no_scored_rows(
    paste0(
      "None of the ", n_rows, " rows of the evaluation data can be scored: ",
      reason, "."
    ),
    n_rows = n_rows, reason = reason
  )
}

#' The classes a classification model was trained on
#'
#' @param object A supervised tidylearn classification model
#' @return A character vector of classes, the positive class second
#' @keywords internal
#' @noRd
tl_model_classes <- function(object) {
  # A model built outside tl_model() -- tl_tune_xgboost()'s, for one --
  # records no response_levels, so read them off its training response
  object$spec$response_levels %||%
    levels(tl_normalise_response(object$data[[object$spec$response_var]]))
}

#' The observed response a model is scored against
#'
#' A model fits its formula's left-hand side, which is the bare column only
#' when the formula says so. A model of \code{log(mpg) ~ wt + hp} predicts
#' on the log scale, so reading raw \code{mpg} compared logs with miles per
#' gallon: rmse 18.1 for a fit whose residual rmse is 0.106. The left-hand
#' side is evaluated on the scored rows instead, as \code{model.frame()}
#' does when it builds the response a fit uses.
#'
#' @param object A supervised tidylearn model
#' @param new_data The rows being scored
#' @return The response, one value per row of \code{new_data}
#' @keywords internal
#' @noRd
tl_observed_response <- function(object, new_data) {
  response_var <- object$spec$response_var
  if (!response_var %in% names(new_data)) {
    stop(
      "Response variable '", response_var,
      "' not found in the evaluation data.",
      call. = FALSE
    )
  }

  formula <- object$spec$formula
  if (!inherits(formula, "formula") || length(formula) < 3L ||
        is.name(formula[[2]])) {
    return(new_data[[response_var]])
  }

  lhs <- formula[[2]]
  observed <- tryCatch(
    eval(lhs, new_data, environment(formula) %||% baseenv()),
    error = function(e) {
      stop(
        "The response '", deparse1(lhs), "' could not be computed on the ",
        "evaluation data: ", conditionMessage(e),
        call. = FALSE
      )
    }
  )
  # scale() and the like return a one-column matrix
  if (is.matrix(observed) && ncol(observed) == 1L) {
    observed <- observed[, 1]
  }
  if (length(observed) != nrow(new_data)) {
    stop(
      "The response '", deparse1(lhs), "' gives ", length(observed),
      " value(s) for the ", nrow(new_data), " row(s) of the evaluation ",
      "data, so it cannot be scored row by row.",
      call. = FALSE
    )
  }
  observed
}

#' Evaluate a tidylearn model
#'
#' Scores a supervised model's predictions against the observed response.
#'
#' The observed response is the formula's left-hand side evaluated on the
#' scored rows, so a model of \code{log(mpg)} is scored against
#' \code{log(mpg)}, the scale it predicts on. For classification, the
#' observed classes are read against the classes the model was trained on,
#' whose second is the positive class; rows of a class the model never saw
#' are left out with a warning. Rows missing the response or a prediction
#' are dropped. \code{\link{tl_calc_classification_metrics}} describes
#' \code{"auc"} and \code{"pr_auc"} when the scored rows hold a single
#' class or lack one.
#'
#' With no row left to score -- \code{new_data} has no rows, or every row
#' is dropped -- it is an error of class \code{tidylearn_no_scored_rows},
#' naming the reason. \code{\link{tl_cv}} catches that class and leaves the
#' fold out.
#'
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
#'   \code{"rsq"}. Any other name is an error.
#' @param ... Additional arguments passed to \code{predict()}
#' @return A \link[tibble]{tibble} with columns \code{metric} (character)
#'   and \code{value} (numeric), containing one row per requested metric,
#'   and for \code{"auc"} on more than two classes an \code{auc_<class>}
#'   row per class as well. An unsupervised model has no response to score
#'   against and returns the single row \code{metric = "completed"},
#'   \code{value = 1}.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_evaluate(model)
#' tl_evaluate(model, metrics = c("rmse", "mape"))
#' }
#' @export
tl_evaluate <- function(object, new_data = NULL, metrics = NULL, ...) {
  if (!inherits(object, "tidylearn_supervised")) {
    # Unsupervised evaluation (placeholder)
    return(tibble::tibble(metric = "completed", value = 1))
  }

  is_classification <- isTRUE(object$spec$is_classification)
  if (is.null(metrics)) {
    metrics <- tl_default_metrics(is_classification)
  }
  tl_check_metric_names(metrics, is_classification)

  # With no new_data the stored rows are scored, and predict() is left to
  # read them itself. A model fitted on engineered features -- the PCA and
  # cluster candidates of tl_auto_ml() -- stores the engineered columns,
  # and predict() rebuilds those features on any data it is handed, so
  # passing the stored rows back built them twice and failed.
  predict_data <- new_data
  if (is.null(new_data)) {
    new_data <- object$data
  }
  actuals <- tl_observed_response(object, new_data)
  if (nrow(new_data) == 0L) {
    tl_stop_no_scored_rows(
      paste0(
        "'new_data' has no rows, so there is nothing to score. Check the ",
        "split or filter that produced it."
      ),
      n_rows = 0L
    )
  }

  predict_as <- function(type) {
    predict(object, new_data = predict_data, type = type, ...)
  }

  # Ask for the prediction type each metric family needs. The default
  # predict type is method-dependent -- logistic returns probabilities
  # for type = "response" -- so classification must request "class".
  extract_pred <- function(type) {
    preds <- predict_as(type)
    if (!".pred" %in% names(preds)) {
      stop(
        "predict() for method '", object$spec$method, "' did not return a ",
        "'.pred' column for type = '", type, "'.",
        call. = FALSE
      )
    }
    preds$.pred
  }

  if (is_classification) {
    # The model's classes, not the scored rows' declared levels, fix the
    # class set and with it the positive class
    model_levels <- tl_model_classes(object)
    aligned <- tl_align_classes(actuals, model_levels)
    predicted <- factor(
      as.character(extract_pred("class")), levels = model_levels
    )

    predicted_probs <- NULL
    if (any(c("auc", "pr_auc") %in% metrics)) {
      predicted_probs <- predict_as("prob")
    }

    no_prediction <- is.na(predicted)
    if (!is.null(predicted_probs)) {
      no_prediction <- no_prediction | !stats::complete.cases(predicted_probs)
    }
    tl_check_scored_rows(
      no_response = is.na(actuals),
      no_prediction = no_prediction,
      unseen = !is.na(actuals) & !aligned$keep
    )

    tl_calc_classification_metrics(
      actuals = aligned$actuals,
      predicted = predicted,
      predicted_probs = predicted_probs,
      metrics = metrics
    )
  } else {
    predicted <- extract_pred("response")
    tl_check_scored_rows(
      no_response = is.na(actuals),
      no_prediction = is.na(predicted)
    )

    tl_calc_regression_metrics(
      actuals = actuals,
      predicted = predicted,
      metrics = metrics
    )
  }
}

#' Cross-validation for tidylearn models
#'
#' Each fold's model is scored with \code{\link{tl_evaluate}}, so the
#' response is read as that function reads it: a transformed left-hand
#' side on its own scale, and classes against the fold model's.
#'
#' @param data Data frame
#' @param formula Model formula
#' @param method Modeling method
#' @param folds Number of cross-validation folds, a whole number between 2
#'   and \code{nrow(data)}
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
#' @param ... Additional arguments passed to \code{\link{tl_model}} for
#'   every fold. Arguments holding one value per row of \code{data} --
#'   \code{weights}, \code{subset}, \code{offset}, \code{foldid} and
#'   \code{strata} -- are refused, since they cannot follow the rows into
#'   a fold.
#' @return A list with two elements:
#'   \describe{
#'     \item{\code{$folds}}{A list of per-fold evaluation
#'       \link[tibble]{tibble}s, each with \code{metric} and
#'       \code{value} columns.}
#'     \item{\code{$summary}}{A \link[tibble]{tibble} with columns
#'       \code{metric}, \code{mean}, and \code{sd} summarizing
#'       performance across folds. A metric undefined on a fold -- auc
#'       on a fold holding one class -- is \code{NA} there and left out
#'       of the mean and sd. So is every metric of a fold none of whose
#'       rows can be scored, with a warning giving the reason.}
#'   }
#' @examples
#' \donttest{
#' cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3)
#' cv$summary
#' }
#' @export
tl_cv <- function(data, formula, method, folds = 5, metrics = NULL,
                  transform = NULL, ...) {
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }
  formula <- tl_as_formula(formula)
  n <- nrow(data)
  # A fractional count went through rep(seq_len(folds)), which truncates
  # it: folds = 2.5 ran two folds without a word
  if (!is.numeric(folds) || length(folds) != 1L || is.na(folds) ||
        folds != round(folds) || folds < 2 || folds > n) {
    stop(
      "'folds' must be a whole number between 2 and nrow(data) (", n,
      "). Got: ", tl_describe_value(folds), ".",
      call. = FALSE
    )
  }

  # An argument with one value per row of `data` reached every fold whole,
  # and the fit failed on "variable lengths differ". tl_compare_cv()
  # refuses the same arguments.
  per_row <- intersect(names2(list(...)), tl_per_row_args())
  if (length(per_row) > 0) {
    several <- length(per_row) > 1L
    stop(
      "tl_cv() cannot re-split '", paste(per_row, collapse = "', '"),
      "' across folds: ", if (several) "they hold" else "it holds",
      " one value per row of `data`, and each fold fits a subset of the ",
      "rows. Leave ", if (several) "them" else "it",
      " out, or loop over the folds with tl_model() to pass each fold its ",
      "own rows' values.",
      call. = FALSE
    )
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

    # Split data. A one-column frame drops to a bare vector without
    # drop = FALSE, and tl_model() refuses that.
    train_data <- data[train_indices, , drop = FALSE]
    test_data <- data[test_indices, , drop = FALSE]

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

    # Evaluate. tl_evaluate() refuses a fold with no row it can score --
    # every predictor missing, say -- and one such fold is no reason to
    # stop the run, so it is left out of the summary like any undefined
    # score.
    eval_result <- tryCatch(
      tl_evaluate(model, new_data = test_data, metrics = metrics),
      tidylearn_no_scored_rows = function(e) {
        warning(
          "Fold ", i, " is left out of the summary, since ",
          if (is.null(e$reason)) {
            "it has no rows to score."
          } else {
            paste0(
              "none of its ", e$n_rows, " rows can be scored: ", e$reason, "."
            )
          },
          call. = FALSE
        )
        tibble::tibble(
          metric = metrics %||% tl_default_metrics(
            isTRUE(model$spec$is_classification)
          ),
          value = NA_real_
        )
      }
    )

    cv_results[[i]] <- eval_result
    test_sizes[i] <- nrow(test_data)
  }

  # Combine results
  all_results <- dplyr::bind_rows(cv_results)

  # Calculate mean and sd for each metric
  summary_results <- all_results |>
    dplyr::group_by(.data$metric) |>
    dplyr::summarize(
      mean = mean(.data$value, na.rm = TRUE),
      sd = stats::sd(.data$value, na.rm = TRUE),
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
