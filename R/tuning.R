#' @title Hyperparameter Tuning Functions for tidylearn
#' @name tidylearn-tuning
#' @description
#' Functions for automatic hyperparameter tuning and selection
#' @importFrom stats model.matrix as.formula
#' @importFrom dplyr filter select mutate arrange
#' @importFrom tidyr crossing
NULL

#' Should a metric be maximised?
#'
#' The direction is \code{tl_metric_higher_better()}'s, the rule pipelines
#' and AutoML rank models by, so that a search and a leaderboard cannot
#' disagree about which score is best. A list kept here read every name
#' outside its error metrics as higher-is-better, unknown ones included.
#' The tuners refuse a name tidylearn does not compute before asking, and
#' one that reaches here anyway has no direction to give.
#'
#' @param metric Metric name
#' @return \code{TRUE} to maximise, \code{FALSE} to minimise
#' @keywords internal
#' @noRd
tl_metric_maximize <- function(metric) {
  higher <- tl_metric_higher_better(metric)
  if (anyNA(higher)) {
    stop(
      "Metric \"", metric, "\" has no direction to optimise: it is not one ",
      "tidylearn computes. Name one of: ",
      paste(c(tl_known_metrics(TRUE), tl_known_metrics(FALSE)),
            collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  higher
}

#' Tune hyperparameters for a model using grid search
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param method The modeling method to tune, one of the supervised methods
#'   \code{\link{tl_model}} fits
#' @param param_grid A named list of candidate values, one element for each
#'   \code{\link{tl_model}} argument to tune, named after it
#' @param folds Number of cross-validation folds, a whole number between 2
#'   and \code{nrow(data)}. \code{nrow(data)} leaves each row out in turn.
#' @param metric Metric to optimize: one of the names
#'   \code{\link{tl_evaluate}} computes for the task. Defaults to
#'   \code{"accuracy"} for classification and \code{"rmse"} for regression.
#' @param maximize Logical; whether to maximize (TRUE)
#'   or minimize (FALSE) the metric. \code{NULL}, the default, maximizes
#'   every metric but the error metrics \code{"rmse"}, \code{"mse"},
#'   \code{"mae"} and \code{"mape"}.
#' @param verbose Logical; whether to print progress
#' @param ... Additional arguments passed to \code{\link{tl_model}} for
#'   every fold and for the final fit. Arguments holding one value per row
#'   of \code{data} -- \code{weights}, \code{subset}, \code{offset},
#'   \code{foldid} and \code{strata} -- are refused, since they cannot
#'   follow the rows into a fold.
#' @return A tidylearn model object fitted with the best hyperparameters.
#'   Tuning results are stored as an attribute \code{"tuning_results"},
#'   a list containing \code{param_grid}, \code{results}, \code{best_params},
#'   \code{best_metric}, \code{metric}, and \code{maximize}.
#'
#'   \code{results} has one row per evaluated combination: \code{mean_metric}
#'   (the mean over the folds that produced a score), \code{n_folds_ok} (how
#'   many of the \code{folds} did), and a column per parameter. A parameter
#'   with a vector-valued candidate, such as \code{hidden_layers}, is a list
#'   column. A fold produces no score when its fit fails, when the metric
#'   is undefined on it -- \code{"auc"} on a fold holding one class, or
#'   \code{"precision"} on one where nothing is predicted positive -- or
#'   when none of its rows can be scored, as when every predictor is
#'   missing there, which is warned about.
#'
#'   Only combinations with \code{n_folds_ok} equal to \code{folds} are
#'   eligible to be best, since a mean over the folds that happened to
#'   succeed is not comparable with a mean over all of them. If no
#'   combination was scored on every fold, the best of those scored on the
#'   most folds is used, with a warning saying whether fits failed or the
#'   metric was undefined. If no combination was scored on any fold, the
#'   function stops.
#'
#'   For \code{method = "forest"}, an \code{mtry} above the number of
#'   predictors is capped at that number, with a warning, and duplicate
#'   combinations that result are evaluated once. Predictors are counted as
#'   randomForest counts them, so a column removed with \code{- id} is not
#'   one.
#' @examples
#' \donttest{
#' model <- tl_tune_grid(iris, Species ~ ., method = "tree",
#'   param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
#'   folds = 2, verbose = FALSE)
#' }
#' @export
tl_tune_grid <- function(data, formula, method,
                         param_grid, folds = 5,
                         metric = NULL,
                         maximize = NULL,
                         verbose = TRUE, ...) {
  # Input validation
  formula <- tl_as_formula(formula)
  if (!is.list(param_grid)) {
    stop("param_grid must be a named list", call. = FALSE)
  }
  tl_check_param_names(param_grid, "param_grid")
  # An empty candidate vector crosses to zero combinations, which read
  # later as "every parameter set failed in every fold"
  empty <- names(param_grid)[lengths(param_grid) == 0]
  if (length(empty) > 0) {
    stop("param_grid gives no candidate values for: ",
         paste(empty, collapse = ", "), ".", call. = FALSE)
  }
  tl_check_tuning_inputs(data, method, folds)
  dots <- list(...)
  tl_check_per_row_args(c(names2(dots), names(param_grid)), "tl_tune_grid()")

  # Determine if classification or regression, as tl_model() does: from the
  # response the formula computes, so factor(am) ~ . is a classification
  # although the am column is numeric. tl_model() also fits logistic
  # regression as classification whatever the response is stored as, so a
  # 0/1 numeric response has to default to a classification metric too.
  is_classification <- tl_tuning_task(formula, data, method)

  # Default metric based on problem type
  if (is.null(metric)) {
    metric <- if (is_classification) "accuracy" else "rmse"
  }
  tl_check_tuning_metric(metric, is_classification)

  # Optimisation direction. Derived from the metric itself, not from
  # whether the caller supplied one -- naming a metric while leaving
  # `maximize` at its NULL default previously left it NULL, and the
  # `if (maximize)` below then failed with "argument is of length zero".
  if (is.null(maximize)) {
    maximize <- tl_metric_maximize(metric)
  }
  tl_check_maximize(maximize)

  # Create parameter grid, one list of arguments per combination
  param_df <- do.call(tidyr::crossing, param_grid)
  param_combinations <- lapply(
    seq_len(nrow(param_df)), function(i) tl_tune_grid_row(param_df, i)
  )

  # Capping can turn two candidates into the same one, and fitting a
  # combination twice would only repeat its score
  param_combinations <- unique(
    tl_tune_cap_mtry(param_combinations, method, formula, data)
  )
  n_sets <- length(param_combinations)

  if (verbose) {
    message(
      "Tuning ", method, " model with ",
      n_sets, " parameter combinations"
    )
    message(
      "Cross-validation with ", folds, " folds"
    )
    message(
      "Optimizing for ", metric, ", ",
      ifelse(maximize, "maximizing", "minimizing")
    )
  }

  # Create cross-validation splits
  cv_splits <- tl_resample_folds(data, folds)

  # Initialize results storage
  tuning_results <- list()

  # Shared by every fit of this search, the final one included, so the
  # response-conversion warning is given once
  conversion <- tl_warn_once()

  # Loop through parameter combinations
  for (i in seq_len(n_sets)) {
    params <- param_combinations[[i]]

    if (verbose) {
      message(
        "Parameter set ", i, " of ", n_sets, ": ",
        tl_tune_format_params(params)
      )
    }

    tuning_results[[i]] <- tl_tune_score_set(
      params, cv_splits, formula, method, metric, dots, conversion
    )

    if (verbose) {
      message(
        "  Mean ", metric, ": ",
        round(tuning_results[[i]]$mean_metric, 4)
      )
    }
  }

  # Convert results to data frame
  results_df <- tl_tune_results_frame(
    tuning_results, param_combinations, names(param_grid)
  )

  # Find best parameter set among those scored on every fold
  best_idx <- tl_tune_select_best(
    results_df, maximize, folds,
    vapply(param_combinations, tl_tune_format_params, character(1)),
    n_failed = vapply(tuning_results, function(x) x$n_fit_failed,
                      integer(1)),
    metric = metric
  )

  # Taken from the combinations rather than the results frame, which holds
  # vector-valued candidates as list cells
  best_params <- param_combinations[[best_idx]]

  if (verbose) {
    message(
      "Best parameters found: ",
      tl_tune_format_params(best_params)
    )
    message(
      "Best ", metric, ": ",
      round(results_df$mean_metric[best_idx], 4)
    )
  }

  # Fit final model with best parameters
  final_model_args <- c(
    list(
      data = data,
      formula = formula,
      method = method
    ),
    best_params,
    dots
  )

  final_model <- withCallingHandlers(
    do.call(tl_model, final_model_args),
    tidylearn_response_conversion = conversion
  )

  # Add tuning results to model
  attr(final_model, "tuning_results") <- list(
    param_grid = param_grid,
    results = results_df,
    best_params = best_params,
    best_metric = results_df$mean_metric[best_idx],
    metric = metric,
    maximize = maximize
  )

  final_model
}

#' Refuse a metric the evaluation cannot produce
#'
#' @param metric The requested metric name
#' @param eval_metrics The tibble returned by \code{tl_evaluate()}
#' @return `TRUE`, invisibly, when the metric is present
#' @keywords internal
#' @noRd
tl_check_metric_available <- function(metric, eval_metrics,
                                      model = NULL, new_data = NULL) {
  if (metric %in% eval_metrics$metric) {
    return(invisible(TRUE))
  }

  # tl_evaluate() filters to the metrics it was asked for, so an
  # unrecognised name leaves nothing behind to list. Ask again without
  # the filter, on this error path only, to find out what this task
  # actually produces.
  available <- eval_metrics$metric
  if (length(available) == 0 && !is.null(model)) {
    available <- tryCatch(
      suppressWarnings(suppressMessages(
        tl_evaluate(model, new_data = new_data)$metric
      )),
      error = function(e) character()
    )
  }

  stop(
    "Metric \"", metric, "\" was not produced for this task",
    if (length(available)) {
      paste0(". Available: ", paste0("\"", available, "\"", collapse = ", "))
    } else {
      ""
    },
    ". Classification metrics are not computed for a numeric response, ",
    "nor regression metrics for a factor one.",
    call. = FALSE
  )
}

#' Refuse a metric argument the tuners cannot optimise
#'
#' The name is checked before anything is fitted. \code{metric =
#' c("rmse", "mae")} used to reach the first fold and fail there with "the
#' condition has length > 1", and an unknown name failed with whatever
#' \code{tl_evaluate()} said about it.
#'
#' @param metric The requested metric
#' @param is_classification Whether the task is classification
#' @return `TRUE`, invisibly, when the metric is usable
#' @keywords internal
#' @noRd
tl_check_tuning_metric <- function(metric, is_classification) {
  if (!is.character(metric) || length(metric) != 1L || is.na(metric)) {
    stop(
      "'metric' must be a single metric name, such as \"",
      if (is_classification) "accuracy" else "rmse", "\"; got ",
      tl_describe_value(metric), ".",
      call. = FALSE
    )
  }
  tl_check_metric_available(
    metric, list(metric = tl_known_metrics(is_classification))
  )
}

#' Refuse a maximize argument that is not a single logical
#'
#' @param maximize The optimisation direction, already defaulted
#' @return `TRUE`, invisibly, when it is TRUE or FALSE
#' @keywords internal
#' @noRd
tl_check_maximize <- function(maximize) {
  if (!is.logical(maximize) || length(maximize) != 1L || is.na(maximize)) {
    stop(
      "'maximize' must be TRUE, FALSE or NULL, which follows the metric; ",
      "got ", tl_describe_value(maximize), ".",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Refuse a parameter list whose elements are not all named
#'
#' Each element is passed to \code{tl_model()} under its name. An unnamed
#' one went in positionally, where \code{rpart.control()} and the like
#' discard it, so \code{list(c(0.01, 0.1))} fitted the same model once per
#' candidate and reported a best value under no name at all.
#'
#' @param params \code{param_grid} or \code{param_space}
#' @param arg Its argument name, for the message
#' @return `TRUE`, invisibly, when every element has a name of its own
#' @keywords internal
#' @noRd
tl_check_param_names <- function(params, arg) {
  param_names <- names2(params)
  unnamed <- which(is.na(param_names) | param_names == "")
  if (length(unnamed) > 0) {
    stop(
      arg, " must be a named list: ",
      if (length(unnamed) == 1L) "element " else "elements ",
      paste(unnamed, collapse = ", "),
      if (length(unnamed) == 1L) " has" else " have",
      " no name. Name each element after the tl_model() argument it sets, ",
      "as in list(cp = c(0.01, 0.1)).",
      call. = FALSE
    )
  }

  repeated <- unique(param_names[duplicated(param_names)])
  if (length(repeated) > 0) {
    stop(
      arg, " names must be unique, since each sets one tl_model() ",
      "argument; repeated: ", paste(repeated, collapse = ", "), ".",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

#' Refuse data, a method or a fold count the tuners cannot use
#'
#' rsample checks the fold count itself, but its messages name \code{v},
#' which is not an argument of anything the caller wrote. An unknown method
#' failed every fit, and the search then advised checking the candidate
#' values.
#'
#' @param data The training data
#' @param method The method to tune
#' @param folds The number of folds
#' @return `TRUE`, invisibly, when all three are usable
#' @keywords internal
#' @noRd
tl_check_tuning_inputs <- function(data, method, folds) {
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }

  methods <- tl_supervised_methods()
  if (!is.character(method) || length(method) != 1L ||
        !method %in% methods) {
    stop(
      "'method' must be one of the supervised methods: ",
      paste(methods, collapse = ", "), "; got ",
      tl_describe_value(method), ".",
      call. = FALSE
    )
  }

  tl_check_folds(folds, data)
}

#' Whether a search is a classification
#'
#' Decided as \code{tl_model()} decides it, so the default metric and the
#' metric check agree with the model each fold fits. The response is the
#' one the formula computes: reading the raw column took
#' \code{factor(am) ~ wt + hp} for a regression, defaulted to rmse, which
#' the classifier does not produce, and refused \code{"accuracy"}. Logistic
#' regression is a classification whatever the response is stored as.
#'
#' @param formula The model formula
#' @param data The training data
#' @param method The method searched
#' @return \code{TRUE} for classification
#' @keywords internal
#' @noRd
tl_tuning_task <- function(formula, data, method) {
  y <- tl_formula_response(formula, data)
  is.factor(y) || is.character(y) || method == "logistic"
}

#' The folds a resampling search runs over
#'
#' \code{rsample::vfold_cv()} refuses \code{v = nrow(data)} and points to
#' \code{loo_cv()}, so a fold count that \code{tl_check_folds()} and
#' \code{tl_cv()} both accept failed there. It is leave-one-out, and is
#' run as such.
#'
#' @param data The data to split
#' @param folds The number of folds, already checked
#' @return An rsample rset whose \code{splits} are the folds
#' @keywords internal
#' @noRd
tl_resample_folds <- function(data, folds) {
  if (folds == nrow(data)) {
    rsample::loo_cv(data)
  } else {
    rsample::vfold_cv(data, v = folds)
  }
}

#' Refuse a fold count that cannot split the data
#'
#' @param folds The number of folds
#' @param data The data to split
#' @return `TRUE`, invisibly, when the count is usable
#' @keywords internal
#' @noRd
tl_check_folds <- function(folds, data) {
  n <- nrow(data)
  if (!is.numeric(folds) || length(folds) != 1L || is.na(folds) ||
        folds != round(folds) || folds < 2 || folds > n) {
    stop(
      "'folds' must be a whole number between 2 and nrow(data) (", n,
      "). Got: ", tl_describe_value(folds), ".",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Refuse a per-row argument a resampling search cannot split
#'
#' An argument with one value per row of the data reached every fold
#' whole, so each fit failed on "variable lengths differ", and the search
#' reported that every parameter set had failed.
#'
#' @param arg_names Names of the arguments to be passed to tl_model()
#' @param caller The function refusing them, for the message
#' @return `TRUE`, invisibly, when none holds one value per row
#' @keywords internal
#' @noRd
tl_check_per_row_args <- function(arg_names, caller) {
  per_row <- intersect(arg_names, tl_per_row_args())
  if (length(per_row) == 0) {
    return(invisible(TRUE))
  }

  several <- length(per_row) > 1L
  stop(
    caller, " cannot re-split '", paste(per_row, collapse = "', '"),
    "' across folds: ", if (several) "they hold" else "it holds",
    " one value per row of `data`, and each fold fits a subset of the ",
    "rows. Tune without ", if (several) "them" else "it", ", then pass ",
    if (several) "them" else "it", " to tl_model() with the parameters ",
    "chosen.",
    call. = FALSE
  )
}

#' Cross-validate one parameter set
#'
#' A fold yields no score for one of two reasons: its fit failed, or the
#' fit succeeded and the fold could not be scored -- the metric is
#' undefined on its rows (auc on a fold holding one class, precision where
#' nothing is predicted positive), or no row of it can be scored at all.
#' Both leave the score \code{NA}. Failed fits are counted separately so
#' that the messages can say which happened.
#'
#' @param params The parameter values, a named list
#' @param cv_splits The folds, from \code{rsample::vfold_cv()}
#' @param formula,method Passed to \code{tl_model()}
#' @param metric The metric scored
#' @param dots Further arguments for \code{tl_model()}
#' @param conversion Handler for \code{tl_model()}'s response-conversion
#'   warning, shared by every fit of one search; see
#'   \code{tl_warn_once()}
#' @return A list: \code{mean_metric} over the scored folds,
#'   \code{n_folds_ok}, \code{n_fit_failed} and \code{fold_metrics}
#' @keywords internal
#' @noRd
tl_tune_score_set <- function(params, cv_splits, formula, method, metric,
                              dots, conversion = tl_warn_once()) {
  n_folds <- length(cv_splits$splits)
  fold_metrics <- rep(NA_real_, n_folds)
  fit_failed <- logical(n_folds)

  for (j in seq_len(n_folds)) {
    split <- cv_splits$splits[[j]]
    model_args <- c(
      list(data = rsample::analysis(split), formula = formula,
           method = method),
      params,
      dots
    )

    # tl_model() notes things about the response -- that a numeric column
    # with few distinct values is being treated as regression, say -- and
    # every fold refit repeated it. The final fit on all the rows gives it
    # once, as tl_cv() leaves it to the caller's own fit. The warning that
    # a 0/1 response is converted for logistic regression repeated the same
    # way, and is let through once per search.
    fold_model <- tryCatch(
      withCallingHandlers(
        suppressMessages(do.call(tl_model, model_args)),
        tidylearn_response_conversion = conversion
      ),
      error = function(e) {
        warning(
          "Error fitting model with parameters: ",
          tl_tune_format_params(params),
          ". Error: ", conditionMessage(e),
          call. = FALSE
        )
        NULL
      }
    )
    if (is.null(fold_model)) {
      fit_failed[j] <- TRUE
      next
    }

    # tl_evaluate() refuses a fold on which no row can be scored -- every
    # predictor missing, say. The fit did not fail, and one such fold is no
    # reason to stop the search, so it goes unscored like a fold whose
    # metric is undefined.
    valid_fold <- rsample::assessment(split)
    eval_metrics <- tryCatch(
      tl_evaluate(fold_model, valid_fold, metrics = metric),
      tidylearn_no_scored_rows = function(e) {
        warning(
          "Fold ", j, " is left out of the score for ",
          if (length(params) > 0) {
            tl_tune_format_params(params)
          } else {
            "the method's defaults"
          },
          ", since ", tl_unscored_fold_reason(e),
          call. = FALSE
        )
        NULL
      }
    )
    if (is.null(eval_metrics)) {
      next
    }

    # A metric the evaluation did not produce left a zero-length
    # right-hand side, and the assignment failed with "replacement has
    # length zero", which says nothing about the metric asked for
    tl_check_metric_available(metric, eval_metrics, fold_model, valid_fold)
    fold_metrics[j] <- eval_metrics$value[eval_metrics$metric == metric]
  }

  # The count is kept alongside the mean because a mean over fewer folds
  # is not comparable with one over all of them
  n_folds_ok <- sum(!is.na(fold_metrics))
  list(
    mean_metric = if (n_folds_ok > 0) {
      mean(fold_metrics, na.rm = TRUE)
    } else {
      NA_real_
    },
    n_folds_ok = n_folds_ok,
    n_fit_failed = sum(fit_failed),
    fold_metrics = fold_metrics
  )
}

#' A warning handler that lets the first warning through
#'
#' A search refits the model once per fold and set, and a warning about the
#' data rather than the fit -- the response converted to a factor for
#' logistic regression -- said the same thing every time: seven times for
#' a 2-set, 3-fold search. One handler is shared by every fit of a run, so
#' the warning is given once.
#'
#' @return A function for \code{withCallingHandlers()}
#' @keywords internal
#' @noRd
tl_warn_once <- function() {
  warned <- FALSE
  function(w) {
    if (warned) {
      invokeRestart("muffleWarning")
    }
    warned <<- TRUE
  }
}

#' Why a resampling fold could not be scored
#'
#' Worded as \code{tl_cv()} words it, for the warnings of the other
#' resampling functions.
#'
#' @param e A condition of class \code{tidylearn_no_scored_rows}
#' @return The end of a sentence, from "it has" or "none of its" to the
#'   full stop
#' @keywords internal
#' @noRd
tl_unscored_fold_reason <- function(e) {
  if (is.null(e$reason)) {
    "it has no rows to score."
  } else {
    paste0("none of its ", e$n_rows, " rows can be scored: ", e$reason, ".")
  }
}

#' Refuse a parameter range that runs the wrong way
#'
#' A continuous range is sampled with \code{runif(1, min, max)} and a
#' log-uniform one with \code{exp(runif(1, log(min), log(max)))}. Both
#' return \code{NaN} when \code{min > max}, and R gives only a warning,
#' so \code{c(0.1, 0.001)} instead of \code{c(0.001, 0.1)} produced a
#' full grid of NaN parameters, fitted models with them, and reported
#' \code{best_params} of NaN -- without failing anywhere.
#'
#' Integer ranges are unaffected: \code{500:100} is a valid descending
#' sequence and \code{sample()} draws from it happily.
#'
#' @param param_space The space passed to \code{tl_tune_random()}
#' @return `TRUE`, invisibly, when every range is usable
#' @keywords internal
#' @noRd
tl_check_param_space <- function(param_space) {
  # An empty candidate vector failed in the draw with "invalid first
  # argument", which names neither the parameter nor the cause
  empty <- names(param_space)[lengths(param_space) == 0]
  if (length(empty) > 0) {
    stop("param_space gives no candidate values for: ",
         paste(empty, collapse = ", "), ".", call. = FALSE)
  }

  for (param_name in names(param_space)) {
    param_def <- param_space[[param_name]]
    # A function is sampled by calling it, and a list is a set of whole
    # candidates -- list(10, 1, "log") is three candidates, not a range
    if (is.function(param_def) || is.list(param_def)) {
      next
    }

    is_log_spec <- length(param_def) == 3 &&
      identical(as.character(param_def[3]), "log") &&
      !anyNA(suppressWarnings(as.numeric(param_def[1:2])))

    bounds <- if (is_log_spec) {
      as.numeric(param_def[1:2])
    } else if (is.numeric(param_def) && length(param_def) == 2 &&
                 !all(param_def == floor(param_def))) {
      param_def
    } else {
      next
    }

    if (!all(is.finite(bounds))) {
      stop(
        "param_space$", param_name,
        " has a non-finite bound: c(", paste(bounds, collapse = ", "), ").",
        call. = FALSE
      )
    }

    if (bounds[1] == bounds[2]) {
      # Reversing equal bounds changes nothing, so the old advice to write
      # c(20.5, 20.5) as c(20.5, 20.5) was no help
      stop(
        "param_space$", param_name, " is a range with equal ends, ",
        bounds[1], ". To fix the parameter, give the single value ",
        bounds[1], ".",
        call. = FALSE
      )
    }

    if (bounds[1] > bounds[2]) {
      stop(
        "param_space$", param_name, " runs from ", bounds[1], " to ",
        bounds[2], ", but a range is c(min, max). Sampling it would give ",
        "NaN for every iteration. Write it as c(", bounds[2], ", ",
        bounds[1], if (is_log_spec) ", \"log\")" else ")", ".",
        call. = FALSE
      )
    }

    if (is_log_spec && bounds[1] <= 0) {
      stop(
        "param_space$", param_name,
        " is log-uniform, so both bounds must be positive, but the lower ",
        "bound is ", bounds[1], ".",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}

#' Draw one value from a random-search parameter space
#'
#' @param param_def One element of \code{param_space}
#' @param param_name Its name, for the error message
#' @return A single draw
#' @keywords internal
#' @noRd
tl_draw_param <- function(param_def, param_name = "parameter") {
  # Order matters here. A log-uniform spec c(min, max, "log") is a
  # CHARACTER vector -- c() coerces -- so it has to be recognised before
  # any is.numeric() branch, and the whole-number test has to come before
  # the continuous one or an integer set like c(100, 500) gets sampled
  # with runif() and yields 234.66.
  is_log_spec <- !is.list(param_def) && length(param_def) == 3 &&
    identical(as.character(param_def[3]), "log") &&
    !anyNA(suppressWarnings(as.numeric(param_def[1:2])))

  if (is.function(param_def)) {
    # Custom sampling function
    param_def()
  } else if (is.list(param_def) && length(param_def) > 0) {
    # A list is a set of candidates that are not single values, such as
    # hidden_layers = list(c(10), c(20, 10)). Each is drawn whole.
    param_def[[sample.int(length(param_def), 1)]]
  } else if (is_log_spec) {
    # Log-uniform range: [min, max, "log"]
    bounds <- as.numeric(param_def[1:2])
    exp(runif(1, log(bounds[1]), log(bounds[2])))
  } else if (is.atomic(param_def) && length(param_def) == 1) {
    # A single value is a fixed setting. sample() cannot be trusted with
    # it: sample(20, 1) draws from 1:20, so minsplit = 20 was tuned as
    # though it were a range.
    param_def
  } else if (is.integer(param_def) ||
               (is.numeric(param_def) &&
                  all(param_def == floor(param_def)))) {
    if (length(param_def) == 2) {
      # Integer range: [min, max]. Indexed rather than sampled, because a
      # range whose ends match, c(20, 20), is the single number 20, and
      # sample() would draw from 1:20.
      candidates <- param_def[1]:param_def[2]
      candidates[sample.int(length(candidates), 1)]
    } else {
      # Discrete values
      sample(param_def, 1)
    }
  } else if (is.numeric(param_def) && length(param_def) == 2) {
    # Continuous range: [min, max]
    runif(1, param_def[1], param_def[2])
  } else if (is.numeric(param_def) && length(param_def) >= 3) {
    # Discrete set of any numbers, e.g. c(0.001, 0.01, 0.1). Only whole
    # numbers reached the discrete branch above, so the natural way to
    # write a set of candidate cp or alpha values -- the parameters that
    # are never integers -- was rejected as an "Unsupported parameter
    # space definition", while tl_tune_grid() took the same vector without
    # complaint.
    sample(param_def, 1)
  } else if (is.character(param_def) || is.factor(param_def)) {
    # Categorical parameter
    sample(param_def, 1)
  } else if (is.logical(param_def)) {
    # Logical parameter, drawn from the values supplied. c(TRUE, TRUE)
    # reduces to one value, which is returned rather than handed to
    # sample().
    values <- unique(param_def)
    if (length(values) == 1) values else sample(values, 1)
  } else {
    stop(
      "Unsupported parameter space definition ",
      "for ", param_name, call. = FALSE
    )
  }
}

#' Extract one grid row as arguments for tl_model()
#'
#' \code{tidyr::crossing()} stores a candidate that is not a single value,
#' such as \code{hidden_layers = c(10, 5)}, in a list column, so the row
#' holds \code{list(c(10, 5))}. Passed on as it stands, the model receives
#' a list where it expects the vector. Each list cell is unwrapped exactly
#' once, which leaves a candidate that is itself a list intact.
#'
#' @param param_df The grid from \code{tidyr::crossing()}
#' @param i Row index
#' @return A named list of arguments
#' @keywords internal
#' @noRd
tl_tune_grid_row <- function(param_df, i) {
  row <- as.list(param_df[i, , drop = FALSE])
  lapply(row, function(value) if (is.list(value)) value[[1]] else value)
}

#' Describe a parameter set for messages
#'
#' @param params A named list of parameter values
#' @return A single string such as \code{"cp=0.01, hidden_layers=c(10, 5)"}
#' @keywords internal
#' @noRd
tl_tune_format_params <- function(params) {
  # round() on the whole set failed with "non-numeric argument to
  # mathematical function" as soon as one parameter was a string, and
  # paste() spread a vector-valued parameter across several entries
  values <- vapply(params, tl_tune_format_value, character(1))
  paste(names(params), values, sep = "=", collapse = ", ")
}

#' Describe one parameter value
#'
#' @param value A candidate value: a single value, a vector or a list
#' @return A single string, such as \code{"0.01"} or \code{"c(10, 5)"}
#' @keywords internal
#' @noRd
tl_tune_format_value <- function(value) {
  if (is.atomic(value) && length(value) == 1) {
    format(value, digits = 4)
  } else {
    paste(deparse(value), collapse = "")
  }
}

#' Assemble the tuning results data frame
#'
#' @param tuning_results Per-set lists of \code{mean_metric} and
#'   \code{n_folds_ok}
#' @param param_combinations Per-set lists of parameter values
#' @param param_names The tuned parameters, in column order
#' @param iteration Whether to lead with an \code{iteration} column
#' @return A data frame with one row per parameter set
#' @keywords internal
#' @noRd
tl_tune_results_frame <- function(tuning_results, param_combinations,
                                  param_names, iteration = FALSE) {
  results_df <- data.frame(
    mean_metric = vapply(tuning_results, function(x) x$mean_metric,
                         numeric(1)),
    n_folds_ok = vapply(tuning_results, function(x) x$n_folds_ok,
                        integer(1))
  )
  if (iteration) {
    results_df <- cbind(
      data.frame(iteration = seq_len(nrow(results_df))), results_df
    )
  }

  # A parameter whose candidates are all single values becomes an ordinary
  # column. One with any vector-valued candidate cannot, so it is kept as
  # a list column with one cell per set.
  for (param in param_names) {
    values <- lapply(param_combinations, function(p) p[[param]])
    is_scalar <- vapply(values, function(v) {
      is.atomic(v) && length(v) == 1
    }, logical(1))
    results_df[[param]] <- if (all(is_scalar)) unlist(values) else values
  }

  results_df
}

#' Choose the best parameter set
#'
#' A mean over the folds that happened to succeed is not comparable with a
#' mean over all of them: a set that failed on the hardest fold is scored
#' only on the easier ones, and could win for that reason alone. Only sets
#' scored on every fold are therefore eligible. When none is, the choice
#' falls back to the best of the sets scored on the most folds, with a
#' warning naming it. When no set was scored on any fold, it stops.
#'
#' A fold without a score either failed to fit or fitted and left the
#' metric undefined. The remedies differ -- fix the candidate values, or
#' choose a metric these folds define -- so the messages say which
#' happened. The warning used to blame failed fits in both cases.
#'
#' @param results_df The results frame, with \code{mean_metric} and
#'   \code{n_folds_ok}
#' @param maximize Whether a higher metric is better
#' @param folds The number of folds requested
#' @param labels A description of each set, for the warning
#' @param n_failed The folds on which each set's fit failed. The rest of
#'   its unscored folds had an undefined metric.
#' @param metric The metric name, for the messages
#' @return The row index of the chosen set
#' @keywords internal
#' @noRd
tl_tune_select_best <- function(results_df, maximize, folds, labels,
                                n_failed = folds - results_df$n_folds_ok,
                                metric = NULL) {
  n_ok <- results_df$n_folds_ok
  any_failed <- any(n_failed > 0)
  any_undefined <- any(folds - n_ok - n_failed > 0)
  metric_name <- if (is.null(metric)) {
    "the metric"
  } else {
    paste0("\"", metric, "\"")
  }
  searched <- paste0(
    nrow(results_df), " set", if (nrow(results_df) == 1) "" else "s",
    ", ", folds, " folds"
  )

  # With nothing scored there is nothing to choose, and carrying on left
  # best_params empty for the final fit to fail on obscurely
  if (!any(n_ok > 0)) {
    if (!any_undefined) {
      stop(
        "Every parameter set failed in every fold (", searched, "), so ",
        "there is no score to choose the best from. The warnings above ",
        "give the error from each fit; check the candidate values against ",
        "the arguments the method accepts.",
        call. = FALSE
      )
    }
    if (!any_failed) {
      stop(
        "No parameter set could be scored: every fit succeeded, but ",
        metric_name, " could not be computed on any fold (", searched,
        "). The warnings above say why; tune on a metric these folds can ",
        "produce.",
        call. = FALSE
      )
    }
    stop(
      "No parameter set could be scored on any fold (", searched, "): ",
      "some fits failed, and on the other folds ", metric_name, " could ",
      "not be computed. The warnings above give each cause.",
      call. = FALSE
    )
  }

  complete <- n_ok == folds
  pool <- if (any(complete)) which(complete) else which(n_ok == max(n_ok))
  scores <- results_df$mean_metric[pool]
  best_idx <- pool[if (maximize) which.max(scores) else which.min(scores)]

  if (!any(complete)) {
    cause <- if (!any_undefined) {
      "The warnings above give the error from each failed fit."
    } else if (!any_failed) {
      paste0(
        "No fit failed: ", metric_name, " could not be computed on the ",
        "other folds, and the warnings above say why."
      )
    } else {
      paste0(
        "On some folds the fit failed and on others ", metric_name,
        " could not be computed; the warnings above give each cause."
      )
    }
    warning(
      if (any_undefined) {
        "No parameter set was scored on all "
      } else {
        "No parameter set completed all "
      },
      folds, " folds. Using ", labels[best_idx], ", the best of the sets ",
      "scored on ", max(n_ok), " of them, so its score rests on fewer ",
      "folds than were requested. ", cause,
      call. = FALSE
    )
  }

  best_idx
}

#' Cap a random forest's mtry at the number of predictors
#'
#' A default grid is built without seeing the data, so it cannot know how
#' many predictors there are, and a caller's own grid can overshoot in the
#' same way. randomForest resets an mtry above that count to the count,
#' with a warning, in every fold -- so the fit succeeds, but the results
#' credit the score to an mtry that was never used, and two candidates that
#' both overshoot are the same model scored twice. Capping here instead of
#' dropping the values keeps the all-predictors candidate the caller asked
#' for, and the results record the value each model was actually fitted
#' with.
#'
#' The count is the one randomForest's formula method makes. It rebuilds
#' the model frame from the formula's term labels and samples from the
#' variables in it: a factor counts once, \code{y ~ x1 * x2} has two
#' predictors, not three, and \code{y ~ . - id} does not count \code{id}.
#' The formula's own model frame keeps a column that \code{- id} removed
#' from the terms, so counting its columns overstated the predictors.
#'
#' @param param_combinations Per-set lists of parameter values
#' @param method The model method
#' @param formula The model formula
#' @param data The training data
#' @return \code{param_combinations}, with any mtry capped
#' @keywords internal
#' @noRd
tl_tune_cap_mtry <- function(param_combinations, method, formula, data) {
  if (method != "forest") {
    return(param_combinations)
  }

  is_cappable <- function(p) {
    is.numeric(p$mtry) && length(p$mtry) == 1 && !is.na(p$mtry)
  }
  if (!any(vapply(param_combinations, is_cappable, logical(1)))) {
    return(param_combinations)
  }

  # A formula the data cannot satisfy fails every fit with tl_model()'s
  # own message, which says more than a terms() error would here, and a
  # formula with no predictors fails the same way
  n_predictors <- tryCatch(
    {
      labels <- attr(stats::terms(formula, data = data), "term.labels")
      variables <- attr(stats::terms(stats::reformulate(labels)), "variables")
      length(variables) - 1L
    },
    error = function(e) NULL
  )
  if (is.null(n_predictors) || n_predictors < 1L) {
    return(param_combinations)
  }

  # Below 1 randomForest resets mtry to 1 with its own warning in every
  # fold, and the results credited the value asked for
  too_small <- unique(unlist(lapply(param_combinations, function(p) {
    if (is_cappable(p) && p$mtry < 1) p$mtry
  })))
  if (length(too_small) > 0) {
    warning(
      "mtry = ", paste(sort(too_small), collapse = ", "),
      " is below 1, so it is raised to 1 and the results report that value.",
      call. = FALSE
    )
    param_combinations <- lapply(param_combinations, function(p) {
      if (is_cappable(p)) {
        p$mtry <- max(p$mtry, 1)
      }
      p
    })
  }

  too_large <- unique(unlist(lapply(param_combinations, function(p) {
    if (is_cappable(p) && p$mtry > n_predictors) p$mtry
  })))
  if (length(too_large) == 0) {
    return(param_combinations)
  }

  warning(
    "mtry = ", paste(sort(too_large), collapse = ", "),
    if (length(too_large) == 1) " exceeds" else " exceed",
    " the ", n_predictors, " predictors in ",
    paste(deparse(formula), collapse = " "),
    ". randomForest would reset ",
    if (length(too_large) == 1) "it" else "them",
    " to ", n_predictors, " in every fold, so ",
    if (length(too_large) == 1) "it is" else "they are",
    " capped at ", n_predictors, " and the results report that value.",
    call. = FALSE
  )

  lapply(param_combinations, function(p) {
    if (is_cappable(p)) {
      p$mtry <- min(p$mtry, n_predictors)
    }
    p
  })
}

#' Tune hyperparameters using random search
#'
#' @param data A data frame containing the training
#'   data
#' @param formula A formula specifying the model
#' @param method The modeling method to tune, one of the supervised methods
#'   \code{\link{tl_model}} fits
#' @param param_space A named list of parameter spaces to sample from, one
#'   element for each \code{\link{tl_model}} argument to tune, named after
#'   it. Each element is read by its type and length:
#'   \describe{
#'     \item{a function}{called with no arguments to draw one value}
#'     \item{a list}{a set of candidates, each drawn whole, e.g.
#'       \code{list(c(10), c(20, 10))} for \code{hidden_layers}}
#'     \item{\code{c(min, max, "log")}}{log-uniform draw between
#'       \code{min} and \code{max}}
#'     \item{a single value}{used as given in every iteration}
#'     \item{two whole numbers}{integer range, e.g.
#'       \code{c(10, 20)} draws from 10:20}
#'     \item{three or more numbers}{a discrete set, sampled from as
#'       given, whether or not they are whole}
#'     \item{two other numbers}{uniform draw between them, e.g.
#'       \code{c(0.01, 0.1)}}
#'     \item{character or factor}{categorical, sampled from as given}
#'     \item{logical}{sampled from the values given}
#'   }
#' @param n_iter Number of random parameter
#'   combinations to try, a whole number of at least 1
#' @param folds Number of cross-validation folds, a whole number between 2
#'   and \code{nrow(data)}. \code{nrow(data)} leaves each row out in turn.
#' @param metric Metric to optimize, as for \code{\link{tl_tune_grid}}
#' @param maximize Logical; whether to maximize (TRUE)
#'   or minimize (FALSE) the metric. \code{NULL}, the default, follows the
#'   metric, as for \code{\link{tl_tune_grid}}.
#' @param verbose Logical; whether to print progress
#' @param seed Random seed for reproducibility
#' @param ... Additional arguments passed to \code{\link{tl_model}} for
#'   every fold and for the final fit. Per-row arguments are refused, as for
#'   \code{\link{tl_tune_grid}}.
#' @return A tidylearn model object fitted with the best hyperparameters.
#'   Tuning results are stored as an attribute \code{"tuning_results"},
#'   a list containing \code{param_space}, \code{results}, \code{best_params},
#'   \code{best_metric}, \code{metric}, and \code{maximize}.
#'
#'   \code{results} has one row per iteration: \code{iteration},
#'   \code{mean_metric}, \code{n_folds_ok}, and a column per parameter, as
#'   described for \code{\link{tl_tune_grid}}. The best parameters are chosen
#'   by the same rules, and \code{mtry} is capped the same way; duplicate
#'   draws are kept, so there are always \code{n_iter} rows.
#' @examples
#' \donttest{
#' model <- tl_tune_random(mtcars, mpg ~ ., method = "tree",
#'   param_space = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
#'   n_iter = 3, folds = 2, verbose = FALSE)
#' }
#' @export
tl_tune_random <- function(data, formula, method,
                           param_space,
                           n_iter = 10,
                           folds = 5,
                           metric = NULL,
                           maximize = NULL,
                           verbose = TRUE,
                           seed = NULL, ...) {
  formula <- tl_as_formula(formula)

  # Seed this call without rewriting the caller's random stream
  tl_local_seed(seed)

  # Input validation
  if (!is.list(param_space)) {
    stop(
      "param_space must be a named list",
      call. = FALSE
    )
  }

  tl_check_param_names(param_space, "param_space")
  tl_check_param_space(param_space)
  tl_check_tuning_inputs(data, method, folds)
  # n_iter = 0 searched nothing, then reported that every parameter set
  # had failed in every fold
  if (!is.numeric(n_iter) || length(n_iter) != 1L || !is.finite(n_iter) ||
        n_iter != round(n_iter) || n_iter < 1) {
    stop(
      "'n_iter' must be a single whole number of at least 1; got ",
      tl_describe_value(n_iter), ".",
      call. = FALSE
    )
  }
  dots <- list(...)
  tl_check_per_row_args(
    c(names2(dots), names(param_space)), "tl_tune_random()"
  )

  # Determine if classification or regression as tl_model() does, for the
  # reasons given in tl_tune_grid()
  is_classification <- tl_tuning_task(formula, data, method)

  # Default metric based on problem type
  if (is.null(metric)) {
    metric <- if (is_classification) "accuracy" else "rmse"
  }
  tl_check_tuning_metric(metric, is_classification)

  # See tl_tune_grid(): the direction follows the metric, not whether one
  # was supplied
  if (is.null(maximize)) {
    maximize <- tl_metric_maximize(metric)
  }
  tl_check_maximize(maximize)

  if (verbose) {
    message(
      "Random search for ", method,
      " model with ", n_iter, " iterations"
    )
    message(
      "Cross-validation with ", folds, " folds"
    )
    message(
      "Optimizing for ", metric, ", ",
      ifelse(maximize, "maximizing", "minimizing")
    )
  }

  # Create cross-validation splits
  cv_splits <- tl_resample_folds(data, folds)

  # Initialize results storage
  tuning_results <- list()

  # Shared by every fit of this search, the final one included, so the
  # response-conversion warning is given once
  conversion <- tl_warn_once()

  # Generate random parameter combinations
  param_combinations <- lapply(seq_len(n_iter), function(i) {
    params <- lapply(names(param_space), function(param_name) {
      tl_draw_param(param_space[[param_name]], param_name)
    })
    names(params) <- names(param_space)
    params
  })

  # Unlike the grid, duplicates are kept: n_iter rows were asked for
  param_combinations <- tl_tune_cap_mtry(
    param_combinations, method, formula, data
  )

  # Loop through parameter combinations
  for (i in seq_len(n_iter)) {
    params <- param_combinations[[i]]

    if (verbose) {
      message(
        "Iteration ", i, " of ", n_iter, ": ",
        tl_tune_format_params(params)
      )
    }

    tuning_results[[i]] <- tl_tune_score_set(
      params, cv_splits, formula, method, metric, dots, conversion
    )

    if (verbose) {
      message(
        "  Mean ", metric, ": ",
        round(tuning_results[[i]]$mean_metric, 4)
      )
    }
  }

  # Convert results to data frame
  results_df <- tl_tune_results_frame(
    tuning_results, param_combinations, names(param_space),
    iteration = TRUE
  )

  # Find best parameter set among those scored on every fold
  best_idx <- tl_tune_select_best(
    results_df, maximize, folds,
    vapply(param_combinations, tl_tune_format_params, character(1)),
    n_failed = vapply(tuning_results, function(x) x$n_fit_failed,
                      integer(1)),
    metric = metric
  )

  # See tl_tune_grid(): taken from the combinations, not the results frame
  best_params <- param_combinations[[best_idx]]

  if (verbose) {
    message(
      "Best parameters found: ",
      tl_tune_format_params(best_params)
    )
    message(
      "Best ", metric, ": ",
      round(results_df$mean_metric[best_idx], 4)
    )
  }

  # Fit final model with best parameters
  final_model_args <- c(
    list(
      data = data,
      formula = formula,
      method = method
    ),
    best_params,
    dots
  )

  final_model <- withCallingHandlers(
    do.call(tl_model, final_model_args),
    tidylearn_response_conversion = conversion
  )

  # Add tuning results to model
  attr(final_model, "tuning_results") <- list(
    param_space = param_space,
    results = results_df,
    best_params = best_params,
    best_metric = results_df$mean_metric[best_idx],
    metric = metric,
    maximize = maximize
  )

  final_model
}

#' Plot hyperparameter tuning results
#'
#' @param model A tidylearn model object with tuning
#'   results
#' @param top_n Number of top parameter sets to
#'   highlight
#' @param param1 First parameter to plot (for 2D grid
#'   or scatter plots)
#' @param param2 Second parameter to plot (for 2D grid
#'   or scatter plots)
#' @param plot_type Type of plot: "scatter", "grid",
#'   "parallel", "importance"
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @details A parameter whose candidates are not single values, such as
#'   \code{hidden_layers = list(c(10), c(20, 10))} or a \code{parms} list,
#'   is drawn as a categorical one, each value labelled as the verbose
#'   messages print it. The importance of a numeric parameter is the
#'   absolute correlation of its values with the score; that of a
#'   categorical one is eta squared from a one-way ANOVA of the score,
#'   and 0 when the sets that were scored all share one value.
#' @examples
#' \donttest{
#' model <- tl_tune_grid(iris, Species ~ ., method = "tree",
#'   param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
#'   folds = 2, verbose = FALSE)
#' tl_plot_tuning_results(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_point geom_tile
#'   scale_fill_gradient2 labs theme_minimal
#' @export
tl_plot_tuning_results <- function(model,
                                   top_n = 5,
                                   param1 = NULL,
                                   param2 = NULL,
                                   plot_type =
                                     "scatter") {
  # Check if model has tuning results
  tuning_results <- attr(model, "tuning_results")
  if (is.null(tuning_results)) {
    stop(
      "Model does not have tuning results",
      call. = FALSE
    )
  }

  # Extract results data frame
  results_df <- tuning_results$results

  # Get parameter names
  param_names <- setdiff(
    names(results_df),
    c("iteration", "mean_metric", "n_folds_ok")
  )

  # A candidate that is not a single value -- a hidden_layers vector, an
  # rpart parms list -- is held in a list column, which no geom, scale or
  # order() can take. Each cell is drawn under its printed label instead,
  # in the order the search met them.
  for (param in param_names) {
    if (is.list(results_df[[param]])) {
      labels <- vapply(results_df[[param]], tl_tune_format_value,
                       character(1))
      results_df[[param]] <- factor(labels, levels = unique(labels))
    }
  }
  # Identifies each set's line in the parallel plot. The rank did that
  # before, and sets whose scores tied shared a rank and so one line.
  results_df$.set <- seq_len(nrow(results_df))

  # Default parameters for plotting if not specified
  if (is.null(param1) && length(param_names) > 0) {
    param1 <- param_names[1]
  }

  if (is.null(param2) && length(param_names) > 1) {
    param2 <- param_names[2]
  }

  # Create ranking column
  results_df$rank <- rank(
    if (tuning_results$maximize) {
      -results_df$mean_metric
    } else {
      results_df$mean_metric
    }
  )
  results_df$is_top <- results_df$rank <= top_n

  # Create plot based on plot_type
  if (plot_type == "scatter" &&
        !is.null(param1) && !is.null(param2)) {
    # Scatter plot of two parameters
    p <- ggplot2::ggplot(
      results_df,
      ggplot2::aes(
        x = .data[[param1]],
        y = .data[[param2]],
        color = .data$mean_metric,
        size = .data$is_top,
        shape = .data$is_top
      )
    ) +
      ggplot2::geom_point(alpha = 0.7) +
      ggplot2::scale_color_gradient2(
        low = "blue",
        high = "red",
        mid = "white",
        midpoint = median(
          results_df$mean_metric, na.rm = TRUE
        )
      ) +
      ggplot2::scale_size_manual(
        values = c(3, 5)
      ) +
      ggplot2::scale_shape_manual(
        values = c(16, 18)
      ) +
      ggplot2::labs(
        title = "Hyperparameter Tuning Results",
        subtitle = paste(
          "Optimizing", tuning_results$metric,
          "- top", top_n, "results highlighted"
        ),
        x = param1,
        y = param2,
        color = tuning_results$metric,
        size = "Top Result",
        shape = "Top Result"
      ) +
      ggplot2::theme_minimal()

  } else if (plot_type == "grid" &&
               !is.null(param1) && !is.null(param2)) {
    # Heat map for grid search
    n_unique_p1 <- length(
      unique(results_df[[param1]])
    )
    n_unique_p2 <- length(
      unique(results_df[[param2]])
    )
    if (n_unique_p1 <= 20 && n_unique_p2 <= 20) {
      p <- ggplot2::ggplot(
        results_df,
        ggplot2::aes(
          x = .data[[param1]],
          y = .data[[param2]],
          fill = .data$mean_metric
        )
      ) +
        ggplot2::geom_tile() +
        ggplot2::geom_text(
          ggplot2::aes(
            label = round(.data$mean_metric, 3)
          ),
          color = "black",
          size = 3
        ) +
        ggplot2::scale_fill_gradient2(
          low = "blue",
          high = "red",
          mid = "white",
          midpoint = median(
            results_df$mean_metric, na.rm = TRUE
          )
        ) +
        ggplot2::labs(
          title = "Hyperparameter Tuning Results Grid",
          subtitle = paste(
            "Optimizing", tuning_results$metric
          ),
          x = param1,
          y = param2,
          fill = tuning_results$metric
        ) +
        ggplot2::theme_minimal()
    } else {
      warning(
        "Parameters have too many unique values ",
        "for a grid plot. ",
        "Using scatter plot instead."
      )
      # The result has to be assigned: the function returns `p`, so
      # discarding this call left `p` undefined on the fallback path
      p <- tl_plot_tuning_results(
        model, top_n, param1, param2, "scatter"
      )
    }

  } else if (plot_type == "parallel") {
    # Parallel coordinates plot
    results_norm <- results_df

    for (param in param_names) {
      if (is.numeric(results_df[[param]])) {
        param_min <- min(
          results_df[[param]], na.rm = TRUE
        )
        param_max <- max(
          results_df[[param]], na.rm = TRUE
        )
        norm_col <- paste0(param, "_norm")
        if (param_max > param_min) {
          results_norm[[norm_col]] <-
            (results_df[[param]] - param_min) /
            (param_max - param_min)
        } else {
          results_norm[[norm_col]] <- 0.5
        }
      } else {
        # For categorical parameters
        norm_col <- paste0(param, "_norm")
        n_levels <- length(
          unique(results_df[[param]])
        )
        results_norm[[norm_col]] <-
          as.numeric(
            factor(results_df[[param]])
          ) / n_levels
      }
    }

    # Prepare data for parallel coordinates
    param_norm_names <- paste0(
      param_names, "_norm"
    )

    # Convert to long format
    plot_data <- tidyr::pivot_longer(
      results_norm,
      cols = all_of(param_norm_names),
      names_to = "parameter",
      values_to = "value"
    )

    # Remove _norm suffix for plotting
    plot_data$parameter <- gsub(
      "_norm$", "", plot_data$parameter
    )

    # Create parallel coordinates plot
    p <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(
        x = .data$parameter,
        y = .data$value,
        group = .data$.set,
        color = .data$mean_metric,
        # `size` on a line is deprecated since ggplot2 3.4.0 and warns
        # the caller to file a bug against tidylearn
        linewidth = .data$is_top,
        alpha = .data$is_top
      )
    ) +
      ggplot2::geom_line() +
      ggplot2::scale_color_gradient2(
        low = "blue",
        high = "red",
        mid = "white",
        midpoint = median(
          results_df$mean_metric, na.rm = TRUE
        )
      ) +
      ggplot2::scale_linewidth_manual(
        values = c(0.5, 1.5)
      ) +
      ggplot2::scale_alpha_manual(
        values = c(0.3, 1)
      ) +
      ggplot2::labs(
        title = paste(
          "Parallel Coordinates Plot of",
          "Hyperparameter Tuning Results"
        ),
        subtitle = paste(
          "Optimizing", tuning_results$metric,
          "- top", top_n, "results highlighted"
        ),
        x = "Parameter",
        y = "Normalized Value",
        color = tuning_results$metric,
        linewidth = "Top Result",
        alpha = "Top Result"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        panel.grid.major.x = ggplot2::element_line(
          color = "gray90"
        ),
        panel.grid.minor = ggplot2::element_blank(),
        axis.text.x = ggplot2::element_text(
          angle = 45, hjust = 1
        )
      )

  } else if (plot_type == "importance") {
    # Parameter importance plot
    # A set that failed every fold has no score, so it says nothing about
    # its parameters' effect
    scored <- !is.na(results_df$mean_metric)

    param_importance <- lapply(
      param_names,
      function(param) {
        if (is.numeric(results_df[[param]])) {
          # For numeric parameters, use correlation. A parameter that took
          # one value, or a metric that did not move across the grid, gives
          # cor() a zero-variance input: it warns and returns NA, and the
          # bar silently disappears from the plot. Zero variance means the
          # parameter explained none of the score, so say that instead.
          # Spread is judged on the pairs cor() uses.
          values <- results_df[[param]]
          pairs <- scored & !is.na(values)
          has_spread <- function(x) {
            length(x) > 1L && stats::sd(x) > 0
          }
          cor_val <- if (has_spread(values[pairs]) &&
                           has_spread(results_df$mean_metric[pairs])) {
            cor(values[pairs], results_df$mean_metric[pairs])
          } else {
            0
          }
          data.frame(
            parameter = param,
            importance = abs(cor_val),
            correlation = cor_val
          )
        } else {
          # For categorical parameters, use ANOVA
          results_df[[param]] <- as.factor(
            results_df[[param]]
          )

          # aov() needs two levels among the scored sets, and stopped on a
          # parameter held at one value -- or one whose other value failed
          # every fold -- with "contrasts can be applied only to factors
          # with 2 or more levels". One value explains none of the score.
          observed <- unique(results_df[[param]][scored])
          if (length(observed[!is.na(observed)]) < 2L) {
            return(data.frame(
              parameter = param, importance = 0, correlation = NA
            ))
          }

          # Run ANOVA. The formula is built with stats::reformulate --
          # the .data pronoun is a tidy-eval construct and is not
          # understood by aov(), which evaluates in a plain data frame.
          anova_result <- summary(stats::aov(
            stats::reformulate(param, response = "mean_metric"),
            data = results_df
          ))

          # Calculate eta squared
          ss_total <- sum(
            anova_result[[1]]$"Sum Sq"
          )
          ss_param <- anova_result[[1]]$"Sum Sq"[1]
          # A constant metric makes ss_total zero and eta squared NaN
          eta_squared <- if (isTRUE(ss_total > 0)) ss_param / ss_total else 0

          data.frame(
            parameter = param,
            importance = eta_squared,
            correlation = NA
          )
        }
      }
    )

    # Combine results
    importance_df <- do.call(
      rbind, param_importance
    )

    # Sort by importance
    importance_df <- importance_df[
      order(
        importance_df$importance,
        decreasing = TRUE
      ),
    ]

    # Create bar plot
    p <- ggplot2::ggplot(
      importance_df,
      ggplot2::aes(
        x = reorder(
          .data$parameter,
          .data$importance
        ),
        y = .data$importance,
        fill = .data$correlation
      )
    ) +
      ggplot2::geom_col() +
      ggplot2::scale_fill_gradient2(
        low = "blue",
        high = "red",
        mid = "white",
        midpoint = 0,
        na.value = "gray"
      ) +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = "Hyperparameter Importance",
        subtitle = paste(
          "Correlation with",
          tuning_results$metric
        ),
        x = "Parameter",
        y = "Importance (|Correlation| or Eta Sq)",
        fill = "Correlation\n(numeric only)"
      ) +
      ggplot2::theme_minimal()
  } else if (plot_type %in% c("scatter", "grid")) {
    # One message covered both causes, so the default plot of a
    # one-parameter search said it must be one of "scatter", ... and had
    # got "scatter"
    stop(
      "plot_type = \"", plot_type, "\" needs two tuned parameters, and ",
      "this search tuned ", length(param_names), ". Use ",
      "plot_type = \"parallel\" or \"importance\".",
      call. = FALSE
    )
  } else {
    stop(
      "plot_type must be one of \"scatter\", \"grid\", ",
      "\"parallel\" or \"importance\"; got \"", plot_type, "\".",
      call. = FALSE
    )
  }

  p
}

#' Create pre-defined parameter grids for common models
#'
#' @param method Model method ("tree", "forest",
#'   "boost", "svm", "xgboost", etc.)
#' @param size Grid size: "small", "medium", "large"
#' @param is_classification Whether the grid is for a classification task
#'   (the default) or a regression one. For regression, an \code{"svm"}
#'   grid also tunes \code{epsilon}, the width of the band within which
#'   e1071's regression SVM ignores errors, and the large \code{"forest"}
#'   grid centres \code{nodesize} on randomForest's regression default of
#'   5 rather than its classification default of 1. The other grids are
#'   the same for both tasks.
#' @return A named list of parameter values suitable for passing to
#'   \code{\link{tl_tune_grid}} or \code{\link{tl_tune_random}}. Each
#'   element is a numeric or character vector of candidate values for
#'   that hyperparameter, or for \code{"deep"}'s \code{hidden_layers} a list
#'   of layer-size vectors. The grid is built without the data, so a
#'   \code{"forest"} \code{mtry} can exceed the number of predictors; the
#'   tuners cap it. \code{"polynomial"} tunes \code{degree}. The
#'   \code{"xgboost"} grids draw on the values \code{\link{tl_tune_xgboost}}
#'   searches by default, and add \code{nrounds}, which that function
#'   chooses by early stopping and \code{\link{tl_tune_grid}} has to tune.
#'   \code{"linear"} and \code{"logistic"} have no tuneable
#'   hyperparameter and return an empty list with a warning, as does an
#'   unknown method.
#' @examples
#' \donttest{
#' grid <- tl_default_param_grid("tree", size = "small")
#' grid <- tl_default_param_grid("forest", size = "medium")
#' grid <- tl_default_param_grid("svm", is_classification = FALSE)
#' }
#' @export
tl_default_param_grid <- function(method,
                                  size = "medium",
                                  is_classification =
                                    TRUE) {
  # Input validation
  if (!is.character(method) || length(method) != 1L || is.na(method)) {
    stop(
      "'method' must be a single method name, such as \"tree\"; got ",
      tl_describe_value(method), ".",
      call. = FALSE
    )
  }
  size <- match.arg(size, c("small", "medium", "large"))
  if (!is.logical(is_classification) || length(is_classification) != 1L ||
        is.na(is_classification)) {
    stop(
      "'is_classification' must be TRUE or FALSE; got ",
      tl_describe_value(is_classification), ".",
      call. = FALSE
    )
  }

  # Define default grids for different methods
  if (method == "tree") {
    if (size == "small") {
      list(
        cp = c(0.01, 0.1),
        minsplit = c(10, 20)
      )
    } else if (size == "medium") {
      list(
        cp = c(0.001, 0.01, 0.1),
        minsplit = c(5, 10, 20, 30),
        maxdepth = c(10, 20, 30)
      )
    } else { # large
      list(
        cp = c(0.0001, 0.001, 0.01, 0.1),
        minsplit = c(5, 10, 20, 30, 50),
        maxdepth = c(5, 10, 15, 20, 30)
      )
    }
  } else if (method == "forest") {
    if (size == "small") {
      list(
        mtry = c(2, 3),
        ntree = c(100, 500)
      )
    } else if (size == "medium") {
      list(
        mtry = c(2, 3, 4, 5),
        ntree = c(100, 300, 500)
      )
    } else { # large
      # No sampsize: randomForest reads it as a number of rows, which a
      # grid built without the data cannot choose. mtry values above the
      # predictor count are capped by the tuners (see tl_tune_cap_mtry()),
      # because no fixed ceiling suits every data set. randomForest's
      # default nodesize is 1 for classification and 5 for regression, and
      # the candidates start from the task's own.
      list(
        mtry = c(1, 2, 3, 4, 5, 6),
        ntree = c(100, 300, 500, 1000),
        nodesize = if (is_classification) c(1, 3, 5) else c(3, 5, 10)
      )
    }
  } else if (method == "boost") {
    if (size == "small") {
      list(
        n.trees = c(50, 100),
        interaction.depth = c(2, 3),
        shrinkage = c(0.01, 0.1)
      )
    } else if (size == "medium") {
      list(
        n.trees = c(50, 100, 200),
        interaction.depth = c(1, 2, 3, 4),
        shrinkage = c(0.001, 0.01, 0.1),
        n.minobsinnode = c(5, 10)
      )
    } else { # large
      list(
        n.trees = c(50, 100, 200, 500, 1000),
        interaction.depth = c(1, 2, 3, 4, 5),
        shrinkage = c(0.001, 0.01, 0.05, 0.1),
        n.minobsinnode = c(1, 5, 10, 20),
        bag.fraction = c(0.5, 0.632, 0.8, 1.0)
      )
    }
  } else if (method == "svm") {
    grid <- if (size == "small") {
      list(
        kernel = c("linear", "radial"),
        cost = c(0.1, 1, 10)
      )
    } else if (size == "medium") {
      list(
        kernel = c("linear", "polynomial", "radial"),
        cost = c(0.01, 0.1, 1, 10, 100),
        gamma = c(0.01, 0.1, 1)
      )
    } else { # large
      list(
        kernel = c(
          "linear", "polynomial",
          "radial", "sigmoid"
        ),
        cost = c(
          0.001, 0.01, 0.1, 1, 10, 100, 1000
        ),
        gamma = c(0.001, 0.01, 0.1, 1, 10),
        degree = c(2, 3, 4),
        coef0 = c(0, 0.1, 1)
      )
    }
    # A regression SVM ignores errors within epsilon of the fit, 0.1 by
    # default on the response e1071 has scaled. Classification has no such
    # band, so the parameter would only repeat every fit.
    if (!is_classification) {
      grid$epsilon <- if (size == "small") c(0.1, 0.5) else c(0.01, 0.1, 0.5)
    }
    grid
  } else if (method == "xgboost") {
    # The values tl_tune_xgboost() searches by default. It picks the number
    # of rounds by early stopping, which tl_tune_grid() cannot, so nrounds
    # is tuned here as well.
    if (size == "small") {
      list(
        nrounds = c(50, 100),
        max_depth = c(3, 6),
        eta = c(0.1, 0.3)
      )
    } else if (size == "medium") {
      list(
        nrounds = c(50, 100, 200),
        max_depth = c(3, 6, 9),
        eta = c(0.01, 0.1, 0.3),
        subsample = c(0.7, 1.0)
      )
    } else { # large
      list(
        nrounds = c(100, 200, 500),
        max_depth = c(3, 6, 9),
        eta = c(0.01, 0.1, 0.3),
        subsample = c(0.7, 1.0),
        colsample_bytree = c(0.7, 1.0),
        min_child_weight = c(1, 3, 5),
        gamma = c(0, 0.1, 0.2)
      )
    }
  } else if (method == "nn") {
    if (size == "small") {
      list(
        size = c(3, 5),
        decay = c(0, 0.1)
      )
    } else if (size == "medium") {
      list(
        size = c(2, 5, 10),
        decay = c(0, 0.01, 0.1)
      )
    } else { # large
      list(
        size = c(2, 5, 10, 20, 50),
        decay = c(0, 0.001, 0.01, 0.1, 0.5),
        maxit = c(100, 200, 500)
      )
    }
  } else if (method %in%
               c("ridge", "lasso", "elastic_net")) {
    if (method == "elastic_net") {
      # Alpha values for elastic net
      alpha_values <- if (size == "small") {
        c(0.25, 0.5, 0.75)
      } else if (size == "medium") {
        c(0.1, 0.25, 0.5, 0.75, 0.9)
      } else { # large
        seq(0.1, 0.9, by = 0.1)
      }
    } else {
      # For ridge and lasso, alpha is fixed
      alpha_values <- if (method == "ridge") {
        0
      } else {
        1
      }
    }

    # Lambda values (regularization strength)
    if (size == "small") {
      lambda_values <- c(0.001, 0.01, 0.1, 1)
    } else if (size == "medium") {
      lambda_values <- c(
        0.0001, 0.001, 0.01, 0.1, 1, 10
      )
    } else { # large
      lambda_values <- c(
        0.0001, 0.0005, 0.001, 0.005,
        0.01, 0.05, 0.1, 0.5, 1, 5, 10
      )
    }

    # Combine parameters
    if (method == "elastic_net") {
      list(
        alpha = alpha_values,
        lambda = lambda_values
      )
    } else {
      list(
        lambda = lambda_values
      )
    }
  } else if (method == "logistic") {
    # "logistic" is an unpenalised glm(). The ridge lambda grid returned
    # here before is not a glm() argument, so every fit in a search over it
    # failed. The penalised classifiers are separate methods with grids of
    # their own.
    warning(
      "Method \"logistic\" is fitted with glm(), and glm() has no ",
      "hyperparameter to tune. Returning an empty parameter grid. For a ",
      "regularised logistic model, use method = \"ridge\", \"lasso\" or ",
      "\"elastic_net\" with their default grids.",
      call. = FALSE
    )
    list()
  } else if (method == "linear") {
    # A supported method, so "Unknown method" was the wrong warning; lm()
    # has nothing to tune
    warning(
      "Method \"linear\" is fitted with lm(), which has no hyperparameter ",
      "to tune. Returning an empty parameter grid. For a penalised linear ",
      "model, use method = \"ridge\", \"lasso\" or \"elastic_net\".",
      call. = FALSE
    )
    list()
  } else if (method == "polynomial") {
    # degree is the one setting tl_fit_polynomial() takes. Degree 1 is the
    # linear model, so the grid starts at 2.
    list(degree = switch(size, small = 2:3, medium = 2:4, 2:5))
  } else if (method == "deep") {
    if (size == "small") {
      list(
        hidden_layers = list(c(10), c(10, 5)),
        activation = c("relu", "tanh"),
        dropout = c(0, 0.2)
      )
    } else if (size == "medium") {
      list(
        hidden_layers = list(
          c(10), c(20, 10), c(20, 10, 5)
        ),
        activation = c("relu", "tanh", "sigmoid"),
        dropout = c(0, 0.2, 0.5),
        batch_size = c(16, 32, 64)
      )
    } else { # large
      list(
        hidden_layers = list(
          c(10), c(20), c(50),
          c(20, 10), c(50, 25), c(50, 25, 10)
        ),
        activation = c(
          "relu", "tanh", "sigmoid", "elu"
        ),
        dropout = c(0, 0.1, 0.2, 0.3, 0.5),
        batch_size = c(16, 32, 64, 128),
        learning_rate = c(
          0.0001, 0.001, 0.01, 0.1
        )
      )
    }
  } else {
    # Default empty grid for unknown method
    warning(
      "Unknown method: ", method,
      ". Returning empty parameter grid.",
      call. = FALSE
    )
    list()
  }
}
