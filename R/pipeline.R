#' @title Model Pipeline Functions for tidylearn
#' @name tidylearn-pipeline
#' @description Functions for creating end-to-end model pipelines
#' @importFrom stats formula
#' @importFrom dplyr filter select mutate
NULL

#' Whether a higher value of a metric is better
#'
#' One list for every place that ranks models, covering exactly the metrics
#' \code{tl_known_metrics()} names. The pipeline, its comparison plot and
#' the AutoML leaderboard each kept their own list, and the pipeline's left
#' out sensitivity, specificity and pr_auc, so a stump scoring 0.40 beat a
#' tree scoring 0.90.
#'
#' @param metrics Character vector of metric names
#' @return A logical vector: \code{TRUE} where higher is better,
#'   \code{FALSE} where lower is, \code{NA} for a name tidylearn does not
#'   compute
#' @keywords internal
#' @noRd
tl_metric_higher_better <- function(metrics) {
  higher <- c("accuracy", "precision", "recall", "sensitivity",
              "specificity", "f1", "auc", "pr_auc", "rsq")
  lower <- c("rmse", "mse", "mae", "mape")
  ifelse(metrics %in% higher, TRUE, ifelse(metrics %in% lower, FALSE, NA))
}

#' Fill in and validate a pipeline evaluation specification
#'
#' @param evaluation A named list of evaluation settings, or NULL.
#' @param is_classification Whether the response is categorical.
#' @return The list, with every recognised setting present.
#' @keywords internal
#' @noRd
merge_evaluation_spec <- function(evaluation, is_classification) {
  defaults <- if (is_classification) {
    list(
      metrics = c("accuracy", "precision", "recall", "f1", "auc"),
      validation = "cv",
      cv_folds = 5,
      train_prop = 0.7,
      best_metric = "f1"
    )
  } else {
    list(
      metrics = c("rmse", "mae", "rsq", "mape"),
      validation = "cv",
      cv_folds = 5,
      train_prop = 0.7,
      best_metric = "rmse"
    )
  }

  if (is.null(evaluation)) {
    return(defaults)
  }

  if (!is.list(evaluation) ||
        is.null(names(evaluation)) ||
        any(names(evaluation) == "")) {
    stop(
      "'evaluation' must be a named list: ",
      paste(names(defaults), collapse = ", "), ".",
      call. = FALSE
    )
  }

  unknown <- setdiff(names(evaluation), names(defaults))
  if (length(unknown) > 0) {
    stop(
      "Unknown evaluation setting(s): ",
      paste(unknown, collapse = ", "),
      ". Available settings: ", paste(names(defaults), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  evaluation <- utils::modifyList(defaults, evaluation)

  # modifyList() drops an element set to NULL, so a caller who writes
  # validation = NULL leaves nothing here and `%in%` returns logical(0),
  # which `if` refuses with "argument is of length zero"
  if (!is.character(evaluation$validation) ||
        length(evaluation$validation) != 1L ||
        !evaluation$validation %in% c("cv", "split")) {
    stop(
      "evaluation$validation must be \"cv\" or \"split\"; got ",
      tl_describe_value(evaluation$validation), ".",
      call. = FALSE
    )
  }

  # An empty metric set renders the message below as a bare full stop,
  # and leaves nothing for best_metric to name
  if (length(evaluation$metrics) == 0) {
    stop(
      "evaluation$metrics is empty. Name at least one of: ",
      paste(defaults$metrics, collapse = ", "), ".",
      call. = FALSE
    )
  }

  # An unrecognised metric is not computed, so every score comes back NA
  # and the run warns about the symptom -- "all values NA" -- rather than
  # the cause. tl_tune_grid() names a missing metric; match it.
  known <- tl_known_metrics(is_classification)
  unknown_metric <- setdiff(evaluation$metrics, known)
  if (length(unknown_metric) > 0) {
    stop(
      "Unknown ", if (is_classification) "classification" else "regression",
      " metric(s): ", paste0("\"", unknown_metric, "\"", collapse = ", "),
      ". Available: ", paste(known, collapse = ", "), ".",
      call. = FALSE
    )
  }

  if (!is.character(evaluation$best_metric) ||
        length(evaluation$best_metric) != 1L ||
        !evaluation$best_metric %in% evaluation$metrics) {
    stop(
      "evaluation$best_metric (", tl_describe_value(evaluation$best_metric),
      ") must be one of evaluation$metrics: ",
      paste(evaluation$metrics, collapse = ", "), ".",
      call. = FALSE
    )
  }

  # Unchecked, these surfaced as rsample errors naming `v`, which is not
  # an argument of anything the caller wrote
  folds <- evaluation$cv_folds
  if (!is.numeric(folds) || length(folds) != 1L || is.na(folds) ||
        folds != round(folds) || folds < 2) {
    stop(
      "evaluation$cv_folds must be a single whole number of at least 2; ",
      "got ", paste(format(folds), collapse = ", "), ".",
      call. = FALSE
    )
  }

  # Left unchecked these reached rsample and base R as "result would be
  # too long a vector" (0), a ROCR complaint about class counts (1), and
  # "cannot take a sample larger than the population" (> 1)
  prop <- evaluation$train_prop
  if (!is.numeric(prop) || length(prop) != 1L || is.na(prop) ||
        prop <= 0 || prop >= 1) {
    stop(
      "evaluation$train_prop must be a single number strictly between 0 ",
      "and 1; got ", paste(format(prop), collapse = ", "),
      ". It is the share of rows used for training. Whether that leaves ",
      "both sides of the split non-empty also depends on the row count, ",
      "which tl_run_pipeline() checks.",
      call. = FALSE
    )
  }

  evaluation
}

#' Fill in and validate a pipeline preprocessing specification
#'
#' @param preprocessing A named list of preprocessing switches, or NULL.
#' @return The list, with every recognised step present.
#' @keywords internal
#' @noRd
merge_preprocessing_spec <- function(preprocessing) {
  defaults <- list(
    impute_missing = TRUE,
    standardize = TRUE,
    dummy_encode = TRUE
  )

  if (is.null(preprocessing)) {
    return(defaults)
  }

  if (!is.list(preprocessing) ||
        is.null(names(preprocessing)) ||
        any(names(preprocessing) == "")) {
    stop(
      "'preprocessing' must be a named list of steps: ",
      paste(names(defaults), collapse = ", "), ".",
      call. = FALSE
    )
  }

  # An unrecognised name would otherwise be accepted and silently do
  # nothing, which reads as a step that ran
  unknown <- setdiff(names(preprocessing), names(defaults))
  if (length(unknown) > 0) {
    stop(
      "Unknown preprocessing step(s): ",
      paste(unknown, collapse = ", "),
      ". Available steps: ", paste(names(defaults), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  not_flag <- names(preprocessing)[
    !vapply(preprocessing, function(x) {
      is.logical(x) && length(x) == 1L && !is.na(x)
    }, logical(1))
  ]
  if (length(not_flag) > 0) {
    stop(
      "Preprocessing step(s) must be TRUE or FALSE: ",
      paste(not_flag, collapse = ", "), ".",
      call. = FALSE
    )
  }

  # The pipeline has no encoding step of its own: model.matrix() encodes
  # factors for lm, glm and glmnet, and the tree methods split on them
  # directly. FALSE therefore ran exactly the models TRUE did.
  if (isFALSE(preprocessing$dummy_encode)) {
    stop(
      "preprocessing$dummy_encode = FALSE cannot be honoured. Categorical ",
      "predictors are encoded by each model's own fitting function -- ",
      "model.matrix() for the linear, logistic and glmnet methods, while ",
      "trees and forests split on factors directly -- so there is no ",
      "encoding step to switch off. Leave dummy_encode out, or remove the ",
      "categorical columns from the formula.",
      call. = FALSE
    )
  }

  utils::modifyList(defaults, preprocessing)
}

#' Create a modeling pipeline
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the model
#' @param preprocessing A named list of preprocessing switches, each
#'   \code{TRUE} or \code{FALSE}: \code{impute_missing} (default
#'   \code{TRUE}) replaces missing predictor values with the training
#'   median or mode; \code{standardize} (default \code{TRUE}) centres and
#'   scales numeric predictors, leaving alone any column the formula uses
#'   inside a function call such as \code{log()}, \code{poly()} or
#'   \code{offset()}, so those terms are computed on the column's own
#'   scale; \code{dummy_encode} (default \code{TRUE}) only records that
#'   categorical predictors are encoded by each model's fitting function,
#'   and cannot be set to \code{FALSE}.
#' @param models A list of models to train
#' @param evaluation A list of evaluation criteria
#' @param ... Not used. Anything passed here is an error, so a misspelt
#'   argument such as \code{evalution} is reported rather than ignored.
#' @return A \code{tidylearn_pipeline} object (S3 list) with components
#'   \code{$formula}, \code{$data}, \code{$preprocessing},
#'   \code{$models}, \code{$evaluation}, and \code{$results}
#'   (initially \code{NULL}; populated after \code{\link{tl_run_pipeline}}).
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(tree = list(method = "tree")))
#' print(pipe)
#' }
#' @export
tl_pipeline <- function(data, formula,
                        preprocessing = NULL,
                        models = NULL,
                        evaluation = NULL, ...) {
  # A misspelt argument -- evalution = list(cv_folds = 2) -- landed here
  # and was dropped, leaving every setting at its default without a word
  extra <- names2(list(...))
  if (length(extra) > 0) {
    extra[!nzchar(extra)] <- "<unnamed>"
    stop(
      "tl_pipeline() has no argument(s) ", paste(extra, collapse = ", "),
      ". It takes data, formula, preprocessing, models and evaluation.",
      call. = FALSE
    )
  }

  formula <- tl_as_formula(formula)

  # Fill in whichever preprocessing steps the caller left unnamed. A
  # partial list used to reach `if (preprocessing$standardize)` as NULL
  # and fail with "argument is of length zero" inside tl_run_pipeline(),
  # long after the mistake was made.
  preprocessing <- merge_preprocessing_spec(preprocessing)

  # A response that is not a column of `data` used to reach the default
  # models and the evaluation spec as NULL, which reads as regression.
  # The pipeline then failed inside rpart with "object 'Speces' not
  # found", naming a typo but not where it was made.
  response_var <- all.vars(formula)[1]
  if (!response_var %in% names(data)) {
    stop(
      "The formula's response, '", response_var,
      "', is not a column of `data`. Available: ",
      paste(names(data), collapse = ", "), ".",
      call. = FALSE
    )
  }
  y <- data[[response_var]]
  is_classification <- is.factor(y) || is.character(y)

  # Create default models if not provided
  if (is.null(models)) {
    if (is_classification) {
      # Logistic is binary-only and errors on a three-level response, so
      # offering it as a default candidate would fail the whole pipeline
      # rather than the one model. Same guard as tl_auto_ml().
      models <- list(
        tree = list(method = "tree"),
        forest = list(method = "forest", ntree = 500)
      )
      # Count classes that occur, not levels that are declared -- a
      # subset of a larger frame keeps the levels it no longer uses.
      if (nlevels(tl_normalise_response(y)) == 2) {
        models <- c(list(logistic = list(method = "logistic")), models)
      }
    } else {
      models <- list(
        linear = list(method = "linear"),
        lasso = list(method = "lasso"),
        forest = list(method = "forest", ntree = 500)
      )
    }
  }

  # Fill in whichever evaluation settings the caller left unnamed, for
  # the same reason as the preprocessing spec above.
  #
  # The task has to be settled the way tl_model_supervised() settles it,
  # logistic override included: logistic on a 0/1 integer response is a
  # classification fit whatever the column is stored as, and each fold
  # model reports it as one. Deciding from the column alone called that
  # pipeline regression and refused the accuracy and auc it goes on to
  # compute.
  #
  # Every other supervised method reads the task off the column as it
  # stands, so a numeric response that puts logistic next to any of them
  # gives one run two tasks. A leaderboard carries one set of metrics,
  # whichever way it is chosen the other candidates score NA, and they
  # then drop out of the comparison without a word -- so the mixture is
  # refused here rather than half-scored later.
  spec_methods <- tl_spec_methods(models)
  named <- !is.na(spec_methods)
  forces_factor <- named & spec_methods == "logistic"
  others <- named & !forces_factor
  if (!is_classification && any(forces_factor) && any(others)) {
    stop(
      "'", response_var, "' is numeric, and `models` puts logistic ",
      "regression alongside ",
      paste(unique(spec_methods[others]), collapse = ", "),
      ". Logistic coerces the response to a factor and is scored as ",
      "classification, while the rest take the column as it stands and ",
      "are scored as regression. One leaderboard cannot hold both, so ",
      "whichever metrics were chosen the other models would score NA. ",
      "Run these as two pipelines, or make '", response_var,
      "' a factor and drop the regression methods.",
      call. = FALSE
    )
  }

  evaluation <- merge_evaluation_spec(
    evaluation,
    is_classification = is_classification || any(forces_factor)
  )

  # Create pipeline object
  pipeline <- list(
    formula = formula,
    data = data,
    preprocessing = preprocessing,
    models = models,
    evaluation = evaluation,
    results = NULL
  )

  class(pipeline) <- "tidylearn_pipeline"

  pipeline
}

#' Columns a formula uses inside a function call
#'
#' Standardising happens before the formula is evaluated, so a column the
#' formula transforms would be transformed on the standardised scale:
#' \code{log(hp)} of a below-average car is \code{log()} of a negative
#' number, and \code{offset(0.05 * hp)} adds 0.05 per standard deviation
#' rather than per horsepower. Those columns are left on their own scale.
#'
#' @param formula The model formula
#' @param data The data it is evaluated against, to expand \code{.}
#' @return A character vector of column names, possibly empty
#' @keywords internal
#' @noRd
tl_formula_call_columns <- function(formula, data) {
  variables <- as.list(
    attr(stats::terms(formula, data = data), "variables")
  )[-1]
  in_calls <- variables[!vapply(variables, is.name, logical(1))]
  unique(unlist(lapply(in_calls, all.vars)))
}

#' Learn preprocessing statistics from a training set
#'
#' Split out from \code{tl_run_pipeline()} so that every resampling fold
#' can learn its own statistics. Learning them once on the full dataset
#' and then splitting lets each assessment row influence the centre,
#' scale and median applied to the rows it is scored against, which
#' inflates every reported metric.
#'
#' The response is deliberately excluded from imputation: replacing a
#' missing outcome with the median fabricates both a training target and
#' a piece of evaluation ground truth.
#'
#' @param data The training rows only
#' @param formula The model formula
#' @param preprocessing The pipeline's preprocessing specification
#' @return A list with \code{medians}, \code{modes}, \code{center} and
#'   \code{scale}, each a named list keyed by column
#' @keywords internal
#' @noRd
tl_learn_preprocessing <- function(data, formula, preprocessing) {
  stats_learned <- list(
    medians = list(), modes = list(),
    center = list(), scale = list()
  )

  response_var <- all.vars(formula)[1]

  if (isTRUE(preprocessing$impute_missing)) {
    for (col in setdiff(names(data), response_var)) {
      if (is.numeric(data[[col]])) {
        # Record the median even when this column is complete -- new data
        # may still have gaps here
        stats_learned$medians[[col]] <- median(data[[col]], na.rm = TRUE)
      } else if (is.factor(data[[col]]) || is.character(data[[col]])) {
        tab <- if (is.factor(data[[col]])) {
          table(data[[col]])
        } else {
          table(data[[col]], useNA = "no")
        }
        stats_learned$modes[[col]] <- if (length(tab) > 0) {
          names(tab)[which.max(tab)]
        } else {
          NA_character_
        }
      }
    }
  }

  if (isTRUE(preprocessing$standardize)) {
    numeric_cols <- vapply(data, is.numeric, logical(1))
    numeric_cols[response_var] <- FALSE  # Don't standardize response
    transformed <- intersect(tl_formula_call_columns(formula, data),
                             names(data))
    numeric_cols[transformed] <- FALSE

    for (col in names(data)[numeric_cols]) {
      col_mean <- mean(data[[col]], na.rm = TRUE)
      col_sd <- stats::sd(data[[col]], na.rm = TRUE)

      # A constant column would divide by zero; centre it only
      if (is.na(col_sd) || col_sd == 0) {
        col_sd <- 1
      }

      stats_learned$center[[col]] <- col_mean
      stats_learned$scale[[col]] <- col_sd
    }
  }

  stats_learned
}

#' Apply learned preprocessing statistics to a data frame
#'
#' @param data Rows to transform -- a training fold, an assessment fold,
#'   or unseen data
#' @param preprocessing The pipeline's preprocessing specification
#' @param stats_learned The output of \code{tl_learn_preprocessing()}
#' @return \code{data} with imputation and standardisation applied
#' @keywords internal
#' @noRd
tl_apply_preprocessing <- function(data, preprocessing, stats_learned) {
  if (isTRUE(preprocessing$impute_missing)) {
    for (col in names(data)) {
      na_idx <- is.na(data[[col]])
      if (!any(na_idx)) next

      if (is.numeric(data[[col]])) {
        med <- stats_learned$medians[[col]]
        if (is.null(med)) next
        data[[col]][na_idx] <- med
      } else if (is.factor(data[[col]]) || is.character(data[[col]])) {
        mode_val <- stats_learned$modes[[col]]
        if (is.null(mode_val) || is.na(mode_val)) next
        data[[col]][na_idx] <- mode_val
      }
    }
  }

  if (isTRUE(preprocessing$standardize)) {
    for (col in names(stats_learned$center)) {
      if (!col %in% names(data)) next

      data[[col]] <- (data[[col]] - stats_learned$center[[col]]) /
        stats_learned$scale[[col]]
    }
  }

  data
}

#' Run a tidylearn pipeline
#'
#' @param pipeline A tidylearn pipeline object
#' @param verbose Logical; whether to print progress
#' @return The input \code{tidylearn_pipeline} object with its
#'   \code{$results} component populated. Results include
#'   \code{$processed_data} (the training data after preprocessing),
#'   \code{$preprocessing_stats} (the medians, modes, centres and scales
#'   learned from the training data, replayed by
#'   \code{\link{tl_predict_pipeline}}), \code{$model_results} (a named
#'   list of per-model fits and metrics), \code{$best_model_name},
#'   \code{$best_model} (the winning \code{tidylearn_model}), and
#'   \code{$metric_values}.
#'
#'   The training data is every row when \code{validation = "cv"}: each
#'   model is scored across the folds and then refitted on all of them.
#'   With \code{validation = "split"} it is the training split alone, since
#'   the models kept are the ones fitted there and scored on the test rows.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(tree = list(method = "tree")),
#'   evaluation = list(metrics = "accuracy", validation = "cv",
#'     cv_folds = 2, best_metric = "accuracy"))
#' pipe <- tl_run_pipeline(pipe, verbose = FALSE)
#' }
#' @export
tl_run_pipeline <- function(pipeline, verbose = TRUE) {
  # Check if pipeline is valid
  if (!inherits(pipeline, "tidylearn_pipeline")) {
    stop("Input must be a tidylearn pipeline object", call. = FALSE)
  }

  # Extract components
  data <- pipeline$data
  formula <- pipeline$formula
  preprocessing <- pipeline$preprocessing
  models <- pipeline$models
  evaluation <- pipeline$evaluation

  # An intercept-only formula leaves the design matrix with no columns to
  # preprocess, which surfaced as "result would be too long a vector"
  response_var <- all.vars(formula)[1]
  predictors <- setdiff(all.vars(formula), response_var)
  if (length(predictors) == 0) {
    stop(
      "The formula names no predictors. A pipeline preprocesses and scores ",
      "predictors, so it needs at least one; got ", deparse(formula), ".",
      call. = FALSE
    )
  }

  # Cross-validation cannot make more folds than there are rows, and the
  # rsample message for it names `v` rather than cv_folds
  if (identical(evaluation$validation, "cv") &&
        evaluation$cv_folds > nrow(data)) {
    stop(
      "evaluation$cv_folds is ", evaluation$cv_folds, " but the data has ",
      nrow(data), " row", if (nrow(data) == 1) "" else "s",
      ". Cross-validation needs at least one row per fold.",
      call. = FALSE
    )
  }

  # A train_prop strictly between 0 and 1 still rounds to an empty side
  # on a small frame, and the two failures look nothing alike: an empty
  # training set is "result would be too long a vector", while an empty
  # test set scores nothing and the run finishes with a leaderboard of
  # NAs and a warning about the NAs rather than the split.
  if (identical(evaluation$validation, "split")) {
    n_train <- round(evaluation$train_prop * nrow(data))
    if (n_train < 1 || n_train >= nrow(data)) {
      stop(
        "evaluation$train_prop of ", evaluation$train_prop, " over ",
        nrow(data), " row", if (nrow(data) == 1) "" else "s",
        " puts ", n_train, " row", if (n_train == 1) "" else "s",
        " in the training set and ", nrow(data) - n_train, " in the test ",
        "set. Both sides have to be non-empty, so this needs a train_prop ",
        "nearer 0.5 or more rows.",
        call. = FALSE
      )
    }
  }

  # Without names the training loop below silently does nothing, leaving
  # an empty leaderboard and an unhelpful downstream error
  if (length(models) == 0 || is.null(names(models)) ||
        any(!nzchar(names(models)))) {
    stop(
      "`models` must be a named list of model configurations, e.g. ",
      "list(linear = list(method = \"linear\")).",
      call. = FALSE
    )
  }

  # The loop below indexes models[[model_name]], which resolves to the
  # first match. A repeated name therefore fitted one spec twice and
  # dropped the other without saying so.
  repeated <- unique(names(models)[duplicated(names(models))])
  if (length(repeated) > 0) {
    stop(
      "`models` has repeated name(s): ",
      paste0("'", repeated, "'", collapse = ", "),
      ". Each model needs its own name -- results are keyed by it, so a ",
      "repeat would discard every spec but the first.",
      call. = FALSE
    )
  }

  # A malformed spec otherwise surfaced as a base R internal further in:
  # "missing value where TRUE/FALSE needed" for a spec with no method,
  # "$ operator is invalid for atomic vectors" for a spec that is not a
  # list, neither of them naming the model responsible.
  for (model_name in names(models)) {
    spec <- models[[model_name]]
    if (!is.list(spec)) {
      stop(
        "Model '", model_name, "' must be a list of settings, e.g. ",
        "list(method = \"tree\"); got ", paste(class(spec), collapse = "/"),
        ".",
        call. = FALSE
      )
    }
    if (is.null(spec$method)) {
      stop(
        "Model '", model_name, "' has no 'method'. Every entry in `models` ",
        "needs one, e.g. list(method = \"tree\").",
        call. = FALSE
      )
    }
    if (!is.character(spec$method) || length(spec$method) != 1L) {
      stop(
        "Model '", model_name, "': 'method' must be a single method name; ",
        "got ", paste(class(spec$method), collapse = "/"), " of length ",
        length(spec$method), ".",
        call. = FALSE
      )
    }
    if (!spec$method %in% tl_supervised_methods()) {
      stop(
        "Model '", model_name, "': \"", spec$method, "\" is not a supervised ",
        "method. A pipeline fits a response, so it takes one of: ",
        paste(tl_supervised_methods(), collapse = ", "), ".",
        call. = FALSE
      )
    }
  }

  # Apply preprocessing
  if (verbose) {
    message("Applying preprocessing steps...")
  }

  if (verbose) {
    if (preprocessing$impute_missing) {
      message("  - Imputing missing values")
    }
    if (preprocessing$standardize) {
      message("  - Standardizing numeric features")
    }
    if (preprocessing$dummy_encode) {
      # Left to model.matrix during model fitting
      message("  - Creating dummy variables for categorical features")
    }
  }

  # Set up validation strategy. Splits are drawn from the RAW data.
  if (evaluation$validation == "cv") {
    # Statistics for the final model, which is legitimately fitted on
    # everything. tl_predict_pipeline() replays these on raw new data.
    #
    # Resampling below deliberately does NOT use them: each fold relearns
    # from its own analysis rows, so no assessment row contributes to the
    # transformation it is later scored under.
    preprocessing_stats <- tl_learn_preprocessing(
      data, formula, preprocessing
    )
    processed_data <- tl_apply_preprocessing(
      data, preprocessing, preprocessing_stats
    )

    cv_folds <- evaluation$cv_folds

    if (verbose) {
      message("Setting up ", cv_folds, "-fold cross-validation")
    }

    # Create cross-validation splits
    cv_splits <- rsample::vfold_cv(data, v = cv_folds)
  } else if (evaluation$validation == "split") {
    train_prop <- evaluation$train_prop

    if (verbose) {
      message("Setting up train/test split (", train_prop * 100, "% / ",
              (1 - train_prop) * 100, "%)")
    }

    # Create a single train/test split, then learn the transformation
    # from the training rows alone and replay it on the test rows
    train_idx <- sample(
      nrow(data),
      round(train_prop * nrow(data))
    )
    split_stats <- tl_learn_preprocessing(
      data[train_idx, ], formula, preprocessing
    )
    train_data <- tl_apply_preprocessing(
      data[train_idx, ], preprocessing, split_stats
    )
    test_data <- tl_apply_preprocessing(
      data[-train_idx, ], preprocessing, split_stats
    )

    # The models kept are the ones fitted on the training rows, so these
    # are the statistics tl_predict_pipeline() has to replay. Full-data
    # statistics moved its predictions away from the ones each model was
    # scored on.
    preprocessing_stats <- split_stats
    processed_data <- train_data
  }

  # Train and evaluate models
  model_results <- list()

  # Logistic on a 0/1 numeric response warns that it converts the response
  # to a factor, and the loop below refits once per fold, per model, and
  # again on every row: two specs over three folds gave eight copies. The
  # first is let through and the rest muffled.
  conversion_warned <- FALSE
  fit_model <- function(args) {
    withCallingHandlers(
      do.call(tl_model, args),
      tidylearn_response_conversion = function(w) {
        if (conversion_warned) {
          invokeRestart("muffleWarning")
        }
        conversion_warned <<- TRUE
      }
    )
  }

  for (model_name in names(models)) {
    if (verbose) {
      message("Training model: ", model_name)
    }

    # Extract model configuration
    model_config <- models[[model_name]]
    method <- model_config$method

    # Remove method from config to use rest as parameters
    model_params <- model_config[names(model_config) != "method"]

    if (evaluation$validation == "cv") {
      # Cross-validation approach
      cv_results <- list()

      for (i in 1:cv_folds) {
        if (verbose) {
          message("  - Fold ", i, "/", cv_folds)
        }

        # Get training and testing data for this fold, then learn the
        # transformation from the analysis rows only and replay it on
        # the assessment rows
        raw_train_fold <- rsample::analysis(cv_splits$splits[[i]])
        raw_test_fold <- rsample::assessment(cv_splits$splits[[i]])

        fold_stats <- tl_learn_preprocessing(
          raw_train_fold, formula, preprocessing
        )
        train_fold <- tl_apply_preprocessing(
          raw_train_fold, preprocessing, fold_stats
        )
        test_fold <- tl_apply_preprocessing(
          raw_test_fold, preprocessing, fold_stats
        )

        # Fit model on training fold
        model_args <- c(
          list(
            data = train_fold,
            formula = formula,
            method = method
          ),
          model_params
        )

        fold_model <- fit_model(model_args)

        # Evaluate on test fold. tl_evaluate() refuses a fold with no row
        # it can score -- every response in it missing, say -- and one such
        # fold is no reason to stop the run, so it scores NA and the
        # average is taken over the rest, as in tl_cv(). The split below
        # lets the error through: it has no other rows to score.
        fold_metrics <- tryCatch(
          tl_evaluate(fold_model, test_fold, metrics = evaluation$metrics),
          tidylearn_no_scored_rows = function(e) {
            warning(
              "Fold ", i, " of model '", model_name, "' is left out of its ",
              "average, since ",
              if (is.null(e$reason)) {
                "it has no rows to score."
              } else {
                paste0("none of its ", e$n_rows, " rows can be scored: ",
                       e$reason, ".")
              },
              call. = FALSE
            )
            tibble::tibble(metric = evaluation$metrics, value = NA_real_)
          }
        )

        # Store fold results
        cv_results[[i]] <- list(
          model = fold_model,
          metrics = fold_metrics
        )
      }

      # Calculate average metrics across folds
      all_metrics <- do.call(rbind, lapply(cv_results, function(x) x$metrics))

      avg_metrics <- all_metrics |>
        dplyr::group_by(.data$metric) |>
        dplyr::summarize(
          mean_value = mean(.data$value, na.rm = TRUE),
          sd_value = sd(.data$value, na.rm = TRUE)
        )

      # Train final model on all data
      final_model_args <- c(
        list(
          data = processed_data,
          formula = formula,
          method = method
        ),
        model_params
      )

      final_model <- fit_model(final_model_args)

      # Store results
      model_results[[model_name]] <- list(
        model = final_model,
        cv_results = cv_results,
        avg_metrics = avg_metrics
      )

      if (verbose) {
        for (i in seq_len(nrow(avg_metrics))) {
          metric <- avg_metrics$metric[i]
          mean_val <- avg_metrics$mean_value[i]
          sd_val <- avg_metrics$sd_value[i]

          message("    ", metric, ": ", round(mean_val, 4),
                  " (+/-", round(sd_val, 4), ")")
        }
      }
    } else if (evaluation$validation == "split") {
      # Train/test split approach
      # Fit model on training data
      model_args <- c(
        list(
          data = train_data,
          formula = formula,
          method = method
        ),
        model_params
      )

      split_model <- fit_model(model_args)

      # Evaluate on test data
      test_metrics <- tl_evaluate(
        split_model, test_data,
        metrics = evaluation$metrics
      )

      # Store results
      model_results[[model_name]] <- list(
        model = split_model,
        test_metrics = test_metrics
      )

      if (verbose) {
        for (i in seq_len(nrow(test_metrics))) {
          metric <- test_metrics$metric[i]
          value <- test_metrics$value[i]

          message("    ", metric, ": ", round(value, 4))
        }
      }
    }
  }

  # Select the best model
  best_metric <- evaluation$best_metric

  if (verbose) {
    message("Selecting best model based on ", best_metric)
  }

  # Extract metric values for each model
  metric_values <- sapply(
    names(model_results),
    function(model_name) {
      result <- model_results[[model_name]]

      if (evaluation$validation == "cv") {
        metric_row <- result$avg_metrics$metric ==
          best_metric
        if (any(metric_row)) {
          val <- result$avg_metrics$mean_value[metric_row]
          if (is.nan(val) || is.na(val)) NA_real_ else val
        } else {
          NA_real_
        }
      } else {
        metric_row <- result$test_metrics$metric ==
          best_metric
        if (any(metric_row)) {
          val <- result$test_metrics$value[metric_row]
          if (is.nan(val) || is.na(val)) NA_real_ else val
        } else {
          NA_real_
        }
      }
    }
  )

  # merge_evaluation_spec() admits known metrics only, so this is never NA
  is_higher_better <- tl_metric_higher_better(best_metric)

  # Find best model (use na.rm-safe which.max/which.min)
  valid_values <- !is.na(metric_values)
  if (!any(valid_values)) {
    best_idx <- 1L
    warning(
      "Could not determine best model from metric '",
      best_metric, "' -- all values NA. Using first model.",
      call. = FALSE
    )
  } else if (is_higher_better) {
    best_idx <- which.max(metric_values)
  } else {
    best_idx <- which.min(metric_values)
  }

  best_model_name <- names(model_results)[best_idx]
  best_model <- model_results[[best_model_name]]$model

  if (verbose) {
    message("Best model: ", best_model_name, " with ", best_metric, " = ",
            round(metric_values[best_idx], 4))
  }

  # Update pipeline with results
  pipeline$results <- list(
    processed_data = processed_data,
    preprocessing_stats = preprocessing_stats,
    model_results = model_results,
    best_model_name = best_model_name,
    best_model = best_model,
    metric_values = metric_values
  )

  pipeline
}

#' Get the best model from a pipeline
#'
#' @param pipeline A tidylearn pipeline object with results
#' @return The best \code{tidylearn_model} object from the pipeline,
#'   selected by the metric specified in \code{evaluation$best_metric}.
#'   The model was fitted on the preprocessed training data, so under the
#'   default preprocessing it expects predictors imputed and standardised
#'   the same way. Predict through \code{\link{tl_predict_pipeline}}, which
#'   replays that preprocessing on raw rows; \code{predict()} on the model
#'   itself reads raw values as if they were already standardised, and
#'   returns wrong predictions without a warning.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(tree = list(method = "tree")),
#'   evaluation = list(metrics = "accuracy", validation = "cv",
#'     cv_folds = 2, best_metric = "accuracy"))
#' pipe <- tl_run_pipeline(pipe, verbose = FALSE)
#' best <- tl_get_best_model(pipe)
#' best$spec$method
#'
#' # Predict through the pipeline, which preprocesses the new rows first
#' tl_predict_pipeline(pipe, iris[c(1, 51, 101), ])
#' }
#' @export
tl_get_best_model <- function(pipeline) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop(
      "Pipeline has not been run yet. Use tl_run_pipeline() first.",
      call. = FALSE
    )
  }

  pipeline$results$best_model
}

#' Compare models from a pipeline
#'
#' @param pipeline A tidylearn pipeline object with results
#' @param metrics Character vector of metrics to compare, each one the
#'   pipeline scored (if NULL, uses all available)
#' @return A \code{\link[ggplot2]{ggplot}} object showing a faceted bar
#'   chart comparing metric values across models, with the best model
#'   highlighted.
#' @importFrom ggplot2 ggplot aes geom_col facet_wrap labs theme_minimal
#' @export
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .,
#'   models = list(
#'     tree = list(method = "tree"),
#'     forest = list(method = "forest", ntree = 100)
#'   ),
#'   evaluation = list(validation = "cv", cv_folds = 3))
#' pipe <- tl_run_pipeline(pipe, verbose = FALSE)
#'
#' tl_compare_pipeline_models(pipe)
#'
#' # Restrict the comparison to one metric
#' tl_compare_pipeline_models(pipe, metrics = "accuracy")
#' }
tl_compare_pipeline_models <- function(pipeline, metrics = NULL) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop(
      "Pipeline has not been run yet. Use tl_run_pipeline() first.",
      call. = FALSE
    )
  }

  # Extract model results
  model_results <- pipeline$results$model_results
  is_cv <- !is.null(pipeline$evaluation) &&
    pipeline$evaluation$validation == "cv"

  # A metric the run did not score left nothing to plot, and the failure
  # surfaced inside ggplot2's faceting
  if (!is.null(metrics)) {
    scored <- unique(unlist(lapply(model_results, function(result) {
      if (is_cv) result$avg_metrics$metric else result$test_metrics$metric
    })))
    unscored <- setdiff(metrics, scored)
    if (length(unscored) > 0) {
      stop(
        "Metric(s) not scored by this pipeline: ",
        paste(unscored, collapse = ", "), ". Scored: ",
        paste(scored, collapse = ", "), ".",
        call. = FALSE
      )
    }
  }

  # Create data frame for plotting
  comparison_data <- NULL

  for (model_name in names(model_results)) {
    result <- model_results[[model_name]]

    if (is_cv) {
      # Get from average metrics
      model_metrics <- result$avg_metrics

      # Filter metrics if specified
      if (!is.null(metrics)) {
        model_metrics <- model_metrics[model_metrics$metric %in% metrics, ]
      }

      # Add model name
      model_metrics$model <- model_name

      # Rename columns
      names(model_metrics)[names(model_metrics) == "mean_value"] <- "value"

      # Add to comparison data
      if (is.null(comparison_data)) {
        comparison_data <- model_metrics
      } else {
        comparison_data <- rbind(comparison_data, model_metrics)
      }
    } else {
      # Get from test metrics
      model_metrics <- result$test_metrics

      # Filter metrics if specified
      if (!is.null(metrics)) {
        model_metrics <- model_metrics[model_metrics$metric %in% metrics, ]
      }

      # Add model name
      model_metrics$model <- model_name

      # Add to comparison data
      if (is.null(comparison_data)) {
        comparison_data <- model_metrics
      } else {
        comparison_data <- rbind(comparison_data, model_metrics)
      }
    }
  }

  # Add highlight for best model
  comparison_data$is_best <-
    comparison_data$model ==
    pipeline$results$best_model_name

  # Determine which metrics are "higher is better"
  comparison_data$higher_better <-
    tl_metric_higher_better(comparison_data$metric)

  # Create comparison plot
  p <- ggplot2::ggplot(
    comparison_data,
    ggplot2::aes(
      x = model,
      y = value,
      fill = is_best
    )
  ) +
    ggplot2::geom_col() +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::geom_text(
      ggplot2::aes(label = round(value, 3)),
      vjust = -0.5,
      size = 3
    ) +
    ggplot2::scale_fill_manual(values = c("steelblue", "darkred")) +
    ggplot2::labs(
      title = "Model Comparison",
      subtitle = "Best model highlighted in red",
      x = "Model",
      y = "Metric Value",
      fill = "Best Model"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )

  p
}

#' Make predictions using a pipeline
#'
#' @param pipeline A tidylearn pipeline object with results
#' @param new_data A data frame containing the new data
#' @param type Type of prediction (default: "response")
#' @param model_name Name of model to use (if NULL, uses the best model)
#' @param ... Additional arguments passed to predict
#' @return A \link[tibble]{tibble} with a \code{.pred} column containing
#'   predictions from the selected (or best) pipeline model, after
#'   applying the same preprocessing steps used during training.
#' @export
#' @examples
#' \donttest{
#' train <- iris[c(1:40, 51:90, 101:140), ]
#' test <- iris[c(41:50, 91:100, 141:150), ]
#'
#' pipe <- tl_pipeline(train, Species ~ .,
#'   models = list(
#'     tree = list(method = "tree"),
#'     forest = list(method = "forest", ntree = 100)
#'   ),
#'   evaluation = list(validation = "cv", cv_folds = 3))
#' pipe <- tl_run_pipeline(pipe, verbose = FALSE)
#'
#' # The best model, with the preprocessing learned on the training rows
#' tl_predict_pipeline(pipe, test)
#'
#' # Or a named candidate instead of the winner
#' tl_predict_pipeline(pipe, test, model_name = "tree")
#' }
tl_predict_pipeline <- function(pipeline,
                                new_data,
                                type = "response",
                                model_name = NULL,
                                ...) {
  # Check if pipeline has results
  if (is.null(pipeline$results)) {
    stop(
      "Pipeline has not been run yet. Use tl_run_pipeline() first.",
      call. = FALSE
    )
  }

  # Determine which model to use
  if (is.null(model_name)) {
    # Use best model
    model <- pipeline$results$best_model
  } else {
    # Use specified model
    if (!model_name %in% names(pipeline$results$model_results)) {
      stop("Model not found in pipeline: ", model_name, call. = FALSE)
    }
    model <- pipeline$results$model_results[[model_name]]$model
  }

  # Apply same preprocessing steps to new data
  processed_new_data <- new_data

  if (!is.null(pipeline$preprocessing)) {
    stats_learned <- pipeline$results$preprocessing_stats

    needs_stats <- isTRUE(pipeline$preprocessing$impute_missing) ||
      isTRUE(pipeline$preprocessing$standardize)

    if (needs_stats && is.null(stats_learned)) {
      stop(
        "This pipeline was run by an older version of tidylearn that did ",
        "not record preprocessing statistics, so new data cannot be ",
        "transformed consistently. Re-run it with tl_run_pipeline().",
        call. = FALSE
      )
    }

    # Replay exactly the transformation the final model was fitted under
    processed_new_data <- tl_apply_preprocessing(
      processed_new_data, pipeline$preprocessing, stats_learned
    )
  }

  # Make predictions
  predict(model, processed_new_data, type = type, ...)
}

#' Save a pipeline to disk
#'
#' @param pipeline A tidylearn pipeline object
#' @param file Path to save the pipeline
#' @return Called for its side effect of saving to disk; returns
#'   \code{NULL} invisibly.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' tl_save_pipeline(pipe, tempfile(fileext = ".rds"))
#' }
#' @export
tl_save_pipeline <- function(pipeline, file) {
  # Validate input
  if (!inherits(pipeline, "tidylearn_pipeline")) {
    stop("Input must be a tidylearn pipeline object", call. = FALSE)
  }

  # Save as RDS
  saveRDS(pipeline, file = file)

  invisible(NULL)
}

#' Load a pipeline from disk
#'
#' @param file Path to the pipeline file
#' @return A \code{tidylearn_pipeline} object previously saved with
#'   \code{\link{tl_save_pipeline}}.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' f <- tempfile(fileext = ".rds")
#' tl_save_pipeline(pipe, f)
#' pipe2 <- tl_load_pipeline(f)
#' }
#' @export
tl_load_pipeline <- function(file) {
  # Load RDS
  pipeline <- readRDS(file)

  # Validate
  if (!inherits(pipeline, "tidylearn_pipeline")) {
    stop("Loaded object is not a tidylearn pipeline", call. = FALSE)
  }

  pipeline
}

#' Print a tidylearn pipeline
#'
#' @param x A tidylearn pipeline object
#' @param ... Additional arguments (not used)
#' @return The input pipeline object \code{x}, returned invisibly.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' print(pipe)
#' }
#' @export
print.tidylearn_pipeline <- function(x, ...) {
  # Extract pipeline components
  formula <- x$formula
  data_dims <- dim(x$data)
  preprocessing <- names(x$preprocessing)[sapply(x$preprocessing, isTRUE)]
  models <- names(x$models)

  # Print pipeline summary
  cat("Tidylearn Pipeline\n")
  cat("=================\n")
  cat("Formula:", deparse(formula), "\n")
  cat("Data:", data_dims[1], "observations,", data_dims[2], "variables\n")
  cat("Preprocessing:", paste(preprocessing, collapse = ", "), "\n")
  cat("Models:", paste(models, collapse = ", "), "\n")

  # Print evaluation if available
  if (!is.null(x$evaluation)) {
    validation <- x$evaluation$validation
    metrics <- paste(x$evaluation$metrics, collapse = ", ")

    cat("Evaluation: ", validation)
    if (validation == "cv") {
      cat(" (", x$evaluation$cv_folds, " folds)", sep = "")
    }
    cat("\n")
    cat("Metrics:", metrics, "\n")
    cat("Best metric:", x$evaluation$best_metric, "\n")
  }

  # Print results if available
  if (!is.null(x$results)) {
    cat("\nResults\n")
    cat("=======\n")
    cat("Best model:", x$results$best_model_name, "\n")

    # Print metric values
    cat("Performance:\n")
    for (i in seq_along(x$results$metric_values)) {
      m_name <- names(x$results$metric_values)[i]
      metric_value <- x$results$metric_values[i]
      is_best <- if (m_name == x$results$best_model_name) " (best)" else ""

      cat(
        "  ", m_name, ": ", x$evaluation$best_metric, " = ",
        round(metric_value, 4), is_best, "\n",
        sep = ""
      )
    }
  }

  invisible(x)
}

#' Summarize a tidylearn pipeline
#'
#' @param object A tidylearn pipeline object
#' @param ... Additional arguments (not used)
#' @return The input pipeline \code{object}, returned invisibly. Called
#'   for its side effect of printing detailed pipeline and model results.
#' @examples
#' \donttest{
#' pipe <- tl_pipeline(iris, Species ~ .)
#' summary(pipe)
#' }
#' @export
summary.tidylearn_pipeline <- function(object, ...) {
  # If no results, just print the pipeline
  if (is.null(object$results)) {
    print(object)
    return(invisible(object))
  }

  # Print pipeline info
  print(object)

  # Print more detailed results
  cat("\nDetailed Results\n")
  cat("===============\n")

  for (model_name in names(object$results$model_results)) {
    cat("\nModel:", model_name, "\n")

    result <- object$results$model_results[[model_name]]

    if (!is.null(object$evaluation) && object$evaluation$validation == "cv") {
      # Print average metrics with standard deviation
      cat("Cross-validation metrics:\n")
      for (i in seq_len(nrow(result$avg_metrics))) {
        metric <- result$avg_metrics$metric[i]
        mean_val <- result$avg_metrics$mean_value[i]
        sd_val <- result$avg_metrics$sd_value[i]

        cat("  ", metric, ": ", round(mean_val, 4),
            " (+/-", round(sd_val, 4), ")", "\n", sep = "")
      }
    } else {
      # Print test metrics
      cat("Test metrics:\n")
      for (i in seq_len(nrow(result$test_metrics))) {
        metric <- result$test_metrics$metric[i]
        value <- result$test_metrics$value[i]

        cat("  ", metric, ": ", round(value, 4), "\n", sep = "")
      }
    }
  }

  # Summary of best model. summary() prints and returns the model, so
  # wrapping it in print() showed the model a second time.
  cat("\nBest Model Summary\n")
  cat("=================\n")
  summary(object$results$best_model)

  invisible(object)
}
