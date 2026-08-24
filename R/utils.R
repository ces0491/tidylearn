#' Utility functions for tidylearn
#' @keywords internal
#' @importFrom stats aov coef cor fitted median qqnorm
#'   reorder residuals runif sd setNames terms update var
#' @importFrom utils combn getFromNamespace head packageVersion
#' @noRd

# Suppress R CMD check notes about global variables from tidyverse NSE
utils::globalVariables(c(
  ".", ".id", ".obs_id", ".row_id", ":=",
  "Actual", "Assumption", "Details", "Freq",
  "Predicted", "SE.sim", "Status",
  "abs_shap_value", "actual", "all_of",
  "avg_sil_width", "cluster", "cluster_label",
  "coefficient", "component",
  "conf_lower", "conf_upper", "confidence",
  "cooks_distance", "cost", "cum_variance",
  "decay", "decile", "distance",
  "epoch", "error", "error_lower", "error_upper",
  "feature", "feature_value", "fold",
  "fpr", "frac_pos", "gap",
  "id1", "id2", "interaction_value", "is_best",
  "is_cook_influential", "is_core",
  "is_influential", "is_noise", "is_outlier",
  "is_top", "k", "knn_dist", "label", "lambda",
  "leverage", "lhs", "lift", "loading",
  "mean_pred_prob", "mean_value", "metric",
  "model", "n", "neighbor", "obs_id",
  "observation", "pc_num", "percentage",
  "pred", "pred_lower", "pred_upper",
  "predicted", "prop_variance",
  "residuals", "rhs", "score", "shap_value",
  "sil_width", "size", "sqrt_abs_residuals",
  "std_residual", "support", "tl_plot_model",
  "tl_plot_unsupervised",
  "tl_prediction_intervals", "tot_withinss",
  "tpr", "value", "var_value", "variable",
  "variance", "where", "x", "x_end",
  "y", "y_end", "abs_estimate", "estimate",
  "p_value", "significant", "std_error", "term"
))


# Null-coalescing operator
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Safe extraction of formula variables
#' @keywords internal
#' @noRd
get_formula_vars <- function(formula, data) {
  if (is.null(formula)) {
    return(names(data)[sapply(data, is.numeric)])
  }

  # Check if it's a one-sided formula (unsupervised)
  if (length(formula) == 2) {
    # One-sided: ~ vars
    rhs <- formula[[2]]
    if (rhs == ".") {
      names(data)[sapply(data, is.numeric)]
    } else {
      all.vars(formula)
    }
  } else {
    # Two-sided: response ~ predictors
    vars <- all.vars(formula)
    vars[-1]  # Exclude response
  }
}

#' Validate that a file path exists
#' @keywords internal
#' @noRd
tl_validate_file_path <- function(path) {
  if (!is.character(path) || length(path) != 1) {
    stop("'path' must be a single character string",
         call. = FALSE)
  }
  if (!file.exists(path)) {
    stop("File not found: '", path, "'",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Seed the RNG for this call only
#'
#' \code{set.seed()} rewrites the session's random stream, so a function
#' that takes a \code{seed} argument for its own reproducibility was also
#' deciding what every later \code{sample()} or \code{rnorm()} in the
#' caller's script would return. Two scripts differing only in whether
#' they passed \code{seed} would diverge everywhere downstream.
#'
#' Registers the restore on the calling function's frame, so the stream
#' goes back to what it was however that function exits.
#'
#' @param seed The seed to set, or NULL to leave the RNG untouched
#' @param envir The frame to restore on; defaults to the caller
#' @return `TRUE`, invisibly
#' @keywords internal
#' @noRd
tl_local_seed <- function(seed, envir = parent.frame()) {
  if (is.null(seed)) {
    return(invisible(TRUE))
  }

  # An R session that has not drawn a random number yet has no
  # .Random.seed at all. Restoring one we invented would be its own
  # side effect, so remove it instead.
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    previous <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    restore <- bquote(
      assign(".Random.seed", .(previous), envir = globalenv()) # nolint
    )
  } else {
    restore <- quote(
      suppressWarnings(rm(".Random.seed", envir = globalenv()))
    )
  }

  do.call(base::on.exit, list(restore, add = TRUE), envir = envir)
  set.seed(seed)
  invisible(TRUE)
}

#' Which rows of the training data the fit actually used
#'
#' \code{lm()} and friends drop incomplete cases, so \code{residuals()},
#' \code{fitted()} and every influence measure are shorter than
#' \code{model$data} whenever a predictor was missing. Anything combining
#' the two then fails with "arguments imply differing number of rows".
#'
#' @param model A fitted tidylearn model
#' @return Integer row indices into \code{model$data}
#' @keywords internal
#' @noRd
tl_fitted_rows <- function(model) {
  kept <- seq_len(nrow(model$data))
  omitted <- stats::na.action(model$fit)
  if (is.null(omitted)) {
    return(kept)
  }
  kept[-as.integer(omitted)]
}

#' Refuse data an algorithm cannot fit, naming what is wrong with it
#'
#' Several of the routines tidylearn wraps reject missing values from deep
#' inside C code, and the message that surfaces names neither the column
#' nor the problem: \code{stats::kmeans()} reports "NA/NaN/Inf in foreign
#' function call (arg 1)", and anything looping over k with \pkg{purrr}
#' wraps that again into "In index: 2. Caused by error in `do_one()`".
#' Missing values are the most ordinary thing that can be wrong with a
#' data set, so say so plainly and say where.
#'
#' Not every method needs this. \code{pam()}, \code{clara()},
#' \code{dist()} and \code{daisy()} handle missing values themselves and
#' are left alone.
#'
#' @param data A numeric data frame or matrix, already column-selected
#' @param what What is being fitted, for the message (e.g. "k-means")
#' @param tolerates Methods to suggest instead, or NULL for none
#' @return `TRUE`, invisibly, when the data is usable
#' @keywords internal
#' @noRd
tl_check_complete_numeric <- function(data, what,
                                      tolerates = c("pam", "clara")) {
  as_frame <- as.data.frame(data)
  if (ncol(as_frame) == 0L) {
    stop(
      what, " needs at least one numeric column, but none were found.",
      call. = FALSE
    )
  }

  bad <- vapply(
    as_frame,
    function(column) sum(!is.finite(as.numeric(column))),
    numeric(1)
  )
  offending <- bad[bad > 0]

  if (length(offending) == 0L) {
    return(invisible(TRUE))
  }

  named <- paste0(
    "'", names(offending), "' (", offending, ")",
    collapse = ", "
  )
  suggestion <- if (length(tolerates)) {
    paste0(
      " Impute or drop them first, or use ",
      paste0("method = \"", tolerates, "\"", collapse = " or "),
      ", which accept missing values."
    )
  } else {
    " Impute or drop them first."
  }

  stop(
    what, " cannot use missing or infinite values. Affected columns, with ",
    "counts: ", named, ".", suggestion,
    call. = FALSE
  )
}

#' Normalise a classification response
#'
#' Subsetting a data frame keeps every factor level, so
#' \code{iris[iris$Species != "setosa", ]} carries three levels while
#' holding two classes. Nothing downstream copes with that consistently:
#' \code{randomForest} and \code{glmnet} refuse to fit, \code{rpart}
#' returns a probability column for the class that is not there, and
#' \code{tl_event_level_args()} reads the declared count and so falls back
#' to \code{yardstick}'s first-level default -- silently scoring the wrong
#' class. Dropping the empty levels once, here, leaves every method a
#' response that says what it holds. It does not change any fit:
#' \code{glm()} and friends drop the empty level internally anyway.
#'
#' @param y A response vector
#' @return `y` as a factor whose levels are the ones actually present
#' @keywords internal
#' @noRd
tl_normalise_response <- function(y) {
  if (!is.factor(y)) {
    y <- factor(y)
  }
  droplevels(y)
}

#' Identify rows usable for prediction
#'
#' Several upstream predict methods default to \code{na.omit} and return a
#' vector shorter than the input, so row \emph{i} of the result stops
#' corresponding to row \emph{i} of the data. Callers use this to drop and
#' then re-expand explicitly, keeping predictions aligned.
#'
#' @param formula The model formula
#' @param new_data Data to predict on
#' @return A logical vector of length \code{nrow(new_data)}, TRUE where
#'   every predictor is present
#' @keywords internal
#' @noRd
tl_complete_predictor_rows <- function(formula, new_data) {
  # terms() expands a "." right-hand side against the columns actually
  # present; all.vars() on the raw formula would return nothing for
  # "y ~ ." and the check would silently pass every row.
  predictors <- tryCatch(
    all.vars(stats::delete.response(stats::terms(formula, data = new_data))),
    error = function(e) get_formula_vars(formula, new_data)
  )
  predictors <- intersect(predictors, names(new_data))

  if (length(predictors) == 0) {
    return(rep(TRUE, nrow(new_data)))
  }

  stats::complete.cases(new_data[, predictors, drop = FALSE])
}

#' Re-expand predictions to the full input length
#'
#' @param values Predictions computed on the complete-case subset
#' @param keep The logical vector returned by
#'   \code{tl_complete_predictor_rows()}
#' @return A vector of length \code{length(keep)} with NA in the dropped
#'   positions, preserving factor levels where applicable
#' @keywords internal
#' @noRd
tl_realign_predictions <- function(values, keep) {
  # Row identity in the returned tibble is positional, so names carried
  # over from new_data's rownames are noise -- and dropping them only on
  # the NA path would make the output shape depend on the data.
  if (all(keep)) {
    return(unname(values))
  }

  if (is.factor(values)) {
    out <- factor(rep(NA_character_, length(keep)),
                  levels = levels(values))
  } else {
    out <- rep(NA_real_, length(keep))
  }
  out[keep] <- unname(values)
  out
}

#' Re-expand a probability matrix to the full input length
#'
#' @param probs A matrix or data frame of probabilities, one row per
#'   complete case
#' @param keep The logical vector returned by
#'   \code{tl_complete_predictor_rows()}
#' @return An object of the same type with NA rows reinstated
#' @keywords internal
#' @noRd
tl_realign_prob_matrix <- function(probs, keep) {
  if (all(keep)) {
    return(probs)
  }

  out <- matrix(
    NA_real_, nrow = length(keep), ncol = ncol(probs),
    dimnames = list(NULL, colnames(probs))
  )
  out[keep, ] <- as.matrix(probs)
  out
}

#' Build a predictor design matrix for new data
#'
#' Uses the right-hand side of the formula only, so scoring unlabelled
#' data does not require the response column, and pins the factor levels
#' seen during training so contrast coding stays stable.
#'
#' @param formula The model formula
#' @param new_data Data to predict on
#' @param xlev Factor levels recorded at fit time (may be NULL)
#' @return A model matrix with the intercept column dropped
#' @keywords internal
#' @noRd
tl_predictor_matrix <- function(formula, new_data, xlev = NULL) {
  rhs_terms <- stats::delete.response(stats::terms(formula, data = new_data))

  frame <- if (is.null(xlev)) {
    stats::model.frame(rhs_terms, new_data, na.action = stats::na.pass)
  } else {
    stats::model.frame(rhs_terms, new_data, na.action = stats::na.pass,
                       xlev = xlev)
  }

  mm <- stats::model.matrix(rhs_terms, frame)
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

#' Check if required packages are installed
#' @keywords internal
#' @noRd
tl_check_packages <- function(...) {
  packages <- c(...)

  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Package '", pkg, "' is required but ",
           "not installed. ",
           "Please install it with: install.packages('",
           pkg, "')",
           call. = FALSE)
    }
  }

  invisible(TRUE)
}

#' Resolve a colour specification to a vector
#'
#' \code{color_by} is documented as a column name, but the tibbles these
#' plots draw from carry only an id and the coordinates -- there is nowhere
#' for a grouping variable to live. Accepting a bare vector of the right
#' length makes the documented intent reachable, and a column name still
#' works when the column really is present.
#'
#' @param color_by A column name, or a vector as long as \code{data} has rows.
#' @param data The data frame being plotted.
#' @param arg Argument name, used in error messages.
#' @return A vector as long as \code{nrow(data)}, or NULL.
#' @keywords internal
#' @noRd
resolve_color_by <- function(color_by, data, arg = "color_by") {
  if (is.null(color_by)) {
    return(NULL)
  }

  if (is.character(color_by) && length(color_by) == 1L &&
        color_by %in% names(data)) {
    return(data[[color_by]])
  }

  if (length(color_by) == nrow(data)) {
    return(color_by)
  }

  stop(
    "'", arg, "' must name a column of the data being plotted (",
    paste(names(data), collapse = ", "),
    ") or be a vector of length ", nrow(data),
    "; got ", if (is.character(color_by) && length(color_by) == 1L) {
      paste0("\"", color_by, "\"")
    } else {
      paste0("length ", length(color_by))
    }, ".",
    call. = FALSE
  )
}

#' Coerce a model specification to a formula
#'
#' \code{tl_model()} has always accepted a character specification and
#' coerced it with \code{as.formula()}, but that coercion lives inside
#' \code{tl_model_supervised()}. Every caller that reads
#' \code{all.vars(formula)[1]} before delegating -- \code{tl_pipeline()},
#' \code{tl_prepare_data()}, \code{tl_auto_ml()}, the tuners -- ran that
#' extraction against the raw argument, and \code{all.vars("y ~ x")} is
#' \code{character(0)}. The response name came back \code{NA}, so
#' \code{data[[NA]]} was \code{NULL} and a classification problem was
#' silently treated as regression.
#'
#' Normalising at the entry point instead means the coercion happens once,
#' before anything inspects the formula.
#'
#' @param formula A formula, or a string that parses as one.
#' @param arg The argument name to use in the error message.
#' @return A formula.
#' @keywords internal
#' @noRd
tl_as_formula <- function(formula, arg = "formula") {
  if (inherits(formula, "formula")) {
    return(formula)
  }

  if (is.character(formula) && length(formula) == 1L && !is.na(formula)) {
    coerced <- tryCatch(
      stats::as.formula(formula),
      error = function(e) NULL
    )
    if (!is.null(coerced)) {
      return(coerced)
    }
    stop(
      "'", arg, "' is the string \"", formula,
      "\", which does not parse as a formula.",
      call. = FALSE
    )
  }

  stop(
    "'", arg, "' must be a formula such as y ~ x, or a string that parses ",
    "as one; got ", paste(class(formula), collapse = "/"),
    if (length(formula) != 1L) paste0(" of length ", length(formula)) else "",
    ".",
    call. = FALSE
  )
}

#' Refuse a classification fit whose predictors carry no information
#'
#' randomForest's classification path does not terminate when every
#' predictor has zero variance: it keeps drawing \code{mtry} candidates
#' looking for a split that cannot exist. The loop is C-level, so it
#' ignores interrupts -- the session has to be killed. Regression is
#' unaffected, and so is the case where only some predictors are
#' constant, which is why the check is this narrow.
#'
#' @param data The model frame, response included.
#' @param formula The model formula.
#' @param method The method name, for the message.
#' @return Invisibly \code{TRUE}; called for the error.
#' @keywords internal
#' @noRd
tl_check_predictor_variance <- function(data, formula, method) {
  predictors <- setdiff(all.vars(formula), all.vars(formula)[1])
  if (identical(predictors, ".") || length(predictors) == 0L) {
    predictors <- setdiff(names(data), all.vars(formula)[1])
  }
  predictors <- intersect(predictors, names(data))
  if (length(predictors) == 0L) {
    return(invisible(TRUE))
  }

  constant <- vapply(
    data[predictors],
    function(column) length(unique(column[!is.na(column)])) <= 1L,
    logical(1)
  )

  if (all(constant)) {
    stop(
      "Method \"", method, "\" cannot fit a classification model when every ",
      "predictor is constant. Affected columns: ",
      paste0("'", predictors, "'", collapse = ", "),
      ". There is no split to find, and randomForest does not stop looking ",
      "for one.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

#' The methods tl_model() dispatches on
#'
#' Kept in one place so that callers which need to validate a method name
#' -- tl_pipeline(), for one -- cannot drift from what tl_model() will
#' actually accept.
#'
#' @return A character vector of method names.
#' @keywords internal
#' @noRd
tl_supervised_methods <- function() {
  c(
    "linear", "polynomial", "logistic", "tree",
    "forest", "boost", "ridge", "lasso",
    "elastic_net", "svm", "nn", "deep", "xgboost"
  )
}

#' @rdname tl_supervised_methods
#' @keywords internal
#' @noRd
tl_unsupervised_methods <- function() {
  c("pca", "mds", "kmeans", "pam", "clara", "hclust", "dbscan")
}

#' The metric names tl_evaluate() can compute
#'
#' Matches the branches in \code{tl_calc_classification_metrics()} and
#' \code{tl_calculate_regression_metrics()} and the list documented on
#' \code{tl_evaluate()}. A name outside this set is not computed, so the
#' score comes back NA and the caller reports the symptom rather than the
#' cause -- which is what this lets callers refuse up front.
#'
#' @param is_classification Whether the task is classification.
#' @return A character vector of metric names.
#' @keywords internal
#' @noRd
tl_known_metrics <- function(is_classification) {
  if (is_classification) {
    c("accuracy", "precision", "recall", "sensitivity",
      "specificity", "f1", "auc", "pr_auc")
  } else {
    c("rmse", "mse", "mae", "mape", "rsq")
  }
}
