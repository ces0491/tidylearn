#' Utility functions for tidylearn
#' @keywords internal
#' @importFrom stats aov coef cor fitted median qqnorm
#'   reorder residuals runif sd setNames terms update var
#' @importFrom utils combn getFromNamespace head packageVersion
#' @noRd

# Suppress R CMD check notes about global variables from tidyverse NSE
utils::globalVariables(c(
  ".", ".id", ".obs_id", ":=",
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
#'
#' For a one-sided formula, the columns an unsupervised method fits on. A
#' dot stands for every numeric column, less any the formula subtracts;
#' a column named explicitly is returned whatever its type, for the caller
#' to judge. The unsupervised fitters select these columns by name, so a
#' term that is not a column name -- \code{log(x)}, \code{x:z} -- is
#' refused here: selecting its variable instead fitted the raw column.
#'
#' @keywords internal
#' @noRd
get_formula_vars <- function(formula, data) {
  if (is.null(formula)) {
    return(names(data)[sapply(data, is.numeric)])
  }
  tl_check_subtracted(formula, data)

  # Check if it's a one-sided formula (unsupervised)
  if (length(formula) == 2) {
    # terms() expands the dot and applies `- x`; all.vars() on the formula
    # returned "." itself for ~ . - x
    labels <- attr(stats::terms(formula, data = data), "term.labels")
    terms <- lapply(labels, str2lang)
    bare <- vapply(terms, is.name, logical(1))
    if (!all(bare)) {
      one <- sum(!bare) == 1L
      stop(
        "Formulas for unsupervised methods name columns only, but this one ",
        "has ", paste(labels[!bare], collapse = ", "), ". Add ",
        if (one) "it to the data as a column" else "them as columns",
        ", e.g. with dplyr::mutate(), and name ",
        if (one) "that column" else "those columns", " instead.",
        call. = FALSE
      )
    }
    vars <- vapply(terms, as.character, character(1))

    if ("." %in% all.vars(formula)) {
      named <- setdiff(all.vars(formula), ".")
      numeric_column <- vapply(
        vars,
        function(v) v %in% names(data) && is.numeric(data[[v]]),
        logical(1)
      )
      vars <- vars[numeric_column | vars %in% named]
    }
    unname(vars)
  } else {
    # Two-sided: the variables the expanded terms use. all.vars() on the
    # formula itself returns "." for `y ~ .` and returns `id` for
    # `y ~ . - id`, which is exactly the column the caller excluded.
    labels <- attr(stats::terms(formula, data = data), "term.labels")
    unique(unlist(lapply(labels, function(label) all.vars(str2lang(label)))))
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
#' The rows are read off the fit's model frame where it has one, by row
#' name: that covers every way a row can be left out, where
#' \code{na.action()} records only the missing values. A row kept under
#' \code{na.exclude} is not one the fit used, although \code{residuals()}
#' pads it back in as \code{NA}.
#'
#' @param model A fitted tidylearn model
#' @return Integer row indices into \code{model$data}
#' @keywords internal
#' @noRd
tl_fitted_rows <- function(model) {
  frame <- if (inherits(model$fit, "lm")) {
    tryCatch(stats::model.frame(model$fit), error = function(e) NULL)
  }
  if (!is.null(frame)) {
    rows <- match(rownames(frame), rownames(model$data))
    if (!anyNA(rows)) {
      return(rows)
    }
  }

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

#' The response a two-sided formula fits
#'
#' The left-hand side is the bare column only when the formula says so.
#' \code{factor(cyl) ~ wt} fits a factor and \code{I(mpg > 20) ~ wt} a
#' logical, so deciding the task from the raw column read the wrong
#' vector: the first was fitted by \code{lm()} on the factor codes, the
#' second refused by logistic as numeric with 25 distinct values. The
#' left-hand side is evaluated as \code{model.frame()} evaluates it, in the
#' data with the formula's environment behind it.
#'
#' @param formula A two-sided formula
#' @param data The data it is fitted on
#' @return The response, one value per row of \code{data}; for a bare name
#'   that is not a column, \code{NULL}, as \code{data[[name]]} gives
#' @keywords internal
#' @noRd
tl_formula_response <- function(formula, data) {
  lhs <- formula[[2L]]
  if (is.name(lhs)) {
    return(data[[as.character(lhs)]])
  }

  response <- tryCatch(
    eval(lhs, data, environment(formula) %||% baseenv()),
    error = function(e) {
      stop(
        "The response ", deparse1(lhs), " could not be computed from the ",
        "data: ", conditionMessage(e),
        call. = FALSE
      )
    }
  )
  # scale() and the like return a one-column matrix
  if (is.matrix(response) && ncol(response) == 1L) {
    response <- response[, 1]
  }
  response
}

#' Read observed classes against the levels a model was trained on
#'
#' \code{tl_normalise_response()} cleans the training response, but data
#' scored later never passes through it: a test split of
#' \code{iris[iris$Species != "setosa", ]} still declares setosa, so
#' yardstick refused truth and estimate as having different levels, and the
#' ROC and lift plots read the binary model as multiclass. A test factor
#' whose levels were merely reordered silently moved the positive class.
#' Everything that scores a classification model reads the observed classes
#' through here, so the levels -- and with them the positive class, the
#' second level -- follow the model rather than the data.
#'
#' A class the model never saw (one a CV training fold happened to miss) has
#' no prediction or probability column to compare with, so those rows are
#' left out of the scoring, with a warning naming the classes.
#'
#' @param actuals Observed classes: factor, character, logical or numeric
#' @param model_levels The model's classes, \code{model$spec$response_levels}
#' @return A list: \code{actuals}, a factor with levels \code{model_levels};
#'   \code{keep}, a logical vector, FALSE where the class is missing or one
#'   the model was not trained on
#' @keywords internal
#' @noRd
tl_align_classes <- function(actuals, model_levels) {
  observed <- as.character(actuals)
  unseen <- !is.na(observed) & !observed %in% model_levels
  if (any(unseen)) {
    warning(
      sum(unseen), " row(s) belong to a class the model was not trained ",
      "on (", paste(unique(observed[unseen]), collapse = ", "), ") and ",
      "are left out of the scoring.",
      call. = FALSE
    )
  }
  aligned <- factor(observed, levels = model_levels)
  list(actuals = aligned, keep = !is.na(aligned))
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
  # If the formula cannot be expanded against new_data, fall back to the
  # names written on its right-hand side. get_formula_vars() is no use
  # here: it expands the formula the same way and fails the same way. A
  # column the formula subtracts is not a predictor, so a missing value in
  # it leaves the row usable.
  predictors <- tryCatch(
    all.vars(stats::delete.response(stats::terms(
      tl_fit_formula(formula, new_data, predicting = TRUE),
      data = new_data
    ))),
    error = function(e) all.vars(formula[[length(formula)]])
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
#' @param xlev Factor levels recorded at fit time (may be NULL). Its
#'   \code{"terms"} attribute, when \code{tl_model()} set one, is the
#'   training predictor terms the matrix is built from.
#' @return A model matrix with the intercept column dropped
#' @keywords internal
#' @noRd
tl_predictor_matrix <- function(formula, new_data, xlev = NULL) {
  # The training terms, which tl_model() keeps with the levels, carry the
  # values a data-dependent term was computed with -- the centre and scale
  # of scale(hp), the knots of a spline -- as predict.lm() uses them.
  # Rebuilt from the formula, such a term was recomputed on the rows
  # predicted, so a row scored alone differed from the same row in the
  # full frame. A model without them is rebuilt from the formula, less any
  # column it subtracts, which the matrix never uses.
  rhs_terms <- attr(xlev, "terms")
  if (!inherits(rhs_terms, "terms")) {
    rhs_terms <- stats::delete.response(stats::terms(
      tl_fit_formula(formula, new_data, predicting = TRUE),
      data = new_data
    ))
  }

  # xlev is keyed by column, and model.frame() applies it to the frame's
  # variables. For a computed term such as relevel(f, "b") the variable is
  # the expression, so passing the column's levels warned "variable 'f' is
  # not a factor" on every prediction.
  frame_variables <- vapply(
    as.list(attr(rhs_terms, "variables"))[-1], deparse1, character(1)
  )
  xlev <- xlev[names(xlev) %in% frame_variables]

  frame <- if (length(xlev) == 0L) {
    stats::model.frame(rhs_terms, new_data, na.action = stats::na.pass)
  } else {
    stats::model.frame(rhs_terms, new_data, na.action = stats::na.pass,
                       xlev = xlev)
  }

  mm <- stats::model.matrix(rhs_terms, frame)
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

#' The columns a fitted formula reads
#'
#' Every variable of the formula's terms, with a dot expanded against the
#' training data: the response, the terms, any offset, and a column the
#' formula subtracts. The last is not a predictor, but \code{predict.lm()}
#' and the other model-frame methods evaluate every variable of the terms
#' they stored, and without it \code{y ~ . - qsec} failed with "object
#' 'qsec' not found".
#'
#' @param formula The model formula
#' @param data The training data
#' @return Variable names, or NULL when the formula cannot be expanded
#' @keywords internal
#' @noRd
tl_model_columns <- function(formula, data) {
  if (!inherits(formula, "formula") || is.null(data)) {
    return(NULL)
  }
  model_terms <- tryCatch(
    stats::terms(formula, data = data),
    error = function(e) NULL
  )
  if (is.null(model_terms)) {
    return(NULL)
  }
  all.vars(model_terms)
}

#' A formula without the columns it subtracts
#'
#' \code{terms()} keeps a subtracted column among its variables, so a fit
#' on \code{y ~ . - id} stored terms that name \code{id}, and predict() on
#' rows without it failed with "object 'id' not found". Such a formula is
#' written out from its expanded terms, which name only the columns the
#' model uses. Any other formula is returned as it is: written out, a dot
#' would list every column in the fit's printed call.
#'
#' At fit time a subtracted name that is not a column is refused: it is
#' almost always a misspelling, and written out, the column the caller
#' meant to drop would be fitted. At prediction new data may lack a
#' subtracted column, since the model never uses it.
#'
#' @param formula A model formula
#' @param data The data its dot expands against
#' @param predicting TRUE when \code{data} is new data rather than the
#'   training data
#' @return \code{formula}, written out without its subtracted columns when
#'   it has any
#' @keywords internal
#' @noRd
tl_fit_formula <- function(formula, data, predicting = FALSE) {
  subtracted <- tl_subtracted_vars(formula)
  if (length(subtracted) == 0L) {
    return(formula)
  }
  absent <- setdiff(subtracted, names(data))
  if (!predicting) {
    tl_check_subtracted(formula, data)
  }

  # The dot is expanded against the data's own columns. Only a subtracted
  # column the new data lacks is added, empty: terms() warns "'varlist' has
  # changed ... should no longer happen!" without it. Anything else the
  # formula names but the data lacks -- a vector or scalar in the formula's
  # environment -- stays out, or it would join the dot as a column.
  columns <- data[0, , drop = FALSE]
  for (variable in absent) {
    columns[[variable]] <- logical(0)
  }

  model_terms <- tryCatch(
    stats::terms(formula, data = columns),
    error = function(e) NULL
  )
  if (is.null(model_terms)) {
    return(formula)
  }

  variables <- as.list(attr(model_terms, "variables"))[-1]
  response <- attr(model_terms, "response")
  used <- c(
    if (response > 0) all.vars(variables[[response]]),
    unlist(lapply(
      attr(model_terms, "term.labels"),
      function(label) all.vars(str2lang(label))
    )),
    unlist(lapply(variables[attr(model_terms, "offset")], all.vars))
  )
  if (all(all.vars(model_terms) %in% used)) {
    return(formula)
  }
  stats::formula(stats::terms(formula, data = columns, simplify = TRUE))
}

#' The variables a formula's right-hand side subtracts
#'
#' Read off the formula as written, through its chain of \code{+} and
#' \code{-}, so it needs no \code{terms()} call and no data.
#'
#' @param formula A model formula
#' @return Variable names, without the dot
#' @keywords internal
#' @noRd
tl_subtracted_vars <- function(formula) {
  subtracted <- character(0)
  walk <- function(expr) {
    if (!is.call(expr)) {
      return(invisible(NULL))
    }
    head <- expr[[1L]]
    if (identical(head, as.name("-"))) {
      if (length(expr) == 3L) {
        walk(expr[[2L]])
        subtracted <<- c(subtracted, all.vars(expr[[3L]]))
      } else {
        subtracted <<- c(subtracted, all.vars(expr[[2L]]))
      }
    } else if (identical(head, as.name("+")) ||
                 identical(head, as.name("("))) {
      for (operand in as.list(expr)[-1L]) walk(operand)
    }
  }
  walk(formula[[length(formula)]])
  setdiff(unique(subtracted), ".")
}

#' Refuse a subtracted name the data does not have
#'
#' \code{terms()} subtracts a name that is not a column without complaint,
#' so \code{mpg ~ . - qsce} kept qsec in a fit written out from it, where
#' \code{lm()} on the formula itself stopped with "object 'qsce' not
#' found".
#'
#' @param formula A model formula
#' @param data The training data
#' @return \code{TRUE}, invisibly, when every subtracted name is a column
#' @keywords internal
#' @noRd
tl_check_subtracted <- function(formula, data) {
  absent <- setdiff(tl_subtracted_vars(formula), names(data))
  if (length(absent) > 0L) {
    one <- length(absent) == 1L
    stop(
      "The formula subtracts ", paste0("'", absent, "'", collapse = ", "),
      if (one) ", which is not a column" else ", which are not columns",
      " of the data. Check the spelling: subtracting a name the data does ",
      "not have drops nothing.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Categorical columns a formula uses as they are
#'
#' The variables that make up a term on their own, or an interaction of
#' such, and appear in no term that computes something from them. Only
#' those can be stored as factors without changing what the formula
#' computes: \code{as.numeric()} of a factor reads its level codes, and
#' \code{nchar()} refuses one.
#'
#' @param formula A two-sided model formula
#' @param data The training data
#' @return Variable names
#' @keywords internal
#' @noRd
tl_bare_term_vars <- function(formula, data) {
  labels <- attr(stats::terms(formula, data = data), "term.labels")

  interaction_parts <- function(expr) {
    if (is.call(expr) && identical(expr[[1L]], as.name(":"))) {
      c(interaction_parts(expr[[2L]]), interaction_parts(expr[[3L]]))
    } else {
      list(expr)
    }
  }

  bare <- character(0)
  computed <- character(0)
  parts <- do.call(c, lapply(lapply(labels, str2lang), interaction_parts))
  for (part in parts) {
    if (is.name(part)) {
      bare <- c(bare, as.character(part))
    } else {
      computed <- c(computed, all.vars(part))
    }
  }
  setdiff(unique(bare), computed)
}

#' Quote values for a message
#'
#' @param x Values to list.
#' @param max How many to show before eliding the rest.
#' @return A single string such as \code{"a", "b", ...}.
#' @keywords internal
#' @noRd
tl_quote_list <- function(x, max = 5L) {
  shown <- paste0("\"", utils::head(x, max), "\"", collapse = ", ")
  if (length(x) > max) paste0(shown, ", ...") else shown
}

#' Read new data's categorical predictors against the training levels
#'
#' A backend reads a factor by its declared levels, and new data declares
#' fewer whenever it holds only some of the categories: randomForest
#' refused such a frame for its level count, and coded a character column
#' by the values present, so a row's prediction depended on the rows
#' scored with it. Every column the model recorded levels for is put on
#' those levels here, before any method's predict sees it. A value outside
#' them is refused by name, where gbm alone scored one without complaint.
#'
#' @param new_data Data to predict on.
#' @param xlev Training levels, \code{model$spec$xlev}.
#' @param training The training data, which says whether a factor is
#'   ordered, or NULL to keep new_data's own.
#' @return \code{new_data}, those columns factors on the training levels.
#' @keywords internal
#' @noRd
tl_align_predictor_levels <- function(new_data, xlev, training = NULL) {
  unseen <- character(0)

  for (column in intersect(names(xlev), names(new_data))) {
    trained <- xlev[[column]]
    values <- new_data[[column]]
    # randomForest treats an ordered factor as numeric, and refuses an
    # unordered one in its place
    ordered <- if (is.null(training[[column]])) {
      is.ordered(values)
    } else {
      is.ordered(training[[column]])
    }
    if (is.factor(values) && identical(levels(values), trained) &&
          is.ordered(values) == ordered) {
      next
    }

    observed <- as.character(values)
    new_levels <- unique(observed[!is.na(observed) & !observed %in% trained])
    if (length(new_levels) > 0) {
      unseen <- c(unseen, paste0(
        "'", column, "' has ", tl_quote_list(new_levels),
        " (trained on ", tl_quote_list(trained, max = 10L), ")"
      ))
      next
    }

    new_data[[column]] <- factor(
      observed,
      levels = trained, ordered = ordered,
      exclude = if (anyNA(trained)) NULL else NA
    )
  }

  if (length(unseen) > 0) {
    stop(
      "new_data holds levels the model was not trained on: ",
      paste(unseen, collapse = "; "),
      ". Recode or drop those rows before predicting.",
      call. = FALSE
    )
  }
  new_data
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
#' The predictor set has to come from \code{terms()}. Reading
#' \code{all.vars()} off the raw formula gets \code{y ~ . - id} wrong in
#' both directions: the \code{.} is not a column, so the set collapses to
#' the one column the caller excluded. That refuses a fittable model when
#' the excluded column is constant, and lets the hang through when it is
#' the excluded column that varies.
#'
#' @param data The model frame, response included.
#' @param formula The model formula.
#' @param method The method name, for the message.
#' @return Invisibly \code{TRUE}; called for the error.
#' @keywords internal
#' @noRd
tl_check_predictor_variance <- function(data, formula, method) {
  predictors <- tryCatch(
    {
      labels <- attr(stats::terms(formula, data = data), "term.labels")
      unique(unlist(lapply(labels, function(l) all.vars(str2lang(l)))))
    },
    error = function(e) setdiff(all.vars(formula), all.vars(formula)[1])
  )
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
#' \code{tl_calc_regression_metrics()} and the list documented on
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

#' Describe a value for an error message
#'
#' A setting can be absent as well as wrong. \code{modifyList()} drops an
#' element set to \code{NULL}, so "got NULL" and "got a character vector
#' of length 2" both need saying, and neither survives being pasted into
#' a message as a bare value.
#'
#' @param x The value to describe.
#' @return A single string.
#' @keywords internal
#' @noRd
tl_describe_value <- function(x) {
  if (is.null(x)) {
    return("NULL")
  }
  if (length(x) != 1L) {
    return(paste0(
      paste(class(x), collapse = "/"), " of length ", length(x)
    ))
  }
  if (is.character(x)) {
    return(paste0("\"", x, "\""))
  }
  paste(format(x), collapse = ", ")
}

#' The method each entry of a pipeline spec names
#'
#' @param models A pipeline's \code{models} list, possibly malformed --
#'   \code{tl_run_pipeline()} is where a bad spec is reported, so this
#'   only reads what it safely can.
#' @return A character vector as long as \code{models}, \code{NA} where an
#'   entry does not name a single method.
#' @keywords internal
#' @noRd
tl_spec_methods <- function(models) {
  if (!is.list(models) || length(models) == 0L) {
    return(character(0))
  }
  vapply(
    models,
    function(spec) {
      if (is.list(spec) && is.character(spec$method) &&
            length(spec$method) == 1L) {
        spec$method
      } else {
        NA_character_
      }
    },
    character(1)
  )
}

#' Take the training data out of a fitted model's stored call
#'
#' \code{do.call()} evaluates its arguments before it builds the call, so
#' the \code{match.call()} the wrapped function runs records the whole
#' training frame as a literal. \code{print()} on the fitted object then
#' spills every row, and on a 960-row frame the call alone was 159 Kb of
#' a 1.5 Mb forest. Calling the function directly is not the remedy:
#' leaving an argument out is what lets the wrapped package apply the
#' default it documents, and only \code{do.call()} can leave one out.
#'
#' @param fit A fitted model that stores its call as \code{$call}.
#' @return \code{fit}, its call referring to the data through
#'   \code{tl_hold_call_args()} and naming its function through
#'   \code{tl_call_head()}.
#' @keywords internal
#' @noRd
tl_restore_call_data <- function(fit) {
  if (!is.null(fit$call) && is.call(fit$call) && !is.null(fit$call$data)) {
    fit <- tl_hold_call_args(fit, c("data", "weights", "subset"))
    if (is.name(fit$call[[1]])) {
      fit$call[[1]] <- tl_call_head(as.character(fit$call[[1]]))
    }
  }
  fit
}

#' Hold a stored call's values where re-running the call finds them
#'
#' A call built by \code{do.call()} holds its values literally, and
#' \code{print()} on the fit spills them. Writing each argument's name
#' back in its place printed well but broke \code{update()} and
#' \code{step()}, which re-evaluate the call in their caller's frame:
#' \code{data} there was whatever the caller had called \code{data} -- a
#' script's full frame, so an 18-row fit was refitted on 32 rows -- or
#' \code{utils::data()}, and \code{weights} found \code{stats::weights()}.
#' Each value is kept in an environment the call refers to instead, so the
#' call prints in a line as \code{<environment>$data} and re-evaluating it
#' anywhere reaches the rows and weights the model was fitted with. The
#' environment shares the values with the model rather than copying them,
#' though \code{saveRDS()} writes them out a second time.
#'
#' @param fit A fitted model that stores its call as \code{$call}.
#' @param args Names of the arguments to hold.
#' @return \code{fit}, with those arguments of its call replaced.
#' @keywords internal
#' @noRd
tl_hold_call_args <- function(fit, args) {
  # An argument the backend recorded as a name or expression was never a
  # literal value, so it has nothing to hold
  held <- Filter(
    function(arg) !is.language(fit$call[[arg]]),
    intersect(names(fit$call), args)
  )
  if (length(held) == 0L) {
    return(fit)
  }

  store <- new.env(parent = emptyenv())
  for (arg in held) {
    assign(arg, fit$call[[arg]], envir = store)
    fit$call[[arg]] <- call("$", store, as.name(arg))
  }
  fit
}

#' The head of a stored call, with the package it comes from
#'
#' \code{update()} evaluates a fit's call in the caller's environment, and
#' a session that has loaded tidylearn has not attached rpart,
#' randomForest, e1071, nnet or gbm: a bare \code{rpart(...)} head failed
#' with "could not find function \"rpart\"". Heads from those packages are
#' written \code{pkg::fun}. \code{lm()} and \code{glm()} stay bare, since
#' stats is attached in every session.
#'
#' @param name The function's name, as tidylearn calls it.
#' @param fun The function, or NULL to look the name up as tidylearn sees
#'   it, through its imports.
#' @return A name, or a \code{pkg::fun} call.
#' @keywords internal
#' @noRd
tl_call_head <- function(name, fun = NULL) {
  if (is.null(fun)) {
    fun <- tryCatch(
      get(name, envir = environment(tl_call_head), mode = "function"),
      error = function(e) NULL
    )
  }
  package <- if (is.function(fun) && !is.null(environment(fun))) {
    environmentName(environment(fun))
  } else {
    ""
  }
  if (!nzchar(package) ||
        package %in% c("base", "stats", "R_GlobalEnv", "tidylearn")) {
    return(as.name(name))
  }
  call("::", as.name(package), as.name(name))
}

#' The rows a subset argument selects
#'
#' Indexing the row numbers with \code{subset} selects what
#' \code{model.frame()} selects with it: a logical is recycled and its
#' \code{NA} selects no row, negative numbers drop rows, and names match
#' row names.
#'
#' @param data The training data.
#' @param subset The \code{subset} argument, as passed.
#' @return Integer row positions.
#' @keywords internal
#' @noRd
tl_subset_rows <- function(data, subset) {
  rows <- seq_len(nrow(data))
  names(rows) <- rownames(data)
  kept <- rows[subset]
  unname(kept[!is.na(kept)])
}

#' Fitting arguments that hold one value per training row
#'
#' These cannot follow the rows into a resampling fold, so a model records
#' only their names, and \code{tl_compare_cv()} refuses to refit it.
#'
#' @return A character vector of argument names.
#' @keywords internal
#' @noRd
tl_per_row_args <- function() {
  c("weights", "subset", "offset", "foldid", "strata")
}

#' Names of a list, with "" for unnamed elements
#'
#' \code{names()} is NULL when no element is named, and indexing with
#' \code{!NULL \%in\% x} would then select nothing at all.
#'
#' @param x A list.
#' @return A character vector the length of \code{x}.
#' @keywords internal
#' @noRd
names2 <- function(x) {
  nms <- names(x)
  if (is.null(nms)) rep("", length(x)) else nms
}

#' Names for models compared side by side
#'
#' The comparisons key their rows on these names, so two models sharing
#' one were merged: the plot drew both bars at the same position and the
#' table pivoted them into list cells. The default label is the method and
#' task, which two models of one method share, so repeats are numbered.
#'
#' @param models List of tidylearn models.
#' @param names Caller-supplied names, or NULL.
#' @param label Function from a model to its default label.
#' @return A character vector of unique names, one per model.
#' @keywords internal
#' @noRd
tl_comparison_names <- function(models, names, label) {
  if (is.null(names)) {
    names <- vapply(models, label, character(1))
    repeated <- names %in% names[duplicated(names)]
    names[repeated] <- paste0(
      names[repeated], " #",
      stats::ave(seq_along(names), names, FUN = seq_along)[repeated]
    )
    return(names)
  }

  if (length(names) != length(models)) {
    stop("Length of 'names' (", length(names), ") must match the number ",
         "of models (", length(models), ")", call. = FALSE)
  }
  if (anyNA(names) || !is.character(names)) {
    stop("'names' must be a character vector with no missing values",
         call. = FALSE)
  }
  if (anyDuplicated(names)) {
    stop("'names' must be unique; the comparison keys each model on its ",
         "name. Repeated: ",
         paste(unique(names[duplicated(names)]), collapse = ", "),
         call. = FALSE)
  }
  names
}

#' Call a model-frame fitting function with arguments that are values
#'
#' \code{lm()}, \code{glm()} and \code{rpart()} evaluate \code{weights},
#' \code{subset} and \code{offset} inside the data, with the formula's
#' environment as the fallback. Forwarded through \code{...} that lookup
#' fails with "..1 used in an incorrect context", and a local variable is
#' not in the formula's environment either. \code{do.call()} places the
#' values themselves in the call, which is what those functions can
#' evaluate.
#'
#' The stored call then holds every value literally, so \code{print()} on
#' the fit would spill the whole frame and weight vector. The data, case
#' weights, subset and control list are held by \code{tl_hold_call_args()}
#' instead, and the function name replaces the function object.
#' \code{family} is left to the caller: as a bare symbol it resolves to
#' \code{stats::family()} when \code{update()} or \code{step()} re-runs the
#' call.
#'
#' A vector \code{offset} is refused. It would fit, but \code{predict()}
#' evaluates the stored \code{offset} in the new data, finds
#' \code{stats::offset()} and fails. Only \code{lm()} and \code{glm()} apply
#' an \code{offset()} term of the formula at prediction, so only their
#' refusal points there; the other functions routed here would drop it.
#'
#' @param fun The fitting function.
#' @param fun_name Its name, for the stored call.
#' @param args Named list of arguments.
#' @return The fitted object.
#' @keywords internal
#' @noRd
tl_fit_by_value <- function(fun, fun_name, args) {
  if ("offset" %in% names(args)) {
    if (fun_name %in% c("lm", "glm")) {
      stop(
        "Pass an offset in the formula, as offset(<column>), rather than ",
        "as the 'offset' argument.\nAn argument offset cannot be applied to ",
        "new data at predict().",
        call. = FALSE
      )
    }
    stop(
      fun_name, "() cannot apply an offset to new data at predict(), as ",
      "an argument or in the formula. For a model with an offset, fit ",
      "method = \"linear\" or \"logistic\" with offset(<column>) in the ",
      "formula.",
      call. = FALSE
    )
  }

  fit <- do.call(fun, args)
  if (is.list(fit) && is.call(fit$call)) {
    fit$call[[1]] <- tl_call_head(fun_name, fun)
    fit <- tl_hold_call_args(fit, c("data", "weights", "subset", "control"))
  }
  fit
}

#' Push overlapping label positions apart
#'
#' Labels on a coefficient path are placed at the value each line ends
#' on, and two coefficients can be arbitrarily close: on `mtcars` two of
#' the five sit within 0.02 of each other and print on top of one
#' another. This walks the values in order and lifts any that are nearer
#' than \code{min_gap}, then re-centres the block so the labels stay over
#' the span they started in rather than drifting upward.
#'
#' The result is a drawing position, not a value. Callers keep the true
#' coefficient for the line itself.
#'
#' @param y Numeric positions, in any order.
#' @param min_gap Minimum separation to enforce.
#' @return \code{y}, adjusted, in the order it was given.
#' @keywords internal
#' @noRd
tl_spread_labels <- function(y, min_gap) {
  if (length(y) < 2L || !is.finite(min_gap) || min_gap <= 0) {
    return(y)
  }
  ord <- order(y)
  spread <- y[ord]
  for (i in seq_along(spread)[-1]) {
    if (spread[i] - spread[i - 1L] < min_gap) {
      spread[i] <- spread[i - 1L] + min_gap
    }
  }
  spread <- spread + (mean(range(y)) - mean(range(spread)))
  y[ord] <- spread
  y
}
