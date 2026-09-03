#' @title Coefficient Inference for tidylearn
#' @name tidylearn-coefficients
#' @description Coefficients, standard errors and confidence intervals as a
#'   tibble, for the methods that have them. \code{tl_table_coefficients()}
#'   formats the same numbers as a \code{gt} table.
NULL

#' Model coefficients as a tibble
#'
#' Returns the coefficients of a fitted tidylearn model as a tibble, with
#' standard errors, test statistics, p-values and -- on request -- confidence
#' intervals. Available for \code{"linear"}, \code{"polynomial"},
#' \code{"logistic"}, \code{"ridge"}, \code{"lasso"} and
#' \code{"elastic_net"}. Other methods have no coefficients; use
#' \code{\link{tl_table_importance}} for those.
#'
#' Intervals are Wald intervals, computed from the same standard errors as
#' the \code{statistic} and \code{p_value} columns beside them, so the
#' interval and the p-value always agree about whether zero is excluded. For
#' \code{"linear"} and \code{"polynomial"} that means \emph{t} quantiles on
#' the residual degrees of freedom, which is exactly what
#' \code{\link[stats]{confint}} returns for an \code{lm}. For
#' \code{"logistic"} it means \emph{z} quantiles, which is what the reported
#' \emph{z} statistic implies but not what \code{confint()} gives -- that
#' profiles the likelihood, which is the better interval when the sample is
#' small or a class is nearly separated. Call
#' \code{stats::confint(model$fit)} when you want it.
#'
#' A rank-deficient fit -- two perfectly collinear predictors, or a factor
#' level with no observations -- cannot estimate every term. Those terms are
#' returned with an \code{NA} estimate rather than dropped, so a term named
#' in the formula never disappears from the output without saying so.
#'
#' @param model A tidylearn supervised model object from
#'   \code{\link{tl_model}}.
#' @param conf_int Whether to add \code{conf_low} and \code{conf_high}
#'   columns (default \code{FALSE}). Not available for regularised methods.
#' @param level Confidence level for the interval (default 0.95). Ignored
#'   unless \code{conf_int = TRUE}.
#' @param exponentiate Whether to report \code{estimate} and the interval on
#'   the odds scale rather than the log-odds scale (default \code{FALSE}).
#'   Only meaningful for a classification model, whose coefficients are log
#'   odds. The standard error stays on the log-odds scale it was computed on
#'   and is renamed \code{std_error_log} to say so.
#' @param lambda For regularised methods: \code{"1se"} (default),
#'   \code{"min"}, or a numeric penalty value.
#' @param ... Additional arguments (currently unused).
#' @return A tibble, one row per model term. For \code{"linear"},
#'   \code{"polynomial"} and \code{"logistic"}: \code{term},
#'   \code{estimate}, \code{std_error}, \code{statistic}, \code{p_value},
#'   plus \code{conf_low} and \code{conf_high} when \code{conf_int = TRUE}.
#'   For regularised methods: \code{term}, \code{estimate} and the
#'   \code{lambda} the estimate came from -- glmnet reports no standard
#'   errors, so there is nothing to test or bound.
#' @seealso \code{\link{tl_table_coefficients}} for the same numbers as a
#'   formatted table.
#' @export
#' @examples
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_coefficients(model)
#' tl_coefficients(model, conf_int = TRUE)
#'
#' # Odds ratios from a logistic fit
#' am_data <- transform(mtcars, am = factor(am))
#' model <- tl_model(am_data, am ~ wt, method = "logistic")
#' tl_coefficients(model, conf_int = TRUE, exponentiate = TRUE)
tl_coefficients <- function(model, conf_int = FALSE, level = 0.95,
                            exponentiate = FALSE, lambda = "1se", ...) {
  if (!inherits(model, "tidylearn_model")) {
    stop("'model' must be a tidylearn_model object", call. = FALSE)
  }

  if (inherits(model, "tidylearn_unsupervised")) {
    method <- model$spec$method
    stop(
      "'", method, "' is unsupervised and has no coefficients. ",
      if (method == "pca") {
        "Use get_pca_loadings() for the component loadings."
      } else {
        "Use the tidy_*() function for this method to get its components."
      },
      call. = FALSE
    )
  }

  if (!is.logical(conf_int) || length(conf_int) != 1L || is.na(conf_int)) {
    stop("'conf_int' must be TRUE or FALSE", call. = FALSE)
  }
  if (!is.logical(exponentiate) || length(exponentiate) != 1L ||
        is.na(exponentiate)) {
    stop("'exponentiate' must be TRUE or FALSE", call. = FALSE)
  }
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
        level <= 0 || level >= 1) {
    stop("'level' must be a single number strictly between 0 and 1, ",
         "such as 0.95", call. = FALSE)
  }

  method <- model$spec$method

  # Exponentiating turns a log-odds coefficient into an odds ratio. On a
  # regression coefficient it produces a number with no interpretation, so
  # refuse rather than return one.
  if (exponentiate && !isTRUE(model$spec$is_classification)) {
    stop(
      "'exponentiate' reports odds ratios, which needs coefficients on the ",
      "log-odds scale.\n'", method, "' models ", model$spec$response_var,
      " on its own scale here, so exponentiating it\nwould not mean anything.",
      call. = FALSE
    )
  }

  if (method %in% c("linear", "polynomial", "logistic")) {
    tl_coef_summary(model, conf_int, level, exponentiate)
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    if (conf_int) {
      stop(
        "Confidence intervals are not available for '", method, "'. glmnet ",
        "reports no standard\nerrors, and the shrinkage that produced these ",
        "estimates biases them toward zero,\nso a Wald interval built on ",
        "them would not cover at its stated rate. Fit\nmethod = ",
        if (isTRUE(model$spec$is_classification)) "'logistic'" else "'linear'",
        " if the interval is what you need.",
        call. = FALSE
      )
    }
    tl_coef_regularized(model, lambda, exponentiate)
  } else {
    stop(
      "'", method, "' has no coefficients. Use tl_table_importance() for ",
      "variable importance,\nor reach the fitted object through model$fit.",
      call. = FALSE
    )
  }
}

#' Coefficients from an lm or glm fit
#'
#' Aliased terms are carried through as NA rather than dropped.
#' \code{summary()} omits them, which turns an unestimable term into a
#' missing row -- and a missing row in a coefficient table reads as a term
#' that was never in the model.
#'
#' @param model A tidylearn model wrapping an lm or glm fit.
#' @param conf_int Whether to add interval columns.
#' @param level Confidence level.
#' @param exponentiate Whether to report on the odds scale.
#' @return A tibble of coefficients.
#' @keywords internal
#' @noRd
tl_coef_summary <- function(model, conf_int, level, exponentiate) {
  fit <- model$fit
  coef_mat <- summary(fit)$coefficients
  estimates <- stats::coef(fit)

  coef_tbl <- tibble::tibble(
    term = names(estimates),
    estimate = unname(estimates),
    std_error = NA_real_,
    statistic = NA_real_,
    p_value = NA_real_
  )

  estimated <- match(rownames(coef_mat), coef_tbl$term)
  if (anyNA(estimated)) {
    stop("could not match every summary() row to a model term. ",
         "Please report this with a reproducible example.", call. = FALSE)
  }
  coef_tbl$std_error[estimated] <- coef_mat[, 2]
  coef_tbl$statistic[estimated] <- coef_mat[, 3]
  coef_tbl$p_value[estimated] <- coef_mat[, 4]

  if (conf_int) {
    # z for logistic, where summary() reports a z statistic because the
    # binomial dispersion is fixed at 1; t on the residual df otherwise,
    # which reproduces confint() on the lm exactly.
    alpha <- 1 - level
    crit <- if (model$spec$method == "logistic") {
      stats::qnorm(1 - alpha / 2)
    } else {
      stats::qt(1 - alpha / 2, df = stats::df.residual(fit))
    }
    coef_tbl <- coef_tbl %>%
      dplyr::mutate(
        conf_low = .data$estimate - crit * .data$std_error,
        conf_high = .data$estimate + crit * .data$std_error
      ) %>%
      dplyr::relocate("conf_low", "conf_high", .after = "std_error")
  }

  if (exponentiate) coef_tbl <- tl_coef_exponentiate(coef_tbl)

  coef_tbl
}

#' Coefficients from a glmnet fit at one penalty
#'
#' @param model A tidylearn model wrapping a glmnet fit.
#' @param lambda "1se", "min", or a numeric penalty.
#' @param exponentiate Whether to report on the odds scale.
#' @return A tibble of coefficients.
#' @keywords internal
#' @noRd
tl_coef_regularized <- function(model, lambda, exponentiate) {
  fit <- model$fit

  lambda_val <- if (identical(lambda, "1se")) {
    attr(fit, "lambda_1se")
  } else if (identical(lambda, "min")) {
    attr(fit, "lambda_min")
  } else if (is.numeric(lambda) && length(lambda) == 1L && !is.na(lambda) &&
               lambda >= 0) {
    lambda
  } else {
    stop("'lambda' must be \"1se\", \"min\", or a single non-negative ",
         "number", call. = FALSE)
  }

  # coef(fit, s = NULL) returns the whole penalty path rather than failing,
  # and one column per lambda flattens into a vector of the wrong length
  # against the term names. Refuse instead of returning that.
  if (is.null(lambda_val) || !is.numeric(lambda_val) ||
        length(lambda_val) != 1L || is.na(lambda_val)) {
    stop("this model carries no '", lambda, "' penalty, so there is no ",
         "single set of\ncoefficients to return. Pass lambda = <a number> ",
         "to pick one.", call. = FALSE)
  }

  coefs <- as.matrix(stats::coef(fit, s = lambda_val))
  if (ncol(coefs) != 1L) {
    stop("expected coefficients at one penalty, got ", ncol(coefs),
         " columns. Please report this with a reproducible example.",
         call. = FALSE)
  }

  coef_tbl <- tibble::tibble(
    term = rownames(coefs),
    estimate = as.vector(coefs),
    lambda = lambda_val
  )

  if (exponentiate) coef_tbl <- tl_coef_exponentiate(coef_tbl)

  coef_tbl
}

#' Move a coefficient table to the odds scale
#'
#' The standard error is not exponentiated, because exp() of a standard
#' error is not the standard error of exp(estimate). It is renamed so a
#' log-odds error is never read as an error on the odds ratio beside it.
#'
#' @param coef_tbl A coefficient tibble.
#' @return The tibble with estimate and interval exponentiated.
#' @keywords internal
#' @noRd
tl_coef_exponentiate <- function(coef_tbl) {
  exp_cols <- intersect(c("estimate", "conf_low", "conf_high"),
                        names(coef_tbl))
  coef_tbl <- coef_tbl %>%
    dplyr::mutate(dplyr::across(dplyr::all_of(exp_cols), exp))

  if ("std_error" %in% names(coef_tbl)) {
    coef_tbl <- coef_tbl %>%
      dplyr::rename(std_error_log = "std_error")
  }

  coef_tbl
}
