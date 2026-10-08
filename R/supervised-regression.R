#' @title Regression Functions for tidylearn
#' @name tidylearn-regression
#' @description Linear and polynomial regression functionality
#' @importFrom stats lm predict poly model.matrix
#' @importFrom tibble tibble
#' @importFrom dplyr bind_cols
NULL

#' Fit a linear regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param ... Additional arguments to pass to lm()
#' @return A fitted linear regression model
#' @keywords internal
tl_fit_linear <- function(data, formula, ...) {
  # By value, so weights = <a vector> reaches lm() as a vector
  tl_fit_by_value(
    stats::lm, "lm",
    list(formula = formula, data = data, ...)
  )
}


#' Fit a polynomial regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param degree Degree of the polynomial (default: 2)
#' @param ... Additional arguments to pass to lm()
#' @return A fitted polynomial regression model
#' @keywords internal
tl_fit_polynomial <- function(data, formula, degree = 2, ...) {
  # Fit the polynomial model
  expansion <- tl_polynomial_formula(formula, data, degree)
  poly_model <- tl_fit_by_value(
    stats::lm, "lm",
    list(formula = expansion$formula, data = data, ...)
  )
  poly_model <- tl_polynomial_predvars(poly_model, expansion$predict_as)

  # Store original formula and degree for future reference
  attr(poly_model, "original_formula") <- formula
  attr(poly_model, "poly_degree") <- degree

  poly_model
}

#' Replace each numeric main effect with its polynomial
#'
#' The formula is edited rather than rebuilt. Pasting a new one together
#' from the response's variable name and the term labels fitted
#' \code{log(mpg) ~ wt} as a model of \code{mpg}, dropped \code{offset()}
#' terms and \code{- 1}, put a factor into \code{poly()} -- which fitted
#' it on its integer codes -- and wrapped an existing \code{poly(wt, 3)} in
#' a second one.
#'
#' A numeric term is a numeric vector or a one-column matrix such as
#' \code{scale(wt)}. One that is also part of an interaction keeps its own
#' term and gains \code{I(x^2)} up to \code{I(x^degree)}, so the
#' interaction is coded as written. Terms left as written: interactions,
#' factors and other non-numeric terms, \code{I()} terms, and bases such as
#' \code{poly()} or a spline's.
#'
#' @param formula The model formula.
#' @param data The training data, to expand a \code{.} and to tell which
#'   terms are numeric.
#' @param degree The polynomial degree.
#' @return A list: \code{formula}, the edited formula in the original
#'   formula's environment, and \code{predict_as}, for each added variable
#'   whose term depends on the training data, the variable as fitted and
#'   the same variable with the term as \code{stats::makepredictcall()}
#'   recomputes it, for \code{tl_polynomial_predvars()}.
#' @keywords internal
#' @noRd
tl_polynomial_formula <- function(formula, data, degree) {
  model_terms <- tl_terms(formula, data = data)
  # Expanded against the data, so a term inside `.` can be replaced
  expanded <- stats::formula(model_terms)
  env <- environment(formula)

  order <- attr(model_terms, "order")
  main_effects <- attr(model_terms, "term.labels")[order == 1L]
  values <- lapply(main_effects, function(label) {
    term <- str2lang(label)
    if (is.call(term) && identical(term[[1]], as.name("I"))) {
      return(NULL)
    }
    tryCatch(eval(term, data, env), error = function(e) NULL)
  })
  names(values) <- main_effects
  numeric_terms <- main_effects[vapply(values, function(value) {
    # A one-column matrix such as scale(wt) is a numeric term too. A
    # basis -- poly(), or a spline's -- already is the expansion.
    is.numeric(value) &&
      (is.null(dim(value)) ||
         (NCOL(value) == 1L && !inherits(value, c("poly", "basis"))))
  }, logical(1))]

  if (length(numeric_terms) == 0L) {
    return(list(formula = expanded, predict_as = list()))
  }

  # A term that is also part of an interaction keeps its own column, and
  # gains the powers above it. Replaced by poly(wt), it left cyl_f:wt
  # without its main effect, and model.matrix() then coded cyl_f in full:
  # one coefficient was always NA. The raw polynomial spans the same
  # columns, so the fit is the same model either way.
  factors <- attr(model_terms, "factors")
  in_interaction <- vapply(numeric_terms, function(label) {
    any(factors[label, order > 1L] != 0)
  }, logical(1))

  # The variables a term adds: its raw polynomial, or the powers above it
  # when it keeps its own column
  added <- function(term, label) {
    if (in_interaction[[label]]) {
      # As doubles, so the term reads wt^2 rather than wt^2L
      lapply(as.numeric(seq_len(degree)[-1]), function(power) {
        call("I", call("^", term, power))
      })
    } else {
      list(call("poly", term, degree = degree, raw = TRUE))
    }
  }

  # One update() for every term: done one term at a time, update() put the
  # variables of an interaction such as wt:hp in a new order and renamed
  # its coefficient
  edit <- quote(.)
  for (label in numeric_terms[!in_interaction]) {
    edit <- call("-", edit, str2lang(label))
  }
  predict_as <- list()
  for (label in numeric_terms) {
    term <- str2lang(label)
    fitted_as <- added(term, label)
    for (variable in fitted_as) {
      edit <- call("+", edit, variable)
    }

    # makepredictcall() gives the call that computes a data-dependent term
    # on new rows as it was computed on these: scale(wt) becomes
    # scale(wt, center = 3.217, scale = 0.978). model.frame() records that
    # for a variable it is handed, which here is the poly() or I() around
    # the term, and not for the term inside it.
    at_predict <- stats::makepredictcall(values[[label]], term)
    if (!identical(at_predict, term)) {
      predicted_as <- added(at_predict, label)
      for (k in seq_along(fitted_as)) {
        predict_as[[length(predict_as) + 1L]] <- list(
          fitted = fitted_as[[k]], predicted = predicted_as[[k]]
        )
      }
    }
  }
  edited <- stats::update(expanded, call("~", edit))
  environment(edited) <- env
  list(formula = edited, predict_as = predict_as)
}

#' Recompute an expanded term on new data as it was in training
#'
#' \code{predict.lm()} evaluates each variable through the predvars the
#' fit's terms carry. For \code{poly(scale(wt), degree = 2, raw = TRUE)}
#' those held the variable as written, so \code{scale()} centred and
#' scaled whatever rows \code{predict()} was handed: one row predicted
#' \code{NaN}, and five rows predicted differently from the same rows
#' among the training data. Each such variable's predvars entry is
#' replaced by the one \code{tl_polynomial_formula()} built around the
#' training-time term. The variables keep their names, so the
#' coefficients do too.
#'
#' @param fit The lm fit of the edited formula.
#' @param predict_as The \code{predict_as} list from
#'   \code{tl_polynomial_formula()}.
#' @return \code{fit}, its terms and its model frame's terms patched.
#' @keywords internal
#' @noRd
tl_polynomial_predvars <- function(fit, predict_as) {
  if (length(predict_as) == 0L) {
    return(fit)
  }

  model_terms <- fit$terms
  variables <- as.list(attr(model_terms, "variables"))[-1L]
  predvars <- attr(model_terms, "predvars")
  for (swap in predict_as) {
    # A raw poly() and an I() have nothing of their own to record, so the
    # entry model.frame() wrote is the variable as fitted
    at <- which(vapply(variables, identical, logical(1), swap$fitted))
    for (i in at) {
      predvars[[i + 1L]] <- swap$predicted
    }
  }
  attr(model_terms, "predvars") <- predvars

  fit$terms <- model_terms
  # model = FALSE leaves no frame to patch
  if (!is.null(fit$model)) {
    attr(fit$model, "terms") <- model_terms
  }
  fit
}


#' Plot diagnostics for a regression model
#'
#' @param model A tidylearn regression model object
#' @param which Which plots to create (1:4)
#' @param ... Additional arguments
#' @return A ggplot object (or list of ggplot objects)
#' @importFrom ggplot2 ggplot aes geom_point geom_smooth
#'   geom_hline geom_text
#' @importFrom ggplot2 labs theme_minimal scale_color_gradient
#'   stat_qq stat_qq_line
#' @keywords internal
tl_plot_diagnostics <- function(model, which = 1:4, ...) {
  # Leverage, Cook's distance and standardised residuals are lm and glm
  # quantities. Other fits failed inside rstandard() with a dispatch error
  # that named neither the plot nor the method. The wording is the one
  # plot() uses for the same refusal.
  if (!inherits(model$fit, "lm")) {
    stop(
      "Diagnostic plots need a model fitted by lm() or glm() -- method ",
      "\"linear\", \"polynomial\" or \"logistic\" -- but this is a \"",
      model$spec$method, "\" model. Use type = \"actual_predicted\" or ",
      "\"residuals\" instead.",
      call. = FALSE
    )
  }

  # Get residuals and fitted values
  fitted_vals <- fitted(model$fit)
  residuals <- residuals(model$fit)
  std_residuals <- rstandard(model$fit)

  # Create data frame for plotting
  plot_data <- tibble::tibble(
    fitted = fitted_vals,
    residuals = residuals,
    std_residuals = std_residuals,
    abs_residuals = abs(residuals),
    sqrt_abs_residuals = sqrt(abs(residuals)),
    leverage = hatvalues(model$fit),
    cooks_distance = cooks.distance(model$fit)
  )

  plots <- list()

  # Plot 1: Residuals vs Fitted
  if (1 %in% which) {
    p1 <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = fitted, y = residuals)
    ) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(
        yintercept = 0, linetype = "dashed",
        color = "red"
      ) +
      ggplot2::geom_smooth(
        se = FALSE, color = "blue", method = "loess"
      ) +
      ggplot2::labs(
        title = "Residuals vs Fitted",
        x = "Fitted values",
        y = "Residuals"
      ) +
      ggplot2::theme_minimal()

    plots[["residuals_vs_fitted"]] <- p1
  }

  # Plot 2: Normal Q-Q
  if (2 %in% which) {
    p2 <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(sample = std_residuals)
    ) +
      ggplot2::stat_qq() +
      ggplot2::stat_qq_line(color = "red") +
      ggplot2::labs(
        title = "Normal Q-Q",
        x = "Theoretical Quantiles",
        y = "Standardized Residuals"
      ) +
      ggplot2::theme_minimal()

    plots[["qq"]] <- p2
  }

  # Plot 3: Scale-Location
  if (3 %in% which) {
    p3 <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = fitted, y = sqrt_abs_residuals)
    ) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_smooth(
        se = FALSE, color = "blue", method = "loess"
      ) +
      ggplot2::labs(
        title = "Scale-Location",
        x = "Fitted values",
        y = "sqrt(|Standardized residuals|)"
      ) +
      ggplot2::theme_minimal()

    plots[["scale_location"]] <- p3
  }

  # Plot 4: Cook's distance / Leverage
  if (4 %in% which) {
    p4 <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = leverage, y = std_residuals)
    ) +
      ggplot2::geom_point(
        ggplot2::aes(
          size = cooks_distance,
          color = cooks_distance
        ),
        alpha = 0.6
      ) +
      ggplot2::geom_hline(
        yintercept = c(-2, 0, 2),
        linetype = "dashed", color = "red"
      ) +
      ggplot2::scale_color_gradient(
        low = "blue", high = "red"
      ) +
      ggplot2::labs(
        title = "Residuals vs Leverage",
        x = "Leverage",
        y = "Standardized residuals",
        size = "Cook's distance",
        color = "Cook's distance"
      ) +
      ggplot2::theme_minimal()

    plots[["residuals_vs_leverage"]] <- p4
  }

  # Return a single plot or list of plots
  if (length(plots) == 1) {
    plots[[1]]
  } else {
    plots
  }
}

#' Plot actual vs predicted values for a regression model
#'
#' @param model A tidylearn regression model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_abline
#'   labs theme_minimal
#' @keywords internal
tl_plot_actual_predicted <- function(model,
                                     new_data = NULL,
                                     ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual and predicted values, the actuals on the scale the model was
  # fitted on: the raw column put mpg against predictions of log(mpg), and
  # the correlation in the subtitle compared the two scales
  actuals <- tl_observed_response(model, new_data)
  predictions <- unname(predict(model, new_data)$.pred)

  # A single missing value turned both statistics into NA
  complete <- !is.na(actuals) & !is.na(predictions)
  if (!all(complete)) {
    warning(
      sum(!complete), " row(s) with a missing response or prediction are ",
      "left out of the plot.",
      call. = FALSE
    )
  }

  # Create data frame for plotting
  plot_data <- tibble::tibble(
    actual = actuals[complete],
    predicted = predictions[complete]
  )

  # Calculate correlation
  corr <- round(cor(plot_data$actual, plot_data$predicted), 3)
  r_squared <- round(cor(plot_data$actual, plot_data$predicted)^2, 3)

  # Create the plot
  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = actual, y = predicted)
  ) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::geom_abline(
      intercept = 0, slope = 1,
      color = "red", linetype = "dashed"
    ) +
    ggplot2::labs(
      title = "Actual vs Predicted Values",
      subtitle = paste0(
        "Correlation: ", corr,
        ", R-squared: ", r_squared
      ),
      x = "Actual values",
      y = "Predicted values"
    ) +
    ggplot2::theme_minimal()

  p
}

#' Plot residuals for a regression model
#'
#' @param model A tidylearn regression model object
#' @param type Type of residual plot: "fitted" (default),
#'   "histogram", "predicted"
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_hline
#'   geom_histogram labs theme_minimal
#' @keywords internal
tl_plot_residuals <- function(model, type = "fitted", ...) {
  # Get residuals and fitted values
  plot_data <- tl_fit_residuals(model)

  # Create the plot based on type
  if (type == "fitted") {
    p <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = fitted, y = residuals)
    ) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(
        yintercept = 0, linetype = "dashed",
        color = "red"
      ) +
      ggplot2::labs(
        title = "Residuals vs Fitted Values",
        x = "Fitted values",
        y = "Residuals"
      ) +
      ggplot2::theme_minimal()
  } else if (type == "histogram") {
    p <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = residuals)
    ) +
      ggplot2::geom_histogram(
        bins = 30, fill = "steelblue",
        color = "white", alpha = 0.7
      ) +
      ggplot2::labs(
        title = "Histogram of Residuals",
        x = "Residuals",
        y = "Count"
      ) +
      ggplot2::theme_minimal()
  } else if (type == "predicted") {
    # The predictions for the rows the fit used are its fitted values.
    # Predicting model$data again covered every row, one more than the
    # residuals whenever lm() had dropped one with a missing value.
    plot_data$predicted <- plot_data$fitted

    p <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = predicted, y = residuals)
    ) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::geom_hline(
        yintercept = 0, linetype = "dashed",
        color = "red"
      ) +
      ggplot2::labs(
        title = "Residuals vs Predicted Values",
        x = "Predicted values",
        y = "Residuals"
      ) +
      ggplot2::theme_minimal()
  } else {
    stop(
      "Invalid plot type. ",
      "Use 'fitted', 'histogram', or 'predicted'.",
      call. = FALSE
    )
  }

  p
}

#' Fitted values and residuals for the residual plots
#'
#' An lm or glm fit carries its own. Other fits -- glmnet, trees, forests
#' and the rest -- do not: \code{fitted()} on a glmnet fit is \code{NULL},
#' so the plot was returned and failed when printed. Their residuals come
#' from predictions on the training data instead, against the response on
#' the scale it was fitted on.
#'
#' @param model A tidylearn model.
#' @return A tibble of \code{fitted} and \code{residuals}, one row per
#'   training row the fit could use.
#' @keywords internal
#' @noRd
tl_fit_residuals <- function(model) {
  fit <- model$fit
  if (inherits(fit, "lm")) {
    return(tibble::tibble(
      fitted = unname(stats::fitted(fit)),
      residuals = unname(stats::residuals(fit))
    ))
  }

  if (isTRUE(model$spec$is_classification)) {
    stop(
      "Residual plots are for regression models, and this \"",
      model$spec$method, "\" model is a classifier. Use type = \"confusion\" ",
      "or \"roc\" instead.",
      call. = FALSE
    )
  }

  actual <- tl_observed_response(model, model$data)
  predicted <- unname(predict(model, model$data)$.pred)
  # Without the rows the fit itself dropped for a missing value
  used <- !is.na(actual) & !is.na(predicted)
  tibble::tibble(
    fitted = predicted[used],
    residuals = actual[used] - predicted[used]
  )
}

#' Create confidence and prediction interval plots
#'
#' @param model A tidylearn regression model object
#' @param new_data Optional data frame for prediction
#'   (if NULL, uses training data)
#' @param level Confidence level (default: 0.95)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @importFrom ggplot2 ggplot aes geom_point geom_ribbon
#'   labs theme_minimal
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt, method = "linear")
#' tl_plot_intervals(model)
#' }
#' @export
tl_plot_intervals <- function(model,
                              new_data = NULL,
                              level = 0.95, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
  }

  if (model$spec$is_classification) {
    stop(
      "Interval plots are only available ",
      "for regression models",
      call. = FALSE
    )
  }

  # The intervals come from predict.lm(). glmnet, trees and the other
  # methods have none, and passing interval = to them failed inside the
  # backend -- glmnet asked for a newx argument the caller never had.
  if (!inherits(model$fit, "lm") || inherits(model$fit, "glm")) {
    stop(
      "Interval plots need a \"linear\" or \"polynomial\" model, whose lm() ",
      "fit gives\nconfidence and prediction intervals. This is a \"",
      model$spec$method, "\" model.",
      call. = FALSE
    )
  }

  # The first predictor of the expanded formula: all.vars() on y ~ . gives
  # "." itself, which is not a column
  x_var <- get_formula_vars(model$spec$formula, model$data)[1]
  if (is.na(x_var)) {
    stop("Interval plots need a predictor to plot against, and ",
         deparse1(model$spec$formula), " has none.", call. = FALSE)
  }
  y_label <- deparse1(model$spec$formula[[2]])

  # Sort data by x variable for smooth curves
  sorted_data <- new_data[order(new_data[[x_var]]), , drop = FALSE]

  # Calculate confidence and prediction intervals from the raw model
  conf_int <- stats::predict(model$fit, newdata = sorted_data,
                             interval = "confidence", level = level)
  pred_int <- stats::predict(model$fit, newdata = sorted_data,
                             interval = "prediction", level = level)

  # Create plot data
  plot_data <- tibble::tibble(
    x = sorted_data[[x_var]],
    pred = conf_int[, "fit"],
    conf_lower = conf_int[, "lwr"],
    conf_upper = conf_int[, "upr"],
    pred_lower = pred_int[, "lwr"],
    pred_upper = pred_int[, "upr"]
  )
  # The observed points on the scale the bands are on: the raw column drew
  # mpg against bands for log(mpg). Data to predict on may not carry the
  # response, and then the bands are drawn alone.
  # Every column the response is computed from, not only the first: mpg
  # alone cannot give I(mpg / wt).
  response_vars <- all.vars(model$spec$formula[[2L]])
  observed <- if (all(response_vars %in% names(sorted_data))) {
    tl_observed_response(model, sorted_data)
  }
  if (!is.null(observed)) {
    plot_data$y <- observed
  }

  # A row with no prediction has nothing to draw in any layer, and ggplot
  # warned about it once per layer when the plot was printed
  predicted <- !is.na(plot_data$x) & !is.na(plot_data$pred)
  if (!all(predicted)) {
    warning(
      sum(!predicted), " row(s) with a missing predictor value are left out ",
      "of the plot.",
      call. = FALSE
    )
    plot_data <- plot_data[predicted, , drop = FALSE]
  }

  # Create the plot
  p <- ggplot2::ggplot(
    plot_data, ggplot2::aes(x = x)
  ) +
    # Prediction intervals (wider)
    ggplot2::geom_ribbon(
      ggplot2::aes(
        ymin = pred_lower, ymax = pred_upper
      ),
      fill = "lightblue", alpha = 0.3
    ) +
    # Confidence intervals (narrower)
    ggplot2::geom_ribbon(
      ggplot2::aes(
        ymin = conf_lower, ymax = conf_upper
      ),
      fill = "steelblue", alpha = 0.5
    ) +
    # Fitted line
    ggplot2::geom_line(
      ggplot2::aes(y = pred),
      color = "blue", linewidth = 1
    )
  if (!is.null(observed)) {
    # Actual points. A row missing only its response keeps its band.
    p <- p + ggplot2::geom_point(
      data = plot_data[!is.na(plot_data$y), , drop = FALSE],
      ggplot2::aes(y = .data$y), alpha = 0.6
    )
  }
  p <- p +
    ggplot2::labs(
      title = paste0(
        "Regression with ", level * 100,
        "% Confidence & Prediction Intervals"
      ),
      subtitle = paste0(
        "Dark band: Confidence interval, ",
        "Light band: Prediction interval"
      ),
      x = x_var,
      y = y_label
    ) +
    ggplot2::theme_minimal()

  p
}
