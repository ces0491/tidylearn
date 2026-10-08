#' @title Advanced Diagnostics Functions for tidylearn
#' @name tidylearn-diagnostics
#' @description Functions for advanced model diagnostics,
#'   assumption checking, and outlier detection
#' @importFrom stats influence.measures cooks.distance hatvalues dffits dfbetas
#' @importFrom stats lm.influence rstudent rstandard
#' @importFrom stats shapiro.test bartlett.test kruskal.test
#' @importFrom dplyr filter select mutate arrange
#' @importFrom ggplot2 ggplot aes geom_point geom_text labs theme_minimal
NULL

#' Calculate influence measures for a linear model
#'
#' @param model A tidylearn model object
#' @param threshold_cook Cook's distance threshold (default: 4/n)
#' @param threshold_leverage Leverage threshold (default: 2*(p+1)/n)
#' @param threshold_dffits DFFITS threshold (default: 2*sqrt((p+1)/n))
#' @return A data frame with one row per observation containing influence
#'   measures: \code{cooks_distance}, \code{leverage}, \code{dffits},
#'   \code{std_residual}, \code{stud_residual}, boolean flags for each
#'   threshold (\code{is_cook_influential}, \code{is_leverage_influential},
#'   \code{is_dffits_influential}, \code{is_outlier}), per-coefficient
#'   \code{dfbetas_*} columns, and an overall \code{is_influential} flag.
#'   Threshold values are stored as attributes.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_influence_measures(model)
#' }
#' @export
tl_influence_measures <- function(model, threshold_cook = NULL,
                                  threshold_leverage = NULL,
                                  threshold_dffits = NULL) {
  # cooks.distance(), hatvalues(), dffits() and rstandard() have no
  # glmnet methods, so the penalised fits cannot be supported here even
  # though they are linear-based
  supported_methods <- c("linear", "logistic", "polynomial")

  if (!inherits(model, "tidylearn_model")) {
    stop(
      "Influence measures are only available for linear-based models",
      call. = FALSE
    )
  }

  if (model$spec$method %in% c("ridge", "lasso", "elastic_net")) {
    stop(
      "Influence measures are not available for penalised regression ",
      "('", model$spec$method, "'): glmnet provides no hat values or ",
      "leave-one-out diagnostics. Refit with method = \"linear\" or ",
      "\"logistic\" to diagnose the unpenalised specification.",
      call. = FALSE
    )
  }

  if (!model$spec$method %in% supported_methods) {
    stop(
      "Influence measures are only available for linear-based models",
      call. = FALSE
    )
  }

  # Extract fitted model
  fit <- model$fit

  # Data dimensions. Count the rows the fit actually used, not the rows it
  # was handed: lm() drops incomplete cases, so one missing predictor left
  # every influence measure an observation shorter than model$data and the
  # data frame below failed with "arguments imply differing number of
  # rows: 60, 59".
  fitted_rows <- tl_fitted_rows(model)
  n <- length(fitted_rows)
  # Predictors excluding the intercept, counted from the rank: an aliased
  # coefficient is NA in coef() and was counted as a parameter
  p <- fit$rank - 1

  # Set default thresholds if not provided
  if (is.null(threshold_cook)) threshold_cook <- 4 / n
  if (is.null(threshold_leverage)) {
    threshold_leverage <- 2 * (p + 1) / n
  }
  if (is.null(threshold_dffits)) {
    threshold_dffits <- 2 * sqrt((p + 1) / n)
  }

  # Calculate influence measures
  cooks_d <- tl_drop_excluded(cooks.distance(fit), fit)
  leverage <- tl_drop_excluded(hatvalues(fit), fit)
  dffits_val <- tl_drop_excluded(dffits(fit), fit)
  dfbetas_val <- tl_drop_excluded(dfbetas(fit), fit)

  # Get standardized residuals
  std_resid <- tl_drop_excluded(rstandard(fit), fit)
  stud_resid <- tl_drop_excluded(rstudent(fit), fit)

  # Create data frame
  # Number the observations by their row in the data, so a dropped row
  # does not silently shift every later observation's label.
  influence_df <- data.frame(
    observation = fitted_rows,
    cooks_distance = cooks_d,
    leverage = leverage,
    dffits = dffits_val,
    std_residual = std_resid,
    stud_residual = stud_resid
  )

  # Add flags for influential observations
  influence_df$is_cook_influential <-
    influence_df$cooks_distance > threshold_cook
  influence_df$is_leverage_influential <-
    influence_df$leverage > threshold_leverage
  influence_df$is_dffits_influential <-
    abs(influence_df$dffits) > threshold_dffits
  influence_df$is_outlier <- abs(influence_df$std_residual) > 3

  # Add dfbetas as separate columns. dfbetas() has a column only for the
  # coefficients the fit estimated, so looping over coef() ran past its
  # last column on a rank-deficient fit.
  for (coef_name in colnames(dfbetas_val)) {
    col_name <- paste0("dfbetas_", gsub("[^[:alnum:]]", "_", coef_name))
    influence_df[[col_name]] <- dfbetas_val[, coef_name]
  }

  # Add summary column for overall influence
  influence_df$is_influential <- influence_df$is_cook_influential |
    influence_df$is_leverage_influential |
    influence_df$is_dffits_influential |
    influence_df$is_outlier

  # Add threshold values as attributes
  attr(influence_df, "threshold_cook") <- threshold_cook
  attr(influence_df, "threshold_leverage") <- threshold_leverage
  attr(influence_df, "threshold_dffits") <- threshold_dffits

  influence_df
}

#' Drop the rows na.exclude pads back in
#'
#' Under \code{na.action = na.exclude}, \code{residuals()}, \code{fitted()}
#' and the influence functions pad their result with NA at the rows the fit
#' dropped, so it is longer than the rows the fit used, which is what
#' \code{tl_fitted_rows()} counts. Under \code{na.omit} nothing is padded.
#'
#' @param x A vector, or a matrix with one row per observation
#' @param fit The lm or glm fit it was computed from
#' @return \code{x} without the padded rows
#' @keywords internal
#' @noRd
tl_drop_excluded <- function(x, fit) {
  omitted <- stats::na.action(fit)
  if (!inherits(omitted, "exclude")) {
    return(x)
  }
  dropped <- as.integer(omitted)
  if (is.matrix(x)) x[-dropped, , drop = FALSE] else x[-dropped]
}

#' Plot influence diagnostics
#'
#' @param model A tidylearn model object
#' @param plot_type Type of influence plot: "cook", "leverage", "index"
#' @param threshold_cook Cook's distance threshold (default: 4/n)
#' @param threshold_leverage Leverage threshold (default: 2*(p+1)/n)
#' @param threshold_dffits DFFITS threshold (default: 2*sqrt((p+1)/n))
#' @param n_labels Number of points to label (default: 3)
#' @param label_size Text size for labels (default: 3)
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_plot_influence(model, plot_type = "cook")
#' }
#' @export
tl_plot_influence <- function(model,
                              plot_type = "cook",
                              threshold_cook = NULL,
                              threshold_leverage = NULL,
                              threshold_dffits = NULL,
                              n_labels = 3,
                              label_size = 3) {
  # Get influence measures
  influence_df <- tl_influence_measures(
    model,
    threshold_cook = threshold_cook,
    threshold_leverage = threshold_leverage,
    threshold_dffits = threshold_dffits
  )

  # Get thresholds from attributes
  threshold_cook <- attr(influence_df, "threshold_cook")
  threshold_leverage <- attr(influence_df, "threshold_leverage")
  threshold_dffits <- attr(influence_df, "threshold_dffits")

  # Create plot based on type
  if (plot_type == "cook") {
    # Cook's distance plot
    # Identify top points to label
    n_to_label <- min(n_labels, nrow(influence_df))
    top_idx <- order(
      influence_df$cooks_distance,
      decreasing = TRUE
    )[1:n_to_label]
    influence_df$label <- ifelse(
      influence_df$observation %in% influence_df$observation[top_idx],
      as.character(influence_df$observation),
      ""
    )

    subtitle_text <- paste(
      "Threshold:",
      round(threshold_cook, 4),
      "- Points above are influential"
    )
    p <- ggplot2::ggplot(
      influence_df,
      ggplot2::aes(x = observation, y = cooks_distance)
    ) +
      ggplot2::geom_point(
        ggplot2::aes(color = is_cook_influential),
        size = 3,
        alpha = 0.7
      ) +
      ggplot2::geom_hline(
        yintercept = threshold_cook,
        linetype = "dashed",
        color = "red"
      ) +
      ggplot2::geom_text(
        ggplot2::aes(label = label),
        hjust = -0.3,
        vjust = 0.5,
        size = label_size
      ) +
      ggplot2::scale_color_manual(
        values = c("FALSE" = "blue", "TRUE" = "red"),
        drop = FALSE
      ) +
      ggplot2::labs(
        title = "Cook's Distance Plot",
        subtitle = subtitle_text,
        x = "Observation Index",
        y = "Cook's Distance",
        color = "Influential"
      ) +
      ggplot2::theme_minimal()

  } else if (plot_type == "leverage") {
    # Leverage-Residual plot (Bubble plot with Cook's distance)
    # Identify top points to label
    n_to_label <- min(n_labels, nrow(influence_df))
    top_idx <- order(
      influence_df$cooks_distance,
      decreasing = TRUE
    )[1:n_to_label]
    influence_df$label <- ifelse(
      influence_df$observation %in% influence_df$observation[top_idx],
      as.character(influence_df$observation),
      ""
    )

    p <- ggplot2::ggplot(
      influence_df,
      ggplot2::aes(
        x = leverage,
        y = std_residual,
        size = cooks_distance,
        color = is_influential
      )
    ) +
      ggplot2::geom_point(alpha = 0.7) +
      ggplot2::geom_hline(
        yintercept = c(-3, 0, 3),
        linetype = "dashed",
        color = c("red", "black", "red")
      ) +
      ggplot2::geom_vline(
        xintercept = threshold_leverage,
        linetype = "dashed",
        color = "red"
      ) +
      ggplot2::geom_text(
        ggplot2::aes(label = label),
        hjust = -0.3,
        vjust = 0.5,
        size = label_size
      ) +
      ggplot2::scale_color_manual(
        values = c("FALSE" = "blue", "TRUE" = "red"),
        drop = FALSE
      ) +
      ggplot2::labs(
        title = "Leverage-Residual Plot",
        subtitle = paste(
          "Leverage threshold:", round(threshold_leverage, 4),
          "- Residual threshold: +/-3"
        ),
        x = "Leverage (Hat Values)",
        y = "Standardized Residuals",
        size = "Cook's Distance",
        color = "Influential"
      ) +
      ggplot2::theme_minimal()

  } else if (plot_type == "index") {
    # Index plot of residuals
    # Identify top points to label
    n_to_label <- min(n_labels, nrow(influence_df))
    top_idx <- order(
      abs(influence_df$std_residual),
      decreasing = TRUE
    )[1:n_to_label]
    influence_df$label <- ifelse(
      influence_df$observation %in% influence_df$observation[top_idx],
      as.character(influence_df$observation),
      ""
    )

    p <- ggplot2::ggplot(
      influence_df,
      ggplot2::aes(x = observation, y = std_residual)
    ) +
      ggplot2::geom_point(
        ggplot2::aes(color = is_outlier),
        size = 3,
        alpha = 0.7
      ) +
      ggplot2::geom_hline(
        yintercept = c(-3, 0, 3),
        linetype = "dashed",
        color = c("red", "black", "red")
      ) +
      ggplot2::geom_text(
        ggplot2::aes(label = label),
        hjust = -0.3,
        vjust = 0.5,
        size = label_size
      ) +
      ggplot2::scale_color_manual(
        values = c("FALSE" = "blue", "TRUE" = "red"),
        drop = FALSE
      ) +
      ggplot2::labs(
        title = "Index Plot of Standardized Residuals",
        subtitle = "Points outside +/-3 are considered outliers",
        x = "Observation Index",
        y = "Standardized Residuals",
        color = "Outlier"
      ) +
      ggplot2::theme_minimal()
  } else {
    stop(
      "Invalid plot_type. Use 'cook', 'leverage', or 'index'.",
      call. = FALSE
    )
  }

  p
}

#' Test the linearity assumption
#'
#' A RESET-style specification test: regress the residuals on powers of
#' the fitted values and test whether they explain anything. If the
#' relationship is really linear, no function of the fitted values should
#' predict what the model left over.
#'
#' This replaces a check on \code{cor(fitted, residuals)}, which cannot
#' detect anything: for OLS with an intercept, residuals are orthogonal
#' to fitted values by construction, so that correlation is identically
#' zero however curved the data is.
#'
#' @param fitted_values Fitted values from the model
#' @param residuals Residuals from the model
#' @return A list with \code{assumption}, \code{check}, \code{details}
#'   and \code{recommendation}
#' @keywords internal
#' @noRd
tl_check_linearity <- function(fitted_values, residuals) {
  result <- function(check, details, recommendation) {
    list(
      assumption = "Linearity", check = check,
      details = details, recommendation = recommendation
    )
  }

  # Cubing raw fitted values overflows for large responses, so work on a
  # standardised scale
  scaled_fit <- as.vector(scale(fitted_values))

  # Distinct values are counted after rounding. A one-factor model such as
  # len ~ supp has two fitted values that differ in their last bits; read
  # as four, they let the test run on a fit it cannot judge, and it
  # reported a p-value of 1.
  if (anyNA(scaled_fit) || length(unique(round(scaled_fit, 8))) < 4) {
    return(result(
      NA,
      "Not enough distinct fitted values to test linearity",
      "Inspect a residuals-versus-fitted plot directly"
    ))
  }

  aux_data <- data.frame(
    .tl_resid = as.vector(residuals),
    .tl_fit2 = scaled_fit^2,
    .tl_fit3 = scaled_fit^3
  )

  aux <- try(
    stats::lm(.tl_resid ~ .tl_fit2 + .tl_fit3, data = aux_data),
    silent = TRUE
  )

  f_stat <- if (inherits(aux, "try-error")) NULL else summary(aux)$fstatistic

  if (is.null(f_stat)) {
    return(result(
      NA,
      "Linearity test could not be computed",
      "Inspect a residuals-versus-fitted plot directly"
    ))
  }

  p_value <- stats::pf(
    f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE
  )

  details <- paste(
    "RESET-style test on powers of the fitted values: p-value =",
    format.pval(p_value, digits = 4)
  )

  result(
    unname(p_value >= 0.05),
    details,
    if (p_value < 0.05) {
      "Consider non-linear transformations or polynomial terms"
    } else {
      "Linearity assumption appears satisfied"
    }
  )
}

#' An ordinary least squares assumption a logistic model does not make
#'
#' @param assumption The assumption's label
#' @param reason Why logistic regression does not assume it
#' @return An assumption entry whose \code{check} is NULL, so it is neither
#'   satisfied nor violated
#' @keywords internal
#' @noRd
tl_not_logistic_assumption <- function(assumption, reason) {
  list(
    assumption = assumption,
    check = NULL,
    details = paste0("Not an assumption of logistic regression: ", reason),
    recommendation = "No check needed for a logistic model"
  )
}

#' Check a fit for multicollinearity
#'
#' Uses \code{car::vif()} when it is installed and can run, and otherwise
#' the largest correlation between columns of the fit's design matrix.
#' \code{car::vif()} refuses a model with fewer than two terms, and one with
#' an aliased coefficient.
#'
#' @param fit An lm or glm fit
#' @param verbose Whether to say when VIF could not be computed
#' @return An assumption entry
#' @keywords internal
#' @noRd
tl_check_multicollinearity <- function(fit, verbose) {
  result <- function(check, details, recommendation) {
    list(
      assumption = "No Multicollinearity", check = check,
      details = details, recommendation = recommendation
    )
  }
  single <- result(
    TRUE, "Model has only one predictor",
    "Not applicable with a single predictor"
  )

  # Terms are counted from the fit: all.vars() on the formula counted `.`
  # as one predictor
  if (length(attr(stats::terms(fit), "term.labels")) < 2) {
    return(single)
  }

  vif_values <- NULL
  if (requireNamespace("car", quietly = TRUE)) {
    vif_values <- tryCatch(car::vif(fit), error = function(e) {
      if (verbose) {
        message("VIF calculation failed. Checking correlations instead.")
      }
      NULL
    })
  }

  if (!is.null(vif_values)) {
    # With a term of more than one df, car::vif() returns a table of GVIF,
    # Df and GVIF^(1/(2*Df)), and max() over the table read the Df column:
    # a six-level factor scored 5. GVIF^(1/Df), the square of the
    # size-adjusted GVIF, is on the scale of an ordinary VIF and equals it
    # for a one-df term.
    if (is.matrix(vif_values)) {
      vif_values <- vif_values[, "GVIF"]^(1 / vif_values[, "Df"])
    }
    max_vif <- max(vif_values)
    return(result(
      max_vif < 5,
      paste("Maximum VIF:", round(max_vif, 4)),
      if (max_vif >= 5) {
        paste(
          "Multicollinearity detected.",
          "Consider removing or combining highly correlated predictors."
        )
      } else {
        "No serious multicollinearity detected"
      }
    ))
  }

  design <- stats::model.matrix(fit)
  design <- design[, colnames(design) != "(Intercept)", drop = FALSE]
  if (ncol(design) < 2) {
    return(single)
  }
  cor_matrix <- suppressWarnings(stats::cor(design))
  max_cor <- max(abs(cor_matrix[upper.tri(cor_matrix)]), na.rm = TRUE)

  result(
    max_cor < 0.7,
    paste("Maximum correlation between predictors:", round(max_cor, 4)),
    if (max_cor >= 0.7) {
      paste(
        "Potential multicollinearity.",
        "Consider removing or combining correlated predictors."
      )
    } else {
      "No serious multicollinearity detected"
    }
  )
}

#' Check model assumptions
#'
#' @param model A tidylearn model object
#' @param test Logical; whether to perform statistical tests
#' @param verbose Logical; whether to print test results and explanations
#' @return A named list with one element per assumption checked
#'   (\code{linearity}, \code{independence}, \code{homoscedasticity},
#'   \code{normality}, \code{multicollinearity}, \code{outliers}), each
#'   containing \code{assumption} (character label), \code{check} (logical,
#'   \code{NA} when the test could not decide, or \code{NULL} when no test
#'   was run), \code{details} (character), and \code{recommendation}
#'   (character). An additional \code{overall} element summarises the
#'   number of assumptions checked, violated, and satisfied; an \code{NA} or
#'   \code{NULL} check counts as neither.
#'
#'   Logistic regression assumes neither normal residuals nor a constant
#'   variance, so for a logistic model \code{normality} and
#'   \code{homoscedasticity} have a \code{NULL} check and a note saying so.
#'   For a factor, multicollinearity is judged on \code{GVIF^(1/Df)}, the
#'   generalised VIF on the scale of an ordinary one.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_check_assumptions(model)
#' }
#' @export
tl_check_assumptions <- function(model, test = TRUE, verbose = TRUE) {
  # These checks all run off residuals(), fitted() and the influence
  # measures, none of which glmnet provides -- listing ridge/lasso/
  # elastic_net as supported produced a "no applicable method" error
  # partway through rather than an answer.
  supported <- c("linear", "logistic", "polynomial")
  penalised <- c("ridge", "lasso", "elastic_net")

  if (!inherits(model, "tidylearn_model")) {
    stop(
      "Assumption checking is only available for linear-based models",
      call. = FALSE
    )
  }

  if (model$spec$method %in% penalised) {
    stop(
      "Assumption checking is not available for penalised regression ",
      "('", model$spec$method, "'): glmnet provides no residuals, ",
      "hat values or influence measures. Refit with method = \"linear\" ",
      "or \"logistic\" to diagnose the unpenalised specification.",
      call. = FALSE
    )
  }

  if (!model$spec$method %in% supported) {
    stop(
      "Assumption checking is only available for linear-based models",
      call. = FALSE
    )
  }

  # Extract the fit. Under na.exclude its residuals and fitted values are
  # padded back to every row of the data, so the padding is dropped to
  # leave the rows the fit used.
  fit <- model$fit
  residuals <- tl_drop_excluded(residuals(fit), fit)
  fitted_values <- tl_drop_excluded(fitted(fit), fit)

  # A logistic model assumes neither normal residuals nor a constant
  # variance: the variance of a binary outcome is set by its mean. Testing
  # them reported violations that are not violations, with advice to
  # transform the response.
  is_logistic <- model$spec$method == "logistic"

  # Initialize results list
  assumptions <- list()

  # 1. Linearity
  assumptions$linearity <- tl_check_linearity(fitted_values, residuals)

  # 2. Independence
  # Durbin-Watson statistic for autocorrelation. It is computed directly:
  # car::durbinWatsonTest() also bootstraps a p-value, which drew from the
  # caller's random stream though only the statistic was read.
  if (test) {
    dw_statistic <- sum(diff(residuals)^2) / sum(residuals^2)
    dw_check <- if (is.finite(dw_statistic)) {
      dw_statistic >= 1.5 && dw_statistic <= 2.5
    } else {
      NA
    }
    dw_recommendation <- if (is.na(dw_check)) {
      "Inspect the residuals in observation order directly"
    } else if (!dw_check) {
      paste(
        "Possible autocorrelation in residuals.",
        "Check for time-series structure or clustering."
      )
    } else {
      "Independence assumption appears satisfied"
    }
    assumptions$independence <- list(
      assumption = "Independence",
      check = dw_check,
      details = paste("Durbin-Watson statistic:", round(dw_statistic, 4)),
      recommendation = dw_recommendation
    )
  } else {
    assumptions$independence <- list(
      assumption = "Independence",
      check = NULL,
      details = "No statistical test performed",
      recommendation = paste(
        "Ensure observations are independent.",
        "Consider the data collection process."
      )
    )
  }

  # 3. Homoscedasticity (Equal Variance)
  # Breusch-Pagan test
  if (is_logistic) {
    assumptions$homoscedasticity <- tl_not_logistic_assumption(
      "Homoscedasticity",
      "the variance of a binary outcome is set by its mean, p(1 - p)"
    )
  } else if (test && requireNamespace("lmtest", quietly = TRUE)) {
    bp_test <- lmtest::bptest(fit)
    bp_p_value <- unname(bp_test$p.value)
    bp_recommendation <- if (is.na(bp_p_value)) {
      "Inspect a plot of residuals against fitted values directly"
    } else if (bp_p_value < 0.05) {
      paste(
        "Heteroscedasticity detected.",
        "Consider variance stabilizing transformations or robust SEs."
      )
    } else {
      "Homoscedasticity assumption appears satisfied"
    }
    assumptions$homoscedasticity <- list(
      assumption = "Homoscedasticity",
      check = bp_p_value >= 0.05,
      details = paste("Breusch-Pagan test p-value:", round(bp_p_value, 4)),
      recommendation = bp_recommendation
    )
  } else {
    # Simpler check: correlation between abs(residuals) and fitted values
    het_cor <- cor(abs(residuals), fitted_values)
    het_details <- paste(
      "Correlation between |residuals| and fitted values:",
      round(het_cor, 4)
    )
    het_recommendation <- if (abs(het_cor) >= 0.2) {
      paste(
        "Possible heteroscedasticity.",
        "Consider variance stabilizing transformations or robust SEs."
      )
    } else {
      "Homoscedasticity assumption appears satisfied"
    }
    assumptions$homoscedasticity <- list(
      assumption = "Homoscedasticity",
      check = abs(het_cor) < 0.2,
      details = het_details,
      recommendation = het_recommendation
    )
  }

  # 4. Normality of Residuals
  # Shapiro-Wilk test
  # Shapiro-Wilk limited to 5000 observations
  if (is_logistic) {
    assumptions$normality <- tl_not_logistic_assumption(
      "Normality of Residuals",
      "the residuals of a binary outcome are not expected to be normal"
    )
  } else if (test && length(residuals) <= 5000) {
    sw_test <- shapiro.test(residuals)
    sw_p_value <- sw_test$p.value
    norm_recommendation <- if (sw_p_value < 0.05) {
      paste(
        "Residuals may not be normally distributed.",
        "Consider transformations or robust regression."
      )
    } else {
      "Normality assumption appears satisfied"
    }
    assumptions$normality <- list(
      assumption = "Normality of Residuals",
      check = sw_p_value >= 0.05,
      details = paste("Shapiro-Wilk test p-value:", round(sw_p_value, 4)),
      recommendation = norm_recommendation
    )
  } else {
    # Check skewness and kurtosis
    if (requireNamespace("moments", quietly = TRUE)) {
      skew <- moments::skewness(residuals)
      kurt <- moments::kurtosis(residuals)
      norm_check <- abs(skew) < 0.5 && abs(kurt - 3) < 1
      norm_details <- paste(
        "Skewness:", round(skew, 4),
        "Kurtosis:", round(kurt, 4)
      )
      norm_recommendation <- if (!norm_check) {
        paste(
          "Residuals may not be normally distributed.",
          "Consider transformations or robust regression."
        )
      } else {
        "Normality assumption appears satisfied"
      }
      assumptions$normality <- list(
        assumption = "Normality of Residuals",
        check = norm_check,
        details = norm_details,
        recommendation = norm_recommendation
      )
    } else {
      # Simplified check with quantiles
      q_norm <- qqnorm(residuals, plot = FALSE)
      cor_norm <- cor(q_norm$x, q_norm$y)
      qq_recommendation <- if (cor_norm < 0.98) {
        paste(
          "Residuals may not be normally distributed.",
          "Consider transformations or robust regression."
        )
      } else {
        "Normality assumption appears satisfied"
      }
      assumptions$normality <- list(
        assumption = "Normality of Residuals",
        check = cor_norm >= 0.98,
        details = paste("QQ-plot correlation:", round(cor_norm, 4)),
        recommendation = qq_recommendation
      )
    }
  }

  # 5. Multicollinearity
  assumptions$multicollinearity <- tl_check_multicollinearity(fit, verbose)

  # 6. Outliers and Influential Points. which() leaves out a flag that a
  # NaN measure made NA; sum() would count it as NA.
  influence_df <- tl_influence_measures(model)
  influential_obs <- influence_df$observation[
    which(influence_df$is_influential)
  ]
  n_influential <- length(influential_obs)

  outlier_recommendation <- if (n_influential > 0) {
    obs_to_show <- utils::head(influential_obs, 5)
    paste(
      "Consider inspecting observations:",
      paste(obs_to_show, collapse = ", "),
      "... (and potentially others)"
    )
  } else {
    "No influential outliers detected"
  }
  assumptions$outliers <- list(
    assumption = "No Influential Outliers",
    check = n_influential == 0,
    details = paste(n_influential, "influential observations detected"),
    recommendation = outlier_recommendation
  )

  # Print summary if verbose. A check is NA when its test could not decide,
  # and `if (NA)` stopped the summary partway through.
  if (verbose) {
    message("Model Assumptions Check Summary:")
    message("--------------------------------")
    for (name in names(assumptions)) {
      check <- assumptions[[name]]
      check_status <- if (isTRUE(check$check)) {
        "SATISFIED"
      } else if (isFALSE(check$check)) {
        "VIOLATED"
      } else {
        "UNKNOWN"
      }

      message(
        check$assumption, ": ", check_status, "\n",
        "  Details: ", check$details, "\n",
        "  Recommendation: ", check$recommendation, "\n"
      )
    }
  }

  # Add overall assessment. An undecided (NA) check is neither satisfied nor
  # violated; counted, it made the totals NA.
  checks <- Filter(
    Negate(is.null),
    lapply(assumptions, function(x) x$check)
  )
  checks <- unlist(checks)
  checks <- checks[!is.na(checks)]

  if (length(checks) > 0) {
    overall_status <- if (all(checks)) {
      "All checked assumptions appear to be satisfied."
    } else {
      n_violated <- sum(!checks)
      paste(
        n_violated,
        "assumption(s) appear to be violated. See details."
      )
    }
  } else {
    overall_status <- "Could not perform complete assumption checks."
  }

  assumptions$overall <- list(
    status = overall_status,
    n_checked = length(checks),
    n_violated = sum(!checks),
    n_satisfied = sum(checks)
  )

  assumptions
}

#' Create a comprehensive diagnostic dashboard
#'
#' @param model A tidylearn model object whose fit is an \code{lm} or
#'   \code{glm}: method \code{"linear"}, \code{"polynomial"} or
#'   \code{"logistic"}
#' @param include_influence Logical; whether to include influence diagnostics
#' @param include_assumptions Logical; whether to include assumption checks
#' @param include_performance Logical; whether to include performance metrics
#' @param arrange_plots Layout arrangement (e.g., "grid", "row", "column")
#' @return A \code{\link[gridExtra]{grid.arrange}} object (a
#'   \code{\link[grid]{grob}}) containing the arranged diagnostic plots.
#' @examples
#' \donttest{
#' if (requireNamespace("gridExtra")) {
#'   model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#'   tl_diagnostic_dashboard(model)
#' }
#' }
#' @export
tl_diagnostic_dashboard <- function(model, include_influence = TRUE,
                                    include_assumptions = TRUE,
                                    include_performance = TRUE,
                                    arrange_plots = "grid") {

  # The panels read standardised residuals, hat values and Cook's distance
  # off an lm or glm fit. A tree reached rstandard() and failed with "no
  # applicable method for 'rstandard' applied to an object of class rpart".
  if (!inherits(model, "tidylearn_model") || !inherits(model$fit, "lm")) {
    method <- if (inherits(model, "tidylearn_model")) model$spec$method
    stop(
      "The diagnostic dashboard is only available for linear-based models ",
      "(method \"linear\", \"polynomial\" or \"logistic\")",
      if (!is.null(method)) paste0("; this model's method is \"", method, "\""),
      ".",
      call. = FALSE
    )
  }

  # Check package dependencies
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    stop(
      "Package 'gridExtra' is required for creating dashboards",
      call. = FALSE
    )
  }

  # Initialize plots list
  plots <- list()

  # Add basic diagnostic plots
  plots$residuals_vs_fitted <- tl_plot_residuals(model, type = "fitted")
  plots$residual_hist <- tl_plot_residuals(model, type = "histogram")
  plots$qq_plot <- ggplot2::ggplot(
    data.frame(residuals = rstandard(model$fit)),
    ggplot2::aes(sample = residuals)
  ) +
    ggplot2::stat_qq() +
    ggplot2::stat_qq_line(color = "red") +
    ggplot2::labs(
      title = "Normal Q-Q Plot",
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    ) +
    ggplot2::theme_minimal()

  # Add scale-location plot
  diag_plots <- tl_plot_diagnostics(model, which = 3)
  plots$scale_location <- diag_plots[["scale_location"]]

  # Add influence plots if requested
  if (include_influence) {
    plots$cook_distance <- tl_plot_influence(model, plot_type = "cook")
    plots$leverage_plot <- tl_plot_influence(model, plot_type = "leverage")
  }

  # Add assumption check if requested
  if (include_assumptions) {
    assumptions <- tl_check_assumptions(model, verbose = FALSE)

    # Create plot with assumption check results. A check left NA by a test
    # that could not decide is shown as unknown, not as a missing fill.
    shown <- assumptions[c("linearity", "independence", "homoscedasticity",
                           "normality", "multicollinearity")]
    assumption_results <- data.frame(
      Assumption = vapply(shown, function(x) x$assumption, character(1)),
      Status = vapply(shown, function(x) {
        if (isTRUE(x$check)) {
          "Satisfied"
        } else if (isFALSE(x$check)) {
          "Violated"
        } else {
          "Unknown"
        }
      }, character(1)),
      Details = vapply(shown, function(x) x$details, character(1))
    )

    # Create a textual summary plot
    status_colors <- c(
      "Satisfied" = "green",
      "Violated" = "red",
      "Unknown" = "gray"
    )
    plots$assumptions <- ggplot2::ggplot(
      assumption_results,
      ggplot2::aes(x = 1, y = Assumption, fill = Status)
    ) +
      ggplot2::geom_tile() +
      ggplot2::geom_text(
        ggplot2::aes(label = Details),
        hjust = 0,
        x = 1.05
      ) +
      ggplot2::scale_fill_manual(values = status_colors) +
      ggplot2::labs(title = "Assumption Check Results") +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.title = ggplot2::element_blank(),
        axis.text.x = ggplot2::element_blank(),
        axis.ticks = ggplot2::element_blank(),
        panel.grid = ggplot2::element_blank()
      ) +
      ggplot2::coord_cartesian(xlim = c(0.5, 2.5))
  }

  # Add performance metrics if requested
  if (include_performance) {
    # Calculate model metrics
    metrics <- tl_evaluate(model)

    # Create performance summary plot
    plots$performance <- ggplot2::ggplot(
      metrics,
      ggplot2::aes(x = metric, y = value)
    ) +
      ggplot2::geom_col(fill = "steelblue") +
      ggplot2::geom_text(
        ggplot2::aes(label = round(value, 3)),
        vjust = -0.5
      ) +
      ggplot2::labs(title = "Model Performance Metrics") +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.title = ggplot2::element_blank(),
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
      )
  }

  # Arrange plots based on specified layout
  if (arrange_plots == "grid") {
    # Determine grid dimensions
    n_plots <- length(plots)
    n_cols <- min(3, n_plots)
    n_rows <- ceiling(n_plots / n_cols)

    # Arrange in grid
    combined_plot <- gridExtra::grid.arrange(
      grobs = plots,
      ncol = n_cols,
      nrow = n_rows
    )
  } else if (arrange_plots == "row") {
    # Arrange in a single row
    combined_plot <- gridExtra::grid.arrange(
      grobs = plots,
      ncol = length(plots)
    )
  } else if (arrange_plots == "column") {
    # Arrange in a single column
    combined_plot <- gridExtra::grid.arrange(
      grobs = plots,
      nrow = length(plots)
    )
  } else {
    stop(
      "Invalid arrange_plots value. Use 'grid', 'row', or 'column'.",
      call. = FALSE
    )
  }

  combined_plot
}

#' Detect outliers in the data
#'
#' @param data A data frame containing the data
#' @param variables Character vector of variables to check for outliers
#' @param method Method for outlier detection: "boxplot", "z-score", "cook",
#'   "iqr", "mahalanobis"
#' @param threshold Threshold for outlier detection
#' @param plot Logical; whether to create a plot of outliers
#' @return A list with outlier detection results:
#'   \describe{
#'     \item{method}{The detection method used (character).}
#'     \item{method_name}{Human-readable method name (character).}
#'     \item{threshold}{The threshold value used (numeric).}
#'     \item{threshold_label}{Formatted threshold description (character).}
#'     \item{outlier_flags}{A logical matrix (observations x variables),
#'       \code{NA} where a value is missing. For \code{"cook"} and
#'       \code{"mahalanobis"} a row's flags are the same in every column,
#'       and \code{NA} when any of its values is missing.}
#'     \item{any_outlier}{Logical vector indicating if each observation is an
#'       outlier in any variable. Missing flags are ignored, so a row with
#'       no flag at all is \code{FALSE}.}
#'     \item{outlier_counts}{List with \code{total}, \code{by_variable}, and
#'       \code{by_observation} counts.}
#'     \item{outlier_indices}{Integer vector of outlier row indices.}
#'     \item{plot}{A \code{\link[ggplot2]{ggplot}} object, or \code{NULL} if
#'       \code{plot = FALSE}.}
#'   }
#' @examples
#' \donttest{
#' tl_detect_outliers(mtcars, variables = c("mpg", "wt"), method = "iqr")
#' }
#' @export
tl_detect_outliers <- function(data, variables = NULL, method = "iqr",
                               threshold = NULL, plot = TRUE) {
  # Handle variables selection
  if (is.null(variables)) {
    # Select only numeric variables. vapply() keeps a frame with no columns
    # a logical index, where sapply() returned list().
    variables <- names(data)[vapply(data, is.numeric, logical(1))]
    if (length(variables) == 0) {
      stop("No numeric variables found in the data", call. = FALSE)
    }
  } else {
    # Check if specified variables exist and are numeric
    for (var in variables) {
      if (!var %in% names(data)) {
        stop("Variable not found in data: ", var, call. = FALSE)
      }
      if (!is.numeric(data[[var]])) {
        stop("Variable is not numeric: ", var, call. = FALSE)
      }
    }
  }

  # Extract data for selected variables
  var_data <- data[, variables, drop = FALSE]

  # Set default threshold based on method
  if (is.null(threshold)) {
    threshold <- switch(
      method,
      "boxplot" = 1.5, # IQR multiplier
      "z-score" = 3, # Standard deviations
      "cook" = 4 / nrow(data),
      "iqr" = 1.5,
      "mahalanobis" = 0.975,
      2 # Default multiplier
    )
  }

  # One column of flags per variable. matrix() keeps a single row a 1 x k
  # matrix, which sapply() simplified to a vector, and apply() below failed
  # with "dim(X) must have a positive length".
  flag_each <- function(flag) {
    flags <- vapply(variables, function(var) as.vector(flag(var_data[[var]])),
                    logical(nrow(var_data)))
    matrix(flags, nrow = nrow(var_data), ncol = length(variables),
           dimnames = list(NULL, variables))
  }

  # Detect outliers based on method
  if (method == "boxplot" || method == "iqr") {
    # IQR method for each variable
    outlier_flags <- flag_each(function(x) {
      q1 <- stats::quantile(x, 0.25, na.rm = TRUE)
      q3 <- stats::quantile(x, 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      lower_bound <- q1 - threshold * iqr
      upper_bound <- q3 + threshold * iqr
      x < lower_bound | x > upper_bound
    })

    method_name <- "Interquartile Range (IQR)"
    threshold_label <- paste0("IQR multiplier: ", threshold)

  } else if (method == "z-score") {
    # Z-score method for each variable
    outlier_flags <- flag_each(function(x) abs(scale(x)) > threshold)

    method_name <- "Z-Score"
    threshold_label <- paste0("Standard deviations: ", threshold)

  } else if (method == "cook") {
    # Cook's distance method (requires fitting a model)
    # First need to create a model formula
    if (length(variables) < 2) {
      stop(
        "Cook's distance method requires at least 2 variables",
        call. = FALSE
      )
    }

    # Use first variable as response. The names are backquoted, since a
    # column such as `car weight` pasted in as it stands does not parse.
    formula <- stats::reformulate(
      paste0("`", variables[-1], "`"),
      response = as.name(variables[1])
    )

    # Fit linear model. na.exclude keeps one distance per row of the data,
    # NA where a value is missing: under na.omit the 31 distances of 32 rows
    # were recycled into the flag matrix, shifting every column.
    model <- stats::lm(formula, data = data, na.action = stats::na.exclude)

    # Calculate Cook's distance
    cooks_d <- stats::cooks.distance(model)

    # Create flags (only available for the whole observation, not by variable)
    outlier_flags <- matrix(
      cooks_d > threshold,
      nrow = nrow(data),
      ncol = length(variables)
    )
    colnames(outlier_flags) <- variables

    method_name <- "Cook's Distance"
    threshold_label <- paste0("Threshold: ", threshold)

  } else if (method == "mahalanobis") {
    # Mahalanobis distance for multivariate outlier detection
    if (length(variables) < 2) {
      stop(
        "Mahalanobis distance requires at least 2 variables",
        call. = FALSE
      )
    }

    # Calculate center (means) and covariance matrix
    center <- colMeans(var_data, na.rm = TRUE)
    cov_matrix <- stats::cov(var_data, use = "pairwise.complete.obs")

    # Calculate Mahalanobis distances
    mahal_dist <- stats::mahalanobis(var_data, center, cov_matrix)

    # Convert threshold to chi-square quantile
    chi2_quantile <- stats::qchisq(threshold, df = length(variables))

    # Flag outliers (same for all variables)
    outlier_flags <- matrix(
      mahal_dist > chi2_quantile,
      nrow = nrow(data),
      ncol = length(variables)
    )
    colnames(outlier_flags) <- variables

    method_name <- "Mahalanobis Distance"
    threshold_label <- paste0(
      "Chi-square quantile (p = ", threshold,
      ", df = ", length(variables), "): ", round(chi2_quantile, 2)
    )

  } else {
    stop(
      "Invalid method. Use 'boxplot', 'z-score',",
      " 'cook', 'iqr', or 'mahalanobis'.",
      call. = FALSE
    )
  }

  # Combine flags across variables. A missing value's flag is NA, and any()
  # over it made the row, and with it the total count, NA.
  any_outlier <- apply(outlier_flags, 1, any, na.rm = TRUE)

  # Count outliers
  outlier_counts <- list(
    total = sum(any_outlier),
    by_variable = colSums(outlier_flags, na.rm = TRUE),
    by_observation = rowSums(outlier_flags, na.rm = TRUE)
  )

  # Create plot if requested
  if (plot) {
    if (method == "mahalanobis" && length(variables) >= 2) {
      # For Mahalanobis, create scatter plot matrix with outliers highlighted
      # Create a data frame with outlier flags
      plot_data <- var_data
      plot_data$is_outlier <- any_outlier

      # Create pairs plot
      if (requireNamespace("GGally", quietly = TRUE)) {
        outlier_plot <- GGally::ggpairs(
          plot_data,
          columns = seq_along(variables),
          aes(color = is_outlier),
          progress = FALSE
        ) +
          ggplot2::scale_color_manual(
            values = c("FALSE" = "blue", "TRUE" = "red"),
            drop = FALSE
          ) +
          ggplot2::labs(
            title = paste("Outlier Detection using", method_name),
            subtitle = threshold_label
          )
      } else {
        # Fallback to basic scatter plot of first two variables
        outlier_plot <- ggplot2::ggplot(
          plot_data,
          ggplot2::aes(
            x = .data[[variables[1]]],
            y = .data[[variables[2]]],
            color = is_outlier
          )
        ) +
          ggplot2::geom_point() +
          ggplot2::scale_color_manual(
            values = c("FALSE" = "blue", "TRUE" = "red"),
            drop = FALSE
          ) +
          ggplot2::labs(
            title = paste("Outlier Detection using", method_name),
            subtitle = threshold_label,
            x = variables[1],
            y = variables[2],
            color = "Outlier"
          ) +
          ggplot2::theme_minimal()
      }
    } else if (method == "cook") {
      # For Cook's distance, create index plot. A row the fit dropped has
      # no distance to draw.
      plot_data <- data.frame(
        observation = seq_along(cooks_d),
        cooks_distance = cooks_d,
        is_outlier = cooks_d > threshold
      )
      plot_data <- plot_data[!is.na(plot_data$cooks_distance), , drop = FALSE]

      outlier_plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
          x = observation,
          y = cooks_distance,
          color = is_outlier
        )
      ) +
        ggplot2::geom_point() +
        ggplot2::geom_hline(
          yintercept = threshold,
          linetype = "dashed",
          color = "red"
        ) +
        ggplot2::scale_color_manual(
          values = c("FALSE" = "blue", "TRUE" = "red"),
          drop = FALSE
        ) +
        ggplot2::labs(
          title = paste("Outlier Detection using", method_name),
          subtitle = threshold_label,
          x = "Observation Index",
          y = "Cook's Distance",
          color = "Outlier"
        ) +
        ggplot2::theme_minimal()
    } else {
      # For other methods, create boxplots with outliers highlighted
      # Prepare data for plotting
      plot_data <- tidyr::pivot_longer(
        var_data,
        cols = dplyr::everything(),
        names_to = "variable",
        values_to = "value"
      )

      # Add outlier flag.
      #
      # pivot_longer() emits row-major output: each observation's
      # variables appear consecutively. Deriving the observation as
      # (i - 1) %% nrow(data) + 1 assumes column-major and is correct
      # only when there is exactly one variable -- with more, the flags
      # get attached to the wrong points and the plot colours genuine
      # outliers as normal.
      n_vars <- length(variables)
      obs_idx <- ((seq_len(nrow(plot_data)) - 1L) %/% n_vars) + 1L
      var_idx <- match(plot_data$variable, variables)

      plot_data$is_outlier <- outlier_flags[
        cbind(obs_idx, var_idx)
      ]

      outlier_plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
          x = variable,
          y = value,
          fill = is_outlier
        )
      ) +
        ggplot2::geom_boxplot(outlier.shape = NA) +
        ggplot2::geom_jitter(
          ggplot2::aes(color = is_outlier),
          width = 0.2,
          alpha = 0.7
        ) +
        ggplot2::scale_fill_manual(values = c("lightblue", "lightpink")) +
        ggplot2::scale_color_manual(
          values = c("FALSE" = "blue", "TRUE" = "red"),
          drop = FALSE
        ) +
        ggplot2::labs(
          title = paste("Outlier Detection using", method_name),
          subtitle = threshold_label,
          x = "Variable",
          y = "Value",
          fill = "Outlier",
          color = "Outlier"
        ) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
        )
    }
  } else {
    outlier_plot <- NULL
  }

  # Create output list
  result <- list(
    method = method,
    method_name = method_name,
    threshold = threshold,
    threshold_label = threshold_label,
    outlier_flags = outlier_flags,
    any_outlier = any_outlier,
    outlier_counts = outlier_counts,
    outlier_indices = which(any_outlier),
    plot = outlier_plot
  )

  result
}
