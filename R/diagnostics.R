#' @title Advanced Diagnostics Functions for tidysl
#' @name tidysl-diagnostics
#' @description Functions for advanced model diagnostics, assumption checking, and outlier detection
#' @importFrom stats influence.measures cooks.distance hatvalues dffits dfbetas
#' @importFrom stats lm.influence rstudent rstandard
#' @importFrom stats shapiro.test bartlett.test kruskal.test
#' @importFrom dplyr %>% filter select mutate arrange
#' @importFrom ggplot2 ggplot aes geom_point geom_text labs theme_minimal
NULL

#' Calculate influence measures for a linear model
#'
#' @param model A tidysl model object
#' @param threshold_cook Cook's distance threshold (default: 4/n)
#' @param threshold_leverage Leverage threshold (default: 2*(p+1)/n)
#' @param threshold_dffits DFFITS threshold (default: 2*sqrt((p+1)/n))
#' @return A data frame with influence measures
#' @export
tl_influence_measures <- function(model, threshold_cook = NULL,
                                  threshold_leverage = NULL,
                                  threshold_dffits = NULL) {
  # Check if model is supported
  if (!inherits(model, "tidysl_model") ||
      !model$spec$method %in% c("linear", "logistic", "polynomial", "ridge", "lasso", "elastic_net")) {
    stop("Influence measures are only available for linear-based models", call. = FALSE)
  }

  # Extract fitted model
  fit <- model$fit

  # Data dimensions
  n <- nrow(model$data)
  p <- length(coef(fit)) - 1  # Number of predictors (excluding intercept)

  # Set default thresholds if not provided
  if (is.null(threshold_cook)) threshold_cook <- 4/n
  if (is.null(threshold_leverage)) threshold_leverage <- 2*(p+1)/n
  if (is.null(threshold_dffits)) threshold_dffits <- 2*sqrt((p+1)/n)

  # Calculate influence measures
  cooks_d <- cooks.distance(fit)
  leverage <- hatvalues(fit)
  dffits_val <- dffits(fit)
  dfbetas_val <- dfbetas(fit)

  # Get standardized residuals
  std_resid <- rstandard(fit)
  stud_resid <- rstudent(fit)

  # Create data frame
  influence_df <- data.frame(
    observation = 1:n,
    cooks_distance = cooks_d,
    leverage = leverage,
    dffits = dffits_val,
    std_residual = std_resid,
    stud_residual = stud_resid
  )

  # Add flags for influential observations
  influence_df$is_cook_influential <- influence_df$cooks_distance > threshold_cook
  influence_df$is_leverage_influential <- influence_df$leverage > threshold_leverage
  influence_df$is_dffits_influential <- abs(influence_df$dffits) > threshold_dffits
  influence_df$is_outlier <- abs(influence_df$std_residual) > 3

  # Add dfbetas as separate columns
  coef_names <- names(coef(fit))
  for (i in seq_along(coef_names)) {
    col_name <- paste0("dfbetas_", gsub("[^[:alnum:]]", "_", coef_names[i]))
    influence_df[[col_name]] <- dfbetas_val[, i]
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

  return(influence_df)
}

#' Plot influence diagnostics
#'
#' @param model A tidysl model object
#' @param plot_type Type of influence plot: "cook", "leverage", "index"
#' @param threshold_cook Cook's distance threshold (default: 4/n)
#' @param threshold_leverage Leverage threshold (default: 2*(p+1)/n)
#' @param threshold_dffits DFFITS threshold (default: 2*sqrt((p+1)/n))
#' @param n_labels Number of points to label (default: 3)
#' @param label_size Text size for labels (default: 3)
#' @return A ggplot object
#' @export
tl_plot_influence <- function(model, plot_type = "cook", threshold_cook = NULL,
                              threshold_leverage = NULL, threshold_dffits = NULL,
                              n_labels = 3, label_size = 3) {
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
    top_idx <- order(influence_df$cooks_distance, decreasing = TRUE)[1:min(n_labels, nrow(influence_df))]
    influence_df$label <- ifelse(influence_df$observation %in% influence_df$observation[top_idx],
                                 as.character(influence_df$observation), "")

    p <- ggplot2::ggplot(influence_df, ggplot2::aes(x = observation, y = cooks_distance)) +
      ggplot2::geom_point(ggplot2::aes(color = is_cook_influential), size = 3, alpha = 0.7) +
      ggplot2::geom_hline(yintercept = threshold_cook, linetype = "dashed", color = "red") +
      ggplot2::geom_text(ggplot2::aes(label = label), hjust = -0.3, vjust = 0.5, size = label_size) +
      ggplot2::scale_color_manual(values = c("blue", "red")) +
      ggplot2::labs(
        title = "Cook's Distance Plot",
        subtitle = paste("Threshold:", round(threshold_cook, 4), "- Points above are influential"),
        x = "Observation Index",
        y = "Cook's Distance",
        color = "Influential"
      ) +
      ggplot2::theme_minimal()

  } else if (plot_type == "leverage") {
    # Leverage-Residual plot (Bubble plot with Cook's distance)
    # Identify top points to label
    top_idx <- order(influence_df$cooks_distance, decreasing = TRUE)[1:min(n_labels, nrow(influence_df))]
    influence_df$label <- ifelse(influence_df$observation %in% influence_df$observation[top_idx],
                                 as.character(influence_df$observation), "")

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
      ggplot2::geom_hline(yintercept = c(-3, 0, 3), linetype = "dashed", color = c("red", "black", "red")) +
      ggplot2::geom_vline(xintercept = threshold_leverage, linetype = "dashed", color = "red") +
      ggplot2::geom_text(ggplot2::aes(label = label), hjust = -0.3, vjust = 0.5, size = label_size) +
      ggplot2::scale_color_manual(values = c("blue", "red")) +
      ggplot2::labs(
        title = "Leverage-Residual Plot",
        subtitle = paste(
          "Leverage threshold:", round(threshold_leverage, 4),
          "- Residual threshold: ±3"
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
    top_idx <- order(abs(influence_df$std_residual), decreasing = TRUE)[1:min(n_labels, nrow(influence_df))]
    influence_df$label <- ifelse(influence_df$observation %in% influence_df$observation[top_idx],
                                 as.character(influence_df$observation), "")

    p <- ggplot2::ggplot(influence_df, ggplot2::aes(x = observation, y = std_residual)) +
      ggplot2::geom_point(ggplot2::aes(color = is_outlier), size = 3, alpha = 0.7) +
      ggplot2::geom_hline(yintercept = c(-3, 0, 3), linetype = "dashed", color = c("red", "black", "red")) +
      ggplot2::geom_text(ggplot2::aes(label = label), hjust = -0.3, vjust = 0.5, size = label_size) +
      ggplot2::scale_color_manual(values = c("blue", "red")) +
      ggplot2::labs(
        title = "Index Plot of Standardized Residuals",
        subtitle = "Points outside ±3 are considered outliers",
        x = "Observation Index",
        y = "Standardized Residuals",
        color = "Outlier"
      ) +
      ggplot2::theme_minimal()
  } else {
    stop("Invalid plot_type. Use 'cook', 'leverage', or 'index'.", call. = FALSE)
  }

  return(p)
}

#' Check model assumptions
#'
#' @param model A tidysl model object
#' @param test Logical; whether to perform statistical tests
#' @param verbose Logical; whether to print test results and explanations
#' @return A list with assumption check results
#' @export
tl_check_assumptions <- function(model, test = TRUE, verbose = TRUE) {
  # Check if model is supported
  if (!inherits(model, "tidysl_model") ||
      !model$spec$method %in% c("linear", "logistic", "polynomial", "ridge", "lasso", "elastic_net")) {
    stop("Assumption checking is only available for linear-based models", call. = FALSE)
  }

  # Extract fitted model and data
  fit <- model$fit
  data <- model$data

  # Get residuals
  residuals <- residuals(fit)
  std_residuals <- rstandard(fit)
  fitted_values <- fitted(fit)

  # Initialize results list
  assumptions <- list()

  # 1. Linearity
  # Check correlation between fitted values and residuals
  linearity_cor <- cor(fitted_values, residuals)
  assumptions$linearity <- list(
    assumption = "Linearity",
    check = abs(linearity_cor) < 0.1,
    details = paste("Correlation between fitted values and residuals:", round(linearity_cor, 4)),
    recommendation = if (abs(linearity_cor) >= 0.1) {
      "Consider non-linear transformations or polynomial terms"
    } else {
      "Linearity assumption appears satisfied"
    }
  )

  # 2. Independence
  # Durbin-Watson test for autocorrelation
  if (test && requireNamespace("car", quietly = TRUE)) {
    dw_test <- car::durbinWatsonTest(fit)
    dw_statistic <- dw_test$dw
    assumptions$independence <- list(
      assumption = "Independence",
      check = dw_statistic >= 1.5 && dw_statistic <= 2.5,
      details = paste("Durbin-Watson statistic:", round(dw_statistic, 4)),
      recommendation = if (dw_statistic < 1.5 || dw_statistic > 2.5) {
        "Possible autocorrelation in residuals. Check for time-series structure or clustering."
      } else {
        "Independence assumption appears satisfied"
      }
    )
  } else {
    assumptions$independence <- list(
      assumption = "Independence",
      check = NULL,
      details = "No statistical test performed",
      recommendation = "Ensure observations are independent. Consider the data collection process."
    )
  }

  # 3. Homoscedasticity (Equal Variance)
  # Breusch-Pagan test
  if (test && requireNamespace("lmtest", quietly = TRUE)) {
    bp_test <- lmtest::bptest(fit)
    bp_p_value <- bp_test$p.value
    assumptions$homoscedasticity <- list(
      assumption = "Homoscedasticity",
      check = bp_p_value >= 0.05,
      details = paste("Breusch-Pagan test p-value:", round(bp_p_value, 4)),
      recommendation = if (bp_p_value < 0.05) {
        "Heteroscedasticity detected. Consider variance stabilizing transformations or robust standard errors."
      } else {
        "Homoscedasticity assumption appears satisfied"
      }
    )
  } else {
    # Simpler check: correlation between abs(residuals) and fitted values
    het_cor <- cor(abs(residuals), fitted_values)
    assumptions$homoscedasticity <- list(
      assumption = "Homoscedasticity",
      check = abs(het_cor) < 0.2,
      details = paste("Correlation between |residuals| and fitted values:", round(het_cor, 4)),
      recommendation = if (abs(het_cor) >= 0.2) {
        "Possible heteroscedasticity. Consider variance stabilizing transformations or robust standard errors."
      } else {
        "Homoscedasticity assumption appears satisfied"
      }
    )
  }

  # 4. Normality of Residuals
  # Shapiro-Wilk test
  if (test && length(residuals) <= 5000) {  # Shapiro-Wilk limited to 5000 observations
    sw_test <- shapiro.test(residuals)
    sw_p_value <- sw_test$p.value
    assumptions$normality <- list(
      assumption = "Normality of Residuals",
      check = sw_p_value >= 0.05,
      details = paste("Shapiro-Wilk test p-value:", round(sw_p_value, 4)),
      recommendation = if (sw_p_value < 0.05) {
        "Residuals may not be normally distributed. Consider transformations or robust regression."
      } else {
        "Normality assumption appears satisfied"
      }
    )
  } else {
    # Check skewness and kurtosis
    if (requireNamespace("moments", quietly = TRUE)) {
      skew <- moments::skewness(residuals)
      kurt <- moments::kurtosis(residuals)
      norm_check <- abs(skew) < 0.5 && abs(kurt - 3) < 1
      assumptions$normality <- list(
        assumption = "Normality of Residuals",
        check = norm_check,
        details = paste("Skewness:", round(skew, 4), "Kurtosis:", round(kurt, 4)),
        recommendation = if (!norm_check) {
          "Residuals may not be normally distributed. Consider transformations or robust regression."
        } else {
          "Normality assumption appears satisfied"
        }
      )
    } else {
      # Simplified check with quantiles
      q_norm <- qqnorm(residuals, plot = FALSE)
      cor_norm <- cor(q_norm$x, q_norm$y)
      assumptions$normality <- list(
        assumption = "Normality of Residuals",
        check = cor_norm >= 0.98,
        details = paste("QQ-plot correlation:", round(cor_norm, 4)),
        recommendation = if (cor_norm < 0.98) {
          "Residuals may not be normally distributed. Consider transformations or robust regression."
        } else {
          "Normality assumption appears satisfied"
        }
      )
    }
  }

  # 5. Multicollinearity
  if (requireNamespace("car", quietly = TRUE)) {
    # Attempt to calculate VIF
    tryCatch({
      vif_values <- car::vif(fit)
      max_vif <- max(vif_values)
      assumptions$multicollinearity <- list(
        assumption = "No Multicollinearity",
        check = max_vif < 5,
        details = paste("Maximum VIF:", round(max_vif, 4)),
        recommendation = if (max_vif >= 5) {
          "Multicollinearity detected. Consider removing or combining highly correlated predictors."
        } else {
          "No serious multicollinearity detected"
        }
      )
    }, error = function(e) {
      # If VIF calculation fails, use correlation matrix
      if (verbose) {
        message("VIF calculation failed. Checking correlations instead.")
      }

      # Create correlation matrix of predictors
      formula <- model$spec$formula
      predictor_vars <- all.vars(formula)[-1]  # Remove response variable

      if (length(predictor_vars) > 1) {
        pred_data <- stats::model.matrix(formula, data)[, -1]  # Remove intercept
        cor_matrix <- cor(pred_data)
        max_cor <- max(abs(cor_matrix[upper.tri(cor_matrix)]))

        assumptions$multicollinearity <- list(
          assumption = "No Multicollinearity",
          check = max_cor < 0.7,
          details = paste("Maximum correlation between predictors:", round(max_cor, 4)),
          recommendation = if (max_cor >= 0.7) {
            "Potential multicollinearity. Consider removing or combining highly correlated predictors."
          } else {
            "No serious multicollinearity detected"
          }
        )
      } else {
        assumptions$multicollinearity <- list(
          assumption = "No Multicollinearity",
          check = TRUE,
          details = "Model has only one predictor",
          recommendation = "Not applicable with a single predictor"
        )
      }
    })
  } else {
    # If car package not available, use correlation matrix
    formula <- model$spec$formula
    predictor_vars <- all.vars(formula)[-1]  # Remove response variable

    if (length(predictor_vars) > 1) {
      pred_data <- stats::model.matrix(formula, data)[, -1]  # Remove intercept
      cor_matrix <- cor(pred_data)
      max_cor <- max(abs(cor_matrix[upper.tri(cor_matrix)]))

      assumptions$multicollinearity <- list(
        assumption = "No Multicollinearity",
        check = max_cor < 0.7,
        details = paste("Maximum correlation between predictors:", round(max_cor, 4)),
        recommendation = if (max_cor >= 0.7) {
          "Potential multicollinearity. Consider removing or combining highly correlated predictors."
        } else {
          "No serious multicollinearity detected"
        }
      )
    } else {
      assumptions$multicollinearity <- list(
        assumption = "No Multicollinearity",
        check = TRUE,
        details = "Model has only one predictor",
        recommendation = "Not applicable with a single predictor"
      )
    }
  }

  # 6. Outliers and Influential Points
  influence_df <- tl_influence_measures(model)
  n_influential <- sum(influence_df$is_influential)

  assumptions$outliers <- list(
    assumption = "No Influential Outliers",
    check = n_influential == 0,
    details = paste(n_influential, "influential observations detected"),
    recommendation = if (n_influential > 0) {
      paste(
        "Consider inspecting observations:",
        paste(influence_df$observation[influence_df$is_influential][1:min(5, n_influential)], collapse = ", "),
        "... (and potentially others)"
      )
    } else {
      "No influential outliers detected"
    }
  )

  # Print summary if verbose
  if (verbose) {
    cat("Model Assumptions Check Summary:\n")
    cat("--------------------------------\n")
    for (name in names(assumptions)) {
      check <- assumptions[[name]]
      check_status <- if (is.null(check$check)) {
        "UNKNOWN"
      } else if (check$check) {
        "SATISFIED"
      } else {
        "VIOLATED"
      }

      cat(paste0(
        check$assumption, ": ", check_status, "\n",
        "  Details: ", check$details, "\n",
        "  Recommendation: ", check$recommendation, "\n\n"
      ))
    }
  }

  # Add overall assessment
  checks <- sapply(assumptions, function(x) x$check)
  checks <- checks[!is.null(checks)]

  if (length(checks) > 0) {
    overall_status <- if (all(checks)) {
      "All checked assumptions appear to be satisfied."
    } else {
      n_violated <- sum(!checks)
      paste(n_violated, "assumption(s) appear to be violated. See details for recommendations.")
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

  return(assumptions)
}

#' Create a comprehensive diagnostic dashboard
#'
#' @param model A tidysl model object
#' @param include_influence Logical; whether to include influence diagnostics
#' @param include_assumptions Logical; whether to include assumption checks
#' @param include_performance Logical; whether to include performance metrics
#' @param arrange_plots Layout arrangement (e.g., "grid", "row", "column")
#' @return A plot grid with diagnostic plots
#' @export
tl_diagnostic_dashboard <- function(model, include_influence = TRUE,
                                    include_assumptions = TRUE,
                                    include_performance = TRUE,
                                    arrange_plots = "grid") {

  # Check package dependencies
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    stop("Package 'gridExtra' is required for creating dashboards", call. = FALSE)
  }

  # Initialize plots list
  plots <- list()

  # Add basic diagnostic plots
  plots$residuals_vs_fitted <- tl_plot_residuals(model, type = "fitted")
  plots$residual_hist <- tl_plot_residuals(model, type = "histogram")
  plots$qq_plot <- ggplot2::ggplot(data.frame(residuals = rstandard(model$fit)),
                                   ggplot2::aes(sample = residuals)) +
    ggplot2::stat_qq() +
    ggplot2::stat_qq_line(color = "red") +
    ggplot2::labs(title = "Normal Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles") +
    ggplot2::theme_minimal()

  # Add scale-location plot
  plots$scale_location <- tl_plot_diagnostics(model, which = 3)[[1]]

  # Add influence plots if requested
  if (include_influence) {
    plots$cook_distance <- tl_plot_influence(model, plot_type = "cook")
    plots$leverage_plot <- tl_plot_influence(model, plot_type = "leverage")
  }

  # Add assumption check if requested
  if (include_assumptions) {
    assumptions <- tl_check_assumptions(model, verbose = FALSE)

    # Create plot with assumption check results
    assumption_results <- data.frame(
      Assumption = sapply(assumptions[1:5], function(x) x$assumption),
      Status = sapply(assumptions[1:5], function(x) {
        if (is.null(x$check)) return("Unknown")
        ifelse(x$check, "Satisfied", "Violated")
      }),
      Details = sapply(assumptions[1:5], function(x) x$details)
    )

    # Create a textual summary plot
    plots$assumptions <- ggplot2::ggplot(assumption_results, ggplot2::aes(x = 1, y = Assumption, fill = Status)) +
      ggplot2::geom_tile() +
      ggplot2::geom_text(ggplot2::aes(label = Details), hjust = 0, x = 1.05) +
      ggplot2::scale_fill_manual(values = c("Satisfied" = "green", "Violated" = "red", "Unknown" = "gray")) +
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
    plots$performance <- ggplot2::ggplot(metrics, ggplot2::aes(x = metric, y = value)) +
      ggplot2::geom_col(fill = "steelblue") +
      ggplot2::geom_text(ggplot2::aes(label = round(value, 3)), vjust = -0.5) +
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
    stop("Invalid arrange_plots value. Use 'grid', 'row', or 'column'.", call. = FALSE)
  }

  return(combined_plot)
}

#' Detect outliers in the data
#'
#' @param data A data frame containing the data
#' @param variables Character vector of variables to check for outliers
#' @param method Method for outlier detection: "boxplot", "z-score", "cook", "iqr", "mahalanobis"
#' @param threshold Threshold for outlier detection
#' @param plot Logical; whether to create a plot of outliers
#' @return A list with outlier detection results
#' @export
tl_detect_outliers <- function(data, variables = NULL, method = "iqr",
                               threshold = NULL, plot = TRUE) {
  # Handle variables selection
  if (is.null(variables)) {
    # Select only numeric variables
    variables <- names(data)[sapply(data, is.numeric)]
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
    threshold <- switch(method,
                        "boxplot" = 1.5,     # IQR multiplier
                        "z-score" = 3,       # Standard deviations
                        "cook" = 4/nrow(data), # 4/n rule for Cook's distance
                        "iqr" = 1.5,         # IQR multiplier
                        "mahalanobis" = 0.975, # Quantile for chi-square distribution
                        2                    # Default multiplier
    )
  }

  # Detect outliers based on method
  if (method == "boxplot" || method == "iqr") {
    # IQR method for each variable
    outlier_flags <- sapply(variables, function(var) {
      q1 <- stats::quantile(var_data[[var]], 0.25, na.rm = TRUE)
      q3 <- stats::quantile(var_data[[var]], 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      lower_bound <- q1 - threshold * iqr
      upper_bound <- q3 + threshold * iqr
      var_data[[var]] < lower_bound | var_data[[var]] > upper_bound
    })

    method_name <- "Interquartile Range (IQR)"
    threshold_label <- paste0("IQR multiplier: ", threshold)

  } else if (method == "z-score") {
    # Z-score method for each variable
    outlier_flags <- sapply(variables, function(var) {
      z_scores <- abs(scale(var_data[[var]]))
      z_scores > threshold
    })

    method_name <- "Z-Score"
    threshold_label <- paste0("Standard deviations: ", threshold)

  } else if (method == "cook") {
    # Cook's distance method (requires fitting a model)
    # First need to create a model formula
    if (length(variables) < 2) {
      stop("Cook's distance method requires at least 2 variables (response and predictor)", call. = FALSE)
    }

    # Use first variable as response
    formula <- stats::as.formula(paste(variables[1], "~", paste(variables[-1], collapse = " + ")))

    # Fit linear model
    model <- stats::lm(formula, data = data)

    # Calculate Cook's distance
    cooks_d <- stats::cooks.distance(model)

    # Create flags (only available for the whole observation, not by variable)
    outlier_flags <- matrix(cooks_d > threshold, nrow = nrow(data), ncol = length(variables))
    colnames(outlier_flags) <- variables

    method_name <- "Cook's Distance"
    threshold_label <- paste0("Threshold: ", threshold)

  } else if (method == "mahalanobis") {
    # Mahalanobis distance for multivariate outlier detection
    if (length(variables) < 2) {
      stop("Mahalanobis distance method requires at least 2 variables", call. = FALSE)
    }

    # Calculate center (means) and covariance matrix
    center <- colMeans(var_data, na.rm = TRUE)
    cov_matrix <- stats::cov(var_data, use = "pairwise.complete.obs")

    # Calculate Mahalanobis distances
    mahal_dist <- stats::mahalanobis(var_data, center, cov_matrix)

    # Convert threshold to chi-square quantile
    chi2_quantile <- stats::qchisq(threshold, df = length(variables))

    # Flag outliers (same for all variables)
    outlier_flags <- matrix(mahal_dist > chi2_quantile, nrow = nrow(data), ncol = length(variables))
    colnames(outlier_flags) <- variables

    method_name <- "Mahalanobis Distance"
    threshold_label <- paste0("Chi-square quantile (p = ", threshold, ", df = ", length(variables), "): ", round(chi2_quantile, 2))

  } else {
    stop("Invalid method. Use 'boxplot', 'z-score', 'cook', 'iqr', or 'mahalanobis'.", call. = FALSE)
  }

  # Combine flags across variables
  any_outlier <- apply(outlier_flags, 1, any)

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
          columns = 1:length(variables),
          aes(color = is_outlier),
          progress = FALSE
        ) +
          ggplot2::scale_color_manual(values = c("blue", "red")) +
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
          ggplot2::scale_color_manual(values = c("blue", "red")) +
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
      # For Cook's distance, create index plot
      plot_data <- data.frame(
        observation = 1:length(cooks_d),
        cooks_distance = cooks_d,
        is_outlier = cooks_d > threshold
      )

      outlier_plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
          x = observation,
          y = cooks_distance,
          color = is_outlier
        )
      ) +
        ggplot2::geom_point() +
        ggplot2::geom_hline(yintercept = threshold, linetype = "dashed", color = "red") +
        ggplot2::scale_color_manual(values = c("blue", "red")) +
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

      # Add outlier flag
      plot_data$is_outlier <- FALSE
      for (i in 1:nrow(plot_data)) {
        var_idx <- match(plot_data$variable[i], variables)
        obs_idx <- (i - 1) %% nrow(data) + 1
        plot_data$is_outlier[i] <- outlier_flags[obs_idx, var_idx]
      }

      outlier_plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
          x = variable,
          y = value,
          fill = is_outlier
        )
      ) +
        ggplot2::geom_boxplot(outlier.shape = NA) +
        ggplot2::geom_jitter(ggplot2::aes(color = is_outlier), width = 0.2, alpha = 0.7) +
        ggplot2::scale_fill_manual(values = c("lightblue", "lightpink")) +
        ggplot2::scale_color_manual(values = c("blue", "red")) +
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

  return(result)
}
