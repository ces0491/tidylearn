#' @title Interaction Analysis Functions for tidylearn
#' @name tidylearn-interactions
#' @description Functions for testing, visualizing, and analyzing interactions
#' @importFrom stats lm anova
#' @importFrom dplyr %>% filter select mutate
#' @importFrom ggplot2 ggplot aes geom_line geom_point facet_wrap labs theme_minimal
NULL

#' Test for significant interactions between variables
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the base model without interactions
#' @param var1 First variable to test for interactions
#' @param var2 Second variable to test for interactions (if NULL, tests var1 with all others)
#' @param all_pairs Logical; whether to test all variable pairs
#' @param categorical_only Logical; whether to only test categorical variables
#' @param numeric_only Logical; whether to only test numeric variables
#' @param mixed_only Logical; whether to only test numeric-categorical pairs
#' @param alpha Significance level for interaction tests
#' @return A data frame with interaction test results
#' @export
tl_test_interactions <- function(data, formula, var1 = NULL, var2 = NULL,
                                 all_pairs = FALSE, categorical_only = FALSE,
                                 numeric_only = FALSE, mixed_only = FALSE,
                                 alpha = 0.05) {
  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Extract predictor variables
  predictors <- all.vars(formula)[-1]

  # Categorize variables
  var_types <- sapply(data[predictors], function(x) {
    if (is.factor(x) || is.character(x)) {
      return("categorical")
    } else if (is.numeric(x)) {
      return("numeric")
    } else {
      return("other")
    }
  })

  # Generate pairs to test
  if (all_pairs) {
    # Test all variable pairs
    pairs <- combn(predictors, 2, simplify = FALSE)
  } else if (!is.null(var1) && !is.null(var2)) {
    # Test specific pair
    pairs <- list(c(var1, var2))
  } else if (!is.null(var1)) {
    # Test var1 with all other variables
    pairs <- lapply(setdiff(predictors, var1), function(v) c(var1, v))
  } else {
    stop("Must specify at least var1 or set all_pairs = TRUE", call. = FALSE)
  }

  # Filter pairs based on variable types
  if (categorical_only) {
    pairs <- pairs[sapply(pairs, function(p) all(var_types[p] == "categorical"))]
  } else if (numeric_only) {
    pairs <- pairs[sapply(pairs, function(p) all(var_types[p] == "numeric"))]
  } else if (mixed_only) {
    pairs <- pairs[sapply(pairs, function(p) {
      var_types[p[1]] != var_types[p[2]] &&
        all(var_types[p] %in% c("categorical", "numeric"))
    })]
  }

  # Test interactions
  results <- lapply(pairs, function(pair) {
    # Build base model
    base_model <- lm(formula, data = data)

    # Build model with interaction
    int_formula <- update(formula, paste0(". ~ . + ", pair[1], ":", pair[2]))
    int_model <- lm(int_formula, data = data)

    # Perform ANOVA to compare models
    models_comparison <- anova(base_model, int_model)

    # Extract p-value for interaction term
    p_value <- models_comparison$`Pr(>F)`[2]

    # Calculate interaction effect size (change in R-squared)
    r2_base <- summary(base_model)$r.squared
    r2_int <- summary(int_model)$r.squared
    delta_r2 <- r2_int - r2_base

    # Create result row
    return(data.frame(
      var1 = pair[1],
      var2 = pair[2],
      p_value = p_value,
      significant = p_value < alpha,
      delta_r2 = delta_r2,
      f_statistic = models_comparison$F[2]
    ))
  })

  # Combine results
  all_results <- do.call(rbind, results)

  # Sort by significance
  all_results <- all_results[order(all_results$p_value), ]

  return(all_results)
}

#' Plot interaction effects
#'
#' @param model A tidylearn model object
#' @param var1 First variable in the interaction
#' @param var2 Second variable in the interaction
#' @param n_points Number of points to use for continuous variables
#' @param fixed_values Named list of values for other variables in the model
#' @param confidence Logical; whether to show confidence intervals
#' @param ... Additional arguments to pass to predict()
#' @return A ggplot object
#' @export
tl_plot_interaction <- function(model, var1, var2, n_points = 100, fixed_values = NULL,
                                confidence = TRUE, ...) {
  # Extract data
  data <- model$data

  # Check if variables exist in the model
  formula <- model$spec$formula
  all_vars <- all.vars(formula)[-1] # Remove response variable
  if (!var1 %in% all_vars || !var2 %in% all_vars) {
    stop("Variables not found in model formula", call. = FALSE)
  }

  # Determine variable types
  var1_type <- if (is.factor(data[[var1]]) || is.character(data[[var1]])) "categorical" else "numeric"
  var2_type <- if (is.factor(data[[var2]]) || is.character(data[[var2]])) "categorical" else "numeric"

  # Create grid of values
  if (var1_type == "categorical") {
    if (is.factor(data[[var1]])) {
      var1_values <- levels(data[[var1]])
    } else {
      var1_values <- unique(data[[var1]])
    }
  } else {
    var1_values <- seq(min(data[[var1]], na.rm = TRUE),
                       max(data[[var1]], na.rm = TRUE),
                       length.out = n_points)
  }

  if (var2_type == "categorical") {
    if (is.factor(data[[var2]])) {
      var2_values <- levels(data[[var2]])
    } else {
      var2_values <- unique(data[[var2]])
    }
  } else {
    var2_values <- seq(min(data[[var2]], na.rm = TRUE),
                       max(data[[var2]], na.rm = TRUE),
                       length.out = n_points)
  }

  # Create grid of all combinations
  grid <- expand.grid(
    var1 = var1_values,
    var2 = var2_values
  )
  names(grid) <- c(var1, var2)

  # Add fixed values for other variables
  all_other_vars <- setdiff(all_vars, c(var1, var2))

  for (v in all_other_vars) {
    if (!is.null(fixed_values) && v %in% names(fixed_values)) {
      # Use provided value
      grid[[v]] <- fixed_values[[v]]
    } else {
      # Use median/mode as default
      if (is.numeric(data[[v]])) {
        grid[[v]] <- median(data[[v]], na.rm = TRUE)
      } else if (is.factor(data[[v]]) || is.character(data[[v]])) {
        # Use most frequent value
        tab <- table(data[[v]])
        grid[[v]] <- names(tab)[which.max(tab)]

        # Convert to factor if necessary
        if (is.factor(data[[v]])) {
          grid[[v]] <- factor(grid[[v]], levels = levels(data[[v]]))
        }
      } else {
        # Use first value
        grid[[v]] <- data[[v]][1]
      }
    }
  }

  # Make predictions
  predictions <- predict(model, grid, ...)

  # Add predictions to grid
  if (is.data.frame(predictions)) {
    # Multiple columns (e.g., confidence intervals)
    grid <- cbind(grid, predictions)
    y_col <- "fit"
    lower_col <- "lwr"
    upper_col <- "upr"
  } else {
    # Single column
    grid$prediction <- predictions
    y_col <- "prediction"
    lower_col <- NULL
    upper_col <- NULL
  }

  # Create plot
  if (var1_type == "numeric" && var2_type == "categorical") {
    # Line plot with x = var1, color = var2
    p <- ggplot2::ggplot(grid, ggplot2::aes(x = .data[[var1]], y = .data[[y_col]], color = .data[[var2]])) +
      ggplot2::geom_line() +
      ggplot2::labs(
        title = paste("Interaction between", var1, "and", var2),
        x = var1,
        y = "Predicted Value",
        color = var2
      )

    # Add confidence intervals if available
    if (confidence && !is.null(lower_col) && !is.null(upper_col)) {
      p <- p + ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data[[lower_col]], ymax = .data[[upper_col]], fill = .data[[var2]]),
        alpha = 0.2,
        linetype = 0
      )
    }
  } else if (var1_type == "categorical" && var2_type == "numeric") {
    # Line plot with x = var2, color = var1
    p <- ggplot2::ggplot(grid, ggplot2::aes(x = .data[[var2]], y = .data[[y_col]], color = .data[[var1]])) +
      ggplot2::geom_line() +
      ggplot2::labs(
        title = paste("Interaction between", var1, "and", var2),
        x = var2,
        y = "Predicted Value",
        color = var1
      )

    # Add confidence intervals if available
    if (confidence && !is.null(lower_col) && !is.null(upper_col)) {
      p <- p + ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data[[lower_col]], ymax = .data[[upper_col]], fill = .data[[var1]]),
        alpha = 0.2,
        linetype = 0
      )
    }
  } else if (var1_type == "numeric" && var2_type == "numeric") {
    # Contour plot or heat map
    p <- ggplot2::ggplot(grid, ggplot2::aes(x = .data[[var1]], y = .data[[var2]], z = .data[[y_col]])) +
      ggplot2::geom_contour_filled() +
      ggplot2::labs(
        title = paste("Interaction between", var1, "and", var2),
        x = var1,
        y = var2,
        fill = "Predicted Value"
      )
  } else {
    # Categorical x Categorical: Faceted bar plot
    p <- ggplot2::ggplot(grid, ggplot2::aes(x = .data[[var1]], y = .data[[y_col]], fill = .data[[var2]])) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::labs(
        title = paste("Interaction between", var1, "and", var2),
        x = var1,
        y = "Predicted Value",
        fill = var2
      )
  }

  # Apply minimal theme
  p <- p + ggplot2::theme_minimal()

  return(p)
}

#' Find important interactions automatically
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the base model without interactions
#' @param top_n Number of top interactions to return
#' @param min_r2_change Minimum change in R-squared to consider
#' @param max_p_value Maximum p-value for significance
#' @param exclude_vars Character vector of variables to exclude from interaction testing
#' @return A tidylearn model with important interactions
#' @export
tl_auto_interactions <- function(data, formula, top_n = 3, min_r2_change = 0.01,
                                 max_p_value = 0.05, exclude_vars = NULL) {
  # Extract predictor variables
  predictors <- all.vars(formula)[-1]

  # Remove excluded variables
  if (!is.null(exclude_vars)) {
    predictors <- setdiff(predictors, exclude_vars)
  }

  # Generate all possible pairs
  pairs <- combn(predictors, 2, simplify = FALSE)

  # Test all interactions
  test_results <- tl_test_interactions(data, formula, all_pairs = TRUE, alpha = max_p_value)

  # Filter significant interactions
  significant <- test_results %>%
    dplyr::filter(.data$p_value < max_p_value, .data$delta_r2 >= min_r2_change)

  # Select top interactions
  if (nrow(significant) > top_n) {
    top_interactions <- significant[1:top_n, ]
  } else {
    top_interactions <- significant
  }

  if (nrow(top_interactions) == 0) {
    message("No significant interactions found based on criteria")
    # Return original model
    return(tl_model(data, formula, method = "linear"))
  }

  # Build formula with interactions
  interaction_terms <- apply(top_interactions, 1, function(row) {
    paste0(row["var1"], ":", row["var2"])
  })

  new_formula <- update(formula, paste0(". ~ . +", paste(interaction_terms, collapse = " + ")))

  # Fit model with interactions
  interaction_model <- tl_model(data, new_formula, method = "linear")

  # Add interaction test results to model
  attr(interaction_model, "interaction_tests") <- test_results
  attr(interaction_model, "selected_interactions") <- top_interactions

  return(interaction_model)
}

#' Calculate partial effects based on a model with interactions
#'
#' @param model A tidylearn model object
#' @param var Variable to calculate effects for
#' @param by_var Variable to calculate effects by (interaction variable)
#' @param at_values Named list of values at which to hold other variables
#' @param intervals Logical; whether to include confidence intervals
#' @return A data frame with marginal effects
#' @export
tl_interaction_effects <- function(model, var, by_var, at_values = NULL, intervals = TRUE) {
  # Extract data
  data <- model$data
  formula <- model$spec$formula

  # Check if variables exist in the model
  all_vars <- all.vars(formula)[-1] # Remove response variable
  if (!var %in% all_vars || !by_var %in% all_vars) {
    stop("Variables not found in model formula", call. = FALSE)
  }

  # Check if the interaction term exists in the model
  int_term <- paste0(var, ":", by_var)
  rev_int_term <- paste0(by_var, ":", var)

  has_interaction <- int_term %in% attr(terms(formula), "term.labels") ||
    rev_int_term %in% attr(terms(formula), "term.labels")

  if (!has_interaction) {
    warning(paste("Interaction term", int_term, "not found in model formula"),
            call. = FALSE)
  }

  # Determine by_var values
  if (is.factor(data[[by_var]]) || is.character(data[[by_var]])) {
    if (is.factor(data[[by_var]])) {
      by_values <- levels(data[[by_var]])
    } else {
      by_values <- unique(data[[by_var]])
    }
  } else {
    # For continuous by_var, use quantiles
    by_values <- stats::quantile(data[[by_var]], probs = seq(0, 1, 0.25), na.rm = TRUE)
    names(by_values) <- paste0("Q", seq(0, 100, 25))
  }

  # Set up variable values
  if (is.factor(data[[var]]) || is.character(data[[var]])) {
    if (is.factor(data[[var]])) {
      var_values <- levels(data[[var]])
    } else {
      var_values <- unique(data[[var]])
    }
    var_is_numeric <- FALSE
  } else {
    var_range <- range(data[[var]], na.rm = TRUE)
    var_values <- seq(var_range[1], var_range[2], length.out = 100)
    var_is_numeric <- TRUE
  }

  # Create grid for predictions
  grid_list <- list()

  for (bv in by_values) {
    # Create data frame for this by_value
    grid <- data.frame(var_values)
    names(grid) <- var
    grid[[by_var]] <- bv

    # Add values for other variables
    other_vars <- setdiff(all_vars, c(var, by_var))
    for (v in other_vars) {
      if (!is.null(at_values) && v %in% names(at_values)) {
        grid[[v]] <- at_values[[v]]
      } else {
        # Use median/mode
        if (is.numeric(data[[v]])) {
          grid[[v]] <- median(data[[v]], na.rm = TRUE)
        } else if (is.factor(data[[v]])) {
          grid[[v]] <- levels(data[[v]])[1]
        } else {
          grid[[v]] <- unique(data[[v]])[1]
        }
      }
    }

    # Make predictions
    if (intervals) {
      preds <- predict(model, grid, se.fit = TRUE)
      grid$fit <- preds$fit
      grid$se <- preds$se.fit
      grid$lower <- grid$fit - 1.96 * grid$se
      grid$upper <- grid$fit + 1.96 * grid$se
    } else {
      grid$fit <- predict(model, grid)
    }

    # Add by_value label
    grid$by_value <- bv
    if (!is.null(names(by_values)) && !is.na(match(bv, by_values))) {
      grid$by_label <- names(by_values)[match(bv, by_values)]
    } else {
      grid$by_label <- as.character(bv)
    }

    # Store in list
    grid_list[[length(grid_list) + 1]] <- grid
  }

  # Combine all grids
  final_grid <- do.call(rbind, grid_list)

  # Calculate slopes for numeric variables
  if (var_is_numeric) {
    # Group by by_value
    slopes <- lapply(by_values, function(bv) {
      # Get data for this by_value
      sub_grid <- final_grid[final_grid$by_value == bv, ]

      # If only one value, can't compute slope
      if (nrow(sub_grid) <= 1) {
        return(data.frame(
          by_value = bv,
          by_label = if (is.null(names(by_values))) as.character(bv) else names(by_values)[match(bv, by_values)],
          slope = NA,
          slope_se = NA
        ))
      }

      # Fit linear model to get slope
      slope_model <- lm(fit ~ .data[[var]], data = sub_grid)
      slope_coef <- coef(summary(slope_model))

      return(data.frame(
        by_value = bv,
        by_label = if (is.null(names(by_values))) as.character(bv) else names(by_values)[match(bv, by_values)],
        slope = slope_coef[2, 1],
        slope_se = slope_coef[2, 2]
      ))
    })

    # Combine all slopes
    final_slopes <- do.call(rbind, slopes)

    # Return both grid and slopes
    return(list(
      effects = final_grid,
      slopes = final_slopes
    ))
  } else {
    # For categorical variables, just return the effects
    return(final_grid)
  }
}
