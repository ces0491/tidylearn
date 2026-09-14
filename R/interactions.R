#' @title Interaction Analysis Functions for tidylearn
#' @name tidylearn-interactions
#' @description Functions for testing, visualizing, and analyzing interactions
#' @importFrom stats lm anova
#' @importFrom dplyr filter select mutate
#' @importFrom ggplot2 ggplot aes geom_line geom_point
#'   facet_wrap labs theme_minimal
NULL

#' Test for significant interactions between variables
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the base model without interactions,
#'   or a string that parses as one. \code{.} and \code{- var} are expanded
#'   against \code{data}.
#' @param var1 First variable to test for interactions
#' @param var2 Second variable to test for interactions
#'   (if NULL, tests var1 with all others)
#' @param all_pairs Logical; whether to test all variable pairs
#' @param categorical_only Logical; whether to only test categorical variables
#' @param numeric_only Logical; whether to only test numeric variables
#' @param mixed_only Logical; whether to only test numeric-categorical pairs
#' @param alpha Significance level for interaction tests
#' @return A data frame with one row per tested interaction pair, containing
#'   columns \code{var1}, \code{var2}, \code{p_value}, \code{significant}
#'   (logical), \code{delta_r2} (change in R-squared), and
#'   \code{f_statistic}, sorted by \code{p_value} ascending.
#' @examples
#' \donttest{
#' results <- tl_test_interactions(mtcars, mpg ~ wt + hp + cyl,
#'   var1 = "wt", var2 = "hp")
#' }
#' @export
tl_test_interactions <- function(data, formula, var1 = NULL, var2 = NULL,
                                 all_pairs = FALSE, categorical_only = FALSE,
                                 numeric_only = FALSE, mixed_only = FALSE,
                                 alpha = 0.05) {
  formula <- tl_as_formula(formula)
  tl_check_two_sided(formula, "tl_test_interactions")

  # The candidates are the main-effect terms of the expanded formula.
  # all.vars() cannot see the columns behind `.` and still counts a column
  # removed with `- id`, so the terms have to be expanded against the data.
  model_terms <- stats::terms(formula, data = data)
  term_labels <- attr(model_terms, "term.labels")
  predictors <- term_labels[attr(model_terms, "order") == 1]

  # A term can be a transformation such as log(x), so it is evaluated
  # against the data rather than looked up as a column name.
  var_types <- vapply(predictors, function(term) {
    x <- eval(str2lang(term), data, environment(formula))
    if (is.factor(x) || is.character(x)) {
      "categorical"
    } else if (is.numeric(x)) {
      "numeric"
    } else {
      "other"
    }
  }, character(1))

  # Generate pairs to test
  if (all_pairs) {
    # combn() on a single name counts from 1 to that value instead of
    # failing, so fewer than two predictors must mean no pairs.
    pairs <- if (length(predictors) >= 2) {
      combn(predictors, 2, simplify = FALSE)
    } else {
      list()
    }
  } else if (!is.null(var1) && !is.null(var2)) {
    # Test specific pair
    pairs <- list(c(var1, var2))
  } else if (!is.null(var1)) {
    # Test var1 with all other variables
    pairs <- lapply(setdiff(predictors, var1), function(v) c(var1, v))
  } else {
    stop("Must specify at least var1 or set all_pairs = TRUE", call. = FALSE)
  }

  # Filter pairs based on variable types. vapply() keeps an empty pair list
  # a logical index, where sapply() would return list() and break the subset.
  type_filter <- NULL
  if (categorical_only) {
    type_filter <- "categorical_only = TRUE"
    pairs <- pairs[vapply(
      pairs, function(p) all(var_types[p] %in% "categorical"), logical(1)
    )]
  } else if (numeric_only) {
    type_filter <- "numeric_only = TRUE"
    pairs <- pairs[vapply(
      pairs, function(p) all(var_types[p] %in% "numeric"), logical(1)
    )]
  } else if (mixed_only) {
    type_filter <- "mixed_only = TRUE"
    pairs <- pairs[vapply(pairs, function(p) {
      all(var_types[p] %in% c("categorical", "numeric")) &&
        var_types[p[1]] != var_types[p[2]]
    }, logical(1))]
  }

  # A pair whose interaction the formula already has cannot add anything:
  # both models were the same fit, reported as a row of NA
  existing <- term_labels[attr(model_terms, "order") == 2]
  in_formula <- vapply(pairs, function(p) {
    any(c(paste0(p[1], ":", p[2]), paste0(p[2], ":", p[1])) %in% existing)
  }, logical(1))
  pairs <- pairs[!in_formula]

  if (length(pairs) == 0) {
    # Classed, so tl_auto_interactions() can treat "nothing to test" as a
    # result rather than a failure
    stop(errorCondition(message = paste0(
      "No variable pairs left to test",
      if (!is.null(type_filter)) paste0(" after applying ", type_filter),
      if (any(in_formula)) " once pairs already in the formula are skipped",
      ". The predictors in 'formula' are: ",
      if (length(predictors) > 0) {
        paste(predictors, collapse = ", ")
      } else {
        "none"
      },
      ". Interaction testing needs at least two predictors of the types ",
      "selected."
    ), class = "tidylearn_no_interaction_pairs", call = NULL))
  }

  # The models are built from the expanded term labels. update() cannot add
  # a term to a formula that still contains `.`, because it calls terms()
  # without the data.
  build_formula <- function(labels) {
    built <- stats::reformulate(
      c(labels, tl_offset_terms(model_terms)),
      response = formula[[2]],
      intercept = attr(model_terms, "intercept") == 1
    )
    environment(built) <- environment(formula)
    built
  }
  base_model <- lm(build_formula(term_labels), data = data)

  # Test interactions
  results <- lapply(pairs, function(pair) {
    # Build model with interaction
    int_formula <- build_formula(
      c(term_labels, paste0(pair[1], ":", pair[2]))
    )
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
    data.frame(
      var1 = pair[1],
      var2 = pair[2],
      p_value = p_value,
      significant = p_value < alpha,
      delta_r2 = delta_r2,
      f_statistic = models_comparison$F[2]
    )
  })

  # Combine results
  all_results <- do.call(rbind, results)

  # Sort by significance
  all_results <- all_results[order(all_results$p_value), ]

  all_results
}

#' Plot interaction effects
#'
#' @param model A tidylearn model object
#' @param var1 First variable in the interaction
#' @param var2 Second variable in the interaction
#' @param n_points Number of points to use for continuous variables
#' @param fixed_values Named list of values for other variables in the model
#' @param confidence Logical; whether to show a 95\% confidence band. The
#'   band is drawn when one variable is numeric and the other categorical,
#'   and needs a model whose underlying fit is an \code{lm} or \code{glm};
#'   for a \code{glm} it is built on the link scale and transformed to the
#'   response scale. For any other fit a message says no band was drawn.
#' @param ... Additional arguments to pass to predict()
#' @return A \code{\link[ggplot2]{ggplot}} object. Two numeric variables are
#'   drawn as a filled contour of the prediction; a numeric and a categorical
#'   variable as one line per category; two categorical variables as dodged
#'   bars.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")
#'
#' # Two numeric variables are drawn as a filled contour over both ranges
#' tl_plot_interaction(model, var1 = "wt", var2 = "hp")
#'
#' # A numeric by categorical interaction is drawn as one line per level,
#' # each with a confidence band
#' am_model <- tl_model(transform(mtcars, am = factor(am)), mpg ~ wt * am,
#'   method = "linear")
#' tl_plot_interaction(am_model, var1 = "wt", var2 = "am")
#'
#' # Coarser grid, no band
#' tl_plot_interaction(am_model, var1 = "wt", var2 = "am",
#'   n_points = 20, confidence = FALSE)
#' }
tl_plot_interaction <- function(model, var1, var2,
                                n_points = 100,
                                fixed_values = NULL,
                                confidence = TRUE,
                                ...) {
  # Extract data
  data <- model$data

  # Check if variables exist in the model
  formula <- tl_as_formula(model$spec$formula)
  all_vars <- tl_interaction_variables(formula, data)
  if (!var1 %in% all_vars || !var2 %in% all_vars) {
    stop("Variables not found in model formula", call. = FALSE)
  }

  # Determine variable types
  var1_type <- if (
    is.factor(data[[var1]]) || is.character(data[[var1]])
  ) "categorical" else "numeric"
  var2_type <- if (
    is.factor(data[[var2]]) || is.character(data[[var2]])
  ) "categorical" else "numeric"

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

  # Add fixed values for other variables. The grid needs every column the
  # model frame evaluates, which for `y ~ . - qsec` still includes qsec.
  all_other_vars <- setdiff(tl_model_frame_columns(formula, data),
                            c(var1, var2))

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

  # The line and the confidence band are both on the response scale, so a
  # prediction type passed through ... would draw the line on another
  # one: type = "class" drew class codes 1 and 2 over a probability band
  if ("type" %in% names(list(...))) {
    stop("tl_plot_interaction() draws predictions on the response scale ",
         "and takes no 'type' argument.", call. = FALSE)
  }

  # Make predictions -- tidylearn predict returns a tibble with .pred
  predictions <- predict(model, grid, ...)
  if (is.data.frame(predictions) && ".pred" %in% names(predictions)) {
    grid$prediction <- predictions$.pred
  } else {
    grid$prediction <- predictions
  }
  y_col <- "prediction"
  lower_col <- NULL
  upper_col <- NULL

  # tidylearn's predict() returns no interval, so the band comes from the
  # underlying lm or glm fit. Only the line plots have a place to draw it.
  is_line_plot <- xor(var1_type == "categorical", var2_type == "categorical")
  if (confidence && is_line_plot) {
    band <- tl_interaction_band(model, grid)
    if (is.null(band)) {
      message(
        "No confidence band drawn: it needs standard errors from a linear ",
        "or generalised linear model fit, and this model's fit is of class '",
        class(model$fit)[1], "'. Set confidence = FALSE to skip this message."
      )
    } else {
      grid$.lower <- band$lower
      grid$.upper <- band$upper
      lower_col <- ".lower"
      upper_col <- ".upper"
    }
  }

  # Create plot
  if (var1_type == "numeric" && var2_type == "categorical") {
    # Line plot with x = var1, color = var2
    p <- ggplot2::ggplot(grid, ggplot2::aes(
      x = .data[[var1]], y = .data[[y_col]],
      color = .data[[var2]]
    )) +
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
        ggplot2::aes(
          ymin = .data[[lower_col]],
          ymax = .data[[upper_col]],
          fill = .data[[var2]]
        ),
        alpha = 0.2,
        linetype = 0
      )
    }
  } else if (var1_type == "categorical" && var2_type == "numeric") {
    # Line plot with x = var2, color = var1
    p <- ggplot2::ggplot(grid, ggplot2::aes(
      x = .data[[var2]], y = .data[[y_col]],
      color = .data[[var1]]
    )) +
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
        ggplot2::aes(
          ymin = .data[[lower_col]],
          ymax = .data[[upper_col]],
          fill = .data[[var1]]
        ),
        alpha = 0.2,
        linetype = 0
      )
    }
  } else if (var1_type == "numeric" && var2_type == "numeric") {
    # Contour plot or heat map
    p <- ggplot2::ggplot(grid, ggplot2::aes(
      x = .data[[var1]], y = .data[[var2]],
      z = .data[[y_col]]
    )) +
      ggplot2::geom_contour_filled() +
      ggplot2::labs(
        title = paste("Interaction between", var1, "and", var2),
        x = var1,
        y = var2,
        fill = "Predicted Value"
      )
  } else {
    # Categorical x Categorical: Faceted bar plot
    p <- ggplot2::ggplot(grid, ggplot2::aes(
      x = .data[[var1]], y = .data[[y_col]],
      fill = .data[[var2]]
    )) +
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

  p
}

#' Find important interactions automatically
#'
#' @param data A data frame containing the data
#' @param formula A formula specifying the base model without interactions
#' @param top_n Number of top interactions to return
#' @param min_r2_change Minimum change in R-squared to consider
#' @param max_p_value Maximum p-value for significance
#' @param exclude_vars Character vector of predictor variables that may not
#'   appear in a selected interaction. They stay in the model as main effects.
#'   Every name must be a predictor in \code{formula}.
#' @return A tidylearn model object (class \code{"tidylearn_model"}) fitted
#'   with the top significant interaction terms added to the formula.
#'   The interaction test results and selected interactions are stored as
#'   attributes \code{"interaction_tests"} and
#'   \code{"selected_interactions"}.
#' @examples
#' \donttest{
#' model <- tl_auto_interactions(mtcars, mpg ~ wt + hp + cyl, top_n = 2)
#' }
#' @export
tl_auto_interactions <- function(data, formula, top_n = 3, min_r2_change = 0.01,
                                 max_p_value = 0.05, exclude_vars = NULL) {
  formula <- tl_as_formula(formula)
  tl_check_two_sided(formula, "tl_auto_interactions")

  # A misspelt exclusion would otherwise exclude nothing without a word.
  if (!is.null(exclude_vars)) {
    if (!is.character(exclude_vars)) {
      stop(
        "'exclude_vars' must be a character vector of predictor names; got ",
        paste(class(exclude_vars), collapse = "/"), ".",
        call. = FALSE
      )
    }
    unknown <- setdiff(exclude_vars, tl_interaction_variables(formula, data))
    if (length(unknown) > 0) {
      stop(
        "'exclude_vars' names variables that are not predictors in ",
        "'formula': ", paste(unknown, collapse = ", "), ".",
        call. = FALSE
      )
    }
  }

  # Test all interactions
  # With every pair already in the formula, or fewer than two predictors,
  # there is nothing to add: return the model as specified, as when no
  # interaction is significant
  test_results <- tryCatch(
    tl_test_interactions(data, formula, all_pairs = TRUE,
                         alpha = max_p_value),
    tidylearn_no_interaction_pairs = function(e) NULL
  )
  if (is.null(test_results)) {
    message("No interactions left to test; returning the model as specified")
    return(tl_model(data, formula, method = "linear"))
  }

  # Drop every pair that involves an excluded variable before any selection,
  # so neither the chosen terms nor the stored test results include it. A
  # term such as log(z) involves z, so the check reads the term's variables.
  if (!is.null(exclude_vars)) {
    involves_excluded <- function(term) {
      any(all.vars(str2lang(term)) %in% exclude_vars)
    }
    keep <- !vapply(test_results$var1, involves_excluded, logical(1)) &
      !vapply(test_results$var2, involves_excluded, logical(1))
    test_results <- test_results[keep, , drop = FALSE]
  }

  # Filter significant interactions
  significant <- test_results |>
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
  interaction_terms <- paste0(top_interactions$var1, ":", top_interactions$var2)

  # The formula is rebuilt from its expanded terms, since update() cannot
  # add a term to a formula that contains `.`.
  model_terms <- stats::terms(formula, data = data)
  new_formula <- stats::reformulate(
    c(attr(model_terms, "term.labels"), interaction_terms,
      tl_offset_terms(model_terms)),
    response = formula[[2]],
    intercept = attr(model_terms, "intercept") == 1
  )
  environment(new_formula) <- environment(formula)

  # Fit model with interactions
  interaction_model <- tl_model(data, new_formula, method = "linear")

  # Add interaction test results to model
  attr(interaction_model, "interaction_tests") <- test_results
  attr(interaction_model, "selected_interactions") <- top_interactions

  interaction_model
}

#' Calculate partial effects based on a model with interactions
#'
#' @param model A tidylearn model object
#' @param var Variable to calculate effects for
#' @param by_var Variable to calculate effects by (interaction variable)
#' @param at_values Named list of values at which to hold other variables
#' @param intervals Logical; whether to add 95\% confidence limits as columns
#'   \code{lower} and \code{upper}. This needs a model whose underlying fit is
#'   an \code{lm} or \code{glm}; for any other fit a message is shown and only
#'   point estimates are returned. For a \code{glm} the interval is built on
#'   the link scale and transformed to the response scale.
#' @return For numeric \code{var}: a list with \code{effects} (data frame of
#'   predicted values across the variable range for each value of
#'   \code{by_var}) and \code{slopes} (data frame with the slope of
#'   \code{var} at each value of \code{by_var}). For categorical
#'   \code{var}: a data frame of predicted values at each factor level for
#'   each level of \code{by_var}. A numeric \code{by_var} is evaluated at its
#'   quartiles; quartiles that tie are evaluated once, with a label naming
#'   each quartile they stand for, such as \code{"Q0/Q25"}.
#'
#'   \code{fit}, \code{lower}, \code{upper} and \code{slope} are on the
#'   response scale whatever \code{intervals} is set to: predicted
#'   probabilities for a logistic model. \code{slope} is the slope of a
#'   straight line fitted to \code{fit} across the range of \code{var}, so
#'   for a non-linear link it is an average rate of change over that range.
#'
#'   \code{slopes$slope_se} is the standard error of a straight line fitted
#'   to the prediction grid, not the sampling uncertainty of the marginal
#'   effect. For a linear model the grid is exactly linear in \code{var},
#'   so this is near zero by construction and should not be read as a
#'   precise estimate. Use \code{summary(model$fit)} for inference on the
#'   interaction coefficient itself.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")
#'
#' # How the effect of weight changes across horsepower
#' effects <- tl_interaction_effects(model, var = "wt", by_var = "hp")
#' head(effects$effects)
#' effects$slopes
#'
#' # slopes$slope_se describes the fitted grid, not the sampling
#' # uncertainty of the marginal effect -- for that, read the coefficient
#' summary(model$fit)$coefficients["wt:hp", ]
#' }
tl_interaction_effects <- function(model, var, by_var,
                                   at_values = NULL,
                                   intervals = TRUE) {
  # Extract data
  data <- model$data
  formula <- tl_as_formula(model$spec$formula)

  # Check if variables exist in the model
  all_vars <- tl_interaction_variables(formula, data)
  if (!var %in% all_vars || !by_var %in% all_vars) {
    stop("Variables not found in model formula", call. = FALSE)
  }

  # An effect of var needs var to vary. A constant column fitted as an
  # aliased coefficient and failed later with "subscript out of bounds".
  if (length(unique(stats::na.omit(data[[var]]))) < 2) {
    stop("'", var, "' takes a single value in the model's data, so it has ",
         "no effect to estimate at any level of '", by_var, "'.",
         call. = FALSE)
  }

  # Check if the interaction term exists in the model. terms() needs the
  # data to expand a `.` in the formula.
  int_term <- paste0(var, ":", by_var)
  rev_int_term <- paste0(by_var, ":", var)
  term_labels <- attr(stats::terms(formula, data = data), "term.labels")

  has_interaction <- int_term %in% term_labels || rev_int_term %in% term_labels

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
    # For continuous by_var, use quantiles. A discrete variable such as
    # cyl has tied quartiles, and each tie would otherwise produce a
    # duplicate grid block and a repeated slope row. Tied quartiles are
    # kept once, labelled with every quartile they stand for ("Q0/Q25").
    quartiles <- stats::quantile(
      data[[by_var]], probs = seq(0, 1, 0.25),
      na.rm = TRUE, names = FALSE
    )
    by_values <- unique(quartiles)
    names(by_values) <- vapply(by_values, function(value) {
      paste0("Q", seq(0, 100, 25)[quartiles == value], collapse = "/")
    }, character(1))
  }

  # Standard errors come from the underlying lm or glm fit. For any other
  # fit the point estimates are still valid, and intervals = TRUE is the
  # default, so the function falls back to point estimates with a message
  # rather than failing on a default call.
  if (intervals && !inherits(model$fit, "lm")) {
    message(
      "Confidence intervals need standard errors from a linear or ",
      "generalised linear model fit; this model's fit is of class '",
      class(model$fit)[1], "', so only point estimates are returned. ",
      "Set intervals = FALSE to skip this message."
    )
    intervals <- FALSE
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

    # Add values for other variables, including any the formula names but
    # removes, which the model frame still evaluates
    other_vars <- setdiff(tl_model_frame_columns(formula, data),
                          c(var, by_var))
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

    # Make predictions. tidylearn's predict() has no standard errors, so
    # the interval comes from the underlying fit, on the response scale
    # that predict() reports.
    if (intervals) {
      band <- tl_interaction_band(model, grid)
      grid$fit <- band$fit
      grid$lower <- band$lower
      grid$upper <- band$upper
    } else {
      preds <- predict(model, grid)
      grid$fit <- if (is.data.frame(preds)) preds$.pred else preds
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
        data.frame(
          by_value = bv,
          by_label = if (is.null(names(by_values))) {
            as.character(bv)
          } else {
            names(by_values)[match(bv, by_values)]
          },
          slope = NA,
          slope_se = NA
        )
      }

      # Fit linear model to get slope. The response here is the model's
      # own fitted values on a regular grid, so for a linear model the
      # points lie exactly on a line and summary.lm() warns about an
      # "essentially perfect fit". That is expected by construction, not a
      # problem with the data, so the warning is not passed on.
      #
      # Note this makes slope_se the standard error of the fit to the
      # prediction grid, not the uncertainty in the marginal effect
      # itself -- for a linear model it is near zero by construction.
      slope_formula <- stats::as.formula(paste("fit ~", var))
      slope_model <- lm(slope_formula, data = sub_grid)
      slope_coef <- withCallingHandlers(
        coef(summary(slope_model)),
        warning = function(w) {
          if (grepl("perfect fit", conditionMessage(w), fixed = TRUE)) {
            invokeRestart("muffleWarning")
          }
        }
      )

      data.frame(
        by_value = bv,
        by_label = if (is.null(names(by_values))) {
          as.character(bv)
        } else {
          names(by_values)[match(bv, by_values)]
        },
        slope = slope_coef[2, 1],
        slope_se = slope_coef[2, 2]
      )
    })

    # Combine all slopes
    final_slopes <- do.call(rbind, slopes)

    # Return both grid and slopes
    list(
      effects = final_grid,
      slopes = final_slopes
    )
  } else {
    # For categorical variables, just return the effects
    final_grid
  }
}

#' Variables used by the terms of a model formula
#'
#' Expands the formula against the data, so `.` becomes the data's columns
#' and a column removed with `- id` is left out, then returns the variables
#' the remaining terms use. A term such as log(x) contributes x.
#'
#' @param formula A model formula
#' @param data The data the formula is expanded against
#' @return A character vector of variable names, without the response
#' @keywords internal
#' @noRd
tl_interaction_variables <- function(formula, data) {
  model_terms <- stats::terms(formula, data = data)
  factors <- attr(model_terms, "factors")
  # An offset's row is all zeros too, yet its variables are still needed
  # to evaluate the model on new data
  offset_vars <- unlist(lapply(tl_offset_terms(model_terms), function(term) {
    all.vars(str2lang(term))
  }))
  if (length(factors) == 0) {
    return(unique(as.character(offset_vars)))
  }
  # The response row is all zeros, as is the row of any variable that only
  # appears in a removed term.
  used <- rownames(factors)[rowSums(factors) > 0]
  unique(c(
    unlist(lapply(used, function(term) all.vars(str2lang(term)))),
    offset_vars
  ))
}

#' Data columns a model frame evaluates, excluding the response
#'
#' Wider than \code{tl_interaction_variables()}: a column removed with
#' \code{- qsec} is used by no term but is still evaluated when the model
#' predicts, so a prediction grid without it fails with "object 'qsec' not
#' found".
#'
#' @param formula A model formula
#' @param data The model's data
#' @return A character vector of column names
#' @keywords internal
#' @noRd
tl_model_frame_columns <- function(formula, data) {
  predictor_terms <- stats::delete.response(stats::terms(formula, data = data))
  intersect(all.vars(predictor_terms), names(data))
}

#' Refuse a formula with no response
#'
#' The interaction testers read the response as \code{formula[[2]]}, which
#' on \code{~ a + b} is the right-hand side: the first predictor became
#' the response and the tests reported nonsense beside "essentially
#' perfect fit" warnings.
#'
#' @param formula A formula
#' @param caller Name of the calling function, for the message
#' @return NULL, invisibly
#' @keywords internal
#' @noRd
tl_check_two_sided <- function(formula, caller) {
  if (length(formula) != 3) {
    stop(caller, "() needs a response to test interactions against: ",
         "write the formula as y ~ predictors.", call. = FALSE)
  }
  invisible(NULL)
}

#' Offset terms of a terms object, as text
#'
#' \code{term.labels} never includes an \code{offset()}, so a formula
#' rebuilt from the labels alone loses it -- and the model fitted from that
#' formula differs from the one the caller specified.
#'
#' @param model_terms A \code{terms} object
#' @return A character vector of offset expressions, possibly empty
#' @keywords internal
#' @noRd
tl_offset_terms <- function(model_terms) {
  offsets <- attr(model_terms, "offset")
  if (is.null(offsets)) {
    return(character(0))
  }
  variables <- as.list(attr(model_terms, "variables"))[-1]
  vapply(variables[offsets], function(term) {
    paste(deparse(term), collapse = " ")
  }, character(1))
}

#' Confidence band for predictions from an lm or glm fit
#'
#' @param model A tidylearn model object
#' @param newdata Data frame of predictor values
#' @param level Confidence level
#' @return A data frame with columns fit, lower and upper on the response
#'   scale, or NULL when the fit is not an lm or glm
#' @keywords internal
#' @noRd
tl_interaction_band <- function(model, newdata, level = 0.95) {
  fit <- model$fit
  if (!inherits(fit, "lm")) {
    return(NULL)
  }

  # The underlying fit is called directly, so it needs the same feature
  # construction that predict.tidylearn_model() applies.
  newdata <- apply_feature_transform(model, newdata)

  if (inherits(fit, "glm")) {
    # A glm's standard errors are on the link scale, where the estimate is
    # approximately normal. The interval is built there and mapped through
    # the inverse link, which keeps a probability interval inside [0, 1].
    # pmin/pmax keep lower below upper for a decreasing inverse link.
    link <- stats::predict(fit, newdata = newdata, type = "link",
                           se.fit = TRUE)
    estimate <- as.vector(link$fit)
    margin <- stats::qnorm(1 - (1 - level) / 2) * as.vector(link$se.fit)
    linkinv <- stats::family(fit)$linkinv
    bound_a <- linkinv(estimate - margin)
    bound_b <- linkinv(estimate + margin)
    data.frame(
      fit = linkinv(estimate),
      lower = pmin(bound_a, bound_b),
      upper = pmax(bound_a, bound_b)
    )
  } else {
    ci <- stats::predict(fit, newdata = newdata, interval = "confidence",
                         level = level)
    data.frame(
      fit = unname(ci[, "fit"]),
      lower = unname(ci[, "lwr"]),
      upper = unname(ci[, "upr"])
    )
  }
}
