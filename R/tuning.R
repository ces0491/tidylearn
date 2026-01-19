#' @title Hyperparameter Tuning Functions for tidylearn
#' @name tidylearn-tuning
#' @description Functions for automatic hyperparameter tuning and selection
#' @importFrom stats model.matrix as.formula
#' @importFrom dplyr %>% filter select mutate arrange
#' @importFrom tidyr crossing
NULL

#' Tune hyperparameters for a model using grid search
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param method The modeling method to tune
#' @param param_grid A named list of parameter values to tune
#' @param folds Number of cross-validation folds
#' @param metric Metric to optimize
#' @param maximize Logical; whether to maximize (TRUE) or minimize (FALSE) the metric
#' @param verbose Logical; whether to print progress
#' @param ... Additional arguments passed to tl_model
#' @return A list with the best model and tuning results
#' @export
tl_tune_grid <- function(data, formula, method, param_grid, folds = 5,
                         metric = NULL, maximize = NULL, verbose = TRUE, ...) {
  # Input validation
  if (!is.list(param_grid)) {
    stop("param_grid must be a named list", call. = FALSE)
  }

  # Determine if classification or regression
  response_var <- all.vars(formula)[1]
  y <- data[[response_var]]
  is_classification <- is.factor(y) || is.character(y) || (is.numeric(y) && length(unique(y)) <= 10)

  # Default metric based on problem type
  if (is.null(metric)) {
    if (is_classification) {
      metric <- "accuracy"
      if (is.null(maximize)) maximize <- TRUE
    } else {
      metric <- "rmse"
      if (is.null(maximize)) maximize <- FALSE
    }
  }

  # Create parameter grid
  param_df <- do.call(tidyr::crossing, param_grid)

  if (verbose) {
    message("Tuning ", method, " model with ", nrow(param_df), " parameter combinations")
    message("Cross-validation with ", folds, " folds")
    message("Optimizing for ", metric, ", ", ifelse(maximize, "maximizing", "minimizing"))
  }

  # Create cross-validation splits
  cv_splits <- rsample::vfold_cv(data, v = folds)

  # Initialize results storage
  tuning_results <- list()

  # Loop through parameter combinations
  for (i in 1:nrow(param_df)) {
    if (verbose) {
      message("Parameter set ", i, " of ", nrow(param_df), ": ",
              paste(names(param_df), param_df[i,], sep = "=", collapse = ", "))
    }

    # Extract parameters for this iteration
    params <- as.list(param_df[i,])

    # Initialize metrics storage for this parameter set
    fold_metrics <- numeric(folds)

    # Cross-validation loop
    for (j in 1:folds) {
      # Get training and validation data for this fold
      train_fold <- rsample::analysis(cv_splits$splits[[j]])
      valid_fold <- rsample::assessment(cv_splits$splits[[j]])

      # Fit model with current parameters
      model_args <- c(
        list(
          data = train_fold,
          formula = formula,
          method = method
        ),
        params,
        list(...)
      )

      # Train model
      fold_model <- tryCatch({
        do.call(tl_model, model_args)
      }, error = function(e) {
        warning("Error fitting model with parameters: ",
                paste(names(params), params, sep = "=", collapse = ", "),
                ". Error: ", e$message)
        return(NULL)
      })

      # If model failed, skip this fold
      if (is.null(fold_model)) {
        fold_metrics[j] <- NA
        next
      }

      # Evaluate model
      eval_metrics <- tl_evaluate(fold_model, valid_fold, metrics = metric)

      # Store metric value
      fold_metrics[j] <- eval_metrics$value[eval_metrics$metric == metric]
    }

    # Calculate mean metric across folds
    mean_metric <- mean(fold_metrics, na.rm = TRUE)

    # Store result for this parameter set
    tuning_results[[i]] <- c(
      params,
      list(
        mean_metric = mean_metric,
        fold_metrics = fold_metrics,
        metric_name = metric,
        maximize = maximize
      )
    )

    if (verbose) {
      message("  Mean ", metric, ": ", round(mean_metric, 4))
    }
  }

  # Convert results to data frame
  results_df <- do.call(rbind, lapply(tuning_results, function(x) {
    df <- data.frame(
      mean_metric = x$mean_metric
    )

    # Add parameters
    for (param in names(param_grid)) {
      df[[param]] <- x[[param]]
    }

    return(df)
  }))

  # Find best parameter set
  if (maximize) {
    best_idx <- which.max(results_df$mean_metric)
  } else {
    best_idx <- which.min(results_df$mean_metric)
  }

  # Extract best parameters
  best_params <- as.list(results_df[best_idx, names(param_grid)])

  if (verbose) {
    message("Best parameters found: ",
            paste(names(best_params), best_params, sep = "=", collapse = ", "))
    message("Best ", metric, ": ", round(results_df$mean_metric[best_idx], 4))
  }

  # Fit final model with best parameters
  final_model_args <- c(
    list(
      data = data,
      formula = formula,
      method = method
    ),
    best_params,
    list(...)
  )

  final_model <- do.call(tl_model, final_model_args)

  # Add tuning results to model
  attr(final_model, "tuning_results") <- list(
    param_grid = param_grid,
    results = results_df,
    best_params = best_params,
    best_metric = results_df$mean_metric[best_idx],
    metric = metric,
    maximize = maximize
  )

  return(final_model)
}

#' Tune hyperparameters for a model using random search
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param method The modeling method to tune
#' @param param_space A named list of parameter spaces to sample from
#' @param n_iter Number of random parameter combinations to try
#' @param folds Number of cross-validation folds
#' @param metric Metric to optimize
#' @param maximize Logical; whether to maximize (TRUE) or minimize (FALSE) the metric
#' @param verbose Logical; whether to print progress
#' @param seed Random seed for reproducibility
#' @param ... Additional arguments passed to tl_model
#' @return A list with the best model and tuning results
#' @export
tl_tune_random <- function(data, formula, method, param_space, n_iter = 10, folds = 5,
                           metric = NULL, maximize = NULL, verbose = TRUE, seed = NULL, ...) {
  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Input validation
  if (!is.list(param_space)) {
    stop("param_space must be a named list", call. = FALSE)
  }

  # Determine if classification or regression
  response_var <- all.vars(formula)[1]
  y <- data[[response_var]]
  is_classification <- is.factor(y) || is.character(y) || (is.numeric(y) && length(unique(y)) <= 10)

  # Default metric based on problem type
  if (is.null(metric)) {
    if (is_classification) {
      metric <- "accuracy"
      if (is.null(maximize)) maximize <- TRUE
    } else {
      metric <- "rmse"
      if (is.null(maximize)) maximize <- FALSE
    }
  }

  if (verbose) {
    message("Random search for ", method, " model with ", n_iter, " iterations")
    message("Cross-validation with ", folds, " folds")
    message("Optimizing for ", metric, ", ", ifelse(maximize, "maximizing", "minimizing"))
  }

  # Create cross-validation splits
  cv_splits <- rsample::vfold_cv(data, v = folds)

  # Initialize results storage
  tuning_results <- list()

  # Generate random parameter combinations
  param_combinations <- list()
  for (i in 1:n_iter) {
    params <- list()
    for (param_name in names(param_space)) {
      param_def <- param_space[[param_name]]

      if (is.function(param_def)) {
        # Custom sampling function
        params[[param_name]] <- param_def()
      } else if (is.numeric(param_def) && length(param_def) == 2) {
        # Numeric range: [min, max]
        params[[param_name]] <- runif(1, param_def[1], param_def[2])
      } else if (is.numeric(param_def) && length(param_def) == 3 && param_def[3] == "log") {
        # Log-uniform range: [min, max, "log"]
        params[[param_name]] <- exp(runif(1, log(param_def[1]), log(param_def[2])))
      } else if (is.integer(param_def) || (is.numeric(param_def) && all(param_def == floor(param_def)))) {
        # Integer range or discrete values
        if (length(param_def) == 2) {
          # Integer range: [min, max]
          params[[param_name]] <- sample(param_def[1]:param_def[2], 1)
        } else {
          # Discrete values
          params[[param_name]] <- sample(param_def, 1)
        }
      } else if (is.character(param_def) || is.factor(param_def)) {
        # Categorical parameter
        params[[param_name]] <- sample(param_def, 1)
      } else if (is.logical(param_def)) {
        # Logical parameter
        params[[param_name]] <- sample(c(TRUE, FALSE), 1)
      } else {
        stop("Unsupported parameter space definition for ", param_name, call. = FALSE)
      }
    }
    param_combinations[[i]] <- params
  }

  # Loop through parameter combinations
  for (i in 1:n_iter) {
    params <- param_combinations[[i]]

    if (verbose) {
      message("Iteration ", i, " of ", n_iter, ": ",
              paste(names(params), round(unlist(params), 4), sep = "=", collapse = ", "))
    }

    # Initialize metrics storage for this parameter set
    fold_metrics <- numeric(folds)

    # Cross-validation loop
    for (j in 1:folds) {
      # Get training and validation data for this fold
      train_fold <- rsample::analysis(cv_splits$splits[[j]])
      valid_fold <- rsample::assessment(cv_splits$splits[[j]])

      # Fit model with current parameters
      model_args <- c(
        list(
          data = train_fold,
          formula = formula,
          method = method
        ),
        params,
        list(...)
      )

      # Train model
      fold_model <- tryCatch({
        do.call(tl_model, model_args)
      }, error = function(e) {
        warning("Error fitting model with parameters: ",
                paste(names(params), params, sep = "=", collapse = ", "),
                ". Error: ", e$message)
        return(NULL)
      })

      # If model failed, skip this fold
      if (is.null(fold_model)) {
        fold_metrics[j] <- NA
        next
      }

      # Evaluate model
      eval_metrics <- tl_evaluate(fold_model, valid_fold, metrics = metric)

      # Store metric value
      fold_metrics[j] <- eval_metrics$value[eval_metrics$metric == metric]
    }

    # Calculate mean metric across folds
    mean_metric <- mean(fold_metrics, na.rm = TRUE)

    # Store result for this parameter set
    tuning_results[[i]] <- c(
      params,
      list(
        mean_metric = mean_metric,
        fold_metrics = fold_metrics,
        metric_name = metric,
        maximize = maximize
      )
    )

    if (verbose) {
      message("  Mean ", metric, ": ", round(mean_metric, 4))
    }
  }

  # Convert results to data frame
  results_df <- do.call(rbind, lapply(seq_along(tuning_results), function(i) {
    result <- tuning_results[[i]]

    df <- data.frame(
      iteration = i,
      mean_metric = result$mean_metric
    )

    # Add parameters
    for (param in names(param_space)) {
      df[[param]] <- result[[param]]
    }

    return(df)
  }))

  # Find best parameter set
  if (maximize) {
    best_idx <- which.max(results_df$mean_metric)
  } else {
    best_idx <- which.min(results_df$mean_metric)
  }

  # Extract best parameters
  best_params <- as.list(results_df[best_idx, names(param_space)])

  if (verbose) {
    message("Best parameters found: ",
            paste(names(best_params), round(unlist(best_params), 4), sep = "=", collapse = ", "))
    message("Best ", metric, ": ", round(results_df$mean_metric[best_idx], 4))
  }

  # Fit final model with best parameters
  final_model_args <- c(
    list(
      data = data,
      formula = formula,
      method = method
    ),
    best_params,
    list(...)
  )

  final_model <- do.call(tl_model, final_model_args)

  # Add tuning results to model
  attr(final_model, "tuning_results") <- list(
    param_space = param_space,
    results = results_df,
    best_params = best_params,
    best_metric = results_df$mean_metric[best_idx],
    metric = metric,
    maximize = maximize
  )

  return(final_model)
}

#' Plot hyperparameter tuning results
#'
#' @param model A tidylearn model object with tuning results
#' @param top_n Number of top parameter sets to highlight
#' @param param1 First parameter to plot (for 2D grid or scatter plots)
#' @param param2 Second parameter to plot (for 2D grid or scatter plots)
#' @param plot_type Type of plot: "scatter", "grid", "parallel", "importance"
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_point geom_tile scale_fill_gradient2 labs theme_minimal
#' @export
tl_plot_tuning_results <- function(model, top_n = 5, param1 = NULL, param2 = NULL,
                                   plot_type = "scatter") {
  # Check if model has tuning results
  tuning_results <- attr(model, "tuning_results")
  if (is.null(tuning_results)) {
    stop("Model does not have tuning results", call. = FALSE)
  }

  # Extract results data frame
  results_df <- tuning_results$results

  # Get parameter names
  param_names <- setdiff(names(results_df), c("iteration", "mean_metric"))

  # Default parameters for plotting if not specified
  if (is.null(param1) && length(param_names) > 0) {
    param1 <- param_names[1]
  }

  if (is.null(param2) && length(param_names) > 1) {
    param2 <- param_names[2]
  }

  # Create ranking column
  results_df$rank <- rank(if (tuning_results$maximize) -results_df$mean_metric else results_df$mean_metric)
  results_df$is_top <- results_df$rank <= top_n

  # Create plot based on plot_type
  if (plot_type == "scatter" && !is.null(param1) && !is.null(param2)) {
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
        midpoint = median(results_df$mean_metric, na.rm = TRUE)
      ) +
      ggplot2::scale_size_manual(values = c(3, 5)) +
      ggplot2::scale_shape_manual(values = c(16, 18)) +
      ggplot2::labs(
        title = "Hyperparameter Tuning Results",
        subtitle = paste("Optimizing", tuning_results$metric, "- top", top_n, "results highlighted"),
        x = param1,
        y = param2,
        color = tuning_results$metric,
        size = "Top Result",
        shape = "Top Result"
      ) +
      ggplot2::theme_minimal()

  } else if (plot_type == "grid" && !is.null(param1) && !is.null(param2)) {
    # Heat map for grid search
    # Check if parameters have few unique values for a grid
    if (length(unique(results_df[[param1]])) <= 20 && length(unique(results_df[[param2]])) <= 20) {
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
          ggplot2::aes(label = round(.data$mean_metric, 3)),
          color = "black",
          size = 3
        ) +
        ggplot2::scale_fill_gradient2(
          low = "blue",
          high = "red",
          mid = "white",
          midpoint = median(results_df$mean_metric, na.rm = TRUE)
        ) +
        ggplot2::labs(
          title = "Hyperparameter Tuning Results Grid",
          subtitle = paste("Optimizing", tuning_results$metric),
          x = param1,
          y = param2,
          fill = tuning_results$metric
        ) +
        ggplot2::theme_minimal()
    } else {
      warning("Parameters have too many unique values for a grid plot. Using scatter plot instead.")
      plot_type <- "scatter"
      return(tl_plot_tuning_results(model, top_n, param1, param2, "scatter"))
    }

  } else if (plot_type == "parallel") {
    # Parallel coordinates plot for multidimensional exploration
    # Normalize parameter values to 0-1 scale for better visualization
    results_norm <- results_df

    for (param in param_names) {
      if (is.numeric(results_df[[param]])) {
        param_min <- min(results_df[[param]], na.rm = TRUE)
        param_max <- max(results_df[[param]], na.rm = TRUE)
        if (param_max > param_min) {
          results_norm[[paste0(param, "_norm")]] <- (results_df[[param]] - param_min) / (param_max - param_min)
        } else {
          results_norm[[paste0(param, "_norm")]] <- 0.5
        }
      } else {
        # For categorical parameters, convert to numeric
        results_norm[[paste0(param, "_norm")]] <- as.numeric(factor(results_df[[param]])) / length(unique(results_df[[param]]))
      }
    }

    # Prepare data for parallel coordinates
    param_norm_names <- paste0(param_names, "_norm")

    # Convert to long format
    plot_data <- tidyr::pivot_longer(
      results_norm,
      cols = all_of(param_norm_names),
      names_to = "parameter",
      values_to = "value"
    )

    # Remove _norm suffix for plotting
    plot_data$parameter <- gsub("_norm$", "", plot_data$parameter)

    # Create parallel coordinates plot
    p <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(
        x = .data$parameter,
        y = .data$value,
        group = .data$rank,
        color = .data$mean_metric,
        size = .data$is_top,
        alpha = .data$is_top
      )
    ) +
      ggplot2::geom_line() +
      ggplot2::scale_color_gradient2(
        low = "blue",
        high = "red",
        mid = "white",
        midpoint = median(results_df$mean_metric, na.rm = TRUE)
      ) +
      ggplot2::scale_size_manual(values = c(0.5, 1.5)) +
      ggplot2::scale_alpha_manual(values = c(0.3, 1)) +
      ggplot2::labs(
        title = "Parallel Coordinates Plot of Hyperparameter Tuning Results",
        subtitle = paste("Optimizing", tuning_results$metric, "- top", top_n, "results highlighted"),
        x = "Parameter",
        y = "Normalized Value",
        color = tuning_results$metric,
        size = "Top Result",
        alpha = "Top Result"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        panel.grid.major.x = ggplot2::element_line(color = "gray90"),
        panel.grid.minor = ggplot2::element_blank(),
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
      )

  } else if (plot_type == "importance") {
    # Parameter importance plot
    # Calculate correlation between parameters and metric
    param_importance <- lapply(param_names, function(param) {
      if (is.numeric(results_df[[param]])) {
        # For numeric parameters, use correlation
        cor_val <- cor(results_df[[param]], results_df$mean_metric, use = "pairwise.complete.obs")
        return(data.frame(
          parameter = param,
          importance = abs(cor_val),
          correlation = cor_val
        ))
      } else {
        # For categorical parameters, use ANOVA
        # Convert to factor
        results_df[[param]] <- as.factor(results_df[[param]])

        # Run ANOVA
        anova_result <- summary(aov(mean_metric ~ .data[[param]], data = results_df))

        # Calculate eta squared
        ss_total <- sum(anova_result[[1]]$"Sum Sq")
        ss_param <- anova_result[[1]]$"Sum Sq"[1]
        eta_squared <- ss_param / ss_total

        return(data.frame(
          parameter = param,
          importance = eta_squared,
          correlation = NA
        ))
      }
    })

    # Combine results
    importance_df <- do.call(rbind, param_importance)

    # Sort by importance
    importance_df <- importance_df[order(importance_df$importance, decreasing = TRUE), ]

    # Create bar plot
    p <- ggplot2::ggplot(
      importance_df,
      ggplot2::aes(
        x = reorder(.data$parameter, .data$importance),
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
        subtitle = paste("Correlation with", tuning_results$metric),
        x = "Parameter",
        y = "Importance (|Correlation| or Eta Squared)",
        fill = "Correlation\n(numeric only)"
      ) +
      ggplot2::theme_minimal()
  } else {
    stop("Invalid plot_type or insufficient parameters for plotting", call. = FALSE)
  }

  return(p)
}

#' Create pre-defined parameter grids for common models
#'
#' @param method Model method ("tree", "forest", "boost", "svm", etc.)
#' @param size Grid size: "small", "medium", "large"
#' @param is_classification Whether the task is classification or regression
#' @return A named list of parameter values to tune
#' @export
tl_default_param_grid <- function(method, size = "medium", is_classification = TRUE) {
  # Input validation
  size <- match.arg(size, c("small", "medium", "large"))

  # Define default grids for different methods
  if (method == "tree") {
    if (size == "small") {
      return(list(
        cp = c(0.01, 0.1),
        minsplit = c(10, 20)
      ))
    } else if (size == "medium") {
      return(list(
        cp = c(0.001, 0.01, 0.1),
        minsplit = c(5, 10, 20, 30),
        maxdepth = c(10, 20, 30)
      ))
    } else { # large
      return(list(
        cp = c(0.0001, 0.001, 0.01, 0.1),
        minsplit = c(5, 10, 20, 30, 50),
        maxdepth = c(5, 10, 15, 20, 30)
      ))
    }
  } else if (method == "forest") {
    if (size == "small") {
      return(list(
        mtry = c(2, 3),
        ntree = c(100, 500)
      ))
    } else if (size == "medium") {
      return(list(
        mtry = c(2, 3, 4, 5),
        ntree = c(100, 300, 500)
      ))
    } else { # large
      return(list(
        mtry = c(1, 2, 3, 4, 5, 6),
        ntree = c(100, 300, 500, 1000),
        sampsize = c(0.5, 0.632, 0.8, 1.0),
        nodesize = c(1, 3, 5)
      ))
    }
  } else if (method == "boost") {
    if (size == "small") {
      return(list(
        n.trees = c(50, 100),
        interaction.depth = c(2, 3),
        shrinkage = c(0.01, 0.1)
      ))
    } else if (size == "medium") {
      return(list(
        n.trees = c(50, 100, 200),
        interaction.depth = c(1, 2, 3, 4),
        shrinkage = c(0.001, 0.01, 0.1),
        n.minobsinnode = c(5, 10)
      ))
    } else { # large
      return(list(
        n.trees = c(50, 100, 200, 500, 1000),
        interaction.depth = c(1, 2, 3, 4, 5),
        shrinkage = c(0.001, 0.01, 0.05, 0.1),
        n.minobsinnode = c(1, 5, 10, 20),
        bag.fraction = c(0.5, 0.632, 0.8, 1.0)
      ))
    }
  } else if (method == "svm") {
    if (size == "small") {
      return(list(
        kernel = c("linear", "radial"),
        cost = c(0.1, 1, 10)
      ))
    } else if (size == "medium") {
      return(list(
        kernel = c("linear", "polynomial", "radial"),
        cost = c(0.01, 0.1, 1, 10, 100),
        gamma = c(0.01, 0.1, 1)
      ))
    } else { # large
      return(list(
        kernel = c("linear", "polynomial", "radial", "sigmoid"),
        cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000),
        gamma = c(0.001, 0.01, 0.1, 1, 10),
        degree = c(2, 3, 4),
        coef0 = c(0, 0.1, 1)
      ))
    }
  } else if (method == "nn") {
    if (size == "small") {
      return(list(
        size = c(3, 5),
        decay = c(0, 0.1)
      ))
    } else if (size == "medium") {
      return(list(
        size = c(2, 5, 10),
        decay = c(0, 0.01, 0.1)
      ))
    } else { # large
      return(list(
        size = c(2, 5, 10, 20, 50),
        decay = c(0, 0.001, 0.01, 0.1, 0.5),
        maxit = c(100, 200, 500)
      ))
    }
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    if (method == "elastic_net") {
      # Alpha values for elastic net (0 = ridge, 1 = lasso)
      alpha_values <- if (size == "small") {
        c(0.25, 0.5, 0.75)
      } else if (size == "medium") {
        c(0.1, 0.25, 0.5, 0.75, 0.9)
      } else { # large
        seq(0.1, 0.9, by = 0.1)
      }
    } else {
      # For ridge and lasso, alpha is fixed
      alpha_values <- if (method == "ridge") 0 else 1
    }

    # Lambda values (regularization strength)
    if (size == "small") {
      lambda_values <- c(0.001, 0.01, 0.1, 1)
    } else if (size == "medium") {
      lambda_values <- c(0.0001, 0.001, 0.01, 0.1, 1, 10)
    } else { # large
      lambda_values <- c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)
    }

    # Combine parameters
    if (method == "elastic_net") {
      return(list(
        alpha = alpha_values,
        lambda = lambda_values
      ))
    } else {
      return(list(
        lambda = lambda_values
      ))
    }
  } else if (method == "logistic") {
    # For logistic regression, we can tune regularization parameters
    return(tl_default_param_grid("ridge", size, is_classification))
  } else if (method == "deep") {
    if (size == "small") {
      return(list(
        hidden_layers = list(c(10), c(10, 5)),
        activation = c("relu", "tanh"),
        dropout = c(0, 0.2)
      ))
    } else if (size == "medium") {
      return(list(
        hidden_layers = list(c(10), c(20, 10), c(20, 10, 5)),
        activation = c("relu", "tanh", "sigmoid"),
        dropout = c(0, 0.2, 0.5),
        batch_size = c(16, 32, 64)
      ))
    } else { # large
      return(list(
        hidden_layers = list(c(10), c(20), c(50), c(20, 10), c(50, 25), c(50, 25, 10)),
        activation = c("relu", "tanh", "sigmoid", "elu"),
        dropout = c(0, 0.1, 0.2, 0.3, 0.5),
        batch_size = c(16, 32, 64, 128),
        learning_rate = c(0.0001, 0.001, 0.01, 0.1)
      ))
    }
  } else {
    # Default empty grid for unknown method
    warning("Unknown method: ", method, ". Returning empty parameter grid.")
    return(list())
  }
}
