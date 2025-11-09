#' @title Model Selection Functions for tidysl
#' @name tidysl-model-selection
#' @description Functions for stepwise model selection, cross-validation, and hyperparameter tuning
#' @importFrom stats AIC BIC step
#' @importFrom dplyr %>% filter select mutate
NULL

#' Perform stepwise selection on a linear model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the initial model
#' @param direction Direction of stepwise selection: "forward", "backward", or "both"
#' @param criterion Criterion for selection: "AIC" or "BIC"
#' @param trace Logical; whether to print progress
#' @param steps Maximum number of steps to take
#' @param ... Additional arguments to pass to step()
#' @return A selected model
#' @export
tl_step_selection <- function(data, formula, direction = "backward",
                              criterion = "AIC", trace = FALSE, steps = 1000, ...) {
  # Input validation
  direction <- match.arg(direction, c("forward", "backward", "both"))
  criterion <- match.arg(criterion, c("AIC", "BIC"))

  # Create full model
  full_model <- lm(formula, data = data)

  # Create null model (intercept only) for forward selection
  null_formula <- as.formula(paste(all.vars(formula)[1], "~ 1"))
  null_model <- lm(null_formula, data = data)

  # Set penalty parameter k based on criterion
  k <- if (criterion == "AIC") 2 else log(nrow(data))

  # Determine start and scope based on direction
  if (direction == "forward") {
    start_model <- null_model
    scope <- list(lower = null_formula, upper = formula)
  } else if (direction == "backward") {
    start_model <- full_model
    scope <- formula
  } else {  # "both"
    start_model <- null_model
    scope <- list(lower = null_formula, upper = formula)
  }

  # Run stepwise selection
  selected_model <- stats::step(
    start_model,
    scope = scope,
    direction = direction,
    k = k,
    trace = trace,
    steps = steps,
    ...
  )

  # Create tidysl model wrapper
  is_classification <- is.factor(data[[all.vars(formula)[1]]]) ||
    is.character(data[[all.vars(formula)[1]]])

  # Return selected model as a tidysl model
  model <- structure(
    list(
      spec = list(
        formula = formula(selected_model),
        method = "linear",
        is_classification = is_classification,
        response_var = all.vars(formula)[1],
        selection = list(
          criterion = criterion,
          direction = direction
        )
      ),
      fit = selected_model,
      data = data
    ),
    class = c("tidysl_linear", "tidysl_model")
  )

  return(model)
}

#' Compare models using cross-validation
#'
#' @param data A data frame containing the training data
#' @param models A list of tidysl model objects
#' @param folds Number of cross-validation folds
#' @param metrics Character vector of metrics to compute
#' @param ... Additional arguments
#' @return A tibble with cross-validation results for all models
#' @export
tl_compare_cv <- function(data, models, folds = 5, metrics = NULL, ...) {
  # Check if all models are tidysl models
  is_tl_model <- sapply(models, inherits, "tidysl_model")
  if (!all(is_tl_model)) {
    stop("All models must be tidysl model objects", call. = FALSE)
  }

  # Get model names if not provided
  model_names <- names(models)
  if (is.null(model_names)) {
    model_names <- paste0("Model_", seq_along(models))
  }

  # Check if all models are of the same type (classification or regression)
  model_types <- sapply(models, function(model) model$spec$is_classification)
  if (length(unique(model_types)) > 1) {
    stop("All models must be of the same type (classification or regression)", call. = FALSE)
  }

  is_classification <- model_types[1]
  response_var <- models[[1]]$spec$response_var

  # Default metrics based on problem type
  if (is.null(metrics)) {
    if (is_classification) {
      metrics <- c("accuracy", "precision", "recall", "f1", "auc")
    } else {
      metrics <- c("rmse", "mae", "rsq", "mape")
    }
  }

  # Create cross-validation splits
  cv_splits <- rsample::vfold_cv(data, v = folds)

  # For each model, perform cross-validation
  cv_results <- lapply(seq_along(models), function(i) {
    model <- models[[i]]
    model_name <- model_names[i]

    # For each fold, train model and evaluate
    fold_results <- purrr::map_dfr(seq_len(folds), function(j) {
      # Get training and testing data for this fold
      train_data <- rsample::analysis(cv_splits$splits[[j]])
      test_data <- rsample::assessment(cv_splits$splits[[j]])

      # Extract formula and method from the model
      formula <- model$spec$formula
      method <- model$spec$method

      # Train model on this fold
      fold_model <- tl_model(train_data, formula = formula, method = method, ...)

      # Evaluate model on test data
      fold_metrics <- tl_evaluate(fold_model, test_data, metrics = metrics)

      # Add fold number and model name
      fold_metrics$fold <- j
      fold_metrics$model <- model_name

      return(fold_metrics)
    })

    return(fold_results)
  })

  # Combine results from all models
  all_cv_results <- do.call(rbind, cv_results)

  # Calculate summary statistics for each model
  summary_results <- all_cv_results %>%
    dplyr::group_by(.data$model, .data$metric) %>%
    dplyr::summarize(
      mean_value = mean(.data$value, na.rm = TRUE),
      sd_value = sd(.data$value, na.rm = TRUE),
      min_value = min(.data$value, na.rm = TRUE),
      max_value = max(.data$value, na.rm = TRUE),
      .groups = "drop"
    )

  # Return both detailed and summary results
  return(list(
    fold_metrics = all_cv_results,
    summary = summary_results
  ))
}

#' Plot comparison of cross-validation results
#'
#' @param cv_results Results from tl_compare_cv function
#' @param metrics Character vector of metrics to plot (if NULL, plots all metrics)
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_boxplot facet_wrap labs theme_minimal
#' @export
tl_plot_cv_comparison <- function(cv_results, metrics = NULL) {
  # Extract fold metrics
  fold_metrics <- cv_results$fold_metrics

  # Filter metrics if specified
  if (!is.null(metrics)) {
    fold_metrics <- fold_metrics %>%
      dplyr::filter(.data$metric %in% metrics)
  }

  # Create the plot
  p <- ggplot2::ggplot(
    fold_metrics,
    ggplot2::aes(x = .data$model, y = .data$value, fill = .data$model)
  ) +
    ggplot2::geom_boxplot() +
    ggplot2::facet_wrap(~ .data$metric, scales = "free_y") +
    ggplot2::labs(
      title = "Cross-Validation Results Comparison",
      x = "Model",
      y = "Metric Value",
      fill = "Model"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

  return(p)
}

#' Perform statistical comparison of models using cross-validation
#'
#' @param cv_results Results from tl_compare_cv function
#' @param baseline_model Name of the model to use as baseline for comparison
#' @param test Type of statistical test: "t.test" or "wilcox"
#' @param metric Name of the metric to compare
#' @return A data frame with statistical test results
#' @importFrom stats t.test wilcox.test p.adjust
#' @export
tl_test_model_difference <- function(cv_results, baseline_model = NULL,
                                     test = "t.test", metric = NULL) {
  # Input validation
  test <- match.arg(test, c("t.test", "wilcox"))

  # Extract fold metrics
  fold_metrics <- cv_results$fold_metrics

  # Get unique models and metrics
  models <- unique(fold_metrics$model)

  if (is.null(baseline_model)) {
    baseline_model <- models[1]
  } else if (!baseline_model %in% models) {
    stop("Baseline model not found in CV results", call. = FALSE)
  }

  if (is.null(metric)) {
    metrics <- unique(fold_metrics$metric)
  } else {
    if (!metric %in% unique(fold_metrics$metric)) {
      stop("Metric not found in CV results", call. = FALSE)
    }
    metrics <- metric
  }

  # Perform statistical tests
  results <- lapply(metrics, function(m) {
    # Filter data for current metric
    metric_data <- fold_metrics %>%
      dplyr::filter(.data$metric == m)

    # Get baseline model data
    baseline_data <- metric_data %>%
      dplyr::filter(.data$model == baseline_model) %>%
      dplyr::pull(.data$value)

    # Compare each model to baseline
    model_comparisons <- lapply(setdiff(models, baseline_model), function(model_name) {
      # Get current model data
      model_data <- metric_data %>%
        dplyr::filter(.data$model == model_name) %>%
        dplyr::pull(.data$value)

      # Perform statistical test
      if (test == "t.test") {
        test_result <- stats::t.test(model_data, baseline_data, paired = TRUE)
      } else {
        test_result <- stats::wilcox.test(model_data, baseline_data, paired = TRUE)
      }

      # Calculate mean difference
      mean_diff <- mean(model_data, na.rm = TRUE) - mean(baseline_data, na.rm = TRUE)

      # Return results
      return(data.frame(
        metric = m,
        model = model_name,
        baseline = baseline_model,
        mean_diff = mean_diff,
        p_value = test_result$p.value
      ))
    })

    # Combine results for all models
    model_results <- do.call(rbind, model_comparisons)

    # Adjust p-values for multiple comparisons if needed
    if (length(models) > 2) {
      model_results$p_adj <- stats::p.adjust(model_results$p_value, method = "holm")
    } else {
      model_results$p_adj <- model_results$p_value
    }

    return(model_results)
  })

  # Combine results for all metrics
  all_results <- do.call(rbind, results)

  return(all_results)
}
