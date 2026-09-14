#' @title Model Selection Functions for tidylearn
#' @name tidylearn-model-selection
#' @description Functions for stepwise model selection,
#'   cross-validation, and hyperparameter tuning
#' @importFrom stats AIC BIC step
#' @importFrom dplyr filter select mutate
NULL

#' Perform stepwise selection on a linear model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the initial model
#' @param direction Direction of stepwise selection:
#'   "forward", "backward", or "both"
#' @param criterion Criterion for selection: "AIC" or "BIC"
#' @param trace Logical; whether to print progress
#' @param steps Maximum number of steps to take
#' @param ... Additional arguments to pass to step()
#' @return A \code{tidylearn_model} object of class
#'   \code{tidylearn_linear} wrapping the selected \code{\link[stats]{lm}}
#'   model. Access the underlying model via \code{$fit} and the selected
#'   formula via \code{$spec$formula}.
#' @examples
#' \donttest{
#' model <- tl_step_selection(mtcars, mpg ~ ., direction = "backward")
#' summary(model)
#' }
#' @export
tl_step_selection <- function(data, formula, direction = "backward",
                              criterion = "AIC",
                              trace = FALSE,
                              steps = 1000, ...) {
  # Input validation
  direction <- match.arg(direction, c("forward", "backward", "both"))
  criterion <- match.arg(criterion, c("AIC", "BIC"))

  # Expand `.` and apply `- x` against the data before anything else reads
  # the formula. step() expands a dot in the upper scope against the start
  # model's right-hand side, which for forward selection is `1`, so
  # `y ~ .` offered no terms to add and returned the intercept-only model.
  formula <- stats::formula(stats::terms(tl_as_formula(formula), data = data,
                                         simplify = TRUE))

  # Create full model
  full_model <- lm(formula, data = data)

  # Create null model (intercept only) for forward selection
  # update() keeps the response as written; pasting all.vars()[1] turned
  # log(mpg) into mpg, so forward selection fitted a different response
  null_formula <- stats::update(formula, . ~ 1)
  # step() refits by re-evaluating the model call, which names `data`, and
  # the lookup falls back to the formula's environment. The caller's may
  # hold utils::data rather than this frame's data, so `data` is supplied
  # in an environment in front of the caller's -- which still resolves any
  # other variable the formula uses there.
  formula_env <- new.env(parent = environment(formula))
  formula_env$data <- data
  environment(null_formula) <- formula_env
  null_model <- lm(null_formula, data = data)

  # Set penalty parameter k based on criterion
  # BIC's penalty is log(n) for the rows the model used. nrow(data) counts
  # rows lm() dropped for missing values, which over-penalises and can
  # change the model selected.
  k <- if (criterion == "AIC") 2 else log(stats::nobs(full_model))

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

  # Create tidylearn model wrapper
  is_classification <- is.factor(data[[all.vars(formula)[1]]]) ||
    is.character(data[[all.vars(formula)[1]]])

  # Return selected model as a tidylearn model
  model <- structure(
    list(
      spec = list(
        paradigm = "supervised",
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
    class = c("tidylearn_linear", "tidylearn_supervised", "tidylearn_model")
  )

  model
}

#' Compare models using cross-validation
#'
#' @param data A data frame containing the training data
#' @param models A list of tidylearn model objects
#' @param folds Number of cross-validation folds
#' @param metrics Character vector of metrics to compute
#' @param ... Arguments passed to \code{\link{tl_model}} for every fold
#'   fit. Each model is refitted with the arguments it was built with;
#'   anything given here overrides them.
#' @return A list with two elements:
#'   \describe{
#'     \item{\code{$fold_metrics}}{A data frame with columns
#'       \code{metric}, \code{value}, \code{fold}, and \code{model}
#'       containing per-fold results for every model.}
#'     \item{\code{$summary}}{A data frame with columns \code{model},
#'       \code{metric}, \code{mean_value}, \code{sd_value},
#'       \code{min_value}, and \code{max_value} summarizing
#'       cross-validation performance.}
#'   }
#' @examples
#' \donttest{
#' m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
#' m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
#' cv$summary
#' }
#' @export
tl_compare_cv <- function(data, models, folds = 5, metrics = NULL, ...) {
  # Check if all models are tidylearn models
  is_tl_model <- sapply(models, inherits, "tidylearn_model")
  if (!all(is_tl_model)) {
    stop("All models must be tidylearn model objects", call. = FALSE)
  }

  # Results are keyed by model name. A name left empty in a partly named
  # list became a model called "", and a repeated name pooled two models'
  # folds into one summary row.
  model_names <- names(models)
  if (is.null(model_names)) {
    model_names <- rep("", length(models))
  }
  unnamed <- is.na(model_names) | model_names == ""
  if (anyDuplicated(model_names[!unnamed])) {
    stop(
      "Model names must be unique; the results are keyed on them. ",
      "Repeated: ",
      paste(unique(model_names[!unnamed][duplicated(model_names[!unnamed])]),
            collapse = ", "),
      call. = FALSE
    )
  }
  # Generated names skip any the caller chose, so list(m1, Model_1 = m2)
  # does not collide with a name the caller never repeated
  model_names[unnamed] <- utils::tail(
    make.unique(c(model_names[!unnamed], paste0("Model_", which(unnamed)))),
    sum(unnamed)
  )

  # Check if all models are of the same type (classification or regression)
  model_types <- sapply(models, function(model) model$spec$is_classification)
  if (length(unique(model_types)) > 1) {
    stop(
      "All models must be of the same type ",
      "(classification or regression)",
      call. = FALSE
    )
  }

  is_classification <- model_types[1]

  # Default metrics based on problem type
  if (is.null(metrics)) {
    if (is_classification) {
      metrics <- c("accuracy", "precision", "recall", "f1", "auc")
    } else {
      metrics <- c("rmse", "mae", "rsq", "mape")
    }
  }

  # Each model's own fitting arguments, overridden by any passed here. A
  # model from tl_step_selection() or an older tidylearn has none recorded.
  fit_args <- lapply(models, function(model) {
    recorded <- if (is.null(model$spec$args)) list() else model$spec$args
    # Replace whole arguments. modifyList() would merge a list-valued one
    # such as parms into the recorded list instead of overriding it.
    overrides <- list(...)
    recorded[names(overrides)] <- overrides
    recorded
  })

  # An argument with one value per training row cannot follow the rows
  # into a fold, and replaying it whole would fail on a length mismatch.
  # A model records such arguments by name only; one passed here to
  # tl_compare_cv() is checked as well.
  per_row <- unique(c(
    unlist(lapply(models, function(model) model$spec$per_row_args)),
    intersect(names2(list(...)), tl_per_row_args())
  ))
  if (length(per_row) > 0) {
    stop(
      "tl_compare_cv() cannot re-split '", paste(per_row, collapse = "', '"),
      "' across folds: it holds one value per row of the data a model was ",
      "fitted on.",
      call. = FALSE
    )
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

      # Train model on this fold, with the arguments it was built with.
      # Refitting from formula and method alone scored every model at its
      # method's defaults, so two trees differing only in cp tied.
      fold_model <- do.call(
        tl_model,
        c(
          list(train_data, formula = model$spec$formula,
               method = model$spec$method),
          fit_args[[i]]
        )
      )

      # Evaluate model on test data
      fold_metrics <- tl_evaluate(fold_model, test_data, metrics = metrics)

      # Add fold number and model name
      fold_metrics$fold <- j
      fold_metrics$model <- model_name

      fold_metrics
    })

    fold_results
  })

  # Combine results from all models
  all_cv_results <- do.call(rbind, cv_results)

  # Calculate summary statistics for each model
  summary_results <- all_cv_results |>
    dplyr::group_by(.data$model, .data$metric) |>
    dplyr::summarize(
      mean_value = mean(.data$value, na.rm = TRUE),
      sd_value = sd(.data$value, na.rm = TRUE),
      min_value = min(.data$value, na.rm = TRUE),
      max_value = max(.data$value, na.rm = TRUE),
      .groups = "drop"
    )

  # Return both detailed and summary results
  list(
    fold_metrics = all_cv_results,
    summary = summary_results
  )
}

#' Plot comparison of cross-validation results
#'
#' @param cv_results Results from tl_compare_cv function
#' @param metrics Character vector of metrics to plot
#'   (if NULL, plots all metrics)
#' @return A \code{\link[ggplot2]{ggplot}} object showing boxplots of
#'   cross-validation metric distributions for each model.
#' @importFrom ggplot2 ggplot aes geom_boxplot facet_wrap labs theme_minimal
#' @examples
#' \donttest{
#' m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
#' m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
#' tl_plot_cv_comparison(cv)
#' }
#' @export
tl_plot_cv_comparison <- function(cv_results, metrics = NULL) {
  # Extract fold metrics
  fold_metrics <- cv_results$fold_metrics

  # Filter metrics if specified
  if (!is.null(metrics)) {
    fold_metrics <- fold_metrics |>
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

  p
}

#' Perform statistical comparison of models using cross-validation
#'
#' @param cv_results Results from tl_compare_cv function
#' @param baseline_model Name of the model to use as baseline for comparison
#' @param test Type of statistical test: "t.test" or "wilcox"
#' @param metric Name of the metric to compare
#' @return A data frame with columns \code{metric}, \code{model},
#'   \code{baseline}, \code{mean_diff}, \code{p_value}, and
#'   \code{p_adj} (Holm-adjusted p-value) containing pairwise
#'   statistical comparisons against the baseline model.
#' @importFrom stats t.test wilcox.test p.adjust
#' @examples
#' \donttest{
#' m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
#' m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
#' tl_test_model_difference(cv, baseline_model = "simple", metric = "rmse")
#' }
#' @export
tl_test_model_difference <- function(
    cv_results,
    baseline_model = NULL,
    test = "t.test",
    metric = NULL) {
  # Input validation
  test <- match.arg(test, c("t.test", "wilcox"))

  # Extract fold metrics
  fold_metrics <- cv_results$fold_metrics

  # Get unique models and metrics
  models <- unique(fold_metrics$model)

  if (is.null(baseline_model)) {
    baseline_model <- models[1]
  } else if (!baseline_model %in% models) {
    stop(
      "Baseline model not found in CV results",
      call. = FALSE
    )
  }

  if (is.null(metric)) {
    metrics <- unique(fold_metrics$metric)
  } else {
    if (!metric %in% unique(fold_metrics$metric)) {
      stop(
        "Metric not found in CV results",
        call. = FALSE
      )
    }
    metrics <- metric
  }

  # Perform statistical tests
  results <- lapply(metrics, function(m) {
    # Filter data for current metric
    metric_data <- fold_metrics |>
      dplyr::filter(.data$metric == m)

    # Get baseline model data
    baseline_data <- metric_data |>
      dplyr::filter(.data$model == baseline_model) |>
      dplyr::pull(.data$value)

    # Compare each model to baseline
    other_models <- setdiff(models, baseline_model)
    model_comparisons <- lapply(
      other_models,
      function(model_name) {
        # Get current model data
        model_data <- metric_data |>
          dplyr::filter(
            .data$model == model_name
          ) |>
          dplyr::pull(.data$value)

        # Perform statistical test
        if (test == "t.test") {
          test_result <- stats::t.test(
            model_data, baseline_data,
            paired = TRUE
          )
        } else {
          test_result <- stats::wilcox.test(
            model_data, baseline_data,
            paired = TRUE
          )
        }

        # Calculate mean difference
        mean_diff <-
          mean(model_data, na.rm = TRUE) -
          mean(baseline_data, na.rm = TRUE)

        # Return results
        data.frame(
          metric = m,
          model = model_name,
          baseline = baseline_model,
          mean_diff = mean_diff,
          p_value = test_result$p.value
        )
      }
    )

    # Combine results for all models
    model_results <- do.call(
      rbind, model_comparisons
    )

    # Adjust p-values for multiple comparisons
    if (length(models) > 2) {
      model_results$p_adj <- stats::p.adjust(
        model_results$p_value, method = "holm"
      )
    } else {
      model_results$p_adj <- model_results$p_value
    }

    model_results
  })

  # Combine results for all metrics
  all_results <- do.call(rbind, results)

  all_results
}
