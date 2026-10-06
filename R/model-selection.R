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
#'   formula via \code{$spec$formula}. \code{$data} is the data passed in.
#' @details Every candidate model is fitted on the same rows: those with no
#'   missing value in any variable of \code{formula}. When that leaves out
#'   rows a candidate could otherwise use, a message gives the count, and
#'   the fit records the rows in its \code{na.action}, as \code{lm()}
#'   records the rows it drops itself.
#'
#'   Forward and both-direction selection start from a model holding the
#'   intercept, or none if \code{formula} removes it with \code{- 1}, and
#'   any \code{offset()} terms in \code{formula}. Neither is ever dropped.
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

  # The model is returned with the data as passed. Every fit below sees a
  # copy whose response is blanked on the rows the full model cannot use.
  original_data <- data
  data <- tl_step_fix_rows(data, formula)

  # Create full model
  full_model <- lm(formula, data = data)

  # The starting model for forward and both-direction selection. Built
  # from the formula itself, so the response stays as written --
  # all.vars()[1] turned log(mpg) into mpg -- and the intercept setting
  # and offset() terms stay too: update(formula, . ~ 1) dropped the
  # offsets and put back an intercept removed with - 1.
  null_formula <- tl_step_null_formula(formula)
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
  response <- original_data[[all.vars(formula)[1]]]
  is_classification <- is.factor(response) || is.character(response)

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
      data = original_data
    ),
    class = c("tidylearn_linear", "tidylearn_supervised", "tidylearn_model")
  )

  model
}

#' Fix the rows stepwise selection fits every candidate on
#'
#' \code{step()} fits each candidate on the rows that candidate can use, so
#' a variable with missing values changes the row count as it enters or
#' leaves the model, and \code{step()} stops with "number of rows in use
#' has changed: remove missing values?". Selection runs on the rows the
#' full model uses instead.
#'
#' The other rows are left out by blanking the response there. Every
#' candidate then drops them for a missing response, so the selected fit
#' records them in its \code{na.action} -- where \code{lm()} records the
#' rows it drops itself, and where \code{tl_fitted_rows()} reads them --
#' and the model can keep the data the caller passed. Subsetting the rows
#' instead would also leave a variable the formula takes from the caller's
#' environment longer than the data.
#'
#' @param data The data passed to \code{tl_step_selection()}
#' @param formula The expanded formula of the full model
#' @return \code{data}, with the response set to \code{NA} on the rows the
#'   full model cannot use
#' @keywords internal
#' @noRd
tl_step_fix_rows <- function(data, formula) {
  # A model frame the data cannot build fails in lm() below, with lm()'s
  # own message
  frame <- tryCatch(
    stats::model.frame(formula, data = data, na.action = stats::na.omit),
    error = function(e) NULL
  )
  omitted <- if (is.null(frame)) {
    integer()
  } else {
    as.integer(stats::na.action(frame))
  }

  # A response taken from outside the data cannot be blanked there; those
  # fits behave as lm() and step() make them
  response_vars <- all.vars(formula[[2L]])
  if (length(omitted) == 0L || !all(response_vars %in% names(data))) {
    return(data)
  }

  # Rows missing the response are dropped by every candidate anyway. Only
  # rows missing a predictor change from one candidate to the next.
  has_response <- stats::complete.cases(data[response_vars])
  if (!any(has_response[omitted])) {
    return(data)
  }

  message(
    "Selecting on the ", nrow(data) - length(omitted), " of ", nrow(data),
    " rows with no missing value in the formula's variables, so that ",
    "every candidate model is fitted on the same rows."
  )
  for (variable in response_vars) {
    data[[variable]][omitted] <- NA
  }
  data
}

#' The starting model of forward and both-direction selection
#'
#' @param formula The expanded formula of the full model
#' @return \code{formula} with its right-hand side reduced to the intercept
#'   (or \code{0} when the formula removes it) and its \code{offset()}
#'   terms, in the formula's own environment
#' @keywords internal
#' @noRd
tl_step_null_formula <- function(formula) {
  model_terms <- stats::terms(formula)
  rhs <- if (attr(model_terms, "intercept") == 1L) 1 else 0
  # "offset" indexes the variables, the response among them, and the
  # variables attribute is a call to list() whose first element is `list`
  variables <- attr(model_terms, "variables")
  for (i in attr(model_terms, "offset")) {
    rhs <- call("+", rhs, variables[[i + 1L]])
  }
  null_formula <- formula
  null_formula[[3L]] <- rhs
  null_formula
}

#' Compare models using cross-validation
#'
#' Each model is refitted on every fold from its formula, method and
#' fitting arguments. A model a refit would not reproduce is refused: one
#' built by \code{\link{tl_semisupervised}} or \code{\link{tl_anomaly_aware}},
#' or one of \code{\link{tl_auto_ml}}'s candidates fitted on features it
#' engineered.
#'
#' @param data A data frame containing the training data
#' @param models A named list of supervised tidylearn models, all of them
#'   classification or all regression. An unnamed model is named
#'   \code{Model_<position>}.
#' @param folds Number of cross-validation folds, a whole number between 2
#'   and \code{nrow(data)}
#' @param metrics Character vector of metrics to compute, from those
#'   \code{\link{tl_evaluate}} computes for the task. Defaults to
#'   \code{c("accuracy", "precision", "recall", "f1", "auc")} for
#'   classification and \code{c("rmse", "mae", "rsq", "mape")} for
#'   regression.
#' @param ... Arguments passed to \code{\link{tl_model}} for every fold
#'   fit. Each model is refitted with the arguments it was built with;
#'   anything given here overrides them.
#' @return A list with two elements:
#'   \describe{
#'     \item{\code{$fold_metrics}}{A data frame with columns
#'       \code{metric}, \code{value}, \code{fold}, and \code{model}
#'       containing per-fold results for every model. A metric undefined
#'       on a fold -- \code{"auc"} on a fold holding one class -- is
#'       \code{NA} there, and so is every metric of a fold none of whose
#'       rows can be scored, with a warning naming the fold.}
#'     \item{\code{$summary}}{A data frame with columns \code{model},
#'       \code{metric}, \code{mean_value}, \code{sd_value},
#'       \code{min_value}, and \code{max_value} summarizing
#'       cross-validation performance over the folds with a value. A
#'       metric with no value on any fold is \code{NA} throughout.}
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
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }
  # A bare model is a list too, of its spec, fit and data, so it was read
  # as three models, none of them a tidylearn model
  if (inherits(models, "tidylearn_model")) {
    stop(
      "'models' must be a list of tidylearn models; to cross-validate a ",
      "single model, wrap it in list().",
      call. = FALSE
    )
  }
  # An empty list reached the task check with nothing to read, and failed
  # with "argument is not interpretable as logical"
  if (!is.list(models) || length(models) == 0L) {
    stop(
      "'models' must be a non-empty list of tidylearn models; got ",
      if (is.list(models)) "an empty list" else tl_describe_value(models),
      ".",
      call. = FALSE
    )
  }

  # Check if all models are tidylearn models
  is_tl_model <- vapply(models, inherits, logical(1), "tidylearn_model")
  if (!all(is_tl_model)) {
    stop("All models must be tidylearn model objects", call. = FALSE)
  }
  tl_check_folds(folds, data)

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

  # An unsupervised model has no task to read, and failed the type check
  # below with "argument is of length zero"
  supervised <- vapply(models, inherits, logical(1), "tidylearn_supervised")
  if (!all(supervised)) {
    stop(
      "tl_compare_cv() compares supervised models; ",
      paste0("'", model_names[!supervised], "'", collapse = ", "),
      if (sum(!supervised) == 1L) {
        " is an unsupervised model."
      } else {
        " are unsupervised models."
      },
      call. = FALSE
    )
  }

  # A refit from formula, method and arguments does not repeat what a
  # wrapper did around tl_model(). It scored a semi-supervised model as a
  # tree trained on every fold label, and failed on the column an
  # anomaly-aware model adds.
  refit_problems <- vapply(models, tl_compare_cv_refit_problem, character(1))
  refused <- !is.na(refit_problems)
  if (any(refused)) {
    stop(
      "tl_compare_cv() cannot refit ",
      paste0("'", model_names[refused], "': ", refit_problems[refused],
             collapse = "; "),
      ". Cross-validate models built with tl_model() instead.",
      call. = FALSE
    )
  }

  # Check if all models are of the same type (classification or regression)
  model_types <- vapply(
    models, function(model) isTRUE(model$spec$is_classification), logical(1)
  )
  if (length(unique(model_types)) > 1) {
    stop(
      "All models must be of the same type ",
      "(classification or regression)",
      call. = FALSE
    )
  }

  is_classification <- model_types[[1]]

  # Default metrics based on problem type
  if (is.null(metrics)) {
    if (is_classification) {
      metrics <- c("accuracy", "precision", "recall", "f1", "auc")
    } else {
      metrics <- c("rmse", "mae", "rsq", "mape")
    }
  }

  # A name outside the task's metrics was left out of the results without
  # a word, so a misspelt metric vanished from the summary. Checked before
  # any fold is fitted, with the refusal tl_evaluate() gives.
  tl_check_metric_names(metrics, is_classification)

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
      # method's defaults, so two trees differing only in cp tied. The
      # notes tl_model() gives about the response were given when the
      # model was built, and every fold refit repeated them, as tl_cv()
      # does not. So was the warning that a 0/1 response is converted for
      # logistic regression.
      fold_model <- withCallingHandlers(
        suppressMessages(do.call(
          tl_model,
          c(
            list(train_data, formula = model$spec$formula,
                 method = model$spec$method),
            fit_args[[i]]
          )
        )),
        tidylearn_response_conversion = function(w) {
          invokeRestart("muffleWarning")
        }
      )

      # Evaluate model on test data. tl_evaluate() refuses a fold on which
      # no row can be scored -- every predictor missing, say -- and one
      # such fold is no reason to stop the comparison, so its values are
      # NA and the summary leaves it out.
      fold_metrics <- tryCatch(
        tl_evaluate(fold_model, test_data, metrics = metrics),
        tidylearn_no_scored_rows = function(e) {
          warning(
            "Fold ", j, " is left out of the summary for '", model_name,
            "', since ", tl_unscored_fold_reason(e),
            call. = FALSE
          )
          tibble::tibble(metric = metrics, value = NA_real_)
        }
      )

      # Add fold number and model name
      fold_metrics$fold <- j
      fold_metrics$model <- model_name

      fold_metrics
    })

    fold_results
  })

  # Combine results from all models
  all_cv_results <- do.call(rbind, cv_results)

  # Calculate summary statistics for each model, over the folds with a
  # value. A metric undefined on a fold -- auc on one holding a single
  # class -- is NA there and left out.
  summary_results <- all_cv_results |>
    dplyr::group_by(.data$model, .data$metric) |>
    dplyr::summarize(
      mean_value = tl_summarise_scored(.data$value, mean),
      sd_value = tl_summarise_scored(.data$value, stats::sd),
      min_value = tl_summarise_scored(.data$value, min),
      max_value = tl_summarise_scored(.data$value, max),
      .groups = "drop"
    )

  # Return both detailed and summary results
  list(
    fold_metrics = all_cv_results,
    summary = summary_results
  )
}

#' Summarise the folds that produced a value
#'
#' A metric undefined on every fold has nothing to summarise. With
#' \code{na.rm = TRUE}, \code{mean()} returned \code{NaN} for it and
#' \code{min()} and \code{max()} \code{Inf} and \code{-Inf}, each with a
#' warning.
#'
#' @param values Per-fold values, \code{NA} where undefined
#' @param fun The summary function
#' @return \code{fun} of the values present, or \code{NA} when there are
#'   none
#' @keywords internal
#' @noRd
tl_summarise_scored <- function(values, fun) {
  values <- values[!is.na(values)]
  if (length(values) == 0L) NA_real_ else fun(values)
}

#' Why a model cannot be refitted from its formula and method
#'
#' @param model A supervised tidylearn model
#' @return A sentence fragment naming what a refit would miss, or
#'   \code{NA} when a refit reproduces the model
#' @keywords internal
#' @noRd
tl_compare_cv_refit_problem <- function(model) {
  if (inherits(model, "tidylearn_semisupervised")) {
    return(paste(
      "tl_semisupervised() trained it on labels it propagated through",
      "clusters, which a refit from its formula and method would not repeat"
    ))
  }
  if (inherits(model, "tidylearn_anomaly_aware")) {
    return(paste(
      "tl_anomaly_aware() trained it after acting on the anomalies it",
      "detected, which a refit from its formula and method would not repeat"
    ))
  }
  # tl_auto_ml() marks the candidates it fits on PCA scores or cluster
  # labels this way, so that predict() can rebuild the features
  if (!is.null(model$feature_transform)) {
    return(paste(
      "tl_auto_ml() trained it on features it engineered, which a refit",
      "from its formula and method would not rebuild"
    ))
  }
  NA_character_
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
#' @param test Type of statistical test: \code{"t.test"}, a paired t-test,
#'   or \code{"wilcox"}, a paired Wilcoxon signed-rank test
#'   (\code{"wilcox.test"} is read as \code{"wilcox"}). With no tied or zero
#'   differences the signed-rank test is exact, and the smallest two-sided
#'   p-value it can give for \code{n} folds is \code{2 / 2^n}: 0.0625 for 5
#'   folds, so it needs at least 6 to reach \code{p < 0.05}.
#' @param metric Name of the metric to compare
#' @return A data frame with columns \code{metric}, \code{model},
#'   \code{baseline}, \code{mean_diff}, \code{p_value}, and
#'   \code{p_adj} (Holm-adjusted p-value) containing pairwise
#'   statistical comparisons against the baseline model. Each comparison
#'   pairs the folds both models have a value for, and \code{mean_diff} is
#'   the mean of those paired differences. With fewer than two such folds
#'   the p-values are \code{NA}, with a warning.
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
  # Input validation. "wilcox.test" is the function's own name, and the
  # diagnostics vignette wrote it that way, so it is read as "wilcox".
  if (identical(test, "wilcox.test")) {
    test <- "wilcox"
  }
  test <- tryCatch(
    match.arg(test, c("t.test", "wilcox")),
    error = function(e) {
      stop(
        "'test' must be \"t.test\" or \"wilcox\"; got ",
        tl_describe_value(test), ".",
        call. = FALSE
      )
    }
  )

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
    baseline_data <- metric_data[metric_data$model == baseline_model,
                                 c("fold", "value")]

    # Compare each model to baseline
    other_models <- setdiff(models, baseline_model)
    model_comparisons <- lapply(
      other_models,
      function(model_name) {
        # Pair the two models fold by fold, on the folds both have a
        # value for. A metric undefined on a fold is NA there, and taking
        # each model's mean over its own folds set mean_diff on a
        # different footing from the paired test beside it.
        model_data <- metric_data[metric_data$model == model_name,
                                  c("fold", "value")]
        paired <- merge(model_data, baseline_data, by = "fold",
                        suffixes = c("_model", "_baseline"))
        both <- !is.na(paired$value_model) & !is.na(paired$value_baseline)
        model_values <- paired$value_model[both]
        baseline_values <- paired$value_baseline[both]

        # A paired test needs two pairs. t.test() stopped on fewer, which
        # discarded the results for every other metric as well.
        p_value <- if (sum(both) < 2L) {
          warning(
            "Only ", sum(both), " fold", if (sum(both) == 1L) "" else "s",
            " scored for both '", model_name, "' and '", baseline_model,
            "' on \"", m, "\", too few for a paired test; its p-value is NA.",
            call. = FALSE
          )
          NA_real_
        } else if (test == "t.test") {
          stats::t.test(model_values, baseline_values, paired = TRUE)$p.value
        } else {
          stats::wilcox.test(
            model_values, baseline_values, paired = TRUE
          )$p.value
        }

        # Return results
        data.frame(
          metric = m,
          model = model_name,
          baseline = baseline_model,
          mean_diff = if (any(both)) {
            mean(model_values - baseline_values)
          } else {
            NA_real_
          },
          p_value = p_value
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
