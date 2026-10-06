#' @title XGBoost Functions for tidylearn
#' @name tidylearn-xgboost
#' @description XGBoost-specific implementation for gradient boosting
#' @importFrom stats model.matrix as.formula
#' @importFrom dplyr filter select mutate
NULL

#' Whether the installed xgboost has the 3.x interface
#'
#' xgboost 3.0 renamed and re-based several arguments, so the calls that
#' use them branch on this.
#'
#' @return A single logical
#' @keywords internal
#' @noRd
tl_xgb_v3 <- function() {
  utils::packageVersion("xgboost") >= "3.0.0"
}

#' The rows xgboost trains on: design matrix, labels and weights
#'
#' One model frame supplies the predictors and the response, so the two
#' stay aligned. \code{model.matrix()} on the raw data applied na.omit by
#' itself while the response was read straight from the data, so one
#' missing predictor left the labels a row longer than the matrix and
#' \code{xgb.DMatrix()} refused them. Missing predictors are kept, since
#' xgboost routes them itself; a row whose response or weight is missing
#' has nothing to learn from, and is dropped.
#'
#' @param formula The model formula
#' @param data The training data
#' @param weights Case weights, one per row of \code{data}, or NULL
#' @return A list: \code{x}, the design matrix without the intercept;
#'   \code{y}, the response; \code{weights}, NULL or one per row of \code{x}
#' @keywords internal
#' @noRd
tl_xgb_training_rows <- function(formula, data, weights = NULL) {
  if (!is.null(weights) && length(weights) != nrow(data)) {
    stop(
      "'weights' must have one value per row of data (", nrow(data),
      "); got ", length(weights), ".",
      call. = FALSE
    )
  }

  frame <- stats::model.frame(formula, data = data, na.action = stats::na.pass)
  x <- stats::model.matrix(attr(frame, "terms"), frame)
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  y <- unname(stats::model.response(frame))

  keep <- !is.na(y)
  if (!is.null(weights)) {
    keep <- keep & !is.na(weights)
  }

  list(
    x = x[keep, , drop = FALSE],
    y = y[keep],
    weights = if (!is.null(weights)) weights[keep]
  )
}

#' Fit an XGBoost model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param nrounds Number of boosting rounds (default: 100)
#' @param max_depth Maximum depth of trees (default: 6)
#' @param eta Learning rate (default: 0.3)
#' @param subsample Subsample ratio of observations (default: 1)
#' @param colsample_bytree Subsample ratio of columns (default: 1)
#' @param min_child_weight Minimum sum of instance weight
#'   needed in a child (default: 1)
#' @param gamma Minimum loss reduction to make a further partition (default: 0)
#' @param alpha L1 regularization term (default: 0)
#' @param lambda L2 regularization term (default: 1)
#' @param early_stopping_rounds Early stopping rounds (default: NULL). It
#'   needs data to stop on, which a fit on all the rows does not have, so
#'   it is refused unless a validation set is passed as \code{evals} (or
#'   \code{watchlist} before xgboost 3.0). \code{\link{tl_tune_xgboost}}
#'   chooses the number of rounds by cross-validation instead.
#' @param nthread Number of threads (default: max available)
#' @param verbose Verbose output (default: 0)
#' @param ... Arguments \code{xgb.train()} takes, which go to it; case
#'   \code{weights}, which go to the training \code{xgb.DMatrix()}; and
#'   booster parameters such as \code{max_leaves} or \code{tree_method},
#'   which go into \code{params}. A booster parameter given here, including
#'   \code{objective} and \code{eval_metric}, replaces the value set from
#'   the arguments above.
#' @param compute Compute tier. Either \code{"cpu"} (default) or
#'   \code{"gpu"}; when \code{"gpu"}, the function passes
#'   \code{device = "cuda"} to \code{xgb.train()}. Requires an
#'   xgboost build with CUDA support.
#' @return A fitted XGBoost model
#' @keywords internal
tl_fit_xgboost <- function(data, formula, is_classification = FALSE,
                           nrounds = 100, max_depth = 6, eta = 0.3,
                           subsample = 1, colsample_bytree = 1,
                           min_child_weight = 1, gamma = 0,
                           alpha = 0, lambda = 1,
                           early_stopping_rounds = NULL,
                           nthread = NULL, verbose = 0, ...,
                           compute = "cpu") {
  # Check if xgboost is installed
  tl_check_packages("xgboost")

  # Parse formula
  response_var <- all.vars(formula)[1]

  # Case weights belong to the DMatrix. Forwarded to xgb.train() they were
  # an argument it does not have: xgboost 3.x warned and fitted without.
  dots <- list(...)
  weights <- dots$weights
  dots$weights <- NULL

  rows <- tl_xgb_training_rows(formula, data, weights)
  x_mat <- rows$x
  y <- rows$y

  # Prepare response variable based on problem type
  if (is_classification) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (length(levels(y)) == 2) {
      # Binary classification: convert to 0/1
      y_numeric <- as.integer(y) - 1
      objective <- "binary:logistic"
      eval_metric <- "logloss"
    } else {
      # Multiclass classification: convert to 0-based index
      y_numeric <- as.integer(y) - 1
      objective <- "multi:softprob"
      eval_metric <- "mlogloss"
      num_class <- length(levels(y))
    }
  } else {
    # Regression
    y_numeric <- y
    objective <- "reg:squarederror"
    eval_metric <- "rmse"
  }

  # Create DMatrix object
  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(x_mat), label = y_numeric,
    weight = rows$weights, nthread = nthread
  )

  # Set parameters
  params <- list(
    objective = objective,
    eval_metric = eval_metric,
    max_depth = max_depth,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    alpha = alpha,
    lambda = lambda
  )

  # Add num_class parameter for multiclass classification
  if (is_classification && length(levels(y)) > 2) {
    params$num_class <- num_class
  }

  # Add nthread parameter if specified
  if (!is.null(nthread)) {
    params$nthread <- nthread
  }

  # Route to local CUDA when caller resolved compute to GPU. Requires
  # an xgboost build with CUDA support; otherwise xgb.train will error.
  # `device` requires xgboost >= 2.0.0; tl_check_backend_gpu() screens
  # older versions out before compute ever resolves to "gpu"
  if (identical(compute, "gpu")) {
    params$device <- "cuda"
  }

  # What xgb.train() has no argument for is a booster parameter, and goes
  # in params. xgboost 3.x moves such arguments there itself, but warns
  # that doing so will become an error. The caller's value replaces one set
  # above. objective is an xgb.train() argument in 3.x, which refuses it
  # alongside the one in params, so it is treated as a parameter as well.
  train_formals <- setdiff(
    names(formals(xgboost::xgb.train)), c("...", "params", "objective")
  )
  to_train <- names2(dots) %in% train_formals
  booster_params <- dots[!to_train & names2(dots) != "params"]
  params <- tl_override_args(params, c(dots$params, booster_params))

  # Early stopping needs data to stop on, and a fit on every row has none:
  # xgboost stopped with "For early stopping, 'evals' must have at least
  # one element". Stopping on the training rows would only stop once the
  # fit no longer improved on itself.
  eval_arg <- if (tl_xgb_v3()) "evals" else "watchlist"
  if (!is.null(early_stopping_rounds) && length(dots[[eval_arg]]) == 0L) {
    stop(
      "early_stopping_rounds needs data to stop on, and tl_model() holds no ",
      "rows out. Pass a validation set as ", eval_arg,
      " = list(validation = <xgb.DMatrix>), or use tl_tune_xgboost(), ",
      "which chooses the number of rounds by cross-validation.",
      call. = FALSE
    )
  }

  # Train XGBoost model
  xgb_model <- do.call(
    xgboost::xgb.train,
    tl_override_args(
      list(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        early_stopping_rounds = early_stopping_rounds,
        verbose = verbose
      ),
      dots[to_train]
    )
  )

  # Store additional information for later use
  attr(xgb_model, "feature_names") <- colnames(x_mat)
  attr(xgb_model, "response_var") <- response_var
  attr(xgb_model, "is_classification") <- is_classification

  if (is_classification) {
    attr(xgb_model, "response_levels") <- levels(y)
  }

  xgb_model
}

#' Multiclass xgboost probabilities as an n x k matrix
#'
#' xgboost 3.x returns a matrix directly and has dropped the
#' \code{reshape} argument; earlier versions returned a flat row-major
#' vector unless \code{reshape = TRUE} was passed. Normalise both.
#'
#' @param raw The value \code{predict} returned for the booster
#' @param n_obs Number of observations predicted
#' @param class_levels The response levels recorded at fit time
#' @return A numeric matrix with one named column per class
#' @keywords internal
#' @noRd
tl_xgb_prob_matrix <- function(raw, n_obs, class_levels) {
  k <- length(class_levels)

  out <- if (is.matrix(raw)) {
    raw
  } else {
    # Flat vector is row-major: obs 1's k probabilities, then obs 2's
    matrix(raw, nrow = n_obs, ncol = k, byrow = TRUE)
  }

  colnames(out) <- class_levels
  out
}

#' Predict using an XGBoost model
#'
#' @param model A tidylearn XGBoost model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default),
#'   "prob" (for classification), "class" (for classification)
#' @param iterationrange Boosting iterations to predict with, as
#'   \code{c(start, end)} -- base-1 and inclusive of both ends, so
#'   \code{c(1, 20)} predicts from the first twenty iterations and
#'   \code{end} may not exceed the number fitted. NULL (default) uses
#'   every iteration.
#' @param ntreelimit Deprecated. Use \code{iterationrange} instead.
#'   \code{ntreelimit = n} is translated to \code{c(1, n)}.
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_xgboost <- function(model, new_data,
                               type = "response",
                               iterationrange = NULL,
                               ntreelimit = NULL, ...) {
  # xgboost renamed ntreelimit to iterationrange and made the old name an
  # error-in-waiting. Translate rather than pass it through: "first k
  # trees" is iterations 1 through k.
  if (!is.null(ntreelimit)) {
    warning(
      "'ntreelimit' is deprecated; use iterationrange = c(1, ",
      ntreelimit, ") instead.",
      call. = FALSE
    )
    if (is.null(iterationrange)) {
      iterationrange <- c(1L, as.integer(ntreelimit))
    }
  }

  # Extract XGBoost model
  xgb_model <- model$fit

  # Extract metadata
  feature_names <- attr(xgb_model, "feature_names")
  is_classification <- model$spec$is_classification

  # Build the design matrix from the predictors only, pinned to the
  # training factor levels. Using the full two-sided formula would demand
  # the response column, which unlabelled data does not have, and letting
  # new data supply its own levels would change the contrast coding.
  formula <- model$spec$formula
  x_new <- tl_predictor_matrix(formula, new_data, xlev = model$spec$xlev)

  # Check column names match
  if (!all(colnames(x_new) %in% feature_names)) {
    extra_cols <- setdiff(colnames(x_new), feature_names)
    warning("New data contains columns not in the training data: ",
            paste(extra_cols, collapse = ", "))
  }
  missing_cols <- setdiff(feature_names, colnames(x_new))
  if (length(missing_cols) > 0) {
    stop(
      "New data is missing predictors used at fit time: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  # Create DMatrix for prediction. NAs are passed through rather than
  # dropped -- xgboost routes missing values itself, so predictions stay
  # aligned with the rows of new_data.
  x_subset <- x_new[, feature_names, drop = FALSE]
  dtest <- xgboost::xgb.DMatrix(
    data = as.matrix(x_subset)
  )

  # Make predictions
  if (is_classification) {
    if (type == "prob") {
      # Get class probabilities
      response_levels <- attr(xgb_model, "response_levels")
      n_classes <- length(response_levels)

      if (n_classes == 2) {
        # Binary classification
        prob <- predict(
          xgb_model, newdata = dtest,
          iterationrange = iterationrange
        )

        # One column per class, as a tibble like every other method's.
        # A data.frame here was the one exception.
        tibble::tibble(
          !!response_levels[1] := 1 - prob,
          !!response_levels[2] := prob
        )
      } else {
        # Multiclass classification
        probs <- tl_xgb_prob_matrix(
          predict(
            xgb_model, newdata = dtest,
            iterationrange = iterationrange
          ),
          n_obs = nrow(x_subset), class_levels = response_levels
        )

        tibble::as_tibble(as.data.frame(probs))
      }
    } else if (type == "class" || type == "response") {
      # Get predicted classes
      response_levels <- attr(xgb_model, "response_levels")
      n_classes <- length(response_levels)

      if (n_classes == 2) {
        # Binary classification
        prob <- predict(
          xgb_model, newdata = dtest,
          iterationrange = iterationrange
        )
        pred_classes <- ifelse(
          prob > 0.5,
          response_levels[2],
          response_levels[1]
        )
      } else {
        # Multiclass classification
        probs <- tl_xgb_prob_matrix(
          predict(
            xgb_model, newdata = dtest,
            iterationrange = iterationrange
          ),
          n_obs = nrow(x_subset), class_levels = response_levels
        )
        pred_idx <- max.col(probs, ties.method = "first")
        pred_classes <- response_levels[pred_idx]
      }

      # Convert to factor with original levels
      factor(pred_classes, levels = response_levels)
    } else {
      stop(
        "Invalid prediction type for XGBoost ",
        "classification. Use 'prob', 'class', ",
        "or 'response'.",
        call. = FALSE
      )
    }
  } else {
    # Regression predictions
    predict(
      xgb_model, newdata = dtest,
      iterationrange = iterationrange
    )
  }
}

#' Plot feature importance for an XGBoost model
#'
#' @param model A tidylearn XGBoost model object
#' @param top_n Number of top features to display (default: 10)
#' @param importance_type Type of importance: "gain" (default), "cover" or
#'   "frequency", read from the matching column of
#'   \code{xgboost::xgb.importance()}
#' @param ... Additional arguments passed to \code{xgboost::xgb.importance()}
#' @return A \code{\link[ggplot2]{ggplot}} object. Its data holds the
#'   \code{top_n} features and their \code{importance}, relative to the
#'   most important feature's.
#' @examples
#' \donttest{
#' if (requireNamespace("xgboost", quietly = TRUE)) {
#'   model <- tl_model(mtcars, mpg ~ ., method = "xgboost", nthread = 2)
#'   tl_plot_xgboost_importance(model)
#' }
#' }
#' @export
tl_plot_xgboost_importance <- function(model, top_n = 10,
                                       importance_type = "gain",
                                       ...) {
  # Check if model is an XGBoost model
  if (!inherits(model, "tidylearn_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  measures <- c(gain = "Gain", cover = "Cover", frequency = "Frequency")
  if (!is.character(importance_type) || length(importance_type) != 1L ||
        !importance_type %in% names(measures)) {
    stop(
      "'importance_type' must be one of \"gain\", \"cover\" or ",
      "\"frequency\"; got ", tl_describe_value(importance_type), ".",
      call. = FALSE
    )
  }

  # Extract XGBoost model
  xgb_model <- model$fit

  # Calculate feature importance
  importance <- as.data.frame(xgboost::xgb.importance(
    model = xgb_model,
    feature_names = attr(xgb_model, "feature_names"),
    ...
  ))

  measure <- measures[[importance_type]]
  if (!measure %in% names(importance)) {
    stop(
      "xgb.importance() reports no ", measure, " for this model, only ",
      paste(setdiff(names(importance), "Feature"), collapse = ", "), ".",
      call. = FALSE
    )
  }

  # Drawn with ggplot2 like the package's other importance plots. This
  # returned xgb.plot.importance()'s data.table, drawing base graphics as a
  # side effect, and never read importance_type.
  plot_data <- tibble::tibble(
    feature = importance$Feature,
    importance = importance[[measure]] / max(importance[[measure]])
  ) |>
    dplyr::arrange(dplyr::desc(.data$importance)) |>
    dplyr::slice_head(n = top_n)

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = stats::reorder(.data$feature, .data$importance),
      y = .data$importance
    )
  ) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "XGBoost Feature Importance",
      x = NULL,
      y = paste0("Relative importance (", importance_type, ")")
    ) +
    ggplot2::theme_minimal()
}

#' Plot XGBoost tree visualization
#'
#' @param model A tidylearn XGBoost model object
#' @param tree_index Index of the tree to plot, counting the first tree as
#'   0 (default: 0). A multiclass model grows one tree per class in each
#'   round.
#' @param ... Additional arguments passed to \code{xgboost::xgb.plot.tree()}
#' @return The return value of \code{\link[xgboost]{xgb.plot.tree}}, a
#'   tree diagram rendered via the \pkg{DiagrammeR} package.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("xgboost", quietly = TRUE) &&
#'     requireNamespace("DiagrammeR", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "xgboost", nrounds = 10,
#'     nthread = 2)
#'
#'   # tree_index is zero-based, so this is the first tree
#'   tl_plot_xgboost_tree(model, tree_index = 0)
#' }
#' }
tl_plot_xgboost_tree <- function(model, tree_index = 0, ...) {
  # Check if model is an XGBoost model
  if (!inherits(model, "tidylearn_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  if (!is.numeric(tree_index) || length(tree_index) != 1L ||
        is.na(tree_index) || tree_index < 0 ||
        tree_index != round(tree_index)) {
    stop(
      "'tree_index' must be a single whole number, counting the first tree ",
      "as 0; got ", tl_describe_value(tree_index), ".",
      call. = FALSE
    )
  }

  # Extract XGBoost model
  xgb_model <- model$fit

  n_trees <- length(unique(xgboost::xgb.model.dt.tree(model = xgb_model)$Tree))
  if (tree_index >= n_trees) {
    stop(
      "'tree_index' is ", tree_index, ", but the model has ", n_trees,
      " trees, numbered 0 to ", n_trees - 1, ".",
      call. = FALSE
    )
  }

  # xgboost 3.0 replaced the zero-based `trees` with a one-based
  # `tree_idx`. tree_index reached neither: 3.x dropped it with a warning
  # that it will become an error, and drew the first tree every time.
  if (tl_xgb_v3()) {
    xgboost::xgb.plot.tree(model = xgb_model, tree_idx = tree_index + 1L, ...)
  } else {
    xgboost::xgb.plot.tree(model = xgb_model, trees = tree_index, ...)
  }
}

#' The iteration an xgb.cv() run settled on
#'
#' xgboost 3.0 moved \code{best_iteration} out of the top level of the
#' \code{xgb.cv()} result and into \code{$early_stop}. Reading only the old
#' location returned \code{NULL} against every installed xgboost from 3.0
#' on, so each parameter set scored \code{NULL}, \code{which.min()} over
#' the collected scores returned \code{integer(0)}, and the tuner died on
#' "attempt to select less than one element in get1index" -- for every
#' input, including the documented default call.
#'
#' Both locations are read so the package works either side of that
#' change. Where neither carries one -- early stopping switched off, so
#' the run went the full distance -- the last iteration is the answer.
#'
#' @param cv_result The value of \code{xgboost::xgb.cv()}.
#' @return A single integer iteration index.
#' @keywords internal
#' @noRd
tl_xgb_best_iteration <- function(cv_result) {
  it <- cv_result$early_stop$best_iteration
  if (length(it) != 1L) {
    it <- cv_result$best_iteration
  }
  if (length(it) != 1L || is.na(it) || it < 1) {
    it <- nrow(cv_result$evaluation_log)
  }
  as.integer(it)
}

#' Tune XGBoost hyperparameters
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem. \code{NULL} (default) reads it from the
#'   response, as \code{\link{tl_model}} does: a factor or character
#'   response is classification. \code{FALSE} with such a response is an
#'   error.
#' @param param_grid Named list of parameter values to try. NULL (default)
#'   tries every combination of \code{tl_default_param_grid("xgboost",
#'   size = "large")} without its \code{nrounds}, which early stopping
#'   chooses here.
#' @param cv_folds Number of cross-validation folds (default: 5)
#' @param nrounds Upper bound on boosting rounds per parameter set
#'   (default: 1000). Early stopping normally halts well short of it, so
#'   this is a ceiling rather than a target; lower it to cap the search.
#' @param early_stopping_rounds Early stopping rounds (default: 10)
#' @param verbose Logical indicating whether to print progress (default: TRUE)
#' @param ... Arguments \code{xgboost::xgb.cv()} takes, such as
#'   \code{showsd} or \code{stratified}, which go to it alone; case
#'   \code{weights}, one per row of \code{data}; and booster parameters held
#'   fixed across the grid, such as \code{nthread} or \code{tree_method},
#'   which join each parameter set and the final fit. A value in
#'   \code{param_grid} replaces one given here.
#' @return A \code{tidylearn_model} object (the refit on full data using the
#'   best hyperparameters, built by \code{\link{tl_model}} so that it records
#'   them in \code{$spec$args}) with an attribute \code{"tuning_results"}
#'   containing a list with elements \code{param_grid}, \code{results}
#'   (per-combination CV output), \code{best_params}, \code{best_iteration},
#'   \code{best_score}, and \code{minimize}.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("xgboost", quietly = TRUE)) {
#'   # The default grid is every combination of the large xgboost grid
#'   # without nrounds -- this many:
#'   default_grid <- tl_default_param_grid("xgboost", size = "large")
#'   prod(lengths(default_grid[names(default_grid) != "nrounds"]))
#'
#'   # Name a smaller one to see it run, and cap nrounds so early stopping
#'   # has less ground to cover. nthread = 2 keeps xgboost from taking
#'   # every core it is offered.
#'   tuned <- tl_tune_xgboost(iris, Species ~ .,
#'     param_grid = list(max_depth = c(2, 4)),
#'     cv_folds = 3, nrounds = 20, verbose = FALSE, nthread = 2)
#'
#'   results <- attr(tuned, "tuning_results")
#'   results$best_params
#'   results$best_iteration
#'
#'   # tuned is an ordinary model, refit on all rows at those settings
#'   predict(tuned, iris[1:5, ])
#' }
#' }
tl_tune_xgboost <- function(data, formula, is_classification = NULL,
                            param_grid = NULL, cv_folds = 5,
                            nrounds = 1000,
                            early_stopping_rounds = 10,
                            verbose = TRUE, ...) {
  # Check if xgboost is installed
  tl_check_packages("xgboost")

  # Set default parameter grid if not provided: the large grid
  # tl_default_param_grid() gives tl_tune_grid(), taken from there so the
  # two cannot drift apart, without the nrounds early stopping picks here
  if (is.null(param_grid)) {
    param_grid <- tl_default_param_grid("xgboost", size = "large")
    param_grid$nrounds <- NULL
  }

  # Parse formula, and settle the task from the response
  formula <- tl_as_formula(formula)
  task <- tl_tuner_task(data, formula, is_classification)
  data <- task$data
  is_classification <- task$is_classification

  # xgb.cv()'s own arguments go to it alone. Everything else in ... is a
  # booster parameter, which belongs in params, for the cross-validation
  # and the final fit alike. All of ... used to reach both calls, so an
  # xgb.cv() argument such as showsd reached xgb.train(), which warned
  # that it did not recognise it, and a booster parameter reached each as
  # an argument, which xgboost 3.x warns will become an error.
  dots <- list(...)
  weights <- dots$weights
  dots$weights <- NULL
  cv_formals <- setdiff(
    names(formals(xgboost::xgb.cv)), c("...", "params", "objective")
  )
  to_cv <- names2(dots) %in% cv_formals
  cv_args <- dots[to_cv]
  fixed_params <- c(dots$params, dots[!to_cv & names2(dots) != "params"])

  # Prepare data for XGBoost
  rows <- tl_xgb_training_rows(formula, data, weights)
  x_mat <- rows$x
  y <- rows$y

  # Prepare response variable based on problem type
  if (is_classification) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (length(levels(y)) == 2) {
      # Binary classification: convert to 0/1
      y_numeric <- as.integer(y) - 1
      objective <- "binary:logistic"
      eval_metric <- "logloss"
    } else {
      # Multiclass classification: convert to 0-based index
      y_numeric <- as.integer(y) - 1
      objective <- "multi:softprob"
      eval_metric <- "mlogloss"
      num_class <- length(levels(y))
    }
  } else {
    # Regression
    y_numeric <- y
    objective <- "reg:squarederror"
    eval_metric <- "rmse"
  }

  # Create DMatrix object
  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(x_mat), label = y_numeric,
    weight = rows$weights, nthread = fixed_params$nthread
  )

  # Create parameter grid. A character value such as tree_method stays a
  # string rather than becoming a factor.
  param_df <- expand.grid(param_grid, stringsAsFactors = FALSE)

  if (verbose) {
    message("Tuning XGBoost with ", nrow(param_df), " parameter combinations")
    message("Cross-validation with ", cv_folds, " folds")
  }

  # Initialize storage for results
  results <- list()

  # Loop through parameter combinations
  for (i in seq_len(nrow(param_df))) {
    if (verbose) {
      message(
        "Parameter set ", i, " of ", nrow(param_df),
        ": ",
        paste(
          names(param_df), param_df[i, ],
          sep = "=", collapse = ", "
        )
      )
    }

    # Extract parameters for this iteration. `drop = FALSE` because a
    # one-parameter grid is a single-column data frame, and `[i, ]` on one
    # of those returns a bare vector with the column name gone -- so
    # as.list() produced an unnamed list and xgboost refused the whole fit
    # with "parameter names cannot be empty strings". The same slip was
    # fixed in tl_tune_grid() and tl_tune_random() for 0.4.0; this call
    # site was missed.
    params <- tl_override_args(
      fixed_params, as.list(param_df[i, , drop = FALSE])
    )

    # Set basic parameters. These stay the tuner's own: the score is read
    # from the column named after eval_metric.
    params$objective <- objective
    params$eval_metric <- eval_metric

    # Add num_class parameter for multiclass classification
    if (is_classification && length(levels(y)) > 2) {
      params$num_class <- num_class
    }

    # Run cross-validation
    # nrounds was hardcoded here while `...` went to the same call, so a
    # caller who passed the one argument an xgboost tuner obviously takes
    # got "formal argument \"nrounds\" matched by multiple actual
    # arguments". It is a named argument now, defaulting to the same high
    # ceiling that early stopping is expected to cut short.
    cv_result <- do.call(
      xgboost::xgb.cv,
      tl_override_args(
        list(
          params = params,
          data = dtrain,
          nrounds = nrounds,
          nfold = cv_folds,
          early_stopping_rounds = early_stopping_rounds,
          verbose = ifelse(verbose, 1, 0)
        ),
        cv_args
      )
    )

    # Extract best iteration and performance
    best_iteration <- tl_xgb_best_iteration(cv_result)
    metric_col <- paste0("test_", eval_metric, "_mean")
    best_score <- cv_result$evaluation_log[
      best_iteration,
    ][[metric_col]]

    # Store results
    results[[i]] <- list(
      params = params,
      best_iteration = best_iteration,
      best_score = best_score,
      cv_result = cv_result
    )

    if (verbose) {
      message("  Best iteration: ", best_iteration,
              ", Best ", eval_metric, ": ", round(best_score, 6))
    }
  }

  # Find best parameter set
  best_scores <- sapply(results, function(x) x$best_score)

  # Determine if we should minimize or maximize the metric
  minimize <- eval_metric %in% c("error", "logloss", "mlogloss", "rmse", "mae")

  if (minimize) {
    best_idx <- which.min(best_scores)
  } else {
    best_idx <- which.max(best_scores)
  }

  # Extract best parameters and iteration
  best_params <- results[[best_idx]]$params
  best_iteration <- results[[best_idx]]$best_iteration

  if (verbose) {
    message(
      "Best parameters found: ",
      paste(
        names(best_params), best_params,
        sep = "=", collapse = ", "
      )
    )
    message("Best iteration: ", best_iteration)
    best_sc <- results[[best_idx]]$best_score
    message(
      "Best ", eval_metric, ": ",
      round(best_sc, 6)
    )
  }

  # Refit on all rows through tl_model(), at the winning settings and the
  # iteration the cross-validation settled on. A hand-built model here had
  # no $spec$args, so tl_compare_cv() refitted it at xgboost's defaults in
  # every fold. tl_model() sets objective, eval_metric and num_class from
  # the response, as the cross-validation did, so they are not passed on.
  final_params <- tl_override_args(
    fixed_params, as.list(param_df[best_idx, , drop = FALSE])
  )
  final_params <- final_params[
    !names2(final_params) %in% c("objective", "eval_metric", "num_class")
  ]
  final_args <- c(
    list(data = data, formula = formula, method = "xgboost"),
    final_params,
    list(nrounds = best_iteration)
  )
  if (!is.null(weights)) {
    final_args$weights <- weights
  }
  model <- do.call(tl_model, final_args)

  # Add tuning results to model
  attr(model, "tuning_results") <- list(
    param_grid = param_grid,
    results = results,
    best_params = best_params,
    best_iteration = best_iteration,
    best_score = results[[best_idx]]$best_score,
    minimize = minimize
  )

  model
}

#' SHAP contributions for the rows of data, one matrix per model output
#'
#' The design matrix is built from the predictors alone, pinned to the
#' training factor levels, with incomplete rows kept, so it has one row per
#' row of \code{data} and needs no response column. \code{predict()} lays
#' contributions out differently by version and task: a matrix for one
#' output; for several, a list of matrices before xgboost 3.0 and a row x
#' class x feature array from 3.0 on. Both become a list here.
#'
#' @param model A tidylearn xgboost model
#' @param data Rows to explain
#' @param trees_idx Boosting rounds to use, as a run of consecutive
#'   one-based rounds, or NULL for all
#' @return A list: \code{x}, the design matrix; \code{shap}, a list of
#'   n x p matrices named by feature, one per output; \code{bias}, a list
#'   of matching intercept vectors; \code{classes}, the class of each
#'   output for a multiclass model, NULL otherwise
#' @keywords internal
#' @noRd
tl_xgb_contributions <- function(model, data, trees_idx = NULL) {
  xgb_model <- model$fit
  feature_names <- attr(xgb_model, "feature_names")

  x <- tl_predictor_matrix(model$spec$formula, data, xlev = model$spec$xlev)
  missing_cols <- setdiff(feature_names, colnames(x))
  if (length(missing_cols) > 0) {
    stop(
      "Data is missing predictors used at fit time: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  x <- x[, feature_names, drop = FALSE]

  args <- list(
    xgb_model, xgboost::xgb.DMatrix(data = x),
    predcontrib = TRUE, approxcontrib = FALSE
  )
  if (!is.null(trees_idx)) {
    args$iterationrange <- tl_xgb_round_range(trees_idx, xgb_model)
  }
  raw <- do.call(predict, args)

  n <- nrow(x)
  per_output <- if (is.list(raw)) {
    raw
  } else if (length(dim(raw)) == 3L) {
    lapply(seq_len(dim(raw)[2]), function(k) {
      matrix(raw[, k, ], nrow = n)
    })
  } else {
    list(matrix(raw, nrow = n))
  }

  p <- length(feature_names)
  list(
    x = x,
    shap = lapply(per_output, function(m) {
      out <- m[, seq_len(p), drop = FALSE]
      dimnames(out) <- list(NULL, feature_names)
      out
    }),
    # The intercept is the last column in every layout
    bias = lapply(per_output, function(m) unname(m[, p + 1L])),
    classes = if (length(per_output) > 1L) {
      attr(xgb_model, "response_levels")
    }
  )
}

#' Translate a run of boosting rounds into predict()'s iterationrange
#'
#' xgboost 3.0 reads \code{iterationrange} as inclusive of both ends;
#' earlier versions read its end as exclusive.
#'
#' @param trees_idx One-based rounds, consecutive and increasing
#' @param xgb_model The booster, for its number of rounds
#' @return A length-2 integer vector in the installed version's convention
#' @keywords internal
#' @noRd
tl_xgb_round_range <- function(trees_idx, xgb_model) {
  if (!is.numeric(trees_idx) || length(trees_idx) == 0L ||
        anyNA(trees_idx) || any(trees_idx != round(trees_idx)) ||
        any(trees_idx < 1) || any(diff(trees_idx) != 1)) {
    stop(
      "'trees_idx' must be a run of consecutive rounds counted from 1, such ",
      "as 1:20; got ", paste(utils::head(trees_idx, 10), collapse = ", "),
      if (length(trees_idx) > 10) ", ..." else "", ".",
      call. = FALSE
    )
  }

  rounds <- if (tl_xgb_v3()) {
    xgboost::xgb.get.num.boosted.rounds(xgb_model)
  } else {
    xgb_model$niter
  }
  last <- max(trees_idx)
  if (last > rounds) {
    stop(
      "'trees_idx' runs to round ", last, ", but the model has ", rounds,
      " rounds.",
      call. = FALSE
    )
  }

  first <- as.integer(min(trees_idx))
  last <- as.integer(last)
  if (tl_xgb_v3()) c(first, last) else c(first, last + 1L)
}

#' Generate SHAP values for XGBoost model interpretation
#'
#' @param model A tidylearn XGBoost model object
#' @param data Data for SHAP value calculation
#'   (default: NULL, uses training data). The response column is not
#'   needed.
#' @param n_samples Number of samples to use (default: 100, NULL for all)
#' @param trees_idx Boosting rounds to include, as a run of consecutive
#'   rounds counted from 1 such as \code{1:20} (default: NULL, uses every
#'   round)
#' @return A data frame with one column of SHAP values per feature (the
#'   columns of the model's design matrix), a \code{BIAS} column and a
#'   \code{row_id} column giving the row of the (sampled) data each row
#'   explains. For a multiclass model there is one block of rows per class,
#'   told apart by a \code{class} column. The columns of \code{data} whose
#'   names are not already taken -- the response, and any factor predictor,
#'   whose SHAP columns are named after its levels -- are appended for
#'   reference.
#' @examples
#' \donttest{
#' if (requireNamespace("xgboost", quietly = TRUE)) {
#'   model <- tl_model(mtcars, mpg ~ ., method = "xgboost", nthread = 2)
#'   shap <- tl_xgboost_shap(model, n_samples = 20)
#' }
#' }
#' @export
tl_xgboost_shap <- function(model, data = NULL,
                            n_samples = 100,
                            trees_idx = NULL) {
  # Check if model is an XGBoost model
  if (!inherits(model, "tidylearn_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  # Check if xgboost is installed
  tl_check_packages("xgboost")

  # Use training data if data is not provided
  if (is.null(data)) {
    data <- model$data
  }

  # Sample data if needed
  if (!is.null(n_samples) && n_samples < nrow(data)) {
    sample_idx <- sample(nrow(data), n_samples)
    data <- data[sample_idx, , drop = FALSE]
  }

  # Contributions come from the predictors alone. The full two-sided
  # formula demanded the response, so unlabelled data failed with "object
  # 'mpg' not found"; trees_idx was handed to predict(), which has no such
  # argument, and dropped.
  contrib <- tl_xgb_contributions(model, data, trees_idx)

  # One block of rows per model output. xgboost 3.x returns a row x class x
  # feature array for a multiclass model, and labelling a flattened slice
  # of it put Sepal.Length's SHAP for virginica under Petal.Length and left
  # most of the columns named NA.
  blocks <- lapply(seq_along(contrib$shap), function(k) {
    block <- as.data.frame(contrib$shap[[k]])
    block$BIAS <- contrib$bias[[k]]
    block$row_id <- seq_len(nrow(block))
    if (!is.null(contrib$classes)) {
      block$class <- factor(contrib$classes[k], levels = contrib$classes)
    }
    block
  })
  shap_df <- do.call(rbind, blocks)
  rownames(shap_df) <- NULL

  # Add original data for reference, repeated for each block
  for (col in setdiff(names(data), names(shap_df))) {
    shap_df[[col]] <- rep(data[[col]], times = length(blocks))
  }

  shap_df
}

#' Plot SHAP summary for XGBoost model
#'
#' @param model A tidylearn XGBoost model object
#' @param data Data for SHAP value calculation
#'   (default: NULL, uses training data)
#' @param top_n Number of top features to display (default: 10)
#' @param n_samples Number of samples to use
#'   (default: 100, NULL for all)
#' @return A \code{\link[ggplot2]{ggplot}} object. Features are ranked by
#'   mean absolute SHAP value; a multiclass model is drawn one panel per
#'   class.
#' @importFrom ggplot2 ggplot aes geom_point
#'   scale_color_gradient labs theme_minimal
#' @examples
#' \donttest{
#' if (requireNamespace("xgboost", quietly = TRUE)) {
#'   model <- tl_model(mtcars, mpg ~ ., method = "xgboost", nthread = 2)
#'   tl_plot_xgboost_shap_summary(model, n_samples = 20)
#' }
#' }
#' @export
tl_plot_xgboost_shap_summary <- function(model,
                                         data = NULL,
                                         top_n = 10,
                                         n_samples = 100) {
  if (!inherits(model, "tidylearn_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  # Resolve data and handle sampling here so feature values and SHAP values
  # come from exactly the same rows
  use_data <- if (is.null(data)) model$data else data
  if (!is.null(n_samples) && n_samples < nrow(use_data)) {
    sample_idx <- sample(nrow(use_data), n_samples)
    use_data <- use_data[sample_idx, , drop = FALSE]
  }

  # SHAP values and the feature values they explain, from one design
  # matrix. Feature values are read from that matrix, not the data, so a
  # factor's columns have values too.
  contrib <- tl_xgb_contributions(model, use_data)

  # Rank features by mean absolute SHAP value, over every class
  all_shap <- do.call(rbind, contrib$shap)
  feature_importance <- colMeans(abs(all_shap))
  sorted_features <- names(sort(feature_importance, decreasing = TRUE))
  top_features <- sorted_features[seq_len(min(top_n, length(sorted_features)))]

  plot_data <- do.call(rbind, lapply(seq_along(contrib$shap), function(k) {
    shap_vals <- contrib$shap[[k]][, top_features, drop = FALSE]
    block <- data.frame(
      feature = rep(top_features, each = nrow(shap_vals)),
      feature_value = as.vector(contrib$x[, top_features, drop = FALSE]),
      shap_value = as.vector(shap_vals),
      abs_shap_value = abs(as.vector(shap_vals))
    )
    if (!is.null(contrib$classes)) {
      block$class <- factor(contrib$classes[k], levels = contrib$classes)
    }
    block
  }))

  # Create plot
  shap_aes <- ggplot2::aes(
    x = reorder(feature, abs_shap_value),
    y = shap_value
  )
  color_aes <- ggplot2::aes(color = feature_value)
  grad <- ggplot2::scale_color_gradient2(
    low = "blue", mid = "white",
    high = "red", midpoint = 0
  )
  shap_labs <- ggplot2::labs(
    title = "SHAP Feature Importance",
    subtitle = "Features sorted by mean absolute SHAP value",
    x = NULL,
    y = "SHAP Value (impact on prediction)",
    color = "Feature Value"
  )

  if (requireNamespace("ggforce", quietly = TRUE)) {
    # Use violin plots with jittered points
    p <- ggplot2::ggplot(plot_data, shap_aes) +
      ggforce::geom_sina(
        color_aes, size = 2, alpha = 0.7
      ) +
      grad +
      ggplot2::coord_flip() +
      shap_labs +
      ggplot2::theme_minimal()
  } else {
    # Fall back to basic jittered points
    p <- ggplot2::ggplot(plot_data, shap_aes) +
      ggplot2::geom_jitter(
        color_aes, width = 0.2, alpha = 0.7
      ) +
      grad +
      ggplot2::coord_flip() +
      shap_labs +
      ggplot2::theme_minimal()
  }

  if (!is.null(contrib$classes)) {
    p <- p + ggplot2::facet_wrap(ggplot2::vars(.data$class))
  }

  p
}

#' Plot SHAP dependence for a specific feature
#'
#' @param model A tidylearn XGBoost model object
#' @param feature Feature name to plot
#' @param interaction_feature Feature to use for coloring (default: NULL)
#' @param data Data for SHAP value calculation
#'   (default: NULL, uses training data)
#' @param n_samples Number of samples to use
#'   (default: 100, NULL for all)
#' @return A \code{\link[ggplot2]{ggplot}} object. A multiclass model is
#'   drawn one panel per class, from that class's SHAP values.
#' @importFrom ggplot2 ggplot aes geom_point
#'   geom_smooth scale_color_gradient labs theme_minimal
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("xgboost", quietly = TRUE)) {
#'   model <- tl_model(iris, Species ~ ., method = "xgboost", nrounds = 10,
#'     nthread = 2)
#'
#'   # One panel per class
#'   tl_plot_xgboost_shap_dependence(model, feature = "Petal.Length")
#'
#'   # Colour the points by a second feature to read the interaction
#'   tl_plot_xgboost_shap_dependence(model,
#'     feature = "Petal.Length",
#'     interaction_feature = "Petal.Width")
#' }
#' }
tl_plot_xgboost_shap_dependence <- function( # nolint: object_length_linter.
    model, feature,
    interaction_feature = NULL,
    data = NULL,
    n_samples = 100) {
  if (!inherits(model, "tidylearn_model") || model$spec$method != "xgboost") {
    stop("This function requires an XGBoost model", call. = FALSE)
  }

  # Check if feature exists
  feature_names <- attr(model$fit, "feature_names")
  if (!feature %in% feature_names) {
    stop(
      "Feature not found in model: ", feature,
      call. = FALSE
    )
  }

  # Check interaction feature if provided
  if (!is.null(interaction_feature) &&
        !interaction_feature %in% feature_names) {
    stop(
      "Interaction feature not found in model: ",
      interaction_feature, call. = FALSE
    )
  }

  # Sample once, here, and take the SHAP values and the feature values from
  # the same rows. tl_xgboost_shap() drew its own sample and this function
  # drew a second, independent one for the feature values, so each SHAP
  # value was plotted against another row's feature: for y = 10x the
  # correlation came out near zero.
  use_data <- if (is.null(data)) model$data else data
  if (!is.null(n_samples) && n_samples < nrow(use_data)) {
    sample_idx <- sample(nrow(use_data), n_samples)
    use_data <- use_data[sample_idx, , drop = FALSE]
  }
  contrib <- tl_xgb_contributions(model, use_data)

  # One block of points per model output; a multiclass model has one per
  # class
  plot_data <- do.call(rbind, lapply(seq_along(contrib$shap), function(k) {
    block <- data.frame(
      feature_value = unname(contrib$x[, feature]),
      shap_value = unname(contrib$shap[[k]][, feature])
    )
    if (!is.null(interaction_feature)) {
      block$interaction_value <- unname(contrib$x[, interaction_feature])
    }
    if (!is.null(contrib$classes)) {
      block$class <- factor(contrib$classes[k], levels = contrib$classes)
    }
    block
  }))

  # Create plot
  base_aes <- ggplot2::aes(
    x = feature_value, y = shap_value
  )
  dep_title <- paste(
    "SHAP Dependence Plot for", feature
  )

  if (is.null(interaction_feature)) {
    # Without interaction feature
    p <- ggplot2::ggplot(plot_data, base_aes) +
      ggplot2::geom_point(alpha = 0.7) +
      ggplot2::geom_smooth(
        method = "loess", formula = y ~ x,
        se = TRUE, color = "red"
      ) +
      ggplot2::labs(
        title = dep_title,
        x = feature,
        y = "SHAP Value (impact on prediction)"
      ) +
      ggplot2::theme_minimal()
  } else {
    # With interaction feature
    int_aes <- ggplot2::aes(
      x = feature_value, y = shap_value,
      color = interaction_value
    )
    p <- ggplot2::ggplot(plot_data, int_aes) +
      ggplot2::geom_point(alpha = 0.7) +
      ggplot2::scale_color_gradient2(
        low = "blue", mid = "white",
        high = "red"
      ) +
      ggplot2::labs(
        title = dep_title,
        subtitle = paste(
          "Colored by", interaction_feature
        ),
        x = feature,
        y = "SHAP Value (impact on prediction)",
        color = interaction_feature
      ) +
      ggplot2::theme_minimal()
  }

  if (!is.null(contrib$classes)) {
    p <- p + ggplot2::facet_wrap(ggplot2::vars(.data$class))
  }

  p
}
