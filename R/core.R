#' @title tidylearn: A Unified Tidy Interface to R's Machine Learning Ecosystem
#' @name tidylearn-core
#' @description Core functionality for tidylearn. This package
#'   provides a unified tidyverse-compatible interface to
#'   established R machine learning packages including glmnet,
#'   randomForest, xgboost, e1071, rpart, gbm, nnet, cluster,
#'   and dbscan. The underlying algorithms are unchanged -
#'   tidylearn wraps them with consistent function signatures,
#'   tidy tibble output, and unified ggplot2-based
#'   visualization. Access raw model objects via model$fit.
#' @importFrom magrittr %>%
#' @importFrom rlang .data .env
#' @importFrom dplyr filter select mutate group_by summarize arrange
#' @importFrom tibble tibble as_tibble
#' @importFrom purrr map map_dbl map_lgl map2
#' @importFrom tidyr nest unnest
#' @importFrom stats predict model.matrix formula as.formula
NULL

#' Pipe operator
#'
#' @name %>%
#' @rdname pipe
#' @param lhs A value or the magrittr placeholder.
#' @param rhs A function call using the magrittr semantics.
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#' @return The result of applying rhs to lhs.
#' @description See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
NULL

#' @export
#' @rdname pipe
`%>%` <- magrittr::`%>%`

#' Create a tidylearn model
#'
#' Unified interface for creating machine learning models
#' by wrapping established R packages. This function dispatches
#' to the appropriate underlying package based on the method.
#'
#' The wrapped packages include: stats (lm, glm, prcomp,
#' kmeans, hclust), glmnet, randomForest, xgboost, gbm,
#' e1071, nnet, rpart, cluster, and dbscan. The underlying
#' algorithms are unchanged - this function provides a
#' consistent interface and returns tidy output.
#'
#' Access the raw model object from the underlying package via \code{model$fit}.
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model. For
#'   unsupervised methods, use \code{~ vars} or NULL.
#' @param method The modeling method. Supervised: "linear"
#'   (stats::lm), "logistic" (stats::glm), "tree" (rpart),
#'   "forest" (randomForest), "boost" (gbm),
#'   "ridge"/"lasso"/"elastic_net" (glmnet), "svm" (e1071),
#'   "nn" (nnet), "deep" (keras), "xgboost" (xgboost).
#'   Unsupervised: "pca" (stats::prcomp),
#'   "mds" (stats/MASS/smacof), "kmeans" (stats::kmeans),
#'   "pam"/"clara" (cluster), "hclust" (stats::hclust),
#'   "dbscan" (dbscan).
#' @param compute Compute tier for the fit. One of \code{"cpu"} (default,
#'   existing behaviour), \code{"gpu"} (route to local CUDA when the
#'   method has an upstream GPU path -- xgboost and deep learning today),
#'   \code{"auto"} (consult \code{\link{tl_compute_advisor}} and pick per
#'   call), or \code{"cloud"} (reserved -- not yet wired up). When
#'   \code{"gpu"} is requested for a method without an upstream GPU path
#'   or on a machine without a detected GPU, the call falls back to CPU
#'   with a warning.
#' @param ... Additional arguments passed to the underlying model function
#' @return A \code{tidylearn_model} object (S3) containing the fitted model
#'   (\code{$fit}), model specification (\code{$spec}), and training data
#'   (\code{$data}). The object also inherits from a method-specific class
#'   (e.g., \code{tidylearn_linear}) and a paradigm class
#'   (\code{tidylearn_supervised} or \code{tidylearn_unsupervised}).
#' @export
#' @examples
#' \donttest{
#' # Classification -> wraps randomForest::randomForest()
#' model <- tl_model(iris, Species ~ ., method = "forest")
#' model$fit  # Access the raw randomForest object
#'
#' # Regression -> wraps stats::lm()
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' model$fit  # Access the raw lm object
#'
#' # PCA -> wraps stats::prcomp()
#' model <- tl_model(iris, ~ ., method = "pca")
#' model$fit  # Access the raw prcomp object
#'
#' # Clustering -> wraps stats::kmeans()
#' model <- tl_model(iris, method = "kmeans", k = 3)
#' model$fit  # Access the raw kmeans object
#' }
tl_model <- function(data, formula = NULL, method = "linear", ...,
                     compute = "cpu") {
  # Validate inputs
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }

  # Define supervised and unsupervised methods
  supervised_methods <- c(
    "linear", "polynomial", "logistic", "tree",
    "forest", "boost", "ridge", "lasso",
    "elastic_net", "svm", "nn", "deep", "xgboost"
  )
  unsupervised_methods <- c(
    "pca", "mds", "kmeans", "pam",
    "clara", "hclust", "dbscan"
  )

  # Determine paradigm
  is_supervised <- method %in% supervised_methods
  is_unsupervised <- method %in% unsupervised_methods

  if (!is_supervised && !is_unsupervised) {
    stop(
      "Unknown method: ", method,
      "\nSupervised methods: ",
      paste(supervised_methods, collapse = ", "),
      "\nUnsupervised methods: ",
      paste(unsupervised_methods, collapse = ", "),
      call. = FALSE
    )
  }

  # Route to appropriate function. Both paths thread `compute` through
  # tl_resolve_compute(), which applies the documented fallback rules
  # uniformly — including "cloud" -> error for all methods until Modal
  # lands.
  if (is_supervised) {
    tl_model_supervised(data, formula, method, compute = compute, ...)
  } else {
    tl_model_unsupervised(data, formula, method, compute = compute, ...)
  }
}

#' Create a supervised learning model
#'
#' Internal function for creating supervised models
#' @keywords internal
#' @noRd
tl_model_supervised <- function(data, formula, method, ..., compute = "cpu") {
  if (!inherits(formula, "formula")) {
    formula <- as.formula(formula)
  }

  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Determine if classification or regression
  y <- data[[response_var]]
  is_classification <- is.factor(y) || is.character(y)

  if (!is_classification && is.numeric(y) &&
        length(unique(y)) <= 10) {
    message(
      "Note: Response '", response_var, "' has ",
      length(unique(y)), " unique numeric values. ",
      "Treating as regression. Convert to factor ",
      "for classification."
    )
  }

  # Resolve the effective compute tier (handles auto/gpu fallbacks).
  # CPU-only methods short-circuit so they don't trigger advisor / GPU
  # detection unnecessarily.
  #
  # Forward the caller's runtime-relevant hyperparameters (nrounds,
  # ntree, epochs, ...) so "auto" estimates the job actually being run.
  # Without this the advisor sizes a default job and can be out by the
  # ratio of requested to default.
  dots <- list(...)
  hyperparams <- dots[vapply(
    dots,
    function(value) is.numeric(value) && length(value) == 1L,
    logical(1)
  )]

  effective_compute <- tl_resolve_compute(
    method, data, formula,
    compute = compute, hyperparams = hyperparams
  )

  # Record the training-time factor levels. Predict methods that build
  # their own design matrix need these to keep contrast coding identical
  # to the fit; without them, new data missing a level silently changes
  # the encoding.
  predictor_vars <- intersect(get_formula_vars(formula, data), names(data))
  xlev <- lapply(
    Filter(function(v) is.factor(data[[v]]), predictor_vars),
    function(v) levels(data[[v]])
  )
  names(xlev) <- Filter(function(v) is.factor(data[[v]]), predictor_vars)

  # Create model specification
  model_spec <- list(
    paradigm = "supervised",
    formula = formula,
    method = method,
    is_classification = is_classification,
    response_var = response_var,
    response_levels = if (is_classification) levels(as.factor(y)) else NULL,
    xlev = xlev,
    compute = effective_compute
  )

  # Fit the model based on method. Methods with an upstream GPU path
  # receive the resolved compute tier; others ignore it.
  fitted_model <- switch(
    method,
    "linear" = tl_fit_linear(data, formula, ...),
    "polynomial" = tl_fit_polynomial(data, formula, ...),
    "logistic" = tl_fit_logistic(data, formula, ...),
    "tree" = tl_fit_tree(data, formula, is_classification, ...),
    "forest" = tl_fit_forest(data, formula, is_classification, ...),
    "boost" = tl_fit_boost(data, formula, is_classification, ...),
    "ridge" = tl_fit_ridge(data, formula, is_classification, ...),
    "lasso" = tl_fit_lasso(data, formula, is_classification, ...),
    "elastic_net" = tl_fit_elastic_net(data, formula, is_classification, ...),
    "svm" = tl_fit_svm(data, formula, is_classification, ...),
    "nn" = tl_fit_nn(data, formula, is_classification, ...),
    "deep" = tl_fit_deep(
      data, formula, is_classification,
      compute = effective_compute, ...
    ),
    "xgboost" = tl_fit_xgboost(
      data, formula, is_classification,
      compute = effective_compute, ...
    ),
    stop("Unsupported supervised method: ", method, call. = FALSE)
  )

  # Create and return tidylearn model object
  model <- structure(
    list(
      spec = model_spec,
      fit = fitted_model,
      data = data
    ),
    class = c(
      paste0("tidylearn_", method),
      "tidylearn_supervised", "tidylearn_model"
    )
  )

  model
}

#' Create an unsupervised learning model
#'
#' Internal function for creating unsupervised models
#' @keywords internal
#' @noRd
tl_model_unsupervised <- function(data, formula = NULL, method, ...,
                                  compute = "cpu") {
  # For unsupervised learning, formula can be NULL or ~ vars

  # Resolve the effective compute tier. None of tidylearn's current
  # unsupervised methods have an upstream GPU path, so "gpu" warns and
  # falls back to CPU; "cloud" errors uniformly until Modal lands.
  effective_compute <- tl_resolve_compute(
    method, data, formula, compute = compute
  )

  # Create model specification
  model_spec <- list(
    paradigm = "unsupervised",
    formula = formula,
    method = method,
    compute = effective_compute
  )

  # Fit the model based on method
  fitted_model <- switch(
    method,
    "pca" = tl_fit_pca(data, formula, ...),
    "mds" = tl_fit_mds(data, formula, ...),
    "kmeans" = tl_fit_kmeans(data, formula, ...),
    "pam" = tl_fit_pam(data, formula, ...),
    "clara" = tl_fit_clara(data, formula, ...),
    "hclust" = tl_fit_hclust(data, formula, ...),
    "dbscan" = tl_fit_dbscan(data, formula, ...),
    stop("Unsupported unsupervised method: ", method, call. = FALSE)
  )

  # Create and return tidylearn model object
  model <- structure(
    list(
      spec = model_spec,
      fit = fitted_model,
      data = data
    ),
    class = c(
      paste0("tidylearn_", method),
      "tidylearn_unsupervised", "tidylearn_model"
    )
  )

  model
}

#' Predict using a tidylearn model
#'
#' Unified prediction interface for both supervised and unsupervised models
#'
#' @param object A tidylearn model object
#' @param new_data A data frame containing the new data.
#'   If NULL, uses training data.
#' @param type Type of prediction, for supervised models only:
#'   \code{"response"} (default), \code{"prob"} or \code{"class"}. Note
#'   that \code{"response"} is method-dependent -- logistic regression
#'   returns probabilities, trees and forests return class labels -- so
#'   pass \code{"class"} explicitly when you want labels. Ignored by
#'   unsupervised models, whose output is determined by the method.
#' @param ... Additional arguments
#' @return For supervised models, a \link[tibble]{tibble} with a
#'   \code{.pred} column; with \code{type = "prob"}, one column per class
#'   instead. For unsupervised models, the method's natural output: an
#'   \code{.obs_id} column plus component scores for \code{"pca"} and
#'   \code{"mds"}, or plus a \code{cluster} column for the clustering
#'   methods.
#'
#'   Unsupervised models differ in whether they can handle new data.
#'   \code{"pca"} projects it and \code{"kmeans"} assigns it to the
#'   nearest centre; \code{"pam"}, \code{"clara"}, \code{"dbscan"},
#'   \code{"mds"} and \code{"hclust"} have no out-of-sample projection
#'   and error if \code{new_data} is supplied. For hierarchical
#'   clustering, cut the tree with \code{tidy_cutree()} instead.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' predict(model)
#' predict(model, new_data = mtcars[1:5, ])
#' }
#' @export
predict.tidylearn_model <- function(object,
                                    new_data = NULL,
                                    type = "response",
                                    ...) {
  # Track whether the caller supplied data. Unsupervised methods behave
  # differently for training data than for new observations, and row
  # count is not a reliable way to tell the two apart.
  training <- is.null(new_data)

  if (training) {
    new_data <- object$data
  } else {
    # A model fitted on engineered features cannot read raw new data: the
    # columns it was trained on do not exist there. Models that carry a
    # record of how their features were built rebuild them first.
    new_data <- apply_feature_transform(object, new_data)
  }

  # Route to appropriate predict method
  if (inherits(object, "tidylearn_supervised")) {
    predict_supervised(object, new_data, type, ...)
  } else if (inherits(object, "tidylearn_unsupervised")) {
    predict_unsupervised(object, new_data, type, training = training, ...)
  } else {
    stop("Unknown model type", call. = FALSE)
  }
}

#' Rebuild engineered features on new data
#'
#' `tl_auto_ml()` fits some of its candidates on PCA scores or on a cluster
#' assignment rather than on the raw columns. Those models record how their
#' features were produced, so that predicting on raw new data reproduces the
#' same transformation -- fitted on the training data, replayed here -- instead
#' of failing on a column that only ever existed inside the search.
#'
#' @param object A tidylearn model.
#' @param new_data Raw data supplied to `predict()`.
#' @return `new_data`, with the engineered columns present.
#' @keywords internal
#' @noRd
apply_feature_transform <- function(object, new_data) {
  transform <- object$feature_transform
  if (is.null(transform)) {
    return(new_data)
  }

  response <- transform$response
  has_response <- !is.null(response) && response %in% names(new_data)
  response_values <- if (has_response) new_data[[response]] else NULL
  predictors <- if (has_response) {
    new_data[, setdiff(names(new_data), response), drop = FALSE]
  } else {
    new_data
  }

  out <- switch(
    transform$kind,
    "pca" = {
      scores <- predict(transform$reduction_model, new_data = predictors)
      scores[, setdiff(names(scores), ".obs_id"), drop = FALSE]
    },
    "cluster" = {
      assignment <- predict(transform$cluster_model, new_data = predictors)
      new_data[[transform$column]] <- factor(
        assignment$cluster,
        levels = transform$levels
      )
      return(new_data)
    },
    stop("Unknown feature transform: ", transform$kind, call. = FALSE)
  )

  if (has_response) {
    out[[response]] <- response_values
  }
  out
}

#' Predict using supervised models
#' @keywords internal
#' @noRd
predict_supervised <- function(object, new_data, type = "response", ...) {
  method <- object$spec$method

  # Route to method-specific prediction
  preds <- switch(
    method,
    "linear" = predict(object$fit, newdata = new_data, ...),
    "polynomial" = predict(object$fit, newdata = new_data, ...),
    "logistic" = tl_predict_logistic(object, new_data, type, ...),
    "tree" = tl_predict_tree(object, new_data, type, ...),
    "forest" = tl_predict_forest(object, new_data, type, ...),
    "boost" = tl_predict_boost(object, new_data, type, ...),
    "ridge" = tl_predict_glmnet(object, new_data, type, ...),
    "lasso" = tl_predict_glmnet(object, new_data, type, ...),
    "elastic_net" = tl_predict_glmnet(object, new_data, type, ...),
    "svm" = tl_predict_svm(object, new_data, type, ...),
    "nn" = tl_predict_nn(object, new_data, type, ...),
    "deep" = tl_predict_deep(object, new_data, type, ...),
    "xgboost" = tl_predict_xgboost(object, new_data, type, ...),
    stop(
      "Unsupported supervised method for prediction: ",
      method, call. = FALSE
    )
  )

  # Ensure tibble output
  if (is.data.frame(preds) || inherits(preds, "tbl")) {
    preds
  } else {
    tibble::tibble(.pred = preds)
  }
}

#' Keep only the leading components a reduction was asked for
#'
#' `tl_reduce_dimensions(n_components = k)` trims its returned data to the
#' first k components. The fitted model has to trim its predictions the same
#' way, or projecting a test set yields more columns than the model that
#' consumes them was trained on.
#'
#' @param x Matrix or data frame of component scores, widest first.
#' @param n_components Number of leading components to keep, or NULL for all.
#' @return `x`, trimmed to its first `n_components` columns.
#' @keywords internal
#' @noRd
truncate_components <- function(x, n_components) {
  if (is.null(n_components)) {
    return(x)
  }
  keep <- min(as.integer(n_components), ncol(x))
  x[, seq_len(keep), drop = FALSE]
}

#' Align new data to the columns a fitted unsupervised model was built on
#'
#' Selecting "every numeric column" from `new_data` silently produces a matrix
#' of the wrong width whenever the caller passes extra columns, or the same
#' columns in a different order. Downstream arithmetic then either recycles
#' (k-means centres) or transposes meaning (PCA rotation) without complaint.
#' Matching on name and erroring on a mismatch keeps that failure loud.
#'
#' @param new_data Data frame supplied to `predict()`.
#' @param expected Character vector of column names the fit was built on.
#' @param what Label used in the error message.
#' @return A numeric matrix with columns in `expected` order.
#' @keywords internal
#' @noRd
align_new_data <- function(new_data, expected, what) {
  missing_cols <- setdiff(expected, names(new_data))
  if (length(missing_cols) > 0) {
    stop(
      what, " was fitted on ", length(expected), " column(s) (",
      paste(expected, collapse = ", "), ") but new_data is missing: ",
      paste(missing_cols, collapse = ", "), ".",
      call. = FALSE
    )
  }
  x <- new_data[, expected, drop = FALSE]
  non_numeric <- expected[!vapply(x, is.numeric, logical(1))]
  if (length(non_numeric) > 0) {
    stop(
      what, " requires numeric columns, but new_data has non-numeric: ",
      paste(non_numeric, collapse = ", "), ".",
      call. = FALSE
    )
  }
  as.matrix(x)
}

#' Predict using unsupervised models
#' @keywords internal
#' @noRd
predict_unsupervised <- function(object, new_data, type = "response",
                                 training = FALSE, ...) {
  method <- object$spec$method

  # Methods with no out-of-sample projection: returning the training
  # result for new data would look like a prediction but is not one
  no_out_of_sample <- function(label, hint = NULL) {
    if (!training) {
      stop(
        label, " does not support out-of-sample prediction.",
        if (!is.null(hint)) paste0(" ", hint) else "",
        call. = FALSE
      )
    }
  }

  result <- switch(
    method,
    "pca" = {
      # For PCA, transform the new data
      if (training) {
        object$fit$scores
      } else {
        # Transform new data using the PCA rotation. The rotation's row
        # names are the training predictors, in the order prcomp() saw them.
        x_mat <- align_new_data(
          new_data,
          rownames(object$fit$model$rotation),
          "PCA"
        )
        if (object$fit$settings$center) {
          x_mat <- scale(
            x_mat,
            center = object$fit$model$center,
            scale = FALSE
          )
        }
        if (object$fit$settings$scale) {
          x_mat <- scale(
            x_mat,
            center = FALSE,
            scale = object$fit$model$scale
          )
        }
        scores <- x_mat %*% object$fit$model$rotation
        colnames(scores) <- paste0(
          "PC", seq_len(ncol(scores))
        )
        scores <- truncate_components(scores, object$spec$n_components)
        tibble::as_tibble(scores) %>%
          dplyr::mutate(
            .obs_id = as.character(seq_len(nrow(scores))),
            .before = 1
          )
      }
    },
    "kmeans" = {
      if (training) {
        object$fit$clusters
      } else {
        # Assign to nearest center. Columns are matched to the centre
        # matrix by name: recycling a mismatched row against a centre
        # returns a cluster number that looks valid and is not.
        centers <- object$fit$model$centers
        x_mat <- align_new_data(new_data, colnames(centers), "k-means")
        # apply() drops to a length-k vector when x_mat has a single row,
        # and max.col() then reads that as k rows of one column -- three
        # cluster numbers for one observation, with no error. Pin the
        # shape rather than trusting simplification.
        dists <- matrix(
          apply(centers, 1, function(centre) {
            rowSums((x_mat - rep(centre, each = nrow(x_mat)))^2)
          }),
          nrow = nrow(x_mat),
          ncol = nrow(centers)
        )
        clusters <- max.col(-dists, ties.method = "first")
        tibble::tibble(cluster = as.integer(clusters))
      }
    },
    "pam" = ,
    "clara" = {
      no_out_of_sample(toupper(method))
      object$fit$clusters
    },
    "mds" = {
      no_out_of_sample("Multidimensional scaling")
      truncate_components(object$fit$points, object$spec$n_components)
    },
    "hclust" = {
      no_out_of_sample("Hierarchical clustering")
      # The fit holds the tree, not cluster assignments -- those require
      # choosing a cut height or number of clusters
      stop(
        "Hierarchical clustering models carry a tree, not cluster ",
        "assignments. Use tidy_cutree(model$fit$model, k = ...) to cut ",
        "the tree.",
        call. = FALSE
      )
    },
    "dbscan" = {
      no_out_of_sample("DBSCAN")
      object$fit$clusters
    },
    stop(
      "Unsupported unsupervised method for prediction: ",
      method, call. = FALSE
    )
  )

  result
}


#' Print method for tidylearn models
#' @param x A tidylearn model object
#' @param ... Additional arguments (ignored)
#' @return The input object \code{x}, returned invisibly.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' print(model)
#' }
#' @export
print.tidylearn_model <- function(x, ...) {
  cat("tidylearn Model\n")
  cat("===============\n")
  cat("Paradigm:", x$spec$paradigm, "\n")
  cat("Method:", x$spec$method, "\n")

  if (x$spec$paradigm == "supervised") {
    cat(
      "Task:",
      ifelse(
        x$spec$is_classification,
        "Classification", "Regression"
      ), "\n"
    )
    cat("Formula:", deparse(x$spec$formula), "\n")
  } else {
    cat("Technique:", x$spec$method, "\n")
  }

  cat("\nTraining observations:", nrow(x$data), "\n")
  invisible(x)
}

#' Summary method for tidylearn models
#' @param object A tidylearn model object
#' @param ... Additional arguments (ignored)
#' @return The input \code{object}, returned invisibly. Called for its
#'   side effect of printing model summary and training performance.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' summary(model)
#' }
#' @export
summary.tidylearn_model <- function(object, ...) {
  print(object)

  cat("\n")
  if (inherits(object, "tidylearn_supervised")) {
    # Evaluate on training data
    eval_results <- tl_evaluate(object)
    cat("Training Performance:\n")
    print(eval_results)
  } else {
    # Show unsupervised model details
    cat("Model Components:\n")
    print(names(object$fit))
  }

  invisible(object)
}

#' Plot a supervised tidylearn model
#'
#' Dispatches to the appropriate plotting function based on model type and
#' requested plot type.
#'
#' @param model A tidylearn supervised model object
#' @param type Plot type. For regression: "auto", "actual_predicted",
#'   "residuals", "diagnostics". For classification: "auto", "confusion",
#'   "roc", "precision_recall", "calibration", "lift", "gain".
#'   "importance" is available for tree-based and regularized models.
#' @param ... Additional arguments passed to the underlying plot function
#' @return A ggplot2 object (invisibly for base-graphics plots)
#' @keywords internal
tl_plot_model <- function(model, type = "auto", ...) {
  is_class <- model$spec$is_classification

  if (type == "auto") {
    type <- if (is_class) "confusion" else "actual_predicted"
  }

  switch(
    type,
    # Regression plots
    "actual_predicted" = tl_plot_actual_predicted(model, ...),
    "residuals"        = tl_plot_residuals(model, ...),
    "diagnostics"      = tl_plot_diagnostics(model, ...),
    # Classification plots
    "confusion"        = tl_plot_confusion(model, ...),
    "roc"              = tl_plot_roc(model, ...),
    "precision_recall" = tl_plot_precision_recall(model, ...),
    "calibration"      = tl_plot_calibration(model, ...),
    "lift"             = tl_plot_lift(model, ...),
    "gain"             = tl_plot_gain(model, ...),
    # Shared
    "importance"       = tl_plot_importance(model, ...),
    stop(
      "Unknown plot type '", type, "'. ",
      if (is_class) {
        paste0(
          "Use: 'confusion', 'roc', ",
          "'precision_recall', 'calibration', ",
          "'lift', 'gain', or 'importance'."
        )
      } else {
        paste0(
          "Use: 'actual_predicted', ",
          "'residuals', 'diagnostics', ",
          "or 'importance'."
        )
      },
      call. = FALSE
    )
  )
}

#' Plot an unsupervised tidylearn model
#'
#' Dispatches to the appropriate plotting function based on the unsupervised
#' model method.
#'
#' @param model A tidylearn unsupervised model object
#' @param type Plot type (default: "auto"). Currently unused; reserved for
#'   future sub-type selection.
#' @param ... Additional arguments passed to the underlying plot function
#' @return A ggplot2 object or invisible result
#' @keywords internal
tl_plot_unsupervised <- function(model, type = "auto", ...) {
  method <- model$spec$method

  # The tl_fit_* wrappers unpack the tidy_* objects into plain lists, so
  # the plot helpers have to be handed the pieces they expect rather than
  # the fit itself
  cluster_data <- function() {
    clusters <- model$fit$clusters
    if (is.null(clusters) || !"cluster" %in% names(clusters)) {
      stop(
        "No cluster assignments found in the fitted ", method, " model.",
        call. = FALSE
      )
    }

    data <- model$data
    # As a factor so plot_clusters does not pick it as an axis
    data$cluster <- as.factor(clusters$cluster)
    data
  }

  switch(
    method,
    "pca"    = plot_variance_explained(model$fit$variance_explained, ...),
    "kmeans" = ,
    "pam"    = ,
    "clara"  = ,
    "dbscan" = plot_clusters(cluster_data(), ...),
    "hclust" = plot_dendrogram(model$fit$model, ...),
    "mds"    = plot_mds(
      structure(
        list(
          config = model$fit$points,
          method = model$fit$method,
          stress = model$fit$stress,
          gof = model$fit$gof
        ),
        class = "tidy_mds"
      ),
      ...
    ),
    stop(
      "Plotting not implemented for unsupervised method: ", method,
      call. = FALSE
    )
  )
}

#' Plot method for tidylearn models
#' @param x A tidylearn model object
#' @param type Plot type (default: "auto")
#' @param ... Additional arguments passed to plotting functions
#' @return A \code{\link[ggplot2]{ggplot}} object. The specific plot depends
#'   on the model paradigm and \code{type} argument.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' plot(model, type = "actual_predicted")
#' }
#' @export
plot.tidylearn_model <- function(x, type = "auto", ...) {
  if (inherits(x, "tidylearn_supervised")) {
    tl_plot_model(x, type, ...)
  } else if (inherits(x, "tidylearn_unsupervised")) {
    tl_plot_unsupervised(x, type, ...)
  }
}

#' Get tidylearn version information
#' @return A package_version object containing the version number
#' @examples
#' tl_version()
#' @export
tl_version <- function() {
  packageVersion("tidylearn")
}
