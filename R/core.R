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
tl_model <- function(data, formula = NULL, method = "linear", ...) {
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

  # Route to appropriate function
  if (is_supervised) {
    tl_model_supervised(data, formula, method, ...)
  } else {
    tl_model_unsupervised(data, formula, method, ...)
  }
}

#' Create a supervised learning model
#'
#' Internal function for creating supervised models
#' @keywords internal
#' @noRd
tl_model_supervised <- function(data, formula, method, ...) {
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

  # Create model specification
  model_spec <- list(
    paradigm = "supervised",
    formula = formula,
    method = method,
    is_classification = is_classification,
    response_var = response_var
  )

  # Fit the model based on method
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
    "deep" = tl_fit_deep(data, formula, is_classification, ...),
    "xgboost" = tl_fit_xgboost(data, formula, is_classification, ...),
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
tl_model_unsupervised <- function(data, formula = NULL, method, ...) {
  # For unsupervised learning, formula can be NULL or ~ vars

  # Create model specification
  model_spec <- list(
    paradigm = "unsupervised",
    formula = formula,
    method = method
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
#' @param type Type of prediction. For supervised:
#'   "response" (default), "prob", "class". For
#'   unsupervised: "scores", "clusters", "transform"
#'   depending on method.
#' @param ... Additional arguments
#' @return A \link[tibble]{tibble} with a \code{.pred} column containing
#'   predictions. For classification with \code{type = "prob"}, returns
#'   columns for each class probability.
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
  if (is.null(new_data)) {
    new_data <- object$data
  }

  # Route to appropriate predict method
  if (inherits(object, "tidylearn_supervised")) {
    predict_supervised(object, new_data, type, ...)
  } else if (inherits(object, "tidylearn_unsupervised")) {
    predict_unsupervised(object, new_data, type, ...)
  } else {
    stop("Unknown model type", call. = FALSE)
  }
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

#' Predict using unsupervised models
#' @keywords internal
#' @noRd
predict_unsupervised <- function(object, new_data, type = "response", ...) {
  method <- object$spec$method

  result <- switch(
    method,
    "pca" = {
      # For PCA, transform the new data
      if (is.null(new_data) || nrow(new_data) == nrow(object$data)) {
        object$fit$scores
      } else {
        # Transform new data using the PCA rotation
        x_mat <- new_data %>%
          dplyr::select(where(is.numeric)) %>%
          as.matrix()
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
        tibble::as_tibble(scores) %>%
          dplyr::mutate(
            .obs_id = as.character(seq_len(nrow(scores))),
            .before = 1
          )
      }
    },
    "kmeans" = {
      if (is.null(new_data) || nrow(new_data) == nrow(object$data)) {
        object$fit$clusters
      } else {
        # Assign to nearest center
        x_mat <- new_data %>%
          dplyr::select(where(is.numeric)) %>%
          as.matrix()
        centers <- object$fit$model$centers
        dists <- apply(x_mat, 1, function(x) {
          apply(centers, 1, function(c) sum((x - c)^2))
        })
        clusters <- apply(dists, 2, which.min)
        tibble::tibble(cluster = as.integer(clusters))
      }
    },
    "mds" = {
      # MDS doesn't naturally support new data prediction
      object$fit$points
    },
    "hclust" = {
      # Hierarchical clustering doesn't support standard prediction
      warning(
        "Hierarchical clustering does not support ",
        "standard out-of-sample prediction."
      )
      object$fit$clusters
    },
    "dbscan" = {
      # DBSCAN doesn't support standard prediction
      warning("DBSCAN does not support standard out-of-sample prediction.")
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

  switch(
    method,
    "pca"    = plot_variance_explained(get_pca_variance(model$fit), ...),
    "kmeans" = ,
    "pam"    = ,
    "clara"  = ,
    "dbscan" = plot_clusters(
      cbind(model$data, cluster = model$fit$cluster), ...
    ),
    "hclust" = plot_dendrogram(model$fit, ...),
    "mds"    = plot_mds(model$fit, ...),
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
