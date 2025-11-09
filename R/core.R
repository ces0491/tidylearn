#' @title tidylearn: A Unified Tidy Approach to Machine Learning
#' @name tidylearn-core
#' @description Core functionality for the tidylearn package
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
#' Unified interface for creating both supervised and unsupervised machine learning models.
#' This function automatically detects the learning paradigm based on the method and formula.
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model. For unsupervised methods, use \code{~ vars} or NULL.
#' @param method The modeling method. Supervised: "linear", "logistic", "tree", "forest", "boost",
#'   "ridge", "lasso", "elastic_net", "svm", "nn", "deep", "xgboost".
#'   Unsupervised: "pca", "mds", "kmeans", "pam", "clara", "hclust", "dbscan".
#' @param ... Additional arguments to pass to the underlying model function
#' @return A tidylearn model object
#' @export
#' @examples
#' \dontrun{
#' # Supervised: Classification
#' model <- tl_model(iris, Species ~ ., method = "forest")
#'
#' # Supervised: Regression
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#'
#' # Unsupervised: PCA
#' model <- tl_model(iris, ~ ., method = "pca")
#'
#' # Unsupervised: Clustering
#' model <- tl_model(iris, method = "kmeans", k = 3)
#' }
tl_model <- function(data, formula = NULL, method = "linear", ...) {
  # Validate inputs
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }

  # Define supervised and unsupervised methods
  supervised_methods <- c("linear", "polynomial", "logistic", "tree", "forest", "boost",
                         "ridge", "lasso", "elastic_net", "svm", "nn", "deep", "xgboost")
  unsupervised_methods <- c("pca", "mds", "kmeans", "pam", "clara", "hclust", "dbscan")

  # Determine paradigm
  is_supervised <- method %in% supervised_methods
  is_unsupervised <- method %in% unsupervised_methods

  if (!is_supervised && !is_unsupervised) {
    stop("Unknown method: ", method,
         "\nSupervised methods: ", paste(supervised_methods, collapse = ", "),
         "\nUnsupervised methods: ", paste(unsupervised_methods, collapse = ", "),
         call. = FALSE)
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
  is_classification <- is.factor(y) || is.character(y) || (is.numeric(y) && length(unique(y)) <= 10)

  if (is_classification && is.numeric(y)) {
    warning("Response appears to be categorical but is stored as numeric. Consider converting to factor.")
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
    class = c(paste0("tidylearn_", method), "tidylearn_supervised", "tidylearn_model")
  )

  return(model)
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
    class = c(paste0("tidylearn_", method), "tidylearn_unsupervised", "tidylearn_model")
  )

  return(model)
}

#' Predict using a tidylearn model
#'
#' Unified prediction interface for both supervised and unsupervised models
#'
#' @param object A tidylearn model object
#' @param new_data A data frame containing the new data. If NULL, uses training data.
#' @param type Type of prediction. For supervised: "response" (default), "prob", "class".
#'   For unsupervised: "scores", "clusters", "transform" depending on method.
#' @param ... Additional arguments
#' @return Predictions as a tibble
#' @export
predict.tidylearn_model <- function(object, new_data = NULL, type = "response", ...) {
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

#' Print method for tidylearn models
#' @export
print.tidylearn_model <- function(x, ...) {
  cat("tidylearn Model\n")
  cat("===============\n")
  cat("Paradigm:", x$spec$paradigm, "\n")
  cat("Method:", x$spec$method, "\n")

  if (x$spec$paradigm == "supervised") {
    cat("Task:", ifelse(x$spec$is_classification, "Classification", "Regression"), "\n")
    cat("Formula:", deparse(x$spec$formula), "\n")
  } else {
    cat("Technique:", x$spec$method, "\n")
  }

  cat("\nTraining observations:", nrow(x$data), "\n")
  invisible(x)
}

#' Summary method for tidylearn models
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

#' Plot method for tidylearn models
#' @export
plot.tidylearn_model <- function(x, type = "auto", ...) {
  if (inherits(x, "tidylearn_supervised")) {
    tl_plot_model(x, type, ...)
  } else if (inherits(x, "tidylearn_unsupervised")) {
    tl_plot_unsupervised(x, type, ...)
  }
}

#' Get model version information
#' @export
tl_version <- function() {
  packageVersion("tidylearn")
}
