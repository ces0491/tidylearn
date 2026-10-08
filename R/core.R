#' @title tidylearn: A Unified Tidy Interface to R's Machine Learning Ecosystem
#' @name tidylearn-core
#' @description Core functionality for tidylearn. This package
#'   provides a unified tidyverse-compatible interface to
#'   established R machine learning packages including glmnet,
#'   randomForest, xgboost, e1071, rpart, gbm, nnet, cluster,
#'   and dbscan. The underlying algorithms are unchanged -
#'   tidylearn wraps them with consistent function signatures,
#'   tidy tibble output, and unified ggplot2-based
#'   visualization. Supervised models keep the wrapped object at
#'   model$fit; unsupervised ones put it at model$fit$model,
#'   alongside the tidied components.
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
#' For a supervised method, \code{model$fit} is the object the wrapped
#' function returned. An unsupervised method returns tidied components as
#' well, so the wrapped object sits at \code{model$fit$model} and
#' \code{model$fit} is the list holding both.
#'
#' For classification, the response is reduced to the classes it
#' actually contains: subsetting a data frame keeps every factor level,
#' and a level no row uses would otherwise be reported as a class, given
#' its own (zero) probability column, and counted when deciding whether
#' the problem is binary. The fit is unaffected.
#'
#' Whether a supervised model is a classification or a regression is
#' decided by the response the formula computes, so \code{factor(cyl) ~ wt}
#' is a classification even though \code{cyl} is numeric. A factor or text
#' response is a classification. A logical response is a regression for
#' every method but \code{"logistic"} -- with \code{"linear"}, a linear
#' probability model -- so write \code{factor(y) ~ ...} to classify it.
#'
#' A categorical predictor stored as text, as \code{tl_read()} returns it,
#' is made a factor before the fit and stored as one in \code{$data}, so
#' every method treats it as a category. At \code{predict()}, categorical
#' columns in new data are read against the levels seen in training: new
#' data may hold only some of them, and a level the model was not trained
#' on is an error that names the column.
#'
#' @section Method arguments:
#' Arguments in \code{...} are passed to the function the method wraps,
#' except for these, which tidylearn takes itself:
#' \describe{
#'   \item{\code{"polynomial"}}{\code{degree} (default 2). Each numeric main
#'     effect is replaced by \code{poly(term, degree, raw = TRUE)} and the
#'     result fitted with \code{lm()}. A numeric term is one that computes a
#'     numeric vector, such as \code{wt} or \code{log(wt)}, or a one-column
#'     matrix, such as \code{scale(wt)}. One that is also part of an
#'     interaction keeps its own term and gains \code{I(x^2)} up to
#'     \code{I(x^degree)}, so the interaction is coded as written. Factor
#'     and other non-numeric terms, interactions, \code{I()} terms, bases
#'     such as \code{poly()} or a spline's, the response as written, an
#'     \code{offset()} and a removed intercept are kept as they are.}
#'   \item{\code{"ridge"}, \code{"lasso"}, \code{"elastic_net"}}{
#'     \code{alpha}, glmnet's mixing parameter (by default 0, 1 and 0.5);
#'     \code{lambda}, a single penalty to fit at, a sequence of penalties for
#'     \code{glmnet::cv.glmnet()} to choose from, or \code{NULL} (the
#'     default) to let it choose its own; and \code{cv_folds} (default 5),
#'     the number of folds for that cross-validation, which takes the place
#'     of glmnet's \code{nfolds}. \code{predict()} uses the \code{lambda.1se}
#'     penalty. tidylearn sets \code{x}, \code{y}, \code{family} and
#'     \code{nfolds} itself and refuses them, along with any argument glmnet
#'     does not take.}
#'   \item{\code{"svm"}}{\code{tune} (default \code{FALSE}) and
#'     \code{tune_folds} (default 5), to choose \code{cost} by
#'     cross-validation before the fit, with \code{gamma} for a non-linear
#'     kernel and \code{degree} for a polynomial one.}
#'   \item{\code{"deep"}}{\code{hidden_layers}, \code{activation},
#'     \code{dropout}, \code{epochs}, \code{batch_size},
#'     \code{validation_split} and \code{learning_rate}.}
#'   \item{\code{"pca"}}{\code{scale} and \code{center} (both \code{TRUE}).}
#'   \item{\code{"mds"}}{\code{mds_method}, the variant:
#'     \code{"classical"} (the default, \code{stats::cmdscale()}),
#'     \code{"metric"} or \code{"nonmetric"} (smacof), or \code{"sammon"}
#'     or \code{"kruskal"} (MASS). \code{k}, or its alias \code{ndim}, is
#'     the number of dimensions (default 2).}
#'   \item{\code{"kmeans"}, \code{"pam"}, \code{"clara"}}{\code{k}, the
#'     number of clusters (default 3); for \code{"pam"}, \code{metric}
#'     as well (default \code{"euclidean"}).}
#'   \item{\code{"hclust"}}{\code{hclust_method}, the linkage:
#'     \code{"average"} (the default), \code{"ward.D"}, \code{"ward.D2"},
#'     \code{"single"}, \code{"complete"}, \code{"mcquitty"},
#'     \code{"median"} or \code{"centroid"}; and \code{distance} (default
#'     \code{"euclidean"}).}
#'   \item{\code{"dbscan"}}{\code{eps} (default 0.5), \code{minPts}
#'     (default 5) and \code{distance} (default \code{"euclidean"}).}
#' }
#' Some defaults differ from the wrapped function's: \code{"forest"} computes
#' importance (\code{importance = TRUE}), \code{"boost"} grows trees of
#' \code{interaction.depth = 3}, and \code{"nn"} fits \code{size = 5}
#' hidden units with \code{trace = FALSE}.
#'
#' \code{weights} and \code{subset} take values, one per row of
#' \code{data} (such as \code{weights = data$w}), not column names. A
#' \code{subset} is applied before the fit, and \code{$data} holds only
#' the rows it selects. Case weights are applied by every supervised method
#' except \code{"svm"} and \code{"deep"}, which refuse them. An offset,
#' written as \code{offset()} in the formula, is applied by
#' \code{"linear"}, \code{"polynomial"} and \code{"logistic"}; the other
#' methods refuse one, because their \code{predict()} would not add it
#' back.
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model. For
#'   unsupervised methods, use \code{~ vars} or NULL; a one-sided formula
#'   names columns, and \code{~ . - x} means every numeric column but
#'   \code{x}. Supervised methods need a two-sided formula.
#' @param method The modeling method. Supervised: "linear"
#'   (stats::lm), "polynomial" (stats::lm on polynomial terms),
#'   "logistic" (stats::glm), "tree" (rpart),
#'   "forest" (randomForest), "boost" (gbm),
#'   "ridge"/"lasso"/"elastic_net" (glmnet), "svm" (e1071),
#'   "nn" (nnet), "deep" (keras), "xgboost" (xgboost).
#'   The method and the response have to agree, and a mismatch is an
#'   error rather than a meaningless fit: \code{"linear"} and
#'   \code{"polynomial"} need a numeric response, \code{"logistic"}
#'   needs exactly two classes, and every other supervised method takes
#'   either.
#'   Unsupervised: "pca" (stats::prcomp),
#'   "mds" (stats::cmdscale, or smacof or MASS through \code{mds_method}),
#'   "kmeans" (stats::kmeans), "pam"/"clara" (cluster),
#'   "hclust" (stats::hclust), "dbscan" (dbscan).
#' @param compute Compute tier for the fit. One of \code{"cpu"} (default,
#'   existing behaviour), \code{"gpu"} (route to local CUDA when the
#'   method has an upstream GPU path -- xgboost and deep learning today),
#'   \code{"auto"} (consult \code{\link{tl_compute_advisor}} and pick per
#'   call), or \code{"cloud"} (reserved -- not yet wired up). When
#'   \code{"gpu"} is requested for a method without an upstream GPU path
#'   or on a machine without a detected GPU, the call falls back to CPU
#'   with a warning.
#' @param ... Arguments for the method: see the Method arguments section.
#'   Anything else is passed to the underlying model function.
#' @return A \code{tidylearn_model} object (S3) containing the fitted model
#'   (\code{$fit}, or \code{$fit$model} for an unsupervised method),
#'   model specification (\code{$spec}), and training data
#'   (\code{$data}). \code{update()} and \code{step()} on \code{$fit}
#'   refit on the training rows and weights, whatever is in the calling
#'   environment. The object also inherits from a method-specific class
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
#' model$fit$model  # The raw prcomp object, alongside tidied components
#'
#' # Clustering -> wraps stats::kmeans()
#' model <- tl_model(iris, method = "kmeans", k = 3)
#' model$fit$model  # The raw kmeans object
#' }
tl_model <- function(data, formula = NULL, method = "linear", ...,
                     compute = "cpu") {
  # Validate inputs
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }

  # Normalise once here so every downstream reader of all.vars() sees a
  # formula. Unsupervised methods are called without one.
  if (!is.null(formula)) {
    formula <- tl_as_formula(formula)
  }

  # Define supervised and unsupervised methods
  supervised_methods <- tl_supervised_methods()
  unsupervised_methods <- tl_unsupervised_methods()

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
  if (!is.null(formula) && !inherits(formula, "formula")) {
    formula <- as.formula(formula)
  }

  # A one-sided formula used to have its first variable read as the
  # response, and failed inside the backend with "incompatible dimensions"
  if (is.null(formula) || length(formula) != 3L) {
    stop(
      "Method \"", method, "\" is supervised and needs a two-sided formula ",
      "naming the response, such as y ~ x; got ",
      if (is.null(formula)) "none" else deparse1(formula), ".",
      call. = FALSE
    )
  }
  tl_check_subtracted(formula, data)

  # A subset is applied to the data before anything else reads it, so the
  # model stores the rows it was fitted on. Passed through to the fit, it
  # left model$data holding every row: predict(), tl_evaluate() and the
  # diagnostics then worked on rows the fit never saw, and the influence
  # measures failed on the mismatch. The other per-row arguments are
  # taken down to the same rows, as model.frame() would take them.
  dots <- list(...)
  if (!is.null(dots[["subset"]])) {
    kept <- tl_subset_rows(data, dots[["subset"]])
    if (length(kept) == 0L) {
      stop(
        "'subset' selects none of the ", nrow(data), " rows of data.",
        call. = FALSE
      )
    }
    for (arg in intersect(names2(dots), tl_per_row_args())) {
      if (length(dots[[arg]]) == nrow(data)) {
        dots[[arg]] <- dots[[arg]][kept]
      }
    }
    dots[["subset"]] <- NULL
    model <- do.call(
      tl_model_supervised,
      c(
        list(data = data[kept, , drop = FALSE], formula = formula,
             method = method),
        dots,
        list(compute = compute)
      )
    )
    model$spec$per_row_args <- union(model$spec$per_row_args, "subset")
    return(model)
  }

  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Determine if classification or regression from the response the
  # formula computes, which is the column itself only for a bare name.
  # Messages name the response as written.
  y <- tl_formula_response(formula, data)
  response_label <- if (is.name(formula[[2L]])) {
    response_var
  } else {
    deparse1(formula[[2L]])
  }
  is_classification <- is.factor(y) || is.character(y)

  # Refuse a method that cannot fit this response before anything
  # reinterprets it, so the message describes what the caller passed.
  tl_check_method_response(method, y, response_label, is_classification)

  # Logistic regression is a classification method whatever the response
  # is stored as. Left alone, a 0/1 integer response produced a binomial
  # glm described by a spec that said is_classification = FALSE, so
  # tl_evaluate() scored it with rmse, mae and rsq, and asking it for
  # accuracy returned an empty tibble -- no error, no warning.
  # The warning has a class of its own so that a caller refitting once per
  # fold can let it through once rather than once per fold.
  if (method == "logistic" && !is_classification) {
    warning(warningCondition(
      "Converting response variable to factor for logistic regression",
      class = "tidylearn_response_conversion"
    ))
    is_classification <- TRUE
  }

  if (!is_classification && is.numeric(y) &&
        length(unique(y)) <= 10) {
    message(
      "Note: Response '", response_label, "' has ",
      length(unique(y)), " unique numeric values. ",
      "Treating as regression. Convert to factor ",
      "for classification."
    )
  }

  # Normalise the response once, so the spec, the fitted model and every
  # predict path agree on what the classes are. Writing it back into
  # `data` matters as much as the local copy: `data` is what gets fitted
  # and what is stored on the model, and predict methods read the levels
  # back off it. A computed response is left to the formula: writing
  # I(mpg > 20) back over mpg would change what the formula computes.
  if (is_classification) {
    y <- tl_normalise_response(y)
    if (is.name(formula[[2L]])) {
      data[[response_var]] <- y
    }
  }

  # A category stored as text is made a factor, and stored as one. Left as
  # text, randomForest coded it by the values present in whatever frame it
  # was handed -- so a row's prediction depended on the rows scored with
  # it -- and gbm refused it outright. Only a column the formula uses as it
  # is changes type: one used inside a term keeps the type the term reads.
  text_columns <- Filter(
    function(v) is.character(data[[v]]),
    intersect(tl_bare_term_vars(formula, data), names(data))
  )
  for (column in text_columns) {
    data[[column]] <- factor(data[[column]])
  }

  # Resolve the effective compute tier (handles auto/gpu fallbacks).
  # CPU-only methods short-circuit so they don't trigger advisor / GPU
  # detection unnecessarily.
  #
  # Forward the caller's runtime-relevant hyperparameters (nrounds,
  # ntree, epochs, ...) so "auto" estimates the job actually being run.
  # Without this the advisor sizes a default job and can be out by the
  # ratio of requested to default. Every fit argument goes, whatever its
  # length -- hidden_layers is a vector -- and the advisor reads and checks
  # the ones its estimate uses.
  hyperparams <- dots

  effective_compute <- tl_resolve_compute(
    method, data, formula,
    compute = compute, hyperparams = hyperparams
  )

  # Record the training-time factor levels. predict() puts new data's
  # categorical columns on these before any method sees them, and predict
  # methods that build their own design matrix pass them on, so contrast
  # coding stays identical to the fit when new data holds only some of the
  # categories.
  predictor_vars <- intersect(get_formula_vars(formula, data), names(data))
  xlev <- lapply(
    Filter(function(v) is.factor(data[[v]]), predictor_vars),
    function(v) levels(data[[v]])
  )
  names(xlev) <- Filter(function(v) is.factor(data[[v]]), predictor_vars)

  # A formula that subtracts a column is fitted written out without it, so
  # the stored terms do not ask predict() for a column the model never
  # uses; the spec keeps the formula as the caller wrote it.
  fit_formula <- tl_fit_formula(formula, data)

  # The predictor terms of the training data travel with the levels: their
  # predvars hold what a data-dependent term such as scale(hp) was computed
  # with, and tl_predictor_matrix() builds new data's design from them
  # rather than recomputing the term on the rows predicted.
  attr(xlev, "terms") <- tryCatch(
    {
      predictor_terms <- stats::delete.response(
        stats::terms(fit_formula, data = data)
      )
      attr(
        stats::model.frame(predictor_terms, data, na.action = stats::na.pass),
        "terms"
      )
    },
    error = function(e) NULL
  )

  # The classes are those of the rows the fit uses. Most methods leave out
  # a row with a missing value, and a class whose every row has one is not
  # in the fit: the spec still listed it, so a binary glmnet fit was
  # described as three-class and plotted as multiclass.
  response_levels <- if (is_classification) {
    tl_fitted_classes(y, data, fit_formula, method, response_label)
  }

  # Create model specification
  model_spec <- list(
    paradigm = "supervised",
    formula = formula,
    method = method,
    is_classification = is_classification,
    response_var = response_var,
    response_levels = response_levels,
    xlev = xlev,
    compute = effective_compute,
    # The fitting arguments, so a refit on other rows -- a CV fold -- fits
    # the model the caller built rather than the method's defaults
    args = dots[!names2(dots) %in% tl_per_row_args()],
    # Arguments with one value per training row cannot be replayed on
    # other rows, so only their names are kept: storing the values made a
    # second copy of, say, a weight vector the fit already holds
    per_row_args = intersect(names2(dots), tl_per_row_args())
  )

  # Fit the model based on method. Methods with an upstream GPU path
  # receive the resolved compute tier; others ignore it.
  fitted_model <- switch(
    method,
    "linear" = tl_fit_linear(data, fit_formula, ...),
    "polynomial" = tl_fit_polynomial(data, fit_formula, ...),
    "logistic" = tl_fit_logistic(data, fit_formula, ...),
    "tree" = tl_fit_tree(data, fit_formula, is_classification, ...),
    "forest" = tl_fit_forest(data, fit_formula, is_classification, ...),
    "boost" = tl_fit_boost(data, fit_formula, is_classification, ...),
    "ridge" = tl_fit_ridge(data, fit_formula, is_classification, ...),
    "lasso" = tl_fit_lasso(data, fit_formula, is_classification, ...),
    "elastic_net" = tl_fit_elastic_net(
      data, fit_formula, is_classification, ...
    ),
    "svm" = tl_fit_svm(data, fit_formula, is_classification, ...),
    "nn" = tl_fit_nn(data, fit_formula, is_classification, ...),
    "deep" = tl_fit_deep(
      data, fit_formula, is_classification,
      compute = effective_compute, ...
    ),
    "xgboost" = tl_fit_xgboost(
      data, fit_formula, is_classification,
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
#'   pass \code{"class"} explicitly when you want labels. \code{"prob"}
#'   and \code{"class"} need a classification model, and any other value
#'   is an error. Ignored by unsupervised models, whose output is
#'   determined by the method.
#' @param ... Additional arguments
#' @return For supervised models, a \link[tibble]{tibble} with a
#'   \code{.pred} column; with \code{type = "prob"}, one column per class
#'   instead. It has one row per row of \code{new_data}, in order, and
#'   zero rows for zero-row \code{new_data}. For unsupervised models, the
#'   method's natural output: an \code{.obs_id} column, the row names of
#'   the data predicted on, plus component scores for \code{"pca"} and
#'   \code{"mds"} (as many as the model keeps), or plus a \code{cluster}
#'   column for the clustering methods.
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
#' Data that already holds the engineered columns, and lacks the raw columns
#' they are built from, is returned as it is: the model's own stored data,
#' passed back explicitly by \code{tl_evaluate(m, m$data)} or by a plot that
#' defaults to it, was projected a second time and failed for want of the
#' raw columns.
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

  raw_inputs <- switch(
    transform$kind,
    "pca" = rownames(transform$reduction_model$fit$model$rotation),
    "cluster" = colnames(transform$cluster_model$fit$model$centers),
    NULL
  )
  engineered <- tryCatch(
    get_formula_vars(object$spec$formula, object$data),
    error = function(e) NULL
  )
  if (length(raw_inputs) > 0L && length(engineered) > 0L &&
        all(engineered %in% names(new_data)) &&
        !all(raw_inputs %in% names(new_data))) {
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

#' Methods to offer when the requested one cannot fit the response
#'
#' Every method that takes either a numeric or a categorical response,
#' less the two whose packages are only suggested. \code{"deep"} needs
#' keras and a Python backend and \code{"xgboost"} needs xgboost, so
#' naming them here would answer one error with another on the machines
#' that do not have them. Both still work when installed; they are absent
#' from the advice, not from the package.
#'
#' @keywords internal
#' @noRd
tl_dual_task_suggestions <- function() {
  c("tree", "forest", "boost", "ridge", "lasso",
    "elastic_net", "svm", "nn")
}

#' Refuse a method that cannot fit the response it was handed
#'
#' Both directions of the mismatch used to be accepted.
#' \code{lm()} on a factor estimates from the underlying integer codes, so
#' \code{tl_model(iris, Species ~ ., method = "linear")} returned numbers on
#' a scale where setosa is 1 and virginica is 3 -- and never failed, at any
#' point, so nothing told the caller. Checking here keeps the complaint next
#' to the decision that caused it.
#'
#' @param method The requested method
#' @param y The response, as supplied
#' @param response_var Its name, for the message
#' @param is_classification Whether the response is a factor or character
#' @return `TRUE`, invisibly, when the method can fit the response
#' @keywords internal
#' @noRd
tl_check_method_response <- function(method, y, response_var,
                                     is_classification) {
  quoted <- function(x) paste0("\"", x, "\"", collapse = ", ")

  if (method %in% c("linear", "polynomial") && is_classification) {
    n_classes <- nlevels(tl_normalise_response(y))
    alternatives <- if (n_classes == 2) {
      c("logistic", tl_dual_task_suggestions())
    } else {
      tl_dual_task_suggestions()
    }
    kind <- if (is.factor(y)) "factor" else "character vector"

    # A single-class response is not a task any method can take. The
    # equally-spaced-codes argument does not apply to one class, and the
    # usual "refit with one of these" tail would send the caller round a
    # loop of methods that each refuse it in turn.
    if (n_classes == 1) {
      stop(
        "Method \"", method, "\" fits a numeric response, but '",
        response_var, "' is a ", kind, " holding a single class (",
        quoted(levels(tl_normalise_response(y))),
        "). It is not a regression target, and no classification method ",
        "will fit it either -- there is nothing to discriminate.",
        call. = FALSE
      )
    }

    stop(
      "Method \"", method, "\" fits a numeric response, but '",
      response_var, "' is a ", kind,
      " with ", n_classes, " classes. lm() estimates from the underlying ",
      "integer codes, so the classes would be treated as equally spaced ",
      "points on a scale and the predictions would be numbers between ",
      "them. Refit with method = ", quoted(alternatives), ".",
      call. = FALSE
    )
  }

  # A 0/1 or 1/2 coding is a two-class response stored as a number, and is
  # allowed. Anything with more distinct values is either a measurement or
  # a multiclass coding, and neither is something binomial glm can fit.
  if (method == "logistic" && is.numeric(y)) {
    n_distinct <- length(unique(stats::na.omit(y)))
    if (n_distinct > 2) {
      stop(
        "Logistic regression needs a two-class response, but '",
        response_var, "' is numeric with ", n_distinct,
        " distinct values. If those are measurements, use a regression ",
        "method: ",
        quoted(c("linear", "polynomial", tl_dual_task_suggestions())),
        ". If they encode classes, convert '", response_var,
        "' to a factor first -- though logistic still handles only two.",
        call. = FALSE
      )
    }
  }

  # A response with one class gives every classification method something
  # it cannot fit, but only logistic said so plainly. The rest reported
  # whatever their backend hit first: rpart "number of rows of matrices
  # must match (see arg 2)", glmnet "non-conformable arguments", e1071
  # "Model is empty!", xgboost a complaint about num_class. None of them
  # named the response or the cause.
  if (is_classification && method != "logistic") {
    present <- levels(tl_normalise_response(y))
    if (length(present) == 1L) {
      stop(
        "Method \"", method, "\" needs a response with at least two ",
        "classes, but '", response_var, "' has only one (",
        quoted(present), "). There is nothing to discriminate.",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}

#' The classes among the rows a fit uses
#'
#' rpart, gbm and xgboost route a missing predictor value themselves and
#' fit every row with a response. The other methods leave out any row with
#' a missing value in a variable of the formula, so a class all of whose
#' rows have one is not among the classes they fit.
#'
#' @param y The normalised response, one value per row of \code{data}
#' @param data The training data
#' @param formula The formula the method is fitted on
#' @param method The method
#' @param response_label The response as written, for the message
#' @return The classes, in level order
#' @keywords internal
#' @noRd
tl_fitted_classes <- function(y, data, formula, method, response_label) {
  if (method %in% c("tree", "boost", "xgboost")) {
    return(levels(y))
  }
  columns <- intersect(
    all.vars(stats::terms(formula, data = data)), names(data)
  )
  if (length(columns) == 0L) {
    return(levels(y))
  }

  # A response with one class in every row is refused by the method checks
  # with their own wording; this is for one left by the missing values.
  # The glmnet fitter refuses that case itself, from its model frame.
  classes <- levels(droplevels(y[stats::complete.cases(data[columns])]))
  if (length(classes) == 1L && nlevels(y) > 1L &&
        !method %in% c("ridge", "lasso", "elastic_net")) {
    stop(
      "Method \"", method, "\" leaves out rows with a missing value, and in ",
      "the rows left '", response_label, "' has only one class (\"",
      classes, "\"). There is nothing to discriminate. Impute the missing ",
      "values, or use method = \"tree\" or \"boost\", which fit rows with ",
      "missing predictors.",
      call. = FALSE
    )
  }
  classes
}

#' Predict using supervised models
#' @keywords internal
#' @noRd
predict_supervised <- function(object, new_data, type = "response", ...) {
  method <- object$spec$method
  tl_check_predict_type(object, type)
  new_data <- tl_prepare_new_data(object, new_data)

  predict_rows <- function(rows) {
    # Route to method-specific prediction
    preds <- switch(
      method,
      "linear" = predict(object$fit, newdata = rows, ...),
      "polynomial" = predict(object$fit, newdata = rows, ...),
      "logistic" = tl_predict_logistic(object, rows, type, ...),
      "tree" = tl_predict_tree(object, rows, type, ...),
      "forest" = tl_predict_forest(object, rows, type, ...),
      "boost" = tl_predict_boost(object, rows, type, ...),
      "ridge" = tl_predict_glmnet(object, rows, type, ...),
      "lasso" = tl_predict_glmnet(object, rows, type, ...),
      "elastic_net" = tl_predict_glmnet(object, rows, type, ...),
      "svm" = tl_predict_svm(object, rows, type, ...),
      "nn" = tl_predict_nn(object, rows, type, ...),
      "deep" = tl_predict_deep(object, rows, type, ...),
      "xgboost" = tl_predict_xgboost(object, rows, type, ...),
      stop(
        "Unsupported supervised method for prediction: ",
        method, call. = FALSE
      )
    )

    # Ensure tibble output
    if (is.data.frame(preds)) {
      preds
    } else {
      tibble::tibble(.pred = preds)
    }
  }

  # Zero rows is ordinary input -- a filter that matched nothing -- but each
  # backend fails on it its own way: glm "eta must be a nonempty numeric
  # vector", multinomial glmnet "non-conformable arrays", xgboost a pointer
  # misalignment. One training row is predicted instead and none of it is
  # kept, so the empty result has exactly the columns and types a
  # non-empty one has.
  if (nrow(new_data) == 0L && NROW(object$data) > 0L) {
    template <- tl_prepare_new_data(object, tl_template_row(object))
    return(predict_rows(template)[0, , drop = FALSE])
  }

  predict_rows(new_data)
}

#' Refuse a prediction type the model cannot give
#'
#' Every supervised method's predict takes \code{"response"}, \code{"prob"}
#' and \code{"class"}. A regression model ignored the type, so
#' \code{type = "prob"} on one returned its numeric predictions as if they
#' were probabilities, and a misspelt type went unnoticed.
#'
#' @param object A supervised tidylearn model
#' @param type The requested type
#' @return `TRUE`, invisibly, when the type is one the model can give
#' @keywords internal
#' @noRd
tl_check_predict_type <- function(object, type) {
  if (!is.character(type) || length(type) != 1L || is.na(type) ||
        !type %in% c("response", "prob", "class")) {
    stop(
      "Invalid prediction type ", tl_describe_value(type), ". Use ",
      "type = \"response\", \"prob\" or \"class\".",
      call. = FALSE
    )
  }

  if (type != "response" && !isTRUE(object$spec$is_classification)) {
    response <- if (inherits(object$spec$formula, "formula")) {
      deparse1(object$spec$formula[[2L]])
    } else {
      object$spec$response_var
    }
    stop(
      "type = \"", type, "\" needs a classification model, but this \"",
      object$spec$method, "\" model is a regression of '", response,
      "'. Use type = \"response\".",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

#' Hand a method's predict only the columns and levels it was fitted on
#'
#' Two things about new data used to change predictions without changing
#' anything the model uses. A column the model never saw counted wherever
#' a method expanded the formula's dot over new_data: a mostly-missing
#' notes column turned 27 of 30 svm predictions NA, and gave xgboost a
#' design matrix wider than its features. And a categorical column was read
#' by the levels new_data declared, not the training levels. The columns
#' are cut down to those the formula reads, and the categorical ones put on
#' their training levels, before any method's predict sees them.
#'
#' A predictor missing from new_data is refused first. model.frame() looks
#' a variable up in the data and then in the formula's environment, so a
#' missing column was taken from a same-named object in the caller's
#' session, and the predictions were built from it.
#'
#' @param object A supervised tidylearn model
#' @param new_data Data to predict on
#' @return `new_data`, prepared
#' @keywords internal
#' @noRd
tl_prepare_new_data <- function(object, new_data) {
  missing_cols <- setdiff(tl_predictor_columns(object), names(new_data))
  if (length(missing_cols) > 0) {
    stop(
      "New data is missing predictors used at fit time: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  columns <- tl_model_columns(object$spec$formula, object$data)
  if (!is.null(columns)) {
    new_data <- new_data[, intersect(names(new_data), columns), drop = FALSE]
  }
  tl_align_predictor_levels(new_data, object$spec$xlev, object$data)
}

#' The training columns a model's predictor terms read
#'
#' The variables of the predictor terms, an offset's included, that were
#' columns of the training data. A variable the formula takes from its
#' environment instead -- a scalar in \code{offset(k * disp)} -- is not a
#' column new data has to carry, and a column the formula subtracts is not
#' in the terms at all.
#'
#' @param object A supervised tidylearn model
#' @return Column names; none when the model has no training data
#' @keywords internal
#' @noRd
tl_predictor_columns <- function(object) {
  if (is.null(object$data)) {
    return(character(0))
  }
  predictor_terms <- attr(object$spec$xlev, "terms")
  if (!inherits(predictor_terms, "terms")) {
    predictor_terms <- tryCatch(
      stats::delete.response(stats::terms(
        tl_fit_formula(object$spec$formula, object$data, predicting = TRUE),
        data = object$data
      )),
      error = function(e) NULL
    )
  }
  if (is.null(predictor_terms)) {
    return(character(0))
  }
  intersect(all.vars(predictor_terms), names(object$data))
}

#' One training row to take a prediction's shape from
#'
#' The first with every column the formula reads present, so the template
#' does not depend on how a backend treats missing values.
#'
#' @param object A supervised tidylearn model with training data
#' @return A one-row data frame
#' @keywords internal
#' @noRd
tl_template_row <- function(object) {
  data <- object$data
  columns <- intersect(
    tl_model_columns(object$spec$formula, data) %||% names(data),
    names(data)
  )
  complete <- if (length(columns) > 0L) {
    which(stats::complete.cases(data[columns]))
  }
  data[if (length(complete) > 0L) complete[1] else 1L, , drop = FALSE]
}

#' Keep only the leading components a reduction was asked for
#'
#' `tl_reduce_dimensions(n_components = k)` trims its returned data to the
#' first k components. The fitted model has to trim its predictions the same
#' way, or projecting a test set yields more columns than the model that
#' consumes them was trained on.
#'
#' An `.obs_id` column is kept and not counted: counting it left a 2-D MDS
#' fit predicting `.obs_id` and `Dim1` alone.
#'
#' @param x Matrix or data frame of component scores, widest first.
#' @param n_components Number of leading components to keep, or NULL for all.
#' @return `x`, trimmed to `.obs_id`, if it has one, and its first
#'   `n_components` components.
#' @keywords internal
#' @noRd
truncate_components <- function(x, n_components) {
  if (is.null(n_components)) {
    return(x)
  }
  is_id <- colnames(x) %in% ".obs_id"
  components <- which(!is_id)
  keep <- components[seq_len(min(as.integer(n_components), length(components)))]
  x[, sort(c(which(is_id), keep)), drop = FALSE]
}

#' Identify the rows of a data frame for `.obs_id`
#'
#' The training path names observations by row name, so new data is named
#' the same way rather than numbered from 1.
#'
#' @param data A data frame.
#' @return A character vector, one id per row.
#' @keywords internal
#' @noRd
tl_obs_ids <- function(data) {
  as.character(rownames(data) %||% seq_len(nrow(data)))
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
        truncate_components(object$fit$scores, object$spec$n_components)
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
        tibble::as_tibble(scores) |>
          dplyr::mutate(.obs_id = tl_obs_ids(new_data), .before = 1)
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
        tibble::tibble(
          .obs_id = tl_obs_ids(new_data),
          cluster = as.integer(clusters)
        )
      }
    },
    "pam" = ,
    "clara" = {
      no_out_of_sample(toupper(method))
      object$fit$clusters
    },
    "mds" = {
      no_out_of_sample("Multidimensional scaling")
      # The coordinates carry .obs_id only when the distances were labelled,
      # and a tibble or a frame with automatic row names gives them none
      points <- object$fit$points
      if (!".obs_id" %in% names(points)) {
        ids <- tl_obs_ids(object$data)
        if (length(ids) != nrow(points)) {
          ids <- as.character(seq_len(nrow(points)))
        }
        points <- dplyr::mutate(points, .obs_id = ids, .before = 1)
      }
      truncate_components(points, object$spec$n_components)
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
#'   "diagnostics" needs a model fitted by \code{lm()} or \code{glm()}:
#'   method "linear", "polynomial" or "logistic".
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
    # Shared. A regularised model's importance is its standardised
    # coefficients, which the tree-based plot does not compute
    "importance"       = if (model$spec$method %in%
                               c("ridge", "lasso", "elastic_net")) {
      tl_plot_importance_regularized(model, ...)
    } else {
      tl_plot_importance(model, ...)
    },
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
