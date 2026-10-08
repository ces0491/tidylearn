#' @title Visualization Functions for tidylearn
#' @name tidylearn-visualization
#' @description General visualization functions for tidylearn models
#' @importFrom ggplot2 ggplot aes geom_line geom_point geom_bar geom_boxplot
#' @importFrom ggplot2 geom_histogram geom_density
#' @importFrom ggplot2 geom_jitter scale_color_gradient
#' @importFrom ggplot2 labs theme_minimal
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr mutate filter group_by summarize arrange
NULL

#' Plot feature importance across multiple models
#'
#' Each model's importance is rescaled on its own: its largest value
#' becomes 100, or, when no value is positive (a forest's permutation
#' importance can be negative throughout), its largest magnitude becomes
#' -100. A factor predictor
#' appears once, under its own name, for every model: the largest of its
#' design columns stands for it where a method ranks those columns
#' separately (ridge, lasso, elastic net and xgboost). A predictor a model
#' was given but did not use scores zero for that model; one it was never
#' given has no bar for it. Tree, forest and boost models are given the
#' variables of an interaction such as \code{wt:hp} rather than the
#' interaction itself, so it has no bar for them. Features are ranked on
#' their mean importance over the models that were given them.
#'
#' @param ... tidylearn model objects to compare
#' @param top_n Number of top features to display (default: 10)
#' @param names Optional character vector of model names, one unique name
#'   per model
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' m1 <- tl_model(iris, Sepal.Length ~ ., method = "forest")
#' m2 <- tl_model(iris, Sepal.Length ~ ., method = "boost")
#' tl_plot_importance_comparison(m1, m2, names = c("Forest", "Boost"))
#' }
#' @export
tl_plot_importance_comparison <- function(..., top_n = 10, names = NULL) {
  # Get models
  models <- list(...)

  # The bars are keyed on these names, so two models sharing one were
  # drawn in the same places
  names <- tl_comparison_names(
    models, names %||% paste0("Model ", seq_along(models))
  )

  # Extract importance for each model
  all_importance <- purrr::map2_dfr(models, names, tl_comparison_importance)

  # This used to evaluate NULL without returning it, so the function
  # carried on and failed inside dplyr with "object 'feature' not found"
  if (is.null(all_importance) || nrow(all_importance) == 0) {
    stop(
      "None of the models has feature importance to compare. Supported ",
      "methods: tree, forest, boost, xgboost, ridge, lasso, elastic_net.",
      call. = FALSE
    )
  }

  # Find top features across all models. Each model has a row for every
  # predictor it was given, at zero where it did not use one, so a feature
  # a lasso dropped does not outrank one both models used.
  top_features <- all_importance |>
    dplyr::group_by(.data[["feature"]]) |>
    dplyr::summarize(
      avg_importance = mean(.data[["importance"]]),
      .groups = "drop"
    ) |>
    dplyr::arrange(dplyr::desc(.data[["avg_importance"]])) |>
    dplyr::slice_head(n = top_n) |>
    dplyr::pull(.data[["feature"]])

  # Filter to only top features
  plot_data <- all_importance |>
    dplyr::filter(.data[["feature"]] %in% top_features)

  # Create the plot. A feature only some models were given has fewer bars,
  # and preserving the single-bar width keeps a lone bar from filling the
  # whole slot.
  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = stats::reorder(feature, importance),
      y = importance,
      fill = model
    )
  ) +
    ggplot2::geom_col(
      position = ggplot2::position_dodge(preserve = "single")
    ) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Feature Importance Comparison",
      x = NULL,
      y = "Importance",
      fill = "Model"
    ) +
    ggplot2::theme_minimal()

  p
}

#' One model's importance for the comparison, one row per predictor
#'
#' rpart, randomForest and gbm name a factor predictor by its column
#' (\code{Species}); glmnet and xgboost name its design columns
#' (\code{Speciesversicolor}, \code{Speciesvirginica}). Filling zeros across
#' the two namings said the lasso gave \code{Species} nothing and the forest
#' gave the dummies nothing. Each name is mapped to the formula term it
#' came from -- by position for glmnet, whose design columns can share a
#' name -- and a term takes its largest column, so every model reports on
#' the same names. A predictor the model was given and did not rank
#' scores zero; one it was never given gets no row, so it draws no bar for
#' this model and does not pull down the predictor's average.
#'
#' @param model A tidylearn model.
#' @param name The model's name in the comparison.
#' @return A tibble of \code{feature}, \code{model} and \code{importance},
#'   or NULL for a method with no importance.
#' @keywords internal
#' @noRd
tl_comparison_importance <- function(model, name) {
  method <- model$spec$method
  if (method %in% c("tree", "forest", "boost", "xgboost")) {
    imp <- tl_extract_importance(model)
    # xgboost is fitted on the design matrix, which has a wt:hp column for
    # wt * hp; rpart, randomForest and gbm are handed the variables alone
    given <- if (method == "xgboost") "terms" else "variables"
    mapped <- tl_importance_terms(model, imp$feature, given = given)
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    # Two design columns can share a name -- a factor a with level b beside
    # a numeric column ab -- so a column's term comes from its position
    by_column <- tl_glmnet_column_importance(model)
    kept <- !is.na(by_column$importance) & by_column$importance > 0
    imp <- tibble::tibble(
      feature = by_column$column[kept],
      importance = tl_rescale_importance(by_column$importance[kept])
    )
    mapped <- list(
      terms = attr(stats::terms(model$spec$formula, data = model$data),
                   "term.labels"),
      feature_terms = tl_design_column_terms(model)[kept]
    )
  } else {
    warning(
      "Importance extraction not implemented for model type: ", method,
      call. = FALSE
    )
    return(NULL)
  }

  # An empty importance used to drop the model from the comparison with
  # nothing to say why
  if (nrow(imp) == 0) {
    warning(
      "Model '", name, "' has no feature with non-zero importance: ",
      tl_no_importance_reason(model), ". Its bars are all zero.",
      call. = FALSE
    )
  }

  by_term <- vapply(
    split(imp$importance, mapped$feature_terms), max, numeric(1)
  )

  features <- union(mapped$terms, names(by_term))
  importance <- unname(by_term[features])
  importance[is.na(importance)] <- 0

  # Shown as the predictor is named, without the backquotes a term label
  # puts around a non-syntactic name
  tibble::tibble(
    feature = gsub("`", "", features, fixed = TRUE),
    model = name,
    importance = importance
  )
}

#' The formula term each importance name belongs to
#'
#' @param model A tidylearn supervised model.
#' @param features Feature names an importance extractor returned.
#' @param given What the backend was given: \code{"terms"}, the formula's
#'   terms, for a fit on the design matrix; or \code{"variables"}, the
#'   variables those terms use, for rpart, randomForest and gbm, which take
#'   wt and hp for wt * hp and never a wt:hp column.
#' @return A list: \code{terms}, the predictors the model was given (term
#'   labels, or variables named as term labels name them), and
#'   \code{feature_terms}, the one each of \code{features} belongs to. A
#'   name that matches none is kept as it is.
#' @keywords internal
#' @noRd
tl_importance_terms <- function(model, features,
                                given = c("terms", "variables")) {
  given <- match.arg(given)
  formula <- model$spec$formula
  model_terms <- stats::terms(formula, data = model$data)
  labels <- attr(model_terms, "term.labels")
  if (given == "variables") {
    # Taking every term label as given drew a zero bar for wt:hp, a term
    # the backend never had a column for. A variable that only a
    # subtracted term names is in no term, and was not given either.
    factors <- attr(model_terms, "factors")
    labels <- if (length(factors) > 0L) {
      rownames(factors)[rowSums(factors) > 0]
    } else {
      character(0)
    }
    if (length(labels) > 0L) {
      # The design-column lookup below then maps the columns of a matrix
      # variable, poly(hp, 2)1 and poly(hp, 2)2, to that variable even when
      # only an interaction uses it
      variables_only <- stats::reformulate(labels)
      environment(variables_only) <- environment(formula)
      formula <- variables_only
    }
  }
  # A term label backquotes a non-syntactic name, `car weight`, as glmnet
  # and xgboost columns do; rpart names the variable without them, and
  # randomForest rebuilds its frame with data.frame(), which makes the name
  # syntactic. Unmapped, one predictor became two features, each with a
  # false zero.
  unquoted <- gsub("`", "", labels, fixed = TRUE)
  term_of <- c(
    stats::setNames(labels, labels),
    stats::setNames(labels, unquoted),
    stats::setNames(labels, make.names(unquoted)),
    stats::setNames(labels, make.names(labels))
  )

  if (!all(features %in% names(term_of))) {
    # Design-matrix columns: "assign" gives the term each column came from
    frame <- stats::model.frame(formula, data = model$data)
    design_terms <- stats::terms(frame)
    design <- stats::model.matrix(design_terms, frame)
    assign <- attr(design, "assign")
    in_term <- assign > 0
    term_of <- c(
      term_of,
      stats::setNames(
        attr(design_terms, "term.labels")[assign[in_term]],
        colnames(design)[in_term]
      )
    )
  }

  feature_terms <- unname(term_of[features])
  unmatched <- is.na(feature_terms)
  feature_terms[unmatched] <- features[unmatched]
  list(terms = labels, feature_terms = feature_terms)
}

#' Why a model has no feature importance
#'
#' @param model A tidylearn model.
#' @return A phrase for an error or warning message.
#' @keywords internal
#' @noRd
tl_no_importance_reason <- function(model) {
  method <- model$spec$method
  if (method == "tree") {
    "the tree has no splits"
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    "the penalty dropped every predictor from this model"
  } else {
    "the fit did not use any predictor"
  }
}

#' Rescale importance so the largest value is 100
#'
#' Permutation importance is negative for a feature that does worse than
#' noise, and can be negative for every feature at once. Dividing by the
#' largest value, then itself negative, reversed the ranking: \%IncMSE of
#' -6.20, -0.87 and -6.61 became 709, 100 and 756. With no positive value
#' the largest magnitude sets the scale instead, which keeps the wrapped
#' package's order.
#'
#' @param x Numeric importance values.
#' @return \code{x} rescaled, or unchanged when every value is zero.
#' @keywords internal
#' @noRd
tl_rescale_importance <- function(x) {
  scale <- if (any(x > 0, na.rm = TRUE)) {
    max(x, na.rm = TRUE)
  } else {
    max(abs(x), 0, na.rm = TRUE)
  }
  if (scale == 0) {
    return(x)
  }
  100 * x / scale
}

#' Extract importance from a tree-based model
#'
#' @param model A tidylearn model object
#' @return A data frame with feature importance values, rescaled so the
#'   largest is 100. Empty for a tree with no splits.
#' @keywords internal
tl_extract_importance <- function(model) {
  # Get the model
  fit <- model$fit
  method <- model$spec$method

  if (method == "tree") {
    # rpart leaves variable.importance NULL for a tree with no splits, and
    # a tibble of two NULL columns has neither column
    imp <- fit$variable.importance %||%
      stats::setNames(numeric(0), character(0))

    # Create a data frame for plotting
    importance_df <- tibble::tibble(
      feature = names(imp),
      importance = unname(imp)
    )
  } else if (method == "forest") {
    # Random forest importance
    # Get variable importance from randomForest
    imp <- randomForest::importance(fit)

    # Permutation importance (mean decrease in accuracy, % increase in
    # MSE) exists only when the forest was fitted with importance = TRUE.
    # Otherwise the table holds the impurity measure alone, and asking for
    # the permutation column failed with "subscript out of bounds".
    permutation <- if (model$spec$is_classification) {
      "MeanDecreaseAccuracy"
    } else {
      "%IncMSE"
    }
    impurity <- if (model$spec$is_classification) {
      "MeanDecreaseGini"
    } else {
      "IncNodePurity"
    }
    measure <- if (permutation %in% colnames(imp)) permutation else impurity

    importance_df <- tibble::tibble(
      feature = rownames(imp),
      importance = unname(imp[, measure])
    )
  } else if (method == "xgboost") {
    # Gain: each feature's share of the loss reduction across its splits
    imp <- xgboost::xgb.importance(
      model = fit,
      feature_names = attr(fit, "feature_names")
    )
    importance_df <- tibble::tibble(
      feature = imp$Feature,
      importance = imp$Gain
    )
  } else if (method == "boost") {
    # gbm's relative influence, named after the columns gbm fitted on
    # rather than as summary() names it
    importance_df <- tl_gbm_influence(fit)
  } else {
    stop(
      "Variable importance extraction not implemented for method: ",
      method,
      call. = FALSE
    )
  }

  importance_df$importance <- tl_rescale_importance(importance_df$importance)

  importance_df
}

#' Relative influence of each column a gbm fit was given
#'
#' gbm names its influence after the formula's term labels, but it fits on
#' a frame of the variables those terms use, and an influence belongs to a
#' position in that frame. For wt * hp + qsec the frame is wt, hp and qsec,
#' and summary() added a wt:hp it had no column for, at zero. For
#' wt:hp + qsec the frame is qsec, wt and hp: wt's influence was reported
#' as wt:hp's, hp's had no name, and summary() failed with "row names
#' contain missing values". Each position is named here after the column
#' gbm built, as \code{gbm()} builds it from \code{fit$var.names}. A fit
#' from \code{gbm.fit()}, which has no terms, was handed its columns and
#' names each one.
#'
#' @param fit A \code{gbm} object.
#' @return A tibble of \code{feature} and \code{importance}, the relative
#'   influence over all the fitted trees, one row per column, largest
#'   first as \code{summary()} orders it.
#' @keywords internal
#' @noRd
tl_gbm_influence <- function(fit) {
  features <- if (is.null(fit$Terms)) {
    fit$var.names
  } else {
    rownames(
      attr(stats::terms(stats::reformulate(fit$var.names)), "factors")
    )
  }

  # A column past the last term label that no tree split on is beyond the
  # end of gbm's vector, and has no influence
  influence <- unname(gbm::relative.influence(fit, n.trees = fit$n.trees))
  influence <- c(influence, numeric(max(0L, length(features) -
                                          length(influence))))
  influence <- influence[seq_along(features)]

  ordered <- order(influence, decreasing = TRUE)
  tibble::tibble(
    feature = features[ordered],
    importance = influence[ordered]
  )
}

#' Extract importance from a regularized regression model
#'
#' @param model A tidylearn regularized model object
#' @param lambda Which lambda to use: "1se" (default), "min", or a numeric
#'   penalty within the fitted path
#' @return A data frame with feature importance values: each coefficient's
#'   absolute value times its predictor's standard deviation, so the
#'   ranking does not depend on units, rescaled to a maximum of 100. For a
#'   multiclass model a predictor takes its largest value across classes.
#' @keywords internal
tl_get_importance_regularized <- function(model, lambda = "1se") {
  by_column <- tl_glmnet_column_importance(model, lambda)

  # A penalty large enough to drop every predictor leaves nothing to rank,
  # and an empty column comes back empty
  kept <- !is.na(by_column$importance) & by_column$importance > 0
  tibble::tibble(
    feature = by_column$column[kept],
    importance = tl_rescale_importance(by_column$importance[kept])
  )
}

#' Importance of each design column of a regularised model
#'
#' A coefficient is per unit of its predictor, so |coefficient| ranked
#' predictors by their units: hp / 100 made hp 100 times as important
#' without changing a single prediction. Each is scaled by its column's
#' standard deviation.
#'
#' \code{coef()} lists the intercept and then the design columns in the
#' order the model was fitted on them, once per class for a multinomial
#' fit, so coefficients and standard deviations are matched by position.
#' Matched by name, two columns sharing one -- a factor a with level b
#' beside a numeric column ab -- both took the first column's standard
#' deviation and were merged into one row.
#'
#' @param model A tidylearn regularised model.
#' @param lambda As for \code{tl_get_importance_regularized()}.
#' @return A tibble with one row per design column, in design order:
#'   \code{column}, its name, and \code{importance}, |coefficient| x SD
#'   before rescaling. A multiclass column takes its largest value across
#'   classes.
#' @keywords internal
#' @noRd
tl_glmnet_column_importance <- function(model, lambda = "1se") {
  fit <- model$fit
  lambda_val <- tl_resolve_lambda(fit, lambda)

  # Coefficients at the selected lambda, stacked by class for a
  # multinomial fit, whose coef() is a list that as.matrix() could not use
  coefs <- tl_glmnet_coef_tbl(fit, lambda_val)
  class_of <- if ("class" %in% names(coefs)) {
    coefs$class
  } else {
    rep("", nrow(coefs))
  }
  slope <- coefs$term != "(Intercept)"
  coefs <- coefs[slope, , drop = FALSE]
  class_of <- class_of[slope]
  position <- stats::ave(seq_along(class_of), class_of, FUN = seq_along)
  columns <- coefs$term[class_of == class_of[1]]

  column_sd <- tl_glmnet_column_sd(model)
  if (length(column_sd) != length(columns)) {
    stop("could not match every coefficient to a design column. ",
         "Please report this with a reproducible example.", call. = FALSE)
  }

  # A multiclass predictor matters as much as its largest effect on any
  # class
  effect <- abs(coefs$estimate) * unname(column_sd)[position]
  tibble::tibble(
    column = columns,
    importance = vapply(split(effect, position), max, numeric(1),
                        USE.NAMES = FALSE)
  )
}

#' Standard deviation of each design column of a regularised model
#'
#' The fit records them, in design-column order, on the rows it was fitted
#' on. Computed from the stored data instead, they took in every row,
#' including any that \code{subset} left out of the fit; that is still the
#' fallback for a fit without the record.
#'
#' @param model A tidylearn regularised model.
#' @return A numeric vector, one value per design column, in design order.
#' @keywords internal
#' @noRd
tl_glmnet_column_sd <- function(model) {
  recorded <- attr(model$fit, "tl_x_sd")
  if (!is.null(recorded)) {
    return(recorded)
  }

  # The intercept is dropped by name: a formula with - 1 has none, and
  # dropping the first column took the first predictor's SD with it
  frame <- stats::model.frame(model$spec$formula, data = model$data)
  design <- stats::model.matrix(stats::terms(frame), frame)
  design <- design[, colnames(design) != "(Intercept)", drop = FALSE]
  apply(design, 2, stats::sd)
}

#' The formula term each design column of a model belongs to
#'
#' @param model A tidylearn supervised model.
#' @return The term label of each design column, in design order.
#' @keywords internal
#' @noRd
tl_design_column_terms <- function(model) {
  frame <- stats::model.frame(model$spec$formula, data = model$data)
  design_terms <- stats::terms(frame)
  assign <- attr(stats::model.matrix(design_terms, frame), "assign")
  attr(design_terms, "term.labels")[assign[assign > 0]]
}

#' Plot model comparison
#'
#' @param ... tidylearn model objects to compare
#' @param new_data Optional data frame for evaluation. If NULL, the models
#'   are scored on their training data, which they must share: models
#'   fitted on different data are an error asking for \code{new_data}. A
#'   model fitted on engineered features, as \code{tl_auto_ml()} builds some
#'   of its candidates, is scored on the training data of the others.
#' @param metrics Character vector of metrics to compute
#' @param names Optional character vector of model names
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' m1 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
#' tl_plot_model_comparison(m1, m2, names = c("Linear", "Lasso"))
#' }
#' @export
tl_plot_model_comparison <- function(
    ...,
    new_data = NULL,
    metrics = NULL,
    names = NULL) {
  # Get models
  models <- list(...)

  names <- tl_comparison_names(models, names, function(model) {
    task <- if (model$spec$is_classification) "classification" else "regression"
    paste0(model$spec$method, " (", task, ")")
  })

  # Check if all models are of the same type (classification or regression)
  is_classifications <- purrr::map_lgl(
    models,
    function(model) model$spec$is_classification
  )
  if (length(unique(is_classifications)) > 1) {
    stop(
      "All models must be of the same type (classification or regression)",
      call. = FALSE
    )
  }

  is_classification <- is_classifications[1]

  # Without new_data the models are scored on the training rows they share
  if (is.null(new_data)) {
    new_data <- tl_shared_training_data(models, names)
    message(
      "Evaluating on training data. ",
      "For model validation, provide separate test data."
    )
  }

  # Default metrics based on problem type
  if (is.null(metrics)) {
    if (is_classification) {
      metrics <- c("accuracy", "precision", "recall", "f1", "auc")
    } else {
      metrics <- c("rmse", "mae", "rsq", "mape")
    }
  }

  # Evaluate each model
  model_results <- purrr::map2_dfr(models, names, function(model, name) {
    # Evaluate model
    eval_results <- tl_evaluate(model, new_data, metrics)

    # Add model name
    eval_results$model <- name

    eval_results
  })

  # Create the plot
  p <- ggplot2::ggplot(
    model_results,
    ggplot2::aes(x = model, y = value, fill = metric)
  ) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::labs(
      title = "Model Comparison",
      x = NULL,
      y = "Metric Value",
      fill = "Metric"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )

  p
}

#' The training data compared models share, for when no new_data is given
#'
#' With no \code{new_data} the comparisons scored every model on the first
#' model's training rows, so a model fitted on other rows was scored
#' partly on rows it never saw, without a word. Those rows stand in for
#' \code{new_data} only when the models were fitted on the same data.
#'
#' A model fitted on engineered features -- \code{tl_auto_ml()}'s PCA and
#' cluster candidates -- stores those features, and \code{predict()}
#' rebuilds them from raw rows, so every model is scored on raw rows taken
#' from a model that stores them. A PCA candidate's stored frame cannot be
#' that reference: handed to a model fitted on the raw columns it lacks
#' them, and handed back to the candidate it is projected a second time.
#' It is left out of the check. A cluster candidate stores the raw rows
#' beside its cluster column, and those rows are checked and can be the
#' reference.
#'
#' @param models List of tidylearn models.
#' @param names Their names in the comparison.
#' @return The raw training rows of the first model that stores them, to
#'   score every model on; an error when the models' rows differ, or when
#'   no model stores its rows.
#' @keywords internal
#' @noRd
tl_shared_training_data <- function(models, names) {
  rows <- lapply(models, tl_stored_raw_rows)
  stored <- !vapply(rows, is.null, logical(1))
  if (!any(stored)) {
    stop(
      "None of the models stores the rows it was fitted on: each was ",
      "fitted on engineered features such as PCA scores. Pass the rows to ",
      "compare them on as 'new_data'.",
      call. = FALSE
    )
  }

  first <- which(stored)[1]
  reference <- rows[[first]]
  differs <- stored & !vapply(
    rows,
    function(model_rows) {
      is.null(model_rows) || tl_same_training_data(model_rows, reference)
    },
    logical(1)
  )
  if (any(differs)) {
    stop(
      "The models were fitted on different data, so they share no ",
      "training rows to be compared on. Models whose training data ",
      "differs from that of '", names[first], "': ",
      paste0("'", names[differs], "'", collapse = ", "),
      ". Pass the rows to compare them on as 'new_data'.",
      call. = FALSE
    )
  }
  reference
}

#' The raw rows a model stores
#'
#' @param model A tidylearn model.
#' @return The stored data for a model fitted on its own columns; for a
#'   cluster candidate, the stored data without the cluster column it
#'   added; NULL for a model fitted on PCA scores, which keeps no raw rows.
#' @keywords internal
#' @noRd
tl_stored_raw_rows <- function(model) {
  transform <- model$feature_transform
  if (is.null(transform)) {
    model$data
  } else if (identical(transform$kind, "cluster")) {
    model$data[setdiff(names(model$data), transform$column)]
  }
}

#' Whether two models were fitted on the same data
#'
#' \code{tl_model()} stores the frame it fitted after making a factor of
#' each text column its formula uses, so two models of one frame can store
#' a column as a factor and as text. Text and factor columns are compared
#' by their labels and numbers by value, which keeps those models counted
#' as fitted on the same data.
#'
#' @param a,b The stored data of two models.
#' @return TRUE when both hold the same columns and the same rows.
#' @keywords internal
#' @noRd
tl_same_training_data <- function(a, b) {
  if (!identical(dim(a), dim(b)) || !setequal(names(a), names(b))) {
    return(FALSE)
  }
  same_column <- function(column) {
    x <- a[[column]]
    y <- b[[column]]
    if (is.factor(x) || is.character(x) || is.factor(y) || is.character(y)) {
      identical(as.character(x), as.character(y))
    } else if (is.numeric(x) && is.numeric(y)) {
      identical(as.numeric(x), as.numeric(y))
    } else {
      identical(x, y)
    }
  }
  all(vapply(names(a), same_column, logical(1)))
}

#' Plot cross-validation results
#'
#' @param cv_results Cross-validation results from tl_cv function
#' @param metrics Character vector of metrics to plot
#'   (if NULL, plots all metrics)
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @export
#' @examples
#' \donttest{
#' cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 5)
#' tl_plot_cv_results(cv)
#'
#' # One metric rather than every one the folds scored
#' tl_plot_cv_results(cv, metrics = "rmse")
#' }
tl_plot_cv_results <- function(cv_results, metrics = NULL) {
  # tl_cv() returns $folds (a list of per-fold tibbles, with no fold
  # column) and a $summary keyed on `mean`. Reading $fold_metrics and
  # `mean_value` produced a ggplot that only failed when it was drawn.
  fold_metrics <- cv_results$fold_metrics

  if (is.null(fold_metrics)) {
    if (is.null(cv_results$folds)) {
      stop(
        "'cv_results' does not look like tl_cv() output: expected a ",
        "$folds component.",
        call. = FALSE
      )
    }

    fold_metrics <- dplyr::bind_rows(
      cv_results$folds, .id = "fold"
    )
    fold_metrics$fold <- as.integer(fold_metrics$fold)
  }

  summary_data <- cv_results$summary
  if (!is.null(summary_data) && "mean" %in% names(summary_data) &&
        !"mean_value" %in% names(summary_data)) {
    summary_data$mean_value <- summary_data$mean
  }

  # Filter metrics if specified
  if (!is.null(metrics)) {
    available <- unique(fold_metrics$metric)
    if (!any(metrics %in% available)) {
      stop(
        "None of the requested metrics is in the cross-validation ",
        "results. Available: ", paste(available, collapse = ", "), ".",
        call. = FALSE
      )
    }
    absent <- setdiff(metrics, available)
    if (length(absent) > 0) {
      warning(
        "Metric(s) not in the cross-validation results: ",
        paste(absent, collapse = ", "), ". Available: ",
        paste(available, collapse = ", "), ".",
        call. = FALSE
      )
    }

    fold_metrics <- fold_metrics |>
      dplyr::filter(.data[["metric"]] %in% metrics)
    # The mean lines are a layer of their own. Left unfiltered they gave
    # every metric the folds scored a panel, with a mean line and nothing
    # else in it.
    if (!is.null(summary_data)) {
      summary_data <- summary_data |>
        dplyr::filter(.data[["metric"]] %in% metrics)
    }
  }

  # Create the plot
  p <- ggplot2::ggplot(
    fold_metrics,
    ggplot2::aes(
      x = factor(fold),
      y = value,
      group = metric,
      color = metric
    )
  ) +
    ggplot2::geom_line() +
    ggplot2::geom_point() +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::geom_hline(
      data = summary_data,
      ggplot2::aes(yintercept = mean_value, color = metric),
      linetype = "dashed"
    ) +
    ggplot2::labs(
      title = "Cross-Validation Results",
      subtitle = "Dashed lines represent mean values across folds",
      x = "Fold",
      y = "Metric Value",
      color = "Metric"
    ) +
    ggplot2::theme_minimal()

  p
}

#' Create interactive visualization dashboard for a model
#'
#' @param model A tidylearn model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A \code{\link[shiny]{shinyApp}} object.
#' @examplesIf rlang::is_installed(c("shiny", "shinydashboard", "DT"))
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' app <- tl_dashboard(model)
#' }
#' @export
tl_dashboard <- function(model, new_data = NULL, ...) {
  # Check if required packages are installed
  tl_check_packages(c("shiny", "shinydashboard", "DT"))

  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Define UI
  ui <- shinydashboard::dashboardPage(
    shinydashboard::dashboardHeader(title = "tidylearn Model Dashboard"),

    shinydashboard::dashboardSidebar(
      shinydashboard::sidebarMenu(
        shinydashboard::menuItem(
          "Overview", tabName = "overview",
          icon = shiny::icon("dashboard")
        ),
        shinydashboard::menuItem(
          "Performance", tabName = "performance",
          icon = shiny::icon("chart-line")
        ),
        shinydashboard::menuItem(
          "Predictions", tabName = "predictions",
          icon = shiny::icon("table")
        ),
        shinydashboard::menuItem(
          "Diagnostics", tabName = "diagnostics",
          icon = shiny::icon("chart-area")
        )
      )
    ),

    shinydashboard::dashboardBody(
      shinydashboard::tabItems(
        # Overview tab
        shinydashboard::tabItem(
          tabName = "overview",
          shiny::fluidRow(
            shinydashboard::box(
              title = "Model Summary",
              width = 12,
              shiny::verbatimTextOutput("model_summary")
            )
          ),
          shiny::fluidRow(
            shinydashboard::box(
              title = "Feature Importance",
              width = 12,
              shiny::plotOutput("importance_plot")
            )
          )
        ),

        # Performance tab
        shinydashboard::tabItem(
          tabName = "performance",
          shiny::fluidRow(
            shinydashboard::box(
              title = "Performance Metrics",
              width = 12,
              DT::DTOutput("metrics_table")
            )
          ),
          shiny::conditionalPanel(
            condition = "output.is_classification == true",
            shiny::fluidRow(
              shinydashboard::box(
                title = "ROC Curve",
                width = 6,
                shiny::plotOutput("roc_plot")
              ),
              shinydashboard::box(
                title = "Confusion Matrix",
                width = 6,
                shiny::plotOutput("confusion_plot")
              )
            )
          ),
          shiny::conditionalPanel(
            condition = "output.is_classification == false",
            shiny::fluidRow(
              shinydashboard::box(
                title = "Actual vs Predicted",
                width = 6,
                shiny::plotOutput("actual_predicted_plot")
              ),
              shinydashboard::box(
                title = "Residuals",
                width = 6,
                shiny::plotOutput("residuals_plot")
              )
            )
          )
        ),

        # Predictions tab
        shinydashboard::tabItem(
          tabName = "predictions",
          shiny::fluidRow(
            shinydashboard::box(
              title = "Predictions",
              width = 12,
              DT::DTOutput("predictions_table")
            )
          )
        ),

        # Diagnostics tab
        shinydashboard::tabItem(
          tabName = "diagnostics",
          shiny::conditionalPanel(
            condition = "output.is_classification == false",
            shiny::fluidRow(
              shinydashboard::box(
                title = "Diagnostic Plots",
                width = 12,
                shiny::plotOutput("diagnostics_plot")
              )
            )
          ),
          shiny::conditionalPanel(
            condition = "output.is_classification == true",
            shiny::fluidRow(
              shinydashboard::box(
                title = "Calibration Plot",
                width = 6,
                shiny::plotOutput("calibration_plot")
              ),
              shinydashboard::box(
                title = "Precision-Recall Curve",
                width = 6,
                shiny::plotOutput("pr_curve_plot")
              )
            )
          )
        )
      )
    )
  )

  # Define server logic
  server <- function(input, output, session) {
    # Flag for classification or regression
    output$is_classification <- shiny::reactive({
      model$spec$is_classification
    })
    shiny::outputOptions(output, "is_classification", suspendWhenHidden = FALSE)

    # Model summary
    output$model_summary <- shiny::renderPrint({
      summary(model)
    })

    # Performance metrics
    output$metrics_table <- DT::renderDT({
      metrics <- tl_evaluate(model, new_data)
      DT::datatable(metrics,
                    options = list(pageLength = 10),
                    rownames = FALSE)
    })

    # Feature importance
    output$importance_plot <- shiny::renderPlot({
      plot <- tl_dashboard_importance_plot(model)
      shiny::validate(
        shiny::need(
          !is.null(plot),
          "Feature importance not available for this model type"
        )
      )
      plot
    })

    # Predictions
    output$predictions_table <- DT::renderDT({
      DT::datatable(tl_dashboard_predictions(model, new_data),
                    options = list(pageLength = 10),
                    rownames = FALSE)
    })

    # ROC plot (for classification)
    output$roc_plot <- shiny::renderPlot({
      if (model$spec$is_classification) {
        tl_plot_roc(model, new_data)
      }
    })

    # Confusion matrix (for classification)
    output$confusion_plot <- shiny::renderPlot({
      if (model$spec$is_classification) {
        tl_plot_confusion(model, new_data)
      }
    })

    # Actual vs predicted plot (for regression)
    output$actual_predicted_plot <- shiny::renderPlot({
      if (!model$spec$is_classification) {
        tl_plot_actual_predicted(model, new_data)
      }
    })

    # Residuals plot (for regression)
    output$residuals_plot <- shiny::renderPlot({
      if (!model$spec$is_classification) {
        tl_dashboard_residuals_plot(model, new_data)
      }
    })

    # Diagnostics plots (for regression). They read an lm fit's
    # standardised residuals, leverage and Cook's distance, and failed
    # inside rstandard() for every other method.
    output$diagnostics_plot <- shiny::renderPlot({
      if (!model$spec$is_classification) {
        problem <- tl_dashboard_diagnostics_issue(model)
        shiny::validate(shiny::need(is.null(problem), problem))
        tl_dashboard_diagnostics_plot(model)
      }
    })

    # Calibration plot (for classification)
    output$calibration_plot <- shiny::renderPlot({
      if (model$spec$is_classification) {
        tl_plot_calibration(model, new_data)
      }
    })

    # Precision-Recall curve (for classification)
    output$pr_curve_plot <- shiny::renderPlot({
      if (model$spec$is_classification) {
        tl_plot_precision_recall(model, new_data)
      }
    })
  }

  # Return the Shiny app
  shiny::shinyApp(ui, server)
}

#' Importance plot for the dashboard's importance panel
#'
#' The panel sent ridge, lasso and elastic net to
#' \code{tl_plot_importance()}, which handles tree-based methods only, so
#' for those models the panel showed an error in place of a plot.
#'
#' @param model A tidylearn model.
#' @return A ggplot, or NULL for a method with no importance plot.
#' @keywords internal
#' @noRd
tl_dashboard_importance_plot <- function(model) {
  method <- model$spec$method
  if (method %in% c("tree", "forest", "boost", "xgboost")) {
    tl_plot_importance(model)
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    tl_plot_importance_regularized(model)
  } else {
    NULL
  }
}

#' Rows for the dashboard's predictions table
#'
#' The observed values are the formula's left-hand side evaluated on the
#' data, the scale the model predicts on. Read from the raw column, a
#' \code{log(mpg) ~ wt + hp} model listed mpg beside predictions of
#' log(mpg), and residuals that subtracted one from the other.
#'
#' @param model A tidylearn supervised model.
#' @param new_data The data the dashboard evaluates on.
#' @return A data frame of \code{actual} and \code{predicted}, with
#'   \code{residual} for regression and one probability column per class
#'   for classification.
#' @keywords internal
#' @noRd
tl_dashboard_predictions <- function(model, new_data) {
  actuals <- tl_observed_response(model, new_data)

  if (model$spec$is_classification) {
    pred_class <- predict(model, new_data, type = "class")$.pred
    pred_prob <- predict(model, new_data, type = "prob")
    cbind(
      data.frame(actual = actuals, predicted = pred_class),
      pred_prob
    )
  } else {
    predictions <- unname(predict(model, new_data)$.pred)
    data.frame(
      actual = actuals,
      predicted = predictions,
      residual = actuals - predictions
    )
  }
}

#' Residual plot for the dashboard's residuals panel
#'
#' The panel called \code{tl_plot_residuals(model, new_data)}, which takes
#' a plot type as its second argument, and failed with "the condition has
#' length > 1" for every regression model. The residuals are computed from
#' predictions on the dashboard's data, as the predictions table computes
#' them, so they exist for every method.
#'
#' @param model A tidylearn regression model.
#' @param new_data The data the dashboard evaluates on.
#' @return A ggplot of residuals against predicted values.
#' @keywords internal
#' @noRd
tl_dashboard_residuals_plot <- function(model, new_data) {
  predicted <- unname(predict(model, new_data)$.pred)
  plot_data <- tibble::tibble(
    predicted = predicted,
    residual = tl_observed_response(model, new_data) - predicted
  )
  plot_data <- plot_data[stats::complete.cases(plot_data), ]

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = .data$predicted, y = .data$residual)
  ) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    ggplot2::labs(
      title = "Residuals vs Predicted Values",
      x = "Predicted values",
      y = "Residuals"
    ) +
    ggplot2::theme_minimal()
}

#' The four regression diagnostics, arranged for the dashboard
#'
#' \code{tl_plot_diagnostics()} returns its four plots as a list.
#' \code{renderPlot()} printed the list, and each print replaced the plot
#' before it, so the panel showed only the last.
#'
#' @param model A tidylearn model whose fit is an \code{lm}.
#' @return The arranged \code{gtable}, invisibly. The plots are drawn as a
#'   side effect.
#' @keywords internal
#' @noRd
tl_dashboard_diagnostics_plot <- function(model) {
  gridExtra::grid.arrange(grobs = tl_plot_diagnostics(model), ncol = 2)
}

#' Why the dashboard cannot show the regression diagnostics
#'
#' @param model A tidylearn regression model.
#' @return NULL when the four plots can be drawn; otherwise the message the
#'   panel shows in their place.
#' @keywords internal
#' @noRd
tl_dashboard_diagnostics_issue <- function(model) {
  if (!inherits(model$fit, "lm")) {
    "Diagnostic plots are available for linear and polynomial models only."
  } else if (!requireNamespace("gridExtra", quietly = TRUE)) {
    paste0(
      "The diagnostic plots need the gridExtra package. Install it with: ",
      "install.packages(\"gridExtra\")"
    )
  }
}

#' Bin number for each of n ranked rows
#'
#' Spreads the rows over the bins as evenly as they divide, so every bin
#' is used whenever there are at least as many rows as bins. Sizing bins
#' with ceiling(n / bins) left the last ones empty: 32 rows in 10 bins
#' became 8 bins of 4.
#'
#' @param n Number of rows.
#' @param bins Number of bins requested.
#' @return An integer vector of length n, non-decreasing, from 1 to
#'   min(n, bins).
#' @keywords internal
#' @noRd
tl_bin_index <- function(n, bins) {
  # bins = 0 gave one bin numbered 0, and bins = 2.5 gave three
  if (!is.numeric(bins) || length(bins) != 1L || is.na(bins) ||
        bins < 1 || bins != floor(bins)) {
    stop("'bins' must be a single whole number of at least 1", call. = FALSE)
  }
  if (n == 0) {
    return(integer(0))
  }
  bins <- min(bins, n)
  as.integer(ceiling(seq_len(n) * bins / n))
}

#' Rows ranked by predicted probability, for lift and gain charts
#'
#' Rows with tied probabilities have no order between them, and sorting
#' kept whatever order they arrived in: a tree scores many rows alike, so
#' the same model on the same rows gave a different gain curve when the
#' rows were reversed. Each row's outcome is replaced by the mean outcome
#' of its tie group -- the value any tie-breaking would give on average --
#' so a bin boundary falling inside a group takes its share of that group.
#'
#' A row missing its response or its probability used to turn every
#' cumulative total into NA and empty the chart. Those rows are left out,
#' with a warning giving the count.
#'
#' The observed classes are read against the model's, so the positive
#' class is the model's second class. Read from the data, a test factor
#' whose levels had been reordered ranked the rows by the other class's
#' probability. A row of a class the model never saw is left out, and
#' \code{tl_align_classes()} says so.
#'
#' @param model A binary classification model.
#' @param new_data Data to score.
#' @param model_levels The model's two classes.
#' @return A tibble of \code{prob} and \code{actual}, sorted by
#'   \code{prob} descending, where \code{actual} is the tie-group response
#'   rate for the positive (second) class.
#' @keywords internal
#' @noRd
tl_ranked_response <- function(model, new_data, model_levels) {
  observed <- tl_observed_response(model, new_data)
  aligned <- tl_align_classes(observed, model_levels)

  probs <- predict(model, new_data, type = "prob")
  pos_class <- model_levels[2]
  pos_probs <- probs[[pos_class]]

  missing <- is.na(observed) | (aligned$keep & is.na(pos_probs))
  if (any(missing)) {
    warning(
      sum(missing), " row(s) with a missing response or predicted ",
      "probability are left out of the chart.",
      call. = FALSE
    )
  }

  usable <- aligned$keep & !is.na(pos_probs)
  prob <- pos_probs[usable]
  actual <- as.numeric(aligned$actuals[usable] == pos_class)

  # Every cumulative share divides by the number of responders, so with
  # none the chart is 0 / 0 throughout
  if (!any(actual == 1)) {
    stop(
      "Lift and gain need at least one responder, and the scored rows ",
      "have no row of the positive class ('", pos_class, "').",
      call. = FALSE
    )
  }

  ord <- order(prob, decreasing = TRUE)
  prob <- prob[ord]
  actual <- actual[ord]

  tie_group <- match(prob, unique(prob))
  tibble::tibble(
    prob = prob,
    actual = stats::ave(actual, tie_group, FUN = mean)
  )
}

#' Plot lift chart for a classification model
#'
#' @param model A tidylearn classification model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param bins Number of bins for grouping predictions
#'   (default: 10)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @importFrom ggplot2 ggplot aes geom_line geom_point
#' @importFrom ggplot2 geom_hline labs theme_minimal
#' @examples
#' \donttest{
#' iris_bin <- iris[iris$Species != "setosa", ]
#' iris_bin$Species <- factor(iris_bin$Species)
#' model <- tl_model(iris_bin, Species ~ ., method = "logistic")
#' tl_plot_lift(model)
#' }
#' @export
tl_plot_lift <- function(model, new_data = NULL, bins = 10, ...) {
  if (!model$spec$is_classification) {
    stop(
      "Lift chart is only available for classification models",
      call. = FALSE
    )
  }

  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Binary is a property of the model. Counted from the data, a test split
  # that still declared a class the training rows dropped made a binary
  # model look multiclass.
  model_levels <- tl_model_classes(model)

  # For binary classification
  if (length(model_levels) == 2) {
    ordered_data <- tl_ranked_response(model, new_data, model_levels)

    # Calculate lift by decile. tl_bin_index() splits the rows into the
    # number of bins asked for; rounding the bin size up gave 32 rows in 10
    # bins as 8 groups of 4.
    bin <- tl_bin_index(nrow(ordered_data), bins)
    lift_data <- tibble::tibble(
      decile = integer(),
      cumulative_responders = integer(),
      cumulative_total = integer(),
      cumulative_response_rate = numeric(),
      baseline_rate = numeric(),
      lift = numeric()
    )

    baseline_rate <- mean(ordered_data$actual)
    cumulative_responders <- 0
    cumulative_total <- 0

    for (i in unique(bin)) {
      rows <- which(bin == i)

      # Update cumulative counts
      current_responders <- sum(ordered_data$actual[rows])
      current_total <- length(rows)

      cumulative_responders <- cumulative_responders + current_responders
      cumulative_total <- cumulative_total + current_total

      # Calculate metrics
      cumulative_response_rate <- cumulative_responders / cumulative_total
      lift <- cumulative_response_rate / baseline_rate

      # Add to results
      lift_data <- lift_data |>
        dplyr::add_row(
          decile = i,
          cumulative_responders = cumulative_responders,
          cumulative_total = cumulative_total,
          cumulative_response_rate = cumulative_response_rate,
          baseline_rate = baseline_rate,
          lift = lift
        )
    }

    # Create the plot
    p <- ggplot2::ggplot(lift_data, ggplot2::aes(x = decile, y = lift)) +
      ggplot2::geom_line(color = "blue") +
      ggplot2::geom_point(color = "blue", size = 3) +
      ggplot2::geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
      ggplot2::labs(
        title = "Lift Chart",
        subtitle = "Cumulative lift by decile",
        x = "Decile (sorted by predicted probability)",
        y = "Cumulative Lift"
      ) +
      ggplot2::scale_x_continuous(breaks = unique(bin)) +
      ggplot2::theme_minimal()

    p
  } else {
    stop(
      "Lift chart is currently only implemented for binary classification",
      call. = FALSE
    )
  }
}

#' Plot gain chart for a classification model
#'
#' @param model A tidylearn classification model object
#' @param new_data Optional data frame for evaluation
#'   (if NULL, uses training data)
#' @param bins Number of bins for grouping predictions
#'   (default: 10)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @importFrom ggplot2 ggplot aes geom_line geom_point
#' @importFrom ggplot2 geom_abline labs theme_minimal
#' @examples
#' \donttest{
#' iris_bin <- iris[iris$Species != "setosa", ]
#' iris_bin$Species <- factor(iris_bin$Species)
#' model <- tl_model(iris_bin, Species ~ ., method = "logistic")
#' tl_plot_gain(model)
#' }
#' @export
tl_plot_gain <- function(model, new_data = NULL, bins = 10, ...) {
  if (!model$spec$is_classification) {
    stop(
      "Gain chart is only available for classification models",
      call. = FALSE
    )
  }

  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Binary is a property of the model, as in tl_plot_lift()
  model_levels <- tl_model_classes(model)

  # For binary classification
  if (length(model_levels) == 2) {
    ordered_data <- tl_ranked_response(model, new_data, model_levels)

    # Calculate cumulative metrics
    bin <- tl_bin_index(nrow(ordered_data), bins)
    total_responders <- sum(ordered_data$actual)

    # Calculate gain by decile
    gain_data <- tibble::tibble(
      decile = integer(),
      cumulative_pct_population = numeric(),
      cumulative_pct_responders = numeric()
    )

    cumulative_responders <- 0

    for (i in unique(bin)) {
      rows <- which(bin == i)

      # Update cumulative counts
      current_responders <- sum(ordered_data$actual[rows])
      cumulative_responders <- cumulative_responders + current_responders

      # Calculate metrics
      cumulative_pct_population <- max(rows) / nrow(ordered_data) * 100
      cumulative_pct_responders <-
        cumulative_responders / total_responders * 100

      # Add to results
      gain_data <- gain_data |>
        dplyr::add_row(
          decile = i,
          cumulative_pct_population = cumulative_pct_population,
          cumulative_pct_responders = cumulative_pct_responders
        )
    }

    # Add origin point
    gain_data <- dplyr::bind_rows(
      tibble::tibble(
        decile = 0,
        cumulative_pct_population = 0,
        cumulative_pct_responders = 0
      ),
      gain_data
    )

    # Create the plot
    p <- ggplot2::ggplot(
      gain_data,
      ggplot2::aes(
        x = cumulative_pct_population,
        y = cumulative_pct_responders
      )
    ) +
      ggplot2::geom_line(color = "blue", linewidth = 1) +
      ggplot2::geom_point(color = "blue", size = 3) +
      ggplot2::geom_abline(
        intercept = 0,
        slope = 1,
        linetype = "dashed",
        color = "red"
      ) +
      ggplot2::labs(
        title = "Cumulative Gain Chart",
        subtitle = "Cumulative % of responders by % of population",
        x = "Cumulative % of Population",
        y = "Cumulative % of Responders"
      ) +
      ggplot2::coord_fixed() +
      ggplot2::scale_x_continuous(breaks = seq(0, 100, by = 10)) +
      ggplot2::scale_y_continuous(breaks = seq(0, 100, by = 10)) +
      ggplot2::theme_minimal()

    p
  } else {
    stop(
      "Gain chart is currently only implemented for binary classification",
      call. = FALSE
    )
  }
}
#' Plot Clusters in 2D Space
#'
#' Visualize clustering results using first two dimensions
#' or specified dimensions
#'
#' @param data A data frame with cluster assignments
#' @param cluster_col Name of cluster column (default: "cluster")
#' @param x_col X-axis variable (if NULL, uses the first numeric column
#'   other than \code{cluster_col})
#' @param y_col Y-axis variable (if NULL, uses the second numeric column
#'   other than \code{cluster_col})
#' @param centers Optional data frame of cluster centers
#' @param title Plot title
#' @param color_noise_black If TRUE, color noise points (cluster 0) black
#'
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' km <- tidy_kmeans(iris[, 1:4], k = 3)
#' clustered <- augment_kmeans(km, iris[, 1:4])
#' plot_clusters(clustered)
#' }
#' @export
plot_clusters <- function(data,
                          cluster_col = "cluster",
                          x_col = NULL,
                          y_col = NULL,
                          centers = NULL,
                          title = "Cluster Plot",
                          color_noise_black = TRUE) {

  # Find numeric columns if not specified. Integer cluster labels are
  # numeric too, and as the first numeric column they became the x axis.
  numeric_cols <- setdiff(
    names(data)[vapply(data, is.numeric, logical(1))], cluster_col
  )
  if (length(numeric_cols) == 0 && (is.null(x_col) || is.null(y_col))) {
    stop(
      "plot_clusters() needs a numeric column other than the cluster ",
      "column to plot, or 'x_col' and 'y_col'.",
      call. = FALSE
    )
  }

  if (is.null(x_col)) {
    x_col <- numeric_cols[1]
  }

  if (is.null(y_col)) {
    y_col <- if (length(numeric_cols) > 1) {
      numeric_cols[2]
    } else {
      numeric_cols[1]
    }
  }

  # Ensure cluster column is factor
  plot_data <- data |>
    dplyr::mutate(!!cluster_col := as.factor(!!rlang::sym(cluster_col)))

  # Create base plot
  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = !!rlang::sym(x_col),
      y = !!rlang::sym(y_col),
      color = !!rlang::sym(cluster_col)
    )
  ) +
    ggplot2::geom_point(size = 2.5, alpha = 0.7) +
    ggplot2::labs(
      title = title,
      x = x_col,
      y = y_col,
      color = "Cluster"
    ) +
    ggplot2::theme_minimal()

  # Color noise points black if requested
  if (color_noise_black && "0" %in% unique(plot_data[[cluster_col]])) {
    n_clusters <- length(unique(plot_data[[cluster_col]])) - 1
    other_clusters <- setdiff(unique(plot_data[[cluster_col]]), "0")
    cluster_colors <- setNames(grDevices::rainbow(n_clusters), other_clusters)
    p <- p + ggplot2::scale_color_manual(
      values = c("0" = "black", cluster_colors)
    )
  }

  # Add centers if provided
  if (!is.null(centers)) {
    p <- p + ggplot2::geom_point(
      data = centers,
      ggplot2::aes(x = !!rlang::sym(x_col), y = !!rlang::sym(y_col)),
      color = "black", size = 5, shape = 4, stroke = 2,
      inherit.aes = FALSE
    )
  }

  p
}


#' Create Elbow Plot for K-Means
#'
#' Plot total within-cluster sum of squares vs number of clusters
#'
#' @param wss_data A tibble with columns k and tot_withinss (from calc_wss)
#' @param add_line Add vertical line at suggested optimal k? (default: FALSE)
#' @param suggested_k If add_line=TRUE, which k to highlight
#'
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' wss <- data.frame(k = 2:6, tot_withinss = c(150, 90, 60, 50, 45))
#' plot_elbow(wss)
#' }
#' @export
plot_elbow <- function(wss_data, add_line = FALSE, suggested_k = NULL) {

  p <- ggplot2::ggplot(wss_data, ggplot2::aes(x = k, y = tot_withinss)) +
    ggplot2::geom_line(color = "steelblue", linewidth = 1) +
    ggplot2::geom_point(color = "steelblue", size = 3) +
    ggplot2::labs(
      title = "Elbow Method - Total Within-Cluster Sum of Squares",
      subtitle = "Look for 'elbow' in the curve",
      x = "Number of Clusters (k)",
      y = "Total Within-Cluster SS"
    ) +
    ggplot2::theme_minimal()

  if (add_line && !is.null(suggested_k)) {
    p <- p +
      ggplot2::geom_vline(
        xintercept = suggested_k,
        linetype = "dashed",
        color = "red"
      ) +
      ggplot2::annotate(
        "text",
        x = suggested_k,
        y = max(wss_data$tot_withinss) * 0.9,
        label = sprintf("k = %d", suggested_k),
        color = "red",
        hjust = -0.2
      )
  }

  p
}


#' Create Cluster Comparison Plot
#'
#' Compare multiple clustering results side-by-side
#'
#' @param data Data frame with multiple cluster columns
#' @param cluster_cols Vector of cluster column names
#' @param x_col X-axis variable
#' @param y_col Y-axis variable
#'
#' @return The return value of \code{\link[gridExtra]{grid.arrange}}, a
#'   \code{\link[gtable]{gtable}} drawn as a side effect.
#' @examplesIf requireNamespace("gridExtra", quietly = TRUE)
#' \donttest{
#' df <- iris[, 1:4]
#' df$km3 <- kmeans(df, 3)$cluster
#' df$km4 <- kmeans(df, 4)$cluster
#' plot_cluster_comparison(df, c("km3", "km4"), "Sepal.Length", "Sepal.Width")
#' }
#' @export
plot_cluster_comparison <- function(data, cluster_cols, x_col, y_col) {

  plots <- purrr::map(cluster_cols, function(col) {
    plot_clusters(data, cluster_col = col, x_col = x_col, y_col = y_col,
                  title = paste("Clusters:", col))
  })

  # Combine plots
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    stop(
      "Package 'gridExtra' is required to combine these panels. ",
      "Install it with: install.packages(\"gridExtra\")",
      call. = FALSE
    )
  }

  gridExtra::grid.arrange(
    grobs = plots,
    ncol = ceiling(sqrt(length(plots)))
  )
}


#' Plot Cluster Size Distribution
#'
#' Create bar plot of cluster sizes
#'
#' @param clusters Vector of cluster assignments
#' @param title Plot title (default: "Cluster Size Distribution")
#'
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' clusters <- kmeans(iris[, 1:4], 3)$cluster
#' plot_cluster_sizes(clusters)
#' }
#' @export
plot_cluster_sizes <- function(clusters, title = "Cluster Size Distribution") {

  cluster_counts <- tibble::tibble(cluster = as.factor(clusters)) |>
    dplyr::count(cluster)

  ggplot2::ggplot(
    cluster_counts,
    ggplot2::aes(x = cluster, y = n, fill = cluster)
  ) +
    ggplot2::geom_col() +
    ggplot2::geom_text(ggplot2::aes(label = n), vjust = -0.5) +
    ggplot2::labs(
      title = title,
      x = "Cluster",
      y = "Number of Observations"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none")
}


#' Plot Variance Explained (PCA)
#'
#' Create combined scree plot showing individual and cumulative variance
#'
#' @param variance_tbl Variance tibble from tidy_pca
#' @param threshold Horizontal line for variance threshold
#'   (default: 0.8 for 80%)
#'
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(iris[, 1:4], method = "pca")
#' plot_variance_explained(model$fit$variance_explained)
#' }
#' @export
plot_variance_explained <- function(variance_tbl, threshold = 0.8) {

  # Prepare data for plotting
  plot_data <- variance_tbl |>
    dplyr::mutate(pc_num = seq_len(dplyr::n()))

  # Create dual-axis plot
  subtitle_text <- sprintf(
    "Red line: cumulative variance | Green line: %.0f%% threshold",
    threshold * 100
  )
  p1 <- ggplot2::ggplot(plot_data, ggplot2::aes(x = pc_num)) +
    ggplot2::geom_col(
      ggplot2::aes(y = prop_variance),
      fill = "steelblue",
      alpha = 0.7
    ) +
    ggplot2::geom_line(
      ggplot2::aes(y = cum_variance),
      color = "red",
      linewidth = 1
    ) +
    ggplot2::geom_point(
      ggplot2::aes(y = cum_variance),
      color = "red",
      size = 2
    ) +
    ggplot2::geom_hline(
      yintercept = threshold,
      linetype = "dashed",
      color = "darkgreen"
    ) +
    ggplot2::labs(
      title = "Variance Explained by Principal Components",
      subtitle = subtitle_text,
      x = "Principal Component",
      y = "Proportion of Variance Explained"
    ) +
    ggplot2::scale_y_continuous(
      labels = function(x) paste0(round(x * 100, 1), "%")
    ) +
    ggplot2::theme_minimal()

  p1
}


#' Plot Dendrogram with Cluster Highlights
#'
#' Enhanced dendrogram with colored cluster rectangles
#'
#' @param hclust_obj Hierarchical clustering object: an \code{hclust}, a
#'   \code{tidy_hclust}, or a tidylearn model fitted with
#'   \code{method = "hclust"}
#' @param k Number of clusters to highlight
#' @param title Plot title
#'
#' @return Invisibly returns the \code{\link[stats]{hclust}} object. The
#'   dendrogram is drawn as a side effect.
#' @examples
#' \donttest{
#' hc <- hclust(dist(iris[, 1:4]))
#' plot_dendrogram(hc, k = 3)
#' }
#' @export
plot_dendrogram <- function(hclust_obj,
                            k = NULL,
                            title = "Hierarchical Clustering Dendrogram") {

  # A tidylearn model reached plot() whole, which dispatched to
  # plot.tidylearn_model() and failed on the main and xlab arguments
  if (inherits(hclust_obj, "tidylearn_model")) {
    if (!identical(hclust_obj$spec$method, "hclust")) {
      stop(
        "plot_dendrogram() needs a model fitted with method = \"hclust\", ",
        "not \"", hclust_obj$spec$method, "\".",
        call. = FALSE
      )
    }
    hc <- hclust_obj$fit$model
  } else if (inherits(hclust_obj, "tidy_hclust")) {
    hc <- hclust_obj$model
  } else {
    hc <- hclust_obj
  }

  plot(hc, main = title, xlab = "", ylab = "Height", sub = "", cex = 0.7)

  if (!is.null(k)) {
    stats::rect.hclust(hc, k = k, border = 2:(k + 1))
  }

  invisible(hc)
}


#' Create Summary Dashboard
#'
#' Generate a multi-panel summary of clustering results
#'
#' @param data Data frame with cluster assignments
#' @param cluster_col Cluster column name
#' @param validation_metrics Optional tibble of validation metrics
#'
#' @return Invisibly returns a named list of the
#'   \code{\link[ggplot2]{ggplot}} objects drawn: \code{clusters}, the
#'   scatter plot, when the data has two numeric columns besides
#'   \code{cluster_col}; \code{sizes}; and \code{metrics}, when
#'   \code{validation_metrics} is given. The combined plot grid is drawn as
#'   a side effect via \code{\link[gridExtra]{grid.arrange}}.
#' @examplesIf requireNamespace("gridExtra", quietly = TRUE)
#' \donttest{
#' df <- iris[, 1:4]
#' df$cluster <- kmeans(df, 3)$cluster
#' create_cluster_dashboard(df)
#' }
#' @export
create_cluster_dashboard <- function(data,
                                     cluster_col = "cluster",
                                     validation_metrics = NULL) {

  plots <- list()

  # 1. Cluster scatter plot (first two numeric columns besides the
  # clusters). Skipped, it used to leave a NULL in the list that
  # grid.arrange() could not draw.
  numeric_cols <- setdiff(
    names(data)[vapply(data, is.numeric, logical(1))], cluster_col
  )
  if (length(numeric_cols) >= 2) {
    plots$clusters <- plot_clusters(
      data, cluster_col = cluster_col,
      x_col = numeric_cols[1],
      y_col = numeric_cols[2]
    )
  }

  # 2. Cluster sizes
  plots$sizes <- plot_cluster_sizes(data[[cluster_col]])

  # 3. If validation metrics provided, create metrics plot
  if (!is.null(validation_metrics)) {
    # calc_validation_metrics() has no silhouette without a distance
    # matrix, and reading the absent column warned and printed NA
    silhouette <- if ("avg_silhouette" %in% names(validation_metrics)) {
      sprintf("Avg Silhouette: %.3f", validation_metrics$avg_silhouette)
    }
    metrics_text <- paste(
      c(
        "Validation Metrics",
        "",
        sprintf("Number of Clusters: %d", validation_metrics$k),
        silhouette,
        sprintf("Min Size: %d", validation_metrics$min_size),
        sprintf("Max Size: %d", validation_metrics$max_size)
      ),
      collapse = "\n"
    )

    plots$metrics <- ggplot2::ggplot() +
      ggplot2::annotate(
        "text", x = 0.5, y = 0.5,
        label = metrics_text, size = 5
      ) +
      ggplot2::theme_void()
  }

  # Combine plots
  if (length(plots) > 0) {
    if (!requireNamespace("gridExtra", quietly = TRUE)) {
      stop(
        "Package 'gridExtra' is required to combine these panels. ",
        "Install it with: install.packages(\"gridExtra\")",
        call. = FALSE
      )
    }
    gridExtra::grid.arrange(grobs = plots, ncol = 2)
  }

  invisible(plots)
}


#' Create Distance Heatmap
#'
#' Visualize distance matrix as heatmap
#'
#' @param dist_mat Distance matrix (dist object)
#' @param cluster_order Optional vector to reorder observations by cluster
#' @param title Plot title
#'
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' d <- dist(iris[1:20, 1:4])
#' plot_distance_heatmap(d)
#' }
#' @export
plot_distance_heatmap <- function(dist_mat,
                                  cluster_order = NULL,
                                  title = "Distance Heatmap") {

  # Convert to matrix
  dist_matrix <- as.matrix(dist_mat)

  # Reorder if cluster order provided
  if (!is.null(cluster_order)) {
    order_idx <- order(cluster_order)
    dist_matrix <- dist_matrix[order_idx, order_idx]
  }

  # Convert to long format
  dist_long <- dist_matrix |>
    tibble::as_tibble(rownames = "id1") |>
    tidyr::pivot_longer(-id1, names_to = "id2", values_to = "distance")

  # Pin the axis order to the matrix. Character IDs on a discrete scale
  # otherwise sort alphabetically -- "1", "10", "11", ..., "2" -- which
  # moves the diagonal off the diagonal and undoes any cluster_order
  # reordering applied above, fabricating the block structure the plot
  # exists to show.
  axis_levels <- rownames(dist_matrix)
  dist_long$id1 <- factor(dist_long$id1, levels = axis_levels)
  dist_long$id2 <- factor(dist_long$id2, levels = rev(axis_levels))

  # Create heatmap
  ggplot2::ggplot(dist_long, ggplot2::aes(x = id1, y = id2, fill = distance)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient(low = "white", high = "red") +
    ggplot2::labs(title = title, x = "", y = "", fill = "Distance") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 90, hjust = 1, size = 6),
      axis.text.y = ggplot2::element_text(size = 6)
    )
}
