#' @title Tree-based Methods for tidylearn
#' @name tidylearn-trees
#' @description Decision trees, random forests, and boosting functionality
#' @importFrom rpart rpart rpart.control
#' @importFrom stats predict
#' @importFrom randomForest randomForest importance
#' @importFrom gbm gbm predict.gbm
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr mutate arrange desc
NULL

#' Let the caller's arguments replace a wrapper's defaults
#'
#' A wrapper that sets a value itself -- \code{probability} for svm,
#' \code{maxit} in the nn tuner -- and forwards \code{...} to the same call
#' failed with "formal argument matched by multiple actual arguments"
#' whenever the caller named that argument too. Merging first lets the
#' caller's value win, which is what passing it means.
#'
#' @param defaults Named list of the wrapper's own values
#' @param overrides List of the caller's arguments, usually \code{list(...)}
#' @return \code{defaults}, with each named element of \code{overrides}
#'   replacing the one of the same name and any unnamed ones appended
#' @keywords internal
#' @noRd
tl_override_args <- function(defaults, overrides) {
  named <- names2(overrides) != ""
  defaults[names(overrides)[named]] <- overrides[named]
  c(defaults, overrides[!named])
}

#' Fit a decision tree model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param cp Complexity parameter (default: 0.01)
#' @param minsplit Minimum number of observations in a node for a split
#' @param maxdepth Maximum depth of the tree
#' @param ... Additional arguments to pass to rpart()
#' @return A fitted decision tree model
#' @keywords internal
tl_fit_tree <- function(data, formula, is_classification = FALSE,
                        cp = 0.01, minsplit = 20, maxdepth = 30, ...) {
  # Check if rpart is installed
  tl_check_packages("rpart")

  # Determine method based on problem type
  method <- if (is_classification) "class" else "anova"

  # Everything in ... used to go to rpart.control(), which takes its own
  # ... and discards what it does not recognise -- so weights, cost and
  # parms were accepted and silently had no effect. Send rpart()'s own
  # arguments to rpart().
  dots <- list(...)
  dot_names <- names2(dots)
  rpart_own <- dot_names %in%
    c("weights", "subset", "na.action", "model", "x", "y", "parms", "cost")

  # A whole control list passed as control = starts from the caller's
  # settings; the named arguments here and in ... still apply on top.
  user_control <- dot_names == "control"

  # rpart.control() takes the rest, and only its own arguments: it
  # discards a name it does not recognise, so a misspelt maxdeth = 1
  # fitted the default tree and an offset fitted with none, where rpart()
  # itself stops on both.
  control_arg <- dot_names %in%
    setdiff(names(formals(rpart::rpart.control)), "...")
  unknown <- !(rpart_own | user_control | control_arg)
  if (any(unknown)) {
    bad <- dot_names[unknown]
    bad[bad == ""] <- "<unnamed>"
    stop(
      "Method \"tree\" has no argument",
      if (length(bad) > 1L) "s" else "", " ",
      paste0("'", bad, "'", collapse = ", "),
      ": neither rpart() nor rpart.control() takes ",
      if (length(bad) > 1L) "them" else "it",
      ", so it would have been ignored. See ?rpart::rpart.control for the ",
      "settings a tree takes.",
      call. = FALSE
    )
  }

  control <- do.call(
    rpart::rpart.control,
    c(list(cp = cp, minsplit = minsplit, maxdepth = maxdepth),
      dots[control_arg])
  )
  if (any(user_control)) {
    explicit <- c(names(match.call(expand.dots = FALSE))[-1],
                  dot_names[control_arg])
    given <- dots[[which(user_control)[1]]]

    # rpart.control() derives minbucket from minsplit, so a list built as
    # rpart.control(cp = 0.001) carries minbucket = 7 from its default
    # minsplit = 20. Copied over an explicit minsplit = 5, it left a node
    # needing 14 rows to split, and minsplit did nothing. An explicit
    # minsplit therefore brings its own derived minbucket, unless minbucket
    # is explicit too. The list cannot say which of its two it was given:
    # rpart.control(minbucket = 4) and rpart.control(minsplit = 12) build
    # the same list.
    if ("minsplit" %in% explicit) {
      explicit <- c(explicit, "minbucket")
    }

    for (setting in setdiff(names(given), explicit)) {
      control[[setting]] <- given[[setting]]
    }
  }

  tl_fit_by_value(
    rpart::rpart, "rpart",
    c(list(formula = formula, data = data, method = method,
           control = control),
      dots[rpart_own & !user_control])
  )
}

#' Predict using a decision tree model
#'
#' @param model A tidylearn tree model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default),
#'   "prob" or "class" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_tree <- function(model, new_data, type = "response", ...) {
  # Get the tree model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  if (is_classification) {
    if (type == "prob") {
      # Get class probabilities
      probs <- predict(fit, newdata = new_data, type = "prob")

      # Convert to tibble with appropriate column names
      class_levels <- colnames(probs)
      prob_df <- as.data.frame(probs)
      names(prob_df) <- class_levels

      tibble::as_tibble(prob_df)
    } else if (type == "class") {
      # Get predicted classes
      preds <- predict(fit, newdata = new_data, type = "class")
      preds
    } else if (type == "response") {
      # Get predicted classes (same as "class" for classification)
      preds <- predict(fit, newdata = new_data, type = "class")
      preds
    } else {
      stop(
        "Invalid prediction type for classification trees. ",
        "Use 'prob', 'class', or 'response'.",
        call. = FALSE
      )
    }
  } else {
    # Regression predictions
    preds <- predict(fit, newdata = new_data)
    preds
  }
}

#' Fit a random forest model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param ntree Number of trees to grow (default: 500)
#' @param mtry Number of variables randomly sampled at each split. Left
#'   to \code{randomForest::randomForest()} when \code{NULL}, which uses
#'   \code{floor(sqrt(p))} for classification and
#'   \code{max(floor(p / 3), 1)} for regression over the \code{p}
#'   columns of the design matrix.
#' @param importance Whether to compute variable importance (default: TRUE)
#' @param ... Additional arguments to pass to randomForest()
#' @return A fitted random forest model
#' @keywords internal
tl_fit_forest <- function(data, formula, is_classification = FALSE,
                          ntree = 500, mtry = NULL, importance = TRUE, ...) {
  # Check if randomForest is installed
  tl_check_packages("randomForest")

  # randomForest's classification path hangs rather than erroring when no
  # predictor can produce a split, and the loop is uninterruptible
  if (is_classification) {
    tl_check_predictor_variance(data, formula, "forest")
  }

  # No default mtry. randomForest's own is floor(sqrt(p)) for
  # classification and max(floor(p / 3), 1) for regression, over the p
  # columns of the design matrix. Recomputing those here from
  # ncol(data) - 1 counted every column in the frame rather than the
  # formula's predictors, so `mpg ~ wt + hp` on mtcars asked for 3 of 2
  # and randomForest warned that it had reset the value -- and
  # `Species ~ Sepal.Length + Sepal.Width` on iris asked for 2 of 2,
  # which it accepted in silence, sampling every predictor at every
  # split. That is bagging, not a random forest.
  args <- list(
    formula = formula,
    data = data,
    ntree = ntree,
    importance = importance,
    ...
  )
  if (!is.null(mtry)) {
    args$mtry <- mtry
  }

  # randomForest's formula interface re-evaluates each term on a copy of
  # its model frame whose columns data.frame() has renamed -- log(hp)
  # becomes log.hp. -- so a transformed term such as log(hp) or
  # factor(cyl) failed with "object 'hp' not found", called directly as
  # much as through tidylearn. Those formulas are fitted from a predictor
  # frame built here. The rest keep the formula interface.
  if (tl_forest_has_transformed_term(formula, data)) {
    return(tl_fit_forest_frame(formula, data, args))
  }

  tl_restore_call_data(do.call(randomForest::randomForest, args))
}

#' Whether a formula has a term randomForest's formula interface cannot fit
#'
#' @param formula The model formula
#' @param data The training data, to expand a dot against
#' @return TRUE when a term, or a part of an interaction, is computed from
#'   its columns rather than naming one
#' @keywords internal
#' @noRd
tl_forest_has_transformed_term <- function(formula, data) {
  labels <- attr(stats::terms(formula, data = data), "term.labels")
  parts_of <- function(expr) {
    if (is.call(expr) && identical(expr[[1L]], as.name(":"))) {
      c(parts_of(expr[[2L]]), parts_of(expr[[3L]]))
    } else {
      list(expr)
    }
  }
  parts <- do.call(c, lapply(lapply(labels, str2lang), parts_of))
  !all(vapply(parts, is.name, logical(1)))
}

#' Fit a random forest from a predictor frame built from the formula
#'
#' The predictors are the variables of the formula's terms, evaluated by
#' \code{model.frame()}: a factor stays a factor, so randomForest splits
#' it as a category, as its formula interface does. The terms are kept on
#' the fit so that \code{tl_predict_forest()} can build the same frame
#' from new data, which needs only the raw columns.
#'
#' @param formula The model formula
#' @param data The training data
#' @param args The arguments for \code{randomForest()}, as
#'   \code{tl_fit_forest()} assembled them
#' @return A \code{randomForest} object carrying the predictor terms and
#'   factor levels
#' @keywords internal
#' @noRd
tl_fit_forest_frame <- function(formula, data, args) {
  dots <- args[!names2(args) %in% c("formula", "data")]
  na_action <- if (is.null(dots$na.action)) stats::na.fail else dots$na.action
  dots$na.action <- NULL

  frame <- stats::model.frame(formula, data = data, na.action = na_action)
  model_terms <- attr(frame, "terms")

  # Interactions are read as their parts, as the formula interface reads
  # them. The formula's environment stays behind the terms, so a function
  # the caller defined is still found at predict().
  predictor_formula <- stats::reformulate(attr(model_terms, "term.labels"))
  environment(predictor_formula) <- environment(formula)
  predictor_terms <- stats::terms(predictor_formula)

  frame_vars <- as.list(attr(model_terms, "variables"))[-1]
  columns <- vapply(
    as.list(attr(predictor_terms, "variables"))[-1],
    function(v) which(vapply(frame_vars, identical, logical(1), v))[1],
    integer(1)
  )
  x <- frame[, columns, drop = FALSE]
  y <- stats::model.response(frame)

  # The stored call names its values rather than spelling out the frame,
  # as print() would
  fit <- do.call(randomForest::randomForest, c(list(x = x, y = y), dots))
  for (arg in intersect(names(fit$call), c("x", "y", "weights"))) {
    fit$call[[arg]] <- as.name(arg)
  }
  fit$terms <- model_terms
  fit$tl_predictor_terms <- predictor_terms
  fit$tl_xlevels <- stats::.getXlevels(predictor_terms, x)
  if (!is.null(attr(frame, "na.action"))) {
    fit$na.action <- attr(frame, "na.action")
  }
  fit
}

#' Predict from a forest fitted by tl_fit_forest_frame()
#'
#' predict.randomForest() refuses a missing value in data it was not given
#' a formula for, so incomplete rows are left out and put back as NA.
#'
#' @param fit The randomForest object
#' @param new_data Data holding the formula's raw columns
#' @param type \code{"response"} or \code{"prob"}
#' @return Predictions, one per row of \code{new_data}
#' @keywords internal
#' @noRd
tl_forest_frame_predict <- function(fit, new_data, type) {
  x <- stats::model.frame(
    fit$tl_predictor_terms, new_data,
    na.action = stats::na.pass, xlev = fit$tl_xlevels
  )
  keep <- stats::complete.cases(x)
  rows <- x[keep, , drop = FALSE]

  if (type == "prob") {
    probs <- if (any(keep)) {
      predict(fit, newdata = rows, type = "prob")
    } else {
      matrix(numeric(0), nrow = 0, ncol = length(fit$classes),
             dimnames = list(NULL, fit$classes))
    }
    return(tl_realign_prob_matrix(probs, keep))
  }

  preds <- if (any(keep)) {
    predict(fit, newdata = rows, type = "response")
  } else if (is.null(fit$classes)) {
    numeric(0)
  } else {
    factor(character(0), levels = fit$classes)
  }
  tl_realign_predictions(preds, keep)
}

#' Predict using a random forest model
#'
#' @param model A tidylearn forest model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response"
#'   (default), "prob" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_forest <- function(model, new_data, type = "response", ...) {
  # Get the random forest model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  # A forest fitted from a predictor frame builds the same frame from
  # new_data; the rest go through randomForest's formula interface
  forest_predict <- function(rf_type) {
    if (is.null(fit$tl_predictor_terms)) {
      predict(fit, newdata = new_data, type = rf_type)
    } else {
      tl_forest_frame_predict(fit, new_data, rf_type)
    }
  }

  if (is_classification) {
    if (type == "prob") {
      # Get class probabilities
      probs <- forest_predict("prob")

      # Convert to tibble with appropriate column names
      class_levels <- colnames(probs)
      prob_df <- as.data.frame(probs)
      names(prob_df) <- class_levels

      tibble::as_tibble(prob_df)
    } else if (type == "class" || type == "response") {
      # Get predicted classes
      preds <- forest_predict("response")
      preds
    } else {
      stop(
        "Invalid prediction type for random forests. ",
        "Use 'prob', 'class', or 'response'.",
        call. = FALSE
      )
    }
  } else {
    # Regression predictions
    preds <- forest_predict("response")
    preds
  }
}

#' The classes a boost model was fitted on
#'
#' @param model A tidylearn boost model
#' @return The recorded response levels or, for a model without them, the
#'   classes of the response its formula computes on the training data
#' @keywords internal
#' @noRd
tl_boost_class_levels <- function(model) {
  model$spec$response_levels %||%
    levels(tl_normalise_response(
      tl_formula_lhs(model$spec$formula, model$data)
    ))
}

#' Normalise multinomial gbm probabilities to an n x k matrix
#'
#' \code{predict.gbm} returns a 3-D array \code{[n, nclass, 1]} for the
#' multinomial distribution, so \code{is.matrix()} is FALSE and naive
#' handling collapses every row into one prediction. Drop the trailing
#' dimension and restore the class names.
#'
#' @param probs The raw return value of \code{gbm::predict.gbm}
#' @param model The tidylearn model, used for the class levels
#' @return A numeric matrix with one row per observation and one named
#'   column per class
#' @keywords internal
#' @noRd
tl_gbm_multinomial_matrix <- function(probs, model) {
  class_levels <- tl_boost_class_levels(model)

  dims <- dim(probs)

  out <- if (length(dims) == 3L) {
    matrix(probs, nrow = dims[1], ncol = dims[2])
  } else if (length(dims) == 2L) {
    probs
  } else {
    # A bare vector is one observation's class probabilities
    matrix(probs, nrow = 1)
  }

  if (ncol(out) != length(class_levels)) {
    stop(
      "gbm returned ", ncol(out), " probability columns for ",
      length(class_levels), " classes.",
      call. = FALSE
    )
  }

  colnames(out) <- class_levels
  out
}

#' Fit a gradient boosting model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param n.trees Number of trees (default: 100)
#' @param interaction.depth Depth of interactions
#'   (default: 3)
#' @param shrinkage Learning rate (default: 0.1)
#' @param n.minobsinnode Minimum number of observations
#'   in terminal nodes (default: 10)
#' @param cv.folds Number of cross-validation folds
#'   (default: 0, no CV)
#' @param ... Additional arguments to pass to gbm(), including case
#'   \code{weights}. \code{verbose}, and for regression
#'   \code{distribution}, replace the defaults used here.
#' @return A fitted gradient boosting model
#' @details The distribution follows the response: \code{"gaussian"} for
#'   regression, \code{"bernoulli"} for two classes and
#'   \code{"multinomial"} for more. gbm describes its multinomial
#'   distribution as currently broken, kept only for backwards
#'   compatibility, and warns to that effect on every multiclass fit. For
#'   three or more classes, \code{method = "forest"} or
#'   \code{method = "xgboost"} are the better supported choices.
#' @keywords internal
tl_fit_boost <- function(
    data, formula,
    is_classification = FALSE,
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    n.minobsinnode = 10,
    cv.folds = 0, ...) {
  # Check if gbm is installed
  tl_check_packages("gbm")
  dots <- list(...)

  # Determine distribution based on problem type
  if (is_classification) {
    # The response the formula computes, which for factor(am) ~ . is not
    # the column am
    y <- tl_normalise_response(tl_formula_lhs(formula, data))
    class_levels <- levels(y)

    # Check if binary or multiclass
    if (length(class_levels) == 2) {
      # Binary classification. gbm's bernoulli requires a numeric 0/1
      # response, so encode the second level as the positive class --
      # the same orientation tl_predict_boost assumes
      distribution <- "bernoulli"
      recoded <- as.integer(y == class_levels[2])
    } else {
      # Multiclass classification
      distribution <- "multinomial"
      recoded <- y
    }

    # gbm reads the response through the formula. A bare column is
    # replaced by its recoding. A computed one such as factor(am) cannot
    # be: gbm evaluated it afresh, got a factor and refused it for
    # bernoulli. Its recoding gets a column of its own, fitted against the
    # formula with its dot expanded first, so the predictors stay the ones
    # the formula names.
    if (is.name(formula[[2L]])) {
      data[[as.character(formula[[2L]])]] <- recoded
    } else {
      formula <- stats::formula(stats::terms(formula, data = data))
      data[[".tl_response"]] <- recoded
      formula[[2L]] <- as.name(".tl_response")
    }

    # tl_predict_boost() reads these two distributions only, so for
    # classification the choice is not the caller's to make
    if (!is.null(dots$distribution) &&
          !identical(dots$distribution, distribution)) {
      stop(
        "For classification, method \"boost\" sets gbm's distribution from ",
        "the response: \"bernoulli\" for two classes, \"multinomial\" for ",
        "more. Remove distribution = ",
        tl_describe_value(dots$distribution), ".",
        call. = FALSE
      )
    }
  } else {
    # Regression
    distribution <- "gaussian"
  }

  args <- list(
    formula = formula,
    data = data,
    distribution = distribution,
    n.trees = n.trees,
    interaction.depth = interaction.depth,
    shrinkage = shrinkage,
    n.minobsinnode = n.minobsinnode,
    cv.folds = cv.folds,
    verbose = FALSE
  )

  # gbm() evaluates weights inside its own model frame, which a value
  # forwarded through ... cannot reach: "..1 used in an incorrect
  # context". So the call is built from values. The caller's verbose and
  # regression distribution replace the defaults above rather than
  # colliding with them as duplicate arguments.
  tl_fit_by_value(gbm::gbm, "gbm", tl_override_args(args, dots))
}

#' Predict using a gradient boosting model
#'
#' @param model A tidylearn boost model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response"
#'   (default), "prob" (for classification)
#' @param n.trees Number of trees to use for prediction
#'   (if NULL, uses optimal number)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_boost <- function(
    model, new_data,
    type = "response",
    n.trees = NULL, ...) {
  # Get the boosting model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  # Determine the number of trees to use
  if (is.null(n.trees)) {
    if (fit$cv.folds > 0) {
      # Use the optimal number of trees from CV
      n.trees <- gbm::gbm.perf(
        fit, method = "cv", plot.it = FALSE
      )
    } else {
      # Use all trees
      n.trees <- fit$n.trees
    }
  }

  if (is_classification) {
    # Check distribution
    if (fit$distribution$name == "bernoulli") {
      # Binary classification
      if (type == "prob") {
        # Get probabilities on the scale of the response
        probs <- gbm::predict.gbm(
          fit, newdata = new_data,
          n.trees = n.trees,
          type = "response", ...
        )

        # The model's classes. Read off the training column, a computed
        # response such as factor(am) had the column's values instead.
        class_levels <- tl_boost_class_levels(model)

        # Create a data frame with probabilities
        prob_df <- tibble::tibble(
          !!class_levels[1] := 1 - probs,
          !!class_levels[2] := probs
        )

        prob_df
      } else if (type == "class" || type == "response") {
        # Get probabilities
        probs <- gbm::predict.gbm(
          fit, newdata = new_data,
          n.trees = n.trees,
          type = "response", ...
        )

        class_levels <- tl_boost_class_levels(model)

        # Convert to classes
        pred_classes <- ifelse(
          probs > 0.5,
          class_levels[2],
          class_levels[1]
        )
        pred_classes <- factor(
          pred_classes, levels = class_levels
        )

        pred_classes
      } else {
        stop(
          "Invalid prediction type for boosting. ",
          "Use 'prob', 'class', or 'response'.",
          call. = FALSE
        )
      }
    } else if (fit$distribution$name == "multinomial") {
      # Multiclass classification
      if (type == "prob" || type == "class" || type == "response") {
        # Get class probabilities
        probs <- gbm::predict.gbm(
          fit, newdata = new_data,
          n.trees = n.trees,
          type = "response", ...
        )

        probs <- tl_gbm_multinomial_matrix(probs, model)

        if (type == "prob") {
          prob_df <- as.data.frame(probs)
          names(prob_df) <- colnames(probs)
          tibble::as_tibble(prob_df)
        } else {
          # Find class with highest probability, one row at a time
          class_levels <- colnames(probs)
          class_idx <- max.col(probs, ties.method = "first")
          factor(class_levels[class_idx], levels = class_levels)
        }
      } else {
        stop(
          "Invalid prediction type for boosting. ",
          "Use 'prob', 'class', or 'response'.",
          call. = FALSE
        )
      }
    }
  } else {
    # Regression predictions
    preds <- gbm::predict.gbm(
      fit, newdata = new_data,
      n.trees = n.trees,
      type = "response", ...
    )
    preds
  }
}

#' Plot variable importance for tree-based models
#'
#' @param model A tidylearn tree-based model object
#' @param top_n Number of top features to display (default: 20)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_col coord_flip labs theme_minimal
#' @keywords internal
tl_plot_importance <- function(model, top_n = 20, ...) {
  # The importance tl_table_importance() reports, scaled to a maximum of
  # 100. This function kept its own copy of the extraction, which still
  # failed on a forest fitted with importance = FALSE and had no xgboost
  # branch.
  if (!model$spec$method %in% c("tree", "forest", "boost", "xgboost")) {
    stop(
      "Variable importance plot not implemented for method: ",
      model$spec$method,
      call. = FALSE
    )
  }
  importance_df <- tl_extract_importance(model)

  # A tree with no splits has no importance to draw, and an empty chart
  # said nothing about why
  if (nrow(importance_df) == 0) {
    stop("No feature has non-zero importance: ",
         tl_no_importance_reason(model), ".", call. = FALSE)
  }

  # Filter and sort
  importance_df <- importance_df |>
    dplyr::arrange(dplyr::desc(.data$importance)) |>
    dplyr::slice_head(n = top_n)

  # Create the plot
  p <- ggplot2::ggplot(
    importance_df,
    ggplot2::aes(
      x = stats::reorder(feature, importance),
      y = importance
    )
  ) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Variable Importance",
      x = NULL,
      y = "Importance"
    ) +
    ggplot2::theme_minimal()

  p
}

#' Plot a decision tree
#'
#' @param model A tidylearn tree model object
#' @param ... Additional arguments to pass to rpart.plot()
#' @return The return value of \code{\link[rpart.plot]{rpart.plot}}, called
#'   for its side effect of drawing the tree.
#' @examplesIf requireNamespace("rpart.plot", quietly = TRUE)
#' \donttest{
#' model <- tl_model(iris, Species ~ ., method = "tree")
#' tl_plot_tree(model)
#' }
#' @export
tl_plot_tree <- function(model, ...) {
  # Check if rpart.plot is installed
  tl_check_packages("rpart.plot")

  if (model$spec$method != "tree") {
    stop("Tree plot is only available for decision tree models", call. = FALSE)
  }

  # Plot the tree
  rpart.plot::rpart.plot(model$fit, ...)
}

#' Plot partial dependence for tree-based models
#'
#' @param model A tidylearn tree-based model object
#' @param var Variable name to plot
#' @param n.pts Number of points for continuous
#'   variables (default: 20)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object. Its data has a
#'   \code{var_value} column and the mean prediction over the model's
#'   training rows, \code{y}, at each value. For classification, \code{y}
#'   is a mean class probability and a \code{class} column says which
#'   class: the positive class (the second level) alone for a two-class
#'   model, and every class, one line each, for more.
#' @importFrom ggplot2 ggplot aes geom_line geom_point
#' @importFrom ggplot2 labs theme_minimal
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "forest")
#' tl_plot_partial_dependence(model, var = "wt")
#' }
#' @export
tl_plot_partial_dependence <- function(model, var, n.pts = 20, ...) {
  if (!model$spec$method %in% c("tree", "forest", "boost")) {
    stop(
      "Partial dependence plots are currently ",
      "only implemented for tree-based models",
      call. = FALSE
    )
  }

  # Get the data
  data <- model$data

  # Check if variable exists
  if (!var %in% names(data)) {
    stop(
      "Variable '", var,
      "' not found in the model data",
      call. = FALSE
    )
  }

  # Get variable values
  var_values <- data[[var]]
  categorical <- is.factor(var_values) || is.character(var_values)

  # Create grid of values for the variable
  if (categorical) {
    # For categorical variables, use unique values
    grid_values <- unique(var_values)
  } else {
    # For continuous variables, create a sequence
    grid_values <- seq(
      min(var_values, na.rm = TRUE),
      max(var_values, na.rm = TRUE),
      length.out = n.pts
    )
  }

  # Two classes are drawn as the positive class, the second level. With
  # more, every class gets a curve: the mean probability of the second
  # class alone, unlabelled, said nothing about the others -- and the
  # class column held whichever class had the highest mean instead of the
  # one drawn.
  is_classification <- model$spec$is_classification
  if (is_classification) {
    class_levels <- model$spec$response_levels %||%
      levels(tl_normalise_response(data[[model$spec$response_var]]))
    shown <- if (length(class_levels) == 2L) class_levels[2] else class_levels
  }

  pred_data <- purrr::map_dfr(seq_along(grid_values), function(i) {
    new_data <- data
    new_data[[var]] <- grid_values[i]

    if (is_classification) {
      probs <- predict(model, new_data, type = "prob")
      tibble::tibble(
        var_value = grid_values[i],
        class = factor(shown, levels = class_levels),
        y = vapply(shown, function(cl) mean(probs[[cl]], na.rm = TRUE),
                   numeric(1), USE.NAMES = FALSE)
      )
    } else {
      # predict() returns a tibble, and mean() of the tibble itself is NA
      # with a warning -- which every regression curve used to be
      preds <- predict(model, new_data, type = "response")
      tibble::tibble(
        var_value = grid_values[i],
        y = mean(preds$.pred, na.rm = TRUE)
      )
    }
  })

  multiclass <- is_classification && length(shown) > 1L
  y_lab <- if (!is_classification) {
    "Mean Prediction"
  } else if (multiclass) {
    "Mean Probability"
  } else {
    paste0("Mean Probability of ", shown)
  }
  plot_labs <- ggplot2::labs(
    title = paste0("Partial Dependence Plot for ", var),
    x = var,
    y = y_lab
  )
  # Only a multiclass plot maps the class; a label for an unmapped
  # aesthetic draws a message every time the plot is printed
  class_labs <- if (multiclass) {
    ggplot2::labs(colour = "Class", fill = "Class")
  }

  if (categorical) {
    # Bar plot for categorical variables
    p <- if (multiclass) {
      ggplot2::ggplot(
        pred_data,
        ggplot2::aes(x = .data$var_value, y = .data$y, fill = .data$class)
      ) +
        ggplot2::geom_col(position = "dodge")
    } else {
      ggplot2::ggplot(
        pred_data,
        ggplot2::aes(x = .data$var_value, y = .data$y)
      ) +
        ggplot2::geom_col(fill = "steelblue")
    }
    p <- p +
      plot_labs +
      class_labs +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(
          angle = 45, hjust = 1
        )
      )
  } else {
    # Line plot for continuous variables
    p <- if (multiclass) {
      ggplot2::ggplot(
        pred_data,
        ggplot2::aes(x = .data$var_value, y = .data$y,
                     colour = .data$class, group = .data$class)
      ) +
        ggplot2::geom_line() +
        ggplot2::geom_point()
    } else {
      ggplot2::ggplot(
        pred_data,
        ggplot2::aes(x = .data$var_value, y = .data$y)
      ) +
        ggplot2::geom_line(color = "steelblue") +
        ggplot2::geom_point(color = "steelblue")
    }
    p <- p +
      plot_labs +
      class_labs +
      ggplot2::theme_minimal()
  }

  p
}
