#' @title Tree-based Methods for tidysl
#' @name tidysl-trees
#' @description Decision trees, random forests, and boosting functionality
#' @importFrom rpart rpart rpart.control
#' @importFrom stats predict
#' @importFrom randomForest randomForest importance
#' @importFrom gbm gbm predict.gbm
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr %>% mutate arrange desc
NULL

#' Fit a decision tree model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
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

  # Fit the tree model
  tree_model <- rpart::rpart(
    formula = formula,
    data = data,
    method = method,
    control = rpart::rpart.control(
      cp = cp,
      minsplit = minsplit,
      maxdepth = maxdepth,
      ...
    )
  )

  return(tree_model)
}

#' Predict using a decision tree model
#'
#' @param model A tidysl tree model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification), "class" (for classification)
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

      return(tibble::as_tibble(prob_df))
    } else if (type == "class") {
      # Get predicted classes
      preds <- predict(fit, newdata = new_data, type = "class")
      return(preds)
    } else if (type == "response") {
      # Get predicted classes (same as "class" for classification)
      preds <- predict(fit, newdata = new_data, type = "class")
      return(preds)
    } else {
      stop("Invalid prediction type for classification trees. Use 'prob', 'class', or 'response'.", call. = FALSE)
    }
  } else {
    # Regression predictions
    preds <- predict(fit, newdata = new_data)
    return(preds)
  }
}

#' Fit a random forest model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param ntree Number of trees to grow (default: 500)
#' @param mtry Number of variables randomly sampled at each split
#' @param importance Whether to compute variable importance (default: TRUE)
#' @param ... Additional arguments to pass to randomForest()
#' @return A fitted random forest model
#' @keywords internal
tl_fit_forest <- function(data, formula, is_classification = FALSE,
                          ntree = 500, mtry = NULL, importance = TRUE, ...) {
  # Check if randomForest is installed
  tl_check_packages("randomForest")

  # Parse the formula
  response_var <- all.vars(formula)[1]
  y <- data[[response_var]]

  # Set default mtry if not provided
  if (is.null(mtry)) {
    if (is_classification) {
      # For classification, default is sqrt(p)
      mtry <- floor(sqrt(ncol(data) - 1))
    } else {
      # For regression, default is p/3
      mtry <- max(floor((ncol(data) - 1) / 3), 1)
    }
  }

  # Fit the random forest model
  forest_model <- randomForest::randomForest(
    formula = formula,
    data = data,
    ntree = ntree,
    mtry = mtry,
    importance = importance,
    ...
  )

  return(forest_model)
}

#' Predict using a random forest model
#'
#' @param model A tidysl forest model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_forest <- function(model, new_data, type = "response", ...) {
  # Get the random forest model
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

      return(tibble::as_tibble(prob_df))
    } else if (type == "class" || type == "response") {
      # Get predicted classes
      preds <- predict(fit, newdata = new_data, type = "response")
      return(preds)
    } else {
      stop("Invalid prediction type for random forests. Use 'prob', 'class', or 'response'.", call. = FALSE)
    }
  } else {
    # Regression predictions
    preds <- predict(fit, newdata = new_data)
    return(preds)
  }
}

#' Fit a gradient boosting model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a classification problem
#' @param n.trees Number of trees (default: 100)
#' @param interaction.depth Depth of interactions (default: 3)
#' @param shrinkage Learning rate (default: 0.1)
#' @param n.minobsinnode Minimum number of observations in terminal nodes (default: 10)
#' @param cv.folds Number of cross-validation folds (default: 0, no CV)
#' @param ... Additional arguments to pass to gbm()
#' @return A fitted gradient boosting model
#' @keywords internal
tl_fit_boost <- function(data, formula, is_classification = FALSE,
                         n.trees = 100, interaction.depth = 3, shrinkage = 0.1,
                         n.minobsinnode = 10, cv.folds = 0, ...) {
  # Check if gbm is installed
  tl_check_packages("gbm")

  # Determine distribution based on problem type
  if (is_classification) {
    # Get response variable
    response_var <- all.vars(formula)[1]
    y <- data[[response_var]]

    # Check if binary or multiclass
    if (is.factor(y) && length(levels(y)) == 2) {
      # Binary classification
      distribution <- "bernoulli"
    } else {
      # Multiclass classification
      distribution <- "multinomial"
    }
  } else {
    # Regression
    distribution <- "gaussian"
  }

  # Fit the boosting model
  boost_model <- gbm::gbm(
    formula = formula,
    data = data,
    distribution = distribution,
    n.trees = n.trees,
    interaction.depth = interaction.depth,
    shrinkage = shrinkage,
    n.minobsinnode = n.minobsinnode,
    cv.folds = cv.folds,
    verbose = FALSE,
    ...
  )

  return(boost_model)
}

#' Predict using a gradient boosting model
#'
#' @param model A tidysl boost model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction: "response" (default), "prob" (for classification)
#' @param n.trees Number of trees to use for prediction (if NULL, uses optimal number)
#' @param ... Additional arguments
#' @return Predictions
#' @keywords internal
tl_predict_boost <- function(model, new_data, type = "response", n.trees = NULL, ...) {
  # Get the boosting model
  fit <- model$fit
  is_classification <- model$spec$is_classification

  # Determine the number of trees to use
  if (is.null(n.trees)) {
    if (fit$cv.folds > 0) {
      # Use the optimal number of trees from CV
      n.trees <- gbm::gbm.perf(fit, method = "cv", plot.it = FALSE)
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
        probs <- gbm::predict.gbm(fit, newdata = new_data, n.trees = n.trees, type = "response", ...)

        # Get class levels from training data
        response_var <- all.vars(model$spec$formula)[1]
        class_levels <- levels(factor(model$data[[response_var]]))

        # Create a data frame with probabilities for each class
        prob_df <- tibble::tibble(
          !!class_levels[1] := 1 - probs,
          !!class_levels[2] := probs
        )

        return(prob_df)
      } else if (type == "class" || type == "response") {
        # Get probabilities
        probs <- gbm::predict.gbm(fit, newdata = new_data, n.trees = n.trees, type = "response", ...)

        # Get class levels from training data
        response_var <- all.vars(model$spec$formula)[1]
        class_levels <- levels(factor(model$data[[response_var]]))

        # Convert to classes
        pred_classes <- ifelse(probs > 0.5, class_levels[2], class_levels[1])
        pred_classes <- factor(pred_classes, levels = class_levels)

        return(pred_classes)
      } else {
        stop("Invalid prediction type for boosting. Use 'prob', 'class', or 'response'.", call. = FALSE)
      }
    } else if (fit$distribution$name == "multinomial") {
      # Multiclass classification
      if (type == "prob") {
        # Get class probabilities
        probs <- gbm::predict.gbm(fit, newdata = new_data, n.trees = n.trees, type = "response", ...)

        # Reshape to data frame
        if (is.matrix(probs)) {
          # Get class levels from column names
          class_levels <- colnames(probs)
          prob_df <- as.data.frame(probs)
          names(prob_df) <- class_levels
        } else {
          # For single prediction
          response_var <- all.vars(model$spec$formula)[1]
          class_levels <- levels(factor(model$data[[response_var]]))
          prob_df <- as.data.frame(matrix(probs, nrow = 1))
          names(prob_df) <- class_levels
        }

        return(tibble::as_tibble(prob_df))
      } else if (type == "class" || type == "response") {
        # Get class probabilities
        probs <- gbm::predict.gbm(fit, newdata = new_data, n.trees = n.trees, type = "response", ...)

        # Convert to classes
        if (is.matrix(probs)) {
          # Find class with highest probability
          class_idx <- apply(probs, 1, which.max)
          class_levels <- colnames(probs)
          pred_classes <- factor(class_levels[class_idx], levels = class_levels)
        } else {
          # For single prediction
          response_var <- all.vars(model$spec$formula)[1]
          class_levels <- levels(factor(model$data[[response_var]]))
          class_idx <- which.max(probs)
          pred_classes <- factor(class_levels[class_idx], levels = class_levels)
        }

        return(pred_classes)
      } else {
        stop("Invalid prediction type for boosting. Use 'prob', 'class', or 'response'.", call. = FALSE)
      }
    }
  } else {
    # Regression predictions
    preds <- gbm::predict.gbm(fit, newdata = new_data, n.trees = n.trees, type = "response", ...)
    return(preds)
  }
}

#' Plot variable importance for tree-based models
#'
#' @param model A tidysl tree-based model object
#' @param top_n Number of top features to display (default: 20)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_col coord_flip labs theme_minimal
#' @keywords internal
tl_plot_importance <- function(model, top_n = 20, ...) {
  # Get the model
  fit <- model$fit
  method <- model$spec$method

  if (method == "tree") {
    # Decision tree importance
    # Get variable importance from rpart
    imp <- fit$variable.importance

    # Create a data frame for plotting
    importance_df <- tibble::tibble(
      feature = names(imp),
      importance = as.vector(imp)
    )
  } else if (method == "forest") {
    # Random forest importance
    # Get variable importance from randomForest
    imp <- randomForest::importance(fit)

    # Create a data frame for plotting
    if (model$spec$is_classification) {
      # For classification, use mean decrease in accuracy
      importance_df <- tibble::tibble(
        feature = rownames(imp),
        importance = imp[, "MeanDecreaseAccuracy"]
      )
    } else {
      # For regression, use % increase in MSE
      importance_df <- tibble::tibble(
        feature = rownames(imp),
        importance = imp[, "%IncMSE"]
      )
    }
  } else if (method == "boost") {
    # Gradient boosting importance
    # Get relative influence from gbm
    imp <- summary(fit, plotit = FALSE)

    # Create a data frame for plotting
    importance_df <- tibble::tibble(
      feature = imp$var,
      importance = imp$rel.inf
    )
  } else {
    stop("Variable importance plot not implemented for method: ", method, call. = FALSE)
  }

  # Filter and sort
  importance_df <- importance_df %>%
    dplyr::arrange(dplyr::desc(.data$importance)) %>%
    dplyr::slice_head(n = top_n)

  # Create the plot
  p <- ggplot2::ggplot(importance_df, ggplot2::aes(x = stats::reorder(feature, importance), y = importance)) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Variable Importance",
      x = NULL,
      y = "Importance"
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Plot a decision tree
#'
#' @param model A tidysl tree model object
#' @param ... Additional arguments to pass to rpart.plot()
#' @return A plot of the decision tree
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
#' @param model A tidysl tree-based model object
#' @param var Variable name to plot
#' @param n.pts Number of points for continuous variables (default: 20)
#' @param ... Additional arguments
#' @return A ggplot object
#' @importFrom ggplot2 ggplot aes geom_line geom_point labs theme_minimal
#' @export
tl_plot_partial_dependence <- function(model, var, n.pts = 20, ...) {
  if (!model$spec$method %in% c("tree", "forest", "boost")) {
    stop("Partial dependence plots are currently only implemented for tree-based models", call. = FALSE)
  }

  # Get the data
  data <- model$data

  # Check if variable exists
  if (!var %in% names(data)) {
    stop("Variable '", var, "' not found in the model data", call. = FALSE)
  }

  # Get variable values
  var_values <- data[[var]]

  # Create grid of values for the variable
  if (is.factor(var_values) || is.character(var_values)) {
    # For categorical variables, use unique values
    grid_values <- unique(var_values)
  } else {
    # For continuous variables, create a sequence
    grid_values <- seq(min(var_values, na.rm = TRUE), max(var_values, na.rm = TRUE), length.out = n.pts)
  }

  # Create prediction grid
  pred_data <- purrr::map_dfr(grid_values, function(val) {
    # Make a copy of the original data
    new_data <- data

    # Set the variable to the current value
    new_data[[var]] <- val

    # Make predictions
    if (model$spec$is_classification) {
      # For classification, get probabilities
      probs <- predict(model, new_data, type = "prob")

      # Calculate mean probability for each class
      class_means <- purrr::map_dbl(probs, mean)

      # Create result row
      result <- tibble::tibble(
        var_value = val,
        y = class_means[2]  # For binary classification, use the positive class
      )

      # Add class column for multiclass
      if (length(class_means) > 2) {
        result$class <- names(class_means)[which.max(class_means)]
      }
    } else {
      # For regression, get mean prediction
      preds <- predict(model, new_data, type = "response")
      mean_pred <- mean(preds)

      # Create result row
      result <- tibble::tibble(
        var_value = val,
        y = mean_pred
      )
    }

    return(result)
  })

  # Create the plot
  if (is.factor(var_values) || is.character(var_values)) {
    # Bar plot for categorical variables
    p <- ggplot2::ggplot(pred_data, ggplot2::aes(x = var_value, y = y)) +
      ggplot2::geom_col(fill = "steelblue") +
      ggplot2::labs(
        title = paste0("Partial Dependence Plot for ", var),
        x = var,
        y = if (model$spec$is_classification) "Mean Probability" else "Mean Prediction"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  } else {
    # Line plot for continuous variables
    p <- ggplot2::ggplot(pred_data, ggplot2::aes(x = var_value, y = y)) +
      ggplot2::geom_line(color = "steelblue") +
      ggplot2::geom_point(color = "steelblue") +
      ggplot2::labs(
        title = paste0("Partial Dependence Plot for ", var),
        x = var,
        y = if (model$spec$is_classification) "Mean Probability" else "Mean Prediction"
      ) +
      ggplot2::theme_minimal()
  }

  return(p)
}
