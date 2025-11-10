#' @title Visualization Functions for tidysl
#' @name tidysl-visualization
#' @description General visualization functions for tidysl models
#' @importFrom ggplot2 ggplot aes geom_line geom_point geom_bar geom_boxplot
#' @importFrom ggplot2 geom_histogram geom_density geom_jitter scale_color_gradient
#' @importFrom ggplot2 labs theme_minimal
#' @importFrom tibble tibble as_tibble
#' @importFrom dplyr %>% mutate filter group_by summarize arrange
NULL

#' Plot feature importance across multiple models
#'
#' @param ... tidysl model objects to compare
#' @param top_n Number of top features to display (default: 10)
#' @param names Optional character vector of model names
#' @return A ggplot object with feature importance comparison
#' @export
tl_plot_importance_comparison <- function(..., top_n = 10, names = NULL) {
  # Get models
  models <- list(...)

  # Get model names if not provided
  if (is.null(names)) {
    names <- paste0("Model ", seq_along(models))
  } else if (length(names) != length(models)) {
    stop("Length of 'names' must match the number of models", call. = FALSE)
  }

  # Extract importance for each model
  all_importance <- purrr::map2_dfr(models, names, function(model, name) {
    # Check model type
    if (model$spec$method %in% c("tree", "forest", "boost")) {
      # Tree-based models
      imp_data <- tl_extract_importance(model)

      # Add model name
      imp_data$model <- name

      return(imp_data)
    } else if (model$spec$method %in% c("ridge", "lasso", "elastic_net")) {
      # Regularized regression
      imp_data <- tl_extract_importance_regularized(model)

      # Add model name
      imp_data$model <- name

      return(imp_data)
    } else {
      warning("Importance extraction not implemented for model type: ", model$spec$method, call. = FALSE)
      return(NULL)
    }
  })

  # If no importances could be extracted, return NULL
  if (is.null(all_importance) || nrow(all_importance) == 0) {
    return(NULL)
  }

  # Find top features across all models
  top_features <- all_importance %>%
    dplyr::group_by(.data$feature) %>%
    dplyr::summarize(avg_importance = mean(.data$importance), .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(.data$avg_importance)) %>%
    dplyr::slice_head(n = top_n) %>%
    dplyr::pull(.data$feature)

  # Filter to only top features
  plot_data <- all_importance %>%
    dplyr::filter(.data$feature %in% top_features)

  # Create the plot
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = stats::reorder(feature, importance), y = importance, fill = model)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Feature Importance Comparison",
      x = NULL,
      y = "Importance",
      fill = "Model"
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Extract importance from a tree-based model
#'
#' @param model A tidysl model object
#' @return A data frame with feature importance values
#' @keywords internal
tl_extract_importance <- function(model) {
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
    stop("Variable importance extraction not implemented for method: ", method, call. = FALSE)
  }

  # Normalize importance to 0-100 scale
  importance_df <- importance_df %>%
    dplyr::mutate(importance = 100 * .data$importance / max(.data$importance))

  return(importance_df)
}

#' Extract importance from a regularized regression model
#'
#' @param model A tidysl regularized model object
#' @param lambda Which lambda to use ("1se" or "min", default: "1se")
#' @return A data frame with feature importance values
#' @keywords internal
tl_extract_importance_regularized <- function(model, lambda = "1se") {
  # Extract the glmnet model
  fit <- model$fit

  # Extract lambda value to use
  if (lambda == "1se") {
    lambda_val <- attr(fit, "lambda_1se")
  } else if (lambda == "min") {
    lambda_val <- attr(fit, "lambda_min")
  } else if (is.numeric(lambda)) {
    lambda_val <- lambda
  } else {
    stop("Invalid lambda specification. Use '1se', 'min', or a numeric value.", call. = FALSE)
  }

  # Get coefficients at selected lambda
  lambda_index <- which.min(abs(fit$lambda - lambda_val))
  coefs <- as.matrix(coef(fit, s = lambda_val))

  # Exclude intercept
  coefs <- coefs[-1, , drop = FALSE]

  # Create a data frame for plotting
  importance_df <- tibble::tibble(
    feature = rownames(coefs),
    importance = abs(as.vector(coefs))
  ) %>%
    dplyr::filter(.data$importance > 0)

  # Normalize importance to 0-100 scale
  importance_df <- importance_df %>%
    dplyr::mutate(importance = 100 * .data$importance / max(.data$importance))

  return(importance_df)
}

#' Plot model comparison
#'
#' @param ... tidysl model objects to compare
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param metrics Character vector of metrics to compute
#' @param names Optional character vector of model names
#' @return A ggplot object with model comparison
#' @export
tl_plot_model_comparison <- function(..., new_data = NULL, metrics = NULL, names = NULL) {
  # Get models
  models <- list(...)

  # Get model names if not provided
  if (is.null(names)) {
    names <- purrr::map_chr(models, function(model) {
      paste0(model$spec$method, " (", ifelse(model$spec$is_classification, "classification", "regression"), ")")
    })
  } else if (length(names) != length(models)) {
    stop("Length of 'names' must match the number of models", call. = FALSE)
  }

  # Check if all models are of the same type (classification or regression)
  is_classifications <- purrr::map_lgl(models, function(model) model$spec$is_classification)
  if (length(unique(is_classifications)) > 1) {
    stop("All models must be of the same type (classification or regression)", call. = FALSE)
  }

  is_classification <- is_classifications[1]

  # Use first model's training data if new_data not provided
  if (is.null(new_data)) {
    new_data <- models[[1]]$data
    message("Evaluating on training data. For model validation, provide separate test data.")
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

    return(eval_results)
  })

  # Create the plot
  p <- ggplot2::ggplot(model_results, ggplot2::aes(x = model, y = value, fill = metric)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::labs(
      title = "Model Comparison",
      x = NULL,
      y = "Metric Value",
      fill = "Metric"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

  return(p)
}

#' Plot cross-validation results
#'
#' @param cv_results Cross-validation results from tl_cv function
#' @param metrics Character vector of metrics to plot (if NULL, plots all metrics)
#' @return A ggplot object with cross-validation results
#' @export
tl_plot_cv_results <- function(cv_results, metrics = NULL) {
  # Extract fold metrics
  fold_metrics <- cv_results$fold_metrics

  # Filter metrics if specified
  if (!is.null(metrics)) {
    fold_metrics <- fold_metrics %>%
      dplyr::filter(.data$metric %in% metrics)
  }

  # Create the plot
  p <- ggplot2::ggplot(fold_metrics, ggplot2::aes(x = factor(fold), y = value, group = metric, color = metric)) +
    ggplot2::geom_line() +
    ggplot2::geom_point() +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::geom_hline(data = cv_results$summary,
                        ggplot2::aes(yintercept = mean_value, color = metric),
                        linetype = "dashed") +
    ggplot2::labs(
      title = "Cross-Validation Results",
      subtitle = "Dashed lines represent mean values across folds",
      x = "Fold",
      y = "Metric Value",
      color = "Metric"
    ) +
    ggplot2::theme_minimal()

  return(p)
}

#' Create interactive visualization dashboard for a model
#'
#' @param model A tidysl model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param ... Additional arguments
#' @return A Shiny app object
#' @export
tl_dashboard <- function(model, new_data = NULL, ...) {
  # Check if required packages are installed
  tl_check_packages(c("shiny", "shinydashboard", "DT"))

  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Define UI
  ui <- shinydashboard::dashboardPage(
    shinydashboard::dashboardHeader(title = "tidysl Model Dashboard"),

    shinydashboard::dashboardSidebar(
      shinydashboard::sidebarMenu(
        shinydashboard::menuItem("Overview", tabName = "overview", icon = shiny::icon("dashboard")),
        shinydashboard::menuItem("Performance", tabName = "performance", icon = shiny::icon("chart-line")),
        shinydashboard::menuItem("Predictions", tabName = "predictions", icon = shiny::icon("table")),
        shinydashboard::menuItem("Diagnostics", tabName = "diagnostics", icon = shiny::icon("chart-area"))
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
      return(model$spec$is_classification)
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
      if (model$spec$method %in% c("tree", "forest", "boost", "ridge", "lasso", "elastic_net")) {
        tl_plot_importance(model)
      } else {
        shiny::validate(
          shiny::need(FALSE, "Feature importance not available for this model type")
        )
      }
    })

    # Predictions
    output$predictions_table <- DT::renderDT({
      # Get actual values
      actuals <- new_data[[model$spec$response_var]]

      if (model$spec$is_classification) {
        # Classification
        pred_class <- predict(model, new_data, type = "class")
        pred_prob <- predict(model, new_data, type = "prob")

        # Combine into a data frame
        results <- cbind(
          data.frame(actual = actuals, predicted = pred_class),
          pred_prob
        )
      } else {
        # Regression
        predictions <- predict(model, new_data)$prediction

        # Combine into a data frame
        results <- data.frame(
          actual = actuals,
          predicted = predictions,
          residual = actuals - predictions
        )
      }

      DT::datatable(results,
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
        tl_plot_residuals(model, new_data)
      }
    })

    # Diagnostics plots (for regression)
    output$diagnostics_plot <- shiny::renderPlot({
      if (!model$spec$is_classification) {
        tl_plot_diagnostics(model)
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

#' Plot lift chart for a classification model
#'
#' @param model A tidysl classification model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param bins Number of bins for grouping predictions (default: 10)
#' @param ... Additional arguments
#' @return A ggplot object with lift chart
#' @importFrom ggplot2 ggplot aes geom_line geom_point geom_hline labs theme_minimal
#' @export
tl_plot_lift <- function(model, new_data = NULL, bins = 10, ...) {
  if (!model$spec$is_classification) {
    stop("Lift chart is only available for classification models", call. = FALSE)
  }

  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual values
  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) {
    actuals <- factor(actuals)
  }

  # For binary classification
  if (length(levels(actuals)) == 2) {
    # Get probabilities
    probs <- predict(model, new_data, type = "prob")
    pos_class <- levels(actuals)[2]
    pos_probs <- probs[[pos_class]]

    # Convert actuals to binary (0/1)
    binary_actuals <- as.integer(actuals == pos_class)

    # Order by probability
    ordered_data <- tibble::tibble(
      prob = pos_probs,
      actual = binary_actuals
    ) %>%
      dplyr::arrange(dplyr::desc(.data$prob))

    # Calculate cumulative metrics
    decile_size <- ceiling(nrow(ordered_data) / bins)

    # Calculate lift by decile
    lift_data <- tibble::tibble(
      decile = integer(),
      cumulative_responders = integer(),
      cumulative_total = integer(),
      cumulative_response_rate = numeric(),
      baseline_rate = numeric(),
      lift = numeric()
    )

    baseline_rate <- mean(binary_actuals)
    cumulative_responders <- 0
    cumulative_total <- 0

    for (i in 1:bins) {
      # Get current decile indices
      start_idx <- (i - 1) * decile_size + 1
      end_idx <- min(i * decile_size, nrow(ordered_data))

      # Update cumulative counts
      current_responders <- sum(ordered_data$actual[start_idx:end_idx])
      current_total <- end_idx - start_idx + 1

      cumulative_responders <- cumulative_responders + current_responders
      cumulative_total <- cumulative_total + current_total

      # Calculate metrics
      cumulative_response_rate <- cumulative_responders / cumulative_total
      lift <- cumulative_response_rate / baseline_rate

      # Add to results
      lift_data <- lift_data %>%
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
      ggplot2::scale_x_continuous(breaks = 1:bins) +
      ggplot2::theme_minimal()

    return(p)
  } else {
    stop("Lift chart is currently only implemented for binary classification", call. = FALSE)
  }
}

#' Plot gain chart for a classification model
#'
#' @param model A tidysl classification model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param bins Number of bins for grouping predictions (default: 10)
#' @param ... Additional arguments
#' @return A ggplot object with gain chart
#' @importFrom ggplot2 ggplot aes geom_line geom_point geom_abline labs theme_minimal
#' @export
tl_plot_gain <- function(model, new_data = NULL, bins = 10, ...) {
  if (!model$spec$is_classification) {
    stop("Gain chart is only available for classification models", call. = FALSE)
  }

  if (is.null(new_data)) {
    new_data <- model$data
  }

  # Get actual values
  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) {
    actuals <- factor(actuals)
  }

  # For binary classification
  if (length(levels(actuals)) == 2) {
    # Get probabilities
    probs <- predict(model, new_data, type = "prob")
    pos_class <- levels(actuals)[2]
    pos_probs <- probs[[pos_class]]

    # Convert actuals to binary (0/1)
    binary_actuals <- as.integer(actuals == pos_class)

    # Order by probability
    ordered_data <- tibble::tibble(
      prob = pos_probs,
      actual = binary_actuals
    ) %>%
      dplyr::arrange(dplyr::desc(.data$prob))

    # Calculate cumulative metrics
    decile_size <- ceiling(nrow(ordered_data) / bins)
    total_responders <- sum(binary_actuals)

    # Calculate gain by decile
    gain_data <- tibble::tibble(
      decile = integer(),
      cumulative_pct_population = numeric(),
      cumulative_pct_responders = numeric()
    )

    cumulative_responders <- 0

    for (i in 1:bins) {
      # Get current decile indices
      start_idx <- (i - 1) * decile_size + 1
      end_idx <- min(i * decile_size, nrow(ordered_data))

      # Update cumulative counts
      current_responders <- sum(ordered_data$actual[start_idx:end_idx])
      cumulative_responders <- cumulative_responders + current_responders

      # Calculate metrics
      cumulative_pct_population <- end_idx / nrow(ordered_data) * 100
      cumulative_pct_responders <- cumulative_responders / total_responders * 100

      # Add to results
      gain_data <- gain_data %>%
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
    p <- ggplot2::ggplot(gain_data, ggplot2::aes(x = cumulative_pct_population, y = cumulative_pct_responders)) +
      ggplot2::geom_line(color = "blue", size = 1) +
      ggplot2::geom_point(color = "blue", size = 3) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
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

    return(p)
  } else {
    stop("Gain chart is currently only implemented for binary classification", call. = FALSE)
  }
}
#' Plot Clusters in 2D Space
#'
#' Visualize clustering results using first two dimensions or specified dimensions
#'
#' @param data A data frame with cluster assignments
#' @param cluster_col Name of cluster column (default: "cluster")
#' @param x_col X-axis variable (if NULL, uses first numeric column)
#' @param y_col Y-axis variable (if NULL, uses second numeric column)
#' @param centers Optional data frame of cluster centers
#' @param title Plot title
#' @param color_noise_black If TRUE, color noise points (cluster 0) as black (default: TRUE)
#'
#' @return A ggplot object
#' @export
plot_clusters <- function(data, cluster_col = "cluster", x_col = NULL, y_col = NULL,
                          centers = NULL, title = "Cluster Plot", color_noise_black = TRUE) {

  # Find numeric columns if not specified
  numeric_cols <- names(data)[sapply(data, is.numeric)]

  if (is.null(x_col)) {
    x_col <- numeric_cols[1]
  }

  if (is.null(y_col)) {
    y_col <- if (length(numeric_cols) > 1) numeric_cols[2] else numeric_cols[1]
  }

  # Ensure cluster column is factor
  plot_data <- data %>%
    dplyr::mutate(!!cluster_col := as.factor(!!rlang::sym(cluster_col)))

  # Create base plot
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = !!rlang::sym(x_col),
                                                y = !!rlang::sym(y_col),
                                                color = !!rlang::sym(cluster_col))) +
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
    p <- p + ggplot2::scale_color_manual(
      values = c("0" = "black", setNames(grDevices::rainbow(length(unique(plot_data[[cluster_col]])) - 1),
                                         setdiff(unique(plot_data[[cluster_col]]), "0")))
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
#' @return A ggplot object
#' @export
plot_elbow <- function(wss_data, add_line = FALSE, suggested_k = NULL) {

  p <- ggplot2::ggplot(wss_data, ggplot2::aes(x = k, y = tot_withinss)) +
    ggplot2::geom_line(color = "steelblue", size = 1) +
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
      ggplot2::geom_vline(xintercept = suggested_k, linetype = "dashed", color = "red") +
      ggplot2::annotate("text", x = suggested_k, y = max(wss_data$tot_withinss) * 0.9,
                       label = sprintf("k = %d", suggested_k), color = "red", hjust = -0.2)
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
#' @return A grid of ggplot objects
#' @export
plot_cluster_comparison <- function(data, cluster_cols, x_col, y_col) {

  plots <- purrr::map(cluster_cols, function(col) {
    plot_clusters(data, cluster_col = col, x_col = x_col, y_col = y_col,
                  title = paste("Clusters:", col))
  })

  # Combine plots
  gridExtra::grid.arrange(grobs = plots, ncol = ceiling(sqrt(length(plots))))
}


#' Plot Cluster Size Distribution
#'
#' Create bar plot of cluster sizes
#'
#' @param clusters Vector of cluster assignments
#' @param title Plot title (default: "Cluster Size Distribution")
#'
#' @return A ggplot object
#' @export
plot_cluster_sizes <- function(clusters, title = "Cluster Size Distribution") {

  cluster_counts <- tibble::tibble(cluster = as.factor(clusters)) %>%
    dplyr::count(cluster)

  ggplot2::ggplot(cluster_counts, ggplot2::aes(x = cluster, y = n, fill = cluster)) +
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
#' @param threshold Horizontal line for variance threshold (default: 0.8 for 80%)
#'
#' @return A ggplot object
#' @export
plot_variance_explained <- function(variance_tbl, threshold = 0.8) {

  # Prepare data for plotting
  plot_data <- variance_tbl %>%
    dplyr::mutate(pc_num = seq_len(dplyr::n()))

  # Create dual-axis plot
  p1 <- ggplot2::ggplot(plot_data, ggplot2::aes(x = pc_num)) +
    ggplot2::geom_col(ggplot2::aes(y = prop_variance), fill = "steelblue", alpha = 0.7) +
    ggplot2::geom_line(ggplot2::aes(y = cum_variance), color = "red", size = 1) +
    ggplot2::geom_point(ggplot2::aes(y = cum_variance), color = "red", size = 2) +
    ggplot2::geom_hline(yintercept = threshold, linetype = "dashed", color = "darkgreen") +
    ggplot2::labs(
      title = "Variance Explained by Principal Components",
      subtitle = sprintf("Red line: cumulative variance | Green line: %.0f%% threshold", threshold * 100),
      x = "Principal Component",
      y = "Proportion of Variance Explained"
    ) +
    ggplot2::scale_y_continuous(labels = function(x) paste0(round(x * 100, 1), "%")) +
    ggplot2::theme_minimal()

  p1
}


#' Plot Dendrogram with Cluster Highlights
#'
#' Enhanced dendrogram with colored cluster rectangles
#'
#' @param hclust_obj Hierarchical clustering object (hclust or tidy_hclust)
#' @param k Number of clusters to highlight
#' @param title Plot title
#'
#' @return Invisibly returns hclust object (plots as side effect)
#' @export
plot_dendrogram <- function(hclust_obj, k = NULL, title = "Hierarchical Clustering Dendrogram") {

  if (inherits(hclust_obj, "tidy_hclust")) {
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
#' @return Combined plot grid
#' @export
create_cluster_dashboard <- function(data, cluster_col = "cluster", validation_metrics = NULL) {

  plots <- list()

  # 1. Cluster scatter plot (first two numeric columns)
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  if (length(numeric_cols) >= 2) {
    plots[[1]] <- plot_clusters(data, cluster_col = cluster_col,
                                x_col = numeric_cols[1], y_col = numeric_cols[2])
  }

  # 2. Cluster sizes
  plots[[2]] <- plot_cluster_sizes(data[[cluster_col]])

  # 3. If validation metrics provided, create metrics plot
  if (!is.null(validation_metrics)) {
    # Create a text plot with metrics
    metrics_text <- sprintf(
      "Validation Metrics\n\nNumber of Clusters: %d\nAvg Silhouette: %.3f\nMin Size: %d\nMax Size: %d",
      validation_metrics$k,
      validation_metrics$avg_silhouette %||% NA,
      validation_metrics$min_size,
      validation_metrics$max_size
    )

    plots[[3]] <- ggplot2::ggplot() +
      ggplot2::annotate("text", x = 0.5, y = 0.5, label = metrics_text, size = 5) +
      ggplot2::theme_void()
  }

  # Combine plots
  if (length(plots) > 0) {
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
#' @return A ggplot object
#' @export
plot_distance_heatmap <- function(dist_mat, cluster_order = NULL, title = "Distance Heatmap") {

  # Convert to matrix
  dist_matrix <- as.matrix(dist_mat)

  # Reorder if cluster order provided
  if (!is.null(cluster_order)) {
    order_idx <- order(cluster_order)
    dist_matrix <- dist_matrix[order_idx, order_idx]
  }

  # Convert to long format
  dist_long <- dist_matrix %>%
    tibble::as_tibble(rownames = "id1") %>%
    tidyr::pivot_longer(-id1, names_to = "id2", values_to = "distance")

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
