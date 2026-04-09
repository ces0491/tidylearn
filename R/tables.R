#' @title Table Functions for tidylearn
#' @name tidylearn-tables
#' @description Functions for producing formatted gt tables
#'   from tidylearn models. Provides a parallel interface to
#'   the plot functions: \code{tl_table(model, type)}
#'   dispatches to the appropriate table formatter based on model type.
#'   Requires the gt package (suggested dependency).
NULL

# ── Internal helpers ──────────────────────────────────────────────────────────

#' Apply tidylearn gt theme
#' @param gt_tbl A gt table object
#' @param title Optional table title
#' @param subtitle Optional subtitle
#' @param source_note Optional source note
#' @return A styled gt object
#' @keywords internal
#' @noRd
tl_gt_theme <- function(gt_tbl, title = NULL,
                        subtitle = NULL,
                        source_note = NULL) {
  gt_tbl <- gt_tbl %>%
    gt::tab_options(
      heading.background.color = "#2c3e50",
      heading.title.font.size = gt::px(16),
      heading.subtitle.font.size = gt::px(12),
      column_labels.background.color = "#34495e",
      column_labels.font.weight = "bold",
      row.striping.include_table_body = TRUE,
      row.striping.background_color = "#f8f9fa",
      table.border.top.color = "#2c3e50",
      table.border.bottom.color = "#2c3e50",
      table.font.size = gt::px(13)
    ) %>%
    gt::tab_style(
      style = gt::cell_text(color = "white"),
      locations = gt::cells_column_labels()
    )

  if (!is.null(title)) {
    gt_tbl <- gt_tbl %>% gt::tab_header(title = title, subtitle = subtitle)
  }
  if (!is.null(source_note)) {
    gt_tbl <- gt_tbl %>% gt::tab_source_note(source_note = source_note)
  }

  gt_tbl
}

#' Build model info string for table footnotes
#' @param model A tidylearn model object
#' @return Character string describing the model
#' @keywords internal
#' @noRd
tl_model_info <- function(model) {
  method <- model$spec$method
  if (model$spec$paradigm == "supervised") {
    task <- if (model$spec$is_classification) "classification" else "regression"
    formula_text <- deparse(model$spec$formula)
    paste0("tidylearn | ", method, " (", task, ") | ", formula_text,
           " | n = ", nrow(model$data))
  } else {
    paste0("tidylearn | ", method, " | n = ", nrow(model$data))
  }
}

# ── Main dispatcher ──────────────────────────────────────────────────────────

#' Create formatted tables for tidylearn models
#'
#' Dispatches to the appropriate table function based on model type and
#' requested table type. Requires the gt package.
#'
#' @param model A tidylearn model object
#' @param type Table type (default: "auto"). For supervised models: "metrics",
#'   "coefficients", "confusion", "importance". For unsupervised models:
#'   "variance", "loadings", "clusters". MDS models are not supported.
#' @param ... Additional arguments passed to the underlying table function
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_table(model)
#' tl_table(model, type = "coefficients")
#' }
tl_table <- function(model, type = "auto", ...) {
  tl_check_packages("gt")

  if (!inherits(model, "tidylearn_model")) {
    stop("'model' must be a tidylearn_model object", call. = FALSE)
  }

  if (inherits(model, "tidylearn_supervised")) {
    tl_table_model(model, type, ...)
  } else if (inherits(model, "tidylearn_unsupervised")) {
    tl_table_unsupervised(model, type, ...)
  }
}

#' Supervised table dispatcher
#' @keywords internal
#' @noRd
tl_table_model <- function(model, type = "auto", ...) {
  is_class <- model$spec$is_classification

  if (type == "auto") {
    type <- "metrics"
  }

  switch(
    type,
    "metrics"      = tl_table_metrics(model, ...),
    "coefficients" = tl_table_coefficients(model, ...),
    "confusion"    = tl_table_confusion(model, ...),
    "importance"   = tl_table_importance(model, ...),
    stop(
      "Unknown table type '", type, "'. ",
      if (is_class) {
        "Use: 'metrics', 'coefficients', 'confusion', or 'importance'."
      } else {
        "Use: 'metrics', 'coefficients', or 'importance'."
      },
      call. = FALSE
    )
  )
}

#' Unsupervised table dispatcher
#' @keywords internal
#' @noRd
tl_table_unsupervised <- function(model, type = "auto", ...) {
  method <- model$spec$method

  if (method == "mds") {
    stop("Table output is not supported for MDS models. Use plot() instead.",
         call. = FALSE)
  }

  if (type == "auto") {
    type <- switch(
      method,
      "pca"    = "variance",
      "kmeans" = , "pam" = , "clara" = , "dbscan" = , "hclust" = "clusters",
      stop("No default table type for method '", method, "'.", call. = FALSE)
    )
  }

  switch(
    type,
    "variance" = tl_table_variance(model, ...),
    "loadings" = tl_table_loadings(model, ...),
    "clusters" = tl_table_clusters(model, ...),
    stop(
      "Unknown table type '", type,
      "'. Use: 'variance', 'loadings', or 'clusters'.",
      call. = FALSE
    )
  )
}

# ── Supervised table functions ───────────────────────────────────────────────

#' Formatted evaluation metrics table
#'
#' Produces a styled gt table of model evaluation metrics from
#' \code{\link{tl_evaluate}}.
#'
#' @param model A tidylearn supervised model object
#' @param new_data Optional test data. If NULL, uses training data.
#' @param digits Number of decimal places (default: 4)
#' @param ... Additional arguments passed to \code{tl_evaluate}
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_table_metrics(model)
#' }
tl_table_metrics <- function(model, new_data = NULL, digits = 4, ...) {
  tl_check_packages("gt")

  eval_results <- tl_evaluate(model, new_data = new_data, ...)

  eval_results %>%
    dplyr::mutate(
      metric = gsub("_", " ", .data$metric),
      metric = tools::toTitleCase(.data$metric)
    ) %>%
    gt::gt() %>%
    gt::cols_label(metric = "Metric", value = "Value") %>%
    gt::fmt_number(columns = "value", decimals = digits) %>%
    tl_gt_theme(
      title = "Model Evaluation Metrics",
      source_note = tl_model_info(model)
    )
}

#' Formatted model coefficients table
#'
#' Produces a styled gt table of model coefficients. Supports linear,
#' polynomial, logistic, ridge, lasso, and elastic net models.
#'
#' @param model A tidylearn model object
#' @param lambda For regularised models: "1se" (default) or "min"
#' @param digits Number of decimal places (default: 4)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_table_coefficients(model)
#' }
tl_table_coefficients <- function(model, lambda = "1se", digits = 4, ...) {
  tl_check_packages("gt")

  method <- model$spec$method

  if (method %in% c("linear", "polynomial", "logistic")) {
    summ <- summary(model$fit)
    coef_mat <- summ$coefficients
    coef_tbl <- tibble::tibble(
      term = rownames(coef_mat),
      estimate = coef_mat[, 1],
      std_error = coef_mat[, 2],
      statistic = coef_mat[, 3],
      p_value = coef_mat[, 4]
    ) %>%
      dplyr::mutate(significant = ifelse(.data$p_value < 0.05, "*", ""))

    stat_label <- if (method == "logistic") "z value" else "t value"

    gt_tbl <- coef_tbl %>%
      gt::gt() %>%
      gt::cols_label(
        term = "Term", estimate = "Estimate", std_error = "Std. Error",
        statistic = stat_label, p_value = "p", significant = ""
      ) %>%
      gt::fmt_number(
        columns = c("estimate", "std_error", "statistic"),
        decimals = digits
      ) %>%
      gt::fmt_scientific(columns = "p_value", decimals = 2) %>%
      gt::tab_style(
        style = gt::cell_fill(color = "#d4edda"),
        locations = gt::cells_body(rows = coef_tbl$p_value < 0.05)
      ) %>%
      tl_gt_theme(
        title = paste0(tools::toTitleCase(method), " Model Coefficients"),
        source_note = tl_model_info(model)
      )

  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    fit <- model$fit

    if (lambda == "1se") {
      lambda_val <- attr(fit, "lambda_1se")
    } else if (lambda == "min") {
      lambda_val <- attr(fit, "lambda_min")
    } else {
      lambda_val <- lambda
    }

    coefs <- as.matrix(stats::coef(fit, s = lambda_val))

    coef_tbl <- tibble::tibble(
      term = rownames(coefs),
      estimate = as.vector(coefs)
    ) %>%
      dplyr::mutate(abs_estimate = abs(.data$estimate)) %>%
      dplyr::arrange(dplyr::desc(.data$abs_estimate))

    gt_tbl <- coef_tbl %>%
      gt::gt() %>%
      gt::cols_label(
        term = "Term", estimate = "Coefficient", abs_estimate = "|Coefficient|"
      ) %>%
      gt::fmt_number(
        columns = c("estimate", "abs_estimate"), decimals = digits
      ) %>%
      gt::tab_style(
        style = gt::cell_text(color = "#999999"),
        locations = gt::cells_body(rows = coef_tbl$estimate == 0)
      ) %>%
      tl_gt_theme(
        title = paste0(tools::toTitleCase(method), " Coefficients"),
        subtitle = paste0(
          "lambda = ", signif(lambda_val, 4),
          " (", lambda, ")"
        ),
        source_note = tl_model_info(model)
      )

  } else {
    stop(
      "Coefficient table not available for method '", method,
      "'. Try type = 'importance' instead.",
      call. = FALSE
    )
  }

  gt_tbl
}

#' Formatted confusion matrix table
#'
#' Produces a styled gt confusion matrix with correct predictions highlighted.
#' Only available for classification models.
#'
#' @param model A tidylearn classification model
#' @param new_data Optional test data. If NULL, uses training data.
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(iris, Species ~ ., method = "forest")
#' tl_table_confusion(model)
#' }
tl_table_confusion <- function(model, new_data = NULL, ...) {
  tl_check_packages("gt")

  if (!model$spec$is_classification) {
    stop("Confusion matrix table is only available for classification models",
         call. = FALSE)
  }

  if (is.null(new_data)) new_data <- model$data

  response_var <- model$spec$response_var
  actuals <- new_data[[response_var]]
  if (!is.factor(actuals)) actuals <- factor(actuals)
  predicted <- predict(model, new_data, type = "class")$.pred

  cm <- table(Actual = actuals, Predicted = predicted)
  cm_df <- as.data.frame.matrix(cm)
  cm_df$Actual <- rownames(cm_df)
  cm_df <- cm_df %>% dplyr::select("Actual", dplyr::everything())

  gt_tbl <- gt::gt(cm_df, rowname_col = "Actual") %>%
    gt::tab_spanner(label = "Predicted", columns = -1) %>%
    gt::tab_stubhead(label = "Actual") %>%
    tl_gt_theme(
      title = "Confusion Matrix",
      source_note = tl_model_info(model)
    )

  # Highlight diagonal (correct predictions)
  class_levels <- levels(actuals)
  for (cls in class_levels) {
    if (cls %in% colnames(cm_df)) {
      gt_tbl <- gt_tbl %>%
        gt::tab_style(
          style = gt::cell_fill(color = "#d4edda"),
          locations = gt::cells_body(columns = cls, rows = cls)
        )
    }
  }

  gt_tbl
}

#' Formatted feature importance table
#'
#' Produces a styled gt table of feature importance with a colour gradient.
#' Supports tree-based, regularised, and xgboost models.
#'
#' @param model A tidylearn model object
#' @param top_n Maximum number of features to display (default: 20)
#' @param digits Number of decimal places (default: 2)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(iris, Species ~ ., method = "forest")
#' tl_table_importance(model)
#' }
tl_table_importance <- function(model, top_n = 20, digits = 2, ...) {
  tl_check_packages("gt")

  method <- model$spec$method

  if (method %in% c("tree", "forest", "boost")) {
    imp_df <- tl_extract_importance(model)
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    imp_df <- tl_extract_importance_regularized(model)
  } else {
    stop("Importance table not available for method '", method, "'.",
         call. = FALSE)
  }

  imp_df <- imp_df %>%
    dplyr::arrange(dplyr::desc(.data$importance)) %>%
    dplyr::slice_head(n = top_n)

  imp_df %>%
    gt::gt() %>%
    gt::cols_label(feature = "Feature", importance = "Importance") %>%
    gt::fmt_number(columns = "importance", decimals = digits) %>%
    gt::data_color(
      columns = "importance",
      palette = c("#f8f9fa", "#2c3e50")
    ) %>%
    tl_gt_theme(
      title = "Feature Importance",
      subtitle = paste0("Top ", min(top_n, nrow(imp_df)), " features"),
      source_note = tl_model_info(model)
    )
}

# ── Unsupervised table functions ─────────────────────────────────────────────

#' Formatted PCA variance explained table
#'
#' Produces a styled gt table of variance explained by each principal component,
#' with a colour gradient on cumulative variance.
#'
#' @param model A tidylearn PCA model object
#' @param n_components Maximum number of components to show (default: all)
#' @param digits Number of decimal places (default: 4)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(iris[, 1:4], method = "pca")
#' tl_table_variance(model)
#' }
tl_table_variance <- function(model, n_components = NULL, digits = 4, ...) {
  tl_check_packages("gt")

  if (model$spec$method != "pca") {
    stop("Variance table is only available for PCA models", call. = FALSE)
  }

  var_tbl <- model$fit$variance_explained
  if (!is.null(n_components)) {
    var_tbl <- var_tbl %>% dplyr::slice_head(n = n_components)
  }

  var_tbl %>%
    gt::gt() %>%
    gt::cols_label(
      component = "Component", sdev = "Std. Dev.", variance = "Variance",
      prop_variance = "Proportion", cum_variance = "Cumulative"
    ) %>%
    gt::fmt_number(columns = c("sdev", "variance"), decimals = digits) %>%
    gt::fmt_percent(
      columns = c("prop_variance", "cum_variance"),
      decimals = 1
    ) %>%
    gt::data_color(
      columns = "cum_variance",
      palette = c("#ffffff", "#27ae60")
    ) %>%
    tl_gt_theme(
      title = "PCA Variance Explained",
      source_note = tl_model_info(model)
    )
}

#' Formatted PCA loadings table
#'
#' Produces a styled gt table of variable loadings on each principal component,
#' with a diverging colour scale to highlight strong loadings.
#'
#' @param model A tidylearn PCA model object
#' @param n_components Number of components to show (default: all)
#' @param digits Number of decimal places (default: 3)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(iris[, 1:4], method = "pca")
#' tl_table_loadings(model)
#' }
tl_table_loadings <- function(model, n_components = NULL, digits = 3, ...) {
  tl_check_packages("gt")

  if (model$spec$method != "pca") {
    stop("Loadings table is only available for PCA models", call. = FALSE)
  }

  loadings_wide <- model$fit$loadings
  if (!is.null(n_components)) {
    pc_cols <- paste0("PC", seq_len(n_components))
    loadings_wide <- loadings_wide %>%
      dplyr::select("variable", dplyr::any_of(pc_cols))
  }

  pc_cols <- setdiff(names(loadings_wide), "variable")

  loadings_wide %>%
    gt::gt() %>%
    gt::cols_label(variable = "Variable") %>%
    gt::fmt_number(columns = dplyr::all_of(pc_cols), decimals = digits) %>%
    gt::data_color(
      columns = dplyr::all_of(pc_cols),
      palette = c("#c0392b", "#ffffff", "#2980b9"),
      domain = c(-1, 1)
    ) %>%
    tl_gt_theme(
      title = "PCA Loadings",
      source_note = tl_model_info(model)
    )
}

#' Formatted cluster summary table
#'
#' Produces a styled gt table showing cluster sizes and mean feature values.
#' Supports kmeans, pam, clara, dbscan, and hclust models.
#'
#' @param model A tidylearn clustering model object
#' @param k For hclust models, the number of clusters to cut (default: 3)
#' @param digits Number of decimal places (default: 2)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
#' tl_table_clusters(model)
#' }
tl_table_clusters <- function(model, k = 3, digits = 2, ...) {
  tl_check_packages("gt")

  method <- model$spec$method

  if (method == "kmeans") {
    centers <- model$fit$centers
    sizes <- model$fit$model$size
    summary_tbl <- centers %>%
      dplyr::mutate(size = sizes, .after = "cluster")
  } else if (method %in% c("pam", "clara")) {
    centers <- model$fit$medoids
    cluster_counts <- model$fit$clusters %>%
      dplyr::count(.data$cluster, name = "size")
    summary_tbl <- centers %>%
      dplyr::left_join(cluster_counts, by = "cluster")
  } else if (method == "hclust") {
    clusters <- stats::cutree(model$fit$model, k = k)
    data_with_clusters <- model$data %>%
      dplyr::select(where(is.numeric)) %>%
      dplyr::mutate(cluster = as.integer(clusters))
    summary_tbl <- data_with_clusters %>%
      dplyr::group_by(.data$cluster) %>%
      dplyr::summarise(
        size = dplyr::n(),
        dplyr::across(where(is.numeric) & !dplyr::any_of("cluster"), mean),
        .groups = "drop"
      )
  } else if (method == "dbscan") {
    cluster_assignments <- model$fit$clusters
    data_with_clusters <- model$data %>%
      dplyr::select(where(is.numeric)) %>%
      dplyr::mutate(cluster = cluster_assignments$cluster)
    summary_tbl <- data_with_clusters %>%
      dplyr::group_by(.data$cluster) %>%
      dplyr::summarise(
        size = dplyr::n(),
        dplyr::across(where(is.numeric) & !dplyr::any_of("cluster"), mean),
        .groups = "drop"
      )
  } else {
    stop("Cluster table not available for method '",
         method, "'.", call. = FALSE)
  }

  numeric_cols <- names(summary_tbl)[
    vapply(summary_tbl, is.numeric, logical(1))
  ]
  numeric_cols <- setdiff(numeric_cols, c("cluster", "size", "medoid_index"))

  summary_tbl %>%
    gt::gt() %>%
    gt::cols_label(cluster = "Cluster", size = "Size") %>%
    gt::fmt_number(columns = dplyr::all_of(numeric_cols), decimals = digits) %>%
    gt::fmt_integer(columns = dplyr::any_of(c("cluster", "size"))) %>%
    tl_gt_theme(
      title = "Cluster Summary",
      subtitle = paste0(
        method, " | ",
        length(unique(summary_tbl$cluster)), " clusters"
      ),
      source_note = tl_model_info(model)
    )
}

# ── Standalone comparison function ───────────────────────────────────────────

#' Compare multiple models in a formatted table
#'
#' Evaluates multiple tidylearn models and presents the results side-by-side
#' in a styled gt table.
#'
#' @param ... tidylearn model objects to compare
#' @param new_data Optional test data for evaluation. If NULL, uses the
#'   training data of the first model.
#' @param names Optional character vector of model names
#' @param digits Number of decimal places (default: 4)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examples
#' \donttest{
#' m1 <- tl_model(mtcars, mpg ~ ., method = "linear")
#' m2 <- tl_model(mtcars, mpg ~ ., method = "lasso")
#' tl_table_comparison(m1, m2, names = c("Linear", "Lasso"))
#' }
tl_table_comparison <- function(..., new_data = NULL,
                                names = NULL,
                                digits = 4) {
  tl_check_packages("gt")

  models <- list(...)

  if (length(models) < 2) {
    stop("Provide at least 2 models to compare", call. = FALSE)
  }

  if (is.null(names)) {
    names <- vapply(models, function(m) {
      task <- if (m$spec$paradigm == "supervised") {
        if (m$spec$is_classification) "cls" else "reg"
      } else {
        m$spec$method
      }
      paste0(m$spec$method, " (", task, ")")
    }, character(1))
  }

  if (is.null(new_data)) {
    new_data <- models[[1]]$data
  }

  results <- purrr::map2_dfr(models, names, function(model, name) {
    eval_res <- tl_evaluate(model, new_data = new_data)
    eval_res$model <- name
    eval_res
  })

  wide_results <- results %>%
    dplyr::mutate(
      metric = gsub("_", " ", .data$metric),
      metric = tools::toTitleCase(.data$metric)
    ) %>%
    tidyr::pivot_wider(names_from = "model", values_from = "value")

  wide_results %>%
    gt::gt() %>%
    gt::cols_label(metric = "Metric") %>%
    gt::fmt_number(columns = -"metric", decimals = digits) %>%
    tl_gt_theme(
      title = "Model Comparison",
      subtitle = paste0(length(models), " models compared"),
      source_note = paste0("tidylearn | n = ", nrow(new_data))
    )
}
