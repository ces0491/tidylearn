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
  gt_tbl <- gt_tbl |>
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
    ) |>
    gt::tab_style(
      style = gt::cell_text(color = "white"),
      locations = gt::cells_column_labels()
    )

  if (!is.null(title)) {
    gt_tbl <- gt_tbl |> gt::tab_header(title = title, subtitle = subtitle)
  }
  if (!is.null(source_note)) {
    gt_tbl <- gt_tbl |> gt::tab_source_note(source_note = source_note)
  }

  gt_tbl
}

#' Build model info string for table footnotes
#' @param model A tidylearn model object
#' @param n The number of rows the table describes: by default, the rows
#'   the fit used
#' @return Character string describing the model
#' @keywords internal
#' @noRd
tl_model_info <- function(model, n = tl_fit_rows(model)) {
  method <- model$spec$method
  if (model$spec$paradigm == "supervised") {
    task <- if (model$spec$is_classification) "classification" else "regression"
    # deparse() splits a long formula across strings, which made two notes
    formula_text <- paste(deparse(model$spec$formula), collapse = " ")
    paste0("tidylearn | ", method, " (", task, ") | ", formula_text,
           " | n = ", n)
  } else {
    paste0("tidylearn | ", method, " | n = ", n)
  }
}

#' Rows a fitted model was trained on
#'
#' The table notes reported \code{nrow(model$data)}, which counts the rows
#' a fit dropped: \code{lm()} fitted \code{Ozone ~ Temp + Wind} on 116 of
#' airquality's 153 rows.
#'
#' @param model A tidylearn model.
#' @return The number of training rows the fit used.
#' @keywords internal
#' @noRd
tl_fit_rows <- function(model) {
  fit <- model$fit
  # nobs() has methods for lm, glm and glmnet only. Without a count of its
  # own, a fit fell back to every stored row: 153 for airquality, where
  # xgboost trains on the 116 rows with an Ozone value and svm and nn on
  # the 111 complete ones.
  n <- switch(
    model$spec$method,
    # rpart drops a row missing the response but keeps one missing a
    # predictor, which model.frame()'s count would leave out. Its root node
    # counts the rows it kept.
    tree = fit$frame$n[1],
    forest = length(fit$predicted),
    boost = fit$nTrain,
    # The rows the fit trained on: those with a response, as xgboost
    # routes a missing predictor itself
    xgboost = length(
      tl_xgb_training_rows(model$spec$formula, model$data)$y
    ),
    # Fitted values, one per row fitted; e1071 keeps none with fitted = FALSE
    svm = if (length(fit$fitted) > 0L) length(fit$fitted),
    nn = if (!is.null(fit$fitted.values)) NROW(fit$fitted.values),
    tryCatch(stats::nobs(fit), error = function(e) NULL)
  )
  if (is.numeric(n) && length(n) == 1L && !is.na(n)) n else nrow(model$data)
}

#' Rows a table scored
#'
#' \code{tl_evaluate()} leaves out a row whose response or prediction is
#' missing, and one of a class the model was not trained on, so the count
#' is of the rows left rather than the rows passed in. The rows are read
#' as \code{tl_evaluate()} reads them: the response is the formula's
#' left-hand side, and with no \code{new_data} \code{predict()} is left to
#' read the stored rows itself.
#'
#' @param model A tidylearn model.
#' @param new_data The rows passed for scoring, or NULL for the stored rows.
#' @return The number of rows scored.
#' @keywords internal
#' @noRd
tl_scored_rows <- function(model, new_data = NULL) {
  rows <- new_data %||% model$data
  if (!inherits(model, "tidylearn_supervised")) {
    return(nrow(rows))
  }
  observed <- tl_observed_response(model, rows)
  type <- if (model$spec$is_classification) "class" else "response"
  predicted <- predict(model, new_data, type = type)$.pred

  scored <- !is.na(observed) & !is.na(predicted)
  if (model$spec$is_classification) {
    scored <- scored & as.character(observed) %in% tl_model_classes(model)
  }
  sum(scored)
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
#' @examplesIf requireNamespace("gt", quietly = TRUE)
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
#' @return A \code{\link[gt]{gt}} table object. Its source note counts the
#'   rows scored.
#' @export
#' @examplesIf requireNamespace("gt", quietly = TRUE)
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_table_metrics(model)
#' }
tl_table_metrics <- function(model, new_data = NULL, digits = 4, ...) {
  tl_check_packages("gt")

  eval_results <- tl_evaluate(model, new_data = new_data, ...)
  scored <- tl_scored_rows(model, new_data)

  eval_results |>
    dplyr::mutate(
      metric = gsub("_", " ", .data$metric),
      metric = tools::toTitleCase(.data$metric)
    ) |>
    gt::gt() |>
    gt::cols_label(metric = "Metric", value = "Value") |>
    gt::fmt_number(columns = "value", decimals = digits) |>
    tl_gt_theme(
      title = "Model Evaluation Metrics",
      source_note = tl_model_info(model, n = scored)
    )
}

#' Formatted model coefficients table
#'
#' Produces a styled gt table of model coefficients. Supports linear,
#' polynomial, logistic, ridge, lasso, and elastic net models. The numbers
#' come from \code{\link{tl_coefficients}}, which returns them as a tibble
#' if you would rather format them yourself.
#'
#' @param model A tidylearn model object
#' @param lambda For regularised models: \code{"1se"} (default),
#'   \code{"min"}, or a numeric penalty within the fitted path
#' @param digits Number of decimal places (default: 4)
#' @param conf_int Whether to add a confidence interval (default:
#'   \code{FALSE}). Not available for regularised models.
#' @param level Confidence level for the interval (default: 0.95)
#' @param exponentiate Whether to report odds ratios rather than log odds
#'   (default: \code{FALSE}). Classification models only.
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @seealso \code{\link{tl_coefficients}} for the underlying tibble.
#' @export
#' @examplesIf requireNamespace("gt", quietly = TRUE)
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
#' tl_table_coefficients(model)
#' tl_table_coefficients(model, conf_int = TRUE)
#' }
tl_table_coefficients <- function(model, lambda = "1se", digits = 4,
                                  conf_int = FALSE, level = 0.95,
                                  exponentiate = FALSE, ...) {
  tl_check_packages("gt")

  # ... is kept so existing calls still run, but nothing in it is used. A
  # broom-style conf.int = TRUE landed here and produced a table with no
  # interval, so a named argument is at least reported.
  n_extra <- ...length()
  if (n_extra > 0) {
    ignored <- names(list(...))
    if (is.null(ignored)) {
      ignored <- rep("", n_extra)
    }
    ignored[ignored == ""] <- "<unnamed>"
    warning(
      "Ignoring argument(s) tl_table_coefficients() does not use: ",
      paste(ignored, collapse = ", "),
      if ("conf.int" %in% ignored) ". Did you mean conf_int?" else ".",
      call. = FALSE
    )
  }

  method <- model$spec$method

  # Refuses an unsupported method, so the branches below only have to tell
  # the two coefficient shapes apart.
  coef_tbl <- tl_coefficients(
    model,
    conf_int = conf_int, level = level,
    exponentiate = exponentiate, lambda = lambda
  )

  estimate_label <- if (exponentiate) "Odds Ratio" else "Estimate"

  if (method %in% c("linear", "polynomial", "logistic")) {
    # A rank-deficient fit leaves an unestimable term with no p-value, and
    # ifelse() carries that NA into the flag column, where it prints as the
    # word NA in a column whose other values are a star or nothing.
    coef_tbl <- coef_tbl |>
      dplyr::mutate(
        significant = ifelse(
          !is.na(.data$p_value) & .data$p_value < 0.05, "*", ""
        )
      )

    error_col <- if (exponentiate) "std_error_log" else "std_error"
    labels <- list(
      term = "Term",
      estimate = estimate_label,
      statistic = if (method == "logistic") "z value" else "t value",
      p_value = "p",
      significant = ""
    )
    labels[[error_col]] <- if (exponentiate) {
      "Std. Error (log)"
    } else {
      "Std. Error"
    }
    if (conf_int) {
      bound <- paste0(format(level * 100, trim = TRUE), "%")
      labels$conf_low <- paste("Lower", bound)
      labels$conf_high <- paste("Upper", bound)
    }

    gt_tbl <- coef_tbl |>
      gt::gt() |>
      gt::cols_label(!!!labels) |>
      gt::fmt_number(
        columns = dplyr::any_of(c("estimate", error_col, "conf_low",
                                  "conf_high", "statistic")),
        decimals = digits
      ) |>
      gt::fmt_scientific(columns = "p_value", decimals = 2) |>
      gt::tab_style(
        style = gt::cell_fill(color = "#d4edda"),
        locations = gt::cells_body(rows = coef_tbl$significant == "*")
      ) |>
      tl_gt_theme(
        title = paste0(tools::toTitleCase(method), " Model Coefficients"),
        subtitle = if (conf_int) {
          paste0("Wald ", bound, " intervals")
        },
        source_note = tl_model_info(model)
      )

  } else {
    lambda_val <- coef_tbl$lambda[[1]]

    # Whether a term survived the penalty is a property of the coefficient
    # before it moves to the odds scale, where a dropped term reads as 1.
    dropped <- if (exponentiate) 1 else 0
    # Rank by the size of the effect on the scale it was estimated on. On
    # the odds scale |odds ratio| put a dropped term (1) above a strong
    # negative effect (0.02), so the magnitude is taken of the log odds.
    coef_tbl <- coef_tbl |>
      dplyr::select(-"lambda") |>
      dplyr::mutate(
        abs_estimate = if (exponentiate) {
          abs(log(.data$estimate))
        } else {
          abs(.data$estimate)
        }
      ) |>
      dplyr::arrange(dplyr::desc(.data$abs_estimate))

    # A multiclass fit has one set of coefficients per class. Keep each
    # class's rows together, and show them under its name.
    by_class <- "class" %in% names(coef_tbl)
    if (by_class) {
      coef_tbl <- dplyr::arrange(coef_tbl, .data$class)
    }

    gt_tbl <- coef_tbl |>
      gt::gt(groupname_col = if (by_class) "class") |>
      gt::cols_label(
        term = "Term",
        estimate = if (exponentiate) "Odds Ratio" else "Coefficient",
        abs_estimate = if (exponentiate) "|log Odds Ratio|" else "|Coefficient|"
      ) |>
      gt::fmt_number(
        columns = c("estimate", "abs_estimate"), decimals = digits
      ) |>
      gt::tab_style(
        style = gt::cell_text(color = "#999999"),
        locations = gt::cells_body(rows = coef_tbl$estimate == dropped)
      ) |>
      tl_gt_theme(
        title = paste0(tools::toTitleCase(method), " Coefficients"),
        subtitle = paste0(
          "lambda = ", signif(lambda_val, 4),
          " (", lambda, ")"
        ),
        source_note = tl_model_info(model)
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
#' @return A \code{\link[gt]{gt}} table object, with a row and a column for
#'   each class the model was trained on. Rows of a class the model never
#'   saw are left out, with a warning.
#' @export
#' @examplesIf requireNamespace("gt", quietly = TRUE)
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

  # Read against the model's classes, in the model's order. A test split
  # that still declared a class the training rows dropped gave that class
  # a row of zeros, and reordered levels reordered the matrix.
  class_levels <- tl_model_classes(model)
  observed <- tl_observed_response(model, new_data)
  aligned <- tl_align_classes(observed, class_levels)
  actuals <- aligned$actuals
  predicted <- predict(model, new_data, type = "class")$.pred

  # table() drops a row missing either value, so the counts summed to
  # fewer rows than were passed in, with nothing to say so. A row of a
  # class the model never saw is reported by tl_align_classes().
  incomplete <- is.na(observed) | (aligned$keep & is.na(predicted))
  if (any(incomplete)) {
    warning(
      sum(incomplete), " row(s) with a missing response or prediction are ",
      "left out of the confusion matrix.",
      call. = FALSE
    )
  }

  cm <- table(
    Actual = actuals,
    Predicted = factor(as.character(predicted), levels = class_levels)
  )
  cm_df <- as.data.frame.matrix(cm)
  cm_df$Actual <- rownames(cm_df)
  cm_df <- cm_df |> dplyr::select("Actual", dplyr::everything())

  gt_tbl <- gt::gt(cm_df, rowname_col = "Actual") |>
    gt::tab_spanner(label = "Predicted", columns = -1) |>
    gt::tab_stubhead(label = "Actual") |>
    tl_gt_theme(
      title = "Confusion Matrix",
      source_note = tl_model_info(model, n = sum(cm))
    )

  # Highlight diagonal (correct predictions)
  for (cls in class_levels) {
    if (cls %in% colnames(cm_df)) {
      gt_tbl <- gt_tbl |>
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
#' @examplesIf requireNamespace("gt", quietly = TRUE)
#' \donttest{
#' model <- tl_model(iris, Species ~ ., method = "forest")
#' tl_table_importance(model)
#' }
tl_table_importance <- function(model, top_n = 20, digits = 2, ...) {
  tl_check_packages("gt")

  method <- model$spec$method

  if (method %in% c("tree", "forest", "boost", "xgboost")) {
    imp_df <- tl_extract_importance(model)
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    imp_df <- tl_get_importance_regularized(model)
  } else {
    stop("Importance table not available for method '", method, "'.",
         call. = FALSE)
  }

  if (nrow(imp_df) == 0) {
    stop("No feature has non-zero importance: ",
         tl_no_importance_reason(model), ".", call. = FALSE)
  }

  imp_df <- imp_df |>
    dplyr::arrange(dplyr::desc(.data$importance)) |>
    dplyr::slice_head(n = top_n)

  imp_df |>
    gt::gt() |>
    gt::cols_label(feature = "Feature", importance = "Importance") |>
    gt::fmt_number(columns = "importance", decimals = digits) |>
    gt::data_color(
      columns = "importance",
      palette = c("#f8f9fa", "#2c3e50")
    ) |>
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
#' @examplesIf requireNamespace("gt", quietly = TRUE)
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
    var_tbl <- var_tbl |> dplyr::slice_head(n = n_components)
  }

  var_tbl |>
    gt::gt() |>
    gt::cols_label(
      component = "Component", sdev = "Std. Dev.", variance = "Variance",
      prop_variance = "Proportion", cum_variance = "Cumulative"
    ) |>
    gt::fmt_number(columns = c("sdev", "variance"), decimals = digits) |>
    gt::fmt_percent(
      columns = c("prop_variance", "cum_variance"),
      decimals = 1
    ) |>
    gt::data_color(
      columns = "cum_variance",
      palette = c("#ffffff", "#27ae60")
    ) |>
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
#' @examplesIf requireNamespace("gt", quietly = TRUE)
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
    loadings_wide <- loadings_wide |>
      dplyr::select("variable", dplyr::any_of(pc_cols))
  }

  pc_cols <- setdiff(names(loadings_wide), "variable")

  loadings_wide |>
    gt::gt() |>
    gt::cols_label(variable = "Variable") |>
    gt::fmt_number(columns = dplyr::all_of(pc_cols), decimals = digits) |>
    gt::data_color(
      columns = dplyr::all_of(pc_cols),
      palette = c("#c0392b", "#ffffff", "#2980b9"),
      domain = c(-1, 1)
    ) |>
    tl_gt_theme(
      title = "PCA Loadings",
      source_note = tl_model_info(model)
    )
}

#' Formatted cluster summary table
#'
#' Produces a styled gt table showing cluster sizes and mean feature values
#' for the columns the clustering used. Supports kmeans, pam, clara, dbscan,
#' and hclust models.
#'
#' @param model A tidylearn clustering model object
#' @param k For hclust models, the number of clusters to cut (default: 3)
#' @param digits Number of decimal places (default: 2)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[gt]{gt}} table object.
#' @export
#' @examplesIf requireNamespace("gt", quietly = TRUE)
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
    summary_tbl <- centers |>
      dplyr::mutate(size = sizes, .after = "cluster")
  } else if (method %in% c("pam", "clara")) {
    centers <- model$fit$medoids
    cluster_counts <- model$fit$clusters |>
      dplyr::count(.data$cluster, name = "size")
    summary_tbl <- centers |>
      dplyr::left_join(cluster_counts, by = "cluster")
  } else if (method == "hclust") {
    clusters <- stats::cutree(model$fit$model, k = k)
    data_with_clusters <- model$data[, tl_cluster_fit_columns(model),
                                     drop = FALSE] |>
      dplyr::mutate(cluster = as.integer(clusters))
    summary_tbl <- data_with_clusters |>
      dplyr::group_by(.data$cluster) |>
      dplyr::summarise(
        size = dplyr::n(),
        dplyr::across(where(is.numeric) & !dplyr::any_of("cluster"),
                      function(x) mean(x, na.rm = TRUE)),
        .groups = "drop"
      )
  } else if (method == "dbscan") {
    cluster_assignments <- model$fit$clusters
    data_with_clusters <- model$data[, tl_cluster_fit_columns(model),
                                     drop = FALSE] |>
      dplyr::mutate(cluster = cluster_assignments$cluster)
    summary_tbl <- data_with_clusters |>
      dplyr::group_by(.data$cluster) |>
      dplyr::summarise(
        size = dplyr::n(),
        dplyr::across(where(is.numeric) & !dplyr::any_of("cluster"),
                      function(x) mean(x, na.rm = TRUE)),
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

  # dbscan labels its noise points cluster 0, which is not a cluster
  n_clusters <- length(setdiff(unique(summary_tbl$cluster), 0))

  summary_tbl |>
    gt::gt() |>
    gt::cols_label(cluster = "Cluster", size = "Size") |>
    gt::fmt_number(columns = dplyr::all_of(numeric_cols), decimals = digits) |>
    gt::fmt_integer(columns = dplyr::any_of(c("cluster", "size"))) |>
    tl_gt_theme(
      title = "Cluster Summary",
      subtitle = paste0(
        method, " | ",
        n_clusters, if (n_clusters == 1) " cluster" else " clusters",
        if (method == "dbscan" && 0 %in% summary_tbl$cluster) " + noise"
      ),
      source_note = tl_model_info(model)
    )
}

#' Columns a clustering fit used
#'
#' The fit takes the formula's variables, or the whole frame without a
#' formula, and keeps the numeric ones. The hclust and dbscan tables
#' averaged every numeric column of the data instead, including columns
#' the formula left out.
#'
#' @param model A tidylearn clustering model.
#' @return Names of the columns the fit clustered on.
#' @keywords internal
#' @noRd
tl_cluster_fit_columns <- function(model) {
  data <- model$data
  formula <- model$spec$formula
  vars <- if (is.null(formula)) {
    names(data)
  } else {
    intersect(get_formula_vars(formula, data), names(data))
  }
  vars[vapply(data[vars], is.numeric, logical(1))]
}

# ── Standalone comparison function ───────────────────────────────────────────

#' Compare multiple models in a formatted table
#'
#' Evaluates multiple tidylearn models and presents the results side-by-side
#' in a styled gt table.
#'
#' @param ... tidylearn model objects to compare
#' @param new_data Optional test data for evaluation. If NULL, the models
#'   are scored on their training data, which they must share: models
#'   fitted on different data are an error asking for \code{new_data}. A
#'   model fitted on engineered features, as \code{tl_auto_ml()} builds some
#'   of its candidates, is scored on the training data of the others.
#' @param names Optional character vector of model names
#' @param digits Number of decimal places (default: 4)
#' @return A \code{\link[gt]{gt}} table object. Its source note counts the
#'   rows scored, per model when the models scored different rows.
#' @export
#' @examplesIf requireNamespace("gt", quietly = TRUE)
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

  names <- tl_comparison_names(models, names, function(m) {
    task <- if (m$spec$paradigm == "supervised") {
      if (m$spec$is_classification) "cls" else "reg"
    } else {
      m$spec$method
    }
    paste0(m$spec$method, " (", task, ")")
  })

  # Without new_data the models are scored on the training rows they share
  if (is.null(new_data)) {
    new_data <- tl_shared_training_data(models, names)
  }

  results <- purrr::map2_dfr(models, names, function(model, name) {
    eval_res <- tl_evaluate(model, new_data = new_data)
    eval_res$model <- name
    eval_res
  })

  # Models with different predictors can score different rows of the same
  # data, so each count is given when they differ
  scored <- vapply(models, tl_scored_rows, integer(1), new_data = new_data)
  n_note <- if (length(unique(scored)) == 1L) {
    scored[[1]]
  } else {
    paste0(scored, " (", names, ")", collapse = ", ")
  }

  wide_results <- results |>
    dplyr::mutate(
      metric = gsub("_", " ", .data$metric),
      metric = tools::toTitleCase(.data$metric)
    ) |>
    tidyr::pivot_wider(names_from = "model", values_from = "value")

  wide_results |>
    gt::gt() |>
    gt::cols_label(metric = "Metric") |>
    gt::fmt_number(columns = -"metric", decimals = digits) |>
    tl_gt_theme(
      title = "Model Comparison",
      subtitle = paste0(length(models), " models compared"),
      source_note = paste0("tidylearn | n = ", n_note)
    )
}
