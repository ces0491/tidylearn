#' Utility functions for tidylearn
#' @keywords internal
#' @importFrom stats aov coef cor fitted median qqnorm reorder residuals runif sd setNames terms update var
#' @importFrom utils combn getFromNamespace head packageVersion
#' @noRd

# Suppress R CMD check notes about global variables from tidyverse NSE
utils::globalVariables(c(
  ".", ".id", ".obs_id", ".row_id", ":=", "Actual", "Assumption", "Details", "Freq", "Predicted",
  "SE.sim", "Status", "abs_shap_value", "actual", "all_of", "avg_sil_width", "cluster",
  "cluster_label", "coefficient", "component", "conf_lower", "conf_upper",
  "confidence", "cooks_distance", "cost", "cum_variance", "decay", "decile", "distance",
  "epoch", "error", "feature", "feature_value", "fold", "fpr", "frac_pos", "gap",
  "id1", "id2", "interaction_value", "is_best", "is_cook_influential", "is_core",
  "is_influential", "is_noise", "is_outlier", "is_top", "k", "knn_dist", "label", "lambda",
  "leverage", "lhs", "lift", "loading", "mean_pred_prob", "mean_value", "metric",
  "model", "n", "neighbor", "obs_id", "observation", "pc_num", "percentage",
  "pred", "pred_lower", "pred_upper", "predicted", "prop_variance",
  "residuals", "rhs", "score", "shap_value", "sil_width", "size",
  "sqrt_abs_residuals", "std_residual", "support", "tl_plot_model",
  "tl_plot_unsupervised", "tl_prediction_intervals", "tot_withinss", "tpr",
  "value", "var_value", "variable", "variance", "where", "x", "x_end", "y", "y_end"
))


# Null-coalescing operator
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Extract numeric columns from data
#' @keywords internal
#' @noRd
get_numeric_cols <- function(data, cols = NULL) {
  if (!is.null(cols)) {
    cols_enquo <- rlang::enquo(cols)
    data %>% dplyr::select(!!cols_enquo)
  } else {
    data %>% dplyr::select(where(is.numeric))
  }
}

#' Extract response variable from formula
#' @keywords internal
#' @noRd
extract_response <- function(formula, data) {
  if (is.null(formula)) {
    return(NULL)
  }

  vars <- all.vars(formula)
  if (length(vars) == 0) {
    return(NULL)
  }

  response_var <- vars[1]
  if (response_var %in% names(data)) {
    return(data[[response_var]])
  }
  return(NULL)
}

#' Create observation IDs
#' @keywords internal
#' @noRd
create_obs_ids <- function(data) {
  if (!is.null(rownames(data)) && !all(rownames(data) == as.character(seq_len(nrow(data))))) {
    return(rownames(data))
  }
  paste0("obs_", seq_len(nrow(data)))
}

#' Validate data for modeling
#' @keywords internal
#' @noRd
validate_data <- function(data, allow_missing = FALSE) {
  if (!is.data.frame(data)) {
    stop("data must be a data frame or tibble", call. = FALSE)
  }

  if (nrow(data) == 0) {
    stop("data has no rows", call. = FALSE)
  }

  if (!allow_missing && any(is.na(data))) {
    warning("Missing values detected in data. Consider imputation or removing missing values.")
  }

  invisible(TRUE)
}

#' Safe extraction of formula variables
#' @keywords internal
#' @noRd
get_formula_vars <- function(formula, data) {
  if (is.null(formula)) {
    return(names(data)[sapply(data, is.numeric)])
  }

  # Check if it's a one-sided formula (unsupervised)
  if (length(formula) == 2) {
    # One-sided: ~ vars
    rhs <- formula[[2]]
    if (rhs == ".") {
      return(names(data)[sapply(data, is.numeric)])
    } else {
      return(all.vars(formula))
    }
  } else {
    # Two-sided: response ~ predictors
    vars <- all.vars(formula)
    return(vars[-1])  # Exclude response
  }
}

#' Check if required packages are installed
#' @keywords internal
#' @noRd
tl_check_packages <- function(...) {
  packages <- c(...)

  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Package '", pkg, "' is required but not installed. ",
           "Please install it with: install.packages('", pkg, "')",
           call. = FALSE)
    }
  }

  invisible(TRUE)
}
