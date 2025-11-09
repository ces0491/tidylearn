#' Utility functions for tidylearn
#' @keywords internal
#' @noRd

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
