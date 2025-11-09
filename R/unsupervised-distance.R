#' Tidy Distance Matrix Computation
#'
#' Compute distance matrices with tidy output
#'
#' @param data A data frame or tibble
#' @param method Character; distance method (default: "euclidean"). Options: "euclidean", "manhattan", "maximum", "gower"
#' @param cols Columns to include (tidy select). If NULL, uses all numeric columns.
#' @param ... Additional arguments passed to distance functions
#'
#' @return A dist object with tidy attributes
#' @export
tidy_dist <- function(data, method = "euclidean", cols = NULL, ...) {

  # Select columns
  if (!is.null(cols)) {
    cols_enquo <- rlang::enquo(cols)
    data_selected <- data %>% dplyr::select(!!cols_enquo)
  } else {
    data_selected <- data
  }

  # Compute distance based on method
  if (method == "gower") {
    dist_mat <- tidy_gower(data_selected, ...)
  } else {
    # Convert to matrix for standard methods
    data_matrix <- as.matrix(data_selected %>% dplyr::select(where(is.numeric)))
    dist_mat <- stats::dist(data_matrix, method = method)
  }

  dist_mat
}


#' Gower Distance Calculation
#'
#' Computes Gower distance for mixed data types (numeric, factor, ordered)
#'
#' @param data A data frame or tibble
#' @param weights Optional named vector of variable weights (default: equal weights)
#'
#' @return A dist object containing Gower distances
#'
#' @details
#' Gower distance handles mixed data types:
#' - Numeric: range-normalized Manhattan distance
#' - Factor/Character: 0 if same, 1 if different
#' - Ordered: treated as numeric ranks
#'
#' Formula: d_ij = sum(w_k * d_ijk) / sum(w_k)
#' where d_ijk is the dissimilarity for variable k between obs i and j
#'
#' @examples
#' # Create example data with mixed types
#' car_data <- data.frame(
#'   horsepower = c(130, 250, 180),
#'   weight = c(1200, 1650, 1420),
#'   color = factor(c("red", "black", "blue"))
#' )
#'
#' # Compute Gower distance
#' gower_dist <- tidy_gower(car_data)
#'
#' @export
tidy_gower <- function(data, weights = NULL) {

  # Convert to data frame
  data <- as.data.frame(data)

  n <- nrow(data)
  p <- ncol(data)

  # Set up weights
  if (is.null(weights)) {
    weights <- rep(1, p)
    names(weights) <- colnames(data)
  }

  # Initialize distance matrix
  dist_matrix <- matrix(0, nrow = n, ncol = n)

  # Compute pairwise distances
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {

      total_dist <- 0
      valid_vars <- 0

      # Process each variable
      for (k in 1:p) {

        # Skip if either value is NA
        if (is.na(data[i, k]) || is.na(data[j, k])) {
          next
        }

        valid_vars <- valid_vars + weights[k]

        # Compute dissimilarity based on variable type
        if (is.numeric(data[[k]])) {
          # Numeric: range-normalized Manhattan distance
          var_range <- max(data[[k]], na.rm = TRUE) - min(data[[k]], na.rm = TRUE)

          if (var_range > 0) {
            d_k <- abs(data[i, k] - data[j, k]) / var_range
          } else {
            d_k <- 0  # No variation
          }

        } else if (is.ordered(data[[k]])) {
          # Ordinal: treat as numeric ranks
          ranks <- as.numeric(data[[k]])
          var_range <- max(ranks, na.rm = TRUE) - min(ranks, na.rm = TRUE)

          if (var_range > 0) {
            d_k <- abs(ranks[i] - ranks[j]) / var_range
          } else {
            d_k <- 0
          }

        } else if (is.factor(data[[k]]) || is.character(data[[k]])) {
          # Categorical: 0 if same, 1 if different
          d_k <- ifelse(data[i, k] == data[j, k], 0, 1)

        } else {
          # Default: treat as categorical
          d_k <- ifelse(data[i, k] == data[j, k], 0, 1)
        }

        total_dist <- total_dist + weights[k] * d_k
      }

      # Average over valid variables
      if (valid_vars > 0) {
        dist_matrix[i, j] <- total_dist / valid_vars
        dist_matrix[j, i] <- dist_matrix[i, j]  # Symmetric
      }
    }
  }

  # Convert to dist object
  dist_obj <- stats::as.dist(dist_matrix)

  # Preserve row names if available
  if (!is.null(rownames(data))) {
    attr(dist_obj, "Labels") <- rownames(data)
  }

  attr(dist_obj, "method") <- "gower"

  return(dist_obj)
}


#' Standardize Data
#'
#' Center and/or scale numeric variables
#'
#' @param data A data frame or tibble
#' @param center Logical; center variables? (default: TRUE)
#' @param scale Logical; scale variables to unit variance? (default: TRUE)
#'
#' @return A tibble with standardized numeric variables
#' @export
standardize_data <- function(data, center = TRUE, scale = TRUE) {

  data_std <- data %>%
    dplyr::mutate(
      dplyr::across(
        where(is.numeric),
        ~ if (center && scale) {
          as.numeric(base::scale(.x, center = TRUE, scale = TRUE))
        } else if (center) {
          .x - mean(.x, na.rm = TRUE)
        } else if (scale) {
          .x / stats::sd(.x, na.rm = TRUE)
        } else {
          .x
        }
      )
    )

  data_std
}


#' Compare Distance Methods
#'
#' Compute distances using multiple methods for comparison
#'
#' @param data A data frame or tibble
#' @param methods Character vector of methods to compare
#'
#' @return A list of dist objects named by method
#' @export
compare_distances <- function(data, methods = c("euclidean", "manhattan", "maximum")) {

  data_numeric <- data %>% dplyr::select(where(is.numeric))

  dist_list <- purrr::map(methods, ~tidy_dist(data_numeric, method = .x))
  names(dist_list) <- methods

  dist_list
}
