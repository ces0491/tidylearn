#' Tidy Distance Matrix Computation
#'
#' Compute distance matrices with tidy output
#'
#' @param data A data frame or tibble
#' @param method Character; distance method
#'   (default: "euclidean"). Options: "euclidean",
#'   "manhattan", "maximum", "gower"
#' @param cols Columns to include (tidy select).
#'   If NULL, uses all numeric columns.
#' @param ... Additional arguments passed to distance functions
#'
#' @return A \code{\link[stats]{dist}} object containing the computed
#'   distance matrix.
#'
#' @examples
#' \donttest{
#' d <- tidy_dist(iris[, 1:4], method = "euclidean")
#' }
#'
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
#' @param weights Optional named vector of variable
#'   weights (default: equal weights)
#'
#' @return A \code{\link[stats]{dist}} object containing Gower distances, with
#'   the \code{method} attribute set to \code{"gower"}.
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

  # Pre-computation pass
  #
  # 1. Extract each column to a plain vector in a list.
  #    data[i, k] inside a loop dispatches to [.data.frame on every call:
  #    S3 method lookup + argument matching + potential intermediate allocation.
  #    col_vecs[[k]][i] is a hash-table lookup then a C-level pointer offset —
  #    benchmarks show 10-100x faster for scalar access.
  #
  # 2. Store each column's type as a string so we avoid repeated is.numeric() /
  #    is.ordered() / is.factor() S3 predicate calls inside the hot triple loop.
  #
  # 3. Ranges and rank vectors stay precomputed
  col_vecs   <- as.list(data)           # plain-vector views, no copy
  col_ranges <- vector("numeric",   p)
  col_ranks  <- vector("list",      p)
  col_type   <- vector("character", p)  # "numeric" | "ordered" | "categorical"

  for (k in seq_len(p)) {
    v <- col_vecs[[k]]
    if (is.ordered(v)) {
      r              <- as.numeric(v)
      col_ranks[[k]] <- r
      col_ranges[k]  <- max(r, na.rm = TRUE) - min(r, na.rm = TRUE)
      col_type[k]    <- "ordered"
    } else if (is.numeric(v)) {
      col_ranges[k]  <- max(v, na.rm = TRUE) - min(v, na.rm = TRUE)
      col_type[k]    <- "numeric"
    } else {
      col_type[k]    <- "categorical"
    }
  }

  # Initialize distance matrix
  dist_matrix <- matrix(0, nrow = n, ncol = n)

  # Compute pairwise distances
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {

      total_dist <- 0
      valid_vars <- 0

      # Process each variable
      for (k in seq_len(p)) {

        # Plain vector indexing — avoids [.data.frame overhead on every access
        xi <- col_vecs[[k]][i]
        xj <- col_vecs[[k]][j]

        if (is.na(xi) || is.na(xj)) next

        valid_vars <- valid_vars + weights[k]

        # Column type already resolved — no S3 predicate calls in the hot path
        if (col_type[k] == "numeric") {
          d_k <- if (col_ranges[k] > 0) abs(xi - xj) / col_ranges[k] else 0

        } else if (col_type[k] == "ordered") {
          ri <- col_ranks[[k]][i]
          rj <- col_ranks[[k]][j]
          d_k <- if (col_ranges[k] > 0) abs(ri - rj) / col_ranges[k] else 0

        } else {
          d_k <- if (xi == xj) 0 else 1
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

  dist_obj
}


#' Standardize Data
#'
#' Center and/or scale numeric variables
#'
#' @param data A data frame or tibble
#' @param center Logical; center variables? (default: TRUE)
#' @param scale Logical; scale variables to unit variance? (default: TRUE)
#'
#' @return A tibble with numeric variables centered and/or scaled as specified;
#'   non-numeric columns are returned unchanged.
#'
#' @examples
#' \donttest{
#' std <- standardize_data(iris[, 1:4])
#' }
#'
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
#' @return A named list of \code{\link[stats]{dist}} objects, one per method.
#'
#' @examples
#' \donttest{
#' dists <- compare_distances(iris[, 1:4], methods = c("euclidean", "manhattan"))
#' }
#'
#' @export
compare_distances <- function(
    data,
    methods = c("euclidean", "manhattan", "maximum")) {

  data_numeric <- data %>% dplyr::select(where(is.numeric))

  dist_list <- purrr::map(methods, ~tidy_dist(data_numeric, method = .x))
  names(dist_list) <- methods

  dist_list
}
