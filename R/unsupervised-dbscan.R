#' Identify DBSCAN core points
#'
#' @param data_matrix Either a coordinate matrix or a \code{dist} object,
#'   whichever was clustered
#' @param eps The neighbourhood radius
#' @param minPts Minimum points in a neighbourhood, the point included
#' @return A logical vector, TRUE for core points
#' @keywords internal
#' @noRd
tl_dbscan_core_points <- function(data_matrix, eps, minPts) {
  neighbours <- dbscan::frNN(data_matrix, eps = eps)

  # frNN excludes the point itself, so a core point needs minPts - 1
  # neighbours within eps. lengths() carries the distances' labels, which
  # would make is_core a named vector whenever the dist had labels.
  unname(lengths(neighbours$id) >= (minPts - 1))
}

#' Read a coordinate matrix as a data frame
#'
#' The k-NN and DBSCAN helpers accept a matrix of coordinates, as
#' \code{dbscan::dbscan()} does, and pick their columns with
#' \code{dplyr::select()}, which has no matrix method.
#'
#' @param data A matrix, data frame or dist object
#' @return \code{data}, as a data frame when it was a matrix
#' @keywords internal
#' @noRd
tl_as_coordinates <- function(data) {
  if (is.matrix(data)) as.data.frame(data) else data
}

#' Tidy DBSCAN Clustering
#'
#' Performs density-based clustering with tidy output
#'
#' @param data A data frame, tibble, numeric matrix, or dist object
#' @param eps Neighborhood radius (epsilon)
#' @param minPts Minimum number of points to form a dense region (default: 5)
#' @param cols Columns to include (tidy select).
#'   If NULL, uses all numeric columns, or every column for
#'   \code{distance = "gower"}.
#' @param distance Distance metric if data is not a dist object (default:
#'   "euclidean"): any method \code{\link[stats]{dist}} accepts, or "gower"
#'   for mixed data types
#'
#' @return A list of class "tidy_dbscan" containing:
#' \itemize{
#'   \item clusters: tibble with observation IDs, cluster assignments
#'     (0 = noise), and the logical flags \code{is_noise} and
#'     \code{is_core}
#'   \item summary: tibble with each cluster's size and number of core
#'     points
#'   \item n_clusters: number of clusters (excluding noise)
#'   \item n_noise: number of noise points
#'   \item eps, minPts: the parameters used
#'   \item model: original dbscan object
#' }
#'
#' @examples
#' # Basic DBSCAN
#' db_result <- tidy_dbscan(iris, eps = 0.5, minPts = 5)
#'
#' # With suggested eps from k-NN distance plot
#' eps_suggestion <- suggest_eps(iris, minPts = 5)
#' db_result <- tidy_dbscan(iris, eps = eps_suggestion$eps, minPts = 5)
#'
#' @export
tidy_dbscan <- function(data, eps, minPts = 5,
                        cols = NULL,
                        distance = "euclidean") {

  # Handle distance matrix. dbscan::dbscan() accepts a dist object
  # directly and uses it as a dissimilarity; converting to a matrix
  # instead turns every observation into an n-dimensional coordinate row
  # of its own distances, which silently clusters something else.
  input_is_dist <- inherits(data, "dist")

  if (input_is_dist) {
    data_matrix <- data
    n_obs <- attr(data, "Size")
  } else {
    data <- tl_as_coordinates(data)
    data_selected <- tl_select_columns(
      data, rlang::enquo(cols), all_columns = distance == "gower",
      numeric_only = distance != "gower", what = "DBSCAN"
    )
    n_obs <- nrow(data_selected)

    # dbscan() searches Euclidean neighbourhoods on coordinates and takes
    # any other metric as a dist object, so the metric asked for has to
    # arrive as distances. dbscan() refuses missing values in either form,
    # in words that name neither the rows nor the columns.
    if (distance == "euclidean") {
      tl_check_complete_numeric(data_selected, "DBSCAN", tolerates = NULL)
      data_matrix <- as.matrix(data_selected)
    } else {
      data_matrix <- tidy_dist(data_selected, method = distance)
      tl_check_complete_dist(data_matrix, "DBSCAN")
    }
  }

  # Perform DBSCAN
  db_model <- dbscan::dbscan(data_matrix, eps = eps, minPts = minPts)

  # Count clusters and noise
  n_clusters <- max(db_model$cluster)
  n_noise <- sum(db_model$cluster == 0)

  # Create clusters tibble
  if (!input_is_dist && !is.null(rownames(data))) {
    obs_ids <- rownames(data)
  } else if (input_is_dist && !is.null(attr(data, "Labels"))) {
    obs_ids <- attr(data, "Labels")
  } else {
    obs_ids <- paste0("obs_", seq_len(n_obs))
  }

  # A point is core when it has at least minPts neighbours (itself
  # included) within eps. The result carries no "core" attribute, so
  # reading one back gave every point is_core = FALSE.
  core_flags <- tl_dbscan_core_points(data_matrix, eps, minPts)

  clusters_tbl <- tibble::tibble(
    .obs_id = obs_ids,
    cluster = as.integer(db_model$cluster),
    is_noise = cluster == 0,
    is_core = core_flags
  )

  # Create summary statistics
  cluster_summary <- clusters_tbl |>
    dplyr::filter(!is_noise) |>
    dplyr::group_by(cluster) |>
    dplyr::summarise(
      size = dplyr::n(),
      n_core = sum(is_core),
      .groups = "drop"
    )

  # Return tidy object
  result <- list(
    clusters = clusters_tbl,
    summary = cluster_summary,
    n_clusters = n_clusters,
    n_noise = n_noise,
    eps = eps,
    minPts = minPts,
    model = db_model
  )

  class(result) <- c("tidy_dbscan", "list")
  result
}


#' Compute k-NN Distances
#'
#' Calculate distances to k-th nearest neighbor for each point
#'
#' @param data A data frame or matrix
#' @param k Number of nearest neighbors (default: 4)
#' @param cols Columns to include (tidy select).
#'   If NULL, uses all numeric columns.
#'
#' @return A tibble with columns \code{.obs_id} (observation identifier),
#'   \code{knn_dist} (distance to k-th nearest neighbor), and \code{rank}
#'   (rank of the k-NN distance).
#'
#' @examples
#' \donttest{
#' knn <- tidy_knn_dist(iris[, 1:4], k = 5)
#' }
#'
#' @export
tidy_knn_dist <- function(data, k = 4, cols = NULL) {

  data <- tl_as_coordinates(data)
  data_selected <- tl_select_columns(
    data, rlang::enquo(cols), numeric_only = TRUE,
    what = "The k-NN distance"
  )

  data_matrix <- as.matrix(data_selected)

  # Compute k-NN distances
  knn_distances <- dbscan::kNNdist(data_matrix, k = k)

  # Create tibble
  tibble::tibble(
    .obs_id = rownames(data) %||% paste0("obs_", seq_len(nrow(data))),
    knn_dist = as.numeric(knn_distances),
    rank = rank(knn_distances)
  )
}


#' Suggest eps Parameter for DBSCAN
#'
#' Use k-NN distance plot to suggest eps value
#'
#' @param data A data frame or matrix
#' @param minPts The \code{minPts} you will pass to \code{\link{tidy_dbscan}}
#'   (default: 5). The k-NN distance is read at \code{k = minPts - 1}, the
#'   neighbours a core point needs besides itself, as
#'   \code{\link[dbscan]{kNNdistplot}} does.
#' @param method Method to suggest eps: "percentile" (default), "knee"
#' @param percentile If method="percentile", which
#'   percentile to use (default: 0.95)
#'
#' @return A list containing:
#' \itemize{
#'   \item eps: suggested epsilon value
#'   \item knn_distances: full tibble of k-NN distances
#'   \item method: method used
#' }
#'
#' @examples
#' eps_info <- suggest_eps(iris, minPts = 5)
#' eps_info$eps
#'
#' @export
suggest_eps <- function(data, minPts = 5,
                        method = "percentile",
                        percentile = 0.95) {

  # frNN() and kNNdist() leave the point itself out, so a point with
  # minPts - 1 neighbours within eps is core, and the radius for minPts is
  # the distance to the (minPts - 1)th neighbour. k = minPts would suggest
  # the radius for minPts + 1.
  tl_check_whole_number(minPts, "minPts", min = 2)
  knn_data <- tidy_knn_dist(data, k = minPts - 1)

  # Suggest eps based on method
  if (method == "percentile") {
    eps_suggested <- stats::quantile(knn_data$knn_dist, percentile)

  } else if (method == "knee") {
    # Find knee/elbow in sorted k-NN distances
    sorted_dist <- sort(knn_data$knn_dist)

    # Calculate differences
    diffs <- diff(sorted_dist)

    # Find maximum jump
    max_jump_idx <- which.max(diffs)
    eps_suggested <- sorted_dist[max_jump_idx]

  } else {
    stop("method must be 'percentile' or 'knee'")
  }

  list(
    eps = as.numeric(eps_suggested),
    knn_distances = knn_data,
    method = method
  )
}


#' Plot k-NN Distance Plot
#'
#' Visualize k-NN distances to help choose eps
#'
#' @param data A data frame, matrix, or tidy_knn_dist result
#' @param k If data is a data frame, k for k-NN (default: 4)
#' @param add_suggestion Add suggested eps line? (default: TRUE)
#' @param percentile Percentile for suggestion (default: 0.95)
#'
#' @return A \code{\link[ggplot2]{ggplot}} object.
#'
#' @examples
#' \donttest{
#' plot_knn_dist(iris[, 1:4], k = 5)
#' }
#'
#' @export
plot_knn_dist <- function(data, k = 4,
                          add_suggestion = TRUE,
                          percentile = 0.95) {

  # Get k-NN distances if needed
  if (inherits(data, "tbl_df") && "knn_dist" %in% names(data)) {
    knn_data <- data
  } else {
    knn_data <- tidy_knn_dist(data, k = k)
  }

  # Sort by distance
  knn_data <- knn_data |> dplyr::arrange(knn_dist)

  # Create plot
  p <- ggplot2::ggplot(
    knn_data,
    ggplot2::aes(x = seq_along(knn_dist), y = knn_dist)
  ) +
    ggplot2::geom_line(color = "steelblue", linewidth = 1) +
    ggplot2::labs(
      title = paste0("k-NN Distance Plot (k = ", k, ")"),
      subtitle = "Look for 'elbow' or 'knee' to determine eps",
      x = "Points (sorted by distance)",
      y = paste0(k, "-NN Distance")
    ) +
    ggplot2::theme_minimal()

  # Add suggestion line
  if (add_suggestion) {
    eps_line <- stats::quantile(knn_data$knn_dist, percentile)
    p <- p +
      ggplot2::geom_hline(
        yintercept = eps_line,
        linetype = "dashed", color = "red"
      ) +
      ggplot2::annotate(
        "text",
        x = nrow(knn_data) * 0.7,
        y = eps_line * 1.1,
        # %g, because a percentile such as 0.975 is no whole percent and
        # %d refuses it; even 0.57 * 100 is 56.99999999999999
        label = sprintf(
          "Suggested eps = %.3f\n(%g%% percentile)",
          eps_line, percentile * 100
        ),
        color = "red"
      )
  }

  p
}


#' Augment Data with DBSCAN Cluster Assignments
#'
#' @param dbscan_obj A tidy_dbscan object
#' @param data Original data frame
#'
#' @return A tibble containing the original \code{data} with additional columns
#'   \code{cluster} (factor), \code{is_noise} (logical), and \code{is_core}
#'   (logical).
#'
#' @examples
#' \donttest{
#' db <- tidy_dbscan(iris[, 1:4], eps = 0.5, minPts = 5)
#' augmented <- augment_dbscan(db, iris)
#' }
#'
#' @export
augment_dbscan <- function(dbscan_obj, data) {

  if (!inherits(dbscan_obj, "tidy_dbscan")) {
    stop("dbscan_obj must be a tidy_dbscan object")
  }

  data |>
    dplyr::bind_cols(
      tibble::tibble(
        cluster = as.factor(dbscan_obj$model$cluster),
        is_noise = dbscan_obj$clusters$is_noise,
        is_core = dbscan_obj$clusters$is_core
      )
    )
}


#' Explore DBSCAN Parameters
#'
#' Test multiple eps and minPts combinations
#'
#' @param data A data frame or matrix
#' @param eps_values Vector of eps values to test
#' @param minPts_values Vector of minPts values to test
#'
#' @return A tibble with columns \code{eps}, \code{minPts}, \code{n_clusters},
#'   \code{n_noise}, and \code{prop_noise} for each parameter combination.
#'
#' @examples
#' \donttest{
#' params <- explore_dbscan_params(iris[, 1:4],
#'   eps_values = c(0.3, 0.5, 0.8), minPts_values = c(3, 5))
#' }
#'
#' @export
explore_dbscan_params <- function(data, eps_values, minPts_values) {  # nolint

  data_numeric <- tl_select_columns(tl_as_coordinates(data))

  # Create parameter grid
  param_grid <- expand.grid(
    eps = eps_values,
    minPts = minPts_values,
    stringsAsFactors = FALSE
  )

  # Test each combination
  results <- purrr::map2_dfr(
    param_grid$eps,
    param_grid$minPts,
    function(e, m) {
      db <- tidy_dbscan(
        data_numeric, eps = e, minPts = m
      )
      tibble::tibble(
        eps = e,
        minPts = m,
        n_clusters = db$n_clusters,
        n_noise = db$n_noise,
        prop_noise = db$n_noise / nrow(data_numeric)
      )
    }
  )

  results
}


#' Print Method for tidy_dbscan
#'
#' @param x A tidy_dbscan object
#' @param ... Additional arguments (ignored)
#'
#' @return The input object \code{x}, returned invisibly.
#'
#' @examples
#' \donttest{
#' db <- tidy_dbscan(iris[, 1:4], eps = 0.5, minPts = 5)
#' print(db)
#' }
#'
#' @export
print.tidy_dbscan <- function(x, ...) {
  cat("Tidy DBSCAN Clustering\n")
  cat("======================\n\n")
  cat("Parameters:\n")
  cat("  eps (neighborhood radius):", x$eps, "\n")
  cat("  minPts (minimum points): ", x$minPts, "\n\n")

  cat("Results:\n")
  cat("  Number of clusters:", x$n_clusters, "\n")
  cat("  Number of noise points:", x$n_noise, "\n")
  noise_pct <- (x$n_noise / nrow(x$clusters)) * 100
  cat("  Proportion noise:",
      sprintf("%.1f%%", noise_pct), "\n\n")

  if (nrow(x$summary) > 0) {
    cat("Cluster Summary:\n")
    print(x$summary)
  }

  cat("\nUse augment_dbscan() to add cluster assignments to your data\n")

  invisible(x)
}


#' Fit DBSCAN for tidylearn models
#' @keywords internal
#' @noRd
tl_fit_dbscan <- function(data, formula = NULL, eps = 0.5, minPts = 5,
                          distance = "euclidean", ...) {
  tl_check_packages("dbscan")

  # Without a formula, tidy_dbscan() picks the columns its distance can use
  data <- tl_ungroup(data)
  if (!is.null(formula)) {
    vars <- tl_formula_columns(
      formula, data, "DBSCAN",
      mixed_types = distance == "gower",
      alternative = "distance = \"gower\""
    )
    data <- data[, vars, drop = FALSE]
  }

  # Fit DBSCAN using tidy_dbscan
  db_result <- tidy_dbscan(
    data, eps = eps, minPts = minPts, distance = distance, ...
  )

  # Return in expected format
  list(
    clusters = db_result$clusters,
    summary = db_result$summary,
    n_clusters = db_result$n_clusters,
    n_noise = db_result$n_noise,
    model = db_result$model
  )
}
