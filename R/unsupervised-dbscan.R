#' Tidy DBSCAN Clustering
#'
#' Performs density-based clustering with tidy output
#'
#' @param data A data frame, tibble, or distance matrix
#' @param eps Neighborhood radius (epsilon)
#' @param minPts Minimum number of points to form a dense region (default: 5)
#' @param cols Columns to include (tidy select). If NULL, uses all numeric columns.
#' @param distance Distance metric if data is not a dist object (default: "euclidean")
#'
#' @return A list of class "tidy_dbscan" containing:
#' \itemize{
#'   \item clusters: tibble with observation IDs and cluster assignments (0 = noise)
#'   \item core_points: logical vector indicating core points
#'   \item n_clusters: number of clusters (excluding noise)
#'   \item n_noise: number of noise points
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
tidy_dbscan <- function(data, eps, minPts = 5, cols = NULL, distance = "euclidean") {

  # Handle distance matrix
  if (inherits(data, "dist")) {
    data_matrix <- as.matrix(data)
    data_orig <- NULL
  } else {
    # Select columns
    if (!is.null(cols)) {
      cols_enquo <- rlang::enquo(cols)
      data_selected <- data %>% dplyr::select(!!cols_enquo)
    } else {
      data_selected <- data %>% dplyr::select(where(is.numeric))
    }

    data_matrix <- as.matrix(data_selected)
    data_orig <- data_selected
  }

  # Perform DBSCAN
  db_model <- dbscan::dbscan(data_matrix, eps = eps, minPts = minPts)

  # Count clusters and noise
  n_clusters <- max(db_model$cluster)
  n_noise <- sum(db_model$cluster == 0)

  # Create clusters tibble
  if (!is.null(rownames(data))) {
    obs_ids <- rownames(data)
  } else {
    obs_ids <- paste0("obs_", seq_len(nrow(data_matrix)))
  }

  clusters_tbl <- tibble::tibble(
    .id = obs_ids,
    cluster = as.integer(db_model$cluster),
    is_noise = cluster == 0,
    is_core = seq_along(cluster) %in% attr(db_model, "core")
  )

  # Create summary statistics
  cluster_summary <- clusters_tbl %>%
    dplyr::filter(!is_noise) %>%
    dplyr::group_by(cluster) %>%
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
#' @param cols Columns to include (tidy select). If NULL, uses all numeric columns.
#'
#' @return A tibble with observation IDs and k-NN distances
#' @export
tidy_knn_dist <- function(data, k = 4, cols = NULL) {

  # Select columns
  if (!is.null(cols)) {
    cols_enquo <- rlang::enquo(cols)
    data_selected <- data %>% dplyr::select(!!cols_enquo)
  } else {
    data_selected <- data %>% dplyr::select(where(is.numeric))
  }

  data_matrix <- as.matrix(data_selected)

  # Compute k-NN distances
  knn_distances <- dbscan::kNNdist(data_matrix, k = k)

  # Create tibble
  tibble::tibble(
    .id = rownames(data) %||% paste0("obs_", seq_len(nrow(data))),
    knn_dist = as.numeric(knn_distances),
    rank = rank(knn_distances)
  )
}


#' Suggest eps Parameter for DBSCAN
#'
#' Use k-NN distance plot to suggest eps value
#'
#' @param data A data frame or matrix
#' @param minPts Minimum points parameter (used as k for k-NN)
#' @param method Method to suggest eps: "knee" (default), "percentile"
#' @param percentile If method="percentile", which percentile to use (default: 0.95)
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
suggest_eps <- function(data, minPts = 5, method = "percentile", percentile = 0.95) {

  # Compute k-NN distances
  knn_data <- tidy_knn_dist(data, k = minPts)

  # Suggest eps based on method
  if (method == "percentile") {
    eps_suggested <- stats::quantile(knn_data$knn_dist, percentile)

  } else if (method == "knee") {
    # Find knee/elbow in sorted k-NN distances
    sorted_dist <- sort(knn_data$knn_dist)
    n <- length(sorted_dist)

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
#' @param data A data frame or tidy_knn_dist result
#' @param k If data is a data frame, k for k-NN (default: 4)
#' @param add_suggestion Add suggested eps line? (default: TRUE)
#' @param percentile Percentile for suggestion (default: 0.95)
#'
#' @return A ggplot object
#' @export
plot_knn_dist <- function(data, k = 4, add_suggestion = TRUE, percentile = 0.95) {

  # Get k-NN distances if needed
  if (inherits(data, "tbl_df") && "knn_dist" %in% names(data)) {
    knn_data <- data
  } else {
    knn_data <- tidy_knn_dist(data, k = k)
  }

  # Sort by distance
  knn_data <- knn_data %>% dplyr::arrange(knn_dist)

  # Create plot
  p <- ggplot2::ggplot(knn_data, ggplot2::aes(x = seq_along(knn_dist), y = knn_dist)) +
    ggplot2::geom_line(color = "steelblue", size = 1) +
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
      ggplot2::geom_hline(yintercept = eps_line, linetype = "dashed", color = "red") +
      ggplot2::annotate("text",
                       x = nrow(knn_data) * 0.7,
                       y = eps_line * 1.1,
                       label = sprintf("Suggested eps = %.3f\n(%d%% percentile)", eps_line, percentile * 100),
                       color = "red")
  }

  p
}


#' Augment Data with DBSCAN Cluster Assignments
#'
#' @param dbscan_obj A tidy_dbscan object
#' @param data Original data frame
#'
#' @return Original data with cluster information added
#' @export
augment_dbscan <- function(dbscan_obj, data) {

  if (!inherits(dbscan_obj, "tidy_dbscan")) {
    stop("dbscan_obj must be a tidy_dbscan object")
  }

  data %>%
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
#' @return A tibble with parameter combinations and resulting cluster counts
#' @export
explore_dbscan_params <- function(data, eps_values, minPts_values) {

  data_numeric <- data %>% dplyr::select(where(is.numeric))

  # Create parameter grid
  param_grid <- expand.grid(
    eps = eps_values,
    minPts = minPts_values,
    stringsAsFactors = FALSE
  )

  # Test each combination
  results <- purrr::map2_dfr(param_grid$eps, param_grid$minPts, function(e, m) {
    db <- tidy_dbscan(data_numeric, eps = e, minPts = m)

    tibble::tibble(
      eps = e,
      minPts = m,
      n_clusters = db$n_clusters,
      n_noise = db$n_noise,
      prop_noise = db$n_noise / nrow(data_numeric)
    )
  })

  results
}


#' Print Method for tidy_dbscan
#'
#' @param x A tidy_dbscan object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_dbscan <- function(x, ...) {
  cat("Tidy DBSCAN Clustering\n")
  cat("======================\n\n")
  cat("Parameters:\n")
  cat("  eps (neighborhood radius):", x$eps, "\n")
  cat("  minPts (minimum points):  ", x$minPts, "\n\n")

  cat("Results:\n")
  cat("  Number of clusters:", x$n_clusters, "\n")
  cat("  Number of noise points:", x$n_noise, "\n")
  cat("  Proportion noise:", sprintf("%.1f%%", (x$n_noise / nrow(x$clusters)) * 100), "\n\n")

  if (nrow(x$summary) > 0) {
    cat("Cluster Summary:\n")
    print(x$summary)
  }

  cat("\nUse augment_dbscan() to add cluster assignments to your data\n")

  invisible(x)
}
