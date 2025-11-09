#' Tidy Hierarchical Clustering
#'
#' Performs hierarchical clustering with tidy output
#'
#' @param data A data frame, tibble, or dist object
#' @param method Agglomeration method: "ward.D2", "single", "complete", "average" (default), "mcquitty", "median", "centroid"
#' @param distance Distance metric if data is not a dist object (default: "euclidean")
#' @param cols Columns to include (tidy select). If NULL, uses all numeric columns.
#'
#' @return A list of class "tidy_hclust" containing:
#' \itemize{
#'   \item model: hclust object
#'   \item dist: distance matrix used
#'   \item method: linkage method used
#'   \item data: original data (for plotting)
#' }
#'
#' @examples
#' # Basic hierarchical clustering
#' hc_result <- tidy_hclust(USArrests, method = "average")
#'
#' # With specific distance
#' hc_result <- tidy_hclust(mtcars, method = "complete", distance = "manhattan")
#'
#' @export
tidy_hclust <- function(data, method = "average", distance = "euclidean", cols = NULL) {

  # Handle dist object
  if (inherits(data, "dist")) {
    dist_mat <- data
    data_orig <- NULL
  } else {
    # Select columns
    if (!is.null(cols)) {
      cols_enquo <- rlang::enquo(cols)
      data_selected <- data %>% dplyr::select(!!cols_enquo)
    } else {
      data_selected <- data %>% dplyr::select(where(is.numeric))
    }

    # Compute distance
    dist_mat <- tidy_dist(data_selected, method = distance)
    data_orig <- data_selected
  }

  # Perform hierarchical clustering
  hc_model <- stats::hclust(dist_mat, method = method)

  # Create result object
  result <- list(
    model = hc_model,
    dist = dist_mat,
    method = method,
    distance_method = distance,
    data = data_orig
  )

  class(result) <- c("tidy_hclust", "list")
  result
}


#' Cut Hierarchical Clustering Tree
#'
#' Cut dendrogram to obtain cluster assignments
#'
#' @param hclust_obj A tidy_hclust object or hclust object
#' @param k Number of clusters (optional)
#' @param h Height at which to cut (optional)
#'
#' @return A tibble with observation IDs and cluster assignments
#' @export
tidy_cutree <- function(hclust_obj, k = NULL, h = NULL) {

  if (inherits(hclust_obj, "tidy_hclust")) {
    hc_model <- hclust_obj$model
  } else if (inherits(hclust_obj, "hclust")) {
    hc_model <- hclust_obj
  } else {
    stop("hclust_obj must be a tidy_hclust or hclust object")
  }

  if (is.null(k) && is.null(h)) {
    stop("Either k or h must be specified")
  }

  # Cut tree
  if (!is.null(k)) {
    clusters <- stats::cutree(hc_model, k = k)
  } else {
    clusters <- stats::cutree(hc_model, h = h)
  }

  # Create tibble
  tibble::tibble(
    .id = names(clusters),
    cluster = as.integer(clusters)
  )
}


#' Augment Data with Hierarchical Cluster Assignments
#'
#' Add cluster assignments to original data
#'
#' @param hclust_obj A tidy_hclust object
#' @param data Original data frame
#' @param k Number of clusters (optional)
#' @param h Height at which to cut (optional)
#'
#' @return Original data with cluster column added
#' @export
augment_hclust <- function(hclust_obj, data, k = NULL, h = NULL) {

  if (!inherits(hclust_obj, "tidy_hclust")) {
    stop("hclust_obj must be a tidy_hclust object")
  }

  cluster_assignments <- tidy_cutree(hclust_obj, k = k, h = h)

  # Add clusters to data
  data %>%
    dplyr::mutate(.row_id = dplyr::row_number()) %>%
    dplyr::left_join(
      cluster_assignments %>% dplyr::mutate(.row_id = as.integer(.id)),
      by = ".row_id"
    ) %>%
    dplyr::select(-.row_id, -.id)
}


#' Plot Dendrogram
#'
#' Create dendrogram visualization
#'
#' @param hclust_obj A tidy_hclust object or hclust object
#' @param k Optional; number of clusters to highlight with rectangles
#' @param hang Fraction of plot height to hang labels (default: 0.01)
#' @param cex Label size (default: 0.7)
#'
#' @return Invisibly returns the hclust object (plots as side effect)
#' @export
tidy_dendrogram <- function(hclust_obj, k = NULL, hang = 0.01, cex = 0.7) {

  if (inherits(hclust_obj, "tidy_hclust")) {
    hc_model <- hclust_obj$model
    method_label <- hclust_obj$method
  } else if (inherits(hclust_obj, "hclust")) {
    hc_model <- hclust_obj
    method_label <- hc_model$method
  } else {
    stop("hclust_obj must be a tidy_hclust or hclust object")
  }

  # Plot dendrogram
  plot(hc_model,
       main = paste("Hierarchical Clustering Dendrogram\n(", method_label, " linkage)", sep = ""),
       xlab = "",
       ylab = "Height",
       sub = "",
       hang = hang,
       cex = cex)

  # Add rectangles if k specified
  if (!is.null(k)) {
    stats::rect.hclust(hc_model, k = k, border = 2:(k + 1))
  }

  invisible(hc_model)
}


#' Determine Optimal Number of Clusters for Hierarchical Clustering
#'
#' Use silhouette or gap statistic to find optimal k
#'
#' @param hclust_obj A tidy_hclust object
#' @param method Character; "silhouette" (default) or "gap"
#' @param max_k Maximum number of clusters to test (default: 10)
#'
#' @return A list with optimal k and evaluation results
#' @export
optimal_hclust_k <- function(hclust_obj, method = "silhouette", max_k = 10) {

  if (!inherits(hclust_obj, "tidy_hclust")) {
    stop("hclust_obj must be a tidy_hclust object")
  }

  dist_mat <- hclust_obj$dist
  hc_model <- hclust_obj$model

  if (method == "silhouette") {
    # Compute silhouette for k = 2 to max_k
    sil_widths <- purrr::map_dbl(2:max_k, function(k) {
      clusters <- stats::cutree(hc_model, k = k)
      sil <- cluster::silhouette(clusters, dist_mat)
      mean(sil[, 3])
    })

    optimal_k <- which.max(sil_widths) + 1

    result <- list(
      optimal_k = optimal_k,
      method = "silhouette",
      values = sil_widths,
      k_range = 2:max_k
    )

  } else if (method == "gap") {
    # Use gap statistic
    gap_result <- tidy_gap_stat(
      hclust_obj$data,
      FUN_cluster = function(data, k) {
        hc_temp <- stats::hclust(stats::dist(data), method = hclust_obj$method)
        stats::cutree(hc_temp, k = k)
      },
      max_k = max_k
    )

    result <- gap_result

  } else {
    stop("method must be 'silhouette' or 'gap'")
  }

  result
}


#' Print Method for tidy_hclust
#'
#' @param x A tidy_hclust object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_hclust <- function(x, ...) {
  cat("Tidy Hierarchical Clustering\n")
  cat("=============================\n\n")
  cat("Linkage method:", x$method, "\n")
  cat("Distance method:", x$distance_method, "\n")
  cat("Number of observations:", length(x$model$order), "\n")
  cat("Number of merges:", nrow(x$model$merge), "\n\n")

  cat("Use tidy_cutree() to cut the tree and obtain cluster assignments\n")
  cat("Use tidy_dendrogram() to visualize the dendrogram\n")

  invisible(x)
}
