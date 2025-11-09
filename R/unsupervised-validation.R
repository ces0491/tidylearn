#' Tidy Silhouette Analysis
#'
#' Compute silhouette statistics for cluster validation
#'
#' @param clusters Vector of cluster assignments
#' @param dist_mat Distance matrix (dist object)
#'
#' @return A list of class "tidy_silhouette" containing:
#' \itemize{
#'   \item silhouette_data: tibble with silhouette values for each observation
#'   \item avg_width: average silhouette width
#'   \item cluster_avg: average silhouette width by cluster
#' }
#'
#' @export
tidy_silhouette <- function(clusters, dist_mat) {

  if (!inherits(dist_mat, "dist")) {
    stop("dist_mat must be a dist object")
  }

  # Compute silhouette
  sil <- cluster::silhouette(clusters, dist_mat)

  # Create silhouette tibble
  sil_tbl <- tibble::as_tibble(sil[, 1:3]) %>%
    dplyr::rename(
      cluster = cluster,
      neighbor = neighbor,
      sil_width = sil_width
    ) %>%
    dplyr::mutate(.id = rownames(sil) %||% seq_len(nrow(sil)), .before = 1)

  # Calculate average by cluster
  cluster_avg <- sil_tbl %>%
    dplyr::group_by(cluster) %>%
    dplyr::summarise(
      n = dplyr::n(),
      avg_sil_width = mean(sil_width),
      .groups = "drop"
    )

  # Overall average
  avg_width <- mean(sil_tbl$sil_width)

  result <- list(
    silhouette_data = sil_tbl,
    avg_width = avg_width,
    cluster_avg = cluster_avg
  )

  class(result) <- c("tidy_silhouette", "list")
  result
}


#' Silhouette Analysis Across Multiple k Values
#'
#' @param data A data frame or tibble
#' @param max_k Maximum number of clusters to test (default: 10)
#' @param method Clustering method: "kmeans" (default) or "hclust"
#' @param nstart If kmeans, number of random starts (default: 25)
#' @param dist_method Distance metric (default: "euclidean")
#' @param linkage_method If hclust, linkage method (default: "average")
#'
#' @return A tibble with k and average silhouette widths
#' @export
tidy_silhouette_analysis <- function(data, max_k = 10, method = "kmeans",
                                     nstart = 25, dist_method = "euclidean",
                                     linkage_method = "average") {

  data_numeric <- data %>% dplyr::select(where(is.numeric))
  dist_mat <- stats::dist(data_numeric, method = dist_method)

  # Compute silhouette for k = 2 to max_k
  sil_results <- purrr::map_dfr(2:max_k, function(k) {

    if (method == "kmeans") {
      km <- stats::kmeans(data_numeric, centers = k, nstart = nstart)
      clusters <- km$cluster
    } else if (method == "hclust") {
      hc <- stats::hclust(dist_mat, method = linkage_method)
      clusters <- stats::cutree(hc, k = k)
    } else {
      stop("method must be 'kmeans' or 'hclust'")
    }

    sil <- cluster::silhouette(clusters, dist_mat)

    tibble::tibble(
      k = k,
      avg_sil_width = mean(sil[, 3])
    )
  })

  # Add optimal k
  optimal_k <- sil_results$k[which.max(sil_results$avg_sil_width)]

  attr(sil_results, "optimal_k") <- optimal_k
  attr(sil_results, "method") <- method

  sil_results
}


#' Plot Silhouette Analysis
#'
#' @param sil_obj A tidy_silhouette object or tibble from tidy_silhouette_analysis
#'
#' @return A ggplot object
#' @export
plot_silhouette <- function(sil_obj) {

  if (inherits(sil_obj, "tidy_silhouette")) {
    # Individual silhouette plot
    sil_data <- sil_obj$silhouette_data

    p <- ggplot2::ggplot(sil_data, ggplot2::aes(x = .id, y = sil_width, fill = as.factor(cluster))) +
      ggplot2::geom_col() +
      ggplot2::geom_hline(yintercept = sil_obj$avg_width, linetype = "dashed", color = "red") +
      ggplot2::facet_wrap(~cluster, scales = "free_x") +
      ggplot2::labs(
        title = "Silhouette Plot",
        subtitle = sprintf("Average silhouette width: %.3f", sil_obj$avg_width),
        x = "Observation",
        y = "Silhouette Width",
        fill = "Cluster"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(axis.text.x = ggplot2::element_blank())

  } else if (is.data.frame(sil_obj) && "avg_sil_width" %in% names(sil_obj)) {
    # Silhouette across multiple k values
    optimal_k <- attr(sil_obj, "optimal_k")

    p <- ggplot2::ggplot(sil_obj, ggplot2::aes(x = k, y = avg_sil_width)) +
      ggplot2::geom_line(color = "steelblue", size = 1) +
      ggplot2::geom_point(color = "steelblue", size = 3) +
      ggplot2::geom_point(
        data = sil_obj %>% dplyr::filter(k == optimal_k),
        color = "red", size = 5
      ) +
      ggplot2::labs(
        title = "Average Silhouette Width vs Number of Clusters",
        subtitle = sprintf("Optimal k = %d (red point)", optimal_k),
        x = "Number of Clusters (k)",
        y = "Average Silhouette Width"
      ) +
      ggplot2::theme_minimal()

  } else {
    stop("sil_obj must be a tidy_silhouette object or silhouette analysis tibble")
  }

  p
}


#' Tidy Gap Statistic
#'
#' Compute gap statistic for determining optimal number of clusters
#'
#' @param data A data frame or tibble
#' @param FUN_cluster Clustering function (default: uses kmeans internally)
#' @param max_k Maximum number of clusters (default: 10)
#' @param B Number of bootstrap samples (default: 50)
#' @param nstart If using kmeans, number of random starts (default: 25)
#'
#' @return A list of class "tidy_gap" containing gap statistics
#' @export
tidy_gap_stat <- function(data, FUN_cluster = NULL, max_k = 10, B = 50, nstart = 25) {

  data_numeric <- data %>% dplyr::select(where(is.numeric))

  # Use cluster::clusGap
  if (is.null(FUN_cluster)) {
    gap_result <- cluster::clusGap(
      data_numeric,
      FUN = stats::kmeans,
      nstart = nstart,
      K.max = max_k,
      B = B
    )
  } else {
    gap_result <- cluster::clusGap(
      data_numeric,
      FUN = FUN_cluster,
      K.max = max_k,
      B = B
    )
  }

  # Extract results as tibble
  gap_tbl <- tibble::as_tibble(gap_result$Tab) %>%
    dplyr::mutate(k = 1:max_k, .before = 1)

  # Determine optimal k using different methods
  k_firstSEmax <- cluster::maxSE(gap_result$Tab[, "gap"],
                                 gap_result$Tab[, "SE.sim"],
                                 method = "firstSEmax")

  k_globalmax <- cluster::maxSE(gap_result$Tab[, "gap"],
                                gap_result$Tab[, "SE.sim"],
                                method = "globalmax")

  k_firstmax <- which.max(gap_result$Tab[, "gap"])

  result <- list(
    gap_data = gap_tbl,
    k_firstSEmax = k_firstSEmax,
    k_globalmax = k_globalmax,
    k_firstmax = k_firstmax,
    recommended_k = k_firstSEmax,  # Most conservative
    model = gap_result
  )

  class(result) <- c("tidy_gap", "list")
  result
}


#' Plot Gap Statistic
#'
#' @param gap_obj A tidy_gap object
#' @param show_methods Logical; show all three k selection methods? (default: FALSE)
#'
#' @return A ggplot object
#' @export
plot_gap_stat <- function(gap_obj, show_methods = FALSE) {

  if (!inherits(gap_obj, "tidy_gap")) {
    stop("gap_obj must be a tidy_gap object")
  }

  gap_data <- gap_obj$gap_data

  p <- ggplot2::ggplot(gap_data, ggplot2::aes(x = k, y = gap)) +
    ggplot2::geom_line(color = "steelblue", size = 1) +
    ggplot2::geom_point(color = "steelblue", size = 3) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = gap - SE.sim, ymax = gap + SE.sim),
      width = 0.2, alpha = 0.5
    ) +
    ggplot2::labs(
      title = "Gap Statistic",
      subtitle = sprintf("Recommended k = %d (firstSEmax method)", gap_obj$recommended_k),
      x = "Number of Clusters (k)",
      y = "Gap Statistic"
    ) +
    ggplot2::theme_minimal()

  # Add vertical lines for different methods
  if (show_methods) {
    p <- p +
      ggplot2::geom_vline(xintercept = gap_obj$k_firstSEmax, color = "red", linetype = "dashed") +
      ggplot2::geom_vline(xintercept = gap_obj$k_globalmax, color = "purple", linetype = "dashed") +
      ggplot2::geom_vline(xintercept = gap_obj$k_firstmax, color = "green", linetype = "dashed") +
      ggplot2::annotate("text", x = gap_obj$k_firstSEmax, y = max(gap_data$gap) * 0.95,
                       label = "firstSEmax", color = "red", angle = 90, vjust = -0.5, size = 3) +
      ggplot2::annotate("text", x = gap_obj$k_globalmax, y = max(gap_data$gap) * 0.95,
                       label = "globalmax", color = "purple", angle = 90, vjust = -0.5, size = 3) +
      ggplot2::annotate("text", x = gap_obj$k_firstmax, y = max(gap_data$gap) * 0.95,
                       label = "firstmax", color = "green", angle = 90, vjust = -0.5, size = 3)
  } else {
    p <- p +
      ggplot2::geom_vline(xintercept = gap_obj$recommended_k, color = "red", linetype = "dashed")
  }

  p
}


#' Calculate Cluster Validation Metrics
#'
#' Comprehensive validation metrics for a clustering result
#'
#' @param clusters Vector of cluster assignments
#' @param data Original data frame (for WSS calculation)
#' @param dist_mat Distance matrix (for silhouette)
#'
#' @return A tibble with validation metrics
#' @export
calc_validation_metrics <- function(clusters, data = NULL, dist_mat = NULL) {

  metrics <- list()

  # Number of clusters
  k <- length(unique(clusters[clusters != 0]))  # Exclude noise (0) if present
  metrics$k <- k

  # Cluster sizes
  cluster_sizes <- table(clusters)
  metrics$min_size <- min(cluster_sizes)
  metrics$max_size <- max(cluster_sizes)
  metrics$avg_size <- mean(cluster_sizes)

  # Silhouette if distance matrix provided
  if (!is.null(dist_mat)) {
    sil <- cluster::silhouette(clusters, dist_mat)
    metrics$avg_silhouette <- mean(sil[, 3])
    metrics$min_silhouette <- min(sil[, 3])
  }

  # WSS if data provided
  if (!is.null(data)) {
    data_numeric <- data %>% dplyr::select(where(is.numeric))

    # Total within-cluster sum of squares
    wss <- sum(sapply(unique(clusters), function(cl) {
      cluster_data <- data_numeric[clusters == cl, , drop = FALSE]
      if (nrow(cluster_data) > 1) {
        center <- colMeans(cluster_data)
        sum((t(cluster_data) - center)^2)
      } else {
        0
      }
    }))

    metrics$total_wss <- wss
  }

  tibble::as_tibble(metrics)
}


#' Compare Multiple Clustering Results
#'
#' @param cluster_list Named list of cluster assignment vectors
#' @param data Original data
#' @param dist_mat Distance matrix
#'
#' @return A tibble comparing all clustering results
#' @export
compare_clusterings <- function(cluster_list, data, dist_mat = NULL) {

  if (is.null(dist_mat)) {
    data_numeric <- data %>% dplyr::select(where(is.numeric))
    dist_mat <- stats::dist(data_numeric)
  }

  comparison <- purrr::map_dfr(names(cluster_list), function(name) {
    clusters <- cluster_list[[name]]
    metrics <- calc_validation_metrics(clusters, data, dist_mat)
    metrics %>% dplyr::mutate(method = name, .before = 1)
  })

  comparison
}


#' Print Method for tidy_silhouette
#'
#' @param x A tidy_silhouette object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_silhouette <- function(x, ...) {
  cat("Tidy Silhouette Analysis\n")
  cat("========================\n\n")
  cat("Average silhouette width:", round(x$avg_width, 4), "\n\n")

  cat("Interpretation:\n")
  cat("  > 0.70: Strong structure\n")
  cat("  > 0.50: Reasonable structure\n")
  cat("  > 0.25: Weak structure\n")
  cat("  < 0.25: No substantial structure\n\n")

  cat("By Cluster:\n")
  print(x$cluster_avg)

  invisible(x)
}


#' Print Method for tidy_gap
#'
#' @param x A tidy_gap object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_gap <- function(x, ...) {
  cat("Tidy Gap Statistic\n")
  cat("==================\n\n")
  cat("Recommended k:", x$recommended_k, "(firstSEmax method)\n\n")

  cat("Alternative methods:\n")
  cat("  firstSEmax: k =", x$k_firstSEmax, "(most conservative)\n")
  cat("  globalmax:  k =", x$k_globalmax, "(middle ground)\n")
  cat("  firstmax:   k =", x$k_firstmax, "(most liberal)\n\n")

  cat("Gap Statistics (first 10):\n")
  print(head(x$gap_data, 10))

  invisible(x)
}
