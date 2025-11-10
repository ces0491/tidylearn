#' Tidy K-Means Clustering
#'
#' Performs k-means clustering with tidy output
#'
#' @param data A data frame or tibble
#' @param k Number of clusters
#' @param cols Columns to include (tidy select). If NULL, uses all numeric columns.
#' @param nstart Number of random starts (default: 25)
#' @param iter_max Maximum number of iterations (default: 100)
#' @param algorithm K-means algorithm: "Hartigan-Wong" (default), "Lloyd", "Forgy", "MacQueen"
#'
#' @return A list of class "tidy_kmeans" containing:
#' \itemize{
#'   \item clusters: tibble with observation IDs and cluster assignments
#'   \item centers: tibble of cluster centers
#'   \item metrics: tibble with clustering quality metrics
#'   \item model: original kmeans object
#' }
#'
#' @examples
#' # Basic k-means
#' km_result <- tidy_kmeans(iris, k = 3)
#'
#' @export
tidy_kmeans <- function(data, k, cols = NULL, nstart = 25, iter_max = 100,
                        algorithm = "Hartigan-Wong") {

  # Select columns
  if (!is.null(cols)) {
    cols_enquo <- rlang::enquo(cols)
    data_selected <- data %>% dplyr::select(!!cols_enquo)
  } else {
    data_selected <- data %>% dplyr::select(where(is.numeric))
  }

  # Perform k-means
  km_model <- stats::kmeans(data_selected, centers = k, nstart = nstart,
                            iter.max = iter_max, algorithm = algorithm)

  # Create clusters tibble
  clusters_tbl <- tibble::tibble(
    .id = rownames(data) %||% as.character(seq_len(nrow(data))),
    cluster = as.integer(km_model$cluster)
  )

  # Create centers tibble
  centers_tbl <- tibble::as_tibble(km_model$centers) %>%
    dplyr::mutate(cluster = seq_len(k), .before = 1)

  # Create metrics tibble
  metrics_tbl <- tibble::tibble(
    k = k,
    tot_withinss = km_model$tot.withinss,
    betweenss = km_model$betweenss,
    tot_ss = km_model$totss,
    iter = km_model$iter,
    converged = km_model$ifault == 0
  )

  # Return tidy object
  result <- list(
    clusters = clusters_tbl,
    centers = centers_tbl,
    metrics = metrics_tbl,
    sizes = km_model$size,
    model = km_model
  )

  class(result) <- c("tidy_kmeans", "list")
  result
}


#' Augment Data with K-Means Cluster Assignments
#'
#' @param kmeans_obj A tidy_kmeans object
#' @param data Original data frame
#'
#' @return Original data with cluster column added
#' @export
augment_kmeans <- function(kmeans_obj, data) {

  if (!inherits(kmeans_obj, "tidy_kmeans")) {
    stop("kmeans_obj must be a tidy_kmeans object")
  }

  data %>%
    dplyr::bind_cols(
      tibble::tibble(cluster = as.factor(kmeans_obj$model$cluster))
    )
}


#' Tidy PAM (Partitioning Around Medoids)
#'
#' Performs PAM clustering with tidy output
#'
#' @param data A data frame, tibble, or dist object
#' @param k Number of clusters
#' @param metric Distance metric (default: "euclidean"). Use "gower" for mixed data types.
#' @param cols Columns to include (tidy select). If NULL, uses all columns.
#'
#' @return A list of class "tidy_pam" containing:
#' \itemize{
#'   \item clusters: tibble with observation IDs and cluster assignments
#'   \item medoids: tibble of medoid indices and values
#'   \item silhouette: average silhouette width
#'   \item model: original pam object
#' }
#'
#' @examples
#' # PAM with Euclidean distance
#' pam_result <- tidy_pam(iris, k = 3)
#'
#' # PAM with Gower distance for mixed data
#' pam_result <- tidy_pam(mtcars, k = 3, metric = "gower")
#'
#' @export
tidy_pam <- function(data, k, metric = "euclidean", cols = NULL) {

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
      data_selected <- data
    }

    # Compute distance
    if (metric == "gower") {
      dist_mat <- tidy_gower(data_selected)
    } else {
      dist_mat <- tidy_dist(data_selected, method = metric)
    }

    data_orig <- data_selected
  }

  # Perform PAM
  pam_model <- cluster::pam(dist_mat, k = k, diss = TRUE)

  # Create clusters tibble
  clusters_tbl <- tibble::tibble(
    .id = names(pam_model$clustering),
    cluster = as.integer(pam_model$clustering)
  )

  # Create medoids tibble
  if (!is.null(data_orig)) {
    medoids_tbl <- tibble::as_tibble(data_orig[pam_model$medoids, , drop = FALSE]) %>%
      dplyr::mutate(
        cluster = seq_len(k),
        medoid_index = pam_model$medoids,
        .before = 1
      )
  } else {
    medoids_tbl <- tibble::tibble(
      cluster = seq_len(k),
      medoid_index = pam_model$medoids
    )
  }

  # Return tidy object
  result <- list(
    clusters = clusters_tbl,
    medoids = medoids_tbl,
    silhouette_avg = pam_model$silinfo$avg.width,
    silhouette_data = pam_model$silinfo,
    model = pam_model
  )

  class(result) <- c("tidy_pam", "list")
  result
}


#' Augment Data with PAM Cluster Assignments
#'
#' @param pam_obj A tidy_pam object
#' @param data Original data frame
#'
#' @return Original data with cluster column added
#' @export
augment_pam <- function(pam_obj, data) {

  if (!inherits(pam_obj, "tidy_pam")) {
    stop("pam_obj must be a tidy_pam object")
  }

  data %>%
    dplyr::bind_cols(
      tibble::tibble(cluster = as.factor(pam_obj$model$clustering))
    )
}


#' Tidy CLARA (Clustering Large Applications)
#'
#' Performs CLARA clustering (scalable version of PAM)
#'
#' @param data A data frame or tibble
#' @param k Number of clusters
#' @param metric Distance metric (default: "euclidean")
#' @param samples Number of samples to draw (default: 50)
#' @param sampsize Sample size (default: min(n, 40 + 2*k))
#'
#' @return A list of class "tidy_clara" containing clustering results
#'
#' @examples
#' \donttest{
#' # CLARA for large datasets
#' large_data <- iris[rep(1:nrow(iris), 10), 1:4]
#' clara_result <- tidy_clara(large_data, k = 3, samples = 50)
#' print(clara_result)
#' }
#'
#' @export
tidy_clara <- function(data, k, metric = "euclidean", samples = 50, sampsize = NULL) {

  # Select numeric columns if data frame
  if (!inherits(data, "dist")) {
    data_numeric <- data %>% dplyr::select(where(is.numeric))
  } else {
    data_numeric <- data
  }

  # Set default sampsize if not provided
  if (is.null(sampsize)) {
    n <- nrow(data_numeric)
    sampsize <- min(n, 40 + 2 * k)
  }

  # Perform CLARA
  clara_model <- cluster::clara(
    data_numeric,
    k = k,
    metric = metric,
    samples = samples,
    sampsize = sampsize
  )

  # Create clusters tibble
  clusters_tbl <- tibble::tibble(
    .id = names(clara_model$clustering) %||% as.character(seq_along(clara_model$clustering)),
    cluster = as.integer(clara_model$clustering)
  )

  # Create medoids tibble
  medoids_tbl <- tibble::as_tibble(clara_model$medoids) %>%
    dplyr::mutate(cluster = seq_len(k), .before = 1)

  # Return tidy object
  result <- list(
    clusters = clusters_tbl,
    medoids = medoids_tbl,
    silhouette_avg = clara_model$silinfo$avg.width,
    model = clara_model
  )

  class(result) <- c("tidy_clara", "list")
  result
}


#' Calculate Within-Cluster Sum of Squares for Different k
#'
#' Used for elbow method to determine optimal k
#'
#' @param data A data frame or tibble
#' @param max_k Maximum number of clusters to test (default: 10)
#' @param nstart Number of random starts for each k (default: 25)
#'
#' @return A tibble with k and corresponding total within-cluster SS
#' @export
calc_wss <- function(data, max_k = 10, nstart = 25) {

  data_numeric <- data %>% dplyr::select(where(is.numeric))

  wss_values <- purrr::map_dbl(1:max_k, function(k) {
    if (k == 1) {
      # For k=1, total SS is the total variance
      sum(apply(data_numeric, 2, var) * (nrow(data_numeric) - 1))
    } else {
      km <- stats::kmeans(data_numeric, centers = k, nstart = nstart)
      km$tot.withinss
    }
  })

  tibble::tibble(
    k = 1:max_k,
    tot_withinss = wss_values
  )
}


#' Find Optimal Number of Clusters
#'
#' Use multiple methods to suggest optimal k
#'
#' @param data A data frame or tibble
#' @param max_k Maximum k to test (default: 10)
#' @param methods Vector of methods: "silhouette", "gap", "wss" (default: all)
#'
#' @return A list with results from each method
#' @export
optimal_clusters <- function(data, max_k = 10, methods = c("silhouette", "gap", "wss")) {

  results <- list()

  if ("wss" %in% methods) {
    results$wss <- calc_wss(data, max_k = max_k)
  }

  if ("silhouette" %in% methods) {
    sil_results <- tidy_silhouette_analysis(data, max_k = max_k)
    results$silhouette <- sil_results
  }

  if ("gap" %in% methods) {
    gap_results <- tidy_gap_stat(data, max_k = max_k)
    results$gap <- gap_results
  }

  class(results) <- c("optimal_k_results", "list")
  results
}


#' Print Method for tidy_kmeans
#'
#' @param x A tidy_kmeans object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_kmeans <- function(x, ...) {
  cat("Tidy K-Means Clustering\n")
  cat("=======================\n\n")
  cat("Number of clusters:", x$metrics$k, "\n")
  cat("Cluster sizes:", paste(x$sizes, collapse = ", "), "\n")
  cat("Total within-cluster SS:", round(x$metrics$tot_withinss, 2), "\n")
  cat("Between-cluster SS:", round(x$metrics$betweenss, 2), "\n")
  cat("Iterations:", x$metrics$iter, "\n")
  cat("Converged:", x$metrics$converged, "\n\n")

  cat("Cluster Centers:\n")
  print(x$centers)

  invisible(x)
}


#' Print Method for tidy_pam
#'
#' @param x A tidy_pam object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_pam <- function(x, ...) {
  cat("Tidy PAM Clustering\n")
  cat("===================\n\n")
  cat("Number of clusters:", nrow(x$medoids), "\n")
  cat("Average silhouette width:", round(x$silhouette_avg, 4), "\n\n")

  cat("Medoids:\n")
  print(x$medoids)

  cat("\nCluster sizes:\n")
  print(table(x$clusters$cluster))

  invisible(x)
}

#' Fit K-means for tidylearn models
#' @keywords internal
#' @noRd
tl_fit_kmeans <- function(data, formula = NULL, k = 3, ...) {
  # Extract variables to use
  if (!is.null(formula)) {
    vars <- get_formula_vars(formula, data)
    data_for_km <- data[, vars, drop = FALSE]
  } else {
    data_for_km <- data %>% dplyr::select(where(is.numeric))
  }

  # Fit k-means using tidy_kmeans
  km_result <- tidy_kmeans(data_for_km, k = k, ...)

  # Return in expected format
  list(
    clusters = km_result$clusters,
    centers = km_result$centers,
    metrics = km_result$metrics,
    model = km_result$model
  )
}

#' Fit PAM for tidylearn models
#' @keywords internal
#' @noRd
tl_fit_pam <- function(data, formula = NULL, k = 3, ...) {
  tl_check_packages("cluster")

  # Extract variables to use
  if (!is.null(formula)) {
    vars <- get_formula_vars(formula, data)
    data_for_pam <- data[, vars, drop = FALSE]
  } else {
    data_for_pam <- data
  }

  # Fit PAM using tidy_pam
  pam_result <- tidy_pam(data_for_pam, k = k, ...)

  # Return in expected format
  list(
    clusters = pam_result$clusters,
    medoids = pam_result$medoids,
    silhouette_avg = pam_result$silhouette_avg,
    model = pam_result$model
  )
}

#' Fit CLARA for tidylearn models
#' @keywords internal
#' @noRd
tl_fit_clara <- function(data, formula = NULL, k = 3, ...) {
  tl_check_packages("cluster")

  # Extract variables to use
  if (!is.null(formula)) {
    vars <- get_formula_vars(formula, data)
    data_for_clara <- data[, vars, drop = FALSE]
  } else {
    data_for_clara <- data
  }

  # Fit CLARA using tidy_clara
  clara_result <- tidy_clara(data_for_clara, k = k, ...)

  # Return in expected format
  list(
    clusters = clara_result$clusters,
    medoids = clara_result$medoids,
    silhouette_avg = clara_result$silhouette_avg,
    model = clara_result$model
  )
}
