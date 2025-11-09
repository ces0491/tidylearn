#' Tidy Multidimensional Scaling
#'
#' Unified interface for MDS methods with tidy output
#'
#' @param data A data frame, tibble, or distance matrix
#' @param method Character; "classical" (default), "metric", "nonmetric", "sammon", or "kruskal"
#' @param ndim Number of dimensions for output (default: 2)
#' @param distance Character; distance metric if data is not already a dist object (default: "euclidean")
#' @param ... Additional arguments passed to specific MDS functions
#'
#' @return A list of class "tidy_mds" containing:
#' \itemize{
#'   \item config: tibble of MDS configuration (coordinates)
#'   \item stress: goodness-of-fit measure (if applicable)
#'   \item method: character string of method used
#'   \item model: original model object
#' }
#'
#' @examples
#' # Classical MDS
#' mds_result <- tidy_mds(eurodist, method = "classical")
#'
#' # Non-metric MDS
#' mds_result <- tidy_mds(eurodist, method = "nonmetric")
#'
#' # From data frame
#' mds_result <- tidy_mds(USArrests, method = "sammon", distance = "euclidean")
#'
#' @export
tidy_mds <- function(data, method = "classical", ndim = 2, distance = "euclidean", ...) {

  # Convert to distance matrix if needed
  if (inherits(data, "dist")) {
    dist_mat <- data
  } else {
    data_matrix <- as.matrix(data %>% dplyr::select(where(is.numeric)))
    dist_mat <- stats::dist(data_matrix, method = distance)
  }

  # Call appropriate MDS method
  result <- switch(method,
    classical = tidy_mds_classical(dist_mat, ndim = ndim, ...),
    metric = tidy_mds_smacof(dist_mat, ndim = ndim, type = "ratio", ...),
    nonmetric = tidy_mds_smacof(dist_mat, ndim = ndim, type = "ordinal", ...),
    sammon = tidy_mds_sammon(dist_mat, ndim = ndim, ...),
    kruskal = tidy_mds_kruskal(dist_mat, ndim = ndim, ...),
    stop("method must be one of: classical, metric, nonmetric, sammon, kruskal")
  )

  result
}


#' Classical (Metric) MDS
#'
#' Performs classical multidimensional scaling using cmdscale()
#'
#' @param dist_mat A distance matrix (dist object)
#' @param ndim Number of dimensions (default: 2)
#' @param add_rownames Preserve row names from distance matrix (default: TRUE)
#'
#' @return A tidy_mds object
#' @export
tidy_mds_classical <- function(dist_mat, ndim = 2, add_rownames = TRUE) {

  if (!inherits(dist_mat, "dist")) {
    stop("dist_mat must be a dist object")
  }

  # Perform classical MDS
  mds_result <- stats::cmdscale(dist_mat, k = ndim, eig = TRUE)

  # Extract configuration
  config_matrix <- mds_result$points

  # Create tibble
  colnames(config_matrix) <- paste0("Dim", 1:ndim)

  if (add_rownames && !is.null(attr(dist_mat, "Labels"))) {
    config_tbl <- tibble::as_tibble(config_matrix, .name_repair = "minimal") %>%
      dplyr::mutate(.id = attr(dist_mat, "Labels"), .before = 1)
  } else {
    config_tbl <- tibble::as_tibble(config_matrix)
  }

  # Calculate GOF (goodness of fit)
  eigenvalues <- mds_result$eig
  total_var <- sum(abs(eigenvalues))
  retained_var <- sum(eigenvalues[1:ndim])
  gof <- retained_var / total_var

  result <- list(
    config = config_tbl,
    stress = NA,  # Classical MDS doesn't have stress
    gof = gof,
    eigenvalues = eigenvalues,
    method = "Classical MDS",
    model = mds_result
  )

  class(result) <- c("tidy_mds", "list")
  result
}


#' SMACOF MDS (Metric or Non-metric)
#'
#' Performs MDS using SMACOF algorithm from the smacof package
#'
#' @param dist_mat A distance matrix (dist object)
#' @param ndim Number of dimensions (default: 2)
#' @param type Character; "ratio" for metric, "ordinal" for non-metric (default: "ratio")
#' @param ... Additional arguments passed to smacof::mds()
#'
#' @return A tidy_mds object
#' @export
tidy_mds_smacof <- function(dist_mat, ndim = 2, type = "ratio", ...) {

  if (!inherits(dist_mat, "dist")) {
    stop("dist_mat must be a dist object")
  }

  # Perform SMACOF
  mds_result <- smacof::mds(dist_mat, ndim = ndim, type = type, ...)

  # Extract configuration
  config_matrix <- mds_result$conf
  colnames(config_matrix) <- paste0("Dim", 1:ndim)

  if (!is.null(attr(dist_mat, "Labels"))) {
    config_tbl <- tibble::as_tibble(config_matrix) %>%
      dplyr::mutate(.id = attr(dist_mat, "Labels"), .before = 1)
  } else {
    config_tbl <- tibble::as_tibble(config_matrix)
  }

  method_name <- ifelse(type == "ordinal", "Non-metric SMACOF", "Metric SMACOF")

  result <- list(
    config = config_tbl,
    stress = mds_result$stress,
    method = method_name,
    model = mds_result
  )

  class(result) <- c("tidy_mds", "list")
  result
}


#' Sammon Mapping
#'
#' Performs Sammon's non-linear mapping
#'
#' @param dist_mat A distance matrix (dist object)
#' @param ndim Number of dimensions (default: 2)
#' @param ... Additional arguments passed to MASS::sammon()
#'
#' @return A tidy_mds object
#' @export
tidy_mds_sammon <- function(dist_mat, ndim = 2, ...) {

  if (!inherits(dist_mat, "dist")) {
    stop("dist_mat must be a dist object")
  }

  # Perform Sammon mapping
  mds_result <- MASS::sammon(dist_mat, k = ndim, trace = FALSE, ...)

  # Extract configuration
  config_matrix <- mds_result$points
  colnames(config_matrix) <- paste0("Dim", 1:ndim)

  if (!is.null(attr(dist_mat, "Labels"))) {
    config_tbl <- tibble::as_tibble(config_matrix) %>%
      dplyr::mutate(.id = attr(dist_mat, "Labels"), .before = 1)
  } else {
    config_tbl <- tibble::as_tibble(config_matrix)
  }

  result <- list(
    config = config_tbl,
    stress = mds_result$stress,
    method = "Sammon Mapping",
    model = mds_result
  )

  class(result) <- c("tidy_mds", "list")
  result
}


#' Kruskal's Non-metric MDS
#'
#' Performs Kruskal's isoMDS
#'
#' @param dist_mat A distance matrix (dist object)
#' @param ndim Number of dimensions (default: 2)
#' @param ... Additional arguments passed to MASS::isoMDS()
#'
#' @return A tidy_mds object
#' @export
tidy_mds_kruskal <- function(dist_mat, ndim = 2, ...) {

  if (!inherits(dist_mat, "dist")) {
    stop("dist_mat must be a dist object")
  }

  # Perform isoMDS
  mds_result <- MASS::isoMDS(dist_mat, k = ndim, trace = FALSE, ...)

  # Extract configuration
  config_matrix <- mds_result$points
  colnames(config_matrix) <- paste0("Dim", 1:ndim)

  if (!is.null(attr(dist_mat, "Labels"))) {
    config_tbl <- tibble::as_tibble(config_matrix) %>%
      dplyr::mutate(.id = attr(dist_mat, "Labels"), .before = 1)
  } else {
    config_tbl <- tibble::as_tibble(config_matrix)
  }

  result <- list(
    config = config_tbl,
    stress = mds_result$stress,
    method = "Kruskal's isoMDS",
    model = mds_result
  )

  class(result) <- c("tidy_mds", "list")
  result
}


#' Plot MDS Configuration
#'
#' Visualize MDS results
#'
#' @param mds_obj A tidy_mds object
#' @param color_by Optional variable to color points by
#' @param label_points Logical; add point labels? (default: TRUE)
#' @param dim_x Which dimension for x-axis (default: 1)
#' @param dim_y Which dimension for y-axis (default: 2)
#'
#' @return A ggplot object
#' @export
plot_mds <- function(mds_obj, color_by = NULL, label_points = TRUE,
                     dim_x = 1, dim_y = 2) {

  if (!inherits(mds_obj, "tidy_mds")) {
    stop("mds_obj must be a tidy_mds object")
  }

  config <- mds_obj$config
  dim_x_name <- paste0("Dim", dim_x)
  dim_y_name <- paste0("Dim", dim_y)

  # Base plot
  p <- ggplot2::ggplot(config, ggplot2::aes(x = .data[[dim_x_name]], y = .data[[dim_y_name]]))

  # Add points
  if (!is.null(color_by)) {
    p <- p + ggplot2::geom_point(ggplot2::aes(color = .data[[color_by]]), size = 3, alpha = 0.7)
  } else {
    p <- p + ggplot2::geom_point(size = 3, alpha = 0.7, color = "steelblue")
  }

  # Add labels
  if (label_points && ".id" %in% names(config)) {
    p <- p + ggplot2::geom_text(ggplot2::aes(label = .id), vjust = -0.7, size = 3)
  }

  # Add title with stress if available
  title <- mds_obj$method
  if (!is.na(mds_obj$stress)) {
    title <- paste0(title, sprintf(" (Stress = %.4f)", mds_obj$stress))
  } else if (!is.null(mds_obj$gof)) {
    title <- paste0(title, sprintf(" (GOF = %.2f%%)", mds_obj$gof * 100))
  }

  p <- p +
    ggplot2::labs(
      title = title,
      x = paste("Dimension", dim_x),
      y = paste("Dimension", dim_y)
    ) +
    ggplot2::theme_minimal() +
    ggplot2::coord_equal()

  p
}


#' Print Method for tidy_mds
#'
#' @param x A tidy_mds object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_mds <- function(x, ...) {
  cat("Tidy MDS Analysis\n")
  cat("=================\n\n")
  cat("Method:", x$method, "\n")
  cat("Dimensions:", ncol(x$config) - 1, "\n")
  cat("Observations:", nrow(x$config), "\n")

  if (!is.na(x$stress)) {
    cat("Stress:", sprintf("%.4f", x$stress), "\n")
    cat("\nStress interpretation:\n")
    cat("  < 0.05: Excellent\n")
    cat("  < 0.10: Good\n")
    cat("  < 0.20: Acceptable\n")
    cat("  > 0.20: Poor\n")
  }

  if (!is.null(x$gof)) {
    cat("Goodness of Fit:", sprintf("%.2f%%", x$gof * 100), "\n")
  }

  cat("\nConfiguration (first 6 obs):\n")
  print(head(x$config))

  invisible(x)
}
