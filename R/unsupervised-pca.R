#' Tidy Principal Component Analysis
#'
#' Performs PCA on a dataset using tidyverse principles. Returns a tidy list
#' containing scores, loadings, variance explained, and the original model.
#'
#' @param data A data frame or tibble
#' @param cols Columns to include in PCA (tidy select syntax). If NULL, uses all numeric columns.
#' @param scale Logical; should variables be scaled to unit variance? Default TRUE.
#' @param center Logical; should variables be centered? Default TRUE.
#' @param method Character; "prcomp" (default, recommended) or "princomp"
#'
#' @return A list of class "tidy_pca" containing:
#' \itemize{
#'   \item scores: tibble of PC scores with observation identifiers
#'   \item loadings: tibble of variable loadings in long format
#'   \item variance: tibble of variance explained by each PC
#'   \item model: the original prcomp/princomp object
#'   \item settings: list of scale, center, method used
#' }
#'
#' @examples
#' # Basic PCA
#' pca_result <- tidy_pca(USArrests)
#'
#' # PCA on selected columns
#' pca_result <- tidy_pca(mtcars, cols = c(mpg, hp, wt, qsec))
#'
#' # Access components
#' pca_result$scores
#' pca_result$loadings
#' pca_result$variance
#'
#' @export
tidy_pca <- function(data, cols = NULL, scale = TRUE, center = TRUE, method = "prcomp") {

  # Convert to data frame if needed
  data <- as.data.frame(data)

  # Select columns
  if (!is.null(cols)) {
    cols_enquo <- rlang::enquo(cols)
    data_selected <- data %>% dplyr::select(!!cols_enquo)
  } else {
    # Select only numeric columns
    data_selected <- data %>% dplyr::select(where(is.numeric))
  }

  # Preserve row names/identifiers
  if (!is.null(rownames(data))) {
    obs_id <- rownames(data)
  } else {
    obs_id <- paste0("obs_", seq_len(nrow(data)))
  }

  # Perform PCA
  if (method == "prcomp") {
    pca_model <- stats::prcomp(data_selected, scale. = scale, center = center)
    scores_matrix <- pca_model$x
    loadings_matrix <- pca_model$rotation
    sdev <- pca_model$sdev
  } else if (method == "princomp") {
    pca_model <- stats::princomp(data_selected, cor = scale, scores = TRUE)
    scores_matrix <- pca_model$scores
    loadings_matrix <- pca_model$loadings
    sdev <- pca_model$sdev
  } else {
    stop("method must be 'prcomp' or 'princomp'")
  }

  # Create tidy scores tibble
  scores_tbl <- tibble::as_tibble(scores_matrix) %>%
    dplyr::mutate(.obs_id = obs_id, .before = 1)

  # Create tidy loadings tibble (long format)
  n_pcs <- ncol(loadings_matrix)
  loadings_tbl <- tibble::as_tibble(loadings_matrix, rownames = "variable") %>%
    tidyr::pivot_longer(
      cols = -variable,
      names_to = "component",
      values_to = "loading"
    )

  # Calculate variance explained
  variance <- sdev^2
  prop_var <- variance / sum(variance)
  cumsum_var <- cumsum(prop_var)

  variance_tbl <- tibble::tibble(
    component = colnames(scores_matrix),
    sdev = sdev,
    variance = variance,
    prop_variance = prop_var,
    cum_variance = cumsum_var
  )

  # Return tidy PCA object
  result <- list(
    scores = scores_tbl,
    loadings = loadings_tbl,
    variance = variance_tbl,
    model = pca_model,
    settings = list(scale = scale, center = center, method = method)
  )

  class(result) <- c("tidy_pca", "list")
  return(result)
}


#' Get PCA Loadings in Wide Format
#'
#' @param pca_obj A tidy_pca object
#' @param n_components Number of components to include (default: all)
#'
#' @return A tibble with loadings in wide format
#' @export
get_pca_loadings <- function(pca_obj, n_components = NULL) {
  if (!inherits(pca_obj, "tidy_pca")) {
    stop("pca_obj must be a tidy_pca object")
  }

  loadings <- pca_obj$loadings

  if (!is.null(n_components)) {
    components_to_keep <- unique(loadings$component)[1:n_components]
    loadings <- loadings %>%
      dplyr::filter(component %in% components_to_keep)
  }

  loadings %>%
    tidyr::pivot_wider(
      names_from = component,
      values_from = loading
    )
}


#' Get Variance Explained Summary
#'
#' @param pca_obj A tidy_pca object
#'
#' @return A tibble with variance statistics
#' @export
get_pca_variance <- function(pca_obj) {
  if (!inherits(pca_obj, "tidy_pca")) {
    stop("pca_obj must be a tidy_pca object")
  }

  pca_obj$variance
}


#' Augment Original Data with PCA Scores
#'
#' Add PC scores to the original dataset
#'
#' @param pca_obj A tidy_pca object
#' @param data Original data frame
#' @param n_components Number of PCs to add (default: all)
#'
#' @return Original data with PC scores added
#' @export
augment_pca <- function(pca_obj, data, n_components = NULL) {
  if (!inherits(pca_obj, "tidy_pca")) {
    stop("pca_obj must be a tidy_pca object")
  }

  scores <- pca_obj$scores %>% dplyr::select(-.obs_id)

  if (!is.null(n_components)) {
    scores <- scores %>% dplyr::select(1:n_components)
  }

  dplyr::bind_cols(data, scores)
}


#' Create PCA Scree Plot
#'
#' Visualize variance explained by each principal component
#'
#' @param pca_obj A tidy_pca object
#' @param type Character; "variance" or "proportion" (default)
#' @param add_line Logical; add horizontal line at eigenvalue = 1? (for Kaiser criterion)
#'
#' @return A ggplot object
#' @export
tidy_pca_screeplot <- function(pca_obj, type = "proportion", add_line = TRUE) {
  if (!inherits(pca_obj, "tidy_pca")) {
    stop("pca_obj must be a tidy_pca object")
  }

  var_data <- pca_obj$variance

  if (type == "proportion") {
    p <- ggplot2::ggplot(var_data, ggplot2::aes(x = seq_along(component), y = prop_variance)) +
      ggplot2::geom_line(color = "steelblue", size = 1) +
      ggplot2::geom_point(color = "steelblue", size = 3) +
      ggplot2::labs(
        title = "Scree Plot - Proportion of Variance Explained",
        x = "Principal Component",
        y = "Proportion of Variance"
      )
  } else {
    p <- ggplot2::ggplot(var_data, ggplot2::aes(x = seq_along(component), y = variance)) +
      ggplot2::geom_line(color = "steelblue", size = 1) +
      ggplot2::geom_point(color = "steelblue", size = 3) +
      ggplot2::labs(
        title = "Scree Plot - Variance (Eigenvalues)",
        x = "Principal Component",
        y = "Variance (Eigenvalue)"
      )

    if (add_line && pca_obj$settings$scale) {
      p <- p + ggplot2::geom_hline(yintercept = 1, linetype = "dashed", color = "red")
    }
  }

  p + ggplot2::theme_minimal()
}


#' Create PCA Biplot
#'
#' Visualize both observations and variables in PC space
#'
#' @param pca_obj A tidy_pca object
#' @param pc_x Principal component for x-axis (default: 1)
#' @param pc_y Principal component for y-axis (default: 2)
#' @param color_by Optional column name to color points by
#' @param arrow_scale Scaling factor for variable arrows (default: 1)
#' @param label_obs Logical; label observations? (default: FALSE)
#' @param label_vars Logical; label variables? (default: TRUE)
#'
#' @return A ggplot object
#' @export
tidy_pca_biplot <- function(pca_obj, pc_x = 1, pc_y = 2, color_by = NULL,
                             arrow_scale = 1, label_obs = FALSE, label_vars = TRUE) {
  if (!inherits(pca_obj, "tidy_pca")) {
    stop("pca_obj must be a tidy_pca object")
  }

  # Get scores
  scores <- pca_obj$scores
  pc_x_name <- paste0("PC", pc_x)
  pc_y_name <- paste0("PC", pc_y)

  if (!pc_x_name %in% names(scores) || !pc_y_name %in% names(scores)) {
    stop("Requested PCs not available")
  }

  # Get loadings for these PCs
  loadings_wide <- pca_obj$loadings %>%
    dplyr::filter(component %in% c(pc_x_name, pc_y_name)) %>%
    tidyr::pivot_wider(names_from = component, values_from = loading)

  # Scale factor for arrows
  score_range <- max(abs(c(scores[[pc_x_name]], scores[[pc_y_name]])))
  loading_range <- max(abs(c(loadings_wide[[pc_x_name]], loadings_wide[[pc_y_name]])))
  arrow_scale_factor <- (score_range / loading_range) * 0.8 * arrow_scale

  loadings_wide <- loadings_wide %>%
    dplyr::mutate(
      x_end = .data[[pc_x_name]] * arrow_scale_factor,
      y_end = .data[[pc_y_name]] * arrow_scale_factor
    )

  # Get variance explained
  var_exp <- pca_obj$variance
  var_x <- var_exp$prop_variance[pc_x] * 100
  var_y <- var_exp$prop_variance[pc_y] * 100

  # Create base plot
  p <- ggplot2::ggplot()

  # Add observation points
  if (!is.null(color_by)) {
    p <- p + ggplot2::geom_point(
      data = scores,
      ggplot2::aes(x = .data[[pc_x_name]], y = .data[[pc_y_name]], color = .data[[color_by]]),
      alpha = 0.7, size = 2
    )
  } else {
    p <- p + ggplot2::geom_point(
      data = scores,
      ggplot2::aes(x = .data[[pc_x_name]], y = .data[[pc_y_name]]),
      alpha = 0.7, size = 2, color = "steelblue"
    )
  }

  # Add observation labels if requested
  if (label_obs) {
    p <- p + ggplot2::geom_text(
      data = scores,
      ggplot2::aes(x = .data[[pc_x_name]], y = .data[[pc_y_name]], label = .obs_id),
      size = 2.5, vjust = -0.5
    )
  }

  # Add variable arrows
  p <- p + ggplot2::geom_segment(
    data = loadings_wide,
    ggplot2::aes(x = 0, y = 0, xend = x_end, yend = y_end),
    arrow = ggplot2::arrow(length = ggplot2::unit(0.2, "cm")),
    color = "red", size = 0.8, alpha = 0.7
  )

  # Add variable labels if requested
  if (label_vars) {
    p <- p + ggplot2::geom_text(
      data = loadings_wide,
      ggplot2::aes(x = x_end, y = y_end, label = variable),
      color = "red", size = 3, fontface = "bold", vjust = -0.5
    )
  }

  # Add labels and theme
  p <- p +
    ggplot2::labs(
      title = "PCA Biplot",
      x = sprintf("%s (%.1f%% variance)", pc_x_name, var_x),
      y = sprintf("%s (%.1f%% variance)", pc_y_name, var_y)
    ) +
    ggplot2::theme_minimal() +
    ggplot2::coord_equal()

  p
}


#' Print Method for tidy_pca
#'
#' @param x A tidy_pca object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_pca <- function(x, ...) {
  cat("Tidy PCA Analysis\n")
  cat("=================\n\n")
  cat("Number of observations:", nrow(x$scores), "\n")
  cat("Number of variables:", length(unique(x$loadings$variable)), "\n")
  cat("Number of components:", nrow(x$variance), "\n")
  cat("Settings: scale =", x$settings$scale, ", center =", x$settings$center, "\n\n")

  cat("Variance Explained:\n")
  print(x$variance, n = 5)

  cat("\nAccess components with:\n")
  cat("  $scores    - PC scores for each observation\n")
  cat("  $loadings  - Variable loadings on each PC\n")
  cat("  $variance  - Variance explained by each PC\n")
  cat("  $model     - Original PCA model object\n")

  invisible(x)
}
