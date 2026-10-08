#' Tidy Hierarchical Clustering
#'
#' Performs hierarchical clustering with tidy output
#'
#' @param data A data frame, tibble, or dist object
#' @param method Agglomeration method: "ward.D2",
#'   "single", "complete", "average" (default),
#'   "mcquitty", "median", "centroid"
#' @param distance Distance metric if data is not a
#'   dist object (default: "euclidean")
#' @param cols Columns to include (tidy select).
#'   If NULL, uses all numeric columns, or every column for
#'   \code{distance = "gower"}.
#'
#' @return A list of class "tidy_hclust" containing:
#' \itemize{
#'   \item model: hclust object
#'   \item dist: distance matrix used
#'   \item method: linkage method used
#'   \item distance_method: distance metric used
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
tidy_hclust <- function(data, method = "average",
                        distance = "euclidean",
                        cols = NULL) {

  # Handle dist object
  if (inherits(data, "dist")) {
    dist_mat <- data
    data_orig <- NULL
  } else {
    # Gower handles factors, so with no selection it takes every column;
    # taking only the numeric ones would leave them out of the distance
    data_selected <- tl_select_columns(
      data, rlang::enquo(cols), all_columns = distance == "gower",
      numeric_only = distance != "gower", what = "Hierarchical clustering"
    )

    # Compute distance
    dist_mat <- tidy_dist(data_selected, method = distance)
    tl_check_complete_dist(dist_mat, "Hierarchical clustering")
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
#' @return A tibble with columns \code{.obs_id} (observation identifier) and
#'   \code{cluster} (integer cluster assignment).
#'
#' @examples
#' \donttest{
#' hc <- tidy_hclust(USArrests, method = "ward.D2")
#' clusters <- tidy_cutree(hc, k = 3)
#' }
#'
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

  # cutree() takes a vector of k or h and returns a matrix, one column per
  # cut, which would flatten into one row per observation per cut
  if (!is.null(k)) {
    tl_check_whole_number(k, "k", min = 1)
    clusters <- stats::cutree(hc_model, k = k)
  } else {
    if (!is.numeric(h) || length(h) != 1 || !is.finite(h)) {
      stop(
        "'h' must be a single number, the height to cut the tree at. ",
        "Got: ", paste(utils::head(as.character(h), 5), collapse = ", "),
        ".",
        call. = FALSE
      )
    }
    clusters <- stats::cutree(hc_model, h = h)
  }

  # Create tibble
  tibble::tibble(
    .obs_id = names(clusters) %||% as.character(seq_along(clusters)),
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
#' @return A tibble containing the original \code{data} with an additional
#'   \code{cluster} integer column indicating cluster assignments.
#'
#' @examples
#' \donttest{
#' hc <- tidy_hclust(USArrests, method = "ward.D2")
#' augmented <- augment_hclust(hc, USArrests, k = 3)
#' }
#'
#' @export
augment_hclust <- function(hclust_obj, data, k = NULL, h = NULL) {

  if (!inherits(hclust_obj, "tidy_hclust")) {
    stop("hclust_obj must be a tidy_hclust object")
  }

  cluster_assignments <- tidy_cutree(hclust_obj, k = k, h = h)

  # Attach by position: row_number() counts within each group of a grouped
  # tibble, so a join on it would attach other rows' clusters, and a join
  # pads or cuts short data of another length without a word
  n_tree <- nrow(cluster_assignments)
  if (nrow(data) != n_tree) {
    stop(
      "augment_hclust() adds one cluster per observation the tree was ",
      "built on: the tree has ", n_tree, ", but 'data' has ", nrow(data),
      " rows. Pass the data the tree was built from.",
      call. = FALSE
    )
  }

  dplyr::bind_cols(
    data, tibble::tibble(cluster = cluster_assignments$cluster)
  )
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
#' @return The \code{\link[stats]{hclust}} object, returned invisibly. The
#'   dendrogram is plotted as a side effect.
#'
#' @examples
#' \donttest{
#' hc <- tidy_hclust(USArrests, method = "ward.D2")
#' tidy_dendrogram(hc, k = 3)
#' }
#'
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
       main = paste0(
         "Hierarchical Clustering Dendrogram\n(",
         method_label, " linkage)"
       ),
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
#' @param method Character; "silhouette" (default) or "gap". The gap
#'   statistic resamples the observations' numeric columns, so it refuses a
#'   tree built from a dist object, and one built with
#'   \code{distance = "gower"} on non-numeric columns.
#' @param max_k Maximum number of clusters to test (default: 10)
#'
#' @return A list containing:
#' \itemize{
#'   \item optimal_k: the recommended number of clusters
#'   \item method: the evaluation method used
#'   \item values: numeric vector of evaluation scores (for silhouette)
#'   \item k_range: integer vector of k values tested (for silhouette)
#' }
#' If \code{method = "gap"}, returns a \code{tidy_gap} object instead.
#'
#' @examples
#' \donttest{
#' hc <- tidy_hclust(USArrests, method = "ward.D2")
#' opt <- optimal_hclust_k(hc, method = "silhouette", max_k = 6)
#' }
#'
#' @export
optimal_hclust_k <- function(hclust_obj, method = "silhouette", max_k = 10) {

  if (!inherits(hclust_obj, "tidy_hclust")) {
    stop("hclust_obj must be a tidy_hclust object")
  }

  dist_mat <- hclust_obj$dist
  hc_model <- hclust_obj$model

  if (method == "silhouette") {
    # silhouette() is defined for 2 to n - 1 clusters and returns NA
    # outside them, which would fail below as "incorrect number of
    # dimensions"
    tl_check_whole_number(
      max_k, "max_k", min = 2, max = length(hc_model$order) - 1
    )

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
    # clusGap() resamples the observations, so it needs them. A model
    # built straight from a dist object never kept any.
    if (is.null(hclust_obj$data)) {
      stop(
        "The gap statistic needs the observations, but this model was ",
        "fitted from a distance matrix and did not keep them. Refit with ",
        "tidy_hclust(data) to use method = \"gap\", or use ",
        "method = \"silhouette\", which works from the distances alone.",
        call. = FALSE
      )
    }

    # clusGap() draws its reference data uniformly over the range of each
    # numeric column, so the refit sees only those. For a Gower tree on
    # factors it would score clusterings the tree never made.
    non_numeric <- names(hclust_obj$data)[
      !vapply(hclust_obj$data, is.numeric, logical(1))
    ]
    if (identical(hclust_obj$distance_method, "gower") &&
          length(non_numeric) > 0) {
      stop(
        "The gap statistic draws its reference data uniformly over the ",
        "range of each numeric column, so it cannot evaluate a tree built ",
        "with distance = \"gower\" on the non-numeric column",
        if (length(non_numeric) > 1) "s", ": ",
        paste(non_numeric, collapse = ", "), ". Use method = \"silhouette\", ",
        "which works from the tree's own distances.",
        call. = FALSE
      )
    }

    # Read the model's own settings once. The refit below used
    # stats::dist()'s Euclidean default, so a model built with any other
    # distance was evaluated against clusterings it would never produce
    # -- a fault that could not show itself while the branch errored on
    # its first call.
    linkage <- hclust_obj$method
    distance <- hclust_obj$distance_method %||% "euclidean"

    gap_result <- tidy_gap_stat(
      hclust_obj$data,
      # clusGap() requires FUN to return a list with a `cluster` element.
      # cutree() returns a bare integer vector, so every call died with
      # "$ operator is invalid for atomic vectors".
      FUN_cluster = function(data, k) {
        hc_temp <- stats::hclust(
          tidy_dist(as.data.frame(data), method = distance),
          method = linkage
        )
        list(cluster = stats::cutree(hc_temp, k = k))
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
#' @return The input object \code{x}, returned invisibly.
#'
#' @examples
#' \donttest{
#' hc <- tidy_hclust(USArrests, method = "ward.D2")
#' print(hc)
#' }
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


#' Fit hierarchical clustering for tidylearn models
#'
#' tl_model()'s own \code{method} argument holds "hclust", so the linkage
#' has a name of its own, \code{hclust_method}, as the MDS variant has
#' \code{mds_method}.
#' @keywords internal
#' @noRd
tl_fit_hclust <- function(data, formula = NULL,
                          hclust_method = "average",
                          distance = "euclidean", ...) {
  linkages <- c("ward.D", "ward.D2", "single", "complete", "average",
                "mcquitty", "median", "centroid")
  if (!is.character(hclust_method) || length(hclust_method) != 1 ||
        !hclust_method %in% linkages) {
    stop(
      "'hclust_method' must be one of ",
      paste0("\"", linkages, "\"", collapse = ", "), ". Got: ",
      paste(utils::head(as.character(hclust_method), 5), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  # Without a formula, tidy_hclust() picks the columns its distance can use
  data <- tl_ungroup(data)
  if (!is.null(formula)) {
    vars <- tl_formula_columns(
      formula, data, "Hierarchical clustering",
      mixed_types = distance == "gower",
      alternative = "distance = \"gower\""
    )
    data <- data[, vars, drop = FALSE]
  }

  # Fit hierarchical clustering using tidy_hclust
  hc_result <- tidy_hclust(
    data, method = hclust_method,
    distance = distance, ...
  )

  # Return in expected format
  list(
    model = hc_result$model,
    dist = hc_result$dist,
    method = hc_result$method
  )
}
