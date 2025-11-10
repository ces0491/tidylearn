#' Integration Functions: Combining Supervised and Unsupervised Learning
#'
#' These functions demonstrate the power of tidylearn's unified approach by
#' seamlessly integrating supervised and unsupervised learning techniques.

#' Feature Engineering via Dimensionality Reduction
#'
#' Use PCA, MDS, or other dimensionality reduction as a preprocessing step
#' for supervised learning. This can improve model performance and interpretability.
#'
#' @param data A data frame
#' @param response Response variable name (will be preserved)
#' @param method Dimensionality reduction method: "pca", "mds"
#' @param n_components Number of components to retain
#' @param ... Additional arguments for the dimensionality reduction method
#' @return A list containing the transformed data and the reduction model
#' @export
#' @examples
#' \dontrun{
#' # Reduce dimensions before classification
#' reduced <- tl_reduce_dimensions(iris, response = "Species", method = "pca", n_components = 3)
#' model <- tl_model(reduced$data, Species ~ ., method = "logistic")
#' }
tl_reduce_dimensions <- function(data, response = NULL, method = "pca", n_components = NULL, ...) {
  # Separate response if provided
  if (!is.null(response)) {
    if (!response %in% names(data)) {
      stop("Response variable '", response, "' not found in data", call. = FALSE)
    }
    response_data <- data[[response]]
    predictor_data <- data %>% dplyr::select(-dplyr::all_of(response))
  } else {
    response_data <- NULL
    predictor_data <- data
  }

  # Apply dimensionality reduction
  reduction_model <- tl_model(predictor_data, method = method, ...)

  # Transform data
  if (method == "pca") {
    transformed <- reduction_model$fit$scores

    # Select components
    if (!is.null(n_components)) {
      pc_cols <- paste0("PC", seq_len(n_components))
      transformed <- transformed %>% dplyr::select(.obs_id, dplyr::all_of(pc_cols))
    }

    # Add response back
    if (!is.null(response)) {
      transformed[[response]] <- response_data
    }

  } else if (method == "mds") {
    transformed <- reduction_model$fit$points

    # Select dimensions
    if (!is.null(n_components)) {
      dim_cols <- paste0("Dim", seq_len(n_components))
      transformed <- transformed %>% dplyr::select(.id, dplyr::all_of(dim_cols))
    }

    # Add response back
    if (!is.null(response)) {
      transformed[[response]] <- response_data
    }
  }

  list(
    data = transformed,
    reduction_model = reduction_model,
    original_data = data,
    response = response
  )
}

#' Cluster-Based Features
#'
#' Add cluster assignments as features for supervised learning.
#' This semi-supervised approach can capture non-linear patterns.
#'
#' @param data A data frame
#' @param response Response variable name (will be excluded from clustering)
#' @param method Clustering method: "kmeans", "pam", "hclust", "dbscan"
#' @param ... Additional arguments for clustering
#' @return Original data with cluster assignment column(s) added
#' @export
#' @examples
#' \dontrun{
#' # Add cluster features before supervised learning
#' data_with_clusters <- tl_add_cluster_features(iris, response = "Species",
#'                                                 method = "kmeans", k = 3)
#' model <- tl_model(data_with_clusters, Species ~ ., method = "forest")
#' }
tl_add_cluster_features <- function(data, response = NULL, method = "kmeans", ...) {
  # Separate response if provided
  if (!is.null(response)) {
    if (!response %in% names(data)) {
      stop("Response variable '", response, "' not found in data", call. = FALSE)
    }
    predictor_data <- data %>% dplyr::select(-dplyr::all_of(response))
  } else {
    predictor_data <- data
  }

  # Perform clustering
  cluster_model <- tl_model(predictor_data, method = method, ...)

  # Extract cluster assignments
  if (method %in% c("kmeans", "pam", "clara")) {
    clusters <- cluster_model$fit$clusters$cluster
  } else if (method == "hclust") {
    # For hclust, we need k to cut the tree
    if (!"k" %in% names(list(...))) {
      warning("k not specified for hclust, using k=3")
      k <- 3
    } else {
      k <- list(...)$k
    }
    clusters <- stats::cutree(cluster_model$fit$model, k = k)
  } else if (method == "dbscan") {
    clusters <- cluster_model$fit$clusters$cluster
  }

  # Add to original data
  data_augmented <- data %>%
    dplyr::mutate(
      !!paste0("cluster_", method) := as.factor(clusters)
    )

  attr(data_augmented, "cluster_model") <- cluster_model
  data_augmented
}

#' Semi-Supervised Learning via Clustering
#'
#' Train a supervised model with limited labels by first clustering the data
#' and propagating labels within clusters.
#'
#' @param data A data frame
#' @param formula Model formula
#' @param labeled_indices Indices of labeled observations
#' @param cluster_method Clustering method for label propagation
#' @param supervised_method Supervised learning method for final model
#' @param ... Additional arguments
#' @return A tidylearn model trained on pseudo-labeled data
#' @export
#' @examples
#' \dontrun{
#' # Use only 10% of labels
#' labeled_idx <- sample(nrow(iris), size = 15)
#' model <- tl_semisupervised(iris, Species ~ ., labeled_indices = labeled_idx,
#'                            cluster_method = "kmeans", supervised_method = "logistic")
#' }
tl_semisupervised <- function(data, formula, labeled_indices,
                              cluster_method = "kmeans",
                              supervised_method = "logistic", ...) {
  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Create training data with only labeled observations
  labeled_data <- data[labeled_indices, ]
  unlabeled_data <- data[-labeled_indices, ]

  # Cluster the full dataset (excluding response)
  predictor_data <- data %>% dplyr::select(-dplyr::all_of(response_var))

  # Determine k from labeled data
  k <- length(unique(labeled_data[[response_var]]))

  cluster_model <- tl_model(predictor_data, method = cluster_method, k = k, ...)

  # Get cluster assignments
  clusters <- cluster_model$fit$clusters$cluster

  # Propagate labels within clusters
  cluster_labels <- tibble::tibble(
    obs_id = seq_len(nrow(data)),
    cluster = clusters,
    label = data[[response_var]]
  )

  # For each cluster, find the most common label from labeled data
  label_mapping <- cluster_labels %>%
    dplyr::filter(obs_id %in% labeled_indices) %>%
    dplyr::group_by(cluster) %>%
    dplyr::summarize(
      cluster_label = names(which.max(table(label))),
      .groups = "drop"
    )

  # Assign pseudo-labels to unlabeled data
  pseudo_labeled <- cluster_labels %>%
    dplyr::left_join(label_mapping, by = "cluster") %>%
    dplyr::mutate(
      final_label = dplyr::if_else(obs_id %in% labeled_indices, as.character(label), cluster_label)
    )

  # Create pseudo-labeled dataset
  data_pseudo <- data
  data_pseudo[[response_var]] <- as.factor(pseudo_labeled$final_label)

  # Train supervised model on pseudo-labeled data
  model <- tl_model(data_pseudo, formula, method = supervised_method, ...)

  # Add metadata
  model$semisupervised_info <- list(
    labeled_indices = labeled_indices,
    cluster_model = cluster_model,
    label_mapping = label_mapping
  )

  class(model) <- c("tidylearn_semisupervised", class(model))
  model
}

#' Anomaly-Aware Supervised Learning
#'
#' Detect outliers using DBSCAN or other methods, then optionally
#' remove them or down-weight them before supervised learning.
#'
#' @param data A data frame
#' @param formula Model formula
#' @param response Response variable name
#' @param anomaly_method Method for anomaly detection: "dbscan", "isolation_forest"
#' @param action Action to take: "remove", "flag", "downweight"
#' @param supervised_method Supervised learning method
#' @param ... Additional arguments
#' @return A tidylearn model or list with model and anomaly info
#' @export
#' @examples
#' \dontrun{
#' model <- tl_anomaly_aware(iris, Species ~ ., response = "Species",
#'                            anomaly_method = "dbscan", action = "flag")
#' }
tl_anomaly_aware <- function(data, formula, response,
                              anomaly_method = "dbscan",
                              action = "flag",
                              supervised_method = "logistic", ...) {
  # Separate predictors for anomaly detection
  predictor_data <- data %>% dplyr::select(-dplyr::all_of(response))

  # Detect anomalies
  if (anomaly_method == "dbscan") {
    anomaly_model <- tl_model(predictor_data, method = "dbscan", ...)
    is_anomaly <- anomaly_model$fit$clusters$cluster == 0  # DBSCAN noise points
  } else {
    stop("Unsupported anomaly method: ", anomaly_method, call. = FALSE)
  }

  # Take action based on anomalies
  if (action == "remove") {
    data_clean <- data[!is_anomaly, ]
    model <- tl_model(data_clean, formula, method = supervised_method)
    model$anomalies_removed <- sum(is_anomaly)
  } else if (action == "flag") {
    data_flagged <- data %>%
      dplyr::mutate(is_anomaly = is_anomaly)
    # Include anomaly flag as a feature - manually construct formula
    response_var <- all.vars(formula)[1]
    all_vars <- all.vars(formula)
    predictor_vars <- all_vars[-1]

    if (length(predictor_vars) == 1 && predictor_vars[1] == ".") {
      # Formula was response ~ .
      formula_updated <- as.formula(paste(response_var, "~ . + is_anomaly"))
    } else {
      # Explicit predictors
      formula_updated <- as.formula(paste(response_var, "~", paste(c(predictor_vars, "is_anomaly"), collapse = " + ")))
    }
    model <- tl_model(data_flagged, formula_updated, method = supervised_method)
  } else if (action == "downweight") {
    # Create weights (anomalies get lower weight)
    weights <- ifelse(is_anomaly, 0.1, 1.0)
    model <- tl_model(data, formula, method = supervised_method, weights = weights)
  }

  # Add anomaly detection info
  model$anomaly_info <- list(
    anomaly_model = anomaly_model,
    is_anomaly = is_anomaly,
    n_anomalies = sum(is_anomaly),
    action = action
  )

  class(model) <- c("tidylearn_anomaly_aware", class(model))
  model
}

#' Stratified Features via Clustering
#'
#' Create cluster-specific supervised models for heterogeneous data
#'
#' @param data A data frame
#' @param formula Model formula
#' @param cluster_method Clustering method
#' @param k Number of clusters
#' @param supervised_method Supervised learning method
#' @param ... Additional arguments
#' @return A list of models (one per cluster) plus cluster assignments
#' @export
#' @examples
#' \dontrun{
#' models <- tl_stratified_models(mtcars, mpg ~ ., cluster_method = "kmeans",
#'                                 k = 3, supervised_method = "linear")
#' }
tl_stratified_models <- function(data, formula, cluster_method = "kmeans",
                                 k = 3, supervised_method = "linear", ...) {
  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Cluster the predictors
  predictor_data <- data %>% dplyr::select(-dplyr::all_of(response_var))
  cluster_model <- tl_model(predictor_data, method = cluster_method, k = k, ...)

  # Get cluster assignments
  clusters <- cluster_model$fit$clusters$cluster

  # Train a model for each cluster
  cluster_models <- list()
  for (i in seq_len(k)) {
    cluster_data <- data[clusters == i, ]
    if (nrow(cluster_data) > 0) {
      cluster_models[[paste0("cluster_", i)]] <- tl_model(
        cluster_data, formula, method = supervised_method, ...
      )
    }
  }

  # Return stratified model object
  structure(
    list(
      cluster_model = cluster_model,
      supervised_models = cluster_models,
      formula = formula,
      data = data
    ),
    class = c("tidylearn_stratified", "list")
  )
}

#' Predict from stratified models
#' @param object A tidylearn_stratified model object
#' @param new_data New data for predictions
#' @param ... Additional arguments
#' @export
predict.tidylearn_stratified <- function(object, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- object$data
  }

  # Get response variable
  response_var <- all.vars(object$formula)[1]

  # Assign new data to clusters
  predictor_data <- new_data %>% dplyr::select(-dplyr::all_of(response_var))
  new_clusters <- predict(object$cluster_model, new_data = predictor_data)

  # Predict from appropriate cluster model
  predictions <- vector("list", nrow(new_data))
  for (i in seq_len(nrow(new_data))) {
    cluster_id <- new_clusters$cluster[i]
    model_name <- paste0("cluster_", cluster_id)

    if (model_name %in% names(object$supervised_models)) {
      pred <- predict(object$supervised_models[[model_name]],
                     new_data = new_data[i, , drop = FALSE], ...)
      predictions[[i]] <- pred$.pred[1]
    } else {
      predictions[[i]] <- NA
    }
  }

  tibble::tibble(
    .pred = unlist(predictions),
    .cluster = new_clusters$cluster
  )
}
