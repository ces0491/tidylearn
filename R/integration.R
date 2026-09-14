#' Integration Functions: Combining Supervised and Unsupervised Learning
#'
#' These functions demonstrate the power of tidylearn's unified approach by
#' seamlessly integrating supervised and unsupervised learning techniques.

#' Feature Engineering via Dimensionality Reduction
#'
#' Use PCA, MDS, or other dimensionality reduction as a preprocessing step
#' for supervised learning. This can improve model performance
#' and interpretability.
#'
#' @param data A data frame
#' @param response Response variable name (will be preserved)
#' @param method Dimensionality reduction method: "pca", "mds"
#' @param n_components Number of components to retain
#' @param ... Additional arguments for the dimensionality reduction method
#' @return A list with components:
#'   \describe{
#'     \item{data}{The transformed data frame with reduced-dimension columns
#'       and the response variable (if provided).}
#'     \item{reduction_model}{The fitted tidylearn dimensionality reduction
#'       model.}
#'     \item{original_data}{The original input data frame.}
#'     \item{response}{The response variable name, or \code{NULL}.}
#'   }
#' @export
#' @examples
#' \donttest{
#' # Reduce dimensions before classification
#' reduced <- tl_reduce_dimensions(
#'   iris, response = "Species",
#'   method = "pca", n_components = 3
#' )
#' model <- tl_model(reduced$data, Species ~ ., method = "tree")
#' }
tl_reduce_dimensions <- function(data,
                                 response = NULL,
                                 method = "pca",
                                 n_components = NULL,
                                 ...) {
  # Separate response if provided
  if (!is.null(response)) {
    if (!response %in% names(data)) {
      stop(
        "Response variable '", response,
        "' not found in data", call. = FALSE
      )
    }
    response_data <- data[[response]]
    predictor_data <- data %>% dplyr::select(-dplyr::all_of(response))
  } else {
    response_data <- NULL
    predictor_data <- data
  }

  # Apply dimensionality reduction
  reduction_model <- tl_model(predictor_data, method = method, ...)

  # Record the component budget on the model itself. Trimming only the
  # returned $data leaves predict() projecting a test set onto every
  # component, so the test matrix is wider than the model trained on $data.
  reduction_model$spec$n_components <- n_components

  # Transform data
  if (method == "pca") {
    transformed <- reduction_model$fit$scores

    # Select components
    if (!is.null(n_components)) {
      pc_cols <- paste0("PC", seq_len(n_components))
      transformed <- transformed %>%
        dplyr::select(dplyr::all_of(pc_cols))
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
      transformed <- transformed %>%
        dplyr::select(dplyr::all_of(dim_cols))
    }

    # Add response back
    if (!is.null(response)) {
      transformed[[response]] <- response_data
    }
  }

  # Drop the .obs_id row identifier -- it is internal bookkeeping, not a
  # feature. Leaving it in place lets it reach downstream supervised models
  # as a high-cardinality predictor, which makes tree-based fits intractable.
  if (".obs_id" %in% names(transformed)) {
    transformed <- transformed[, names(transformed) != ".obs_id",
                               drop = FALSE]
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
#' @return The original data frame with an additional factor column named
#'   \code{cluster_<method>} containing cluster assignments. The fitted
#'   cluster model is stored as an attribute \code{"cluster_model"}.
#' @export
#' @examples
#' \donttest{
#' # Add cluster features before supervised learning
#' data_with_clusters <- tl_add_cluster_features(iris, response = "Species",
#'                                                 method = "kmeans", k = 3)
#' model <- tl_model(data_with_clusters, Species ~ ., method = "forest")
#' }
tl_add_cluster_features <- function(data,
                                    response = NULL,
                                    method = "kmeans",
                                    ...) {
  # Separate response if provided
  if (!is.null(response)) {
    if (!response %in% names(data)) {
      stop(
        "Response variable '", response,
        "' not found in data", call. = FALSE
      )
    }
    predictor_data <- data %>% dplyr::select(-dplyr::all_of(response))
  } else {
    predictor_data <- data
  }

  # hclust builds the whole tree and takes no k; k is where it is cut.
  # Passing it on to the fit failed with "unused argument (k = 4)", so
  # only the fallback k = 3 ever worked.
  dots <- list(...)
  if (method == "hclust") {
    k <- dots$k
    dots$k <- NULL
    if (is.null(k)) {
      warning("k not specified for hclust, using k=3")
      k <- 3
    }
  }

  # Perform clustering
  cluster_model <- do.call(
    tl_model, c(list(predictor_data, method = method), dots)
  )

  # Extract cluster assignments
  if (method %in% c("kmeans", "pam", "clara")) {
    clusters <- cluster_model$fit$clusters$cluster
  } else if (method == "hclust") {
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
#' Labels are propagated by majority vote within each cluster, so the
#' response must be categorical. Rows in a cluster that holds no labelled
#' observation have no label to take; they are left out of training, with
#' a warning giving the count.
#'
#' @param data A data frame
#' @param formula Model formula. The response must be a factor, character
#'   or logical column.
#' @param labeled_indices Indices of labeled observations
#' @param cluster_method Clustering method for label propagation
#' @param supervised_method Supervised learning method for the final
#'   model (default: \code{"tree"}, which handles any number of classes).
#'   \code{"logistic"} is binary-only and errors on a response with more
#'   than two levels.
#' @param ... Additional arguments
#' @return A tidylearn model object with additional class
#'   \code{"tidylearn_semisupervised"}, trained on pseudo-labeled data. The
#'   model includes a \code{semisupervised_info} element with
#'   \code{labeled_indices}, \code{cluster_model}, \code{label_mapping},
#'   and \code{n_unlabelled_dropped}, the number of rows left out because
#'   their cluster had no labelled observation.
#' @export
#' @examples
#' \donttest{
#' # Use only 10% of labels
#' labeled_idx <- sample(nrow(iris), size = 15)
#' model <- tl_semisupervised(iris, Species ~ ., labeled_indices = labeled_idx,
#'   cluster_method = "kmeans",
#'   supervised_method = "tree"
#' )
#' }
tl_semisupervised <- function(data, formula, labeled_indices,
                              cluster_method = "kmeans",
                              supervised_method = "tree", ...) {
  formula <- tl_as_formula(formula)

  # A logical selector would otherwise be matched as the positions 0 and 1
  if (is.logical(labeled_indices)) {
    labeled_indices <- which(labeled_indices)
  }

  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Labels are propagated by majority vote within a cluster, which has no
  # meaning for a continuous response -- and factor() below would quietly
  # turn a regression into a classification with one class per value.
  response <- data[[response_var]]
  if (!is.factor(response) && !is.character(response) &&
        !is.logical(response)) {
    stop(
      "tl_semisupervised() propagates class labels, so it needs a ",
      "categorical response.\n'", response_var, "' is ",
      class(response)[1], ". Convert it with factor() if its values ",
      "are classes.",
      call. = FALSE
    )
  }

  # Create training data with only labeled observations
  labeled_data <- data[labeled_indices, ]

  # Cluster the full dataset (excluding response)
  predictor_data <- data %>% dplyr::select(-dplyr::all_of(response_var))

  # Determine k from labeled data
  k <- length(unique(labeled_data[[response_var]]))

  # hclust takes no k -- the tree is cut afterwards -- and keeps no
  # cluster assignments of its own
  if (cluster_method == "hclust") {
    cluster_model <- tl_model(predictor_data, method = cluster_method, ...)
    clusters <- stats::cutree(cluster_model$fit$model, k = k)
  } else {
    cluster_model <- tl_model(predictor_data, method = cluster_method,
                              k = k, ...)
    clusters <- cluster_model$fit$clusters$cluster
  }

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
      final_label = dplyr::if_else(
        obs_id %in% labeled_indices,
        as.character(label), cluster_label
      )
    )

  # A cluster holding no labelled observation has no label to propagate.
  # Its rows used to become NA and vanish at fit time without a word, so
  # the model trained on a fraction of the data it appeared to use.
  unlabelled <- is.na(pseudo_labeled$final_label)
  if (any(unlabelled)) {
    empty_clusters <- sort(unique(pseudo_labeled$cluster[unlabelled]))
    warning(
      sum(unlabelled), " of ", nrow(data), " rows have no label and are ",
      "left out of training: they are unlabelled rows in cluster(s) ",
      paste(empty_clusters, collapse = ", "), ", which hold no labelled ",
      "observation, or labelled rows whose own label is missing. Label ",
      "observations from across the data to use them.",
      call. = FALSE
    )
  }

  # Create pseudo-labeled dataset
  data_pseudo <- data
  data_pseudo[[response_var]] <- as.factor(pseudo_labeled$final_label)
  data_pseudo <- data_pseudo[!unlabelled, , drop = FALSE]

  # Train supervised model on pseudo-labeled data
  model <- tl_model(data_pseudo, formula, method = supervised_method, ...)

  # Add metadata
  model$semisupervised_info <- list(
    labeled_indices = labeled_indices,
    cluster_model = cluster_model,
    label_mapping = label_mapping,
    n_unlabelled_dropped = sum(unlabelled)
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
#' @param anomaly_method Method for anomaly detection. Only "dbscan" is
#'   implemented; its noise points are the anomalies.
#' @param action Action to take: "remove", "flag", "downweight".
#'   \code{"downweight"} gives anomalies a case weight of 0.1, and needs a
#'   \code{supervised_method} that takes case weights: \code{"linear"},
#'   \code{"polynomial"}, \code{"logistic"}, \code{"tree"}, \code{"ridge"},
#'   \code{"lasso"}, \code{"elastic_net"} or \code{"forest"}. A forest reads
#'   them as sampling weights.
#' @param supervised_method Supervised learning method (default:
#'   \code{"tree"}, which handles both regression and classification with
#'   any number of classes). \code{"logistic"} is binary-only and errors
#'   on a response with more than two levels.
#' @param ... Additional arguments
#' @return A tidylearn model object with additional class
#'   \code{"tidylearn_anomaly_aware"}. The model includes an
#'   \code{anomaly_info} element with \code{anomaly_model},
#'   \code{is_anomaly} (logical vector), \code{n_anomalies}, and
#'   \code{action}.
#' @export
#' @examples
#' \donttest{
#' model <- tl_anomaly_aware(iris, Species ~ ., response = "Species",
#'                            anomaly_method = "dbscan", action = "flag")
#' }
tl_anomaly_aware <- function(data, formula, response,
                             anomaly_method = "dbscan",
                             action = "flag",
                             supervised_method = "tree",
                             ...) {
  formula <- tl_as_formula(formula)

  # An action outside the three used to fall through every branch and fail
  # later with "object 'model' not found"
  actions <- c("remove", "flag", "downweight")
  if (!is.character(action) || length(action) != 1L ||
        !action %in% actions) {
    stop("'action' must be one of ",
         paste0("\"", actions, "\"", collapse = ", "), ".", call. = FALSE)
  }
  if (!identical(anomaly_method, "dbscan")) {
    stop("'anomaly_method' must be \"dbscan\", the only method implemented.",
         call. = FALSE)
  }

  # Separate predictors for anomaly detection
  predictor_data <- data %>% dplyr::select(-dplyr::all_of(response))

  # Detect anomalies: DBSCAN's noise points
  anomaly_model <- tl_model(predictor_data, method = "dbscan", ...)
  is_anomaly <- anomaly_model$fit$clusters$cluster == 0

  # Take action based on anomalies
  if (action == "remove") {
    data_clean <- data[!is_anomaly, ]
    model <- tl_model(data_clean, formula, method = supervised_method)
    model$anomalies_removed <- sum(is_anomaly)
  } else if (action == "flag") {
    data_flagged <- data %>%
      dplyr::mutate(is_anomaly = is_anomaly)
    # Add the flag to the formula as given. Rebuilding it from all.vars()
    # put an excluded `- Sepal.Width` back in as a predictor and turned
    # poly(wt, 2) into wt.
    # The formula is expanded against the data first: update() cannot
    # expand `.` itself, and expanding against data_flagged would count
    # is_anomaly twice.
    expanded <- stats::formula(stats::terms(formula, data = data,
                                            simplify = TRUE))
    formula_updated <- stats::update(expanded, . ~ . + is_anomaly)
    model <- tl_model(data_flagged, formula_updated, method = supervised_method)
  } else if (action == "downweight") {
    # Only these backends apply case weights. Of the rest, boost and nn
    # error on them, xgboost warns that it does not recognise them, and
    # svm ignores them without a word. randomForest reads them as
    # sampling weights, so an anomaly is drawn into fewer bootstrap
    # samples rather than weighted in a loss.
    weighted_methods <- c("linear", "polynomial", "logistic", "tree",
                          "ridge", "lasso", "elastic_net", "forest")
    if (!supervised_method %in% weighted_methods) {
      stop(
        "action = \"downweight\" needs a method that takes case weights: ",
        paste0("\"", weighted_methods, "\"", collapse = ", "), ".\n'",
        supervised_method, "' does not, so use action = \"remove\" or ",
        "\"flag\" with it.",
        call. = FALSE
      )
    }

    # Create weights (anomalies get lower weight)
    weights <- ifelse(is_anomaly, 0.1, 1.0)

    # glm() reads binomial weights as trial counts and warns that 0.1 is
    # not a whole number of successes. Here they are case weights, which
    # the weighted likelihood handles correctly, so that warning is
    # expected every time and says nothing about this fit.
    model <- withCallingHandlers(
      tl_model(data, formula, method = supervised_method, weights = weights),
      warning = function(w) {
        message_text <- conditionMessage(w)
        if (grepl("non-integer #successes", message_text, fixed = TRUE)) {
          invokeRestart("muffleWarning")
        }
      }
    )
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
#' @param supervised_method Supervised learning method (default:
#'   \code{"tree"}, which handles both regression and classification).
#'   \code{"linear"} silently fits \code{lm()} to a factor response
#'   rather than refusing it, so it is not a safe default here.
#' @param ... Additional arguments
#' @return A list with class \code{"tidylearn_stratified"} containing:
#'   \describe{
#'     \item{cluster_model}{The fitted clustering model.}
#'     \item{supervised_models}{Named list of tidylearn models, one per
#'       cluster.}
#'     \item{formula}{The model formula.}
#'     \item{data}{The original training data.}
#'   }
#' @export
#' @examples
#' \donttest{
#' models <- tl_stratified_models(mtcars, mpg ~ ., cluster_method = "kmeans",
#'                                 k = 3, supervised_method = "linear")
#' }
tl_stratified_models <- function(data, formula, cluster_method = "kmeans",
                                 k = 3, supervised_method = "tree", ...) {
  formula <- tl_as_formula(formula)

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
#' @return A \link[tibble]{tibble} of the columns each cluster's model
#'   returns for the requested \code{type} -- \code{.pred} by default, one
#'   column per class for \code{type = "prob"} -- and a \code{.cluster}
#'   column with cluster assignments.
#' @examples
#' \donttest{
#' models <- tl_stratified_models(mtcars, mpg ~ .,
#'   cluster_method = "kmeans", k = 2, supervised_method = "linear")
#' preds <- predict(models)
#' }
#' @export
predict.tidylearn_stratified <- function(object, new_data = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- object$data
  }

  # Get response variable
  response_var <- all.vars(object$formula)[1]

  # Assign new data to clusters. any_of(): data to predict on need not
  # carry the response at all.
  predictor_data <- new_data %>% dplyr::select(-dplyr::any_of(response_var))
  new_clusters <- predict(object$cluster_model, new_data = predictor_data)

  # Predict each cluster's rows with its own model, keeping every column
  # the model returns. Reading back only .pred dropped the probability
  # columns type = "prob" returns, leaving a tibble of cluster ids.
  cluster_ids <- new_clusters$cluster
  predictions <- vector("list", nrow(new_data))
  for (cluster_id in unique(cluster_ids)) {
    rows <- which(cluster_ids == cluster_id)
    model_name <- paste0("cluster_", cluster_id)
    if (!model_name %in% names(object$supervised_models)) {
      next
    }
    pred <- predict(
      object$supervised_models[[model_name]],
      new_data = new_data[rows, , drop = FALSE], ...
    )
    for (j in seq_along(rows)) {
      predictions[[rows[j]]] <- pred[j, , drop = FALSE]
    }
  }

  # A row whose cluster has no model gets an all-NA prediction row
  template <- dplyr::bind_rows(predictions)
  if (ncol(template) == 0) {
    return(tibble::tibble(.pred = NA, .cluster = cluster_ids))
  }
  missing_rows <- vapply(predictions, is.null, logical(1))
  if (any(missing_rows)) {
    empty <- template[NA_integer_, , drop = FALSE]
    predictions[missing_rows] <- rep(list(empty), sum(missing_rows))
  }

  result <- dplyr::bind_rows(predictions)

  # Each cluster's model knows only the classes in its own rows, so binding
  # their predictions took the class levels, and the order of probability
  # columns, from whichever rows came first -- the second level, which
  # tidylearn treats as the positive class, changed with row order. A class
  # a cluster never saw came back as NA rather than probability 0. Both are
  # set from the classes in the full training data.
  response <- object$data[[response_var]]
  if (is.factor(response) || is.character(response)) {
    class_levels <- levels(factor(response))
    if (".pred" %in% names(result) &&
          (is.factor(result$.pred) || is.character(result$.pred))) {
      result$.pred <- factor(as.character(result$.pred),
                             levels = class_levels)
    }
    prob_cols <- intersect(class_levels, names(result))
    if (length(prob_cols) > 0) {
      # Only a row that was scored takes 0 for an unseen class; a row whose
      # predictors were missing keeps NA throughout
      scored <- rowSums(!is.na(as.data.frame(result[prob_cols]))) > 0
      for (cls in setdiff(class_levels, prob_cols)) {
        result[[cls]] <- NA_real_
      }
      for (cls in class_levels) {
        result[[cls]][scored & is.na(result[[cls]])] <- 0
      }
      result <- result[c(class_levels, setdiff(names(result), class_levels))]
    }
  }

  result$.cluster <- cluster_ids
  result
}
