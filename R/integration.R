#' Integration Functions: Combining Supervised and Unsupervised Learning
#'
#' These functions demonstrate the power of tidylearn's unified approach by
#' seamlessly integrating supervised and unsupervised learning techniques.
#' @noRd
NULL

#' Feature Engineering via Dimensionality Reduction
#'
#' Use PCA, MDS, or other dimensionality reduction as a preprocessing step
#' for supervised learning. This can improve model performance
#' and interpretability.
#'
#' @param data A data frame
#' @param response Response variable name (will be preserved)
#' @param method Dimensionality reduction method: "pca" or "mds"
#' @param n_components Number of components to retain, at most the number
#'   the method computes: one per numeric column for PCA, and \code{k}
#'   (default 2, passed through \code{...}) for MDS. NULL keeps them all.
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
  # Only these two produce component scores. Any other method fitted, left
  # nothing transformed, and failed with "object 'transformed' not found".
  if (!is.character(method) || length(method) != 1L ||
        !method %in% c("pca", "mds")) {
    stop(
      "'method' must be \"pca\" or \"mds\"; got ",
      tl_describe_value(method), ".",
      call. = FALSE
    )
  }
  if (!is.null(n_components) &&
        (!is.numeric(n_components) || length(n_components) != 1L ||
           is.na(n_components) || n_components < 1 ||
           n_components != round(n_components))) {
    stop(
      "'n_components' must be a single whole number of at least 1, or ",
      "NULL to keep every component; got ", tl_describe_value(n_components),
      ".",
      call. = FALSE
    )
  }

  # Separate response if provided
  if (!is.null(response)) {
    if (!response %in% names(data)) {
      stop(
        "Response variable '", response,
        "' not found in data", call. = FALSE
      )
    }
    response_data <- data[[response]]
    predictor_data <- data |> dplyr::select(-dplyr::all_of(response))
  } else {
    response_data <- NULL
    predictor_data <- data
  }

  # Apply dimensionality reduction
  reduction_model <- tl_model(predictor_data, method = method, ...)

  # Only the fit knows how many components there are. Asking for more
  # failed in the column selection below with "Elements PC5 and PC6 don't
  # exist".
  if (!is.null(n_components)) {
    available <- if (method == "pca") {
      ncol(reduction_model$fit$model$rotation)
    } else {
      sum(grepl("^Dim[0-9]+$", names(reduction_model$fit$points)))
    }
    if (n_components > available) {
      stop(
        "'n_components' is ", n_components, ", but the ", toupper(method),
        " has ", available,
        if (method == "pca") {
          " components, one per numeric column."
        } else {
          paste0(" dimensions. Pass k = ", n_components,
                 " to compute that many.")
        },
        call. = FALSE
      )
    }
  }

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
      transformed <- transformed |>
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
      transformed <- transformed |>
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
    predictor_data <- data |> dplyr::select(-dplyr::all_of(response))
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
  data_augmented <- data |>
    dplyr::mutate(
      !!paste0("cluster_", method) := as.factor(clusters)
    )

  attr(data_augmented, "cluster_model") <- cluster_model
  data_augmented
}

#' The columns a formula uses as predictors
#'
#' The unsupervised step of a combined workflow -- the clustering, the
#' projection, the anomaly detection -- has to see the predictors the
#' supervised formula names and no others. Taking every column but the
#' response let a column the formula excludes (\code{y ~ . - id}) shape the
#' clusters, and through them the labels, the features or the rows the
#' model was fitted on. A row id on data sorted by class carries the class.
#'
#' @param formula A two-sided formula
#' @param data The data it is evaluated against, to expand \code{.}
#' @return A character vector of column names
#' @keywords internal
#' @noRd
tl_formula_predictors <- function(formula, data) {
  setdiff(
    intersect(get_formula_vars(formula, data), names(data)),
    all.vars(formula)[1]
  )
}

#' Check a clustering method for the k-cluster workflows
#'
#' @param cluster_method The \code{cluster_method} argument
#' @return \code{TRUE}, invisibly
#' @keywords internal
#' @noRd
tl_check_cluster_method <- function(cluster_method) {
  supported <- c("kmeans", "pam", "clara", "hclust")
  if (!is.character(cluster_method) || length(cluster_method) != 1L ||
        !cluster_method %in% supported) {
    stop(
      "'cluster_method' must be one of ",
      paste0("\"", supported, "\"", collapse = ", "), "; got ",
      tl_describe_value(cluster_method), ".",
      if (identical(cluster_method, "dbscan")) {
        paste0(" These workflows need k clusters, and dbscan chooses its ",
               "own number of clusters.")
      } else {
        ""
      },
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Check the settings meant for a workflow's clustering step
#'
#' @param cluster_args The \code{cluster_args} argument
#' @param k_owner What sets k instead, for the message when it is given here
#' @return \code{TRUE}, invisibly
#' @keywords internal
#' @noRd
tl_check_cluster_args <- function(cluster_args, k_owner) {
  if (!is.list(cluster_args) ||
        (length(cluster_args) > 0 && !all(nzchar(names2(cluster_args))))) {
    stop(
      "'cluster_args' must be a named list of arguments for the clustering ",
      "step, e.g. list(nstart = 5).",
      call. = FALSE
    )
  }
  if ("k" %in% names(cluster_args)) {
    stop("'cluster_args' cannot set k. ", k_owner, call. = FALSE)
  }
  invisible(TRUE)
}

#' Cluster the rows into k groups
#'
#' hclust builds the whole tree and takes no k, so it is cut at k
#' afterwards; the other methods take k at fit time. Passing k to hclust
#' failed with "unused argument (k = 2)".
#'
#' @param predictor_data The columns to cluster on
#' @param method A method accepted by \code{tl_check_cluster_method()}
#' @param k The number of clusters
#' @param cluster_args Further arguments for the clustering fit
#' @return A list: \code{model}, the fitted clustering, and
#'   \code{clusters}, an integer cluster per row
#' @keywords internal
#' @noRd
tl_assign_clusters <- function(predictor_data, method, k, cluster_args) {
  if (method == "hclust") {
    model <- do.call(
      tl_model, c(list(predictor_data, method = method), cluster_args)
    )
    clusters <- stats::cutree(model$fit$model, k = k)
  } else {
    model <- do.call(
      tl_model, c(list(predictor_data, method = method, k = k), cluster_args)
    )
    clusters <- model$fit$clusters$cluster
  }
  list(model = model, clusters = as.integer(unname(clusters)))
}

#' Semi-Supervised Learning via Clustering
#'
#' Train a supervised model with limited labels by first clustering the data
#' and propagating labels within clusters.
#'
#' Labels are propagated by majority vote within each cluster, so the
#' response must be categorical. The rows are clustered on the predictors
#' the formula names, into as many clusters as the labelled rows hold
#' classes. A labelled row whose label is missing takes no part in the
#' vote. Rows in a cluster where no labelled observation carries a label
#' have no label to take, and labelled rows whose own label is missing
#' have none either; both are left out of training, with a warning giving
#' the counts. The pseudo-labels keep the response's level order, so the
#' second level stays the positive class.
#'
#' @param data A data frame
#' @param formula Model formula. The response must be a factor, character
#'   or logical column.
#' @param labeled_indices Indices of labeled observations
#' @param cluster_method Clustering method for label propagation:
#'   \code{"kmeans"} (default), \code{"pam"}, \code{"clara"} or
#'   \code{"hclust"}, whose tree is cut at k
#' @param supervised_method Supervised learning method for the final
#'   model (default: \code{"tree"}, which handles any number of classes).
#'   \code{"logistic"} is binary-only and errors on a response with more
#'   than two levels.
#' @param ... Additional arguments for the supervised model
#' @param cluster_args A named list of arguments for the clustering step,
#'   such as \code{list(nstart = 5)} for k-means. k is set from the
#'   labelled classes and cannot be given here.
#' @return A tidylearn model object with additional class
#'   \code{"tidylearn_semisupervised"}, trained on pseudo-labeled data. The
#'   model includes a \code{semisupervised_info} element with
#'   \code{labeled_indices}, \code{cluster_model}, \code{label_mapping},
#'   and \code{n_unlabelled_dropped}, the number of rows left out for
#'   having no label to train on.
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
                              supervised_method = "tree", ...,
                              cluster_args = list()) {
  formula <- tl_as_formula(formula)
  tl_check_cluster_method(cluster_method)
  tl_check_cluster_args(
    cluster_args,
    "tl_semisupervised() sets k to the number of labelled classes."
  )

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

  # One cluster per class the labelled rows carry. A missing label is not
  # a class: counting it asked k-means for one cluster more than there are
  # classes to propagate.
  labelled_classes <- unique(
    stats::na.omit(as.character(response[labeled_indices]))
  )
  k <- length(labelled_classes)
  if (k < 2) {
    stop(
      "tl_semisupervised() needs labelled rows from at least two classes; ",
      "found ", k,
      if (k == 1) paste0(" (", labelled_classes, ")") else "",
      ". Label rows from more than one class.",
      call. = FALSE
    )
  }

  # Cluster on the formula's predictors
  predictor_data <- data[, tl_formula_predictors(formula, data),
                         drop = FALSE]
  clustering <- tl_assign_clusters(predictor_data, cluster_method, k,
                                   cluster_args)
  cluster_model <- clustering$model

  # Propagate labels within clusters
  cluster_labels <- tibble::tibble(
    obs_id = seq_len(nrow(data)),
    cluster = clustering$clusters,
    label = response
  )

  # For each cluster, the most common label among its labelled rows. A row
  # whose label is missing takes no part: table() of a factor counts every
  # level, so a cluster whose labels were all NA took the first level -- a
  # class none of its rows carried -- and for a character response the
  # empty table made summarize() fail.
  label_mapping <- cluster_labels |>
    dplyr::filter(.data$obs_id %in% labeled_indices, !is.na(.data$label)) |>
    dplyr::group_by(.data$cluster) |>
    dplyr::summarize(
      cluster_label = names(which.max(table(.data$label))),
      .groups = "drop"
    )

  # Assign pseudo-labels to unlabeled data
  pseudo_labeled <- cluster_labels |>
    dplyr::left_join(label_mapping, by = "cluster") |>
    dplyr::mutate(
      final_label = dplyr::if_else(
        .data$obs_id %in% labeled_indices,
        as.character(.data$label), .data$cluster_label
      )
    )

  # A cluster where no labelled observation carries a label has nothing to
  # propagate. Its rows used to become NA and vanish at fit time without a
  # word, so the model trained on a fraction of the data it appeared to use.
  unlabelled <- is.na(pseudo_labeled$final_label)
  if (any(unlabelled)) {
    is_labelled <- pseudo_labeled$obs_id %in% labeled_indices
    orphaned <- unlabelled & !is_labelled
    missing_label <- unlabelled & is_labelled
    rows <- function(n) if (n == 1) "row" else "rows"
    reasons <- c(
      if (any(orphaned)) {
        paste0(
          sum(orphaned), " unlabelled ", rows(sum(orphaned)),
          " in cluster(s) ",
          paste(sort(unique(pseudo_labeled$cluster[orphaned])),
                collapse = ", "),
          ", where no labelled observation carries a label"
        )
      },
      if (any(missing_label)) {
        paste0(sum(missing_label), " labelled ",
               rows(sum(missing_label)), " whose own label is missing")
      }
    )
    warning(
      sum(unlabelled), " of ", nrow(data), " rows have no label and are ",
      "left out of training: ", paste(reasons, collapse = ", and "),
      ". Label observations from across the data to use them.",
      call. = FALSE
    )
  }

  # Keep the response's level order. as.factor() sorted the labels, which
  # moved the positive class -- the second level -- whenever the declared
  # order was not alphabetical.
  data_pseudo <- data
  data_pseudo[[response_var]] <- factor(
    pseudo_labeled$final_label,
    levels = levels(tl_normalise_response(response))
  )
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
#' DBSCAN runs on the predictors the formula names, on their own scale: its
#' \code{eps} and \code{minPts} (defaults 0.5 and 5, passed through
#' \code{...}) are a distance and a count in those units. If every row comes
#' out as noise the call stops, since no normal data would be left to model.
#'
#' @param data A data frame
#' @param formula Model formula
#' @param response Response variable name, left out of the detection
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
#' @param ... Additional arguments for DBSCAN, such as \code{eps} and
#'   \code{minPts}
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

  if (!is.character(response) || length(response) != 1L ||
        !response %in% names(data)) {
    stop("'response' must name a column of `data`; got ",
         tl_describe_value(response), ".", call. = FALSE)
  }

  # Separate predictors for anomaly detection: those the formula names
  predictor_data <- data[, setdiff(tl_formula_predictors(formula, data),
                                   response), drop = FALSE]

  # Detect anomalies: DBSCAN's noise points
  anomaly_model <- tl_model(predictor_data, method = "dbscan", ...)
  is_anomaly <- anomaly_model$fit$clusters$cluster == 0

  # With every row noise there is no normal data to model. remove then
  # failed in lm() with "0 (non-NA) cases", downweight gave every row the
  # same weight and returned the unweighted fit, and flag added a constant
  # column whose coefficient was NA.
  if (all(is_anomaly)) {
    stop(
      "DBSCAN marked all ", length(is_anomaly), " rows as noise, so no ",
      "normal data is left to model. Its neighbourhood (eps = ",
      anomaly_model$fit$model$eps, ", minPts = ",
      anomaly_model$fit$model$minPts, ") is measured in the predictors' ",
      "own units, which are not scaled first. Raise eps, lower minPts, or ",
      "scale the predictors.",
      call. = FALSE
    )
  }

  # Take action based on anomalies
  if (action == "remove") {
    data_clean <- data[!is_anomaly, ]
    model <- tl_model(data_clean, formula, method = supervised_method)
    model$anomalies_removed <- sum(is_anomaly)
  } else if (action == "flag") {
    data_flagged <- data |>
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
#' The rows are clustered on the predictors the formula names, and a
#' model is fitted to each cluster. A cluster whose rows all hold one class
#' has nothing for a classifier to separate, so it gets no model and its
#' rows are predicted as that class.
#'
#' @param data A data frame
#' @param formula Model formula
#' @param cluster_method Clustering method: \code{"kmeans"} (default),
#'   \code{"pam"}, \code{"clara"} or \code{"hclust"}, whose tree is cut at
#'   \code{k}. Only k-means can assign new rows, so the others predict
#'   their training data alone.
#' @param k Number of clusters
#' @param supervised_method Supervised learning method (default:
#'   \code{"tree"}, which handles both regression and classification).
#'   \code{"linear"} needs a numeric response and refuses a factor.
#' @param ... Additional arguments for the supervised models
#' @param cluster_args A named list of arguments for the clustering step,
#'   such as \code{list(nstart = 5)} for k-means. Pass k as \code{k}.
#' @return A list with class \code{"tidylearn_stratified"} containing:
#'   \describe{
#'     \item{cluster_model}{The fitted clustering model.}
#'     \item{clusters}{The training rows' cluster assignments.}
#'     \item{supervised_models}{Named list of tidylearn models, one per
#'       cluster that holds more than one class.}
#'     \item{single_class_clusters}{Named character vector giving, for each
#'       cluster whose rows all hold one class, that class.}
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
                                 k = 3, supervised_method = "tree", ...,
                                 cluster_args = list()) {
  formula <- tl_as_formula(formula)
  tl_check_cluster_method(cluster_method)
  tl_check_cluster_args(cluster_args, "Pass k as the k argument.")

  # Extract response variable
  response_var <- all.vars(formula)[1]
  response <- data[[response_var]]
  is_categorical <- is.factor(response) || is.character(response)

  # Cluster on the formula's predictors
  predictor_data <- data[, tl_formula_predictors(formula, data),
                         drop = FALSE]
  clustering <- tl_assign_clusters(predictor_data, cluster_method, k,
                                   cluster_args)
  clusters <- clustering$clusters

  # Train a model for each cluster. One whose rows all hold a single class
  # gives a classifier nothing to separate, and every method refuses it,
  # which used to fail the whole call: k-means puts iris's 50 setosa rows
  # in a cluster of their own. Its rows are predicted as that class.
  cluster_models <- list()
  single_class <- character()
  for (i in seq_len(k)) {
    cluster_data <- data[clusters == i, , drop = FALSE]
    if (nrow(cluster_data) == 0) {
      next
    }
    name <- paste0("cluster_", i)
    if (is_categorical) {
      classes <- unique(
        stats::na.omit(as.character(cluster_data[[response_var]]))
      )
      if (length(classes) == 1L) {
        single_class[[name]] <- classes
        next
      }
    }
    cluster_models[[name]] <- tl_model(
      cluster_data, formula, method = supervised_method, ...
    )
  }

  # Return stratified model object
  structure(
    list(
      cluster_model = clustering$model,
      clusters = clusters,
      supervised_models = cluster_models,
      single_class_clusters = single_class,
      formula = formula,
      data = data
    ),
    class = c("tidylearn_stratified", "list")
  )
}

#' Predictions for the rows of a single-class cluster
#'
#' Takes \code{type} where \code{predict()} would, by name or as the first
#' unnamed argument, so these rows come back in the same shape as the
#' other clusters'.
#'
#' @param class_label The cluster's one class
#' @param n The number of rows
#' @param type The prediction type
#' @param ... Ignored
#' @return A tibble: one probability column, all 1, for \code{type =
#'   "prob"}, and a \code{.pred} column of the class otherwise
#' @keywords internal
#' @noRd
tl_predict_single_class <- function(class_label, n, type = "response", ...) {
  if (identical(type, "prob")) {
    tibble::tibble(!!class_label := rep(1, n))
  } else {
    tibble::tibble(.pred = rep(class_label, n))
  }
}

#' Predict from stratified models
#' @param object A tidylearn_stratified model object
#' @param new_data New data for predictions. NULL predicts the training
#'   rows from their stored cluster assignments, which works for every
#'   clustering method; new rows can be assigned by k-means only.
#' @param ... Additional arguments passed to each cluster's model, such as
#'   \code{type}
#' @return A \link[tibble]{tibble} of the columns each cluster's model
#'   returns for the requested \code{type} -- \code{.pred} by default, one
#'   column per class for \code{type = "prob"} -- and a \code{.cluster}
#'   column with cluster assignments. Rows of a single-class cluster are
#'   predicted as that class, with probability 1.
#' @examples
#' \donttest{
#' models <- tl_stratified_models(mtcars, mpg ~ .,
#'   cluster_method = "kmeans", k = 2, supervised_method = "linear")
#' preds <- predict(models)
#' }
#' @export
predict.tidylearn_stratified <- function(object, new_data = NULL, ...) {
  # Get response variable
  response_var <- all.vars(object$formula)[1]

  # The training rows already have their clusters. pam, clara and hclust
  # cannot assign rows at all, so recomputing them failed even here.
  if (is.null(new_data) && !is.null(object$clusters)) {
    new_data <- object$data
    cluster_ids <- object$clusters
  } else {
    if (is.null(new_data)) {
      new_data <- object$data
    }
    # Assign new data to clusters. any_of(): data to predict on need not
    # carry the response at all.
    predictor_data <- new_data |> dplyr::select(-dplyr::any_of(response_var))
    cluster_ids <- predict(object$cluster_model,
                           new_data = predictor_data)$cluster
  }

  # Predict each cluster's rows with its own model, keeping every column
  # the model returns. Reading back only .pred dropped the probability
  # columns type = "prob" returns, leaving a tibble of cluster ids.
  predictions <- vector("list", nrow(new_data))
  for (cluster_id in unique(cluster_ids)) {
    rows <- which(cluster_ids == cluster_id)
    model_name <- paste0("cluster_", cluster_id)
    if (model_name %in% names(object$single_class_clusters)) {
      pred <- tl_predict_single_class(
        object$single_class_clusters[[model_name]], length(rows), ...
      )
    } else if (model_name %in% names(object$supervised_models)) {
      pred <- predict(
        object$supervised_models[[model_name]],
        new_data = new_data[rows, , drop = FALSE], ...
      )
    } else {
      next
    }
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
