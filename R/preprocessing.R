#' Data Preprocessing for tidylearn
#'
#' Unified preprocessing functions that work with both
#' supervised and unsupervised workflows

#' Prepare Data for Machine Learning
#'
#' Comprehensive preprocessing pipeline including imputation, scaling,
#' encoding, and feature engineering
#'
#' The statistics are learned from, and applied to, the data passed in.
#' Preparing a whole dataset and then splitting it lets the test rows
#' shape the imputation values and scaling their own scores are measured
#' against. To evaluate a model, split first, or use
#' \code{\link{tl_pipeline}}, which learns its preprocessing inside each
#' resampling fold.
#'
#' @param data A data frame
#' @param formula Optional formula (for supervised learning). Only its
#'   predictors are processed; a column it excludes, such as \code{- id},
#'   is returned unchanged.
#' @param impute_method Method for imputing a missing numeric value:
#'   "mean", "median" or "mode". A missing categorical value is always
#'   filled with the column's most frequent value.
#' @param scale_method Scaling method: "standardize",
#'   "normalize", "robust", "none"
#' @param encode_categorical Whether to encode categorical
#'   variables (default: TRUE)
#' @param remove_zero_variance Remove zero-variance features (default: TRUE)
#' @param remove_correlated Remove highly correlated features (default: FALSE)
#' @param correlation_cutoff Correlation threshold for removal (default: 0.95)
#' @return A list with components:
#'   \describe{
#'     \item{\code{data}}{The processed data frame.}
#'     \item{\code{original_data}}{The original unprocessed data frame.}
#'     \item{\code{preprocessing_steps}}{A record of each step applied
#'       (imputation values, encoding maps, scaling parameters, etc.). It
#'       is for inspection: no function applies it to new data.}
#'     \item{\code{formula}}{The formula passed in (or \code{NULL}).}
#'   }
#' @export
#' @examples
#' \donttest{
#' processed <- tl_prepare_data(iris, Species ~ ., scale_method = "standardize")
#' model <- tl_model(processed$data, Species ~ ., method = "tree")
#' }
tl_prepare_data <- function(data, formula = NULL,
                            impute_method = "mean",
                            scale_method = "standardize",
                            encode_categorical = TRUE,
                            remove_zero_variance = TRUE,
                            remove_correlated = FALSE,
                            correlation_cutoff = 0.95) {

  if (!is.null(formula)) {
    formula <- tl_as_formula(formula)
  }

  imputers <- c("mean", "median", "mode")
  if (!is.character(impute_method) || length(impute_method) != 1L ||
        !impute_method %in% imputers) {
    stop(
      "'impute_method' must be one of ",
      paste0("\"", imputers, "\"", collapse = ", "), ".",
      call. = FALSE
    )
  }

  processed_data <- data
  preprocessing_steps <- list()

  # Extract response if formula provided
  response_var <- NULL
  if (!is.null(formula)) {
    response_var <- all.vars(formula)[1]
  }

  # Separate predictors and response. Only the formula's predictors are
  # processed: a column it excludes, such as `- id`, used to be one-hot
  # encoded and scaled with the rest. Excluded columns are carried through
  # unchanged, so the same formula still finds them downstream.
  passthrough <- character(0)
  if (!is.null(response_var)) {
    response_data <- processed_data[[response_var]]
    predictors <- intersect(get_formula_vars(formula, data), names(data))
    passthrough <- setdiff(names(data), c(response_var, predictors))
    passthrough_data <- processed_data[passthrough]
    predictor_data <- processed_data[predictors]
  } else {
    response_data <- NULL
    predictor_data <- processed_data
  }

  # 1. Handle missing values
  if (any(is.na(predictor_data))) {
    message("Imputing missing values using method: ", impute_method)
    imputation_info <- impute_missing(predictor_data, method = impute_method)
    predictor_data <- imputation_info$data
    preprocessing_steps$imputation <- imputation_info
  }

  # 2. Encode categorical variables
  if (encode_categorical) {
    cat_vars <- names(predictor_data)[
      sapply(
        predictor_data,
        function(x) is.factor(x) || is.character(x)
      )
    ]

    if (length(cat_vars) > 0) {
      message("Encoding ", length(cat_vars), " categorical variables")
      encoding_info <- encode_categoricals(predictor_data, cat_vars)
      predictor_data <- encoding_info$data
      preprocessing_steps$encoding <- encoding_info
    }
  }

  # 3. Remove zero variance features
  if (remove_zero_variance) {
    zero_var_cols <- find_zero_variance(predictor_data)
    if (length(zero_var_cols) > 0) {
      message("Removing ", length(zero_var_cols), " zero-variance features")
      predictor_data <- predictor_data |>
        dplyr::select(-dplyr::all_of(zero_var_cols))
      preprocessing_steps$zero_variance <- zero_var_cols
    }
  }

  # 4. Remove highly correlated features
  if (remove_correlated) {
    numeric_data <- predictor_data |> dplyr::select(where(is.numeric))
    if (ncol(numeric_data) > 1) {
      cor_matrix <- stats::cor(numeric_data, use = "pairwise.complete.obs")
      high_cor <- find_high_correlation(cor_matrix, cutoff = correlation_cutoff)

      if (length(high_cor) > 0) {
        message("Removing ", length(high_cor), " highly correlated features")
        predictor_data <- predictor_data |>
          dplyr::select(-dplyr::all_of(high_cor))
        preprocessing_steps$high_correlation <- high_cor
      }
    }
  }

  # 5. Scale numeric features
  if (scale_method != "none") {
    numeric_cols <- names(predictor_data)[sapply(predictor_data, is.numeric)]

    if (length(numeric_cols) > 0) {
      message("Scaling numeric features using method: ", scale_method)
      scaling_info <- scale_features(
        predictor_data, numeric_cols,
        method = scale_method
      )
      predictor_data <- scaling_info$data
      preprocessing_steps$scaling <- scaling_info
    }
  }

  # Recombine with response and the columns the formula left out
  if (!is.null(response_var)) {
    processed_data <- predictor_data |>
      dplyr::bind_cols(passthrough_data) |>
      dplyr::mutate(!!response_var := response_data)
  } else {
    processed_data <- predictor_data
  }

  list(
    data = processed_data,
    original_data = data,
    preprocessing_steps = preprocessing_steps,
    formula = formula
  )
}

#' Impute missing values
#' @keywords internal
#' @noRd
impute_missing <- function(data, method = "mean") {
  imputed_data <- data
  imputation_values <- list()

  # "mode" and "knn" used to fall through to the mean while the message
  # named the method asked for, and a categorical column was never filled
  # -- its NA rows then broke one-hot encoding with a recycling error. A
  # categorical column takes its most frequent value whatever the method,
  # since a mean or median of categories does not exist.
  for (col in names(data)) {
    values <- data[[col]]
    # An entirely missing column has nothing to impute from and is left
    # as it is. The zero-variance step removes it when it is numeric.
    if (!anyNA(values) || all(is.na(values))) {
      next
    }

    impute_val <- if (is.numeric(values) && method == "mean") {
      mean(values, na.rm = TRUE)
    } else if (is.numeric(values) && method == "median") {
      stats::median(values, na.rm = TRUE)
    } else {
      tl_most_frequent(values)
    }

    imputed_data[[col]][is.na(values)] <- impute_val
    imputation_values[[col]] <- impute_val
  }

  list(
    data = imputed_data,
    method = method,
    imputation_values = imputation_values
  )
}

#' Most frequent non-missing value
#'
#' Ties go to the value seen first. Works on the values themselves rather
#' than on \code{table()} names, which would turn a number into a string.
#'
#' @param x A vector with at least one non-missing value.
#' @return A length-one vector of the same type as \code{x}.
#' @keywords internal
#' @noRd
tl_most_frequent <- function(x) {
  observed <- x[!is.na(x)]
  candidates <- unique(observed)
  candidates[which.max(tabulate(match(observed, candidates)))]
}

#' Encode categorical variables
#' @keywords internal
#' @noRd
encode_categoricals <- function(data, cat_vars) {
  encoded_data <- data
  encoding_map <- list()

  for (var in cat_vars) {
    if (is.character(data[[var]])) {
      encoded_data[[var]] <- as.factor(data[[var]])
    }

    # One-hot encode if more than 2 levels
    if (nlevels(encoded_data[[var]]) > 2) {
      # Create dummy variables
      dummies <- stats::model.matrix(
        ~ . - 1,
        data = data.frame(x = encoded_data[[var]])
      )
      colnames(dummies) <- paste0(var, "_", gsub("^x", "", colnames(dummies)))

      # Remove original column and add dummies
      encoded_data <- encoded_data |>
        dplyr::select(-dplyr::all_of(var)) |>
        dplyr::bind_cols(as.data.frame(dummies))

      encoding_map[[var]] <- colnames(dummies)
    }
  }

  list(
    data = encoded_data,
    encoding_map = encoding_map
  )
}

#' Find zero variance columns
#' @keywords internal
#' @noRd
find_zero_variance <- function(data) {
  numeric_data <- data |> dplyr::select(where(is.numeric))

  zero_var <- sapply(numeric_data, function(x) {
    stats::var(x, na.rm = TRUE) == 0 || all(is.na(x))
  })

  names(zero_var)[zero_var]
}

#' Find highly correlated features
#' @keywords internal
#' @noRd
find_high_correlation <- function(cor_matrix, cutoff = 0.95) {
  # Remove one feature at a time: take the most correlated remaining pair,
  # drop whichever member is more correlated with everything else still
  # present, and look again. Deciding every pair up front against a matrix
  # whose lower triangle had been zeroed dropped both ends of a chain
  # x1 - x2 - x3 and kept x2, the one feature correlated with both.
  abs_cor <- abs(cor_matrix)
  diag(abs_cor) <- NA
  remaining <- colnames(abs_cor)
  to_remove <- character()

  repeat {
    current <- abs_cor[remaining, remaining, drop = FALSE]
    if (length(remaining) < 2 || !any(current > cutoff, na.rm = TRUE)) {
      break
    }

    pair <- which(current == max(current, na.rm = TRUE), arr.ind = TRUE)[1, ]
    candidates <- remaining[pair]
    mean_cor <- colMeans(current[, candidates, drop = FALSE], na.rm = TRUE)
    drop <- candidates[which.max(mean_cor)]

    to_remove <- c(to_remove, drop)
    remaining <- setdiff(remaining, drop)
  }

  to_remove
}

#' Scale numeric features
#' @keywords internal
#' @noRd
scale_features <- function(data, numeric_cols, method = "standardize") {
  scaled_data <- data
  scaling_params <- list()

  for (col in numeric_cols) {
    # A column with fewer than two observed values has no spread to scale
    # by; its sd is NA, and `if (NA > 0)` stopped the whole call
    if (sum(!is.na(data[[col]])) < 2) {
      next
    }

    if (method == "standardize") {
      # Z-score standardization
      mean_val <- mean(data[[col]], na.rm = TRUE)
      sd_val <- stats::sd(data[[col]], na.rm = TRUE)

      if (sd_val > 0) {
        scaled_data[[col]] <- (data[[col]] - mean_val) / sd_val
        scaling_params[[col]] <- list(mean = mean_val, sd = sd_val)
      }

    } else if (method == "normalize") {
      # Min-max normalization
      min_val <- min(data[[col]], na.rm = TRUE)
      max_val <- max(data[[col]], na.rm = TRUE)

      if (max_val > min_val) {
        scaled_data[[col]] <- (data[[col]] - min_val) / (max_val - min_val)
        scaling_params[[col]] <- list(min = min_val, max = max_val)
      }

    } else if (method == "robust") {
      # Robust scaling using median and IQR
      median_val <- stats::median(data[[col]], na.rm = TRUE)
      q1 <- stats::quantile(data[[col]], 0.25, na.rm = TRUE)
      q3 <- stats::quantile(data[[col]], 0.75, na.rm = TRUE)
      iqr_val <- q3 - q1

      if (iqr_val > 0) {
        scaled_data[[col]] <- (data[[col]] - median_val) / iqr_val
        scaling_params[[col]] <- list(median = median_val, iqr = iqr_val)
      }
    }
  }

  list(
    data = scaled_data,
    method = method,
    scaling_params = scaling_params
  )
}

#' Split data into train and test sets
#'
#' @param data A data frame
#' @param prop Proportion for training set (default: 0.8)
#' @param stratify Column name for stratified splitting
#' @param seed Random seed for reproducibility
#' @return A list with two elements:
#'   \describe{
#'     \item{\code{$train}}{A data frame containing the training subset.}
#'     \item{\code{$test}}{A data frame containing the test subset.}
#'   }
#' @export
#' @examples
#' \donttest{
#' split_data <- tl_split(iris, prop = 0.7, stratify = "Species")
#' train <- split_data$train
#' test <- split_data$test
#' }
tl_split <- function(data, prop = 0.8, stratify = NULL, seed = NULL) {
  # prop = 1.5 gave a 31/1 split of 32 rows and prop = -1 a 1/31 one,
  # because the per-group size is clamped to leave a row on each side
  if (!is.numeric(prop) || length(prop) != 1L || is.na(prop) ||
        prop <= 0 || prop >= 1) {
    stop("'prop' must be a single number strictly between 0 and 1, ",
         "such as 0.8", call. = FALSE)
  }

  # Seed this call without rewriting the caller's random stream
  tl_local_seed(seed)

  n <- nrow(data)

  if (!is.null(stratify)) {
    if (!stratify %in% names(data)) {
      stop("Stratify variable not found in data", call. = FALSE)
    }

    # Stratified sampling. Each stratum keeps at least one training row
    # and one test row where it has the rows to spare, so a small group
    # cannot vanish from the training set entirely.
    # Index into idx rather than sampling it: sample() on a single number
    # draws from 1:idx, so a one-row stratum took a row from some other
    # stratum -- sometimes one already drawn -- and left its own in test.
    # split() drops an NA group, so rows missing the stratify value were in
    # no stratum, never drawn, and all landed in test. They form their own.
    groups <- split(seq_len(n), addNA(factor(data[[stratify]]), ifany = TRUE))
    train_indices <- unlist(lapply(groups, function(idx) {
      idx[sample.int(length(idx), size = tl_train_size(length(idx), prop))]
    }))

  } else {
    # Simple random sampling
    train_indices <- sample(seq_len(n), size = tl_train_size(n, prop))
  }

  # data[-integer(0), ] selects NO rows, so an empty training set would
  # hand back an empty test set too and silently lose every observation
  if (length(train_indices) == 0) {
    stop(
      "Splitting ", n, " row(s) at prop = ", prop,
      " leaves no training data. Use a larger sample or a higher prop.",
      call. = FALSE
    )
  }

  # drop = FALSE, or a one-column frame comes back as a bare vector
  list(
    train = data[train_indices, , drop = FALSE],
    test = data[-train_indices, , drop = FALSE]
  )
}

#' Number of training rows to draw from a group
#'
#' \code{floor()} alone returns 0 for small groups, which empties the
#' training set -- and \code{data[-integer(0), ]} then empties the test
#' set as well.
#'
#' @param n_group Rows available in this group
#' @param prop Target training proportion
#' @return An integer count, at least 1 and at most \code{n_group - 1}
#'   whenever the group has two or more rows
#' @keywords internal
#' @noRd
tl_train_size <- function(n_group, prop) {
  if (n_group == 0) {
    return(0L)
  }
  if (n_group == 1L) {
    return(1L)
  }

  size <- floor(n_group * prop)
  min(max(size, 1L), n_group - 1L)
}
