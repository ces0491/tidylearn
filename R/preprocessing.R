#' Data Preprocessing for tidylearn
#'
#' Unified preprocessing functions that work with both supervised and unsupervised workflows

#' Prepare Data for Machine Learning
#'
#' Comprehensive preprocessing pipeline including imputation, scaling,
#' encoding, and feature engineering
#'
#' @param data A data frame
#' @param formula Optional formula (for supervised learning)
#' @param impute_method Method for missing value imputation: "mean", "median", "mode", "knn"
#' @param scale_method Scaling method: "standardize", "normalize", "robust", "none"
#' @param encode_categorical Whether to encode categorical variables (default: TRUE)
#' @param remove_zero_variance Remove zero-variance features (default: TRUE)
#' @param remove_correlated Remove highly correlated features (default: FALSE)
#' @param correlation_cutoff Correlation threshold for removal (default: 0.95)
#' @return A list containing processed data and preprocessing metadata
#' @export
#' @examples
#' \dontrun{
#' processed <- tl_prepare_data(iris, Species ~ ., scale_method = "standardize")
#' model <- tl_model(processed$data, Species ~ ., method = "logistic")
#' }
tl_prepare_data <- function(data, formula = NULL,
                            impute_method = "mean",
                            scale_method = "standardize",
                            encode_categorical = TRUE,
                            remove_zero_variance = TRUE,
                            remove_correlated = FALSE,
                            correlation_cutoff = 0.95) {

  processed_data <- data
  preprocessing_steps <- list()

  # Extract response if formula provided
  response_var <- NULL
  if (!is.null(formula)) {
    response_var <- all.vars(formula)[1]
  }

  # Separate predictors and response
  if (!is.null(response_var)) {
    response_data <- processed_data[[response_var]]
    predictor_data <- processed_data %>% dplyr::select(-dplyr::all_of(response_var))
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
    cat_vars <- names(predictor_data)[sapply(predictor_data, function(x) is.factor(x) || is.character(x))]

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
      predictor_data <- predictor_data %>% dplyr::select(-dplyr::all_of(zero_var_cols))
      preprocessing_steps$zero_variance <- zero_var_cols
    }
  }

  # 4. Remove highly correlated features
  if (remove_correlated) {
    numeric_data <- predictor_data %>% dplyr::select(where(is.numeric))
    if (ncol(numeric_data) > 1) {
      cor_matrix <- stats::cor(numeric_data, use = "pairwise.complete.obs")
      high_cor <- find_high_correlation(cor_matrix, cutoff = correlation_cutoff)

      if (length(high_cor) > 0) {
        message("Removing ", length(high_cor), " highly correlated features")
        predictor_data <- predictor_data %>% dplyr::select(-dplyr::all_of(high_cor))
        preprocessing_steps$high_correlation <- high_cor
      }
    }
  }

  # 5. Scale numeric features
  if (scale_method != "none") {
    numeric_cols <- names(predictor_data)[sapply(predictor_data, is.numeric)]

    if (length(numeric_cols) > 0) {
      message("Scaling numeric features using method: ", scale_method)
      scaling_info <- scale_features(predictor_data, numeric_cols, method = scale_method)
      predictor_data <- scaling_info$data
      preprocessing_steps$scaling <- scaling_info
    }
  }

  # Recombine with response
  if (!is.null(response_var)) {
    processed_data <- predictor_data %>%
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

  numeric_cols <- names(data)[sapply(data, is.numeric)]

  for (col in numeric_cols) {
    if (any(is.na(data[[col]]))) {
      if (method == "mean") {
        impute_val <- mean(data[[col]], na.rm = TRUE)
      } else if (method == "median") {
        impute_val <- stats::median(data[[col]], na.rm = TRUE)
      } else {
        impute_val <- mean(data[[col]], na.rm = TRUE)
      }

      imputed_data[[col]][is.na(imputed_data[[col]])] <- impute_val
      imputation_values[[col]] <- impute_val
    }
  }

  list(
    data = imputed_data,
    method = method,
    imputation_values = imputation_values
  )
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
      dummies <- stats::model.matrix(~ . - 1, data = data.frame(x = encoded_data[[var]]))
      colnames(dummies) <- paste0(var, "_", gsub("^x", "", colnames(dummies)))

      # Remove original column and add dummies
      encoded_data <- encoded_data %>%
        dplyr::select(-dplyr::all_of(var)) %>%
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
  numeric_data <- data %>% dplyr::select(where(is.numeric))

  zero_var <- sapply(numeric_data, function(x) {
    stats::var(x, na.rm = TRUE) == 0 || all(is.na(x))
  })

  names(zero_var)[zero_var]
}

#' Find highly correlated features
#' @keywords internal
#' @noRd
find_high_correlation <- function(cor_matrix, cutoff = 0.95) {
  cor_matrix[lower.tri(cor_matrix, diag = TRUE)] <- 0

  high_cor_pairs <- which(abs(cor_matrix) > cutoff, arr.ind = TRUE)

  if (nrow(high_cor_pairs) == 0) {
    return(character(0))
  }

  # For each pair, remove the one with higher average correlation
  to_remove <- character()
  for (i in seq_len(nrow(high_cor_pairs))) {
    row_idx <- high_cor_pairs[i, 1]
    col_idx <- high_cor_pairs[i, 2]

    row_name <- rownames(cor_matrix)[row_idx]
    col_name <- colnames(cor_matrix)[col_idx]

    # Calculate average correlation
    row_avg_cor <- mean(abs(cor_matrix[row_idx, ]), na.rm = TRUE)
    col_avg_cor <- mean(abs(cor_matrix[, col_idx]), na.rm = TRUE)

    if (row_avg_cor > col_avg_cor) {
      to_remove <- c(to_remove, row_name)
    } else {
      to_remove <- c(to_remove, col_name)
    }
  }

  unique(to_remove)
}

#' Scale numeric features
#' @keywords internal
#' @noRd
scale_features <- function(data, numeric_cols, method = "standardize") {
  scaled_data <- data
  scaling_params <- list()

  for (col in numeric_cols) {
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
#' @return A list with train and test data frames
#' @export
#' @examples
#' \dontrun{
#' split_data <- tl_split(iris, prop = 0.7, stratify = "Species")
#' train <- split_data$train
#' test <- split_data$test
#' }
tl_split <- function(data, prop = 0.8, stratify = NULL, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  n <- nrow(data)

  if (!is.null(stratify)) {
    if (!stratify %in% names(data)) {
      stop("Stratify variable not found in data", call. = FALSE)
    }

    # Stratified sampling
    groups <- split(seq_len(n), data[[stratify]])
    train_indices <- unlist(lapply(groups, function(idx) {
      sample(idx, size = floor(length(idx) * prop))
    }))

  } else {
    # Simple random sampling
    train_indices <- sample(seq_len(n), size = floor(n * prop))
  }

  list(
    train = data[train_indices, ],
    test = data[-train_indices, ]
  )
}
