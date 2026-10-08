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
#' @param data A data frame. A grouped tibble is prepared as a whole and
#'   returned ungrouped.
#' @param formula Optional two-sided formula (for supervised learning),
#'   whose response must be a column of \code{data}. Only its predictors are
#'   processed; a column it excludes, such as \code{- id}, is returned
#'   unchanged. Leave it \code{NULL} to process every column.
#' @param impute_method Method for imputing a missing numeric value:
#'   "mean", "median" or "mode". A missing categorical value is always
#'   filled with the column's most frequent value.
#' @param scale_method Scaling method: "standardize",
#'   "normalize", "robust", "none". A column whose spread is zero or not
#'   finite, such as one holding an \code{Inf}, is left unscaled.
#' @param encode_categorical Whether to encode categorical
#'   variables (default: TRUE)
#' @param remove_zero_variance Remove zero-variance features (default: TRUE)
#' @param remove_correlated Remove highly correlated features (default: FALSE)
#' @param correlation_cutoff Correlation threshold for removal, greater than
#'   0 and at most 1 (default: 0.95)
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
    # all.vars(~ x1 + x2)[1] is x1, which was then passed through
    # unprocessed as if it were the response
    if (length(formula) != 3L) {
      stop(
        "'formula' has no response: write it as y ~ predictors, or leave ",
        "formula = NULL to process every column.",
        call. = FALSE
      )
    }
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

  # An unrecognised method fell through every branch of scale_features()
  # and returned the data unscaled, under a message saying it had scaled
  scalers <- c("standardize", "normalize", "robust", "none")
  if (!is.character(scale_method) || length(scale_method) != 1L ||
        !scale_method %in% scalers) {
    stop(
      "'scale_method' must be one of ",
      paste0("\"", scalers, "\"", collapse = ", "), ".",
      call. = FALSE
    )
  }

  # No correlation exceeds 1, so a cutoff of 95, meant as a percentage,
  # removed nothing without a word
  if (!is.numeric(correlation_cutoff) || length(correlation_cutoff) != 1L ||
        is.na(correlation_cutoff) || correlation_cutoff <= 0 ||
        correlation_cutoff > 1) {
    stop(
      "'correlation_cutoff' must be a single number greater than 0 and at ",
      "most 1, such as 0.95; got ", tl_describe_value(correlation_cutoff),
      ".",
      call. = FALSE
    )
  }

  # A grouped tibble kept its groups through the column selections below,
  # so the response was added back once per group and failed on its length
  processed_data <- dplyr::ungroup(data)
  preprocessing_steps <- list()

  # Extract response if formula provided
  response_var <- NULL
  if (!is.null(formula)) {
    response_var <- all.vars(formula)[1]
    # A misspelt response matched no column, so the real one was scaled as
    # a predictor and the typo was dropped without a word
    if (!response_var %in% names(data)) {
      stop(
        "The formula's response, '", response_var,
        "', is not a column of `data`. Available: ",
        paste(names(data), collapse = ", "), ".",
        call. = FALSE
      )
    }
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

  # 2. Encode categorical variables. vapply() keeps an empty predictor set
  # a logical index, where sapply() returned list() and the subset failed
  # with "invalid subscript type 'list'". An entirely missing column has
  # no value to encode -- model.matrix() dropped every one of its rows --
  # and is left as it is, as imputation leaves it.
  if (encode_categorical) {
    cat_vars <- names(predictor_data)[
      vapply(
        predictor_data,
        function(x) (is.factor(x) || is.character(x)) && !all(is.na(x)),
        logical(1)
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
    numeric_cols <- names(predictor_data)[
      vapply(predictor_data, is.numeric, logical(1))
    ]

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

  # vapply(), because sapply() over a frame with no numeric column returns
  # list(). isTRUE(), because the variance of a single row, or of a column
  # holding an Inf, is NA, and an NA in the selection stopped the call with
  # "Selections can't have missing values".
  zero_var <- vapply(numeric_data, function(x) {
    all(is.na(x)) || isTRUE(stats::var(x, na.rm = TRUE) == 0)
  }, logical(1))

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
    # by; its sd is NA, and `if (NA > 0)` stopped the whole call. An Inf
    # leaves sd() NaN and the range infinite, so each spread below must be
    # finite as well as positive.
    if (sum(!is.na(data[[col]])) < 2) {
      next
    }

    if (method == "standardize") {
      # Z-score standardization
      mean_val <- mean(data[[col]], na.rm = TRUE)
      sd_val <- stats::sd(data[[col]], na.rm = TRUE)

      if (is.finite(sd_val) && sd_val > 0) {
        scaled_data[[col]] <- (data[[col]] - mean_val) / sd_val
        scaling_params[[col]] <- list(mean = mean_val, sd = sd_val)
      }

    } else if (method == "normalize") {
      # Min-max normalization
      min_val <- min(data[[col]], na.rm = TRUE)
      max_val <- max(data[[col]], na.rm = TRUE)

      if (is.finite(max_val - min_val) && max_val > min_val) {
        scaled_data[[col]] <- (data[[col]] - min_val) / (max_val - min_val)
        scaling_params[[col]] <- list(min = min_val, max = max_val)
      }

    } else if (method == "robust") {
      # Robust scaling using median and IQR
      median_val <- stats::median(data[[col]], na.rm = TRUE)
      q1 <- stats::quantile(data[[col]], 0.25, na.rm = TRUE)
      q3 <- stats::quantile(data[[col]], 0.75, na.rm = TRUE)
      iqr_val <- q3 - q1

      if (is.finite(iqr_val) && iqr_val > 0) {
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
#' @param stratify Column name for stratified splitting. Each stratum is
#'   split at \code{prop}. A numeric column with more than five distinct
#'   values is stratified by its quartiles, and rows missing the value form
#'   a stratum of their own. A stratum of a single row goes to the training
#'   set, unless such strata together hold more than a tenth of the rows, as
#'   in an ID column; then they are pooled into one stratum and split
#'   together.
#' @param seed Random seed for reproducibility
#' @return A list with two elements:
#'   \describe{
#'     \item{\code{$train}}{A data frame containing the training subset.}
#'     \item{\code{$test}}{A data frame containing the test subset.}
#'   }
#'   A split that leaves the test set empty, as a single row does, warns.
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

  # A vector of two names failed the column lookup with "the condition has
  # length > 1"
  if (!is.null(stratify)) {
    if (!is.character(stratify) || length(stratify) != 1L ||
          is.na(stratify)) {
      stop("'stratify' must be the name of one column of 'data'; got ",
           tl_describe_value(stratify), ".", call. = FALSE)
    }
    if (!stratify %in% names(data)) {
      stop("Stratify variable not found in data: '", stratify, "'.",
           call. = FALSE)
    }
  }

  # Seed this call without rewriting the caller's random stream
  tl_local_seed(seed)

  n <- nrow(data)

  if (!is.null(stratify)) {
    # Stratified sampling. Each stratum keeps at least one training row
    # and one test row where it has the rows to spare, and a stratum of
    # one row goes to training, unless tl_split_strata() pooled the
    # one-row strata because together they are over a tenth of the rows.
    # Index into idx rather than sampling it: sample() on a single number
    # draws from 1:idx, so a one-row stratum took a row from some other
    # stratum -- sometimes one already drawn -- and left its own in test.
    groups <- split(seq_len(n), tl_split_strata(data[[stratify]]))
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

  # A single row cannot sit on both sides, and an empty test set scores
  # as NaN rather than failing
  if (length(train_indices) == n) {
    warning(
      "Splitting ", n, " row(s) at prop = ", prop, " leaves no test data, ",
      "so the test set is empty. Use a larger sample.",
      call. = FALSE
    )
  }

  # drop = FALSE, or a one-column frame comes back as a bare vector
  list(
    train = data[train_indices, , drop = FALSE],
    test = data[-train_indices, , drop = FALSE]
  )
}

#' Strata for a stratified split
#'
#' Each distinct value used to be a stratum of its own. A continuous or ID
#' column then made every row a stratum of one, and a stratum of one goes
#' wholly to training, so the test set came back empty. A numeric column
#' with more than five distinct values is cut at its quartiles instead,
#' which is the default of rsample's \code{make_strata()}.
#'
#' A stratum of one row still goes to training, so a level seen once is one
#' the model was trained on; in test, \code{predict()} would refuse it as
#' new. When such strata together hold more than a tenth of the rows -- the
#' threshold of \code{make_strata()}'s \code{pool = 0.1} -- the column is
#' ID-like, and they are pooled into one stratum that is split like any
#' other.
#'
#' \code{make_strata()} goes further: it uses fewer bins when a bin would
#' hold under 20 rows, pools any stratum under a tenth of the data, and
#' assigns missing values to strata at random. A single split needs only
#' two rows in a stratum, and missing values keep a stratum of their own.
#'
#' @param x The stratify column
#' @return An integer stratum code per row
#' @keywords internal
#' @noRd
tl_split_strata <- function(x) {
  if (is.numeric(x) && length(unique(x[!is.na(x)])) > 5) {
    breaks <- unique(stats::quantile(x, probs = seq(0, 1, 0.25),
                                     na.rm = TRUE, names = FALSE))
    x <- cut(x, breaks = breaks, include.lowest = TRUE)
  }

  # split() drops an NA group, so rows missing the stratify value were in
  # no stratum, never drawn, and all landed in test. They form their own.
  strata <- as.integer(addNA(factor(x), ifany = TRUE))
  single <- tabulate(strata)[strata] == 1L
  # One-row strata holding more than a tenth of the rows
  if (10 * sum(single) > length(strata)) {
    strata[single] <- 0L
  }
  strata
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
