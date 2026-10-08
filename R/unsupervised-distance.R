#' Tidy Distance Matrix Computation
#'
#' Compute distance matrices with tidy output
#'
#' @param data A data frame or tibble
#' @param method Character; distance method
#'   (default: "euclidean"). Options: "euclidean",
#'   "manhattan", "maximum", "gower"
#' @param cols Columns to include (tidy select).
#'   If NULL, uses all numeric columns, or every column for
#'   \code{method = "gower"}.
#' @param ... Additional arguments passed to distance functions
#'
#' @return A \code{\link[stats]{dist}} object containing the computed
#'   distance matrix.
#'
#' @examples
#' \donttest{
#' d <- tidy_dist(iris[, 1:4], method = "euclidean")
#' }
#'
#' @export
tidy_dist <- function(data, method = "euclidean", cols = NULL, ...) {

  # Every column to begin with: Gower reads factors as they are, and the
  # other methods keep the numeric ones below
  data_selected <- tl_select_columns(
    data, rlang::enquo(cols), all_columns = TRUE,
    numeric_only = method != "gower",
    what = paste0("The ", method, " distance")
  )

  usable <- if (method == "gower") {
    data_selected
  } else {
    dplyr::select(data_selected, where(is.numeric))
  }

  # With no column to compare on, every distance is undefined, and the
  # methods downstream would report that as missing values in every row
  if (ncol(usable) == 0) {
    stop(
      "The ", method, " distance needs at least one ",
      if (method == "gower") "column" else "numeric column",
      ", but none were found.",
      call. = FALSE
    )
  }

  # Compute distance based on method
  if (method == "gower") {
    dist_mat <- tidy_gower(usable, ...)
  } else {
    dist_mat <- stats::dist(as.matrix(usable), method = method)
  }

  dist_mat
}


#' Gower Distance Calculation
#'
#' Computes Gower distance for mixed data types (numeric, factor, ordered)
#'
#' @param data A data frame or tibble
#' @param weights Optional named vector of variable
#'   weights (default: equal weights)
#'
#' @return A \code{\link[stats]{dist}} object containing Gower distances, with
#'   the \code{method} attribute set to \code{"gower"}. A pair of rows with
#'   no variable observed in both has no defined distance and is \code{NA},
#'   as in \code{\link[cluster]{daisy}}.
#'
#' @details
#' Gower distance handles mixed data types:
#' - Numeric: range-normalized Manhattan distance
#' - Factor/Character: 0 if same, 1 if different
#' - Ordered: treated as numeric ranks
#'
#' Formula: d_ij = sum(w_k * d_ijk) / sum(w_k)
#' where d_ijk is the dissimilarity for variable k between obs i and j
#'
#' @examples
#' # Create example data with mixed types
#' car_data <- data.frame(
#'   horsepower = c(130, 250, 180),
#'   weight = c(1200, 1650, 1420),
#'   color = factor(c("red", "black", "blue"))
#' )
#'
#' # Compute Gower distance
#' gower_dist <- tidy_gower(car_data)
#'
#' @export
tidy_gower <- function(data, weights = NULL) {

  # Convert to data frame
  data <- as.data.frame(data)

  n <- nrow(data)
  p <- ncol(data)

  # Set up weights. The loop below indexes them positionally, so a named
  # vector has to be reordered to match the columns first -- otherwise
  # weights = c(color = 100, x = 1) for data.frame(x, color) silently
  # applies the 100 to x.
  if (is.null(weights)) {
    weights <- rep(1, p)
    names(weights) <- colnames(data)
  } else if (!is.null(names(weights)) && all(nzchar(names(weights)))) {
    missing_weights <- setdiff(colnames(data), names(weights))
    if (length(missing_weights) > 0) {
      stop(
        "'weights' is named but has no entry for: ",
        paste(missing_weights, collapse = ", "),
        call. = FALSE
      )
    }
    weights <- weights[colnames(data)]
  } else if (length(weights) != p) {
    stop(
      "'weights' must have one entry per column (", p, "). Got: ",
      length(weights), ".",
      call. = FALSE
    )
  }

  # Pre-computation pass
  #
  # 1. Extract each column to a plain vector in a list.
  #    data[i, k] inside a loop dispatches to [.data.frame on every call:
  #    S3 method lookup + argument matching + potential intermediate allocation.
  #    col_vecs[[k]][i] is a hash-table lookup then a C-level pointer offset —
  #    benchmarks show 10-100x faster for scalar access.
  #
  # 2. Store each column's type as a string so we avoid repeated is.numeric() /
  #    is.ordered() / is.factor() S3 predicate calls inside the hot triple loop.
  #
  # 3. Ranges and rank vectors stay precomputed
  col_vecs   <- as.list(data)           # plain-vector views, no copy
  col_ranges <- vector("numeric",   p)
  col_ranks  <- vector("list",      p)
  col_type   <- vector("character", p)  # numeric, ordered, or categorical

  for (k in seq_len(p)) {
    v <- col_vecs[[k]]
    if (is.ordered(v)) {
      r              <- as.numeric(v)
      col_ranks[[k]] <- r
      col_ranges[k]  <- max(r, na.rm = TRUE) - min(r, na.rm = TRUE)
      col_type[k]    <- "ordered"
    } else if (is.numeric(v)) {
      col_ranges[k]  <- max(v, na.rm = TRUE) - min(v, na.rm = TRUE)
      col_type[k]    <- "numeric"
    } else {
      col_type[k]    <- "categorical"
    }
  }

  # Initialize distance matrix
  dist_matrix <- matrix(0, nrow = n, ncol = n)

  # Compute pairwise distances. seq_len() yields an empty sequence when
  # n < 2, so single-row input returns an empty dist instead of erroring.
  for (i in seq_len(n - 1)) {
    for (j in (i + 1):n) {

      total_dist <- 0
      valid_vars <- 0

      # Process each variable
      for (k in seq_len(p)) {

        # Plain vector indexing — avoids [.data.frame overhead on every access
        xi <- col_vecs[[k]][i]
        xj <- col_vecs[[k]][j]

        if (is.na(xi) || is.na(xj)) next

        valid_vars <- valid_vars + weights[k]

        # Column type already resolved — no S3 predicate calls in the hot path
        if (col_type[k] == "numeric") {
          d_k <- if (col_ranges[k] > 0) abs(xi - xj) / col_ranges[k] else 0

        } else if (col_type[k] == "ordered") {
          ri <- col_ranks[[k]][i]
          rj <- col_ranks[[k]][j]
          d_k <- if (col_ranges[k] > 0) abs(ri - rj) / col_ranges[k] else 0

        } else {
          d_k <- if (xi == xj) 0 else 1
        }

        total_dist <- total_dist + weights[k] * d_k
      }

      # Average over valid variables. A pair with no variable observed in
      # both has no distance, so it is NA, as in cluster::daisy(); the
      # matrix's starting 0 would make the two rows identical.
      dist_matrix[i, j] <- if (valid_vars > 0) {
        total_dist / valid_vars
      } else {
        NA_real_
      }
      dist_matrix[j, i] <- dist_matrix[i, j]  # Symmetric
    }
  }

  # Convert to dist object
  dist_obj <- stats::as.dist(dist_matrix)

  # Preserve row names if available
  if (!is.null(rownames(data))) {
    attr(dist_obj, "Labels") <- rownames(data)
  }

  attr(dist_obj, "method") <- "gower"

  dist_obj
}


#' Standardize Data
#'
#' Center and/or scale numeric variables
#'
#' @param data A data frame or tibble
#' @param center Logical; center variables? (default: TRUE)
#' @param scale Logical; scale variables to unit variance? (default: TRUE)
#'
#' @return A tibble with numeric variables centered and/or scaled as specified;
#'   non-numeric columns are returned unchanged. A grouped tibble is
#'   standardised within each group, as \code{dplyr::mutate()} works on it;
#'   a rowwise tibble is standardised over its whole columns, since a single
#'   value has no spread. Grouping and rowwise identifier columns are left
#'   as they are.
#'
#' @examples
#' \donttest{
#' std <- standardize_data(iris[, 1:4])
#' }
#'
#' @export
standardize_data <- function(data, center = TRUE, scale = TRUE) {

  standardise <- function(x) {
    if (center && scale) {
      as.numeric(base::scale(x, center = TRUE, scale = TRUE))
    } else if (center) {
      x - mean(x, na.rm = TRUE)
    } else if (scale) {
      x / stats::sd(x, na.rm = TRUE)
    } else {
      x
    }
  }

  # mutate() on a rowwise tibble works one row at a time, and one value has
  # no spread, so standardising row by row gives NaN everywhere. Its
  # columns are standardised whole, and the rowwise structure is put back.
  if (inherits(data, "rowwise_df")) {
    ids <- dplyr::group_vars(data)
    standardised <- dplyr::ungroup(data) |>
      dplyr::mutate(dplyr::across(
        where(is.numeric) & !dplyr::all_of(ids), standardise
      ))
    return(dplyr::rowwise(standardised, dplyr::all_of(ids)))
  }

  data |>
    dplyr::mutate(dplyr::across(where(is.numeric), standardise))
}


#' Compare Distance Methods
#'
#' Compute distances using multiple methods for comparison
#'
#' @param data A data frame or tibble
#' @param methods Character vector of methods to compare
#'
#' @return A named list of \code{\link[stats]{dist}} objects, one per method.
#'
#' @examples
#' \donttest{
#' dists <- compare_distances(
#'   iris[, 1:4], methods = c("euclidean", "manhattan")
#' )
#' }
#'
#' @export
compare_distances <- function(
    data,
    methods = c("euclidean", "manhattan", "maximum")) {

  data_numeric <- tl_select_columns(data)

  dist_list <- purrr::map(methods, function(method) {
    tidy_dist(data = data_numeric, method = method)
  })
  names(dist_list) <- methods

  dist_list
}


# ---- input handling shared by the unsupervised routines --------------

#' Drop dplyr grouping from an unsupervised routine's input
#'
#' dplyr adds a grouped tibble's grouping variables back to any column
#' selection ("Adding missing grouping variables"), so selecting the
#' numeric columns of \code{group_by(iris, Species)} returns Species too,
#' and a clustering would run on its factor codes. None of these routines
#' has a per-group meaning, so the grouping is dropped first.
#'
#' @param data Anything; only a data frame is changed
#' @return \code{data}, ungrouped when it is a data frame
#' @keywords internal
#' @noRd
tl_ungroup <- function(data) {
  if (is.data.frame(data)) dplyr::ungroup(data) else data
}

#' Choose the columns an unsupervised routine works on
#'
#' @param data A data frame
#' @param cols The caller's \code{cols} argument, captured with
#'   \code{rlang::enquo()} rather than tested with \code{is.null()}, which
#'   would evaluate it: a bare column name such as \code{Sepal.Length} is
#'   not an object in the caller's environment.
#' @param all_columns When \code{cols} is empty, TRUE keeps every column
#'   (Gower distance reads factors) and FALSE the numeric ones
#' @param numeric_only TRUE when the method can use numeric columns only.
#'   A non-numeric column \code{cols} selects is then left out with a
#'   warning, as one a formula names is, and a selection with no numeric
#'   column is refused.
#' @param what The method, as the messages should name it
#' @return The selected columns of the ungrouped data
#' @keywords internal
#' @noRd
tl_select_columns <- function(data, cols = rlang::quo(NULL),
                              all_columns = FALSE, numeric_only = FALSE,
                              what = NULL) {
  data <- tl_ungroup(data)
  cols <- tl_resolve_cols(cols, data)

  if (rlang::quo_is_null(cols)) {
    return(if (all_columns) data else dplyr::select(data, where(is.numeric)))
  }

  selected <- dplyr::select(data, !!cols)
  if (!numeric_only) {
    return(selected)
  }

  is_number <- vapply(selected, is.numeric, logical(1))
  if (!any(is_number)) {
    stop(
      what, " needs at least one numeric column, but 'cols' selected none",
      if (ncol(selected) > 0) {
        paste0("; it selected ", paste(names(selected), collapse = ", "))
      },
      ".",
      call. = FALSE
    )
  }
  if (!all(is_number)) {
    left_out <- names(selected)[!is_number]
    warning(
      what, " uses only numeric columns, so it left out the non-numeric ",
      "column", if (length(left_out) > 1) "s", " selected by 'cols': ",
      paste(left_out, collapse = ", "), ".",
      call. = FALSE
    )
  }

  selected[is_number]
}

#' Read a cols argument that a wrapper forwarded
#'
#' A function that takes \code{cols = NULL} and passes it on hands over the
#' symbol \code{cols}, not the NULL it holds. tidyselect reads such a
#' symbol by its value -- a NULL as an empty selection, which left zero
#' columns, and a character vector with a deprecation warning -- so the
#' value is read here instead. A symbol that names a column of the data is
#' that column, as tidyselect reads it; an expression that cannot be
#' evaluated outside a selection, such as \code{starts_with("x")}, is left
#' for tidyselect.
#'
#' @param cols A quosure of the caller's cols argument
#' @param data The data the selection is made from
#' @return A quosure: NULL for an empty selection, \code{all_of()} for a
#'   forwarded vector of names or positions, or \code{cols} as it was
#' @keywords internal
#' @noRd
tl_resolve_cols <- function(cols, data) {
  if (rlang::quo_is_null(cols)) {
    return(cols)
  }

  expr <- rlang::quo_get_expr(cols)
  if (is.symbol(expr) && as.character(expr) %in% names(data)) {
    return(cols)
  }

  value <- tryCatch(
    list(suppressWarnings(rlang::eval_tidy(cols, data = data))),
    error = function(e) NULL
  )
  if (is.null(value)) {
    return(cols)
  }

  value <- value[[1]]
  if (is.null(value)) {
    return(rlang::quo(NULL))
  }
  if (is.symbol(expr) && (is.character(value) || is.numeric(value))) {
    return(rlang::quo(dplyr::all_of(!!value)))
  }

  cols
}

#' Refuse pass-through options that break the result
#'
#' The tidy wrappers forward \code{...} to the routine they wrap, then read
#' its full result. An option that changes that result's shape, or one the
#' wrapper sets itself, is refused before the call.
#'
#' R matches an abbreviated argument name to the argument it starts, so
#' \code{cluster.o = TRUE} reaches \code{pam()} as \code{cluster.only =
#' TRUE}. The names in \code{...} are expanded the same way, with
#' \code{pmatch()} against the routine's arguments, before they are
#' checked. \code{pmatch()} leaves a prefix that two arguments share
#' unmatched, where R would pick the one the wrapper has not set itself;
#' for \code{pam()} and \code{clara()} those prefixes (\code{d}, \code{m},
#' \code{me}) match the wrapper's own \code{data} or \code{metric} first,
#' so they never reach \code{...}.
#'
#' @param dots The wrapper's \code{list(...)}
#' @param target The routine \code{...} is forwarded to
#' @param refused A named list with one element per option, itself a list
#'   of \code{reason}, why the option is refused, and \code{allowed}, the
#'   one value that leaves the result whole (the routine's default), or
#'   NULL to refuse every value
#' @param fun The wrapper's name, for the message
#' @return \code{TRUE}, invisibly, when no option is refused
#' @keywords internal
#' @noRd
tl_refuse_options <- function(dots, target, refused, fun) {
  given <- names(dots) %||% character(length(dots))
  options <- names(formals(target))
  full <- options[pmatch(given, options, duplicates.ok = TRUE)]

  for (i in which(full %in% names(refused))) {
    rule <- refused[[full[i]]]
    if (!is.null(rule$allowed) && identical(unname(dots[[i]]), rule$allowed)) {
      next
    }

    named <- if (given[i] == full[i]) {
      paste0("'", full[i], "'")
    } else {
      paste0("'", given[i], "', short for '", full[i], "',")
    }
    verdict <- if (is.null(rule$allowed)) {
      paste0(" cannot be passed to ", fun, "()")
    } else {
      paste0(" must be ", deparse(rule$allowed), " in ", fun, "(), its default")
    }
    stop(named, verdict, ": ", rule$reason, ".", call. = FALSE)
  }

  invisible(TRUE)
}

#' Columns a one-sided formula selects for an unsupervised method
#'
#' \code{get_formula_vars()} reads \code{~ .} as every numeric column,
#' which suits the methods that do arithmetic on the columns. Gower
#' distance is defined for factors too, so there a dot stands for every
#' column, less any the formula subtracts.
#'
#' A column the formula names but the method cannot use is reported with a
#' warning, since it was asked for by name and the fit goes ahead without
#' it. Columns a dot expanded to are not reported, since for these methods
#' the dot means the numeric columns.
#'
#' @param formula A one-sided formula
#' @param data The ungrouped training data
#' @param what The method, as the warning should name it
#' @param mixed_types TRUE when the method uses non-numeric columns
#' @param alternative The argument that would let the method use them,
#'   for the warning, or NULL when there is none
#' @return Column names to fit on
#' @keywords internal
#' @noRd
tl_formula_columns <- function(formula, data, what, mixed_types = FALSE,
                               alternative = NULL) {
  vars <- get_formula_vars(formula, data)

  if (mixed_types) {
    if ("." %in% all.vars(formula)) {
      labels <- attr(stats::terms(formula, data = data), "term.labels")
      vars <- unique(unlist(lapply(
        labels, function(label) all.vars(str2lang(label))
      )))
    }
    return(vars)
  }

  named <- intersect(intersect(vars, all.vars(formula)), names(data))
  dropped <- named[!vapply(data[named], is.numeric, logical(1))]

  if (length(dropped) > 0) {
    it <- if (length(dropped) == 1) "it" else "them"
    warning(
      what, " uses only numeric columns, so it ignored the formula's ",
      "non-numeric column", if (length(dropped) > 1) "s", ": ",
      paste(dropped, collapse = ", "), ". Remove ", it, " from the formula",
      if (is.null(alternative)) {
        paste0(", or convert ", it, " to numbers first.")
      } else {
        paste0(", or pass ", alternative, ", which can use ", it, ".")
      },
      call. = FALSE
    )
  }

  vars
}

#' Refuse a count argument that is not one whole number in range
#'
#' Ranges such as \code{2:max_k} run backwards when the bound is too small
#' -- \code{2:1} is \code{c(2, 1)} -- and a vector \code{k} makes
#' \code{cutree()} return a matrix, so a bad count would surface far from
#' the argument that caused it.
#'
#' @param x The value passed
#' @param arg The argument's name, for the message
#' @param min,max The accepted range
#' @return \code{TRUE}, invisibly
#' @keywords internal
#' @noRd
tl_check_whole_number <- function(x, arg, min = 1, max = Inf) {
  ok <- is.numeric(x) && length(x) == 1L && is.finite(x) &&
    x == round(x) && x >= min && x <= max

  if (!ok) {
    got <- if (length(x) == 0) {
      "nothing"
    } else {
      paste(utils::head(as.character(x), 5), collapse = ", ")
    }
    stop(
      "'", arg, "' must be a single whole number of at least ", min,
      if (is.finite(max)) paste0(" and at most ", max),
      ". Got: ", got, ".",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

#' Refuse a distance matrix with undefined entries
#'
#' A pair of rows with no variable observed in both has no distance:
#' \code{stats::dist()} and \code{tidy_gower()} return NA for it, as
#' \code{cluster::daisy()} does. \code{pam()} rejects that with "NA values
#' in the dissimilarity matrix not allowed" and \code{hclust()} with
#' "NA/NaN/Inf in foreign function call", neither of which says which rows
#' or why.
#'
#' @param dist_mat A dist object
#' @param what The method, for the message
#' @param alternative A way round the gap, for the message, or NULL
#' @return \code{TRUE}, invisibly, when every distance is defined
#' @keywords internal
#' @noRd
tl_check_complete_dist <- function(dist_mat, what, alternative = NULL) {
  if (!anyNA(dist_mat)) {
    return(invisible(TRUE))
  }

  undefined <- which(is.na(as.matrix(dist_mat)), arr.ind = TRUE)
  undefined <- undefined[undefined[, 1] < undefined[, 2], , drop = FALSE]
  shown <- utils::head(seq_len(nrow(undefined)), 3)

  stop(
    what, " cannot use undefined distances: ", nrow(undefined),
    if (nrow(undefined) == 1) " pair of rows has" else " pairs of rows have",
    " no variable observed in both (",
    paste(
      sprintf("rows %d and %d", undefined[shown, 1], undefined[shown, 2]),
      collapse = "; "
    ),
    if (nrow(undefined) > length(shown)) "; ..." else "",
    "). Drop or impute the missing values in those rows first",
    if (is.null(alternative)) "." else paste0(", or ", alternative, "."),
    call. = FALSE
  )
}
