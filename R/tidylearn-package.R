#' @details
#' tidylearn wraps established R machine learning packages behind one
#' consistent interface. The main entry points are:
#'
#' \describe{
#'   \item{\code{\link{tl_read}}}{Read data from files, databases and
#'     cloud sources into a tidy tibble.}
#'   \item{\code{\link{tl_model}}}{Fit any supported supervised or
#'     unsupervised method.}
#'   \item{\code{\link{tl_evaluate}}}{Score a fitted model.}
#'   \item{\code{\link{tl_table}}}{Render results as formatted
#'     \pkg{gt} tables.}
#'   \item{\code{\link{tl_auto_ml}}}{Search across methods automatically.}
#' }
#'
#' Every fitted model keeps the underlying package's own object in its
#' \code{$fit} slot, so package-specific functionality remains available.
#'
#' See \code{vignette("getting-started", package = "tidylearn")} for a
#' walkthrough.
#'
#' @keywords internal
"_PACKAGE"
