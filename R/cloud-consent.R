#' @title Cloud data-egress consent for tidylearn
#' @name tidylearn-cloud-consent
#' @description The consent gate that stands between a `compute =
#'   "cloud"` call and any training data leaving the user's machine.
NULL

# Session-level consent lock. Deliberately an environment in the package
# namespace and nothing else: not an option, not a file, not a cached
# credential. It dies with the R session (T2).
.tl_cloud_state <- new.env(parent = emptyenv())
.tl_cloud_state$consent <- FALSE

#' Grant or revoke cloud upload consent for this R session
#'
#' Fitting with `compute = "cloud"` uploads your training data to your
#' own Modal account, which is a third party. tidylearn will not do that
#' without explicit consent on every call.
#'
#' There are two ways to give it. Pass `confirm_upload = TRUE` to each
#' `tl_model()` call, or call `tl_cloud_consent()` once to opt in for the
#' rest of the session. The session lock exists for batch and
#' non-interactive work, where a per-call argument is repetitive.
#'
#' The lock is **not** persisted. It is forgotten when the R session
#' ends, and it is never written to disk. Revoke it early with
#' `tl_cloud_consent(FALSE)`.
#'
#' tidylearn never prompts interactively for consent, so scripts, CI and
#' `Rscript` behave the same as an interactive session.
#'
#' @param consent `TRUE` to allow cloud uploads for the rest of the
#'   session, `FALSE` to revoke.
#' @return The previous consent state, invisibly.
#' @seealso The full contract is in
#'   `system.file("security/threat-model.md", package = "tidylearn")`.
#' @export
#' @examples
#' # Opt in for the session, then revoke.
#' old <- tl_cloud_consent(TRUE)
#' tl_cloud_consent(FALSE)
tl_cloud_consent <- function(consent = TRUE) {
  if (!is.logical(consent) || length(consent) != 1L || is.na(consent)) {
    stop("'consent' must be TRUE or FALSE.", call. = FALSE)
  }

  previous <- .tl_cloud_state$consent
  .tl_cloud_state$consent <- consent

  if (consent) {
    message(
      "Cloud uploads enabled for this R session. Training data passed to ",
      "tl_model(compute = 'cloud') will be sent to your Modal account. ",
      "Revoke with tl_cloud_consent(FALSE)."
    )
  } else {
    message("Cloud uploads disabled for this R session.")
  }

  invisible(previous)
}

#' Is the session-level consent lock in place?
#'
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_cloud_consent_active <- function() {
  isTRUE(.tl_cloud_state$consent)
}

#' Gate a cloud submission on consent
#'
#' Called before anything is serialised or transmitted. Errors unless
#' the caller passed `confirm_upload = TRUE` or the session lock is in
#' place (T2).
#'
#' @param confirm_upload The per-call argument, as supplied by the user.
#' @return `TRUE`, invisibly, when the upload may proceed.
#' @keywords internal
#' @noRd
tl_cloud_require_consent <- function(confirm_upload = FALSE) {
  if (!is.logical(confirm_upload) || length(confirm_upload) != 1L ||
        is.na(confirm_upload)) {
    stop("'confirm_upload' must be TRUE or FALSE.", call. = FALSE)
  }

  if (isTRUE(confirm_upload) || tl_cloud_consent_active()) {
    return(invisible(TRUE))
  }

  stop(
    "Cloud compute would upload your training data to a third party ",
    "(your Modal account) and consent has not been given. Pass ",
    "confirm_upload = TRUE to this call, or call tl_cloud_consent() ",
    "once to opt in for the session. See ",
    "system.file('security/threat-model.md', package = 'tidylearn').",
    call. = FALSE
  )
}
