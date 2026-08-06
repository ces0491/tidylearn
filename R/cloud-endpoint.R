#' @title Cloud endpoint resolution for tidylearn
#' @name tidylearn-cloud-endpoint
#' @description Resolving and validating the Modal Web Function endpoint
#'   that `compute = "cloud"` submits to.
NULL

# Hosts tidylearn is willing to send user data to. Deployed Modal Web
# Functions live under *.modal.run; modal.com covers Modal's own API
# hosts. Deliberately a fixed constant rather than user-extensible --
# see T9 in inst/security/threat-model.md. Extending this for Modal
# customers on custom domains would need its own opt-in.
.tl_modal_hosts <- c("modal.run", "modal.com")

# The environment variable tidylearn reads for the endpoint. A
# tidylearn-owned name on purpose: it must not collide with, or be
# mistaken for, the Modal account credentials that the Modal CLI keeps
# in its own config file. tidylearn never reads those (T1).
.tl_cloud_endpoint_var <- "TIDYLEARN_MODAL_ENDPOINT"

#' Is a host one of Modal's?
#'
#' Matches the host itself or any subdomain of it. The subdomain test is
#' anchored on a leading dot so that lookalikes such as
#' `modal.run.example.com` or `evil-modal.run` do not match.
#'
#' @param host A single host name, lower case.
#' @return `TRUE` if the host belongs to Modal.
#' @keywords internal
#' @noRd
tl_is_modal_host <- function(host) {
  if (!is.character(host) || length(host) != 1L || is.na(host) ||
        !nzchar(host)) {
    return(FALSE)
  }

  any(vapply(
    .tl_modal_hosts,
    function(domain) {
      identical(host, domain) ||
        endsWith(host, paste0(".", domain))
    },
    logical(1)
  ))
}

#' Validate a Modal endpoint URL before any data crosses the network
#'
#' The endpoint is user-supplied configuration, so a typo or a modified
#' environment variable could otherwise send training data to a host
#' that is not Modal. This is the single choke point that prevents that
#' (T9); every cloud request must be built from its return value.
#'
#' @param url A single URL string.
#' @return The URL, invisibly, if it is acceptable. Errors otherwise.
#' @keywords internal
#' @noRd
tl_validate_modal_url <- function(url) {
  if (!is.character(url) || length(url) != 1L || is.na(url) ||
        !nzchar(url)) {
    stop(
      "Cloud endpoint must be a single non-empty URL string.",
      call. = FALSE
    )
  }

  tl_check_packages("httr2")
  parsed <- tryCatch(
    httr2::url_parse(url),
    error = function(e) {
      stop("Cloud endpoint is not a valid URL: ", url, call. = FALSE)
    }
  )

  if (!identical(parsed$scheme, "https")) {
    stop(
      "Cloud endpoint must use https. Got: ",
      if (is.null(parsed$scheme)) "no scheme" else parsed$scheme,
      ". Training data is never sent over an unencrypted connection.",
      call. = FALSE
    )
  }

  host <- tolower(parsed$hostname %||% "")

  if (!tl_is_modal_host(host)) {
    stop(
      "Cloud endpoint host '", host, "' is not a Modal host. ",
      "tidylearn only uploads data to ",
      paste0("*.", .tl_modal_hosts, collapse = " or "),
      ". Check ", .tl_cloud_endpoint_var, ".",
      call. = FALSE
    )
  }

  invisible(url)
}

#' Resolve the configured Modal endpoint
#'
#' Read from an environment variable rather than an R option: an option
#' can be set silently by a shared `.Rprofile`, which is a weaker
#' position against T9.
#'
#' @return The validated endpoint URL.
#' @keywords internal
#' @noRd
tl_cloud_endpoint <- function() {
  url <- Sys.getenv(.tl_cloud_endpoint_var, unset = "")

  if (!nzchar(url)) {
    stop(
      "No cloud endpoint configured. Set ", .tl_cloud_endpoint_var,
      " to the URL of the Modal Web Function you deployed with ",
      "tl_cloud_setup().",
      call. = FALSE
    )
  }

  tl_validate_modal_url(url)

  url
}
