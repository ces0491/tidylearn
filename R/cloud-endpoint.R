#' @title Cloud endpoint resolution for tidylearn
#' @name tidylearn-cloud-endpoint
#' @description Resolving and validating the Modal Web Function endpoint
#'   that `compute = "cloud"` submits to.
NULL

# Hosts tidylearn is willing to send user data to by default. Deployed
# Modal Web Functions live under *.modal.run; modal.com covers Modal's
# own API hosts. Any Modal workspace's endpoint matches *.modal.run, not
# only the user's own. Modal customers on custom domains can add to this
# with tl_cloud_allow_host(), which is a deliberate per-session action --
# see T9 in inst/security/threat-model.md.
.tl_modal_hosts <- c("modal.run", "modal.com")

#' Validate a host name offered for the allowlist
#'
#' Rejects anything that is not a bare host: URLs, ports, paths,
#' wildcards, empty labels, and names of fewer than three labels. A
#' single label such as `"com"` would widen the allowlist to an entire
#' TLD, and two labels can be a public suffix such as `"co.uk"` or
#' `"github.io"`, under which anyone can register a site; either defeats
#' T9. tidylearn has no copy of the Public Suffix List, so the label
#' count is the rule: it refuses every two-label suffix, and a longer
#' public suffix still passes.
#'
#' The trailing dot of a fully qualified name (`"fits.example.com."`) is
#' dropped: an endpoint's host never carries it, so a host stored with it
#' matched nothing.
#'
#' @param host A character vector of candidate host names.
#' @return The hosts, lower-cased and without a trailing dot. Errors if any
#'   is unacceptable.
#' @keywords internal
#' @noRd
tl_validate_host_name <- function(host) {
  if (!is.character(host) || length(host) == 0L || anyNA(host) ||
        !all(nzchar(host))) {
    stop(
      "'host' must be a character vector of non-empty host names.",
      call. = FALSE
    )
  }

  host <- sub("\\.$", "", tolower(trimws(host)))

  for (h in host) {
    if (grepl("[/@?#*[:space:]]|://|:[0-9]+$", h)) {
      stop(
        "'", h, "' is not a bare host name. Give a host such as ",
        "'fits.example.com', not a URL, port, path or wildcard.",
        call. = FALSE
      )
    }

    labels <- strsplit(h, ".", fixed = TRUE)[[1]]

    # strsplit() drops a final empty field, so a dot left at the end after
    # the root dot was removed is checked for directly
    if (any(!nzchar(labels)) || endsWith(h, ".")) {
      stop(
        "'", h, "' is not a valid host name: it has an empty label.",
        call. = FALSE
      )
    }

    # Two labels passed, so allowing co.uk let attacker.co.uk through
    if (length(labels) < 3L) {
      stop(
        "'", h, "' is too broad to allow: a name with fewer than three ",
        "labels can be a top-level domain or a public suffix such as ",
        "'co.uk' or 'github.io', and allowing it would admit every site ",
        "under it. Give the endpoint's full host name, such as ",
        "'fits.example.com'.",
        call. = FALSE
      )
    }
  }

  host
}

#' Allow an additional host for cloud uploads in this R session
#'
#' tidylearn uploads training data only to Modal's own hosts
#' (`*.modal.run`, `*.modal.com`). Modal customers serving Web Functions
#' from a custom domain can add that domain here.
#'
#' This widens the set of destinations your data may be sent to, so it
#' is deliberately a per-session call rather than an option or an
#' environment variable: a shared `.Rprofile` or an inherited environment
#' should not be able to add a destination without you writing the call.
#' Additions are never persisted and are forgotten when the session ends.
#'
#' Hosts match themselves and their subdomains. Adding
#' `"fits.example.com"` accepts `https://fits.example.com` and
#' `https://a.fits.example.com`, and nothing else.
#'
#' Give the endpoint's full host name. A name of fewer than three labels,
#' such as `"example.com"` or `"co.uk"`, is refused: two labels can be a
#' public suffix, under which anyone can register a site, and allowing
#' one would admit all of them.
#'
#' @param host A character vector of host names to allow, or `NULL` to
#'   clear every host added this session. Bare host names of at least
#'   three labels only — not URLs, ports, paths or wildcards.
#' @return The full allowlist after the change, invisibly.
#' @seealso [tl_cloud_allowed_hosts()], and T9 in
#'   `system.file("security/threat-model.md", package = "tidylearn")`.
#' @export
#' @examples
#' tl_cloud_allow_host("fits.example.com")
#' tl_cloud_allowed_hosts()
#' tl_cloud_allow_host(NULL)
tl_cloud_allow_host <- function(host) {
  if (is.null(host)) {
    .tl_cloud_state$extra_hosts <- character(0)
    message("Cleared all additional cloud hosts for this R session.")
    return(invisible(tl_cloud_allowed_hosts()))
  }

  host <- tl_validate_host_name(host)

  .tl_cloud_state$extra_hosts <- union(
    .tl_cloud_state$extra_hosts, host
  )

  message(
    "Cloud uploads may now also go to: ",
    paste(host, collapse = ", "),
    ". This is in addition to Modal's own hosts and lasts for this R ",
    "session only."
  )

  invisible(tl_cloud_allowed_hosts())
}

#' Hosts tidylearn will currently upload to
#'
#' Modal's own hosts, plus anything added with [tl_cloud_allow_host()]
#' during this session.
#'
#' @return A character vector of host names.
#' @seealso [tl_cloud_allow_host()]
#' @export
#' @examples
#' tl_cloud_allowed_hosts()
tl_cloud_allowed_hosts <- function() {
  union(.tl_modal_hosts, .tl_cloud_state$extra_hosts)
}

# The environment variable tidylearn reads for the endpoint. A
# tidylearn-owned name on purpose: it must not collide with, or be
# mistaken for, the Modal account credentials that the Modal CLI keeps
# in its own config file. tidylearn never reads those (T1).
.tl_cloud_endpoint_var <- "TIDYLEARN_MODAL_ENDPOINT"

#' Is a host on the current allowlist?
#'
#' Matches the host itself or any subdomain of it. The subdomain test is
#' anchored on a leading dot so that lookalikes such as
#' `modal.run.example.com` or `evil-modal.run` do not match.
#'
#' @param host A single host name, lower case.
#' @return `TRUE` if the host is allowed as an upload destination.
#' @keywords internal
#' @noRd
tl_is_allowed_host <- function(host) {
  if (!is.character(host) || length(host) != 1L || is.na(host) ||
        !nzchar(host)) {
    return(FALSE)
  }

  allowed <- tl_cloud_allowed_hosts()

  any(vapply(
    allowed,
    function(domain) {
      identical(host, domain) ||
        endsWith(host, paste0(".", domain))
    },
    logical(1)
  ))
}

#' Is a host one of Modal's own, rather than a session addition?
#'
#' Used to flag a non-default destination in the pre-upload summary, so
#' an added host is visible at the moment consent is acted on.
#'
#' @param host A single host name, lower case.
#' @return `TRUE` if the host belongs to Modal itself.
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

  if (!tl_is_allowed_host(host)) {
    stop(
      "Cloud endpoint host '", host, "' is not an allowed upload ",
      "destination. tidylearn uploads only to ",
      paste0("*.", tl_cloud_allowed_hosts(), collapse = " or "),
      ". Check ", .tl_cloud_endpoint_var,
      ", or allow a custom Modal domain with tl_cloud_allow_host().",
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
