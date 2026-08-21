#' @title Cost controls for tidylearn cloud compute
#' @name tidylearn-cloud-cost
#' @description Bounding what a cloud fit can cost, and keeping in-flight
#'   jobs visible.
NULL

# A submitted Modal job runs to completion on Modal whatever the R
# session does afterwards. Ctrl-C, a closed IDE, a crashed session, a
# closed laptop -- none of them stop it, because the session was only
# ever polling for a result, never keeping the work alive.
#
# That has a consequence worth stating plainly: every client-side
# control here is best-effort. The only two things that bound spend when
# the client dies are the timeout set on the job at submission, and the
# spend budget the user sets on their own Modal workspace. So the
# timeout is the primary control, not the fallback, and tidylearn never
# submits without one.
#
# Modal's own default is 300s and its ceiling is 24h. tidylearn does not
# inherit either: it derives a timeout from the advisor's estimate and
# caps it well below Modal's maximum.

# Multiple of the estimated runtime allowed before the job is killed.
# The advisor is order-of-magnitude, so some headroom is needed; 3x is
# enough to absorb a bad estimate without turning a 10-minute job into
# an overnight one.
.tl_cloud_timeout_factor <- 3

# Never below this -- a job killed during cold start bills for nothing
# useful.
.tl_cloud_timeout_floor <- 60

# Never above this without the user deliberately raising it.
.tl_cloud_timeout_cap_default <- 3600

# Worst-case spend allowed for a single call before tidylearn refuses.
.tl_cloud_max_cost_default <- 5

#' Derive the timeout to set on a cloud job
#'
#' @param est_seconds Estimated total runtime from the advisor.
#' @param timeout_cap Upper bound, in seconds.
#' @return An integer number of seconds.
#' @keywords internal
#' @noRd
tl_cloud_timeout_seconds <- function(est_seconds,
                                     timeout_cap =
                                       .tl_cloud_timeout_cap_default) {
  if (!is.numeric(est_seconds) || length(est_seconds) != 1L ||
        is.na(est_seconds) || est_seconds < 0) {
    stop("'est_seconds' must be a single non-negative number.",
         call. = FALSE)
  }
  if (!is.numeric(timeout_cap) || length(timeout_cap) != 1L ||
        is.na(timeout_cap) || timeout_cap <= 0) {
    stop("'timeout_cap' must be a single positive number of seconds.",
         call. = FALSE)
  }

  wanted <- ceiling(est_seconds * .tl_cloud_timeout_factor)

  as.integer(min(timeout_cap, max(.tl_cloud_timeout_floor, wanted)))
}

#' Worst-case cost of a cloud job
#'
#' What the call bills if it runs until the timeout kills it. This is the
#' number the user is asked to accept, not the expected cost: the
#' estimate is order-of-magnitude, the timeout is the actual bound.
#'
#' @param timeout_seconds Timeout that will be set on the job.
#' @param tier_name Name of the Modal tier, as chosen by the advisor.
#' @return Cost in USD.
#' @keywords internal
#' @noRd
tl_cloud_worst_case_cost <- function(timeout_seconds, tier_name) {
  tier <- .tl_modal_tiers[[tier_name]]

  if (is.null(tier)) {
    stop("Unknown Modal tier: '", tier_name, "'.", call. = FALSE)
  }

  timeout_seconds * tier$rate_per_sec
}

#' Pre-flight cost check for a cloud submission
#'
#' Refuses two situations outright:
#'
#' * an estimate that already exceeds the timeout cap, because the job
#'   would be killed before finishing and bill for the full cap having
#'   produced nothing;
#' * a worst-case cost above `max_cost`.
#'
#' @param advice A `tidylearn_compute_advice` object.
#' @param max_cost Maximum acceptable worst-case spend, in USD.
#' @param timeout_cap Maximum timeout, in seconds.
#' @return A list with `timeout_seconds`, `worst_case_cost`,
#'   `expected_cost` and `tier_label`.
#' @keywords internal
#' @noRd
tl_cloud_check_budget <- function(advice,
                                  max_cost = .tl_cloud_max_cost_default,
                                  timeout_cap =
                                    .tl_cloud_timeout_cap_default) {
  if (!inherits(advice, "tidylearn_compute_advice")) {
    stop("'advice' must be a tidylearn_compute_advice object.",
         call. = FALSE)
  }
  if (!is.numeric(max_cost) || length(max_cost) != 1L ||
        is.na(max_cost) || max_cost <= 0) {
    stop("'max_cost' must be a single positive number of dollars.",
         call. = FALSE)
  }

  cloud <- advice$cloud
  est_seconds <- cloud$est_seconds

  if (est_seconds >= timeout_cap) {
    stop(
      "This fit is estimated at ", tl_cloud_format_duration(est_seconds),
      ", which is at or beyond the ",
      tl_cloud_format_duration(timeout_cap), " timeout cap. Submitting ",
      "it would bill for the full timeout and then kill the job before ",
      "it finished. Reduce the problem, or raise 'timeout_cap' ",
      "deliberately if you mean to run something this long.",
      call. = FALSE
    )
  }

  timeout_seconds <- tl_cloud_timeout_seconds(est_seconds, timeout_cap)
  worst_case <- tl_cloud_worst_case_cost(timeout_seconds,
                                         cloud$tier_name)

  if (worst_case > max_cost) {
    stop(
      "Worst-case cost for this fit is ",
      tl_cloud_format_cost(worst_case), " (tier ", cloud$tier_label,
      ", timeout ", tl_cloud_format_duration(timeout_seconds),
      "), above the ", tl_cloud_format_cost(max_cost), " limit. ",
      "Raise 'max_cost' if that is acceptable. This is the most the ",
      "call can bill, not the expected cost, which is ",
      tl_cloud_format_cost(cloud$est_cost_usd), ".",
      call. = FALSE
    )
  }

  list(
    timeout_seconds = timeout_seconds,
    worst_case_cost = worst_case,
    expected_cost   = cloud$est_cost_usd,
    tier_label      = cloud$tier_label
  )
}

#' Format a duration for user-facing messages
#'
#' @param seconds A number of seconds.
#' @return A short string.
#' @keywords internal
#' @noRd
tl_cloud_format_duration <- function(seconds) {
  if (seconds < 60) {
    return(paste0(round(seconds), "s"))
  }
  if (seconds < 3600) {
    return(paste0(round(seconds / 60, 1), " min"))
  }
  paste0(round(seconds / 3600, 1), " h")
}

#' Format a cost for user-facing messages
#'
#' @param usd Cost in dollars.
#' @return A short string.
#' @keywords internal
#' @noRd
tl_cloud_format_cost <- function(usd) {
  if (usd < 0.01) {
    return("<$0.01")
  }
  paste0("$", format(round(usd, 2), nsmall = 2, trim = TRUE))
}

#' Build the pre-upload summary
#'
#' Metadata only -- dimensions, tier labels, times and costs. Never row
#' values (T6). Shows the destination host and flags it when it is one
#' the user added rather than Modal's own (T9), and states the worst-case
#' spend rather than only the estimate.
#'
#' @param method Method name.
#' @param host Destination host.
#' @param n_rows,n_cols Problem dimensions.
#' @param size_mb Estimated upload size.
#' @param budget Result of `tl_cloud_check_budget()`.
#' @param est_seconds Estimated runtime.
#' @return A character vector of lines.
#' @keywords internal
#' @noRd
tl_cloud_upload_summary <- function(method, host, n_rows, n_cols,
                                    size_mb, budget, est_seconds) {
  destination <- host
  if (!tl_is_modal_host(host)) {
    destination <- paste0(host, "  [host added this session]")
  }

  c(
    "Uploading to Modal:",
    paste0("  Method:        ", method),
    paste0("  Destination:   ", destination),
    paste0("  Rows x cols:   ", format(n_rows, big.mark = ","), " x ",
           n_cols),
    paste0("  Estimated MB:  ", round(size_mb)),
    paste0("  Modal tier:    ", budget$tier_label),
    paste0("  Estimated:     ", tl_cloud_format_duration(est_seconds),
           ", ", tl_cloud_format_cost(budget$expected_cost)),
    paste0("  Timeout:       ",
           tl_cloud_format_duration(budget$timeout_seconds),
           " (job is killed at this point)"),
    paste0("  Most it can bill: ",
           tl_cloud_format_cost(budget$worst_case_cost))
  )
}

#' Record a submitted job so it cannot run invisibly
#'
#' @param call_id Modal call id.
#' @param method Method name.
#' @param timeout_seconds Timeout set on the job.
#' @param worst_case_cost Worst-case spend.
#' @return Invisibly, the call id.
#' @keywords internal
#' @noRd
tl_cloud_register_job <- function(call_id, method, timeout_seconds,
                                  worst_case_cost) {
  .tl_cloud_state$jobs[[call_id]] <- list(
    call_id         = call_id,
    method          = method,
    submitted_at    = Sys.time(),
    timeout_seconds = timeout_seconds,
    worst_case_cost = worst_case_cost
  )

  invisible(call_id)
}

#' Forget a job that has completed or been cancelled
#'
#' @param call_id Modal call id.
#' @return Invisibly, `TRUE` if a job was removed.
#' @keywords internal
#' @noRd
tl_cloud_forget_job <- function(call_id) {
  existed <- !is.null(.tl_cloud_state$jobs[[call_id]])
  .tl_cloud_state$jobs[[call_id]] <- NULL

  invisible(existed)
}

#' Cloud jobs submitted in this R session
#'
#' Every cloud submission is recorded here so that no job runs
#' invisibly. A job disappears from this list when its result is
#' collected or it is cancelled.
#'
#' A submitted job runs on Modal whatever this R session does. If the
#' session ends while a job is still listed here, that job keeps running
#' and keeps billing until it finishes or its timeout kills it — this
#' list cannot survive the session, but the timeout does.
#'
#' @return A tibble with one row per in-flight job: `call_id`, `method`,
#'   `submitted_at`, `timeout_seconds` and `worst_case_cost`. Zero rows
#'   when nothing is in flight.
#' @export
#' @examples
#' # Nothing in flight in a fresh session
#' tl_cloud_jobs()
tl_cloud_jobs <- function() {
  jobs <- .tl_cloud_state$jobs

  if (length(jobs) == 0L) {
    return(tibble::tibble(
      call_id         = character(0),
      method          = character(0),
      submitted_at    = as.POSIXct(character(0)),
      timeout_seconds = integer(0),
      worst_case_cost = numeric(0)
    ))
  }

  tibble::tibble(
    call_id         = vapply(jobs, function(j) j$call_id, character(1)),
    method          = vapply(jobs, function(j) j$method, character(1)),
    submitted_at    = as.POSIXct(
      vapply(jobs, function(j) as.numeric(j$submitted_at), numeric(1)),
      origin = "1970-01-01"
    ),
    timeout_seconds = vapply(
      jobs, function(j) as.integer(j$timeout_seconds), integer(1)
    ),
    worst_case_cost = vapply(
      jobs, function(j) as.numeric(j$worst_case_cost), numeric(1)
    )
  )
}
