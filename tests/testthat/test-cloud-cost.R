# ---- Cloud cost controls ----
#
# A submitted Modal job runs to completion whatever the R session does,
# so the timeout set at submission is the only thing that bounds spend
# when the client goes away. These tests cover the pre-flight gates that
# decide that timeout and refuse the submission when the worst case is
# unacceptable. T10 in inst/security/threat-model.md.

# ---- Timeout derivation ----

test_that("the timeout gives headroom over the estimate", {
  # 3x the estimate, so a bad order-of-magnitude estimate still finishes.
  expect_equal(tl_cloud_timeout_seconds(100), 300L)
  expect_equal(tl_cloud_timeout_seconds(200), 600L)
})

test_that("the timeout has a floor", {
  # A job killed during cold start bills for nothing useful.
  expect_equal(tl_cloud_timeout_seconds(1), 60L)
  expect_equal(tl_cloud_timeout_seconds(0), 60L)
})

test_that("the timeout is capped", {
  expect_equal(tl_cloud_timeout_seconds(5000, timeout_cap = 3600), 3600L)
  expect_equal(tl_cloud_timeout_seconds(100, timeout_cap = 120), 120L)
})

test_that("timeout arguments are validated", {
  expect_error(tl_cloud_timeout_seconds(-1), "non-negative")
  expect_error(tl_cloud_timeout_seconds(NA_real_), "non-negative")
  expect_error(tl_cloud_timeout_seconds("100"), "non-negative")
  expect_error(tl_cloud_timeout_seconds(100, timeout_cap = 0), "positive")
})

# ---- Worst-case cost ----

test_that("worst-case cost is the timeout at the tier's rate", {
  # Not the estimate: the estimate is order-of-magnitude, the timeout is
  # what actually bounds the bill.
  rate <- .tl_modal_tiers[["cpu-large"]]$rate_per_sec
  expect_equal(
    tl_cloud_worst_case_cost(600, "cpu-large"),
    600 * rate
  )
})

test_that("an unknown tier is an error, not a zero cost", {
  expect_error(tl_cloud_worst_case_cost(600, "no-such-tier"), "Unknown Modal tier")
})

# ---- Budget gate ----

fake_advice <- function(est_seconds, est_cost = 0.10,
                        tier = "cpu-large") {
  structure(
    list(
      problem = list(),
      cloud = list(
        est_seconds  = est_seconds,
        est_cost_usd = est_cost,
        tier_name    = tier,
        tier_label   = .tl_modal_tiers[[tier]]$label
      )
    ),
    class = "tidylearn_compute_advice"
  )
}

test_that("an affordable job passes and reports its bound", {
  budget <- tl_cloud_check_budget(fake_advice(120), max_cost = 5)

  expect_equal(budget$timeout_seconds, 360L)
  expect_equal(
    budget$worst_case_cost,
    360 * .tl_modal_tiers[["cpu-large"]]$rate_per_sec
  )
  expect_equal(budget$expected_cost, 0.10)
})

test_that("a job that cannot finish inside the cap is refused", {
  # Submitting this would bill the full timeout and then kill the job
  # before it produced anything.
  expect_error(
    tl_cloud_check_budget(fake_advice(4000), timeout_cap = 3600),
    "beyond the.*timeout cap"
  )
})

test_that("a job above the cost ceiling is refused", {
  # cpu-xlarge is the most expensive CPU tier; a long timeout on it
  # should breach a low ceiling.
  expect_error(
    tl_cloud_check_budget(
      fake_advice(1000, tier = "cpu-xlarge"),
      max_cost = 0.05
    ),
    "Worst-case cost"
  )
})

test_that("the refusal names the worst case, not the estimate", {
  err <- tryCatch(
    tl_cloud_check_budget(
      fake_advice(1000, est_cost = 0.01, tier = "cpu-xlarge"),
      max_cost = 0.05
    ),
    error = function(e) conditionMessage(e)
  )

  expect_match(err, "most the call can bill")
  expect_match(err, "Raise 'max_cost'")
})

test_that("budget arguments are validated", {
  expect_error(tl_cloud_check_budget(list()), "compute_advice")
  expect_error(
    tl_cloud_check_budget(fake_advice(100), max_cost = 0),
    "positive number of dollars"
  )
  expect_error(
    tl_cloud_check_budget(fake_advice(100), max_cost = -1),
    "positive number of dollars"
  )
})

# ---- Formatting ----

test_that("durations and costs read sensibly", {
  expect_equal(tl_cloud_format_duration(45), "45s")
  expect_equal(tl_cloud_format_duration(90), "1.5 min")
  expect_equal(tl_cloud_format_duration(5400), "1.5 h")

  expect_equal(tl_cloud_format_cost(0.001), "<$0.01")
  expect_equal(tl_cloud_format_cost(0.4267), "$0.43")
  expect_equal(tl_cloud_format_cost(12), "$12.00")
})

# ---- Pre-upload summary ----

test_that("the summary states the destination and the worst case", {
  budget <- tl_cloud_check_budget(fake_advice(120), max_cost = 5)
  lines <- tl_cloud_upload_summary(
    method = "xgboost", host = "ws--fit.modal.run",
    n_rows = 1500000, n_cols = 47, size_mb = 564,
    budget = budget, est_seconds = 120
  )
  txt <- paste(lines, collapse = "\n")

  expect_match(txt, "ws--fit\\.modal\\.run")
  expect_match(txt, "1,500,000 x 47")
  expect_match(txt, "Most it can bill")
  expect_match(txt, "killed at this point")

  # Metadata only -- no row values anywhere (T6).
  expect_false(grepl("[0-9]+\\.[0-9]{4,}", txt))
})

test_that("a session-added destination is flagged in the summary", {
  on.exit(suppressMessages(tl_cloud_allow_host(NULL)))
  suppressMessages(tl_cloud_allow_host("fits.example.com"))

  budget <- tl_cloud_check_budget(fake_advice(120), max_cost = 5)
  lines <- tl_cloud_upload_summary(
    method = "forest", host = "fits.example.com",
    n_rows = 100, n_cols = 4, size_mb = 1,
    budget = budget, est_seconds = 120
  )

  expect_match(paste(lines, collapse = "\n"), "host added this session")
})

# ---- Job registry ----

test_that("a fresh session has no jobs in flight", {
  .tl_cloud_state$jobs <- list()
  jobs <- tl_cloud_jobs()

  expect_s3_class(jobs, "tbl_df")
  expect_equal(nrow(jobs), 0L)
  expect_equal(
    names(jobs),
    c("call_id", "method", "submitted_at", "timeout_seconds",
      "worst_case_cost")
  )
})

test_that("submitted jobs are visible and forgettable", {
  .tl_cloud_state$jobs <- list()
  on.exit(.tl_cloud_state$jobs <- list())

  tl_cloud_register_job("fc-001", "xgboost", 600L, 0.20)
  tl_cloud_register_job("fc-002", "forest", 300L, 0.10)

  jobs <- tl_cloud_jobs()
  expect_equal(nrow(jobs), 2L)
  expect_setequal(jobs$call_id, c("fc-001", "fc-002"))
  expect_setequal(jobs$method, c("xgboost", "forest"))
  expect_true(all(jobs$worst_case_cost > 0))

  expect_true(tl_cloud_forget_job("fc-001"))
  expect_equal(nrow(tl_cloud_jobs()), 1L)

  # Forgetting something unknown is not an error, just FALSE.
  expect_false(tl_cloud_forget_job("fc-999"))
})
