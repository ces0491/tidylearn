# ---- Cloud security guards: endpoint validation and consent ----
#
# These implement two commitments in inst/security/threat-model.md and
# are the checks that stand between a cloud fit and user data leaving the
# machine. T9 (egress to a non-Modal host) and T2 (silent upload).

# ---- T9: host allowlist ----

test_that("Modal hosts and their subdomains are accepted", {
  expect_true(tl_is_modal_host("modal.run"))
  expect_true(tl_is_modal_host("modal.com"))
  expect_true(tl_is_modal_host("my-workspace--tidylearn-fit.modal.run"))
  expect_true(tl_is_modal_host("api.modal.com"))
})

test_that("lookalike hosts are rejected", {
  # Suffix-shaped attacks: the allowlisted domain appears in the name but
  # the request would not reach Modal.
  expect_false(tl_is_modal_host("modal.run.example.com"))
  expect_false(tl_is_modal_host("evil-modal.run"))
  expect_false(tl_is_modal_host("notmodal.com"))
  expect_false(tl_is_modal_host("modal.run.evil.test"))
  expect_false(tl_is_modal_host("example.com"))
})

test_that("degenerate hosts are rejected", {
  expect_false(tl_is_modal_host(""))
  expect_false(tl_is_modal_host(NA_character_))
  expect_false(tl_is_modal_host(character(0)))
  expect_false(tl_is_modal_host(c("modal.run", "modal.run")))
})

# ---- T9: URL validation ----

test_that("a well-formed Modal https URL validates", {
  skip_if_not_installed("httr2")

  url <- "https://my-workspace--tidylearn-fit.modal.run/fit"
  expect_silent(tl_validate_modal_url(url))
  expect_equal(tl_validate_modal_url(url), url)
})

test_that("non-https endpoints are refused", {
  skip_if_not_installed("httr2")

  expect_error(
    tl_validate_modal_url("http://my-workspace.modal.run/fit"),
    "must use https"
  )
})

test_that("non-Modal hosts are refused", {
  skip_if_not_installed("httr2")

  expect_error(
    tl_validate_modal_url("https://example.com/fit"),
    "not a Modal host"
  )
  expect_error(
    tl_validate_modal_url("https://modal.run.evil.test/fit"),
    "not a Modal host"
  )
})

test_that("malformed endpoint values are refused", {
  expect_error(tl_validate_modal_url(""), "single non-empty URL")
  expect_error(tl_validate_modal_url(NA_character_), "single non-empty URL")
  expect_error(tl_validate_modal_url(character(0)), "single non-empty URL")
  expect_error(
    tl_validate_modal_url(c("https://a.modal.run", "https://b.modal.run")),
    "single non-empty URL"
  )
})

# ---- T9: endpoint resolution ----

test_that("a missing endpoint variable is a clear error", {
  withr_env <- Sys.getenv("TIDYLEARN_MODAL_ENDPOINT", unset = NA)
  on.exit({
    if (is.na(withr_env)) {
      Sys.unsetenv("TIDYLEARN_MODAL_ENDPOINT")
    } else {
      Sys.setenv(TIDYLEARN_MODAL_ENDPOINT = withr_env)
    }
  })

  Sys.unsetenv("TIDYLEARN_MODAL_ENDPOINT")
  expect_error(tl_cloud_endpoint(), "No cloud endpoint configured")
})

test_that("a configured endpoint is validated, not trusted", {
  skip_if_not_installed("httr2")

  original <- Sys.getenv("TIDYLEARN_MODAL_ENDPOINT", unset = NA)
  on.exit({
    if (is.na(original)) {
      Sys.unsetenv("TIDYLEARN_MODAL_ENDPOINT")
    } else {
      Sys.setenv(TIDYLEARN_MODAL_ENDPOINT = original)
    }
  })

  Sys.setenv(TIDYLEARN_MODAL_ENDPOINT = "https://ws--fit.modal.run/fit")
  expect_equal(tl_cloud_endpoint(), "https://ws--fit.modal.run/fit")

  # A hostile or mistyped value must not slip through just because it
  # was configured.
  Sys.setenv(TIDYLEARN_MODAL_ENDPOINT = "https://evil.test/fit")
  expect_error(tl_cloud_endpoint(), "not a Modal host")
})

# ---- T2: consent ----

test_that("cloud uploads are refused without consent", {
  suppressMessages(tl_cloud_consent(FALSE))
  on.exit(suppressMessages(tl_cloud_consent(FALSE)))

  expect_error(tl_cloud_require_consent(), "consent has not been given")
  expect_error(
    tl_cloud_require_consent(confirm_upload = FALSE),
    "consent has not been given"
  )
})

test_that("per-call confirm_upload permits a single upload", {
  suppressMessages(tl_cloud_consent(FALSE))
  on.exit(suppressMessages(tl_cloud_consent(FALSE)))

  expect_true(tl_cloud_require_consent(confirm_upload = TRUE))

  # Consenting to one call must not arm the session.
  expect_false(tl_cloud_consent_active())
  expect_error(tl_cloud_require_consent(), "consent has not been given")
})

test_that("the session lock permits uploads until revoked", {
  on.exit(suppressMessages(tl_cloud_consent(FALSE)))

  expect_message(tl_cloud_consent(TRUE), "enabled for this R session")
  expect_true(tl_cloud_consent_active())
  expect_true(tl_cloud_require_consent())

  expect_message(tl_cloud_consent(FALSE), "disabled")
  expect_false(tl_cloud_consent_active())
  expect_error(tl_cloud_require_consent(), "consent has not been given")
})

test_that("tl_cloud_consent returns the previous state invisibly", {
  on.exit(suppressMessages(tl_cloud_consent(FALSE)))

  suppressMessages(tl_cloud_consent(FALSE))
  expect_false(suppressMessages(tl_cloud_consent(TRUE)))
  expect_true(suppressMessages(tl_cloud_consent(FALSE)))
})

test_that("consent arguments are validated", {
  expect_error(tl_cloud_consent("yes"), "must be TRUE or FALSE")
  expect_error(tl_cloud_consent(NA), "must be TRUE or FALSE")
  expect_error(tl_cloud_consent(c(TRUE, TRUE)), "must be TRUE or FALSE")
  expect_error(
    tl_cloud_require_consent(confirm_upload = "yes"),
    "must be TRUE or FALSE"
  )
})
