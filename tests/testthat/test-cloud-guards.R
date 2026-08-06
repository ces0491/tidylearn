# ---- Cloud security guards: endpoint validation and consent ----
#
# These implement two commitments in inst/security/threat-model.md and
# are the checks that stand between a cloud fit and user data leaving the
# machine. T9 (egress to a non-Modal host) and T2 (silent upload).

# ---- T9: host allowlist ----

test_that("Modal hosts and their subdomains are accepted", {
  expect_true(tl_is_allowed_host("modal.run"))
  expect_true(tl_is_allowed_host("modal.com"))
  expect_true(tl_is_allowed_host("my-workspace--tidylearn-fit.modal.run"))
  expect_true(tl_is_allowed_host("api.modal.com"))
})

test_that("lookalike hosts are rejected", {
  # Suffix-shaped attacks: the allowlisted domain appears in the name but
  # the request would not reach Modal.
  expect_false(tl_is_allowed_host("modal.run.example.com"))
  expect_false(tl_is_allowed_host("evil-modal.run"))
  expect_false(tl_is_allowed_host("notmodal.com"))
  expect_false(tl_is_allowed_host("modal.run.evil.test"))
  expect_false(tl_is_allowed_host("example.com"))
})

test_that("degenerate hosts are rejected", {
  expect_false(tl_is_allowed_host(""))
  expect_false(tl_is_allowed_host(NA_character_))
  expect_false(tl_is_allowed_host(character(0)))
  expect_false(tl_is_allowed_host(c("modal.run", "modal.run")))
})

# ---- T9: extending the allowlist ----

test_that("the default allowlist is Modal's hosts only", {
  suppressMessages(tl_cloud_allow_host(NULL))
  on.exit(suppressMessages(tl_cloud_allow_host(NULL)))

  expect_setequal(tl_cloud_allowed_hosts(), c("modal.run", "modal.com"))
})

test_that("an added host and its subdomains become allowed", {
  suppressMessages(tl_cloud_allow_host(NULL))
  on.exit(suppressMessages(tl_cloud_allow_host(NULL)))

  expect_false(tl_is_allowed_host("fits.example.com"))

  expect_message(
    tl_cloud_allow_host("fits.example.com"),
    "may now also go to"
  )

  expect_true(tl_is_allowed_host("fits.example.com"))
  expect_true(tl_is_allowed_host("a.fits.example.com"))

  # Adding a host must not widen anything above it.
  expect_false(tl_is_allowed_host("example.com"))
  expect_false(tl_is_allowed_host("evil-fits.example.com"))
  expect_false(tl_is_allowed_host("fits.example.com.evil.test"))
})

test_that("added hosts are still distinguishable from Modal's own", {
  suppressMessages(tl_cloud_allow_host(NULL))
  on.exit(suppressMessages(tl_cloud_allow_host(NULL)))

  suppressMessages(tl_cloud_allow_host("fits.example.com"))

  # Allowed, but the pre-upload summary must be able to say it is not
  # a Modal host.
  expect_true(tl_is_allowed_host("fits.example.com"))
  expect_false(tl_is_modal_host("fits.example.com"))
  expect_true(tl_is_modal_host("ws.modal.run"))
})

test_that("clearing removes session additions but keeps Modal's hosts", {
  suppressMessages(tl_cloud_allow_host("fits.example.com"))
  expect_true(tl_is_allowed_host("fits.example.com"))

  expect_message(tl_cloud_allow_host(NULL), "Cleared all additional")

  expect_false(tl_is_allowed_host("fits.example.com"))
  expect_setequal(tl_cloud_allowed_hosts(), c("modal.run", "modal.com"))
})

test_that("only bare host names may be added", {
  on.exit(suppressMessages(tl_cloud_allow_host(NULL)))

  # A single label would open an entire TLD.
  expect_error(tl_cloud_allow_host("com"), "single label")
  expect_error(tl_cloud_allow_host("localhost"), "single label")

  # URLs, ports, paths and wildcards are not host names.
  expect_error(tl_cloud_allow_host("https://x.example.com"), "bare host")
  expect_error(tl_cloud_allow_host("x.example.com/fit"), "bare host")
  expect_error(tl_cloud_allow_host("x.example.com:8443"), "bare host")
  expect_error(tl_cloud_allow_host("*.example.com"), "bare host")
  expect_error(tl_cloud_allow_host("a b.example.com"), "bare host")

  expect_error(tl_cloud_allow_host(""), "non-empty host names")
  expect_error(tl_cloud_allow_host(NA_character_), "non-empty host names")
  expect_error(tl_cloud_allow_host(character(0)), "non-empty host names")
  expect_error(tl_cloud_allow_host(42), "non-empty host names")
})

test_that("a configured endpoint on an added host validates", {
  skip_if_not_installed("httr2")
  on.exit(suppressMessages(tl_cloud_allow_host(NULL)))

  url <- "https://fits.example.com/fit"
  expect_error(tl_validate_modal_url(url), "not an allowed upload")

  suppressMessages(tl_cloud_allow_host("fits.example.com"))
  expect_equal(tl_validate_modal_url(url), url)

  # http is still refused on an added host.
  expect_error(
    tl_validate_modal_url("http://fits.example.com/fit"),
    "must use https"
  )
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
    "not an allowed upload"
  )
  expect_error(
    tl_validate_modal_url("https://modal.run.evil.test/fit"),
    "not an allowed upload"
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
  expect_error(tl_cloud_endpoint(), "not an allowed upload")
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
