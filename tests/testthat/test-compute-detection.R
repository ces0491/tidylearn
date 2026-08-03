# ---- Compute detection (tl_check_gpu and helpers) ----

test_that("tl_check_gpu returns expected structure", {
  result <- tl_check_gpu()
  expect_s3_class(result, "tidylearn_gpu_check")
  expect_named(result, c("any_gpu", "cuda", "backends", "messages"))
  expect_type(result$any_gpu, "logical")
  expect_length(result$any_gpu, 1L)
  expect_type(result$cuda, "list")
  expect_named(
    result$cuda,
    c("driver_present", "device_count", "device_names", "driver_version")
  )
  expect_type(result$backends, "list")
  expect_named(
    result$backends,
    c("xgboost", "tensorflow", "keras", "torch")
  )
})

test_that("tl_check_gpu reports CPU-only when CUDA absent (mocked)", {
  testthat::local_mocked_bindings(
    tl_detect_cuda_internal = function() {
      list(
        driver_present = FALSE,
        device_count   = 0L,
        device_names   = character(0),
        driver_version = NA_character_
      )
    }
  )
  result <- tl_check_gpu()
  expect_false(result$any_gpu)
  expect_false(result$cuda$driver_present)
  expect_match(result$messages[1], "No NVIDIA CUDA driver detected")
})

test_that("tl_check_gpu reports CUDA present when nvidia-smi succeeds", {
  testthat::local_mocked_bindings(
    tl_detect_cuda_internal = function() {
      list(
        driver_present = TRUE,
        device_count   = 1L,
        device_names   = "Tesla T4",
        driver_version = "525.85.12"
      )
    }
  )
  result <- tl_check_gpu()
  expect_true(result$cuda$driver_present)
  expect_equal(result$cuda$device_names, "Tesla T4")
})

test_that("tl_detect_cuda_internal handles missing nvidia-smi gracefully", {
  result <- tl_detect_cuda_internal()
  expect_type(result, "list")
  expect_named(
    result,
    c("driver_present", "device_count", "device_names", "driver_version")
  )
  expect_type(result$driver_present, "logical")
})

test_that("tl_check_backend_gpu reports not installed for missing package", {
  fake_cuda <- list(driver_present = TRUE)
  result <- tl_check_backend_gpu("nonexistent_pkg_xyz_123", fake_cuda)
  expect_false(result$installed)
  expect_false(result$gpu_likely_works)
  expect_match(result$notes, "is not installed")
})

test_that("tl_check_backend_gpu reports CPU only when driver absent", {
  fake_cuda <- list(driver_present = FALSE)
  result <- tl_check_backend_gpu("tidylearn", fake_cuda)
  expect_true(result$installed)
  expect_false(result$gpu_likely_works)
  expect_match(result$notes, "no CUDA driver detected")
})

test_that("tl_check_backend_gpu reports GPU likely when both present", {
  fake_cuda <- list(driver_present = TRUE)
  result <- tl_check_backend_gpu("tidylearn", fake_cuda)
  expect_true(result$installed)
  expect_true(result$gpu_likely_works)
  expect_match(result$notes, "Installed and CUDA driver present")
})

test_that("tl_gpu_capable_methods with NULL returns all GPU-eligible methods", {
  result <- tl_gpu_capable_methods(NULL)
  expect_type(result, "character")
  expect_true("xgboost" %in% result)
  expect_true("deep" %in% result)
})

test_that("tl_gpu_capable_methods filters by gpu_check", {
  fake_check <- structure(
    list(
      any_gpu = TRUE,
      cuda    = list(driver_present = TRUE),
      backends = list(
        xgboost    = list(installed = TRUE,  gpu_likely_works = TRUE),
        tensorflow = list(installed = FALSE, gpu_likely_works = FALSE),
        keras      = list(installed = FALSE, gpu_likely_works = FALSE),
        torch      = list(installed = FALSE, gpu_likely_works = FALSE)
      )
    ),
    class = "tidylearn_gpu_check"
  )
  result <- tl_gpu_capable_methods(fake_check)
  expect_equal(result, "xgboost")
})

test_that("tl_gpu_capable_methods returns empty when no GPU-capable backend", {
  fake_check <- structure(
    list(
      any_gpu = FALSE,
      cuda    = list(driver_present = FALSE),
      backends = list(
        xgboost    = list(installed = FALSE, gpu_likely_works = FALSE),
        tensorflow = list(installed = FALSE, gpu_likely_works = FALSE),
        keras      = list(installed = FALSE, gpu_likely_works = FALSE),
        torch      = list(installed = FALSE, gpu_likely_works = FALSE)
      )
    ),
    class = "tidylearn_gpu_check"
  )
  result <- tl_gpu_capable_methods(fake_check)
  expect_length(result, 0L)
})

test_that("tl_gpu_capable_methods rejects non-gpu-check input", {
  expect_error(
    tl_gpu_capable_methods(list()),
    "tidylearn_gpu_check object"
  )
})

test_that("print.tidylearn_gpu_check runs without error", {
  result <- tl_check_gpu()
  expect_output(print(result), "tidylearn GPU detection")
})

test_that("print.tidylearn_gpu_check renders driver details when present", {
  fake <- structure(
    list(
      any_gpu = TRUE,
      cuda    = list(
        driver_present = TRUE,
        device_count   = 2L,
        device_names   = c("Tesla T4", "Tesla T4"),
        driver_version = "525.85.12"
      ),
      backends = list(
        xgboost    = list(installed = TRUE,  gpu_likely_works = TRUE),
        tensorflow = list(installed = FALSE, gpu_likely_works = FALSE),
        keras      = list(installed = FALSE, gpu_likely_works = FALSE),
        torch      = list(installed = FALSE, gpu_likely_works = FALSE)
      ),
      messages = character(0)
    ),
    class = "tidylearn_gpu_check"
  )
  output <- capture.output(print(fake))
  expect_true(any(grepl("Tesla T4", output)))
  expect_true(any(grepl("525.85.12", output)))
})
