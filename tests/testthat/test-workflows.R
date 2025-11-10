test_that("tl_auto_ml runs with basic settings", {
  skip_on_cran()
  skip_if_not_installed("randomForest")

  # Run with short time budget for testing
  result <- tl_auto_ml(iris, Species ~ .,
                      use_reduction = TRUE,
                      use_clustering = TRUE,
                      time_budget = 10)

  expect_type(result, "list")
  expect_true("models" %in% names(result))
  expect_true("results" %in% names(result))
  expect_true("best_model" %in% names(result))

  # Should have trained at least one model
  expect_gte(length(result$models), 1)
})

test_that("tl_auto_ml detects task type automatically", {
  skip_on_cran()

  # Classification task
  result_class <- tl_auto_ml(iris, Species ~ .,
                            task = "auto",
                            time_budget = 5)

  expect_type(result_class, "list")

  # Regression task
  result_reg <- tl_auto_ml(mtcars, mpg ~ wt + hp,
                          task = "auto",
                          time_budget = 5)

  expect_type(result_reg, "list")
})

test_that("tl_auto_ml respects time budget", {
  skip_on_cran()

  # Very short time budget
  start_time <- Sys.time()
  result <- tl_auto_ml(iris, Species ~ .,
                      time_budget = 5,
                      use_reduction = FALSE,
                      use_clustering = FALSE)
  end_time <- Sys.time()

  # Should complete within reasonable time (with some buffer)
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
  expect_lt(elapsed, 20)  # Should finish well before 20 seconds
})

test_that("tl_auto_ml can disable reduction and clustering", {
  skip_on_cran()

  result <- tl_auto_ml(iris, Species ~ .,
                      use_reduction = FALSE,
                      use_clustering = FALSE,
                      time_budget = 10)

  expect_type(result, "list")

  # Should still have baseline models
  expect_gte(length(result$models), 1)

  # Model names should not include "pca_" or "clustered_"
  model_names <- names(result$models)
  expect_false(any(grepl("^pca_", model_names)))
  expect_false(any(grepl("^clustered_", model_names)))
})

test_that("tl_auto_ml handles small datasets", {
  skip_on_cran()

  # Small dataset
  small_data <- iris[1:30, ]

  result <- tl_auto_ml(small_data, Species ~ .,
                      time_budget = 5,
                      cv_folds = 3)

  expect_type(result, "list")
  expect_gte(length(result$models), 1)
})

test_that("tl_auto_ml returns best model", {
  skip_on_cran()

  result <- tl_auto_ml(iris, Species ~ .,
                      time_budget = 10)

  expect_true("best_model" %in% names(result))
  expect_s3_class(result$best_model, "tidylearn_model")

  # Best model should be one of the trained models
  expect_true(!is.null(result$best_model))
})

test_that("tl_auto_ml works with regression tasks", {
  skip_on_cran()

  result <- tl_auto_ml(mtcars, mpg ~ .,
                      task = "regression",
                      time_budget = 10)

  expect_type(result, "list")
  expect_gte(length(result$models), 1)
})

test_that("tl_auto_ml handles errors gracefully", {
  skip_on_cran()

  # Should not crash even if some models fail
  result <- tl_auto_ml(iris, Species ~ .,
                      time_budget = 5)

  expect_type(result, "list")
  # Should have trained at least one successful model
  expect_gte(length(result$models), 1)
})
