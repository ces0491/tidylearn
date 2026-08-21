# inst/examples/unified_workflow.R ships in the tarball and is the first
# runnable thing a new user is pointed at. Nothing else executes it, so a
# rename or a signature change breaks it silently.

test_that("the shipped unified workflow example runs end to end", {
  skip_on_cran()

  path <- system.file("examples", "unified_workflow.R", package = "tidylearn")
  skip_if(path == "", "example script not installed")

  env <- new.env(parent = globalenv())
  output <- utils::capture.output(
    expect_no_warning(
      expect_no_error(source(path, local = env, echo = FALSE))
    )
  )

  expect_true(any(grepl("All examples completed", output)))

  # iris has four predictors; asking for three components must report three
  expect_true(any(grepl("Reduced from 4 to 3 features", output, fixed = TRUE)))
})
