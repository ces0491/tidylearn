# inst/examples/unified_workflow.R ships in the tarball and is the first
# runnable thing a new user is pointed at. Nothing else executes it, so a
# rename or a signature change breaks it silently.

# One test, because sourcing the script fits every model in it, so each
# summary line is checked against that single run
test_that("the shipped example runs end to end and reports what it built", {
  skip_on_cran()

  path <- system.file("examples", "unified_workflow.R", package = "tidylearn")
  skip_if(path == "", "example script not installed")

  env <- new.env(parent = globalenv())
  output <- utils::capture.output(
    expect_no_warning(
      expect_no_error(
        suppressMessages(source(path, local = env, echo = FALSE))
      )
    )
  )

  expect_match(output, "All examples completed", fixed = TRUE, all = FALSE)

  # iris has four predictors; asking for three components must report three
  expect_match(output, "Reduced from 4 to 3 features", fixed = TRUE,
               all = FALSE)

  # A transfer-learning result keeps its method at $method. The script read
  # $spec$method, which that result does not have, and printed
  # "built on  over 3 principal components"
  expect_match(
    output, "Transfer learning model: forest fitted on 3 principal components",
    fixed = TRUE, all = FALSE
  )

  # The script blanks ten values in one column of five, and said all five
  # features had missing values
  expect_match(output, "Original data: 5 features, 1 with missing values",
               fixed = TRUE, all = FALSE)

  # Sepal.Length and its noisy copy, and Petal.Length and Petal.Width, are
  # correlated above the 0.95 cutoff, so one of each pair goes; mean
  # imputation leaves no gaps
  expect_match(output, "Processed data: 3 features, 0 with missing values",
               fixed = TRUE, all = FALSE)
})
