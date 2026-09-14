test_that("tl_prepare_data handles missing values", {
  # Create data with missing values
  data_missing <- iris
  data_missing[1:5, "Sepal.Length"] <- NA
  data_missing[10:15, "Petal.Width"] <- NA

  # Prepare data with imputation
  result <- tl_prepare_data(data_missing, Species ~ .,
                            impute_method = "mean",
                            scale_method = "none",
                            encode_categorical = FALSE)

  # Check that NAs are imputed
  expect_false(any(is.na(result$data)))
  expect_true("imputation" %in% names(result$preprocessing_steps))
})

test_that("tl_prepare_data scales features correctly", {
  # Standardization
  result_std <- tl_prepare_data(iris, Species ~ .,
                                impute_method = "mean",
                                scale_method = "standardize",
                                encode_categorical = FALSE)

  numeric_cols <- sapply(result_std$data, is.numeric)
  numeric_data <- result_std$data[, numeric_cols]

  # Check means are close to 0 and sds close to 1 (excluding response)
  means <- colMeans(numeric_data[, names(numeric_data) != "Species"])
  expect_true(all(abs(means) < 1e-10))

  # Normalization
  result_norm <- tl_prepare_data(iris, Species ~ .,
                                 impute_method = "mean",
                                 scale_method = "normalize",
                                 encode_categorical = FALSE)

  numeric_data_norm <- result_norm$data[, numeric_cols]
  # Check values are in [0, 1]
  expect_true(
    all(numeric_data_norm >= 0 & numeric_data_norm <= 1, na.rm = TRUE)
  )
})

test_that("tl_prepare_data encodes categorical variables", {
  # Create data with categorical variable
  test_data <- data.frame(
    x1 = rnorm(100),
    x2 = rnorm(100),
    cat_var = factor(rep(c("A", "B", "C"), length.out = 100)),
    y = rnorm(100)
  )

  result <- tl_prepare_data(test_data, y ~ .,
                            encode_categorical = TRUE,
                            scale_method = "none")

  # Original categorical variable should be replaced with dummies
  expect_false("cat_var" %in% names(result$data))
  expect_true(any(grepl("cat_var_", names(result$data))))
})

test_that("tl_prepare_data removes zero variance features", {
  # Create data with zero variance column
  test_data <- iris
  test_data$zero_var <- 1

  result <- tl_prepare_data(test_data, Species ~ .,
                            remove_zero_variance = TRUE,
                            scale_method = "none",
                            encode_categorical = FALSE)

  # Zero variance column should be removed
  expect_false("zero_var" %in% names(result$data))
  expect_true("zero_variance" %in% names(result$preprocessing_steps))
})

test_that("tl_prepare_data removes highly correlated features", {
  # Create data with highly correlated columns
  test_data <- iris
  test_data$Sepal.Length.Copy <-
    test_data$Sepal.Length + rnorm(nrow(iris), 0, 0.01)

  result <- tl_prepare_data(test_data, Species ~ .,
                            remove_correlated = TRUE,
                            correlation_cutoff = 0.95,
                            scale_method = "none",
                            encode_categorical = FALSE)

  # One of the correlated columns should be removed
  has_original <- "Sepal.Length" %in% names(result$data)
  has_copy <- "Sepal.Length.Copy" %in% names(result$data)

  expect_true(xor(has_original, has_copy))
})

test_that("tl_split creates train/test splits correctly", {
  # Simple split
  split <- tl_split(iris, prop = 0.7, seed = 123)

  expect_type(split, "list")
  expect_equal(names(split), c("train", "test"))
  expect_equal(nrow(split$train), 105)
  expect_equal(nrow(split$test), 45)
  expect_equal(nrow(split$train) + nrow(split$test), nrow(iris))

  # Check no overlap
  train_idx <- as.numeric(rownames(split$train))
  test_idx <- as.numeric(rownames(split$test))
  expect_equal(length(intersect(train_idx, test_idx)), 0)
})

test_that("tl_split supports stratified splitting", {
  # Stratified split
  split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)

  # Check proportions are maintained
  train_props <- prop.table(table(split$train$Species))
  test_props <- prop.table(table(split$test$Species))
  original_props <- prop.table(table(iris$Species))

  # Proportions should be similar (within 5%)
  expect_true(all(abs(train_props - original_props) < 0.05))
  expect_true(all(abs(test_props - original_props) < 0.05))
})

test_that("tl_split validates inputs", {
  expect_error(
    tl_split(iris, prop = 0.7, stratify = "NonexistentColumn"),
    "Stratify variable not found"
  )
})

test_that("tl_prepare_data preserves response variable", {
  result <- tl_prepare_data(iris, Species ~ .,
                            scale_method = "standardize",
                            encode_categorical = FALSE)

  # Response should be present and unchanged
  expect_true("Species" %in% names(result$data))
  expect_equal(result$data$Species, iris$Species)
})

test_that("a one-row stratum keeps the split a partition", {
  # sample() on a single number draws from 1:n, so a stratum holding row
  # 10 alone drew some other row -- possibly one already drawn -- and
  # left row 10 in test.
  d <- data.frame(x = 1:10, g = c(rep("a", 9), "b"))
  split <- tl_split(d, stratify = "g", seed = 1)

  expect_setequal(c(split$train$x, split$test$x), d$x)
  expect_equal(anyDuplicated(split$train$x), 0L)
  expect_true(10 %in% split$train$x)

  # Every distinct mpg is a stratum, and most hold a single car
  split <- tl_split(mtcars, stratify = "mpg", seed = 1)
  expect_equal(nrow(split$train) + nrow(split$test), nrow(mtcars))
  expect_length(intersect(rownames(split$train), rownames(split$test)), 0)
})

test_that("rows missing the stratify value are split, not all sent to test", {
  d <- iris
  d$Species[c(1, 2, 60, 61, 120, 121)] <- NA
  split <- tl_split(d, stratify = "Species", seed = 1)
  expect_equal(nrow(split$train) + nrow(split$test), nrow(d))
  expect_gt(sum(is.na(split$train$Species)), 0)
})

test_that("tl_split keeps a one-column data frame a data frame", {
  split <- tl_split(data.frame(x = 1:10), seed = 1)
  expect_s3_class(split$train, "data.frame")
  expect_s3_class(split$test, "data.frame")
  expect_named(split$train, "x")
})

test_that("imputation does what the method says, and refuses one it lacks", {
  d <- mtcars
  d$mpg[1:3] <- NA
  expect_error(
    tl_prepare_data(d, cyl ~ ., impute_method = "knn", scale_method = "none"),
    "impute_method"
  )
  mode_fit <- suppressMessages(
    tl_prepare_data(transform(d, gear = gear), cyl ~ .,
                    impute_method = "mode", scale_method = "none")
  )
  # mpg has several values tied for most frequent; any of them is a mode
  observed <- mtcars$mpg[-(1:3)]
  imputed <- mode_fit$data$mpg[1]
  expect_equal(sum(observed == imputed), max(table(observed)))
  expect_false(isTRUE(all.equal(imputed, mean(observed))))

  # categorical gaps are filled with the most frequent level
  d2 <- iris
  d2$Species[1:2] <- NA
  out <- suppressMessages(
    tl_prepare_data(d2, Sepal.Length ~ ., scale_method = "none",
                    encode_categorical = FALSE)
  )
  expect_false(anyNA(out$data$Species))
  expect_equal(nrow(out$data), nrow(iris))
})

test_that("correlated removal drops the feature that clears every pair", {
  set.seed(1)
  x1 <- rnorm(200)
  x2 <- x1 + rnorm(200, sd = .1)
  x3 <- x2 + rnorm(200, sd = .1)
  out <- suppressMessages(tl_prepare_data(
    data.frame(x1, x2, x3), remove_correlated = TRUE,
    scale_method = "none", correlation_cutoff = .99
  ))
  expect_named(out$data, c("x1", "x3"))
})

test_that("columns the formula excludes are passed through untouched", {
  d <- data.frame(id = as.character(1:20), y = rnorm(20), x = rnorm(20))
  out <- suppressMessages(tl_prepare_data(d, y ~ . - id))
  expect_setequal(names(out$data), c("id", "y", "x"))
  expect_identical(out$data$id, d$id)
  # x is still scaled
  expect_equal(mean(out$data$x), 0)
})

test_that("tl_split refuses a proportion outside (0, 1)", {
  for (prop in list(1.5, -1, 0, 1, NA_real_, "0.8", c(0.5, 0.6))) {
    expect_error(tl_split(mtcars, prop = prop, seed = 1),
                 "'prop' must be a single number strictly between 0 and 1")
  }
  expect_equal(nrow(tl_split(mtcars, prop = 0.5, seed = 1)$train), 16)
})

test_that("scaling leaves a column with no spread to measure alone", {
  d <- data.frame(y = rnorm(10), x = rnorm(10), empty = NA_real_)
  out <- suppressMessages(
    tl_prepare_data(d, y ~ ., remove_zero_variance = FALSE)
  )
  expect_true(all(is.na(out$data$empty)))
  expect_equal(mean(out$data$x), 0)
})
