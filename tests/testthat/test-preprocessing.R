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
  expect_true(all(numeric_data_norm >= 0 & numeric_data_norm <= 1, na.rm = TRUE))
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
  test_data$Sepal.Length.Copy <- test_data$Sepal.Length + rnorm(nrow(iris), 0, 0.01)

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
