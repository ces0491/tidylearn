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

test_that("tl_split refuses a stratify that is not one column name", {
  # A vector of two names failed the column lookup with "the condition has
  # length > 1"
  for (stratify in list(c("am", "vs"), 1, NA_character_, TRUE)) {
    expect_error(
      tl_split(mtcars, stratify = stratify, seed = 1),
      "'stratify' must be the name of one column of 'data'",
      fixed = TRUE
    )
  }
  expect_error(
    tl_split(mtcars, stratify = "amm", seed = 1),
    "Stratify variable not found in data: 'amm'",
    fixed = TRUE
  )
  expect_equal(nrow(tl_split(mtcars, stratify = "am", seed = 1)$train), 25)
})

test_that("stratifying on a continuous column splits by its quartiles", {
  # Every distinct value was a one-row stratum, and a one-row stratum goes
  # wholly to training, so 50 rows split 50/0 and the empty test set gave
  # tl_evaluate() an rmse of NaN
  set.seed(1)
  d <- data.frame(y = stats::rnorm(50), x = stats::rnorm(50))
  split <- tl_split(d, prop = 0.8, stratify = "y", seed = 1)

  breaks <- stats::quantile(d$y, probs = seq(0, 1, 0.25))
  bins <- cut(d$y, breaks, include.lowest = TRUE)
  train_bins <- cut(split$train$y, breaks, include.lowest = TRUE)
  expect_equal(as.vector(table(train_bins)),
               floor(as.vector(table(bins)) * 0.8))
  expect_equal(nrow(split$train) + nrow(split$test), 50)
  expect_length(intersect(split$train$x, split$test$x), 0)

  # Rows missing the value still form a stratum of their own
  d$y[1:6] <- NA
  split <- tl_split(d, prop = 0.8, stratify = "y", seed = 1)
  expect_equal(sum(is.na(split$train$y)), 4)
  expect_equal(sum(is.na(split$test$y)), 2)
})

test_that("one-row strata are pooled only when they are over a tenth", {
  # An ID column made every row a stratum of one, all sent to training. The
  # pooled stratum is every row, so the draw is the unstratified one.
  ids <- data.frame(id = sprintf("id%02d", 1:50), x = seq_len(50))
  split <- tl_split(ids, prop = 0.8, stratify = "id", seed = 1)
  expect_equal(nrow(split$train), 40)
  expect_equal(nrow(split$test), 10)
  expect_identical(split, tl_split(ids, prop = 0.8, seed = 1))

  # Two levels seen once in 22 rows stay in training, where a model can
  # learn them. Pooled, d's row went to test, a level predict() refuses.
  few <- data.frame(g = factor(c("a", "d", rep("b", 10), rep("c", 10))),
                    x = 1:22)
  split <- tl_split(few, stratify = "g", seed = 1)
  expect_equal(as.vector(table(split$train$g)), c(1, 8, 8, 1))
  expect_equal(as.vector(table(split$test$g)), c(0, 2, 2, 0))

  # Three one-row strata in 30 rows are exactly a tenth and stay in
  # training; in 29 rows they are more than a tenth and are pooled
  singles <- c("a", "b", "c")
  at_tenth <- data.frame(g = c(singles, rep("x", 27)), x = 1:30)
  split <- tl_split(at_tenth, stratify = "g", seed = 1)
  expect_true(all(singles %in% split$train$g))
  over_tenth <- data.frame(g = c(singles, rep("x", 26)), x = 1:29)
  split <- tl_split(over_tenth, stratify = "g", seed = 1)
  expect_equal(sum(singles %in% split$train$g), 2)
  expect_equal(sum(singles %in% split$test$g), 1)

  # A numeric column with few distinct values is still split by value.
  # Binning cyl by its quartiles would put 4 and 6 in one bin and draw 25.
  split <- tl_split(mtcars, prop = 0.8, stratify = "cyl", seed = 1)
  expect_equal(as.vector(table(split$train$cyl)), c(8, 5, 11))
})

test_that("tl_split warns when the test set comes back empty", {
  expect_warning(
    split <- tl_split(data.frame(x = 1), seed = 1),
    "Splitting 1 row(s) at prop = 0.8 leaves no test data",
    fixed = TRUE
  )
  expect_equal(nrow(split$train), 1)
  expect_equal(nrow(split$test), 0)

  expect_no_warning(tl_split(data.frame(x = 1:2), seed = 1))
  expect_no_warning(tl_split(data.frame(x = 1:2, g = 1:2), stratify = "g",
                             seed = 1))
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

test_that("tl_prepare_data refuses a correlation cutoff outside (0, 1]", {
  # 95, meant as a percentage, is never exceeded by a correlation, so
  # remove_correlated = TRUE removed nothing without a word
  set.seed(1)
  x1 <- rnorm(200)
  chain <- data.frame(x1, x2 = x1 + rnorm(200, sd = .1))
  for (cutoff in list(95, 0, -0.5, NA_real_, "0.9", c(0.8, 0.9))) {
    expect_error(
      tl_prepare_data(chain, remove_correlated = TRUE,
                      correlation_cutoff = cutoff, scale_method = "none"),
      paste0("'correlation_cutoff' must be a single number greater than 0 ",
             "and at most 1"),
      fixed = TRUE
    )
  }

  removed <- suppressMessages(tl_prepare_data(
    chain, remove_correlated = TRUE, correlation_cutoff = 0.9,
    scale_method = "none"
  ))
  expect_length(names(removed$data), 1)
  # No correlation exceeds 1, so the boundary is accepted and removes nothing
  kept <- tl_prepare_data(chain, remove_correlated = TRUE,
                          correlation_cutoff = 1, scale_method = "none")
  expect_named(kept$data, c("x1", "x2"))
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

  # An Inf has an undefined variance, and the NA it gave the zero-variance
  # step stopped the call with "Selections can't have missing values"
  d <- data.frame(y = 1:5, x = c(1, 2, Inf, 4, 5), z = c(2, 7, 1, 8, 2))
  out <- suppressMessages(tl_prepare_data(d, y ~ .))
  expect_identical(out$data$x, d$x)
  expect_equal(out$data$z, (d$z - mean(d$z)) / stats::sd(d$z))
})

test_that("tl_prepare_data runs when no formula predictor is numeric", {
  # sapply() over a frame with no numeric column returns list(), and
  # indexing names() with that failed with "invalid subscript type 'list'"
  d <- transform(mtcars, am = factor(am, labels = c("auto", "manual")))
  out <- suppressMessages(tl_prepare_data(d, mpg ~ am))
  expect_equal(dim(out$data), c(32L, 11L))
  expect_identical(out$data$am, d$am)
  expect_identical(out$data$mpg, d$mpg)
  expect_identical(out$data$disp, d$disp)

  logical_pred <- transform(mtcars, manual = am == 1)
  out <- suppressMessages(tl_prepare_data(logical_pred, mpg ~ manual))
  expect_identical(out$data$manual, logical_pred$manual)

  # A three-level factor is still one-hot encoded
  d$gear <- factor(d$gear)
  out <- suppressMessages(
    tl_prepare_data(d, mpg ~ gear, scale_method = "none")
  )
  expect_equal(out$data$gear_4, as.numeric(d$gear == "4"))

  intercept_only <- suppressMessages(tl_prepare_data(mtcars, mpg ~ 1))
  expect_equal(as.list(intercept_only$data[names(mtcars)]), as.list(mtcars))
})

test_that("tl_prepare_data refuses a response that is not a column", {
  # mgp ~ . standardised mpg as one of the predictors, and the misspelt
  # response was dropped without a word
  expect_error(
    tl_prepare_data(mtcars, mgp ~ .),
    "The formula's response, 'mgp', is not a column of `data`",
    fixed = TRUE
  )
  out <- suppressMessages(tl_prepare_data(mtcars[, 1:4], log(mpg) ~ .))
  expect_identical(out$data$mpg, mtcars$mpg)
})

test_that("tl_prepare_data refuses a scaling method it lacks", {
  # "zscore" reported scaling and returned the data unchanged
  expect_error(
    tl_prepare_data(mtcars[, 1:3], mpg ~ ., scale_method = "zscore"),
    paste0("'scale_method' must be one of \"standardize\", \"normalize\", ",
           "\"robust\", \"none\"."),
    fixed = TRUE
  )
  expect_error(
    tl_prepare_data(mtcars[, 1:3], mpg ~ .,
                    scale_method = c("standardize", "none")),
    "'scale_method' must be one of",
    fixed = TRUE
  )

  robust <- suppressMessages(
    tl_prepare_data(mtcars[, 1:3], mpg ~ ., scale_method = "robust")
  )
  expect_equal(robust$data$disp,
               (mtcars$disp - stats::median(mtcars$disp)) /
                 stats::IQR(mtcars$disp),
               ignore_attr = TRUE)
  unscaled <- tl_prepare_data(mtcars[, 1:3], mpg ~ ., scale_method = "none")
  expect_identical(unscaled$data$disp, mtcars$disp)
})

test_that("an entirely missing factor is left as it is", {
  # model.matrix() drops NA rows, so the factor's dummies had no rows and
  # binding them back failed with "Can't recycle `..1` (size 6) to match
  # `..2` (size 0)"
  d <- data.frame(
    y = c(1, 4, 2, 8, 5, 7), x = c(3, 1, 4, 1, 5, 9),
    g = factor(rep(NA, 6), levels = c("a", "b", "c"))
  )
  out <- suppressMessages(tl_prepare_data(d, y ~ .))
  expect_identical(out$data$g, d$g)
  expect_equal(out$data$x, (d$x - mean(d$x)) / stats::sd(d$x))
})

test_that("tl_prepare_data reports imputing only when it imputes", {
  # The only missing column was entirely missing, so nothing was filled,
  # yet the call said "Imputing missing values using method: mean" and
  # recorded an imputation step with no values
  d <- data.frame(
    y = c(1, 4, 2, 8, 5, 7), x = c(3, 1, 4, 1, 5, 9),
    f = factor(rep(NA, 6), levels = c("a", "b", "c"))
  )
  expect_no_message(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none"),
    message = "Imputing missing values"
  )
  expect_null(out$preprocessing_steps$imputation)
  expect_identical(out$data$f, d$f)

  # A column with a value to impute from is still filled, and said so
  d$x[2] <- NA
  expect_message(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none"),
    "Imputing missing values using method: mean",
    fixed = TRUE
  )
  expect_identical(out$preprocessing_steps$imputation$imputation_values,
                   list(x = mean(d$x, na.rm = TRUE)))
  expect_equal(out$data$x[2], mean(d$x, na.rm = TRUE))
})

test_that("tl_prepare_data reports scaling only when it scales", {
  # A numeric column whose spread is not finite is left unscaled, yet with
  # no other numeric predictor the call still said "Scaling numeric
  # features" and recorded a scaling step with no parameters
  d <- data.frame(y = 1:5, x = c(1, 2, Inf, 4, 5))
  expect_no_message(
    out <- tl_prepare_data(d, y ~ .),
    message = "Scaling numeric features"
  )
  expect_null(out$preprocessing_steps$scaling)
  expect_identical(out$data$x, d$x)

  d$z <- c(2, 7, 1, 8, 2)
  expect_message(
    out <- tl_prepare_data(d, y ~ .),
    "Scaling numeric features using method: standardize",
    fixed = TRUE
  )
  expect_named(out$preprocessing_steps$scaling$scaling_params, "z")
  expect_equal(out$data$z, (d$z - mean(d$z)) / stats::sd(d$z))
})

test_that("tl_prepare_data reports encoding only for what it one-hot encodes", {
  # A two-level factor is left as it is, yet the call said "Encoding 1
  # categorical variables" and recorded an encoding step with an empty map
  d <- data.frame(
    y = c(1, 4, 2, 8, 5, 7), x = c(3, 1, 4, 1, 5, 9),
    am = factor(c("a", "m", "a", "m", "m", "a"))
  )
  expect_no_message(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none"),
    message = "Encoding"
  )
  expect_null(out$preprocessing_steps$encoding)
  expect_identical(out$data$am, d$am)

  # A two-level text column still becomes a factor
  d$side <- c("l", "r", "r", "l", "r", "l")
  out <- tl_prepare_data(d, y ~ ., scale_method = "none")
  expect_identical(out$data$side, factor(d$side))

  # Only a factor with more levels is counted, in the singular for one
  d$g <- factor(c("p", "q", "r", "p", "q", "r"))
  messages <- testthat::capture_messages(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none")
  )
  expect_identical(messages, "Encoding 1 categorical variable\n")
  expect_identical(out$preprocessing_steps$encoding$encoding_map,
                   list(g = c("g_p", "g_q", "g_r")))
  expect_equal(out$data$g_q, as.numeric(d$g == "q"))

  d$h <- as.character(rev(d$g))
  messages <- testthat::capture_messages(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none")
  )
  expect_identical(messages, "Encoding 2 categorical variables\n")
  expect_named(out$preprocessing_steps$encoding$encoding_map, c("g", "h"))
})

test_that("tl_prepare_data counts one removed feature in the singular", {
  # "Removing 1 zero-variance features"
  d <- data.frame(y = c(1, 4, 2, 8, 5, 7), x = c(3, 1, 4, 1, 5, 9),
                  flat = 2)
  messages <- testthat::capture_messages(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none")
  )
  expect_identical(messages, "Removing 1 zero-variance feature\n")
  expect_identical(out$preprocessing_steps$zero_variance, "flat")

  d <- data.frame(y = c(1, 4, 2, 8, 5, 7), x = c(3, 1, 4, 1, 5, 9),
                  twice = c(6, 2, 8, 2, 10, 18), z = c(2, 7, 1, 8, 2, 8))
  messages <- testthat::capture_messages(
    out <- tl_prepare_data(d, y ~ ., scale_method = "none",
                           remove_correlated = TRUE)
  )
  expect_identical(messages, "Removing 1 highly correlated feature\n")
  expect_length(out$preprocessing_steps$high_correlation, 1L)
})

test_that("a single row is returned without scaling", {
  # One value has an NA variance, and NA in the zero-variance selection
  # failed with "Selections can't have missing values"
  out <- suppressMessages(
    tl_prepare_data(data.frame(y = 1, x = 2, z = 3), y ~ .)
  )
  expect_equal(as.list(out$data[c("y", "x", "z")]),
               list(y = 1, x = 2, z = 3))
})

test_that("a grouped tibble is prepared as a whole", {
  # The response was added back inside each group: "`mpg` must be size 11
  # or 1, not 32"
  grouped <- dplyr::group_by(tibble::as_tibble(mtcars[, 1:4]), cyl)
  out <- suppressMessages(tl_prepare_data(grouped, mpg ~ .))
  expect_false(dplyr::is_grouped_df(out$data))
  expect_equal(out$data$mpg, mtcars$mpg)
  expect_equal(out$data$disp,
               (mtcars$disp - mean(mtcars$disp)) / stats::sd(mtcars$disp))
})

test_that("tl_prepare_data refuses a formula with no response", {
  # all.vars(~ x1 + x2)[1] made x1 the response, so it was left unscaled
  # and only x2 was processed
  d <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(10, 20, 30, 45))
  expect_error(
    tl_prepare_data(d, ~ x1 + x2),
    "'formula' has no response",
    fixed = TRUE
  )
  expect_equal(suppressMessages(tl_prepare_data(d))$data$x1,
               (d$x1 - mean(d$x1)) / stats::sd(d$x1))
})
