test_that("tl_model creates supervised models correctly", {
  # Test with classification. Logistic is binary only, so this needs a
  # two-level response -- three levels is an error at fit time.
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model <- tl_model(binary_iris, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_model")
  expect_s3_class(model, "tidylearn_supervised")
  expect_s3_class(model, "tidylearn_logistic")
  expect_equal(model$spec$paradigm, "supervised")
  expect_equal(model$spec$method, "logistic")
  expect_true(model$spec$is_classification)
  expect_equal(model$spec$response_var, "Species")
})

test_that("tl_model creates unsupervised models correctly", {
  # Test PCA
  model <- tl_model(iris[, 1:4], method = "pca")

  expect_s3_class(model, "tidylearn_model")
  expect_s3_class(model, "tidylearn_unsupervised")
  expect_s3_class(model, "tidylearn_pca")
  expect_equal(model$spec$paradigm, "unsupervised")
  expect_equal(model$spec$method, "pca")
})

test_that("tl_model validates inputs", {
  # Invalid data type
  expect_error(
    tl_model("not_a_dataframe", method = "linear"),
    "data.*must be a data frame"
  )

  # Unknown method
  expect_error(
    tl_model(iris, Species ~ ., method = "unknown_method"),
    "Unknown method"
  )
})

test_that("tl_model determines task type correctly", {
  # Classification with factor
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model_factor <- tl_model(binary_iris, Species ~ ., method = "logistic")
  expect_true(model_factor$spec$is_classification)

  # Regression with numeric
  model_numeric <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_false(model_numeric$spec$is_classification)
})

test_that("predict.tidylearn_model works for supervised models", {
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model <- tl_model(binary_iris, Species ~ ., method = "logistic")

  # Predict on training data
  pred_train <- predict(model)
  expect_s3_class(pred_train, "tbl_df")
  expect_equal(nrow(pred_train), nrow(binary_iris))

  # Predict on new data
  pred_new <- predict(model, new_data = binary_iris[1:10, ])
  expect_equal(nrow(pred_new), 10)
})

test_that("predict.tidylearn_model works for unsupervised models", {
  model <- tl_model(iris[, 1:4], method = "pca")

  # Transform data
  transformed <- predict(model)
  expect_s3_class(transformed, "tbl_df")
})

test_that("print.tidylearn_model displays correctly", {
  model <- tl_model(iris, Species ~ ., method = "forest")

  # Should print without error
  expect_output(print(model), "tidylearn Model")
  expect_output(print(model), "Paradigm: supervised")
  expect_output(print(model), "Method: forest")
  expect_output(print(model), "Task: Classification")
})

test_that("tl_version returns package version", {
  version <- tl_version()
  expect_s3_class(version, "package_version")
})

test_that("tl_align_classes() reads observed classes against the model's", {
  # Declared but unused levels drop out, and the model's order wins
  observed <- factor(c("virginica", "versicolor"),
                     levels = c("virginica", "setosa", "versicolor"))
  aligned <- tidylearn:::tl_align_classes(
    observed, c("versicolor", "virginica")
  )
  expect_identical(levels(aligned$actuals), c("versicolor", "virginica"))
  expect_identical(as.character(aligned$actuals), c("virginica", "versicolor"))
  expect_identical(aligned$keep, c(TRUE, TRUE))

  # A 0/1 numeric response reads against the character levels the spec holds
  aligned <- tidylearn:::tl_align_classes(c(0, 1, 1), c("0", "1"))
  expect_identical(as.character(aligned$actuals), c("0", "1", "1"))

  # Missing values are not scored, without a warning of their own
  expect_no_warning(
    aligned <- tidylearn:::tl_align_classes(c("a", NA), c("a", "b"))
  )
  expect_identical(aligned$keep, c(TRUE, FALSE))

  # A class the model never saw is left out, and the warning names it
  expect_warning(
    aligned <- tidylearn:::tl_align_classes(c("a", "c", "c"), c("a", "b")),
    "2 row\\(s\\) belong to a class the model was not trained on \\(c\\)"
  )
  expect_identical(aligned$keep, c(TRUE, FALSE, FALSE))
})

test_that("a missing file's message keeps a password out", {
  # A libpq connection string read with a file format reached the file
  # check, and "File not found" printed it, password and all
  err <- expect_error(
    tl_read("host=db user=u password=secret dbname=x", format = "csv",
            .quiet = TRUE),
    "File not found: 'host=db user=u password=*** dbname=x'", fixed = TRUE
  )
  expect_false(grepl("secret", conditionMessage(err), fixed = TRUE))

  # An ordinary path prints as given
  paths <- c(
    "nonexistent.csv", "data/train set.csv", "C:/Users/me/data.csv",
    "C:\\Users\\me\\data.csv", "~/data/x.parquet",
    file.path(tempdir(), "missing.csv")
  )
  for (path in paths) {
    expect_error(
      tidylearn:::tl_validate_file_path(path),
      paste0("File not found: '", path, "'"), fixed = TRUE
    )
  }
})

# ---- Categorical predictors ----

# What tl_read() hands back for a CSV: the category is a character column.
character_frame <- function() {
  set.seed(7)
  d <- data.frame(
    grp = rep(c("a", "b", "c"), 40), x = stats::rnorm(120),
    stringsAsFactors = FALSE
  )
  d$y <- ifelse(d$grp == "c", 10, 0) + d$x + stats::rnorm(120, sd = 0.1)
  d
}

test_that("a character predictor scores a row the same in any company", {
  # randomForest's data.matrix() coded a character column by the values
  # present in new_data, so three "c" rows scored alone predicted -0.11,
  # -0.15 and 0.45, and 9.47, 9.26 and 9.78 inside the full frame
  d <- character_frame()
  set.seed(11)
  model <- tl_model(d, y ~ grp + x, method = "forest", ntree = 50)
  rows <- which(d$grp == "c")[1:3]

  alone <- predict(model, new_data = d[rows, ])$.pred
  expect_equal(alone, predict(model, new_data = d)$.pred[rows])
  expect_true(all(abs(alone - d$y[rows]) < 2))

  # The forest splits on grp as a category, exactly as for a factor
  expect_equal(model$fit$forest$ncat[["grp"]], 3)
  expect_identical(model$spec$xlev$grp, c("a", "b", "c"))
  expect_true(is.factor(model$data$grp))
  as_factor <- d
  as_factor$grp <- factor(as_factor$grp)
  set.seed(11)
  reference <- tl_model(as_factor, y ~ grp + x, method = "forest", ntree = 50)
  expect_equal(
    predict(model, new_data = d)$.pred,
    predict(reference, new_data = as_factor)$.pred
  )

  # gbm refused the character column outright
  boost <- tl_model(d, y ~ grp + x, method = "boost", n.trees = 20)
  expect_equal(
    predict(boost, new_data = d[rows, ])$.pred,
    predict(boost, new_data = d)$.pred[rows]
  )
})

test_that("a character column used only inside a term keeps its type", {
  # Only bare categorical terms become factors: as.numeric() of a factor
  # reads its level codes, not its values
  d <- character_frame()
  d$code <- sprintf("%02d", rep(c(1, 5, 12), 40))
  model <- tl_model(d, y ~ as.numeric(code) + x, method = "linear")

  expect_type(model$data$code, "character")
  expect_equal(
    coef(model$fit),
    coef(lm(y ~ as.numeric(code) + x, data = d))
  )
})

test_that("new data may declare fewer levels than the training factor", {
  # randomForest compares level counts, so a row whose factor declared only
  # its own level failed with "Type of predictors in new data do not match"
  d <- mtcars
  d$cyl <- factor(d$cyl)
  set.seed(3)
  model <- tl_model(d, mpg ~ cyl + wt, method = "forest", ntree = 50)
  full_levels <- data.frame(
    cyl = factor("6", levels = c("4", "6", "8")), wt = 3
  )
  expected <- unname(predict(model$fit, newdata = full_levels))
  scored <- function(new) unname(predict(model, new_data = new)$.pred)

  expect_equal(scored(data.frame(cyl = factor("6"), wt = 3)), expected)
  # A value given as text or as the number it was coded from reads the same
  expect_equal(scored(data.frame(cyl = "6", wt = 3)), expected)
  expect_equal(scored(data.frame(cyl = 6, wt = 3)), expected)
})

test_that("a level the model was not trained on is refused by name", {
  # gbm scored level "z" without complaint, where tree, forest, svm and
  # xgboost refused it
  set.seed(1)
  d <- data.frame(
    g = factor(sample(c("a", "b", "c"), 200, TRUE)), x = stats::rnorm(200)
  )
  d$y <- as.numeric(d$g) + d$x
  model <- tl_model(d, y ~ g + x, method = "boost", n.trees = 20)

  expect_error(
    predict(model, new_data = data.frame(g = factor("z"), x = 0)),
    "levels the model was not trained on: 'g' has \"z\""
  )
  expect_error(
    predict(model, new_data = data.frame(g = c("b", "q"), x = 0)),
    "'g' has \"q\" \\(trained on \"a\", \"b\", \"c\"\\)"
  )
  # A trained level, and a missing value, still predict
  ok <- predict(model, new_data = data.frame(g = c("b", NA), x = 0))
  expect_equal(nrow(ok), 2)
})

test_that("an ordered factor stays ordered when new data is re-levelled", {
  # randomForest treats an ordered factor as numeric, and refused an
  # unordered one in its place
  d <- mtcars
  d$gear <- factor(d$gear, ordered = TRUE)
  set.seed(5)
  model <- tl_model(d, mpg ~ gear + wt, method = "forest", ntree = 50)

  new <- data.frame(gear = factor("4"), wt = 3)
  reference <- data.frame(
    gear = factor("4", levels = c("3", "4", "5"), ordered = TRUE), wt = 3
  )
  expect_equal(
    unname(predict(model, new_data = new)$.pred),
    unname(predict(model$fit, newdata = reference))
  )
})

# ---- Columns the model does not use ----

test_that("a column the model does not use cannot turn predictions NA", {
  # terms() expanded `.` over new_data, so a mostly-NA notes column counted
  # as a predictor and 27 of 30 svm predictions came back NA
  set.seed(3)
  train <- iris[sample(150, 120), ]
  test <- iris[-as.integer(rownames(train)), ]
  model <- tl_model(train, Species ~ ., method = "svm")
  noted <- test
  noted$notes <- NA_character_
  noted$notes[1:3] <- "ok"

  expect_identical(
    predict(model, new_data = noted)$.pred,
    predict(model, new_data = test)$.pred
  )
  expect_false(anyNA(predict(model, new_data = noted)$.pred))
})

test_that("a column the formula subtracts is not needed to predict", {
  # terms() keeps a subtracted column among its variables, so a model of
  # mpg ~ . - qsec stored terms naming qsec, and predict() on rows without
  # it failed with "object 'qsec' not found"
  model <- tl_model(mtcars, mpg ~ . - qsec, method = "linear")
  scored <- function(new) unname(predict(model, new_data = new)$.pred)

  without <- mtcars[1:5, setdiff(names(mtcars), "qsec")]
  expect_equal(scored(without), scored(mtcars[1:5, ]))
  expect_false("qsec" %in% all.vars(stats::terms(model$fit)))
  expect_equal(coef(model$fit), coef(lm(mpg ~ . - qsec, data = mtcars)))

  # The spec, and so print() and every refit, keep the formula as written
  expect_identical(deparse1(model$spec$formula), "mpg ~ . - qsec")
  expect_output(print(model), "Formula: mpg ~ . - qsec", fixed = TRUE)

  # A missing value in the subtracted column leaves the row scored
  noted <- mtcars[1:5, ]
  noted$qsec[2] <- NA
  svm <- tl_model(mtcars, mpg ~ . - qsec, method = "svm")
  expect_false(anyNA(predict(svm, new_data = noted)$.pred))
})

test_that("a training column missing from new data is refused, not looked up", {
  # model.frame() took a same-named object from the caller when new data
  # lacked the column, so predictions came from a global hp
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  hp <- c(110, 110, 93, 110, 175)
  expect_error(
    predict(model, new_data = mtcars[1:5, "wt", drop = FALSE]),
    "New data is missing predictors used at fit time: hp"
  )
  poly <- tl_model(mtcars, mpg ~ wt + hp, method = "polynomial")
  expect_error(
    predict(poly, new_data = mtcars[1:5, "wt", drop = FALSE]),
    "New data is missing predictors used at fit time: hp"
  )

  # Every column present still predicts
  expect_equal(
    unname(predict(model, new_data = mtcars[1:5, c("wt", "hp")])$.pred),
    unname(stats::predict(model$fit, newdata = mtcars[1:5, ]))
  )

  # A variable the formula takes from its environment is not a column to
  # require
  k <- 0.01
  offset <- tl_model(mtcars, mpg ~ wt + offset(k * disp), method = "linear")
  expect_equal(
    unname(predict(offset, new_data = mtcars[1:5, c("wt", "disp")])$.pred),
    unname(stats::predict(offset$fit, newdata = mtcars[1:5, ]))
  )
})

test_that("writing out a subtracted formula keeps what the formula means", {
  # Writing y ~ . - x out added an empty column for every variable the
  # data lacked: a global vector became an extra main effect, a local
  # scalar in an offset failed with "variable lengths differ", and a
  # misspelt subtracted column was fitted as if it had been dropped
  expo <- mtcars$wt
  k <- 0.01
  global <- tl_model(mtcars, mpg ~ . - qsec + I(expo^2), method = "linear")
  expect_equal(
    coef(global$fit), coef(lm(mpg ~ . - qsec + I(expo^2), data = mtcars))
  )
  offset <- tl_model(mtcars, mpg ~ . - qsec + offset(k * disp),
                     method = "linear")
  expect_equal(
    coef(offset$fit),
    coef(lm(mpg ~ . - qsec + offset(k * disp), data = mtcars))
  )

  expect_error(
    tl_model(mtcars, mpg ~ . - qsce, method = "linear"),
    "The formula subtracts 'qsce', which is not a column of the data"
  )
  expect_error(
    tl_model(iris, ~ . - Sepal.Widht, method = "kmeans", k = 3),
    "The formula subtracts 'Sepal.Widht', which is not a column of the data"
  )
})

test_that("a formula may subtract a variable it finds in its environment", {
  # The check for a misspelt subtracted column refused every name the data
  # lacked, so mpg ~ wt * z - z, which lm() fits with z from the formula's
  # environment, failed with "The formula subtracts 'z', which is not a
  # column of the data"
  z <- mtcars$wt * 2
  model <- tl_model(mtcars, mpg ~ wt * z - z, method = "linear")
  reference <- lm(mpg ~ wt * z - z, data = mtcars)
  expect_equal(coef(model$fit), coef(reference))
  expect_named(coef(model$fit), c("(Intercept)", "wt", "wt:z"))
  expect_equal(unname(predict(model)$.pred), unname(fitted(reference)))

  # A misspelling is still refused; without a dot it would keep the term it
  # meant to remove. A function is no variable model.frame() can use, so a
  # misspelling that happens to name one, such as df, is refused too
  expect_error(
    tl_model(mtcars, mpg ~ wt * hp - wt:hpp, method = "linear"),
    "The formula subtracts 'hpp', which is not a column of the data"
  )
  expect_error(
    tl_model(mtcars, mpg ~ . - df, method = "linear"),
    "The formula subtracts 'df', which is not a column of the data"
  )

  # A one-sided formula names columns only, so an unsupervised method
  # refuses a subtracted name the data lacks whatever the environment holds
  Sepal.Widht <- iris$Sepal.Width # nolint: object_name_linter.
  expect_error(
    tl_model(iris, ~ . - Sepal.Widht, method = "kmeans", k = 3),
    "The formula subtracts 'Sepal.Widht', which is not a column of the data"
  )
  pca <- tl_model(iris, ~ . - Sepal.Width, method = "pca")
  expect_identical(
    rownames(pca$fit$model$rotation),
    c("Sepal.Length", "Petal.Length", "Petal.Width")
  )
})

test_that("a dot formula naming an outside variable warns as lm() does", {
  # terms() warns "'varlist' has changed ... should no longer happen!" for a
  # dot formula that names a variable the data lacks. lm() gives it once;
  # tidylearn's own terms() calls passed it on three or four more times
  varlist_warnings <- function(expr) {
    n <- 0L
    withCallingHandlers(expr, warning = function(w) {
      if (grepl("'varlist' has changed", conditionMessage(w), fixed = TRUE)) {
        n <<- n + 1L
        invokeRestart("muffleWarning")
      }
    })
    n
  }
  z <- mtcars$wt * 2
  expect_identical(varlist_warnings(lm(mpg ~ . + z, data = mtcars)), 1L)
  expect_identical(
    varlist_warnings(model <- tl_model(mtcars, mpg ~ . + z, method = "linear")),
    1L
  )
  expect_equal(coef(model$fit),
               suppressWarnings(coef(lm(mpg ~ . + z, data = mtcars))))

  # A classification fit reads its classes through one more terms() call
  d <- mtcars[c("am", "wt", "hp")]
  d$am <- factor(d$am)
  expect_identical(
    varlist_warnings(glm(am ~ . + z, family = binomial, data = d)), 1L
  )
  expect_identical(
    varlist_warnings(tl_model(d, am ~ . + z, method = "logistic")), 1L
  )

  # lm() has nothing to warn about once the dot is written out, and
  # neither has tl_model()
  expect_identical(
    varlist_warnings(tl_model(mtcars, mpg ~ wt * z - z, method = "linear")),
    0L
  )

  # Only that warning is muffled
  muffle <- tidylearn:::tl_muffle_varlist
  expect_warning(
    withCallingHandlers(warning("something else"), warning = muffle),
    "something else"
  )
  real <- tryCatch(
    stats::terms(mpg ~ . + z, data = mtcars),
    warning = function(w) w
  )
  expect_no_warning(withCallingHandlers(warning(real), warning = muffle))
})

test_that("a data-dependent term is computed with the training values", {
  skip_if_not_installed("xgboost")
  # The design was rebuilt from the formula on the rows predicted, so
  # scale(hp) was centred on whatever was scored: a row predicted
  # differently alone than inside mtcars, and a single row was NA
  model <- tl_model(mtcars, mpg ~ scale(hp) + wt, method = "xgboost",
                    nrounds = 20)
  within <- predict(model, new_data = mtcars)$.pred

  alone <- predict(model, new_data = mtcars[5, ])$.pred
  expect_false(is.na(alone))
  expect_equal(alone, within[5])
  expect_equal(predict(model, new_data = mtcars[1:3, ])$.pred, within[1:3])
})

test_that("deep computes a data-dependent term with the training values", {
  skip_on_cran()
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  has_backend <- tryCatch(
    !is.null(tensorflow::tf_version()),
    error = function(e) FALSE
  )
  if (!isTRUE(has_backend)) {
    skip("No TensorFlow backend available")
  }

  # The design was rebuilt from the formula on the rows predicted, as for
  # xgboost: scale(hp) of a single row is NaN, so the row predicted NA
  tensorflow::set_random_seed(1)
  model <- tl_model(mtcars, mpg ~ scale(hp) + wt, method = "deep",
                    epochs = 2, hidden_layers = 4, verbose = 0)
  within <- predict(model, new_data = mtcars)$.pred
  alone <- predict(model, new_data = mtcars[5, ])$.pred
  expect_false(is.na(alone))

  # keras on the design the training rows give, scaled as tl_fit_deep() does
  x <- cbind(`scale(hp)` = scale(mtcars$hp)[, 1], wt = mtcars$wt)
  x <- scale(x, center = model$fit$x_means, scale = model$fit$x_sds)
  direct <- stats::predict(model$fit$model, x[5, , drop = FALSE], verbose = 0)
  # keras rounds batches of different sizes slightly differently
  expect_equal(alone, as.numeric(direct), tolerance = 1e-6)
  expect_equal(alone, within[5], tolerance = 1e-6)
  expect_equal(predict(model, new_data = mtcars[1:3, ])$.pred, within[1:3],
               tolerance = 1e-6)
})

test_that("an extra column does not reach xgboost's design matrix", {
  skip_if_not_installed("xgboost")
  set.seed(3)
  train <- iris[sample(150, 120), ]
  test <- iris[-as.integer(rownames(train)), ]
  model <- tl_model(train, Species ~ ., method = "xgboost", nrounds = 5)
  extra <- test
  extra$batch_id <- seq_len(nrow(test))

  expect_no_warning(scored <- predict(model, new_data = extra))
  expect_identical(scored$.pred, predict(model, new_data = test)$.pred)
})

# ---- Refitting the wrapped object ----

test_that("update() and step() on model$fit refit on the training rows", {
  # The stored call named `data`, and update() evaluates it in the caller's
  # frame: a script that called its full frame `data` refitted on all 32
  # rows of an 18-row fit
  data <- mtcars
  train <- data[data$cyl != 8, ]
  model <- tl_model(train, mpg ~ wt + hp + qsec, method = "linear")

  updated <- stats::update(model$fit, . ~ . - qsec)
  expect_equal(stats::nobs(updated), nrow(train))
  expect_equal(coef(updated), coef(lm(mpg ~ wt + hp, data = train)))
  expect_equal(
    coef(stats::step(model$fit, trace = 0)),
    coef(stats::step(lm(mpg ~ wt + hp + qsec, data = train), trace = 0))
  )

  # A weighted fit stored `weights`, which found stats::weights()
  w <- train$cyl
  weighted <- tl_model(train, mpg ~ wt + hp, method = "linear", weights = w)
  expect_equal(
    coef(stats::update(weighted$fit, . ~ . - hp)),
    coef(lm(mpg ~ wt, data = train, weights = w))
  )

  # rpart and randomForest record their calls the same way
  tree <- tl_model(train, mpg ~ wt + hp, method = "tree", minsplit = 5)
  expect_equal(stats::update(tree$fit, . ~ . - hp)$frame$n[1], nrow(train))
  forest <- tl_model(train, mpg ~ wt + hp, method = "forest", ntree = 5)
  expect_length(stats::update(forest$fit, ntree = 5)$y, nrow(train))

  # The call still prints in a line rather than spilling the data
  expect_lt(nchar(paste(deparse(weighted$fit$call), collapse = "")), 150)
})

test_that("update() on a fit finds its fitting function outside tidylearn", {
  # The stored call named the function bare, which a session that has not
  # attached rpart, randomForest, e1071, nnet or gbm cannot find: update()
  # failed with "could not find function \"rpart\""
  d <- mtcars[rep(seq_len(nrow(mtcars)), 2), ]
  fits <- list(
    tree = tl_model(d, mpg ~ wt + hp, method = "tree")$fit,
    forest = tl_model(d, mpg ~ wt + hp, method = "forest", ntree = 5)$fit,
    svm = tl_model(d, mpg ~ wt + hp, method = "svm")$fit,
    nn = tl_model(d, mpg ~ wt + hp, method = "nn", size = 2,
                  trace = FALSE)$fit,
    boost = tl_model(d, mpg ~ wt + hp, method = "boost", n.trees = 10)$fit
  )

  # load_all() puts tidylearn's imports on the search path, so the call is
  # run where only R's default packages are attached, as in a session that
  # has loaded tidylearn and nothing else
  outside <- new.env(parent = as.environment("package:stats"))
  for (name in names(fits)) {
    outside$fit <- fits[[name]]
    refit <- eval(quote(stats::update(fit)), outside)
    expect_s3_class(refit, class(fits[[name]])[1])
  }
})

test_that("update() on model$fit needs no `data` in scope", {
  # With nothing called `data` in scope it found utils::data() and failed
  # with "'data' must be a data.frame"
  model <- tl_model(mtcars[1:20, ], mpg ~ wt + hp, method = "linear")
  expect_equal(stats::nobs(stats::update(model$fit, . ~ . - hp)), 20)
})

test_that("an offset argument is refused with advice that fits the method", {
  # Every method was told to put offset() in its formula, which only lm()
  # and glm() apply at predict(): rpart, gbm and nnet drop it
  fit <- function(fun, name) {
    tidylearn:::tl_fit_by_value(
      fun, name,
      list(formula = mpg ~ wt, data = mtcars, offset = mtcars$hp / 100)
    )
  }
  expect_error(fit(stats::lm, "lm"), "in the formula, as offset\\(<column>\\)")
  expect_error(
    fit(rpart::rpart, "rpart"),
    "rpart\\(\\) cannot apply an offset to new data"
  )
  err <- tryCatch(fit(rpart::rpart, "rpart"), error = conditionMessage)
  expect_match(err, "method = \"linear\" or \"logistic\"")
  expect_no_match(err, "Pass an offset in the formula")
})

# ---- Fitting on a subset ----

test_that("a subset fit keeps only the rows it was fitted on", {
  # model$data kept all 32 rows of an 11-row fit, so the influence
  # measures failed with "differing number of rows: 32, 11" and
  # tl_evaluate() reported an rmse of 3.24 over rows the model never saw
  # The 11 rows hold few distinct values of mpg, which tl_model() notes
  fit <- function(...) suppressMessages(tl_model(...))
  four <- mtcars[mtcars$cyl == 4, ]
  model <- fit(mtcars, mpg ~ wt, method = "linear", subset = mtcars$cyl == 4)

  expect_identical(rownames(model$data), rownames(four))
  expect_equal(coef(model$fit), coef(lm(mpg ~ wt, data = four)))
  expect_equal(nrow(tl_influence_measures(model)), 11)
  expect_equal(
    tl_evaluate(model, metrics = "rmse")$value,
    sqrt(mean(residuals(lm(mpg ~ wt, data = four))^2))
  )
  expect_equal(nrow(predict(model)), 11)
  expect_true("subset" %in% model$spec$per_row_args)

  # The other per-row arguments follow the same rows
  w <- seq_len(32)
  weighted <- fit(mtcars, mpg ~ wt, method = "linear",
                  subset = mtcars$cyl == 4, weights = w)
  expect_equal(
    coef(weighted$fit),
    coef(lm(mpg ~ wt, data = mtcars, subset = cyl == 4, weights = w))
  )

  # glmnet has no subset argument and used to fit every row
  ridge <- fit(mtcars, mpg ~ wt + hp, method = "ridge",
               subset = mtcars$cyl != 8, lambda = 0.1)
  expect_equal(ridge$fit$nobs, sum(mtcars$cyl != 8))

  expect_error(
    fit(mtcars, mpg ~ wt, method = "linear", subset = mtcars$cyl > 10),
    "'subset' selects none of the 32 rows"
  )
})

# ---- Unsupervised formulas ----

test_that("a one-sided formula takes `- x` out of the dot", {
  # all.vars() returned "." for `~ . - Sepal.Width`, and selecting a column
  # called "." failed with "undefined columns selected"
  kept <- c("Sepal.Length", "Petal.Length", "Petal.Width")
  expect_identical(tidylearn:::get_formula_vars(~ . - Sepal.Width, iris), kept)
  model <- tl_model(iris, ~ . - Sepal.Width, method = "kmeans", k = 3)
  expect_identical(colnames(model$fit$model$centers), kept)

  # The dot still means every numeric column, and named columns are kept
  expect_identical(tidylearn:::get_formula_vars(~ ., iris), names(iris)[1:4])
  expect_identical(
    tidylearn:::get_formula_vars(~ Sepal.Length + Species, iris),
    c("Sepal.Length", "Species")
  )
  odd <- data.frame(`a b` = 1:3, c = 4:6, check.names = FALSE)
  expect_identical(
    tidylearn:::get_formula_vars(~ `a b` + c, odd), c("a b", "c")
  )
})

test_that("a transformation in a one-sided formula is refused by name", {
  # PCA ran on the raw column: ~ log(Sepal.Length) + Sepal.Width centred
  # at 5.84, the mean of Sepal.Length itself
  expect_error(
    tl_model(iris, ~ log(Sepal.Length) + Sepal.Width, method = "pca"),
    "name columns only, but this one has log\\(Sepal.Length\\)"
  )
  expect_error(
    tl_model(iris, ~ Sepal.Length:Sepal.Width, method = "kmeans", k = 2),
    "has Sepal.Length:Sepal.Width"
  )
})

test_that("a one-sided formula naming a non-column is refused by name", {
  # The name was kept as a column to select, and every method failed with
  # "undefined columns selected", with a dot or without one, whether or not
  # the caller's session had an object of that name
  z <- mtcars$wt * 2
  expect_error(
    tl_model(mtcars, ~ . + z, method = "kmeans", k = 2),
    "name columns only, but 'z' is not a column of the data"
  )
  expect_error(
    tl_model(mtcars, ~ wt + zz, method = "pca"),
    "name columns only, but 'zz' is not a column of the data"
  )
  expect_error(
    tl_model(mtcars, ~ zz + wt + yy, method = "hclust"),
    "name columns only, but 'zz', 'yy' are not columns of the data"
  )

  # Columns still fit, named or under a dot
  set.seed(1)
  named <- tl_model(mtcars, ~ wt + hp, method = "kmeans", k = 2)
  fit <- named$fit$model
  expect_identical(colnames(fit$centers), c("wt", "hp"))
  # Each centre is the mean of its cluster's rows over those two columns
  expect_equal(
    unname(fit$centers[1, ]),
    unname(colMeans(mtcars[fit$cluster == 1, c("wt", "hp")]))
  )
  dotted <- tl_model(mtcars, ~ ., method = "pca")
  expect_identical(rownames(dotted$fit$model$rotation), names(mtcars))
})

# ---- Unsupervised predict() output ----

test_that("predict() on a reduction keeps .obs_id and the kept components", {
  # truncate_components() counted .obs_id as a component, so a 2-D MDS fit
  # predicted .obs_id and Dim1, and PCA's training projection returned
  # every component however many were kept
  mds <- tl_reduce_dimensions(USArrests, method = "mds", n_components = 2)
  scores <- predict(mds$reduction_model)
  expect_identical(names(scores), c(".obs_id", "Dim1", "Dim2"))
  expect_identical(scores$.obs_id, rownames(USArrests))
  expect_equal(
    unname(as.matrix(scores[, -1])),
    unname(stats::cmdscale(stats::dist(USArrests), k = 2))
  )

  plain <- tl_reduce_dimensions(tibble::as_tibble(USArrests),
                                method = "mds", n_components = 2)
  expect_identical(names(predict(plain$reduction_model)),
                   c(".obs_id", "Dim1", "Dim2"))

  pca <- tl_reduce_dimensions(USArrests, method = "pca", n_components = 2)
  trained <- predict(pca$reduction_model)
  projected <- predict(pca$reduction_model, new_data = USArrests)
  expect_identical(names(trained), c(".obs_id", "PC1", "PC2"))
  expect_identical(names(projected), names(trained))
  expect_equal(as.matrix(trained[, -1]), as.matrix(projected[, -1]))
})

test_that("unsupervised predict() names observations by row on both paths", {
  # k-means scored new data without .obs_id, and PCA numbered new rows
  # "1", "2" whatever they were called
  states <- c("Ohio", "Texas")
  km <- tl_model(USArrests, method = "kmeans", k = 3)
  assigned <- predict(km, new_data = USArrests[states, ])
  expect_identical(names(assigned), names(predict(km)))
  expect_identical(assigned$.obs_id, states)
  all_states <- predict(km, new_data = USArrests)
  expect_identical(
    assigned$cluster,
    all_states$cluster[match(states, all_states$.obs_id)]
  )

  pca <- tl_model(USArrests, method = "pca")
  expect_identical(predict(pca, new_data = USArrests[states, ])$.obs_id, states)
})

# ---- Task detection ----

test_that("the task is read from the response the formula computes", {
  # The checks read the raw column: factor(cyl) ~ wt was fitted by lm() on
  # the factor codes, and ridge treated it as regression and failed with
  # "invalid to change the storage mode of a factor"
  expect_error(
    tl_model(mtcars, factor(cyl) ~ wt, method = "linear"),
    "'factor\\(cyl\\)' is a factor with 3 classes"
  )

  # Doubled, so no class falls under glmnet's eight-observation warning
  doubled <- mtcars[rep(seq_len(nrow(mtcars)), 2), ]
  ridge <- tl_model(doubled, factor(cyl) ~ wt + hp, method = "ridge")
  expect_true(ridge$spec$is_classification)
  expect_identical(ridge$spec$response_levels, c("4", "6", "8"))
  expect_identical(
    levels(predict(ridge, type = "class")$.pred), c("4", "6", "8")
  )
  lasso <- tl_model(doubled, factor(am) ~ wt + hp, method = "lasso")
  expect_identical(lasso$spec$response_levels, c("0", "1"))
  expect_setequal(names(predict(lasso, type = "prob")), c("0", "1"))

  set.seed(2)
  forest <- tl_model(mtcars, factor(cyl) ~ wt + hp, method = "forest",
                     ntree = 50)
  expect_true(forest$spec$is_classification)
  expect_identical(forest$fit$type, "classification")
  expect_setequal(names(predict(forest, type = "prob")), c("4", "6", "8"))

  # The computed response is what the checks see; the column is left alone
  computed <- tidylearn:::tl_formula_response(I(mpg > 20) ~ wt, mtcars)
  expect_identical(as.vector(computed), mtcars$mpg > 20)
  expect_identical(
    tidylearn:::tl_formula_response(mpg ~ wt, mtcars), mtcars$mpg
  )

  # A response that is numeric however it is computed is still regression
  logged <- tl_model(mtcars, log(mpg) ~ wt, method = "linear")
  expect_false(logged$spec$is_classification)
  expect_identical(logged$data, mtcars)
})

test_that("the classes are those of the rows the fit uses", {
  # Every setosa row has a missing predictor, so glmnet, svm and nnet fit
  # two classes; the spec listed three, a binary model was plotted as
  # multiclass, and nn failed with "returned 2 output columns for 3 classes"
  d <- iris
  d$Sepal.Width[d$Species == "setosa"] <- NA
  fitted <- c("versicolor", "virginica")

  lasso <- tl_model(d, Species ~ ., method = "lasso")
  expect_identical(lasso$spec$response_levels, fitted)
  expect_identical(
    lasso$spec$response_levels, attr(lasso$fit, "response_levels")
  )
  expect_identical(
    names(predict(lasso, new_data = iris[51:60, ], type = "prob")), fitted
  )

  nn <- suppressWarnings(
    tl_model(d, Species ~ ., method = "nn", size = 2, trace = FALSE)
  )
  expect_identical(nn$spec$response_levels, fitted)
  expect_equal(nrow(predict(nn, new_data = iris[51:60, ], type = "prob")), 10)

  # A method that fits rows with missing predictors keeps every class
  tree <- tl_model(d, Species ~ ., method = "tree")
  expect_identical(tree$spec$response_levels, levels(iris$Species))

  # One class left is refused, not fitted
  binary <- droplevels(iris[iris$Species != "virginica", ])
  binary$Sepal.Width[binary$Species == "setosa"] <- NA
  expect_error(
    tl_model(binary, Species ~ ., method = "logistic"),
    "in the rows left 'Species' has only one class \\(\"versicolor\"\\)"
  )
})

test_that("a supervised method refuses a formula with no response", {
  # ~ wt read wt as the response and failed with "incompatible
  # dimensions"; no formula at all failed with "argument is not a valid
  # model"
  expect_error(
    tl_model(mtcars, ~ wt, method = "linear"),
    "\"linear\" is supervised and needs a two-sided formula naming the resp"
  )
  expect_error(
    tl_model(mtcars, method = "forest"),
    "\"forest\" is supervised and needs a two-sided formula naming the resp"
  )
})

test_that("the logistic conversion warning has a class of its own", {
  # Resampling refits once per fold and warned once per fold; the class
  # lets a refitting loop let the first through and muffle the rest
  set.seed(1)
  d <- data.frame(x = stats::rnorm(40))
  d$y <- as.integer(d$x + stats::rnorm(40) > 0)
  expect_warning(
    tl_model(d, y ~ x, method = "logistic"),
    "Converting response variable to factor for logistic regression",
    class = "tidylearn_response_conversion"
  )
})

test_that("compute = \"auto\" hands the advisor the hyperparameters it reads", {
  # Only numeric scalars were forwarded, so hidden_layers = c(512, 512)
  # was dropped and the advisor sized the default c(32, 16)
  seen <- NULL
  local_mocked_bindings(
    tl_resolve_compute = function(method, data, formula, compute = "cpu",
                                  hyperparams = list()) {
      seen <<- hyperparams
      stop("resolved")
    }
  )
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "deep", compute = "auto",
             hidden_layers = c(512, 512), epochs = 3, verbose = 0),
    "resolved"
  )
  expect_identical(seen$hidden_layers, c(512, 512))
  expect_identical(seen$epochs, 3)
})

# ---- predict() types ----

test_that("a classification prediction type on a regression is an error", {
  # type = "prob" and "class" were ignored, and both returned the
  # regression's numeric predictions
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")

  expect_error(
    predict(model, type = "prob"),
    "type = \"prob\" needs a classification model"
  )
  expect_error(
    predict(model, type = "class"),
    "type = \"class\" needs a classification model"
  )
  expect_error(
    predict(model, type = "probability"),
    "Invalid prediction type \"probability\""
  )
  expect_equal(
    unname(predict(model, type = "response")$.pred),
    unname(fitted(model$fit))
  )

  forest <- tl_model(iris, Species ~ ., method = "forest", ntree = 10)
  expect_error(predict(forest, type = "probs"), "Invalid prediction type")
  expect_equal(nrow(predict(forest, type = "class")), 150)
})

# ---- Engineered features ----

test_that("a PCA-feature model reads its own stored data as projected", {
  # The stored data holds the component scores, not the raw columns, so
  # passing it back explicitly -- tl_evaluate(m, m$data), or a plot that
  # defaults to it -- projected it again and failed with "PCA was fitted
  # on 4 column(s) ... new_data is missing"
  reduced <- tl_reduce_dimensions(
    iris, response = "Species", method = "pca", n_components = 3
  )
  model <- tl_model(reduced$data, Species ~ PC1 + PC2 + PC3, method = "tree")
  model$feature_transform <- list(
    kind = "pca", reduction_model = reduced$reduction_model,
    response = "Species"
  )

  stored <- predict(model)$.pred
  expect_identical(predict(model, new_data = model$data)$.pred, stored)
  # Raw data is still projected first
  expect_identical(predict(model, new_data = iris)$.pred, stored)
  expect_equal(
    tl_evaluate(model, model$data, metrics = "accuracy")$value,
    mean(stored == iris$Species)
  )
})

# ---- plot() dispatch ----

test_that("plot() routes regularised importance and refuses its diagnostics", {
  # type = "importance" said "not implemented" though the plot exists, and
  # type = "diagnostics" failed inside rstandard()
  ridge <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "ridge")
  p <- plot(ridge, type = "importance")
  geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
  expect_true("GeomCol" %in% geoms)
  expect_equal(p$data, tl_plot_importance_regularized(ridge)$data)

  expect_error(
    plot(ridge, type = "diagnostics"),
    "Diagnostic plots need a model fitted by lm\\(\\) or glm\\(\\)"
  )

  # lm and glm fits keep theirs
  linear <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_length(plot(linear, type = "diagnostics"), 4)
})

test_that("magrittr's %>% stays exported for existing user code", {
  # The package itself pipes with |>. The re-export is kept so code that
  # used %>% after library(tidylearn) alone does not break.
  expect_true("%>%" %in% getNamespaceExports("tidylearn"))
  expect_identical(tidylearn::`%>%`, magrittr::`%>%`)

  # nolint start: pipe_consistency_linter.
  pred <- tl_model(mtcars, mpg ~ wt, method = "linear") %>%
    predict(new_data = mtcars[1:3, ])
  # nolint end
  expect_equal(nrow(pred), 3)
})
