# ---- Degenerate inputs to tl_pipeline, the supervised grid, and the
# ---- credentialled read backends.
#
# Every case here produced either a hang, a silently wrong result, or a
# message from a backend that named neither the argument nor the cause.

# ---- The forest hang ----

test_that("forest refuses a classification fit with no varying predictor", {
  # randomForest's classification path does not terminate here: it keeps
  # drawing mtry candidates looking for a split that cannot exist, in C,
  # so the session has to be killed. The guard has to fire before the
  # call, not around it.
  d <- data.frame(
    x1 = 1, x2 = 1, x3 = 1,
    y = factor(rep(c("hi", "lo"), length.out = 20))
  )

  expect_error(
    tl_model(d, y ~ ., method = "forest"),
    "every predictor is constant"
  )
})

test_that("the constant-predictor guard is narrow", {
  set.seed(1)
  n <- 30

  # Regression is unaffected -- a regression forest predicts the mean
  d_reg <- data.frame(x1 = 1, x2 = 1, y = rnorm(n))
  expect_s3_class(tl_model(d_reg, y ~ ., method = "forest"), "tidylearn_model")

  # So is a frame where only some predictors are constant
  d_mixed <- data.frame(
    x1 = 1, x2 = rnorm(n),
    y = factor(rep(c("hi", "lo"), length.out = n))
  )
  expect_s3_class(
    tl_model(d_mixed, y ~ ., method = "forest"), "tidylearn_model"
  )
})

# ---- Formula normalisation ----

test_that("a character formula is coerced at every entry point", {
  # tl_model() always coerced with as.formula(), but the coercion lives
  # inside tl_model_supervised(). Callers that read all.vars(formula)[1]
  # first saw character(0), took the response name as NA, and treated a
  # classification problem as regression.
  expect_s3_class(
    tl_model(iris, "Species ~ .", method = "tree"), "tidylearn_model"
  )
  expect_type(tl_prepare_data(iris, "Species ~ ."), "list")
  expect_type(tl_cv(iris, "Species ~ .", method = "tree", folds = 3), "list")
  expect_s3_class(
    tl_tune_grid(iris, "Species ~ .",
      method = "tree",
      param_grid = list(cp = 0.01), folds = 3, verbose = FALSE
    ),
    "tidylearn_model"
  )
})

test_that("a character formula reaches the pipeline as classification", {
  pipe <- tl_pipeline(iris, "Species ~ .",
    models = list(t = list(method = "tree"))
  )

  # The tell is the metric set: regression defaults meant the task was
  # misread, and every score came back NA
  expect_true("accuracy" %in% pipe$evaluation$metrics)
  expect_false("rmse" %in% pipe$evaluation$metrics)

  run <- tl_run_pipeline(pipe, verbose = FALSE)
  expect_false(any(is.na(run$results$metric_values)))
})

test_that("something that is not a formula is refused by name", {
  models <- list(t = list(method = "tree"))

  expect_error(tl_pipeline(iris, 42, models = models), "must be a formula")
  expect_error(tl_pipeline(iris, NULL, models = models), "must be a formula")
  expect_error(
    tl_pipeline(iris, c("a ~ b", "c ~ d"), models = models), "length 2"
  )
  expect_error(
    tl_pipeline(iris, "not a formula", models = models), "does not parse"
  )
})

# ---- Pipeline model specifications ----

test_that("a repeated model name is refused", {
  # models[[name]] resolves to the first match, so the loop fitted one
  # spec twice and dropped the other without saying so
  pipe <- tl_pipeline(iris, Species ~ ., models = list(
    a = list(method = "tree"), a = list(method = "forest")
  ))

  expect_error(tl_run_pipeline(pipe, verbose = FALSE), "repeated name")
})

test_that("a malformed model spec names the model responsible", {
  mk <- function(models) tl_pipeline(iris, Species ~ ., models = models)

  expect_error(
    tl_run_pipeline(mk(list(tree = list())), verbose = FALSE),
    "Model 'tree' has no 'method'"
  )
  expect_error(
    tl_run_pipeline(mk(list(tree = "tree")), verbose = FALSE),
    "Model 'tree' must be a list"
  )
  expect_error(
    tl_run_pipeline(
      mk(list(m = list(method = c("tree", "forest")))),
      verbose = FALSE
    ),
    "single method name"
  )
  expect_error(
    tl_run_pipeline(mk(list(m = list(method = "banana"))), verbose = FALSE),
    "not a supervised method"
  )
  # An unsupervised method has no response to fit
  expect_error(
    tl_run_pipeline(mk(list(m = list(method = "kmeans"))), verbose = FALSE),
    "not a supervised method"
  )
})

test_that("a formula with no predictors is refused", {
  pipe <- tl_pipeline(iris, Species ~ 1,
    models = list(t = list(method = "tree"))
  )

  expect_error(tl_run_pipeline(pipe, verbose = FALSE), "names no predictors")
})

# ---- Pipeline evaluation specifications ----

test_that("cv_folds is validated against its own name", {
  mk <- function(folds) {
    tl_pipeline(iris, Species ~ .,
      models = list(t = list(method = "tree")),
      evaluation = list(cv_folds = folds)
    )
  }

  for (bad in list(0, 1, -3, 2.7)) {
    expect_error(mk(bad), "cv_folds must be a single whole number")
  }
  # More folds than rows is a different mistake and says so
  expect_error(
    tl_run_pipeline(mk(1000), verbose = FALSE), "but the data has 150 rows"
  )
})

test_that("train_prop is validated against its own name", {
  mk <- function(prop) {
    tl_pipeline(iris, Species ~ .,
      models = list(t = list(method = "tree")),
      evaluation = list(validation = "split", train_prop = prop)
    )
  }

  for (bad in list(0, 1, 1.5, -0.2)) {
    expect_error(mk(bad), "train_prop must be a single number")
  }
  expect_s3_class(mk(0.7), "tidylearn_pipeline")
})

test_that("an unknown metric is refused rather than scored NA", {
  mk <- function(metrics, best) {
    tl_pipeline(iris, Species ~ .,
      models = list(t = list(method = "tree")),
      evaluation = list(metrics = metrics, best_metric = best)
    )
  }

  # A metric that is never computed left every score NA, and the run
  # warned about the symptom rather than the cause
  expect_error(mk("banana", "banana"), "Unknown classification metric")
  expect_error(mk("rmse", "rmse"), "Unknown classification metric")
  expect_error(mk(character(0), "accuracy"), "metrics is empty")

  # Regression is judged against its own list
  expect_error(
    tl_pipeline(mtcars, mpg ~ .,
      models = list(t = list(method = "linear")),
      evaluation = list(metrics = "accuracy", best_metric = "accuracy")
    ),
    "Unknown regression metric"
  )
})

# ---- Single-class responses ----

test_that("every classification method names a single-class response", {
  set.seed(2)
  n <- 30
  d <- data.frame(
    x1 = rnorm(n), x2 = rnorm(n),
    y = factor(rep("hi", n), levels = c("hi", "lo"))
  )

  # Only logistic used to say this. The rest reported whatever their
  # backend hit first -- rpart "number of rows of matrices must match",
  # glmnet "non-conformable arguments", e1071 "Model is empty!".
  methods <- c(
    "logistic", "tree", "forest", "boost", "ridge",
    "lasso", "elastic_net", "svm", "nn", "xgboost"
  )

  for (method in methods) {
    expect_error(
      tl_model(d, y ~ ., method = method),
      "one|two levels|two classes",
      info = method
    )
  }
})

# ---- Read backends ----

test_that("tl_read_s3 reports a malformed source rather than indexing it", {
  skip_if_not_installed("paws.storage")

  # strsplit(character(0), ...)[[1]] was "subscript out of bounds", the
  # one input that missed the Invalid S3 URI message
  expect_error(tl_read_s3(character(0)), "single S3 URI")
  expect_error(tl_read_s3(c("s3://a/b.csv", "s3://c/d.csv")), "length 2")
  expect_error(tl_read_s3(NULL), "must be an S3 URI string")
  expect_error(tl_read_s3(list(1)), "must be an S3 URI string")

  # A well-formed URI still parses, and fails later for its own reasons
  expect_error(tl_read_s3("s3://bucket"), "Invalid S3 URI")
  expect_error(tl_read_s3("s3://bucket/key"), "Cannot detect file format")
})

test_that("tl_read_db validates its connection and query", {
  skip_if_not_installed("RSQLite")
  conn <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
  on.exit(try(DBI::dbDisconnect(conn), silent = TRUE), add = TRUE)
  DBI::dbWriteTable(conn, "t", head(mtcars, 5))

  expect_error(tl_read_db("not a conn", "SELECT 1"), "DBI connection")
  expect_error(tl_read_db(NULL, "SELECT 1"), "DBI connection")
  expect_error(tl_read_db(conn, ""), "non-empty SQL string")
  expect_error(tl_read_db(conn, NA), "non-empty SQL string")
  expect_error(tl_read_db(conn, character(0)), "non-empty SQL string")

  expect_equal(nrow(tl_read_db(conn, "SELECT mpg FROM t")), 5)
  # An empty result is a result, and says so
  expect_warning(
    empty <- tl_read_db(conn, "SELECT * FROM t WHERE mpg > 9999"),
    "0 rows"
  )
  expect_equal(nrow(empty), 0)
})
