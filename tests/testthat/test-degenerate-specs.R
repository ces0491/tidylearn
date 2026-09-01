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
  expect_s3_class(
    tl_tune_random(iris, "Species ~ .",
      method = "tree",
      param_space = list(cp = c(0.001, 0.1)), n_iter = 2, folds = 3,
      verbose = FALSE, seed = 1
    ),
    "tidylearn_model"
  )

  # tl_auto_ml() announced "task: regression" for this and returned an
  # unranked leaderboard. Its own tests are skip_on_cran(), so the
  # coercion is asserted here as well.
  auto <- suppressMessages(
    tl_auto_ml(iris, "Species ~ .",
      use_reduction = FALSE, use_clustering = FALSE, time_budget = 60
    )
  )
  expect_equal(auto$task, "classification")
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

  # In range, but it rounds to an empty side on a small frame. An empty
  # training set was "result would be too long a vector"; an empty test
  # set scored nothing and the run finished with a leaderboard of NAs.
  small <- function(prop) {
    tl_pipeline(head(iris, 10), Species ~ .,
      models = list(t = list(method = "tree")),
      evaluation = list(
        validation = "split", train_prop = prop,
        metrics = "accuracy", best_metric = "accuracy"
      )
    )
  }
  expect_error(
    tl_run_pipeline(small(0.01), verbose = FALSE),
    "0 rows in the training set"
  )
  expect_error(
    tl_run_pipeline(small(0.99), verbose = FALSE),
    "0 in the test set"
  )
})

test_that("validation and best_metric are refused by name when absent", {
  mk <- function(evaluation) {
    tl_pipeline(iris, Species ~ .,
      models = list(t = list(method = "tree")), evaluation = evaluation
    )
  }

  # modifyList() drops an element set to NULL, so these reached `%in%`
  # as logical(0) and failed with "argument is of length zero"
  expect_error(mk(list(validation = NULL)), "validation must be")
  expect_error(mk(list(best_metric = NULL)), "best_metric")
  expect_error(mk(list(validation = 3)), "validation must be")
  expect_error(mk(list(best_metric = c("f1", "auc"))), "best_metric")
})

test_that("a logistic pipeline on a 0/1 response is scored as classification", {
  set.seed(5)
  n <- 60
  d <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  d$y <- as.integer(d$x1 + rnorm(n) > 0)

  # tl_model() coerces the response for logistic and fits a classifier,
  # so each fold reports accuracy and auc. Deciding the task from the
  # column alone called this regression and refused both.
  pipe <- tl_pipeline(d, y ~ .,
    models = list(lg = list(method = "logistic")),
    evaluation = list(
      metrics = c("accuracy", "auc"), best_metric = "accuracy",
      cv_folds = 3
    )
  )

  run <- suppressWarnings(tl_run_pipeline(pipe, verbose = FALSE))
  expect_false(any(is.na(run$results$metric_values)))

  # A regression pipeline on the same column still gets regression metrics
  expect_true(
    "rmse" %in% tl_pipeline(d, y ~ .,
      models = list(ln = list(method = "linear"))
    )$evaluation$metrics
  )

  # Both at once is two tasks in one run. The leaderboard scored logistic
  # 0.6998 and linear NA, and the linear candidate then dropped out of the
  # comparison without a word
  expect_error(
    tl_pipeline(d, y ~ .,
      models = list(
        lg = list(method = "logistic"), ln = list(method = "linear")
      )
    ),
    "puts logistic regression alongside linear"
  )

  # Once the response is a factor there is only one task, and a mixed
  # spec is the ordinary case
  d$y <- factor(d$y)
  expect_s3_class(
    tl_pipeline(d, y ~ .,
      models = list(lg = list(method = "logistic"), tr = list(method = "tree"))
    ),
    "tidylearn_pipeline"
  )
})

test_that("a response that is not a column is refused where it is read", {
  # This read as NULL, set regression defaults, and failed several steps
  # later inside rpart with "object 'Speces' not found"
  expect_error(
    tl_pipeline(iris, Speces ~ ., models = list(t = list(method = "tree"))),
    "is not a column of `data`"
  )
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
  #
  # The assertion has to be the guard's own wording. glmnet's message is
  # "one multinomial or binomial class has 1 or 0 observations", so a
  # pattern loose enough to match "one" passes without the guard.
  shared <- c(
    "tree", "forest", "boost", "ridge",
    "lasso", "elastic_net", "svm", "nn", "deep", "xgboost"
  )

  for (method in shared) {
    expect_error(
      tl_model(d, y ~ ., method = method),
      "needs a response with at least two classes",
      info = method
    )
  }

  # Logistic keeps its own wording, and the two numeric-response methods
  # keep theirs -- but all thirteen refuse, and say which response
  expect_error(
    tl_model(d, y ~ ., method = "logistic"),
    "needs a response with two levels"
  )
  for (method in c("linear", "polynomial")) {
    expect_error(
      tl_model(d, y ~ ., method = method),
      "holding a single class", info = method
    )
    # A method that would also refuse it is no remedy to suggest
    expect_error(
      tl_model(d, y ~ ., method = method),
      "no classification method will fit it either", info = method
    )
  }

  # The multiclass wording is untouched: the equally-spaced-codes
  # argument is the point there, and the alternatives can fit
  expect_error(
    tl_model(iris, Species ~ ., method = "linear"),
    "equally spaced"
  )
})

test_that("the constant-predictor guard reads an exclusion from the formula", {
  set.seed(4)
  n <- 30
  d <- data.frame(
    id = 1,                             # constant, and excluded below
    x1 = rnorm(n), x2 = rnorm(n),
    y = factor(rep(c("hi", "lo"), length.out = n))
  )

  # all.vars() on the raw formula collapses `y ~ . - id` to the one column
  # the caller excluded, so a fittable frame was refused for the state of
  # a column the model never sees
  expect_s3_class(
    tl_model(d, y ~ . - id, method = "forest"), "tidylearn_model"
  )

  # And the mirror case: the excluded column is the only one that varies,
  # so the set the guard has to judge really is all constant
  d2 <- data.frame(
    id = rnorm(n), x1 = 1, x2 = 1,
    y = factor(rep(c("hi", "lo"), length.out = n))
  )
  expect_error(
    tl_model(d2, y ~ . - id, method = "forest"),
    "every predictor is constant"
  )
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

# ---- Backend defaults over a formula narrower than the frame ----

test_that("forest and svm defaults count the formula's predictors", {
  skip_if_not_installed("randomForest")
  skip_if_not_installed("e1071")

  # These were derived from ncol(data) - 1, so an explicit formula over a
  # wider frame got a default meant for every column in it
  expect_silent(forest <- tl_model(mtcars, mpg ~ wt + hp, method = "forest"))
  expect_equal(forest$fit$mtry, max(floor(2 / 3), 1))

  # The silent case, and the one that matters: mtry equal to the number
  # of predictors samples all of them at every split, which is bagging
  iris_forest <- tl_model(iris, Species ~ Sepal.Length + Sepal.Width,
    method = "forest"
  )
  expect_equal(iris_forest$fit$mtry, floor(sqrt(2)))

  # e1071's default kernel width is 1 / ncol(design matrix)
  svm_fit <- tl_model(mtcars, mpg ~ wt + hp, method = "svm")
  expect_equal(svm_fit$fit$gamma, 1 / 2)

  # A `y ~ .` formula was always right, and stays so
  expect_equal(
    tl_model(iris, Species ~ ., method = "forest")$fit$mtry, floor(sqrt(4))
  )
  expect_equal(tl_model(mtcars, mpg ~ ., method = "svm")$fit$gamma, 1 / 10)

  # An explicit value still wins over the backend default
  expect_equal(
    tl_model(mtcars, mpg ~ ., method = "forest", mtry = 4)$fit$mtry, 4
  )
  expect_equal(
    tl_model(mtcars, mpg ~ ., method = "svm", gamma = 0.25)$fit$gamma, 0.25
  )
})

test_that("the backends store a call that does not carry the data", {
  skip_if_not_installed("randomForest")
  skip_if_not_installed("e1071")

  # Leaving an argument out takes do.call(), which evaluates first, so
  # match.call() inside the backend recorded the whole frame as a
  # literal. print() on the fit spilled every row, and on a 960-row
  # frame the call alone was 159 Kb of a 1.5 Mb forest.
  big <- mtcars[rep(seq_len(nrow(mtcars)), 30), ]

  for (fit in list(
    tl_model(big, mpg ~ wt + hp, method = "forest", ntree = 10)$fit,
    tl_model(big, mpg ~ wt + hp, method = "svm")$fit
  )) {
    expect_identical(fit$call$data, quote(data))
    expect_lt(as.numeric(utils::object.size(fit$call)), 10000)
  }
})
