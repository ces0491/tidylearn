# Metric values checked against hand computations.
#
# The existing metric tests assert only that a value is present and not
# NA. That passes just as happily when every binary metric describes the
# negative class, which is exactly what happened: yardstick defaults to
# event_level = "first" while the rest of the package treats the second
# factor level as positive. A number that is merely present proves
# nothing -- these check the number itself.

# 90 negatives, 10 positives. The model calls everything positive, so
# with "pos" as the positive class:
# nolint start: commented_code_linter.
#   TP = 10, FP = 90, TN = 0, FN = 0
#   sensitivity = TP / (TP + FN) = 10/10  = 1.0
#   specificity = TN / (TN + FP) =  0/90  = 0.0
#   precision   = TP / (TP + FP) = 10/100 = 0.1
#   F1          = 2PR / (P + R)           = 0.1818...
# nolint end
all_positive_truth <- factor(
  c(rep("neg", 90), rep("pos", 10)),
  levels = c("neg", "pos")
)
all_positive_pred <- factor(rep("pos", 100), levels = c("neg", "pos"))

test_that("binary metrics describe the second factor level", {
  result <- suppressWarnings(tl_calc_classification_metrics(
    all_positive_truth, all_positive_pred,
    metrics = c("accuracy", "precision", "recall",
                "sensitivity", "specificity", "f1")
  ))

  value_of <- function(name) result$value[result$metric == name]

  expect_equal(value_of("accuracy"), 0.10)
  expect_equal(value_of("sensitivity"), 1.00)
  expect_equal(value_of("specificity"), 0.00)
  expect_equal(value_of("recall"), 1.00)
  expect_equal(value_of("precision"), 0.10)
  expect_equal(value_of("f1"), 2 * 0.1 * 1 / (0.1 + 1), tolerance = 1e-8)
})

test_that("binary metrics agree with a hand-built confusion matrix", {
  # TP = 30, FN = 20, TN = 35, FP = 15
  truth <- factor(
    c(rep("no", 50), rep("yes", 50)),
    levels = c("no", "yes")
  )
  pred <- factor(
    c(rep("no", 35), rep("yes", 15), rep("yes", 30), rep("no", 20)),
    levels = c("no", "yes")
  )

  result <- tl_calc_classification_metrics(
    truth, pred,
    metrics = c("accuracy", "precision", "recall", "specificity", "f1")
  )
  value_of <- function(name) result$value[result$metric == name]

  tp <- 30
  fn <- 20
  tn <- 35
  fp <- 15
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)

  expect_equal(value_of("accuracy"), (tp + tn) / 100)
  expect_equal(value_of("recall"), recall)
  expect_equal(value_of("specificity"), tn / (tn + fp))
  expect_equal(value_of("precision"), precision)
  expect_equal(
    value_of("f1"),
    2 * precision * recall / (precision + recall)
  )
})

test_that("the positive class matches the one AUC and thresholds use", {
  # A probability column named for the second level is what the AUC
  # branch reads, so a perfectly ranked score must give AUC 1, not 0.
  truth <- factor(
    c(rep("neg", 20), rep("pos", 20)),
    levels = c("neg", "pos")
  )
  pos_prob <- c(seq(0.01, 0.40, length.out = 20),
                seq(0.60, 0.99, length.out = 20))
  probs <- data.frame(neg = 1 - pos_prob, pos = pos_prob)

  result <- tl_calc_classification_metrics(
    truth,
    factor(ifelse(pos_prob > 0.5, "pos", "neg"), levels = levels(truth)),
    predicted_probs = probs,
    metrics = c("auc", "sensitivity", "specificity")
  )
  value_of <- function(name) result$value[result$metric == name]

  expect_equal(value_of("auc"), 1)
  expect_equal(value_of("sensitivity"), 1)
  expect_equal(value_of("specificity"), 1)
})

test_that("threshold metrics move in the right direction", {
  # Raising the threshold on a ranked score can only make the classifier
  # more conservative: precision rises, recall falls. Scoring against the
  # wrong class inverts both.
  truth <- factor(
    c(rep("neg", 50), rep("pos", 50)),
    levels = c("neg", "pos")
  )
  set.seed(11)
  pos_prob <- c(stats::runif(50, 0, 0.6), stats::runif(50, 0.4, 1))

  result <- tl_evaluate_thresholds(
    actuals = truth, probs = pos_prob,
    thresholds = c(0.3, 0.7), pos_class = "pos"
  )
  value_of <- function(name) result$value[result$metric == name]

  expect_gte(value_of("precision_t0.7"), value_of("precision_t0.3"))
  expect_lte(value_of("recall_t0.7"), value_of("recall_t0.3"))
})

test_that("multiclass metrics are unaffected by the event level", {
  truth <- iris$Species
  pred <- iris$Species
  pred[1:10] <- "versicolor"

  result <- tl_calc_classification_metrics(
    truth, pred,
    metrics = c("accuracy", "precision", "recall", "f1")
  )
  value_of <- function(name) result$value[result$metric == name]

  expect_equal(value_of("accuracy"), 140 / 150)
  # Macro-averaged recall: setosa 40/50, versicolor 50/50, virginica 50/50
  expect_equal(value_of("recall"), mean(c(40 / 50, 1, 1)))
  expect_false(is.na(value_of("precision")))
})

# ---- cross-validation covers every row -------------------------------

test_that("tl_cv assigns every row to exactly one assessment fold", {
  # Sizing folds by floor(n / folds) and slicing forward left the last
  # n %% folds rows in no test set at all -- 30 of 32 rows scored on
  # mtcars at folds = 5.
  for (n in c(32L, 41L, 100L)) {
    for (folds in c(3L, 5L, 7L)) {
      fold_id <- rep(seq_len(folds), length.out = n)

      expect_equal(length(fold_id), n)
      expect_setequal(unique(fold_id), seq_len(folds))
      # Folds differ in size by at most one row
      expect_lte(diff(range(table(fold_id))), 1)
    }
  }
})

test_that("tl_cv scores all observations", {
  set.seed(99)
  cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear",
              folds = 5, metrics = "rmse")

  expect_length(cv$folds, 5)
  expect_true(all(c("metric", "mean", "sd") %in% names(cv$summary)))
  expect_false(any(is.na(cv$summary$mean)))
})

test_that("tl_cv rejects fold counts it cannot honour", {
  expect_error(
    tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 1),
    "must be between 2"
  )
  expect_error(
    tl_cv(mtcars[1:4, ], mpg ~ wt, method = "linear", folds = 10),
    "must be between 2"
  )
})

# ---- cross-validation on folds too small for a metric ----------------

test_that("tl_cv explains a metric that no fold could compute", {
  # folds = nrow(data) is leave-one-out, so every test fold holds one
  # observation and rsq -- which needs variation in the truth -- is
  # undefined. mean() over nothing then put a bare NaN in the summary,
  # which reads as a malfunction rather than as a property of the request.
  set.seed(1)
  n <- 10
  d <- data.frame(x = stats::rnorm(n))
  d$y <- d$x * 2 + stats::rnorm(n, sd = 0.2)

  expect_message(
    suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = n)),
    "rsq could not be computed for any fold"
  )
  expect_message(
    suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = n)),
    "smallest fold holds 1 observation"
  )

  # rmse and mae are defined for a single observation and still are
  cv <- suppressWarnings(suppressMessages(
    tl_cv(d, y ~ x, method = "linear", folds = n)
  ))
  defined <- cv$summary[cv$summary$metric %in% c("rmse", "mae"), ]
  expect_equal(nrow(defined), 2L)
  expect_true(all(is.finite(defined$mean)))
})

test_that("a fold count that leaves room for every metric says nothing", {
  set.seed(1)
  n <- 40
  d <- data.frame(x = stats::rnorm(n))
  d$y <- d$x * 2 + stats::rnorm(n, sd = 0.2)

  expect_no_message(
    suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = 5))
  )
  cv <- suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = 5))
  expect_true(all(is.finite(cv$summary$mean)))
})

test_that("tl_cv does not repeat tl_model's notes once per fold", {
  # The response note is about the data, not the fold, and fired k times.
  set.seed(1)
  n <- 40
  d <- data.frame(x = stats::rnorm(n))
  d$y <- as.numeric(d$x > 0)

  emitted <- character()
  withCallingHandlers(
    suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = 5)),
    message = function(m) {
      emitted <<- c(emitted, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_false(any(grepl("Treating as regression", emitted)))
})
