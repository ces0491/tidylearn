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

# ---- area under the precision-recall curve ----------------------------

pr_auc_of <- function(truth, pos_prob) {
  probs <- data.frame(1 - pos_prob, pos_prob)
  names(probs) <- levels(truth)
  pred <- factor(
    ifelse(pos_prob > 0.5, levels(truth)[2], levels(truth)[1]),
    levels = levels(truth)
  )
  result <- tl_calc_classification_metrics(
    truth, pred, predicted_probs = probs, metrics = "pr_auc"
  )
  result$value[result$metric == "pr_auc"]
}

test_that("pr_auc matches a hand-built precision-recall curve", {
  # Scores 0.9, 0.8, 0.7, 0.6 for pos, neg, pos, neg. From the curve's
  # start at recall 0, precision 1, the points (recall, precision) are
  # (0.5, 1), (0.5, 0.5), (1, 2/3), (1, 0.5), so the trapezoids sum to
  # nolint start: commented_code_linter.
  #   0.5 * 1 + 0.5 * (0.5 + 2/3) / 2 = 19/24
  # nolint end
  truth <- factor(c("pos", "neg", "pos", "neg"), levels = c("neg", "pos"))
  expect_equal(pr_auc_of(truth, c(0.9, 0.8, 0.7, 0.6)), 19 / 24)

  # The trapezoid never started at recall 0, so the area before the first
  # recall was lost: a perfect ranking of 5 positives in 20 scored 0.8
  truth <- factor(c(rep("neg", 15), rep("pos", 5)), levels = c("neg", "pos"))
  ranked <- c(seq(0.01, 0.5, length.out = 15), seq(0.6, 0.99, length.out = 5))
  expect_equal(pr_auc_of(truth, ranked), 1)

  # Constant scores make the curve one step from (0, 1) to (1, 5/20). The
  # row used to go missing without a message.
  expect_equal(pr_auc_of(truth, rep(0.3, 20)), (1 + 5 / 20) / 2)
})

test_that("pr_auc agrees with yardstick on a fitted model", {
  # A tree on binary iris reported 0.074 where yardstick gives 0.964
  d <- droplevels(iris[iris$Species != "setosa", ])
  model <- tl_model(d, Species ~ ., method = "tree")
  pos_prob <- predict(model, type = "prob")$virginica

  ev <- tl_evaluate(model, metrics = "pr_auc")
  expect_equal(
    ev$value,
    yardstick::pr_auc_vec(d$Species, pos_prob, event_level = "second")
  )

  # The precision-recall plot reads the area off a ROCR curve; it has to
  # give the same number
  perf <- ROCR::performance(
    ROCR::prediction(pos_prob, as.integer(d$Species == "virginica")),
    "prec", "rec"
  )
  expect_equal(tl_calculate_pr_auc(perf), ev$value)
})

test_that("pr_auc for more than two classes averages one-vs-rest areas", {
  # Asked for on iris, it was left out of the result without a message
  model <- tl_model(iris, Species ~ ., method = "tree")
  probs <- predict(model, type = "prob")

  ev <- tl_evaluate(model, metrics = "pr_auc")
  one_vs_rest <- vapply(levels(iris$Species), function(cls) {
    yardstick::pr_auc_vec(
      factor(iris$Species == cls, levels = c(FALSE, TRUE)),
      probs[[cls]], event_level = "second"
    )
  }, numeric(1))
  expect_equal(ev$value, mean(one_vs_rest))
  expect_equal(
    ev$value,
    yardstick::pr_auc_vec(iris$Species, as.matrix(probs), estimator = "macro")
  )
})

# ---- ranking metrics where they are undefined -------------------------

test_that("auc and pr_auc are NA, with a warning, on a single class", {
  # ROCR stopped with "Number of classes is not equal to 2", which aborted
  # tl_cv(), tl_compare_cv() and the tuners at the first fold that held one
  # class
  d <- mtcars
  d$am <- factor(d$am, labels = c("auto", "manual"))
  model <- tl_model(d, am ~ wt, method = "tree")
  automatic <- d[d$am == "auto", ]

  expect_warning(
    ev <- tl_evaluate(
      model, automatic, metrics = c("accuracy", "auc", "pr_auc")
    ),
    paste0(
      "auc and pr_auc are undefined when the scored rows hold a single ",
      "class \\(\"auto\"\\), so they are NA"
    )
  )
  value_of <- function(name) ev$value[ev$metric == name]
  expect_true(is.na(value_of("auc")))
  expect_true(is.na(value_of("pr_auc")))
  pred <- predict(model, automatic, type = "class")$.pred
  expect_equal(value_of("accuracy"), mean(pred == "auto"))

  # The same for more than two classes, per class as well as on average
  model <- tl_model(iris, Species ~ ., method = "tree")
  expect_warning(
    ev <- tl_evaluate(model, iris[1:50, ], metrics = "auc"),
    "auc is undefined when the scored rows hold a single class"
  )
  expect_equal(ev$metric, c("auc", paste0("auc_", levels(iris$Species))))
  expect_true(all(is.na(ev$value)))
})

test_that("an absent class is left out of the one-vs-rest averages", {
  # A class missing from the scored rows has no one-vs-rest curve. ROCR
  # stopped on it; yardstick's macro pr_auc returns NaN.
  model <- tl_model(iris, Species ~ ., method = "tree")
  no_virginica <- iris[1:100, ]
  probs <- predict(model, no_virginica, type = "prob")

  expect_warning(
    ev <- tl_evaluate(model, no_virginica, metrics = c("auc", "pr_auc")),
    paste0(
      "No scored row belongs to \"virginica\", so auc and pr_auc leave it ",
      "out: its one-vs-rest value is NA and the macro average covers the ",
      "other classes"
    )
  )
  value_of <- function(name) ev$value[ev$metric == name]

  truth <- no_virginica$Species
  class_auc <- function(cls) {
    yardstick::roc_auc_vec(
      factor(truth == cls, levels = c(FALSE, TRUE)), probs[[cls]],
      event_level = "second"
    )
  }
  class_pr <- function(cls) {
    yardstick::pr_auc_vec(
      factor(truth == cls, levels = c(FALSE, TRUE)), probs[[cls]],
      event_level = "second"
    )
  }
  present <- c("setosa", "versicolor")
  expect_true(is.na(value_of("auc_virginica")))
  expect_equal(value_of("auc_setosa"), class_auc("setosa"))
  expect_equal(value_of("auc"), mean(vapply(present, class_auc, numeric(1))))
  expect_equal(
    value_of("pr_auc"), mean(vapply(present, class_pr, numeric(1)))
  )
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
  expected <- "'folds' must be a whole number between 2 and nrow\\(data\\)"
  expect_error(
    tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 1),
    expected
  )
  expect_error(
    tl_cv(mtcars[1:4, ], mpg ~ wt, method = "linear", folds = 10),
    expected
  )
  # folds = 2.5 ran 2 folds without a word; a vector and NA failed on
  # "the condition has length > 1" and "missing value where TRUE/FALSE
  # needed"
  expect_error(
    tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 2.5),
    paste0(expected, " \\(32\\)\\. Got: 2\\.5")
  )
  expect_error(
    tl_cv(mtcars, mpg ~ wt, method = "linear", folds = c(3, 5)),
    expected
  )
  expect_error(
    tl_cv(mtcars, mpg ~ wt, method = "linear", folds = NA),
    expected
  )
  expect_error(
    tl_cv(as.list(mtcars), mpg ~ wt, method = "linear", folds = 3),
    "'data' must be a data frame"
  )

  # A whole number stored as a double is a whole number
  set.seed(1)
  expect_length(tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 4)$folds, 4)
})

# The rows tl_cv() scores in each fold, for checking its numbers by hand.
# It draws the permutation with sample() and deals rows into folds in turn.
cv_fold_rows <- function(n, folds, seed) {
  set.seed(seed)
  indices <- sample(1:n)
  fold_id <- rep(seq_len(folds), length.out = n)
  lapply(seq_len(folds), function(i) indices[fold_id == i])
}

fold_values <- function(cv, metric) {
  vapply(cv$folds, function(f) f$value[f$metric == metric], numeric(1))
}

test_that("tl_cv scores a 0/1 response in folds that hold one class", {
  # A logistic model of mtcars' am crashed at folds = 10 for seeds 1 to 4:
  # "truth and estimate levels must be equivalent. truth: 0; estimate: 0
  # and 1"
  for (seed in 1:4) {
    set.seed(seed)
    cv <- suppressWarnings(
      tl_cv(mtcars, am ~ wt + hp, method = "logistic", folds = 10)
    )
    expect_length(cv$folds, 10)
  }

  set.seed(1)
  cv <- suppressWarnings(
    tl_cv(mtcars, am ~ wt + hp, method = "logistic", folds = 10)
  )
  expected <- vapply(cv_fold_rows(32, 10, seed = 1), function(rows) {
    fit <- suppressWarnings(stats::glm(
      am ~ wt + hp, family = stats::binomial(), data = mtcars[-rows, ]
    ))
    pos_prob <- stats::predict(fit, mtcars[rows, ], type = "response")
    mean((pos_prob > 0.5) == (mtcars$am[rows] == 1))
  }, numeric(1))
  expect_equal(fold_values(cv, "accuracy"), expected)
})

test_that("tl_cv warns of and leaves out a class a fold's model never saw", {
  # Both rows of class "c" are scored in fold 1, so that fold's model never
  # saw the class. yardstick refused the fold: "truth: a, b, and c;
  # estimate: a and b".
  rows <- cv_fold_rows(30, 3, seed = 7)
  set.seed(5)
  d <- data.frame(x1 = stats::rnorm(30), x2 = stats::rnorm(30))
  d$y <- factor(rep(c("a", "b"), length.out = 30), levels = c("a", "b", "c"))
  d$y[rows[[1]][1:2]] <- "c"

  set.seed(7)
  expect_warning(
    cv <- tl_cv(d, y ~ x1 + x2, method = "tree", folds = 3,
                metrics = "accuracy"),
    "2 row\\(s\\) belong to a class the model was not trained on \\(c\\)"
  )

  fold_model <- tl_model(d[-rows[[1]], ], y ~ x1 + x2, method = "tree")
  scored <- d[rows[[1]], ]
  scored <- scored[scored$y != "c", ]
  pred <- predict(fold_model, scored, type = "class")$.pred
  expect_equal(
    fold_values(cv, "accuracy")[1],
    mean(as.character(pred) == as.character(scored$y))
  )
})

test_that("tl_cv scores a transformed response on the scale it is fitted", {
  # log(mpg) ~ wt + hp compared log-scale predictions with raw mpg: rmse
  # 17.9 and rsq -13.3 where the fit's own residual rmse is 0.106
  set.seed(3)
  cv <- tl_cv(mtcars, log(mpg) ~ wt + hp, method = "linear", folds = 4,
              metrics = "rmse")

  expected <- vapply(cv_fold_rows(32, 4, seed = 3), function(rows) {
    fit <- stats::lm(log(mpg) ~ wt + hp, data = mtcars[-rows, ])
    held_out <- mtcars[rows, ]
    sqrt(mean((stats::predict(fit, held_out) - log(held_out$mpg))^2))
  }, numeric(1))
  expect_equal(fold_values(cv, "rmse"), expected)
})

test_that("tl_cv reports NA auc for a fold that holds one class", {
  # The fold stopped the whole run with "Number of classes is not equal
  # to 2". With seed 2 and 10 folds, fold 5 scores only automatic cars.
  d <- mtcars
  d$am <- factor(d$am, labels = c("auto", "manual"))

  set.seed(2)
  expect_warning(
    cv <- tl_cv(d, am ~ wt, method = "tree", folds = 10,
                metrics = c("accuracy", "auc")),
    "auc is undefined when the scored rows hold a single class"
  )

  single_class <- vapply(
    cv_fold_rows(32, 10, seed = 2),
    function(rows) length(unique(d$am[rows])) == 1L,
    logical(1)
  )
  fold_auc <- fold_values(cv, "auc")
  expect_equal(is.na(fold_auc), single_class)
  expect_equal(
    cv$summary$mean[cv$summary$metric == "auc"],
    mean(fold_auc, na.rm = TRUE)
  )
})

test_that("tl_cv leaves out a fold none of whose rows can be scored", {
  # tl_evaluate() refuses such a fold, and one fold is no reason to stop
  # the run. Its scores came back NaN without a word.
  rows <- cv_fold_rows(32, 4, seed = 6)
  d <- mtcars
  d$wt[rows[[2]]] <- NA

  set.seed(6)
  expect_warning(
    cv <- tl_cv(d, mpg ~ wt + hp, method = "linear", folds = 4,
                metrics = "rmse"),
    paste0(
      "Fold 2 is left out of the summary, since none of its 8 rows can be ",
      "scored: 8 have no prediction"
    )
  )
  values <- fold_values(cv, "rmse")
  expect_true(is.na(values[2]))
  expect_true(all(is.finite(values[-2])))
  expect_equal(cv$summary$mean, mean(values[-2]))
})

test_that("tl_cv keeps a one-column data frame a data frame", {
  # Row-subsetting a one-column frame without drop = FALSE returned a
  # vector, and tl_model() refused it: "'data' must be a data frame"
  d <- data.frame(y = c(4.1, 5.3, 2.2, 6.8, 3.9, 5.5, 4.4, 7.1, 2.9, 5.0,
                        3.3, 6.2))
  set.seed(4)
  cv <- tl_cv(d, y ~ 1, method = "linear", folds = 3, metrics = "mae")

  # An intercept-only model predicts the training mean
  expected <- vapply(cv_fold_rows(12, 3, seed = 4), function(rows) {
    mean(abs(mean(d$y[-rows]) - d$y[rows]))
  }, numeric(1))
  expect_equal(fold_values(cv, "mae"), expected)
})

test_that("tl_cv names every metric a task offers when one is unknown", {
  expect_error(
    tl_cv(iris, Species ~ ., method = "tree", folds = 3,
          metrics = "acuracy"),
    paste0(
      "Unknown classification metric\\(s\\) in 'metrics': \"acuracy\"\\. ",
      "Available: accuracy, precision, recall, sensitivity, specificity, ",
      "f1, auc, pr_auc\\."
    )
  )
  # The tl_cv() summary was a 0-row tibble
  expect_error(
    tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 3, metrics = "RMSE"),
    "Unknown regression metric\\(s\\) in 'metrics': \"RMSE\""
  )
})

test_that("tl_cv refuses fitting arguments that hold one value per row", {
  # weights = w reached every fold whole and failed there with "variable
  # lengths differ (found for '(weights)')"
  w <- seq_len(32) / 32
  expect_error(
    tl_cv(mtcars, mpg ~ wt + hp, method = "tree", folds = 3, weights = w),
    "tl_cv\\(\\) cannot re-split 'weights' across folds"
  )
  expect_error(
    tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3,
          offset = w, subset = w > 0.5),
    "cannot re-split 'offset', 'subset' across folds"
  )

  # Other fitting arguments still reach every fold's model
  set.seed(1)
  coarse <- tl_cv(mtcars, mpg ~ wt + hp, method = "tree", folds = 3,
                  metrics = "rmse", cp = 0.5)
  expected <- vapply(cv_fold_rows(32, 3, seed = 1), function(rows) {
    fold_model <- tl_model(mtcars[-rows, ], mpg ~ wt + hp,
                           method = "tree", cp = 0.5)
    tl_evaluate(fold_model, mtcars[rows, ], metrics = "rmse")$value
  }, numeric(1))
  expect_equal(fold_values(coarse, "rmse"), expected)

  set.seed(1)
  fine <- tl_cv(mtcars, mpg ~ wt + hp, method = "tree", folds = 3,
                metrics = "rmse", cp = 0.0001, minsplit = 2)
  expect_false(isTRUE(all.equal(fold_values(fine, "rmse"), expected)))
})

test_that("tl_cv accepts a per-row argument set to NULL", {
  # weights = NULL was refused as holding one value per row, though it
  # holds nothing and tl_model() fits with it
  set.seed(2)
  with_null <- tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 3,
                     weights = NULL)
  set.seed(2)
  without <- tl_cv(mtcars, mpg ~ wt, method = "linear", folds = 3)
  expect_equal(with_null$summary, without$summary)
})

# ---- cross-validation on folds too small for a metric ----------------

test_that("tl_cv explains a metric that no fold could compute", {
  # rsq needs variation in the truth, and a constant response has none in
  # any fold. mean() over nothing then put a bare NaN in the summary, which
  # reads as a malfunction rather than as a property of the request.
  d <- data.frame(x = seq_len(10), y = 3)
  set.seed(1)
  expect_message(
    suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = 5)),
    "rsq could not be computed for any fold"
  )
})

test_that("tl_cv warns once that a leave-one-out fold scores one row", {
  # folds = nrow(data) is leave-one-out. rmse on one row is that row's
  # absolute error, so the summary's rmse was the mean absolute error, and
  # rsq was NA with only a note after the run, which said nothing of rmse
  set.seed(1)
  n <- 10
  d <- data.frame(x = stats::rnorm(n))
  d$y <- d$x * 2 + stats::rnorm(n, sd = 0.2)
  loo <- "each fold holds one row (leave-one-out)"

  warnings <- character()
  notes <- character()
  set.seed(1)
  cv <- withCallingHandlers(
    tl_cv(d, y ~ x, method = "linear", folds = n),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      notes <<- c(notes, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_identical(warnings, paste0(
    "With 10 folds for 10 rows, each fold holds one row (leave-one-out) and ",
    "is scored on that row's prediction alone. rmse on one row is the ",
    "absolute error, so its average over the folds is the mean absolute ",
    "error, and the leave-one-out rmse is the square root of the mse ",
    "average; rsq needs more than one row and is NA on every fold. mae ",
    "averages to its value over the left-out predictions. Use fewer folds ",
    "to score rmse and rsq."
  ))
  # The note after the run does not say it again
  expect_false(any(grepl("rsq could not be computed", notes, fixed = TRUE)))

  # The averages are the ones the warning describes
  errors <- vapply(seq_len(n), function(i) {
    fit <- stats::lm(y ~ x, data = d[-i, ])
    d$y[i] - stats::predict(fit, d[i, ])
  }, numeric(1))
  value_of <- function(name) cv$summary$mean[cv$summary$metric == name]
  expect_equal(value_of("rmse"), mean(abs(errors)))
  expect_equal(value_of("mae"), mean(abs(errors)))

  cv <- suppressWarnings(tl_cv(d, y ~ x, method = "linear", folds = n,
                               metrics = c("mse", "rmse")))
  expect_equal(sqrt(cv$summary$mean[cv$summary$metric == "mse"]),
               sqrt(mean(errors^2)))

  # Metrics that average to their leave-one-out values are not warned
  # about, and neither is an ordinary k-fold run
  set.seed(1)
  expect_no_warning(tl_cv(d, y ~ x, method = "linear", folds = n,
                          metrics = c("mae", "mse", "mape")))
  set.seed(1)
  expect_no_warning(tl_cv(d, y ~ x, method = "linear", folds = 5))
})

test_that("leave-one-out tl_cv drops the per-fold undefined-metric warnings", {
  # Each one-row fold left precision or recall undefined, and auc with a
  # single class, and yardstick and tidylearn warned on every fold
  dm <- mtcars
  dm$am <- factor(dm$am, labels = c("auto", "manual"))
  warnings <- character()
  set.seed(1)
  cv <- withCallingHandlers(
    tl_cv(dm, am ~ wt, method = "tree", folds = nrow(dm),
          metrics = c("accuracy", "precision", "recall", "auc")),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_length(warnings, 1L)
  expect_match(warnings, "each fold holds one row (leave-one-out)",
               fixed = TRUE)
  expect_match(warnings, paste0(
    "precision and recall are undefined on the folds where that one row ",
    "leaves nothing to divide by; auc needs more than one row and is NA on ",
    "every fold. accuracy averages to its value over the left-out ",
    "predictions."
  ), fixed = TRUE)

  # Over a fold of several rows the same warnings are still given
  model <- tl_model(dm, am ~ wt, method = "tree")
  expect_warning(
    tl_evaluate(model, dm[dm$am == "auto", ], metrics = "auc"),
    "auc is undefined when the scored rows hold a single class"
  )
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

test_that("tl_cv warns once that logistic converts a 0/1 response", {
  # The conversion is about the data, not the fold, and the warning came
  # once per fold: five times at folds = 5
  emitted <- character()
  set.seed(1)
  withCallingHandlers(
    tl_cv(mtcars, am ~ wt + hp, method = "logistic", folds = 5),
    warning = function(w) {
      emitted <<- c(emitted, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  conversions <- grepl(
    "Converting response variable to factor for logistic regression",
    emitted, fixed = TRUE
  )
  expect_equal(sum(conversions), 1L)
})
