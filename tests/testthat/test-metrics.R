make_binary_data <- function(seed = 101, n = 40) {
  set.seed(seed)
  data.frame(
    x = c(stats::rnorm(n, -1), stats::rnorm(n, 1)),
    y = factor(rep(c("a", "b"), each = n))
  )
}

test_that("tl_evaluate scores classification on class labels", {
  d <- make_binary_data()
  model <- tl_model(d, y ~ x, method = "logistic")

  ev <- tl_evaluate(model)
  manual <- mean(predict(model, new_data = d, type = "class")$.pred == d$y)

  expect_equal(ev$metric, "accuracy")
  expect_equal(ev$value, manual)
  # The default predict type for logistic returns probabilities; comparing
  # those to class labels scores every model at zero
  expect_gt(ev$value, 0.5)
})

test_that("tl_evaluate honours the metrics argument for classification", {
  d <- make_binary_data()
  model <- tl_model(d, y ~ x, method = "logistic")

  requested <- c("accuracy", "precision", "recall", "f1", "auc")
  ev <- tl_evaluate(model, metrics = requested)

  expect_setequal(ev$metric, requested)
  expect_false(any(is.na(ev$value)))
})

test_that("tl_evaluate honours the metrics argument for regression", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_equal(tl_evaluate(model)$metric, c("rmse", "mae", "rsq"))

  requested <- c("rmse", "mse", "mae", "mape", "rsq")
  ev <- tl_evaluate(model, metrics = requested)
  expect_setequal(ev$metric, requested)

  value_of <- function(m) ev$value[ev$metric == m]
  expect_equal(value_of("rmse"), sqrt(value_of("mse")))
  expect_equal(
    value_of("rsq"),
    summary(stats::lm(mpg ~ wt + hp, data = mtcars))$r.squared
  )
})

test_that("tl_evaluate errors when the response is missing from new data", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_error(
    tl_evaluate(model, new_data = mtcars[, c("wt", "hp")]),
    "not found in the evaluation data"
  )
})

# ---- metric names ------------------------------------------------------

test_that("tl_evaluate refuses a metric it does not compute", {
  # Each of these returned a 0-row tibble with no message
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_error(
    tl_evaluate(model, metrics = "RMSE"),
    "Unknown regression metric\\(s\\) in 'metrics': \"RMSE\""
  )
  # A classification metric on a regression model, with the list to use
  expect_error(
    tl_evaluate(model, metrics = c("rmse", "accuracy")),
    "\"accuracy\"\\. Available: rmse, mse, mae, mape, rsq\\."
  )
  expect_error(
    tl_evaluate(model, metrics = character(0)),
    "'metrics' is empty"
  )

  classifier <- tl_model(make_binary_data(), y ~ x, method = "logistic")
  expect_error(
    tl_evaluate(classifier, metrics = "rmse"),
    paste0(
      "Unknown classification metric\\(s\\) in 'metrics': \"rmse\"\\. ",
      "Available: accuracy, precision, recall, sensitivity, specificity, ",
      "f1, auc, pr_auc\\."
    )
  )
  expect_error(
    tl_calc_classification_metrics(
      factor(c("a", "b")), factor(c("a", "b")), metrics = "acc"
    ),
    "Unknown classification metric\\(s\\) in 'metrics': \"acc\""
  )

  # Every documented name is still accepted, and each comes back once
  ev <- tl_evaluate(classifier, metrics = "sensitivity")
  expect_equal(ev$metric, "sensitivity")
})

# ---- observed classes are read against the model's -------------------

# A test split of a subsetted frame still declares the level the subset
# removed. tl_model() drops it from the training response, so the scored
# rows and the predictions disagreed about the classes.
undropped <- iris[iris$Species != "setosa", ]
undropped_train <- undropped[c(1:35, 51:85), ]
undropped_test <- undropped[c(36:50, 86:100), ]

test_that("tl_evaluate scores a split that declares a level it lost", {
  # yardstick stopped with "truth and estimate levels must be equivalent"
  expect_equal(nlevels(undropped_test$Species), 3L)
  model <- suppressWarnings(
    tl_model(undropped_train, Species ~ ., method = "logistic")
  )

  ev <- tl_evaluate(
    model, undropped_test,
    metrics = c("accuracy", "precision", "recall", "auc", "pr_auc")
  )
  value_of <- function(name) ev$value[ev$metric == name]

  truth <- as.character(undropped_test$Species)
  pred <- as.character(predict(model, undropped_test, type = "class")$.pred)
  pos_prob <- predict(model, undropped_test, type = "prob")$virginica
  # virginica, the model's second class, is the positive class
  tp <- sum(pred == "virginica" & truth == "virginica")

  expect_equal(value_of("accuracy"), mean(pred == truth))
  expect_equal(value_of("precision"), tp / sum(pred == "virginica"))
  expect_equal(value_of("recall"), tp / sum(truth == "virginica"))
  expect_equal(
    value_of("auc"),
    yardstick::roc_auc_vec(
      factor(truth), pos_prob, event_level = "second"
    )
  )
  expect_equal(
    value_of("pr_auc"),
    yardstick::pr_auc_vec(factor(truth), pos_prob, event_level = "second")
  )
})

test_that("a reordered factor does not move the positive class", {
  model <- suppressWarnings(
    tl_model(undropped_train, Species ~ ., method = "logistic")
  )
  wanted <- c("accuracy", "precision", "recall", "specificity", "pr_auc")

  reordered <- undropped_test
  reordered$Species <- factor(
    reordered$Species, levels = c("virginica", "versicolor", "setosa")
  )

  expect_equal(
    tl_evaluate(model, reordered, metrics = wanted),
    tl_evaluate(model, droplevels(undropped_test), metrics = wanted)
  )
})

test_that("rows of a class the model never saw are left out, with a warning", {
  model <- tl_model(undropped_train, Species ~ ., method = "tree")
  scored <- rbind(undropped_test, iris[1:3, ])

  expect_warning(
    ev <- tl_evaluate(model, scored, metrics = "accuracy"),
    "3 row\\(s\\) belong to a class the model was not trained on \\(setosa\\)"
  )
  kept <- tl_evaluate(model, undropped_test, metrics = "accuracy")
  expect_equal(ev$value, kept$value)
})

# ---- the response the formula fits ------------------------------------

test_that("tl_evaluate scores a transformed response on its own scale", {
  # log(mpg) ~ wt + hp was scored against raw mpg: rmse 18.1, rsq -8.26,
  # for a fit whose residual rmse is 0.106
  model <- tl_model(mtcars, log(mpg) ~ wt + hp, method = "linear")
  reference <- stats::lm(log(mpg) ~ wt + hp, data = mtcars)

  ev <- tl_evaluate(model, metrics = c("rmse", "rsq"))
  expect_equal(
    ev$value[ev$metric == "rmse"],
    sqrt(mean(stats::residuals(reference)^2))
  )
  expect_equal(
    ev$value[ev$metric == "rsq"],
    summary(reference)$r.squared
  )

  # The same on new rows, and for a method that is not lm
  tree <- tl_model(mtcars[1:24, ], log(mpg) ~ wt + hp, method = "tree")
  held_out <- mtcars[25:32, ]
  scored <- tl_evaluate(tree, held_out, metrics = "mae")
  pred <- predict(tree, held_out)$.pred
  expect_equal(scored$value, mean(abs(pred - log(held_out$mpg))))
})

# ---- incomplete rows ---------------------------------------------------

test_that("tl_evaluate drops incomplete rows for every classification metric", {
  # An NA predictor stopped auc with "'predictions' contains NA", and an
  # NA response with "Not enough distinct predictions"; accuracy alone
  # dropped the same rows without a word
  d <- droplevels(undropped)
  model <- suppressWarnings(tl_model(d, Species ~ ., method = "logistic"))
  wanted <- c("accuracy", "f1", "auc", "pr_auc")

  holed <- d
  holed$Sepal.Width[1:2] <- NA
  holed$Species[60] <- NA

  expect_equal(
    tl_evaluate(model, holed, metrics = wanted),
    tl_evaluate(model, d[-c(1, 2, 60), ], metrics = wanted)
  )
})

test_that("tl_evaluate refuses when no row can be scored", {
  # Zero rows, or rows that are all incomplete, came back as NaN accuracy,
  # rmse and mae without a message
  regression <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  classifier <- tl_model(iris, Species ~ ., method = "tree")

  expect_error(
    tl_evaluate(regression, mtcars[0, ]),
    "'new_data' has no rows, so there is nothing to score",
    class = "tidylearn_no_scored_rows"
  )
  expect_error(
    tl_evaluate(classifier, iris[0, ], metrics = c("accuracy", "auc")),
    "'new_data' has no rows, so there is nothing to score",
    class = "tidylearn_no_scored_rows"
  )

  no_predictor <- mtcars[1:5, ]
  no_predictor$wt <- NA_real_
  expect_error(
    tl_evaluate(regression, no_predictor),
    paste0(
      "None of the 5 rows of the evaluation data can be scored: 5 have no ",
      "prediction, which happens wherever a predictor is missing\\."
    ),
    class = "tidylearn_no_scored_rows"
  )

  no_response <- iris[c(1, 51, 101), ]
  no_response$Species[] <- NA
  expect_error(
    tl_evaluate(classifier, no_response),
    paste0(
      "None of the 3 rows of the evaluation data can be scored: 3 are ",
      "missing the response\\."
    ),
    class = "tidylearn_no_scored_rows"
  )

  binary <- tl_model(droplevels(iris[51:150, ]), Species ~ ., method = "tree")
  expect_warning(
    expect_error(
      tl_evaluate(binary, iris[1:4, ]),
      paste0(
        "None of the 4 rows of the evaluation data can be scored: 4 belong ",
        "to a class the model was not trained on\\."
      ),
      class = "tidylearn_no_scored_rows"
    ),
    "4 row\\(s\\) belong to a class the model was not trained on"
  )

  # A single row that can be scored is enough
  partial <- mtcars[1:5, ]
  partial$wt[1:4] <- NA
  ev <- tl_evaluate(regression, partial, metrics = "mae")
  expect_equal(
    ev$value,
    unname(abs(predict(regression, partial[5, ])$.pred - partial$mpg[5]))
  )
})

test_that("tl_calc_classification_metrics refuses when nothing can be scored", {
  # accuracy came back NaN without a message
  two_classes <- c("a", "b")
  expect_error(
    tl_calc_classification_metrics(
      factor(two_classes), factor(c(NA, NA), levels = two_classes)
    ),
    paste0(
      "None of the 2 observations can be scored: each is missing its ",
      "observed class, its prediction or a probability\\."
    ),
    class = "tidylearn_no_scored_rows"
  )
  expect_error(
    tl_calc_classification_metrics(
      factor(character(0), levels = two_classes),
      factor(character(0), levels = two_classes)
    ),
    "'actuals' is empty, so there is nothing to score",
    class = "tidylearn_no_scored_rows"
  )
})

# ---- models fitted on engineered features ------------------------------

test_that("a model fitted on PCA features can be evaluated on its own data", {
  # tl_auto_ml()'s pca_* candidates store the PCA scores they were fitted
  # on. tl_evaluate() passed those back to predict(), which projected them
  # a second time and failed: "PCA was fitted on 4 column(s) ... but
  # new_data is missing". summary() and the training-score fallback on
  # the leaderboard went through the same call.
  reduced <- tl_reduce_dimensions(
    iris, response = "Species", method = "pca", n_components = 2
  )
  model <- tl_model(reduced$data, Species ~ PC1 + PC2, method = "tree")
  model$feature_transform <- list(
    kind = "pca",
    reduction_model = reduced$reduction_model,
    response = "Species"
  )

  ev <- tl_evaluate(model)
  stored_pred <- predict(model)$.pred
  expect_equal(ev$value, mean(stored_pred == model$data$Species))

  # Raw data is still projected once, onto the same scores
  expect_equal(tl_evaluate(model, iris)$value, ev$value)
  expect_output(summary(model), "Training Performance")
})

# ---- the shapes tl_calc_classification_metrics() takes and returns -----

test_that("predicted_probs must have a column per class", {
  truth <- factor(c("a", "b", "a", "b"))
  pred <- factor(c("a", "b", "b", "b"))
  probs <- data.frame(a = c(0.9, 0.2, 0.4, 0.3), b = c(0.1, 0.8, 0.6, 0.7))

  # A vector failed with "argument is of length zero"
  expect_error(
    tl_calc_classification_metrics(truth, pred, probs$b, metrics = "auc"),
    "'predicted_probs' must be a data frame with a probability column per"
  )
  expect_error(
    tl_calc_classification_metrics(truth, pred, probs["a"], metrics = "auc"),
    "Missing: \"b\""
  )

  # A matrix failed with "subscript out of bounds"; it is read by its
  # column names, like the data frame predict() returns
  from_frame <- tl_calc_classification_metrics(
    truth, pred, probs, metrics = c("auc", "pr_auc")
  )
  expect_equal(
    tl_calc_classification_metrics(
      truth, pred, as.matrix(probs), metrics = c("auc", "pr_auc")
    ),
    from_frame
  )
})

test_that("a class missing from a hand-built prediction factor still counts", {
  # A factor built from the predictions alone lacks any class the model
  # did not predict, and yardstick refused it: "truth and estimate levels
  # must be equivalent". Those rows are errors the model made.
  truth <- c("neg", "pos", "pos", "neg", "pos")
  all_pos <- factor(rep("pos", 5))
  res <- tl_calc_classification_metrics(
    truth, all_pos, metrics = c("accuracy", "precision", "recall")
  )
  value_of <- function(name) res$value[res$metric == name]
  # TP = 3, FP = 2, FN = 0, with "pos" the second class
  expect_equal(value_of("accuracy"), 3 / 5)
  expect_equal(value_of("precision"), 3 / 5)
  expect_equal(value_of("recall"), 1)

  # Rows 3 and 4 are class "c", which the predictions never name
  three_classes <- c("a", "b", "c", "c", "a", "b")
  two_levels <- factor(c("a", "b", "a", "b", "a", "b"))
  res <- tl_calc_classification_metrics(
    three_classes, two_levels, metrics = "accuracy"
  )
  expect_equal(res$value, 4 / 6)
})

test_that("auc asked for by name without probabilities says it is left out", {
  truth <- factor(c("a", "b", "a", "b"))
  pred <- factor(c("a", "b", "b", "b"))

  # It was dropped from the result without a word
  expect_warning(
    res <- tl_calc_classification_metrics(
      truth, pred, metrics = c("accuracy", "auc")
    ),
    "auc needs 'predicted_probs' and is left out of the result"
  )
  expect_equal(res$metric, "accuracy")

  # The default metrics include auc, and leaving it out there is the
  # documented behaviour
  expect_no_warning(res <- tl_calc_classification_metrics(truth, pred))
  expect_equal(res$metric, c("accuracy", "precision", "recall", "f1"))
})

test_that("thresholds add rows and a threshold column, for binary tasks", {
  truth <- factor(c("a", "b", "a", "b", "b"))
  probs <- data.frame(
    a = c(0.9, 0.2, 0.4, 0.3, 0.6), b = c(0.1, 0.8, 0.6, 0.7, 0.4)
  )
  pred <- factor(ifelse(probs$b > 0.5, "b", "a"), levels = c("a", "b"))

  res <- tl_calc_classification_metrics(
    truth, pred, probs, metrics = "accuracy", thresholds = c(0.5, 0.65)
  )
  expect_equal(res$metric[1], "accuracy")
  expect_true(is.na(res$threshold[1]))
  expect_equal(res$threshold[-1], rep(c(0.5, 0.65), each = 6))
  # At 0.65 only rows 2 and 4 are called "b": 4 of 5 right
  expect_equal(res$value[res$metric == "accuracy_t0.65"], 4 / 5)

  # They were ignored without a word for more than two classes, and
  # without probabilities
  iris_probs <- predict(tl_model(iris, Species ~ ., method = "tree"),
                        type = "prob")
  expect_warning(
    tl_calc_classification_metrics(
      iris$Species, iris$Species, iris_probs,
      metrics = "accuracy", thresholds = 0.5
    ),
    "'thresholds' apply to binary classification only"
  )
  expect_warning(
    tl_calc_classification_metrics(
      truth, pred, metrics = "accuracy", thresholds = 0.5
    ),
    "'thresholds' need 'predicted_probs'"
  )
})

test_that("tl_cv forwards metrics to tl_evaluate", {
  set.seed(102)
  cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3,
              metrics = c("rmse", "mape"))

  expect_setequal(cv$summary$metric, c("rmse", "mape"))
  expect_false(any(is.na(cv$summary$mean)))
})

test_that("tl_calc_regression_metrics handles degenerate inputs", {
  # Constant actuals leave no variance to explain
  res <- tl_calc_regression_metrics(rep(2, 5), rep(2, 5), metrics = "rsq")
  expect_true(is.na(res$value))

  # Zero actuals are excluded from MAPE rather than producing Inf
  res <- tl_calc_regression_metrics(c(0, 2, 4), c(0, 3, 5), metrics = "mape")
  expect_equal(res$value, mean(c(0.5, 0.25)) * 100)

  # Missing values are dropped pairwise
  res <- tl_calc_regression_metrics(c(1, NA, 3), c(1, 5, 3), metrics = "mae")
  expect_equal(res$value, 0)
})
