make_regression_pipeline <- function(data = mtcars) {
  tl_pipeline(
    data, mpg ~ wt + hp,
    models = list(linear = list(method = "linear")),
    evaluation = list(
      metrics = "rmse", validation = "cv",
      cv_folds = 3, best_metric = "rmse"
    )
  )
}

test_that("tl_run_pipeline records preprocessing statistics on the raw scale", {
  set.seed(201)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)
  stats_learned <- res$results$preprocessing_stats

  expect_equal(stats_learned$center$wt, mean(mtcars$wt))
  expect_equal(stats_learned$scale$wt, stats::sd(mtcars$wt))
  expect_equal(stats_learned$medians$wt, stats::median(mtcars$wt))

  # The response is never standardized
  expect_null(stats_learned$center$mpg)

  # The stored training data is standardized, so deriving the centre and
  # scale from it would give 0 and 1
  expect_equal(mean(res$results$processed_data$wt), 0)
  expect_equal(stats::sd(res$results$processed_data$wt), 1)
})

test_that("tl_predict_pipeline standardizes new data with training stats", {
  set.seed(202)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)

  preds <- tl_predict_pipeline(res, new_data = mtcars)
  in_sample <- predict(
    res$results$best_model,
    new_data = res$results$processed_data
  )$.pred

  expect_equal(preds$.pred, in_sample)

  # Predictions must land on the response scale, not hundreds of units away
  expect_true(all(abs(preds$.pred - mtcars$mpg) < 10))
})

test_that("columns the formula transforms are left on their own scale", {
  # log(hp) was evaluated on the standardised column, so every car with
  # below-average hp gave NaN: the final lm kept 15 of 32 rows and
  # tl_predict_pipeline() returned NaN for 5 of the first 6 cars
  set.seed(1)
  expect_no_warning(run <- tl_run_pipeline(
    tl_pipeline(
      mtcars, mpg ~ log(hp) + wt,
      models = list(lin = list(method = "linear")),
      evaluation = list(cv_folds = 4)
    ),
    verbose = FALSE
  ))
  best <- tl_get_best_model(run)
  direct <- lm(mpg ~ log(hp) + wt, data = mtcars)

  expect_equal(stats::nobs(best$fit), nrow(mtcars))
  expect_equal(coef(best$fit)[["log(hp)"]], coef(direct)[["log(hp)"]])
  expect_equal(tl_predict_pipeline(run, mtcars)$.pred, fitted(direct))

  # wt enters as a plain term, so it is still standardised; hp is not
  expect_equal(run$results$preprocessing_stats$center$wt, mean(mtcars$wt))
  expect_null(run$results$preprocessing_stats$center$hp)

  # The offset was computed from standardised hp, which put the
  # predictions up to 8.5 mpg from lm()'s
  set.seed(1)
  offset_run <- tl_run_pipeline(
    tl_pipeline(
      mtcars, mpg ~ wt + offset(0.05 * hp),
      models = list(lin = list(method = "linear")),
      evaluation = list(cv_folds = 4)
    ),
    verbose = FALSE
  )
  direct_offset <- lm(mpg ~ wt + offset(0.05 * hp), data = mtcars)
  expect_equal(
    tl_predict_pipeline(offset_run, mtcars)$.pred,
    predict(direct_offset, mtcars)
  )
})

test_that("standardising leaves every formula shape's model unchanged", {
  # Centring a plain column changes the model when nothing absorbs the
  # shift: a formula without an intercept, or an interaction whose
  # lower-order terms are missing. These predicted differently from lm().
  shapes <- list(
    mpg ~ log(hp) + wt - 1,
    mpg ~ log(hp) + wt:qsec,
    mpg ~ wt - 1,
    mpg ~ wt * qsec
  )
  for (shape in shapes) {
    set.seed(217)
    run <- tl_run_pipeline(
      tl_pipeline(
        mtcars, shape,
        models = list(lin = list(method = "linear")),
        evaluation = list(metrics = "rmse", best_metric = "rmse",
                          cv_folds = 3)
      ),
      verbose = FALSE
    )
    expect_equal(tl_predict_pipeline(run, mtcars)$.pred,
                 fitted(lm(shape, data = mtcars)), info = deparse(shape))
  }

  # A full interaction keeps every lower-order term, so its columns are
  # still standardised
  expect_equal(run$results$preprocessing_stats$center$wt, mean(mtcars$wt))
  expect_equal(run$results$preprocessing_stats$center$qsec,
               mean(mtcars$qsec))

  binary_iris <- droplevels(iris[iris$Species != "setosa", ])
  no_intercept <- Species ~ Sepal.Length + Sepal.Width - 1
  set.seed(218)
  logit <- tl_run_pipeline(
    tl_pipeline(
      binary_iris, no_intercept,
      models = list(logit = list(method = "logistic")),
      evaluation = list(cv_folds = 3)
    ),
    verbose = FALSE
  )
  expect_equal(
    tl_predict_pipeline(logit, binary_iris, type = "response")$.pred,
    fitted(glm(no_intercept, data = binary_iris, family = binomial)),
    tolerance = 1e-6
  )
})

test_that("a computed factor response makes a classification pipeline", {
  # factor(am) ~ wt + hp is a classification for tl_model(), but the
  # pipeline read the task off the numeric column am and set up the
  # regression models and metrics, which the run then refused
  pipe <- tl_pipeline(mtcars, factor(am) ~ wt + hp)
  expect_true(all(c("accuracy", "f1") %in% pipe$evaluation$metrics))
  expect_true("logistic" %in% names(pipe$models))

  set.seed(219)
  run <- tl_run_pipeline(
    tl_pipeline(mtcars, factor(am) ~ wt + hp,
                models = list(tree = list(method = "tree")),
                evaluation = list(cv_folds = 3)),
    verbose = FALSE
  )
  expect_false(anyNA(run$results$metric_values))
  expect_true(run$results$best_model$spec$is_classification)
})

test_that("a transformed term in a logistic pipeline matches glm()", {
  # The standardised Sepal.Length gave NaN under log(), so the final glm
  # kept 51 of the 100 rows
  binary_iris <- droplevels(iris[iris$Species != "setosa", ])
  set.seed(2)
  run <- tl_run_pipeline(
    tl_pipeline(
      binary_iris, Species ~ log(Sepal.Length) + Petal.Width,
      models = list(logit = list(method = "logistic")),
      evaluation = list(cv_folds = 4)
    ),
    verbose = FALSE
  )
  direct <- glm(Species ~ log(Sepal.Length) + Petal.Width,
                data = binary_iris, family = binomial)

  best <- tl_get_best_model(run)
  expect_equal(stats::nobs(best$fit), nrow(binary_iris))
  expect_equal(coef(best$fit)[["log(Sepal.Length)"]],
               coef(direct)[["log(Sepal.Length)"]])
  expect_equal(
    tl_predict_pipeline(run, binary_iris, type = "response")$.pred,
    fitted(direct)
  )
})

test_that("tl_predict_pipeline imputes with the raw training median", {
  set.seed(203)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)

  new_data <- mtcars[1:5, ]
  new_data$wt[1] <- NA

  preds <- tl_predict_pipeline(res, new_data = new_data)

  # Substituting the median by hand must give the same answer
  manual <- mtcars[1:5, ]
  manual$wt[1] <- stats::median(mtcars$wt)
  expect_equal(preds$.pred, tl_predict_pipeline(res, new_data = manual)$.pred)
  expect_false(any(is.na(preds$.pred)))
})

test_that("tl_predict_pipeline rejects pipelines without recorded stats", {
  set.seed(204)
  res <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)
  res$results$preprocessing_stats <- NULL

  expect_error(
    tl_predict_pipeline(res, new_data = mtcars[1:5, ]),
    "did not record preprocessing statistics"
  )
})

test_that("tl_run_pipeline requires a named models list", {
  pipe <- tl_pipeline(mtcars, mpg ~ wt + hp, models = "linear")

  expect_error(tl_run_pipeline(pipe, verbose = FALSE), "named list")
})

test_that("tl_run_pipeline selects a best model using the default metrics", {
  skip_if_not_installed("rpart")

  binary_iris <- droplevels(subset(iris, Species != "virginica"))

  # Default evaluation asks for f1, which requires tl_evaluate to honour
  # the metrics argument
  pipe <- tl_pipeline(
    binary_iris, Species ~ .,
    models = list(tree = list(method = "tree"))
  )
  pipe$evaluation$cv_folds <- 3

  set.seed(205)
  res <- tl_run_pipeline(pipe, verbose = FALSE)

  expect_equal(res$results$best_model_name, "tree")
  expect_false(is.na(res$results$metric_values[["tree"]]))
})

test_that("every metric a pipeline can score has a direction", {
  known <- c(tl_known_metrics(TRUE), tl_known_metrics(FALSE))
  expect_false(anyNA(tl_metric_higher_better(known)))
  expect_equal(
    tl_metric_higher_better(
      c("sensitivity", "specificity", "pr_auc", "rsq", "rmse", "mape")
    ),
    c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE)
  )
  expect_true(is.na(tl_metric_higher_better("mystery")))

  # A multiclass auc is reported per class as well, as auc_<class>
  expect_true(tl_metric_higher_better("auc_setosa"))
})

test_that("the comparison plot knows the direction of per-class auc", {
  # auc_setosa and the other per-class rows came back with an NA direction
  set.seed(220)
  run <- tl_run_pipeline(
    tl_pipeline(iris, Species ~ .,
                models = list(tree = list(method = "tree")),
                evaluation = list(metrics = c("accuracy", "auc"),
                                  best_metric = "accuracy", cv_folds = 3)),
    verbose = FALSE
  )
  plotted <- tl_compare_pipeline_models(run)$data
  expect_true(any(grepl("^auc_", plotted$metric)))
  expect_true(all(plotted$higher_better))
})

test_that("sensitivity, specificity and pr_auc select the highest score", {
  # These were missing from the higher-is-better list, so the lowest score
  # won: a cp = 1 stump at sensitivity 0.40 over a tree at 0.90, and a tree
  # over a logistic model with the higher pr_auc
  binary_iris <- droplevels(iris[iris$Species != "setosa", ])
  candidates <- list(
    sensitivity = list(good = list(method = "tree"),
                       stump = list(method = "tree", cp = 1)),
    specificity = list(good = list(method = "tree"),
                       stump = list(method = "tree", cp = 1)),
    pr_auc = list(tree = list(method = "tree"),
                  logistic = list(method = "logistic"))
  )

  for (metric in names(candidates)) {
    set.seed(10)
    run <- suppressWarnings(tl_run_pipeline(
      tl_pipeline(
        binary_iris, Species ~ .,
        models = candidates[[metric]],
        evaluation = list(metrics = c(metric, "accuracy"),
                          best_metric = metric, cv_folds = 5)
      ),
      verbose = FALSE
    ))
    values <- run$results$metric_values
    expect_false(isTRUE(all.equal(min(values), max(values))), info = metric)
    expect_equal(run$results$best_model_name, names(which.max(values)),
                 info = metric)

    plotted <- tl_compare_pipeline_models(run)$data
    expect_true(all(plotted$higher_better[plotted$metric == metric]),
                info = metric)
  }
})

test_that("split validation predicts under the split's own statistics", {
  # The model was fitted on the training rows under their centre and
  # scale, but tl_predict_pipeline() replayed the full-data ones: wt was
  # centred on 3.217 rather than 3.279, and predictions on the test rows
  # moved by up to 0.685 mpg from the ones the model was scored on
  set.seed(3)
  run <- tl_run_pipeline(
    tl_pipeline(
      mtcars, mpg ~ wt + hp,
      models = list(lin = list(method = "linear")),
      evaluation = list(validation = "split", train_prop = 0.7)
    ),
    verbose = FALSE
  )
  train_rows <- rownames(tl_get_best_model(run)$data)
  test_rows <- setdiff(rownames(mtcars), train_rows)

  expect_length(train_rows, round(0.7 * nrow(mtcars)))
  expect_equal(run$results$preprocessing_stats$center$wt,
               mean(mtcars[train_rows, "wt"]))
  expect_equal(nrow(run$results$processed_data), length(train_rows))

  preds <- tl_predict_pipeline(run, mtcars[test_rows, ])$.pred
  direct <- lm(mpg ~ wt + hp, data = mtcars[train_rows, ])
  expect_equal(preds, predict(direct, mtcars[test_rows, ]))

  # The reported test score is the score of those same predictions
  scored <- run$results$model_results$lin$test_metrics
  expect_equal(scored$value[scored$metric == "rmse"],
               sqrt(mean((mtcars[test_rows, "mpg"] - preds)^2)))
})

test_that("a fold with no row to score is left out of the average", {
  # tl_evaluate() refuses a fold none of whose rows can be scored -- here
  # every response in it is missing -- and unhandled, that one fold
  # stopped the whole run
  scored <- mtcars[, c("mpg", "wt", "hp")]
  set.seed(209)
  folds <- rsample::vfold_cv(scored, v = 4)
  scored$mpg[rsample::complement(folds$splits[[1]])] <- NA

  # vfold_cv() draws its folds from the row count alone, so the same seed
  # gives the pipeline the same folds
  set.seed(209)
  expect_warning(
    run <- tl_run_pipeline(
      tl_pipeline(
        scored, mpg ~ wt + hp,
        models = list(lin = list(method = "linear")),
        evaluation = list(metrics = "rmse", best_metric = "rmse",
                          cv_folds = 4)
      ),
      verbose = FALSE
    ),
    "Fold 1 of model 'lin' is left out of its average"
  )

  fold_scores <- vapply(
    run$results$model_results$lin$cv_results,
    function(fold) fold$metrics$value[fold$metrics$metric == "rmse"],
    numeric(1)
  )
  expect_true(is.na(fold_scores[1]))
  expect_false(anyNA(fold_scores[-1]))
  expect_equal(run$results$metric_values[["lin"]], mean(fold_scores[-1]))

  # A single split has nothing else to score, so the error stands there
  test_only <- mtcars[, c("mpg", "wt", "hp")]
  set.seed(210)
  train_rows <- sample(nrow(test_only), round(0.7 * nrow(test_only)))
  test_only$mpg[-train_rows] <- NA
  set.seed(210)
  expect_error(
    tl_run_pipeline(
      tl_pipeline(
        test_only, mpg ~ wt + hp,
        models = list(lin = list(method = "linear")),
        evaluation = list(validation = "split", train_prop = 0.7)
      ),
      verbose = FALSE
    ),
    class = "tidylearn_no_scored_rows"
  )
})

test_that("a run warns about the logistic response conversion once", {
  # Logistic on a 0/1 numeric response warns that it converts the response
  # to a factor, and the run refits once per fold, per model, and again on
  # every row: two specs over three folds gave eight copies of the warning
  set.seed(211)
  binary <- data.frame(x = stats::rnorm(60))
  binary$y <- as.integer(binary$x + stats::rnorm(60) > 0)
  pipe <- tl_pipeline(
    binary, y ~ x,
    models = list(plain = list(method = "logistic"),
                  capped = list(method = "logistic", maxit = 50)),
    evaluation = list(metrics = "accuracy", best_metric = "accuracy",
                      cv_folds = 3)
  )

  count_conversions <- function(expr) {
    seen <- 0
    withCallingHandlers(
      expr,
      tidylearn_response_conversion = function(w) {
        seen <<- seen + 1
        invokeRestart("muffleWarning")
      }
    )
    seen
  }
  set.seed(212)
  expect_equal(count_conversions(tl_run_pipeline(pipe, verbose = FALSE)), 1)

  split_pipe <- pipe
  split_pipe$evaluation$validation <- "split"
  set.seed(213)
  expect_equal(
    count_conversions(tl_run_pipeline(split_pipe, verbose = FALSE)), 1
  )
})

test_that("tl_pipeline refuses arguments and settings it would ignore", {
  # A misspelt argument was swallowed by `...`, leaving cv_folds at 5
  expect_error(
    tl_pipeline(iris, Species ~ ., evalution = list(cv_folds = 2)),
    "tl_pipeline\\(\\) has no argument\\(s\\) evalution"
  )
  expect_error(
    tl_pipeline(iris, Species ~ ., NULL, NULL, NULL, "extra"),
    "has no argument\\(s\\) <unnamed>"
  )

  # dummy_encode = FALSE ran the same models as TRUE: model.matrix() and
  # the tree methods encode factors whatever the switch says
  expect_error(
    tl_pipeline(iris, Sepal.Length ~ .,
                preprocessing = list(dummy_encode = FALSE)),
    "dummy_encode = FALSE cannot be honoured"
  )

  # The real arguments, and dummy_encode = TRUE, are still taken
  pipe <- tl_pipeline(iris, Species ~ .,
                      preprocessing = list(dummy_encode = TRUE),
                      evaluation = list(cv_folds = 2))
  expect_equal(pipe$evaluation$cv_folds, 2)
  expect_true(pipe$preprocessing$dummy_encode)
})

test_that("tl_compare_pipeline_models names a metric the run did not score", {
  set.seed(207)
  run <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)

  # An unscored metric left nothing to plot and failed inside ggplot2's
  # faceting
  expect_error(
    tl_compare_pipeline_models(run, metrics = "nope"),
    "not scored by this pipeline: nope. Scored: rmse"
  )
  plotted <- tl_compare_pipeline_models(run, metrics = "rmse")
  expect_s3_class(plotted, "ggplot")
  expect_equal(unique(plotted$data$metric), "rmse")
})

test_that("summary() of a run pipeline shows the best model once", {
  set.seed(208)
  run <- tl_run_pipeline(make_regression_pipeline(), verbose = FALSE)

  # The best model's summary was printed, then printed again by print()
  out <- utils::capture.output(summary(run))
  expect_equal(sum(grepl("^tidylearn Model", out)), 1)
  expect_true(any(grepl("^Training Performance", out)))
})

test_that("tl_run_pipeline standardizes constant columns safely", {
  data <- mtcars[, c("mpg", "wt", "hp")]
  data$constant <- 1

  pipe <- tl_pipeline(
    data, mpg ~ wt + hp + constant,
    models = list(linear = list(method = "linear")),
    evaluation = list(
      metrics = "rmse", validation = "cv",
      cv_folds = 3, best_metric = "rmse"
    )
  )

  set.seed(206)
  res <- tl_run_pipeline(pipe, verbose = FALSE)

  expect_equal(res$results$preprocessing_stats$scale$constant, 1)
  expect_false(any(is.na(res$results$processed_data$constant)))
})
