test_that("linear regression models work", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  expect_s3_class(model, "tidylearn_linear")
  expect_false(model$spec$is_classification)

  # Predictions should be numeric
  preds <- predict(model)
  expect_type(preds$.pred, "double")
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("logistic regression models work for classification", {
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model <- tl_model(binary_iris, Species ~ ., method = "logistic")

  expect_s3_class(model, "tidylearn_logistic")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(binary_iris))
})

test_that("logistic regression refuses a response it cannot model", {
  # glm(binomial) takes a three-level factor without complaint and fits
  # the first level against the rest. The failure used to surface at
  # predict() or tl_evaluate(), three calls after the mistake.
  expect_error(
    tl_model(iris, Species ~ ., method = "logistic"),
    "binary only.*3 levels"
  )

  # The message has to name a way forward, not just refuse
  err <- tryCatch(
    tl_model(iris, Species ~ ., method = "logistic"),
    error = function(e) conditionMessage(e)
  )
  expect_match(err, "forest")
  expect_match(err, "xgboost")

  # One level is a different problem and says so
  one_class <- droplevels(subset(iris, Species == "setosa"))
  expect_error(
    tl_model(one_class, Species ~ ., method = "logistic"),
    "only one"
  )
})

test_that("logistic fits and scores a response the formula computes", {
  # The response was read off the column named first, so I(mpg > 20) ~ wt
  # was refused as "binary only, but 'mpg' has 25 levels"
  model <- withCallingHandlers(
    tl_model(mtcars, I(mpg > 20) ~ wt, method = "logistic"),
    tidylearn_response_conversion = function(w) invokeRestart("muffleWarning")
  )
  reference <- stats::glm(I(mpg > 20) ~ wt, data = mtcars,
                          family = stats::binomial())
  prob <- unname(stats::predict(reference, mtcars, type = "response"))

  probs <- predict(model, mtcars, type = "prob")
  expect_named(probs, c("FALSE", "TRUE"))
  expect_equal(unname(probs[["TRUE"]]), prob)
  expect_equal(unname(probs[["FALSE"]]), 1 - prob)
  classes <- predict(model, mtcars, type = "class")$.pred
  expect_identical(levels(classes), c("FALSE", "TRUE"))
  expect_identical(as.character(classes), ifelse(prob > 0.5, "TRUE", "FALSE"))
  expect_equal(tl_coefficients(model)$estimate,
               unname(stats::coef(reference)))

  # The plots score the computed response too. By hand: the AUC is the
  # chance a car over 20 mpg outscores one that is not.
  pos <- prob[mtcars$mpg > 20]
  neg <- prob[mtcars$mpg <= 20]
  auc <- mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
  expect_identical(plot(model, type = "roc")$labels$subtitle,
                   paste0("AUC = ", round(auc, 3)))
  confusion <- plot(model, type = "confusion")$data
  expect_equal(sum(confusion$Freq), nrow(mtcars))

  # A computed response with three classes is refused under its own name
  expect_error(tl_model(mtcars, cut(mpg, 3) ~ wt, method = "logistic"),
               "binary only, but 'cut\\(mpg, 3\\)' has 3 levels")

  # Computed as text, or as numbers other than 0 and 1, it is fitted as the
  # two classes it encodes. glm() stopped with "y values must be
  # 0 <= y <= 1".
  text <- tl_model(mtcars, ifelse(mpg > 20, "hi", "lo") ~ wt,
                   method = "logistic")
  expect_named(predict(text, mtcars, type = "prob"), c("hi", "lo"))
  expect_equal(unname(predict(text, mtcars, type = "prob")$lo), 1 - prob)

  coded <- withCallingHandlers(
    tl_model(mtcars, I(am + 1) ~ wt, method = "logistic"),
    tidylearn_response_conversion = function(w) invokeRestart("muffleWarning")
  )
  am_reference <- stats::glm(am ~ wt, data = mtcars,
                             family = stats::binomial())
  expect_named(predict(coded, mtcars, type = "prob"), c("1", "2"))
  expect_equal(unname(predict(coded, mtcars, type = "prob")[["2"]]),
               unname(stats::predict(am_reference, mtcars,
                                     type = "response")))

  # A computed factor whose first level no row has would make glm() count
  # every row as the other class; without that level it fits as glm() does
  bands <- c(0, 5, 20, 50)
  expect_error(tl_model(mtcars, cut(mpg, bands) ~ wt, method = "logistic"),
               "declares '\\(0,5\\]' first, a class no row has")
  dropped <- tl_model(mtcars, droplevels(cut(mpg, bands)) ~ wt,
                      method = "logistic")
  expect_equal(
    unname(stats::coef(dropped$fit)),
    unname(stats::coef(stats::glm(I(mpg > 20) ~ wt, data = mtcars,
                                  family = stats::binomial())))
  )
})

test_that("tree models work for classification", {
  skip_if_not_installed("rpart")

  model <- tl_model(iris, Species ~ Sepal.Length + Sepal.Width, method = "tree")

  expect_s3_class(model, "tidylearn_tree")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("tree models work for regression", {
  skip_if_not_installed("rpart")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")

  expect_s3_class(model, "tidylearn_tree")
  expect_false(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_type(preds$.pred, "double")
})

test_that("random forest models work for classification", {
  skip_if_not_installed("randomForest")

  model <- tl_model(iris, Species ~ ., method = "forest")

  expect_s3_class(model, "tidylearn_forest")
  expect_true(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("random forest models work for regression", {
  skip_if_not_installed("randomForest")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "forest")

  expect_s3_class(model, "tidylearn_forest")
  expect_false(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_type(preds$.pred, "double")
})

test_that("ridge regression works", {
  skip_if_not_installed("glmnet")

  model <- tl_model(mtcars, mpg ~ ., method = "ridge")

  expect_s3_class(model, "tidylearn_ridge")

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("lasso regression works", {
  skip_if_not_installed("glmnet")

  model <- tl_model(mtcars, mpg ~ ., method = "lasso")

  expect_s3_class(model, "tidylearn_lasso")

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("elastic net works", {
  skip_if_not_installed("glmnet")

  model <- tl_model(mtcars, mpg ~ ., method = "elastic_net", alpha = 0.5)

  expect_s3_class(model, "tidylearn_elastic_net")

  # Predictions
  preds <- predict(model)
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("polynomial regression works", {
  model <- tl_model(mtcars, mpg ~ wt, method = "polynomial", degree = 2)

  expect_s3_class(model, "tidylearn_polynomial")
  expect_false(model$spec$is_classification)

  # Predictions
  preds <- predict(model)
  expect_type(preds$.pred, "double")
})

test_that("polynomial adds its terms to the formula as written", {
  # The formula was rebuilt from variable names as response ~ poly(...), so
  # log(mpg) ~ wt was fitted on the mpg scale, offset() and - 1 were
  # dropped, and poly(wt, 3) became poly(poly(wt, 3), ...)
  cases <- list(
    list(log(mpg) ~ wt, log(mpg) ~ poly(wt, degree = 2, raw = TRUE)),
    list(mpg ~ wt + offset(disp / 100),
         mpg ~ poly(wt, degree = 2, raw = TRUE) + offset(disp / 100)),
    list(mpg ~ wt - 1, mpg ~ poly(wt, degree = 2, raw = TRUE) - 1),
    list(mpg ~ poly(wt, 3) + hp, mpg ~ poly(wt, 3) +
           poly(hp, degree = 2, raw = TRUE)),
    list(mpg ~ wt * hp + I(qsec^2),
         mpg ~ wt * hp + I(wt^2) + I(hp^2) + I(qsec^2))
  )
  for (case in cases) {
    model <- tl_model(mtcars, case[[1]], method = "polynomial")
    reference <- stats::lm(case[[2]], data = mtcars)
    expect_equal(unname(predict(model, mtcars)$.pred),
                 unname(stats::predict(reference, mtcars)),
                 info = deparse(case[[1]]))
    # The same terms, in whatever order update() leaves them
    expect_setequal(names(stats::coef(model$fit)),
                    names(stats::coef(reference)))
    expect_equal(stats::coef(model$fit)[names(stats::coef(reference))],
                 stats::coef(reference), info = deparse(case[[1]]))
  }

  cubic <- tl_model(mtcars, log(mpg) ~ wt, method = "polynomial", degree = 3)
  expect_equal(
    unname(predict(cubic, mtcars)$.pred),
    unname(stats::predict(
      stats::lm(log(mpg) ~ poly(wt, degree = 3, raw = TRUE), data = mtcars),
      mtcars
    ))
  )
})

test_that("polynomial expands a one-column matrix term such as scale()", {
  # Only plain vectors were expanded, so mpg ~ scale(wt) + hp kept
  # scale(wt) linear, and mpg ~ scale(wt) with degree = 3 fitted a straight
  # line without a message
  model <- tl_model(mtcars, mpg ~ scale(wt) + hp, method = "polynomial")
  reference <- stats::lm(mpg ~ poly(scale(wt), degree = 2, raw = TRUE) +
                           poly(hp, degree = 2, raw = TRUE), data = mtcars)
  expect_equal(unname(stats::fitted(model$fit)),
               unname(stats::fitted(reference)))

  cubic <- tl_model(mtcars, mpg ~ scale(wt), method = "polynomial",
                    degree = 3)
  reference <- stats::lm(mpg ~ poly(scale(wt), degree = 3, raw = TRUE),
                         data = mtcars)
  expect_equal(unname(stats::fitted(cubic$fit)),
               unname(stats::fitted(reference)))

  # A basis is left as written, a one-column one included
  basis <- tl_model(mtcars, mpg ~ poly(wt, 1) + hp, method = "polynomial")
  reference <- stats::lm(mpg ~ poly(wt, 1) + poly(hp, degree = 2, raw = TRUE),
                         data = mtcars)
  expect_equal(stats::coef(basis$fit), stats::coef(reference))
})

test_that("polynomial predicts a scale() term with the training centre", {
  # predict.lm() keeps the centre and scale of a scale() term it fits, but
  # not of one inside poly() or I(), so new rows were scaled on their own:
  # one row predicted NaN, and mtcars[1:5, ] gave 20.48 for its second row
  # against a fitted 21.71
  for (formula in list(mpg ~ scale(wt) + hp, mpg ~ scale(wt) * hp)) {
    model <- tl_model(mtcars, formula, method = "polynomial")
    fitted_values <- unname(stats::fitted(model$fit))
    expect_equal(unname(predict(model, mtcars[1, ])$.pred), fitted_values[1],
                 info = deparse(formula))
    expect_equal(unname(predict(model, mtcars[1:5, ])$.pred),
                 fitted_values[1:5], info = deparse(formula))
  }

  # Held-out rows are scaled on the rows the model was fitted on, as the
  # same model written out with that centre and scale predicts them
  train <- mtcars[1:24, ]
  test <- mtcars[25:32, ]
  centre <- mean(train$wt)
  spread <- stats::sd(train$wt)
  train$z <- (train$wt - centre) / spread
  test$z <- (test$wt - centre) / spread
  cases <- list(
    list(mpg ~ scale(wt) + hp,
         mpg ~ poly(z, degree = 2, raw = TRUE) +
           poly(hp, degree = 2, raw = TRUE)),
    list(mpg ~ scale(wt) * hp, mpg ~ z * hp + I(z^2) + I(hp^2))
  )
  for (case in cases) {
    model <- tl_model(train, case[[1]], method = "polynomial")
    expected <- unname(stats::predict(stats::lm(case[[2]], data = train),
                                      test))
    expect_equal(unname(predict(model, test)$.pred), expected,
                 info = deparse(case[[1]]))
    expect_equal(unname(predict(model, test[1, ])$.pred), expected[1],
                 info = deparse(case[[1]]))
  }

  # The terms keep the names they were fitted under
  model <- tl_model(mtcars, mpg ~ scale(wt) + hp, method = "polynomial")
  expect_identical(names(stats::coef(model$fit))[2:3],
                   paste0("poly(scale(wt), degree = 2, raw = TRUE)", 1:2))
})

test_that("polynomial keeps a numeric term that is part of an interaction", {
  # wt was replaced by poly(wt), which left cyl_f:wt with no wt main
  # effect. model.matrix() then coded it with every level of cyl_f, and one
  # coefficient was NA.
  mt <- transform(mtcars, cyl_f = factor(cyl))
  model <- tl_model(mt, mpg ~ cyl_f * wt, method = "polynomial")
  reference <- stats::lm(mpg ~ cyl_f * wt + I(wt^2), data = mt)
  expect_false(anyNA(stats::coef(model$fit)))
  expect_setequal(names(stats::coef(model$fit)),
                  names(stats::coef(reference)))
  expect_equal(stats::coef(model$fit)[names(stats::coef(reference))],
               stats::coef(reference))
  expect_equal(unname(predict(model, mt)$.pred),
               unname(stats::predict(reference, mt)))

  cubic <- tl_model(mt, mpg ~ cyl_f * wt, method = "polynomial", degree = 3)
  expect_equal(
    unname(stats::fitted(cubic$fit)),
    unname(stats::fitted(stats::lm(mpg ~ cyl_f * wt + I(wt^2) + I(wt^3),
                                   data = mt)))
  )
})

test_that("polynomial leaves factor predictors as factors", {
  # Species went into poly() and was fitted on its integer codes, so the
  # same virginica row predicted 6.96 with every level declared and 7.66
  # after droplevels()
  model <- tl_model(iris, Sepal.Length ~ ., method = "polynomial")
  reference <- stats::lm(
    Sepal.Length ~ poly(Sepal.Width, degree = 2, raw = TRUE) +
      poly(Petal.Length, degree = 2, raw = TRUE) +
      poly(Petal.Width, degree = 2, raw = TRUE) + Species,
    data = iris
  )
  expect_equal(unname(predict(model, iris)$.pred),
               unname(stats::predict(reference, iris)))

  row <- iris[101, ]
  expect_equal(unname(predict(model, droplevels(row))$.pred),
               unname(stats::predict(reference, row)))
})

test_that("supervised models handle new data correctly", {
  # Split data
  split <- tl_split(iris, prop = 0.7, seed = 123)

  # Train on training set
  model <- tl_model(split$train, Species ~ ., method = "forest")

  # Predict on test set
  preds <- predict(model, new_data = split$test)

  expect_equal(nrow(preds), nrow(split$test))
})

test_that("supervised models work with formula variations", {
  # Formula with interaction
  model1 <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")
  expect_s3_class(model1, "tidylearn_linear")

  # Formula with all variables
  # versicolor and virginica overlap. setosa is linearly separable from
  # both, and glm() cannot converge on a perfectly separable response.
  binary_iris <- droplevels(subset(iris, Species != "setosa"))
  model2 <- tl_model(binary_iris, Species ~ ., method = "logistic")
  expect_s3_class(model2, "tidylearn_logistic")

  # Formula with subset of variables
  model3 <- tl_model(binary_iris, Species ~ Sepal.Length + Petal.Length,
                     method = "logistic")
  expect_s3_class(model3, "tidylearn_logistic")
})

# Neural networks. Nothing in the suite fitted one before, which is how a
# two-class fit came to be broken for as long as it was.

test_that("neural networks fit two-class problems", {
  skip_if_not_installed("nnet")

  # nnet.formula() supplies entropy = TRUE itself for a two-level factor.
  # Naming it again here reached nnet.default() twice and stopped with
  # "formal argument 'entropy' matched by multiple actual arguments" -- for
  # every binary classification, on any data.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  model <- tl_model(iris_binary, Species ~ ., method = "nn", trace = FALSE)

  expect_s3_class(model, "tidylearn_nn")
  expect_true(model$spec$is_classification)
  expect_equal(nrow(predict(model)), nrow(iris_binary))
})

test_that("the error criterion follows the number of classes", {
  skip_if_not_installed("nnet")

  # Left to nnet: cross-entropy on two levels, softmax on three or more.
  # Multiclass tolerated the duplicate argument only because
  # nnet.default() sets entropy <- FALSE whenever softmax is on, so this
  # also pins that the multiclass fit is unchanged by the repair.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  binary <- tl_model(iris_binary, Species ~ ., method = "nn", trace = FALSE)
  expect_true(binary$fit$entropy)
  expect_false(binary$fit$softmax)

  set.seed(1)
  multiclass <- tl_model(iris, Species ~ ., method = "nn", trace = FALSE)
  expect_false(multiclass$fit$entropy)
  expect_true(multiclass$fit$softmax)
})

test_that("two-class neural network prediction covers every type", {
  skip_if_not_installed("nnet")

  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)
  levels_expected <- levels(iris_binary$Species)

  set.seed(1)
  model <- tl_model(iris_binary, Species ~ ., method = "nn", trace = FALSE)

  labels <- predict(model, type = "class")
  expect_true(all(labels$.pred %in% levels_expected))

  probs <- predict(model, type = "prob")
  expect_named(probs, levels_expected)
  expect_equal(rowSums(probs), rep(1, nrow(iris_binary)), tolerance = 1e-6)

  expect_equal(nrow(predict(model, new_data = iris_binary[1, ])), 1)

  metrics <- tl_evaluate(model, metrics = "accuracy")
  expect_gt(metrics$value[metrics$metric == "accuracy"], 0.8)
})

test_that("neural network arguments still reach nnet", {
  skip_if_not_installed("nnet")

  # The repair removed an argument from the call; the pass-through that
  # shares that `...` has to keep working.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  model <- tl_model(
    iris_binary, Species ~ .,
    method = "nn", size = 3, decay = 0.1, maxit = 50, trace = FALSE
  )

  expect_equal(model$fit$n[2], 3)
  expect_equal(model$fit$decay, 0.1)
})

test_that("neural network tuning runs for every response type", {
  skip_if_not_installed("nnet")

  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  tuned <- tl_tune_nn(
    iris_binary, Species ~ .,
    is_classification = TRUE,
    sizes = c(2, 3), decays = c(0, 0.1), folds = 2
  )

  expect_true(tuned$best_size %in% c(2, 3))
  expect_true(tuned$best_decay %in% c(0, 0.1))
  expect_equal(nrow(tuned$tuning_results), 4)
})

# ---- deep learning: the learning rate has to reach the optimizer -----

test_that("tl_fit_deep(learning_rate=) sets the optimizer's learning rate", {
  skip_on_cran()
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  usable <- tryCatch({
    keras::keras_model_sequential()
    TRUE
  }, error = function(e) FALSE)
  skip_if_not(usable, "No TensorFlow backend available")

  # tl_tune_deep() passed optimizer = optimizer_adam(learning_rate = lr)
  # into tl_fit_deep(), which has no such formal, so it landed in ... and
  # went to keras::fit(). The model is already compiled by then, and
  # compile() is what sets the optimizer -- so the whole learning_rates
  # grid searched over a value that never changed anything.
  set.seed(1)
  d <- data.frame(x1 = stats::rnorm(60), x2 = stats::rnorm(60))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(60, sd = 0.3)

  rate_of <- function(lr) {
    fit <- suppressWarnings(suppressMessages(
      tl_fit_deep(d, y ~ x1 + x2, is_classification = FALSE,
                  hidden_layers = c(4), epochs = 1, verbose = 0,
                  learning_rate = lr)
    ))
    as.numeric(keras::k_get_value(fit$model$optimizer$learning_rate))
  }

  expect_equal(rate_of(0.5), 0.5, tolerance = 1e-6)
  expect_equal(rate_of(0.01), 0.01, tolerance = 1e-6)

  # Left alone, keras keeps its own default rather than being forced
  expect_true(is.finite(rate_of(NULL)))
})

test_that("tl_tune_deep actually searches over the learning rate", {
  skip_on_cran()
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  usable <- tryCatch({
    keras::keras_model_sequential()
    TRUE
  }, error = function(e) FALSE)
  skip_if_not(usable, "No TensorFlow backend available")

  set.seed(1)
  d <- data.frame(x1 = stats::rnorm(80), x2 = stats::rnorm(80))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(80, sd = 0.3)

  tuned <- suppressWarnings(suppressMessages(tl_tune_deep(
    d, y ~ x1 + x2,
    hidden_layers_options = list(c(4)),
    learning_rates = c(0.5, 0.0001),
    batch_sizes = c(16),
    epochs = 3
  )))

  scored <- tuned$tuning_results$val_loss
  expect_equal(length(scored), 2L)
  expect_true(all(is.finite(scored)))

  # Two very different rates over the same data and architecture cannot
  # score identically unless the rate is being ignored, which is exactly
  # what happened when it was routed to fit() instead of compile().
  expect_false(isTRUE(all.equal(scored[1], scored[2])))

  # And the winner has to reach the model that gets returned -- the final
  # refit routed it through optimizer = too. The returned model is a
  # tidylearn_model, so the keras model is at $fit$model.
  final_rate <- as.numeric(
    keras::k_get_value(tuned$model$fit$model$optimizer$learning_rate)
  )
  expect_equal(final_rate, tuned$best_learning_rate, tolerance = 1e-6)
})

# ---- glmnet and missing predictors -----------------------------------

test_that("ridge/lasso/elastic_net tolerate a missing predictor value", {
  # The response was read straight from `data` while the design matrix
  # came from model.frame(), which applies na.omit. One missing predictor
  # therefore left y one row longer than x, and glmnet reported "number of
  # observations in y (60) not equal to the number of rows of x (59)" --
  # a dimension mismatch that names neither missing values nor the column.
  set.seed(1)
  n <- 60
  d <- data.frame(x1 = stats::rnorm(n), x2 = stats::rnorm(n))
  d$y <- 2 * d$x1 - d$x2 + stats::rnorm(n, sd = 0.3)
  d$x1[3] <- NA

  for (method in c("ridge", "lasso", "elastic_net")) {
    model <- suppressWarnings(suppressMessages(
      tl_model(d, y ~ x1 + x2, method = method)
    ))
    expect_s3_class(model, paste0("tidylearn_", method))
    # The same rows lm() keeps, so the family behaves consistently
    expect_equal(model$fit$nobs, 59L, info = method)
  }

  # lm() is the reference for what "drop the incomplete row" means here
  linear <- suppressWarnings(tl_model(d, y ~ x1 + x2, method = "linear"))
  expect_equal(unname(stats::nobs(linear$fit)), 59L)
})

test_that("a missing predictor does not break glmnet classification", {
  di <- droplevels(iris[iris$Species != "setosa", ])
  di$Sepal.Length[2] <- NA

  for (method in c("ridge", "lasso")) {
    model <- suppressWarnings(suppressMessages(
      tl_model(di, Species ~ ., method = method)
    ))
    expect_s3_class(model, paste0("tidylearn_", method))
  }
})

test_that("weights and foldid follow the rows a missing value removes", {
  # model.frame() dropped the incomplete row from x and y but not from the
  # per-row vectors, so glmnet reported "number of elements in weights (32)
  # not equal to the number of rows of x (31)"
  d <- mtcars
  d$wt[3] <- NA
  complete <- !is.na(d$wt)
  x <- as.matrix(d[complete, c("wt", "hp")])
  w <- mtcars$cyl / 4

  model <- tl_model(d, mpg ~ wt + hp, method = "lasso", lambda = 0.1,
                    weights = w)
  reference <- glmnet::glmnet(x, d$mpg[complete], lambda = 0.1,
                              weights = w[complete])
  expect_equal(as.vector(stats::coef(model$fit)),
               as.vector(stats::coef(reference)))

  # foldid failed with "logical subscript too long"
  folds <- rep(1:4, 8)
  cv_model <- tl_model(d, mpg ~ wt + hp, method = "lasso", foldid = folds)
  reference_cv <- glmnet::cv.glmnet(x, d$mpg[complete],
                                    foldid = folds[complete])
  expect_equal(attr(cv_model$fit, "cv_results")$cvm, reference_cv$cvm)
})

test_that("a formula without an intercept keeps every glmnet predictor", {
  # The design's first column was dropped as the intercept, so
  # mpg ~ wt + hp + disp - 1 lost wt without a message
  x <- as.matrix(mtcars[, c("wt", "hp", "disp")])
  reference <- glmnet::glmnet(x, mtcars$mpg, lambda = 0.01)
  for (f in list(mpg ~ wt + hp + disp - 1, mpg ~ wt + hp + disp + 0)) {
    model <- tl_model(mtcars, f, method = "lasso", lambda = 0.01)
    expect_identical(attr(model$fit, "tl_colnames"), c("wt", "hp", "disp"))
    expect_equal(predict(model, mtcars)$.pred,
                 as.vector(stats::predict(reference, newx = x)))
  }

  # Without an intercept the first factor is coded in full, as
  # model.matrix() does
  mt <- transform(mtcars, cyl = factor(cyl))
  coded <- tl_model(mt, mpg ~ cyl + wt - 1, method = "ridge", lambda = 0.1)
  expect_identical(attr(coded$fit, "tl_colnames"),
                   c("cyl4", "cyl6", "cyl8", "wt"))
  expect_length(predict(coded, mt[1:3, ])$.pred, 3)
})

test_that("ridge/lasso/elastic_net refuse an offset in either form", {
  # offset() in the formula was left out of the fit without a word, and an
  # offset argument fitted but left predict() failing for want of newoffset
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp + offset(disp / 100), method = "lasso",
             lambda = 0.1),
    "cannot use the formula's offset\\(disp/100\\)"
  )
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "ridge",
             offset = mtcars$disp / 100),
    "do not take an offset"
  )
  # The model without the offset still fits
  expect_s3_class(
    tl_model(mtcars, mpg ~ wt + hp, method = "lasso", lambda = 0.1),
    "tidylearn_lasso"
  )
})

test_that("an argument glmnet does not take is an error, not ignored", {
  # cv.glmnet() and glmnet() drop names they do not know, so a misspelt
  # standardise = FALSE changed nothing while $spec$args recorded it
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso", lambda = 0.5,
             standardise = FALSE),
    "no argument named 'standardise'"
  )
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "lasso", strata = mtcars$cyl),
    "no argument named 'strata'"
  )

  # A glmnet argument spelt right still reaches glmnet
  x <- as.matrix(mtcars[, c("wt", "hp", "disp")])
  unscaled <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                       lambda = 0.5, standardize = FALSE)
  reference <- glmnet::glmnet(x, mtcars$mpg, lambda = 0.5,
                              standardize = FALSE)
  expect_equal(as.vector(stats::coef(unscaled$fit)),
               as.vector(stats::coef(reference)))

  # A cross-validation argument has nothing to act on at a fixed penalty
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "lasso", lambda = 0.1,
             type.measure = "mae"),
    "only applies to the cross-validation"
  )
  # and reaches cv.glmnet() when the penalty is chosen by it
  set.seed(1)
  by_mae <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                     type.measure = "mae")
  expect_identical(unname(attr(by_mae$fit, "cv_results")$name),
                   "Mean Absolute Error")
})

test_that("subset chooses the rows a glmnet model is fitted on", {
  # It reached glmnet, which has no subset argument, so all 32 rows were
  # fitted while $spec$per_row_args recorded the subset
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso", lambda = 0.1,
                    subset = 1:16)
  reference <- glmnet::glmnet(as.matrix(mtcars[1:16, c("wt", "hp")]),
                              mtcars$mpg[1:16], lambda = 0.1)
  expect_equal(model$fit$nobs, 16)
  expect_equal(as.vector(stats::coef(model$fit)),
               as.vector(stats::coef(reference)))
})

test_that("a glmnet fit records each design column's spread on its rows", {
  # Importance scales a coefficient by its predictor's standard deviation,
  # and could only take it from every row of model$data -- including rows
  # the fit dropped for a missing value
  d <- mtcars
  d$wt[1:4] <- NA
  model <- tl_model(d, mpg ~ wt + hp + factor(cyl), method = "lasso",
                    lambda = 0.1)
  used <- stats::model.matrix(mpg ~ wt + hp + factor(cyl), d)[, -1]
  expect_equal(nrow(used), 28)
  expect_identical(names(attr(model$fit, "tl_x_sd")),
                   attr(model$fit, "tl_colnames"))
  expect_equal(unname(attr(model$fit, "tl_x_sd")),
               unname(apply(used, 2, stats::sd)))

  # In design order, so two columns that share a name keep their own
  set.seed(3)
  dup <- data.frame(
    y = stats::rnorm(40),
    a = factor(sample(c("x", "b"), 40, TRUE), levels = c("x", "b")),
    ab = stats::rnorm(40), z = stats::rnorm(40)
  )
  model <- tl_model(dup, y ~ a + ab + z, method = "ridge", lambda = 0.1)
  expect_identical(names(attr(model$fit, "tl_x_sd")), c("ab", "ab", "z"))
  expect_equal(unname(attr(model$fit, "tl_x_sd")),
               c(stats::sd(dup$a == "b"), stats::sd(dup$ab), stats::sd(dup$z)))
})

test_that("arguments tidylearn sets for glmnet are refused by name", {
  # They reached cv.glmnet() a second time and failed with R's "formal
  # argument "nfolds" matched by multiple actual arguments"
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "lasso", nfolds = 4),
    "tidylearn sets 'nfolds' from cv_folds"
  )
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "lasso", family = "poisson"),
    "tidylearn sets 'family' from the response"
  )
  # A relaxed fit would be ignored: predict() and the coefficients read the
  # unrelaxed one
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso", relax = TRUE),
    "relax = TRUE fits a relaxed lasso"
  )
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso", gamma = 0.5),
    "'gamma' chooses among relaxed fits"
  )
  # relax = FALSE is glmnet's default, and changes nothing
  set.seed(1)
  plain <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso")
  set.seed(1)
  unrelaxed <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                        relax = FALSE)
  expect_equal(predict(unrelaxed, mtcars), predict(plain, mtcars))

  # The number of folds is cv_folds
  set.seed(1)
  four <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso", cv_folds = 4,
                   keep = TRUE)
  expect_equal(sort(unique(attr(four$fit, "cv_results")$foldid)), 1:4)
})

test_that("a glmnet model without stored classes takes the spec's", {
  # A model fitted before the classes were stored on the fit fell back to
  # the raw column, so cut(mpg, ...) ~ . had 25 "classes" and failed
  d <- transform(mtcars, band = cut(mpg, c(0, 20, 50)))
  model <- tl_model(d, cut(mpg, c(0, 20, 50)) ~ wt + hp + qsec,
                    method = "ridge", lambda = 0.1)
  expected <- predict(model, d, type = "prob")
  attr(model$fit, "response_levels") <- NULL
  expect_named(predict(model, d, type = "prob"), levels(d$band))
  expect_equal(predict(model, d, type = "prob"), expected)
})

test_that("a class the fitted rows lack is dropped from a glmnet response", {
  # Missing predictor values removed every setosa row, which left setosa as
  # an empty level, and glmnet stopped on a class with no observations
  d <- iris
  d$Sepal.Width[d$Species == "setosa"] <- NA
  set.seed(1)
  model <- tl_model(d, Species ~ ., method = "lasso")
  expect_identical(attr(model$fit, "response_levels"),
                   c("versicolor", "virginica"))
  expect_s3_class(model$fit, "lognet")

  rows <- d[d$Species != "setosa", ]
  x <- as.matrix(rows[, 1:4])
  cv <- attr(model$fit, "cv_results")
  expect_equal(
    predict(model, rows, type = "prob")$virginica,
    as.vector(stats::predict(cv, newx = x, s = "lambda.1se",
                             type = "response"))
  )

  # One class left is nothing to tell apart
  one_left <- iris
  one_left$Sepal.Width[one_left$Species != "virginica"] <- NA
  expect_error(tl_model(one_left, Species ~ ., method = "lasso"),
               "need rows of at least two classes")
})

test_that("a single design column is refused in tidylearn's words", {
  # glmnet's "x should be a matrix with 2 or more columns" names an
  # argument the caller never passed
  expect_error(tl_model(mtcars, mpg ~ wt, method = "lasso"),
               "need at least two predictor columns")
  expect_error(tl_model(mtcars, mpg ~ wt, method = "lasso"),
               "mpg ~ wt gives one \\(wt\\)")
  expect_error(tl_model(mtcars, mpg ~ wt, method = "ridge", lambda = 0.1),
               "need at least two predictor columns")
  # One factor with three levels is two columns, and fits
  mt <- transform(mtcars, cyl = factor(cyl))
  expect_s3_class(tl_model(mt, mpg ~ cyl, method = "ridge"),
                  "tidylearn_ridge")
})

test_that("a tree sends rpart()'s own arguments to rpart()", {
  # Everything in ... went to rpart.control(), which discards what it does
  # not recognise, so weights had no effect and raised no error
  w <- rep(c(1, 10), length.out = nrow(iris))
  weighted <- tl_model(iris, Species ~ ., method = "tree", weights = w)
  plain <- tl_model(iris, Species ~ ., method = "tree")
  expect_false(isTRUE(all.equal(weighted$fit$frame$wt, plain$fit$frame$wt)))

  # Control arguments still reach rpart.control()
  tuned <- tl_model(iris, Species ~ ., method = "tree", cp = 0.001, xval = 0)
  expect_equal(tuned$fit$control$cp, 0.001)
  expect_equal(tuned$fit$control$xval, 0)
})

test_that("linear and logistic fits take case weights", {
  w <- rep(c(1, 3), length.out = nrow(mtcars))
  lin <- tl_model(mtcars, mpg ~ wt, method = "linear", weights = w)
  expect_equal(coef(lin$fit), coef(lm(mpg ~ wt, data = mtcars, weights = w)))
  # The call refers to the frame and the weight vector rather than holding
  # 32 rows and 32 weights literally
  expect_true(is.language(lin$fit$call$data))
  expect_true(is.language(lin$fit$call$weights))

  am <- transform(mtcars, am = factor(am))
  logit <- tl_model(am, am ~ wt, method = "logistic", weights = w)
  reference <- suppressWarnings(
    glm(am ~ wt, data = am, family = binomial(), weights = w)
  )
  expect_equal(unname(coef(logit$fit)), unname(coef(reference)))
})

test_that("a logistic fit can still be re-run by update() and step()", {
  # With family stored as the bare symbol `family`, re-evaluating the call
  # found stats::family() and failed
  two_class <- droplevels(iris[iris$Species != "setosa", ])
  fit <- tl_model(two_class, Species ~ Sepal.Length + Sepal.Width + Petal.Width,
                  method = "logistic")$fit
  # update() re-evaluates the stored call, which names `data`
  data <- two_class
  expect_identical(fit$call$family, quote(binomial()))
  expect_s3_class(stats::update(fit, . ~ . - Sepal.Width), "glm")
  expect_s3_class(stats::step(fit, trace = 0), "glm")
})

test_that("an offset argument is refused for offset() in the formula", {
  expect_error(
    tl_model(mtcars, mpg ~ wt, method = "linear", offset = mtcars$hp / 100),
    "offset\\(<column>\\)"
  )
  # offset() in the formula fits and predicts
  model <- tl_model(transform(mtcars, off = hp / 100),
                    mpg ~ wt + offset(off), method = "linear")
  expect_equal(nrow(predict(model, transform(mtcars, off = hp / 100))), 32)
})

# ---- regression plots ------------------------------------------------

# The built data of a plot's first layer drawn with `geom`
layer_of_geom <- function(p, geom) {
  index <- which(vapply(p$layers, function(l) inherits(l$geom, geom),
                        logical(1)))
  ggplot2::layer_data(p, index[1])
}

test_that("residual plots work for regression methods without an lm fit", {
  # fitted() and residuals() are NULL on a glmnet fit, so the plot was
  # returned and failed only when printed
  set.seed(1)
  ridge <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "ridge")
  preds <- predict(ridge, mtcars)$.pred

  points <- layer_of_geom(plot(ridge, type = "residuals"), "GeomPoint")
  expect_equal(points$x, preds)
  expect_equal(points$y, mtcars$mpg - preds)
  bars <- layer_of_geom(tl_plot_residuals(ridge, type = "histogram"),
                        "GeomBar")
  expect_equal(sum(bars$count), nrow(mtcars))

  # On the scale the model was fitted on
  logged <- tl_model(mtcars, log(mpg) ~ wt + hp, method = "lasso",
                     lambda = 0.01)
  points <- layer_of_geom(plot(logged, type = "residuals"), "GeomPoint")
  expect_equal(points$y, log(mtcars$mpg) - predict(logged, mtcars)$.pred)

  # A classification model without an lm or glm fit has no residuals
  forest <- tl_model(iris, Species ~ ., method = "forest")
  expect_error(plot(forest, type = "residuals"),
               "Residual plots are for regression models")
})

test_that("residuals against predictions skip the rows lm() dropped", {
  # The predictions covered all 32 rows and the residuals the 31 lm() kept,
  # so type = "predicted" failed with "Can't recycle input of size 32"
  d <- mtcars
  d$wt[3] <- NA
  linear <- tl_model(d, mpg ~ wt + hp, method = "linear")
  points <- layer_of_geom(tl_plot_residuals(linear, type = "predicted"),
                          "GeomPoint")
  expect_equal(points$x, unname(stats::fitted(linear$fit)))
  expect_equal(points$y, unname(stats::residuals(linear$fit)))
})

test_that("diagnostic plots are refused by name without an lm or glm fit", {
  # rstandard() has no method for glmnet, and the error said only that
  ridge <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "ridge")
  expect_error(tl_plot_diagnostics(ridge),
               "Diagnostic plots need a model fitted by lm\\(\\) or glm\\(\\)")
  expect_error(plot(ridge, type = "diagnostics"),
               "Diagnostic plots need a model fitted by lm\\(\\) or glm\\(\\)")
  # The methods that have one still get all four plots
  linear <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_length(plot(linear, type = "diagnostics"), 4)
})

test_that("actual vs predicted leaves incomplete rows out of its statistics", {
  # One missing value made the subtitle "Correlation: NA, R-squared: NA"
  d <- mtcars
  d$wt[3] <- NA
  model <- tl_model(d, mpg ~ wt + hp, method = "linear")
  expect_warning(
    p <- plot(model, type = "actual_predicted"),
    "1 row\\(s\\) with a missing response or prediction are left out"
  )
  complete <- d[-3, ]
  preds <- unname(stats::predict(model$fit, complete))
  r <- stats::cor(complete$mpg, preds)
  expect_identical(p$labels$subtitle,
                   paste0("Correlation: ", round(r, 3),
                          ", R-squared: ", round(r^2, 3)))
  expect_equal(layer_of_geom(p, "GeomPoint")$x, complete$mpg)
})

test_that("actual vs predicted compares on the scale the model was fitted on", {
  # The actuals were the raw column, so log(mpg) ~ wt plotted mpg against
  # predictions of log(mpg) and correlated the two scales in the subtitle
  logged <- tl_model(mtcars, log(mpg) ~ wt, method = "linear")
  p <- plot(logged, type = "actual_predicted")
  fitted_log <- unname(stats::fitted(logged$fit))

  points <- layer_of_geom(p, "GeomPoint")
  expect_equal(points$x, log(mtcars$mpg))
  expect_equal(points$y, fitted_log)
  r <- stats::cor(log(mtcars$mpg), fitted_log)
  expect_identical(p$labels$subtitle,
                   paste0("Correlation: ", round(r, 3),
                          ", R-squared: ", round(r^2, 3)))
})

test_that("interval plots take the formula's predictors and response scale", {
  # all.vars() on mpg ~ . gave "." as the x variable, and log(mpg) ~ wt
  # drew the raw mpg points against bands on the log scale
  dot <- tl_model(mtcars[, c("mpg", "wt", "hp")], mpg ~ ., method = "linear")
  p <- tl_plot_intervals(dot)
  expect_identical(p$labels$x, "wt")
  expect_equal(layer_of_geom(p, "GeomPoint")$x, sort(mtcars$wt))

  logged <- tl_model(mtcars, log(mpg) ~ wt, method = "linear")
  p <- tl_plot_intervals(logged)
  sorted <- mtcars[order(mtcars$wt), ]
  expect_identical(p$labels$y, "log(mpg)")
  expect_equal(layer_of_geom(p, "GeomPoint")$y, log(sorted$mpg))
  expect_equal(layer_of_geom(p, "GeomLine")$y,
               unname(stats::predict(logged$fit, sorted)))

  # Data to predict on need not carry the response: the bands are drawn
  # without the points
  p <- tl_plot_intervals(logged, new_data = mtcars[, c("wt", "hp")])
  expect_false(any(vapply(p$layers, function(l) inherits(l$geom, "GeomPoint"),
                          logical(1))))
  # Nor all the columns a computed response is made of: mpg alone cannot
  # give mpg / wt
  ratio <- tl_model(mtcars, I(mpg / wt) ~ hp, method = "linear")
  p <- tl_plot_intervals(ratio, new_data = mtcars[, c("mpg", "hp")])
  expect_false(any(vapply(p$layers, function(l) inherits(l$geom, "GeomPoint"),
                          logical(1))))

  # glmnet has no intervals to give, and failed asking for newx
  ridge <- tl_model(mtcars, mpg ~ wt + hp, method = "ridge")
  expect_error(tl_plot_intervals(ridge),
               "Interval plots need a \"linear\" or \"polynomial\" model")
})

test_that("interval plots leave out a row with no prediction, once", {
  # A missing predictor value left an NA in every layer, and ggplot warned
  # about the same row once per layer when the plot was drawn
  d <- mtcars
  d$wt[3] <- NA
  model <- tl_model(d, mpg ~ wt, method = "polynomial")
  expect_warning(
    p <- tl_plot_intervals(model),
    "1 row\\(s\\) with a missing predictor value are left out of the plot"
  )
  expect_equal(nrow(layer_of_geom(p, "GeomLine")), 31)
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  expect_no_warning(print(p))

  # A row missing only its response keeps its band and loses its point
  d2 <- mtcars
  d2$mpg[3] <- NA
  model2 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_no_warning(p2 <- tl_plot_intervals(model2, new_data = d2))
  expect_equal(nrow(layer_of_geom(p2, "GeomLine")), 32)
  expect_equal(nrow(layer_of_geom(p2, "GeomPoint")), 31)
})

# ---- regularisation plots --------------------------------------------

# Inches from the panel's left edge to the left end of each path label,
# for the plot drawn `width` x `height` inches: negative means clipped
path_label_room <- function(p, width = 7, height = 5) {
  grDevices::pdf(NULL, width = width, height = height)
  on.exit(grDevices::dev.off())
  built <- ggplot2::ggplot_build(p)
  gtab <- ggplot2::ggplot_gtable(built)
  panel_in <- width -
    sum(grid::convertWidth(gtab$widths, "in", valueOnly = TRUE))
  text <- layer_of_geom(p, "GeomText")
  x_range <- built$layout$panel_params[[1]]$x.range
  font <- grid::gpar(fontsize = text$size[1] * ggplot2::.pt)
  label_in <- vapply(text$label, function(label) {
    grob <- grid::textGrob(label, gp = font)
    grid::convertWidth(grid::grobWidth(grob), "in", valueOnly = TRUE)
  }, numeric(1))
  stats::setNames((text$x - x_range[1]) / diff(x_range) * panel_in - label_in,
                  text$label)
}

test_that("path labels fit inside the panel however long the names", {
  # The room left of the paths was a fixed share of the axis, so at 7 x 5
  # inches Speciesversicolor lost its first letters at the panel edge
  set.seed(1)
  long_names <- tl_model(iris, Sepal.Length ~ ., method = "lasso")
  room <- path_label_room(tl_plot_regularization_path(long_names))
  expect_true(all(room > 0), info = paste(names(room), collapse = ", "))

  # Short names keep the layout they had: the labels on mtcars still clear
  # the edge, and the left expansion is the 0.16 they always had
  set.seed(1)
  short_names <- tl_model(mtcars, mpg ~ ., method = "lasso")
  p <- tl_plot_regularization_path(short_names)
  expect_true(all(path_label_room(p) > 0))
  expect_equal(p$scales$get_scales("x")$expand[1:2], c(0.16, 0))
})

test_that("labelled paths are drawn in the accent colour", {
  # Unnamed scale values gave TRUE the first entry whenever no path was
  # FALSE, so with label_n or fewer predictors every path was grey and thin
  set.seed(1)
  few <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "lasso")
  lines <- layer_of_geom(tl_plot_regularization_path(few), "GeomLine")
  expect_identical(unique(lines$colour), "steelblue")
  expect_identical(unique(lines$linewidth), 1.2)
  expect_identical(unique(lines$alpha), 1)

  # With more predictors than labels, the rest stay grey and thin
  set.seed(1)
  many <- tl_model(mtcars, mpg ~ ., method = "lasso")
  p <- tl_plot_regularization_path(many, label_n = 2)
  lines <- layer_of_geom(p, "GeomLine")
  n_lambda <- length(many$fit$lambda)
  expect_equal(sum(lines$colour == "steelblue"), 2 * n_lambda)
  expect_equal(sum(lines$colour == "gray"), 8 * n_lambda)
  expect_true(all(lines$linewidth[lines$colour == "gray"] == 0.5))

  # A sequence of penalties marks one lambda.min and one lambda.1se, not
  # one dashed line per penalty
  set.seed(1)
  sequence <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                       lambda = c(1, 0.5, 0.1))
  p <- tl_plot_regularization_path(sequence)
  vlines <- which(vapply(p$layers, function(l) inherits(l$geom, "GeomVline"),
                         logical(1)))
  expect_length(vlines, 2)
  for (i in vlines) {
    expect_equal(nrow(ggplot2::layer_data(p, i)), 1)
  }
})

test_that("a regularisation path needs more than one penalty", {
  # Fitted at one lambda, every term was a single point, so no line was
  # drawn, under a subtitle naming a lambda.min and lambda.1se that no
  # cross-validation had chosen
  single <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                     lambda = 0.5)
  expect_error(tl_plot_regularization_path(single),
               "fitted at the single penalty lambda = 0.5")

  # A path with no cross-validation behind it, as a model fitted along a
  # sequence of penalties by an earlier version, marks no lambda.min or
  # lambda.1se
  set.seed(1)
  path_only <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso")
  attr(path_only$fit, "cv_results") <- NULL
  p <- tl_plot_regularization_path(path_only)
  expect_false(any(vapply(p$layers, function(l) inherits(l$geom, "GeomVline"),
                          logical(1))))
  expect_match(p$labels$subtitle, "No cross-validation")
})

test_that("the regularisation path draws a multiclass model class by class", {
  # coef() on a multinomial fit is a list, and the path failed with
  # "Tibble columns must have compatible sizes"
  set.seed(1)
  model <- tl_model(iris, Species ~ ., method = "lasso")
  p <- tl_plot_regularization_path(model)
  expect_setequal(as.character(unique(p$data$class)), levels(iris$Species))

  # Each class's path is glmnet's
  path <- as.matrix(stats::coef(model$fit)$versicolor)
  rows <- p$data[p$data$class == "versicolor" &
                   p$data$feature == "Petal.Width", ]
  expect_equal(rows$lambda, model$fit$lambda)
  expect_equal(rows$coefficient, unname(path["Petal.Width", ]))

  # One panel per class, each with its own labels
  built <- ggplot2::ggplot_build(p)
  expect_equal(nrow(built$layout$layout), 3)
  expect_equal(length(unique(layer_of_geom(p, "GeomText")$PANEL)), 3)
})

test_that("the cross-validation plot names the measure it shows", {
  # The label was "Binomial Deviance" for every classifier, a multinomial
  # one included, and "Mean Squared Error" for every regression
  multi <- tl_model(iris, Species ~ ., method = "lasso")
  expect_identical(tl_plot_regularization_cv(multi)$labels$y,
                   "Multinomial Deviance")

  set.seed(2)
  by_mae <- tl_model(mtcars, mpg ~ ., method = "lasso", type.measure = "mae")
  expect_identical(tl_plot_regularization_cv(by_mae)$labels$y,
                   unname(attr(by_mae$fit, "cv_results")$name))
  expect_identical(tl_plot_regularization_cv(by_mae)$labels$y,
                   "Mean Absolute Error")
})

# ---- classification plots --------------------------------------------

test_that("classification plots read classes from the model, not the data", {
  # The observed classes were read off the scored rows, so a test split of
  # iris[iris$Species != "setosa", ] still declaring setosa made the binary
  # model look multiclass, and a test factor with its levels reordered
  # switched the class the plots treated as positive
  skip_if_not_installed("glmnet")
  skip_if_not_installed("randomForest")
  iris2 <- iris[iris$Species != "setosa", ]
  split <- tl_split(iris2, prop = 0.7, seed = 1)
  dropped <- droplevels(split$test)
  reordered <- dropped
  reordered$Species <- factor(as.character(dropped$Species),
                              levels = c("virginica", "versicolor"))

  for (method in c("logistic", "lasso", "forest")) {
    set.seed(1)
    model <- tl_model(split$train, Species ~ Sepal.Length + Sepal.Width,
                      method = method)

    for (type in c("roc", "precision_recall", "calibration", "confusion")) {
      expected <- plot(model, type = type, new_data = dropped)
      expect_equal(plot(model, type = type, new_data = split$test)$data,
                   expected$data, info = paste(method, type))
      expect_equal(plot(model, type = type, new_data = reordered)$data,
                   expected$data, info = paste(method, type))
    }

    # virginica, the model's second class, is the positive one. By hand:
    # the AUC is the chance a virginica row outscores a versicolor row.
    prob <- predict(model, dropped, type = "prob")$virginica
    pos <- prob[dropped$Species == "virginica"]
    neg <- prob[dropped$Species == "versicolor"]
    auc <- mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
    expect_identical(
      plot(model, type = "roc", new_data = split$test)$labels$subtitle,
      paste0("AUC = ", round(auc, 3)),
      info = method
    )
    calibration <- plot(model, type = "calibration",
                        new_data = reordered)$data
    expect_equal(sum(calibration$frac_pos * calibration$n),
                 sum(dropped$Species == "virginica"), info = method)
  }
})

test_that("ROC and precision-recall need both classes among the rows", {
  # Decided from the data, one class present was reported as a multiclass
  # problem; ROCR itself says "Number of classes is not equal to 2"
  binary <- droplevels(iris[iris$Species != "setosa", ])
  model <- tl_model(binary, Species ~ Sepal.Length + Sepal.Width,
                    method = "logistic")
  one_class <- binary[binary$Species == "virginica", ]
  expect_error(plot(model, type = "roc", new_data = one_class),
               "needs rows of both classes")
  expect_error(plot(model, type = "precision_recall", new_data = one_class),
               "needs rows of both classes")
  # A calibration curve has something to show for one class
  expect_s3_class(plot(model, type = "calibration", new_data = one_class),
                  "ggplot")

  linear <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_error(plot(linear, type = "roc"),
               "only available for classification models")
})

test_that("classification plots leave incomplete rows out and count them", {
  # ROC and precision-recall stopped with ROCR's "'predictions' contains
  # NA", and the confusion counts summed to 30 of 32 with no message
  d <- transform(mtcars, am = factor(am))
  d$wt[c(3, 7)] <- NA
  model <- tl_model(d, am ~ wt + hp, method = "logistic")
  complete <- d[!is.na(d$wt), ]

  for (type in c("roc", "precision_recall", "calibration")) {
    expect_warning(
      p <- plot(model, type = type),
      "2 row\\(s\\) with a missing response or predicted probability"
    )
    expect_equal(p$data, plot(model, type = type, new_data = complete)$data,
                 info = type)
  }

  expect_warning(
    p <- plot(model, type = "confusion"),
    "2 row\\(s\\) with a missing response or prediction are left out"
  )
  by_hand <- table(complete$am,
                   predict(model, complete, type = "class")$.pred)
  expect_equal(p$data$Freq, as.vector(by_hand))
  expect_equal(sum(p$data$percentage), 100)

  # A missing response is counted the same way
  d2 <- transform(mtcars, am = factor(am))
  model2 <- tl_model(d2, am ~ wt + hp, method = "logistic")
  d2$am[5] <- NA
  expect_warning(plot(model2, type = "roc", new_data = d2), "1 row\\(s\\)")
})

test_that("a tree takes a whole control list", {
  model <- tl_model(iris, Species ~ ., method = "tree",
                    control = rpart::rpart.control(cp = 0.2, xval = 0))
  expect_equal(model$fit$control$cp, 0.2)
  expect_equal(model$fit$control$xval, 0)

  # an explicit argument still wins over the list
  model <- tl_model(iris, Species ~ ., method = "tree", cp = 0.05,
                    control = rpart::rpart.control(cp = 0.2))
  expect_equal(model$fit$control$cp, 0.05)
})
