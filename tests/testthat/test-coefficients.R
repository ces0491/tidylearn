# tl_coefficients() is the tibble the gt coefficient table formats, and the
# only route to a confidence interval on a tidylearn coefficient. The two
# things worth pinning are that the intervals are the ones they claim to be
# -- checked against confint() and against the reported standard error
# rather than against a recorded number -- and that a term the fit could
# not estimate still appears.

am_data <- transform(mtcars, am = factor(am))

test_that("tl_coefficients returns a row per term, no interval by default", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  coefs <- tl_coefficients(model)

  expect_s3_class(coefs, "tbl_df")
  expect_identical(
    names(coefs),
    c("term", "estimate", "std_error", "statistic", "p_value")
  )
  expect_identical(coefs$term, c("(Intercept)", "wt", "hp"))
  expect_equal(coefs$estimate, unname(stats::coef(model$fit)))
})

test_that("the linear interval is the one confint() gives for the lm", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  coefs <- tl_coefficients(model, conf_int = TRUE)

  expect_true(all(c("conf_low", "conf_high") %in% names(coefs)))

  reference <- stats::confint(model$fit, level = 0.95)
  expect_equal(coefs$conf_low, unname(reference[, 1]))
  expect_equal(coefs$conf_high, unname(reference[, 2]))

  narrower <- tl_coefficients(model, conf_int = TRUE, level = 0.8)
  expect_true(all(narrower$conf_high - narrower$conf_low <
                    coefs$conf_high - coefs$conf_low))
  reference_80 <- stats::confint(model$fit, level = 0.8)
  expect_equal(narrower$conf_low, unname(reference_80[, 1]))
})

test_that("the interval and the p-value agree about zero", {
  # The reason for computing a Wald interval rather than profiling the
  # likelihood: an interval excluding zero beside p > 0.05 in the same row
  # is the one output a coefficient table must never produce.
  model <- tl_model(mtcars, mpg ~ wt + hp + disp + drat, method = "linear")
  coefs <- tl_coefficients(model, conf_int = TRUE)

  excludes_zero <- coefs$conf_low > 0 | coefs$conf_high < 0
  expect_identical(excludes_zero, coefs$p_value < 0.05)
})

test_that("the logistic interval uses the z the summary reports", {
  model <- tl_model(am_data, am ~ wt, method = "logistic")
  coefs <- tl_coefficients(model, conf_int = TRUE)

  crit <- stats::qnorm(0.975)
  expect_equal(coefs$conf_low, coefs$estimate - crit * coefs$std_error)
  expect_equal(coefs$conf_high, coefs$estimate + crit * coefs$std_error)
  expect_identical(coefs$term, c("(Intercept)", "wt"))
})

test_that("exponentiate reports odds ratios and marks the error's scale", {
  model <- tl_model(am_data, am ~ wt, method = "logistic")
  log_odds <- tl_coefficients(model, conf_int = TRUE)
  odds <- tl_coefficients(model, conf_int = TRUE, exponentiate = TRUE)

  expect_equal(odds$estimate, exp(log_odds$estimate))
  expect_equal(odds$conf_low, exp(log_odds$conf_low))
  expect_equal(odds$conf_high, exp(log_odds$conf_high))

  # The standard error is not exponentiated, and says so in its name.
  expect_false("std_error" %in% names(odds))
  expect_equal(odds$std_error_log, log_odds$std_error)

  # The statistic and p-value are unchanged by the scale.
  expect_equal(odds$p_value, log_odds$p_value)
})

test_that("exponentiate is refused where coefficients are not log odds", {
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_error(
    tl_coefficients(model, exponentiate = TRUE),
    "log-odds scale"
  )
})

test_that("the exponentiate refusal names the response as written", {
  # For log(mpg) ~ wt it said the model was "of mpg on its own scale"
  logged <- tl_model(mtcars, log(mpg) ~ wt, method = "linear")
  expect_error(tl_coefficients(logged, exponentiate = TRUE),
               "'linear' models log\\(mpg\\) here, not the log odds")
})

test_that("a term the fit could not estimate is still a row", {
  collinear <- transform(mtcars, wt_doubled = wt * 2)
  model <- tl_model(collinear, mpg ~ wt + wt_doubled, method = "linear")

  # lm() drops the aliased term from summary()$coefficients entirely, so
  # reading the table from there loses a term named in the formula.
  expect_identical(
    rownames(summary(model$fit)$coefficients),
    c("(Intercept)", "wt")
  )

  coefs <- tl_coefficients(model, conf_int = TRUE)
  expect_identical(coefs$term, c("(Intercept)", "wt", "wt_doubled"))

  aliased <- coefs[coefs$term == "wt_doubled", ]
  expect_true(is.na(aliased$estimate))
  expect_true(is.na(aliased$std_error))
  expect_true(is.na(aliased$p_value))
  expect_true(is.na(aliased$conf_low))
})

test_that("two terms with the same name keep their own statistics", {
  # A factor `a` with level "b" makes the term ab, which is also the name of
  # a numeric column. The statistics were matched to terms by name, so both
  # summary() rows landed on the first ab and the second had none.
  set.seed(3)
  d <- data.frame(
    y = stats::rnorm(40),
    a = factor(sample(c("x", "b"), 40, TRUE), levels = c("x", "b")),
    ab = stats::rnorm(40)
  )
  model <- tl_model(d, y ~ a + ab, method = "linear")
  coefs <- tl_coefficients(model, conf_int = TRUE)
  reference <- summary(model$fit)$coefficients

  expect_identical(coefs$term, c("(Intercept)", "ab", "ab"))
  expect_equal(coefs$estimate, unname(reference[, "Estimate"]))
  expect_equal(coefs$std_error, unname(reference[, "Std. Error"]))
  expect_equal(coefs$statistic, unname(reference[, "t value"]))
  expect_equal(coefs$p_value, unname(reference[, "Pr(>|t|)"]))
  # By hand: confint() looks the standard errors up by name as well, so it
  # gives the second ab the first one's interval
  crit <- stats::qt(0.975, df = stats::df.residual(model$fit))
  by_hand <- reference[, "Estimate"] - crit * reference[, "Std. Error"]
  expect_equal(coefs$conf_low, unname(by_hand))

  # An aliased term among them still takes no statistics of its own
  d$ab_doubled <- d$ab * 2
  aliased <- tl_coefficients(tl_model(d, y ~ a + ab + ab_doubled,
                                      method = "linear"))
  expect_identical(aliased$term, c("(Intercept)", "ab", "ab", "ab_doubled"))
  expect_equal(aliased$std_error[1:3], unname(reference[, "Std. Error"]))
  expect_true(is.na(aliased$std_error[4]))

  skip_if_not_installed("gt")
  tbl <- tl_table_coefficients(model)[["_data"]]
  expect_equal(tbl$std_error, unname(reference[, "Std. Error"]))
})

test_that("regularised models report the penalty and no interval", {
  model <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso")
  coefs <- tl_coefficients(model)

  expect_identical(names(coefs), c("term", "estimate", "lambda"))
  expect_identical(coefs$lambda, rep(attr(model$fit, "lambda_1se"),
                                     nrow(coefs)))

  at_min <- tl_coefficients(model, lambda = "min")
  expect_identical(at_min$lambda[[1]], attr(model$fit, "lambda_min"))

  at_value <- tl_coefficients(model, lambda = 0.5)
  expect_identical(at_value$lambda[[1]], 0.5)
})

test_that("regularised coefficients are those of the fit cv.glmnet() scored", {
  # The model was a second glmnet() run along cv.glmnet()'s penalties. It
  # agreed with cv.glmnet()'s own fit only to the solver's tolerance, so on
  # mtcars the predictions at lambda_1se differed by up to 0.004.
  set.seed(1)
  model <- tl_model(mtcars, mpg ~ ., method = "lasso")
  cv <- attr(model$fit, "cv_results")

  expect_equal(tl_coefficients(model)$estimate,
               as.vector(stats::coef(cv, s = "lambda.1se")))
  expect_equal(tl_coefficients(model, lambda = "min")$estimate,
               as.vector(stats::coef(cv, s = "lambda.min")))
  x <- as.matrix(mtcars[, -1])
  expect_equal(predict(model, mtcars)$.pred,
               as.vector(stats::predict(cv, newx = x, s = "lambda.1se")))
})

test_that("a penalty that keeps no predictor says it dropped them all", {
  # When cross-validation chooses the largest penalty, every slope is zero.
  # The refit at that penalty left one at about -6e-17, which importance
  # rescaled to 100 and the coefficient table did not grey out.
  set.seed(3)
  d <- data.frame(y = stats::rnorm(40), a = stats::rnorm(40),
                  b = stats::rnorm(40), c = stats::rnorm(40))
  model <- tl_model(d, y ~ ., method = "lasso")
  # the intercept-only model is the one chosen here
  expect_equal(attr(model$fit, "lambda_1se"), max(model$fit$lambda))

  coefs <- tl_coefficients(model)
  expect_identical(coefs$estimate[coefs$term != "(Intercept)"], c(0, 0, 0))
  expect_error(tl_plot_importance_regularized(model),
               "the penalty dropped every predictor")

  skip_if_not_installed("gt")
  expect_error(tl_table_importance(model),
               "the penalty dropped every predictor")
})

test_that("an interval on a shrunk coefficient is refused, not faked", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
  expect_error(
    tl_coefficients(model, conf_int = TRUE),
    "no standard"
  )
})

test_that("tl_coefficients rejects a bad lambda rather than the whole path", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")

  expect_error(tl_coefficients(model, lambda = "1SE"), "non-negative")
  expect_error(tl_coefficients(model, lambda = c(0.1, 0.2)), "non-negative")
  expect_error(tl_coefficients(model, lambda = -1), "non-negative")
})

test_that("methods without coefficients say what to reach for instead", {
  forest <- tl_model(mtcars, mpg ~ wt + hp, method = "forest")
  expect_error(tl_coefficients(forest), "importance")

  pca <- tl_model(mtcars[, c("mpg", "wt", "hp")], ~ ., method = "pca")
  expect_error(tl_coefficients(pca), "loadings")

  expect_error(tl_coefficients(mtcars), "tidylearn_model")
})

test_that("tl_coefficients validates its own arguments", {
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")

  expect_error(tl_coefficients(model, conf_int = NA), "TRUE or FALSE")
  expect_error(tl_coefficients(model, exponentiate = "yes"), "TRUE or FALSE")
  expect_error(tl_coefficients(model, conf_int = TRUE, level = 95),
               "between 0 and 1")
  expect_error(tl_coefficients(model, conf_int = TRUE, level = 0),
               "between 0 and 1")
  # Checked without an interval too, as documented
  expect_error(tl_coefficients(model, level = 95), "between 0 and 1")
})

test_that("the gt table formats the same numbers the tibble carries", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table_coefficients(model, conf_int = TRUE)
  expect_s3_class(tbl, "gt_tbl")

  coefs <- tl_coefficients(model, conf_int = TRUE)
  expect_equal(tbl[["_data"]]$conf_low, coefs$conf_low)
  expect_equal(tbl[["_data"]]$estimate, coefs$estimate)

  # An aliased term has no p-value, and ifelse() would put an NA in the
  # flag column next to a star and a blank.
  collinear <- transform(mtcars, wt_doubled = wt * 2)
  aliased <- tl_table_coefficients(
    tl_model(collinear, mpg ~ wt + wt_doubled, method = "linear")
  )[["_data"]]
  expect_identical(aliased$term, c("(Intercept)", "wt", "wt_doubled"))
  expect_identical(aliased$significant, c("*", "*", ""))

  lasso <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso")
  expect_s3_class(tl_table_coefficients(lasso), "gt_tbl")
  expect_error(tl_table_coefficients(lasso, conf_int = TRUE), "no standard")
})

test_that("a multiclass regularised model gives numbers per class", {
  # coef() on a multinomial glmnet is a list of sparse matrices, which
  # became a column of dgCMatrix objects
  model <- tl_model(iris, Species ~ ., method = "ridge")
  coefs <- tl_coefficients(model)

  expect_identical(names(coefs), c("class", "term", "estimate", "lambda"))
  expect_type(coefs$estimate, "double")
  expect_equal(nrow(coefs), 3 * 5)
  expect_setequal(coefs$class, levels(iris$Species))

  reference <- coef(model$fit, s = attr(model$fit, "lambda_1se"))$setosa
  expect_equal(coefs$estimate[coefs$class == "setosa"],
               as.vector(as.matrix(reference)))

  # There is no reference class, so no odds ratio to report
  expect_error(tl_coefficients(model, exponentiate = TRUE), "multiclass")

  skip_if_not_installed("gt")
  expect_s3_class(tl_table_coefficients(model), "gt_tbl")
})

test_that("a penalty outside the fitted path is refused, not relabelled", {
  model <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso")
  path <- range(model$fit$lambda)
  expect_error(tl_coefficients(model, lambda = 0), "outside the fitted path")
  expect_error(tl_coefficients(model, lambda = path[2] * 10),
               "outside the fitted path")
  # both ends of the path are still accepted
  expect_equal(tl_coefficients(model, lambda = path[1])$lambda[[1]], path[1])
  expect_equal(tl_coefficients(model, lambda = path[2])$lambda[[1]], path[2])

  single <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                     lambda = 0.5)
  expect_error(tl_coefficients(single, lambda = 0.01),
               "outside the fitted path")
  expect_equal(tl_coefficients(single, lambda = 0.5)$lambda[[1]], 0.5)
})

test_that("an odds-ratio table ranks terms by the size of the log odds", {
  skip_if_not_installed("gt")
  bin <- transform(mtcars, am = factor(am))
  model <- tl_model(bin, am ~ wt + hp + qsec + drat, method = "lasso")
  data <- tl_table_coefficients(model, exponentiate = TRUE,
                                lambda = "min")[["_data"]]
  log_size <- abs(log(data$estimate))
  expect_false(is.unsorted(rev(log_size)))
  # a dropped term (odds ratio exactly 1) sorts last
  expect_equal(data$estimate[nrow(data)], 1)
})

test_that("a misspelt argument to tl_coefficients() is an error", {
  # broom spells it conf.int; ... used to swallow it and return no interval
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_error(tl_coefficients(model, conf.int = TRUE), "unused argument")
  skip_if_not_installed("gt")
  expect_warning(tl_table_coefficients(model, conf.int = TRUE),
                 "conf.int")
})

test_that("an empty interaction cell is the aliased row the docs describe", {
  d <- data.frame(
    y = c(1, 2, 3, 4, 5, 6, 7, 8.5),
    f = factor(c("a", "a", "b", "b", "a", "a", "b", "b")),
    g = factor(c("u", "v", "u", "u", "u", "v", "u", "u"))
  )
  coefs <- tl_coefficients(tl_model(d, y ~ f * g, method = "linear"))
  expect_true("fb:gv" %in% coefs$term)
  expect_true(is.na(coefs$estimate[coefs$term == "fb:gv"]))
})

test_that("a sequence of penalties is cross-validated down to one", {
  # Fitted as a path with no penalty chosen, the model could not predict:
  # predict() and tl_evaluate() failed asking for a lambda that predict()
  # does not take
  set.seed(1)
  model <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                    lambda = c(1, 0.1))
  cv <- attr(model$fit, "cv_results")
  expect_equal(model$fit$lambda, c(1, 0.1))
  expect_identical(attr(model$fit, "lambda_1se"), cv$lambda.1se)
  expect_identical(attr(model$fit, "lambda_min"), cv$lambda.min)

  x <- as.matrix(mtcars[, c("wt", "hp", "disp")])
  preds <- as.vector(stats::predict(cv, newx = x, s = "lambda.1se"))
  expect_equal(predict(model, mtcars)$.pred, preds)
  expect_equal(tl_coefficients(model)$estimate,
               as.vector(stats::coef(cv, s = "lambda.1se")))
  rmse <- tl_evaluate(model, metrics = "rmse")
  expect_equal(rmse$value[rmse$metric == "rmse"],
               sqrt(mean((mtcars$mpg - preds)^2)))
})

test_that("a model that chose no single penalty says so", {
  # A model fitted at several penalties before they were cross-validated
  # stored all of them as its "1se" penalty
  model <- tl_model(mtcars, mpg ~ wt + hp + disp, method = "lasso",
                    lambda = 0.5)
  attr(model$fit, "lambda_1se") <- c(1, 0.1)
  expect_error(tl_coefficients(model), "several penalties")
  expect_equal(tl_coefficients(model, lambda = 0.5)$lambda[[1]], 0.5)
})

test_that("predictions come from the coefficients tl_coefficients() reports", {
  # predict() used lambda_min while the coefficients defaulted to
  # lambda_1se, so the numbers on display did not produce the predictions
  set.seed(1)
  model <- tl_model(mtcars, mpg ~ wt + hp + disp + drat + qsec,
                    method = "lasso")
  expect_false(isTRUE(all.equal(attr(model$fit, "lambda_min"),
                                attr(model$fit, "lambda_1se"))))

  coefs <- tl_coefficients(model)
  design <- cbind(1, as.matrix(mtcars[coefs$term[-1]]))
  by_hand <- as.vector(design %*% coefs$estimate)
  expect_equal(predict(model, mtcars)$.pred, by_hand)
})
