# ---- Cloud round trip: fitted models must survive serialisation ----
#
# `compute = "cloud"` fits on a remote R worker and ships the fitted model
# back as raw bytes. A model whose internals are an external pointer
# (xgboost) or a handle to a Python object (keras) can deserialise into an
# object that looks fine but cannot predict, so the round trip is asserted
# per method rather than assumed.
#
# `unserialize(serialize(x))` in one process is a genuine check here because
# the byte stream is self-contained -- see the payload-size test at the end
# of this file. If a backend ever stopped embedding its model in the stream,
# the restored object would carry a dangling pointer and predict() would
# fail, which is exactly what these tests catch.

reg_train <- mtcars[1:24, ]
reg_test <- mtcars[25:32, ]
reg_formula <- mpg ~ wt + hp

clf_data <- mtcars
clf_data$am <- factor(clf_data$am, levels = c(0, 1),
                      labels = c("auto", "manual"))
clf_train <- clf_data[1:24, ]
clf_test <- clf_data[25:32, ]
clf_formula <- am ~ wt + hp

# gbm needs more rows than a 24-row mtcars split provides.
set.seed(42)
big_n <- 400
big_data <- data.frame(
  wt = runif(big_n, 1.5, 5.5),
  hp = runif(big_n, 50, 340)
)
big_data$mpg <- 37 - 4.5 * big_data$wt - 0.02 * big_data$hp +
  rnorm(big_n, sd = 1.5)
big_train <- big_data[1:360, ]
big_test <- big_data[361:400, ]

# method -> the package that supplies its implementation
roundtrip_methods <- list(
  linear      = "stats",
  polynomial  = "stats",
  logistic    = "stats",
  tree        = "rpart",
  forest      = "randomForest",
  boost       = "gbm",
  ridge       = "glmnet",
  lasso       = "glmnet",
  elastic_net = "glmnet",
  svm         = "e1071",
  nn          = "nnet",
  xgboost     = "xgboost"
)

# randomForest resets mtry with only two predictors, so give it a wider
# formula rather than let the tests emit a warning.
wide_formula <- mpg ~ wt + hp + disp + drat

roundtrip_fixture <- function(method) {
  if (method == "logistic") {
    list(train = clf_train, test = clf_test, formula = clf_formula)
  } else if (method == "boost") {
    list(train = big_train, test = big_test, formula = reg_formula)
  } else if (method == "forest") {
    list(train = reg_train, test = reg_test, formula = wide_formula)
  } else {
    list(train = reg_train, test = reg_test, formula = reg_formula)
  }
}

for (method_name in names(roundtrip_methods)) {
  local({
    method <- method_name
    pkg <- roundtrip_methods[[method]]

    test_that(paste0("'", method, "' models survive a serialise round trip"), {
      skip_if_not_installed(pkg)

      fx <- roundtrip_fixture(method)
      model <- tl_model(fx$train, fx$formula, method = method)

      restored <- unserialize(serialize(model, connection = NULL))

      expect_s3_class(restored, class(model)[1])

      # The restored model must predict, and predict identically to the
      # model that was never serialised.
      reference <- predict(model, new_data = fx$test)
      restored_preds <- predict(restored, new_data = fx$test)

      expect_equal(
        as.data.frame(restored_preds),
        as.data.frame(reference)
      )
    })
  })
}

test_that("xgboost serialises its booster into the byte stream", {
  skip_if_not_installed("xgboost")

  # An xgb.Booster holds an external pointer. It round-trips only because
  # xgboost embeds the booster in the serialised stream; without that the
  # payload would collapse to a few KB of R-level metadata wrapping a null
  # pointer, and the restored model would fail to predict. Assert the
  # payload is substantial so a regression in that behaviour is caught
  # here rather than against a live Modal worker.
  model <- tl_model(reg_train, reg_formula, method = "xgboost")
  payload <- serialize(model, connection = NULL)

  expect_gt(length(payload), 20000)
})
