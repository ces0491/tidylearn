# ---- Cloud model serialisation ----
#
# The cloud path ships a fitted model back from a remote R worker as
# bytes. tl_cloud_serialize_model() / tl_cloud_restore_model() are the
# two ends of that. Twelve methods go through base serialisation; 'deep'
# needs its keras slot lifted out first.

# keras needs more rows than mtcars provides to train at all.
set.seed(42)
deep_n <- 400
big_deep <- data.frame(
  wt = runif(deep_n, 1.5, 5.5),
  hp = runif(deep_n, 50, 340)
)
big_deep$mpg <- 37 - 4.5 * big_deep$wt - 0.02 * big_deep$hp +
  rnorm(deep_n, sd = 1.5)
big_deep_train <- big_deep[1:360, ]
big_deep_test <- big_deep[361:400, ]

test_that("a plain model round-trips through the cloud helpers", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  reference <- predict(model, new_data = mtcars[1:5, ])

  bundle <- tl_cloud_serialize_model(model)

  expect_type(bundle$payload, "raw")
  expect_null(bundle$keras_payload)

  restored <- tl_cloud_restore_model(bundle$payload, bundle$keras_payload)

  expect_equal(
    as.data.frame(predict(restored, new_data = mtcars[1:5, ])),
    as.data.frame(reference)
  )
})

test_that("xgboost round-trips without a separate payload", {
  testthat::skip_if_not_installed("xgboost")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost")
  reference <- predict(model, new_data = mtcars[1:5, ])

  bundle <- tl_cloud_serialize_model(model)
  expect_null(bundle$keras_payload)

  restored <- tl_cloud_restore_model(bundle$payload)

  expect_equal(
    as.data.frame(predict(restored, new_data = mtcars[1:5, ])),
    as.data.frame(reference)
  )
})

test_that("non-models are refused", {
  expect_error(tl_cloud_serialize_model(mtcars), "fitted tidylearn model")
  expect_error(tl_cloud_serialize_model(NULL), "fitted tidylearn model")
  expect_error(tl_cloud_restore_model("not raw"), "must be a raw vector")
  expect_error(
    tl_cloud_restore_model(serialize(1, NULL), keras_payload = "no"),
    "raw vector or NULL"
  )
})

test_that("tl_model_has_keras is false for pure-R models", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_false(tl_model_has_keras(model))

  # $fit is not a list at all for several methods.
  expect_false(tl_model_has_keras(list(fit = NULL)))
  expect_false(tl_model_has_keras(list(fit = list(model = NULL))))
})

# ---- 'deep' needs a TensorFlow backend, which CI does not provide ----

skip_if_no_tensorflow <- function() {
  testthat::skip_if_not_installed("keras")
  testthat::skip_if_not_installed("tensorflow")
  ok <- tryCatch(
    !is.null(tensorflow::tf_version()),
    error = function(e) FALSE
  )
  if (!isTRUE(ok)) {
    testthat::skip("No TensorFlow backend available")
  }
}

test_that("'deep' is detected as holding a Python object", {
  skip_if_no_tensorflow()

  model <- tl_model(big_deep_train, mpg ~ wt + hp, method = "deep",
                    epochs = 3, verbose = 0)

  expect_true(tl_model_has_keras(model))
})

test_that("'deep' round-trips through the cloud helpers", {
  skip_if_no_tensorflow()

  model <- tl_model(big_deep_train, mpg ~ wt + hp, method = "deep",
                    epochs = 3, verbose = 0)
  reference <- predict(model, new_data = big_deep_test)

  bundle <- tl_cloud_serialize_model(model)

  # The keras model must travel in its own payload, not the main one.
  expect_type(bundle$payload, "raw")
  expect_type(bundle$keras_payload, "raw")
  expect_gt(length(bundle$keras_payload), 1000)

  restored <- tl_cloud_restore_model(bundle$payload, bundle$keras_payload)

  expect_equal(
    as.data.frame(predict(restored, new_data = big_deep_test)),
    as.data.frame(reference)
  )
})

test_that("dropping the keras payload leaves an unusable model", {
  skip_if_no_tensorflow()

  model <- tl_model(big_deep_train, mpg ~ wt + hp, method = "deep",
                    epochs = 3, verbose = 0)

  bundle <- tl_cloud_serialize_model(model)

  # Restoring without the keras half must not quietly produce something
  # that looks like a working model.
  restored <- tl_cloud_restore_model(bundle$payload)
  expect_null(restored$fit$model)
  expect_error(predict(restored, new_data = big_deep_test))
})
