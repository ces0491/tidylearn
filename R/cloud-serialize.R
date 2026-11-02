#' @title Model serialisation for tidylearn cloud compute
#' @name tidylearn-cloud-serialize
#' @description Converting a fitted tidylearn model to bytes and back,
#'   so it can cross the boundary between a remote worker and the
#'   caller's R session.
NULL

# Twelve of the thirteen supervised methods survive base R serialisation
# intact, xgboost included -- its booster is embedded in the byte stream
# rather than left behind as a dangling pointer.
#
# 'deep' is the exception. A keras model is a reference to a Python
# object, so base serialisation produces an object that looks intact and
# then fails on predict.
#
# So the keras slot is lifted out and saved in Keras's own .keras archive
# format, and everything around it -- the scaling statistics, formula,
# factor levels -- goes through base serialisation as usual.

#' Serialise a fitted tidylearn model to raw bytes
#'
#' @param model A fitted tidylearn model.
#' @return A list with `payload` (raw vector) and, for keras-backed
#'   models, `keras_payload` (raw vector). `keras_payload` is `NULL` for
#'   every other method.
#' @keywords internal
#' @noRd
tl_cloud_serialize_model <- function(model) {
  if (!inherits(model, "tidylearn_model")) {
    stop("'model' must be a fitted tidylearn model.", call. = FALSE)
  }

  keras_payload <- NULL

  if (tl_model_has_keras(model)) {
    tl_check_packages("keras3")
    keras_payload <- tl_keras_to_raw(model$fit$model)
    model$fit$model <- NULL
  }

  list(
    payload = serialize(model, connection = NULL),
    keras_payload = keras_payload
  )
}

#' Restore a fitted tidylearn model from raw bytes
#'
#' @param payload Raw vector produced by `tl_cloud_serialize_model()`.
#' @param keras_payload Raw vector holding the keras model, or `NULL`.
#' @return The restored tidylearn model.
#' @keywords internal
#' @noRd
tl_cloud_restore_model <- function(payload, keras_payload = NULL) {
  if (!is.raw(payload)) {
    stop("'payload' must be a raw vector.", call. = FALSE)
  }

  model <- unserialize(payload)

  if (!is.null(keras_payload)) {
    if (!is.raw(keras_payload)) {
      stop("'keras_payload' must be a raw vector or NULL.", call. = FALSE)
    }
    tl_check_packages("keras3")
    model$fit$model <- tl_keras_from_raw(keras_payload)
  }

  model
}

#' Convert a keras model to raw bytes
#'
#' keras3 saves to a file only, so the bytes go through a temporary
#' `.keras` archive.
#'
#' @param model A keras model.
#' @return A raw vector.
#' @keywords internal
#' @noRd
tl_keras_to_raw <- function(model) {
  path <- tempfile(fileext = ".keras")
  on.exit(unlink(path), add = TRUE)

  keras3::save_model(model, path)
  readBin(path, what = "raw", n = file.size(path))
}

#' Restore a keras model from raw bytes
#'
#' @param payload Raw vector from `tl_keras_to_raw()`.
#' @return A keras model.
#' @keywords internal
#' @noRd
tl_keras_from_raw <- function(payload) {
  path <- tempfile(fileext = ".keras")
  on.exit(unlink(path), add = TRUE)

  writeBin(payload, path)
  keras3::load_model(path)
}

#' Does this model hold a Python object that needs special handling?
#'
#' Tests the object rather than the method name, so anything storing a
#' Python handle is caught however it got there.
#'
#' `python.builtin.object` is reticulate's marker on every Python object
#' and is the right thing to match on. The concrete keras classes are not:
#' keras renamed them from `keras.engine.*` to `keras.src.engine.*`
#' between versions, so matching those would silently stop detecting
#' models on one side of that change and hand back an object that fails
#' only when predicted from.
#'
#' @param model A fitted tidylearn model.
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_model_has_keras <- function(model) {
  fit <- model$fit

  if (!is.list(fit) || is.null(fit$model)) {
    return(FALSE)
  }

  inherits(fit$model, "python.builtin.object")
}
