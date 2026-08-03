constant_frame <- function(data, value = 999) {
  data[] <- value
  data
}

test_that("PCA prediction projects new data rather than replaying scores", {
  model <- tl_model(iris[, 1:4], method = "pca")

  training <- predict(model)
  same_rows <- predict(model, new_data = iris[, 1:4])

  # Projecting the training data must reproduce the training scores
  expect_equal(
    unname(as.matrix(training[, -1])),
    unname(as.matrix(same_rows[, -1])),
    tolerance = 1e-8
  )

  # New data with the same row count must not be mistaken for training
  # data -- row count is not a reliable signal
  constant <- predict(model, new_data = constant_frame(iris[, 1:4]))
  expect_false(isTRUE(all.equal(
    as.data.frame(training), as.data.frame(constant)
  )))
})

test_that("kmeans prediction assigns new data to nearest centre", {
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  expect_identical(predict(model)$cluster, model$fit$clusters$cluster)

  # Identical rows must all land in the same cluster
  constant <- predict(model, new_data = constant_frame(iris[, 1:4]))
  expect_length(unique(constant$cluster), 1)
})

test_that("methods without out-of-sample projection say so", {
  pam_model <- tl_model(iris[, 1:4], method = "pam", k = 3)
  expect_equal(nrow(predict(pam_model)), 150)
  expect_error(
    predict(pam_model, new_data = constant_frame(iris[, 1:4])),
    "does not support out-of-sample prediction"
  )

  mds_model <- tl_model(iris[1:20, 1:4], method = "mds")
  expect_error(
    predict(mds_model, new_data = iris[21:40, 1:4]),
    "does not support out-of-sample prediction"
  )
})

test_that("hclust prediction points at tidy_cutree instead of returning NULL", {
  model <- tl_model(iris[1:30, 1:4], method = "hclust")

  expect_error(predict(model), "tidy_cutree")
})

test_that("plot() returns a ggplot for every unsupervised method", {
  fits <- list(
    pca    = tl_model(iris[, 1:4], method = "pca"),
    kmeans = tl_model(iris[, 1:4], method = "kmeans", k = 3),
    pam    = tl_model(iris[, 1:4], method = "pam", k = 3),
    clara  = tl_model(iris[, 1:4], method = "clara", k = 3),
    dbscan = tl_model(iris[, 1:4], method = "dbscan", eps = 0.5, minPts = 5),
    mds    = tl_model(iris[1:30, 1:4], method = "mds")
  )

  # ggplot objects are built without touching a device
  for (name in names(fits)) {
    expect_s3_class(plot(fits[[name]]), "ggplot")
  }
})

test_that("plot() draws a dendrogram for hclust models", {
  model <- tl_model(iris[1:30, 1:4], method = "hclust")

  # Dendrograms are base graphics, so draw to a null device rather than
  # leaving an Rplots.pdf behind
  draw <- function(x) {
    grDevices::pdf(NULL)
    on.exit(grDevices::dev.off(), add = TRUE)
    plot(x)
  }

  expect_s3_class(draw(model), "hclust")
})
