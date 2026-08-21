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
test_that("unsupervised prediction matches new_data columns by name", {
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  reference <- predict(model, new_data = iris[101:150, 1:4])$cluster

  # Column order must not change the answer
  shuffled <- iris[101:150, c(4, 2, 1, 3)]
  expect_identical(predict(model, new_data = shuffled)$cluster, reference)

  # Extra columns are ignored rather than widening the distance calculation
  extra <- iris[101:150, 1:4]
  extra$noise <- seq_len(nrow(extra))
  expect_identical(predict(model, new_data = extra)$cluster, reference)
})

test_that("a column the fit needs is an error, not a recycled distance", {
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  # Silently recycling a 3-column row against a 4-column centre returns a
  # cluster number that looks valid and is not
  expect_error(
    predict(model, new_data = iris[1:5, 1:3]),
    "missing: Petal.Width"
  )
  expect_silent(predict(model, new_data = iris[1:5, 1:4]))

  pca <- tl_model(iris[, 1:4], method = "pca")
  expect_error(
    predict(pca, new_data = iris[1:5, 1:2]),
    "missing: Petal.Length, Petal.Width"
  )
})

test_that("non-numeric columns are reported rather than coerced", {
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  bad <- iris[1:5, 1:4]
  bad$Petal.Width <- as.character(bad$Petal.Width)
  expect_error(predict(model, new_data = bad), "non-numeric: Petal.Width")
})

test_that("tl_reduce_dimensions projects new data to the components it kept", {
  split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)
  reduced <- tl_reduce_dimensions(
    split$train,
    response = "Species", method = "pca", n_components = 3
  )

  kept <- grep("^PC", names(reduced$data), value = TRUE)
  expect_identical(kept, c("PC1", "PC2", "PC3"))

  projected <- predict(
    reduced$reduction_model,
    new_data = split$test[, setdiff(names(split$test), "Species")]
  )

  # The projection has to be the same width as the data the downstream
  # model was trained on, or the two cannot be used together
  expect_identical(grep("^PC", names(projected), value = TRUE), kept)
})

test_that("a reduce-then-cluster workflow assigns test rows without warning", {
  split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)
  reduced <- tl_reduce_dimensions(
    split$train,
    response = "Species", method = "pca", n_components = 3
  )
  clustered <- tl_add_cluster_features(
    reduced$data,
    response = "Species", method = "kmeans", k = 3
  )

  test_pca <- predict(
    reduced$reduction_model,
    new_data = split$test[, setdiff(names(split$test), "Species")]
  )

  cluster_model <- attr(clustered, "cluster_model")
  expect_silent(assignments <- predict(cluster_model, new_data = test_pca))
  expect_true(all(assignments$cluster %in% seq_len(3)))
  expect_equal(nrow(assignments), nrow(split$test))
})

test_that("an unset component budget still returns every component", {
  model <- tl_model(iris[, 1:4], method = "pca")
  expect_null(model$spec$n_components)
  expect_identical(
    grep("^PC", names(predict(model, new_data = iris[1:5, 1:4])), value = TRUE),
    c("PC1", "PC2", "PC3", "PC4")
  )
})

test_that("prediction returns one row per observation, down to one", {
  # apply() simplifies a one-row result to a bare vector, and max.col() then
  # reads that as k rows of one column: three cluster numbers for a single
  # observation, returned without an error. Row counts are checked at every
  # size a caller can actually pass.
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  for (n in c(10, 2, 1)) {
    assignment <- predict(model, new_data = iris[seq_len(n), 1:4])
    expect_equal(nrow(assignment), n)
  }

  expect_equal(nrow(predict(model, new_data = iris[0, 1:4])), 0)

  pca <- tl_model(iris[, 1:4], method = "pca")
  expect_equal(nrow(predict(pca, new_data = iris[1, 1:4])), 1)
})

test_that("predicting one row at a time agrees with the whole frame", {
  # The property that catches any shape collapse, whichever function grows
  # one next: an observation's cluster cannot depend on what it was
  # submitted alongside.
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  batch <- predict(model, new_data = iris[, 1:4])$cluster
  one_at_a_time <- vapply(
    seq_len(nrow(iris)),
    function(i) predict(model, new_data = iris[i, 1:4])$cluster,
    integer(1)
  )

  expect_identical(one_at_a_time, batch)
})

test_that("a single row survives the reduce-then-cluster path", {
  reduced <- tl_reduce_dimensions(
    iris,
    response = "Species", method = "pca", n_components = 2
  )
  projected <- predict(reduced$reduction_model, new_data = iris[1, 1:4])
  expect_equal(nrow(projected), 1)
  expect_identical(grep("^PC", names(projected), value = TRUE), c("PC1", "PC2"))

  clustered <- tl_add_cluster_features(
    iris,
    response = "Species", method = "kmeans", k = 3
  )
  cluster_model <- attr(clustered, "cluster_model")
  expect_equal(nrow(predict(cluster_model, new_data = iris[1, 1:4])), 1)
})
