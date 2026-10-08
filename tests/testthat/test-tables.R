test_that("tl_table dispatches correctly for supervised models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table dispatches correctly for unsupervised models", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "pca")
  tbl <- tl_table(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table rejects non-tidylearn objects", {
  skip_if_not_installed("gt")

  expect_error(tl_table(lm(mpg ~ wt, data = mtcars)), "tidylearn_model")
})

test_that("tl_table_metrics returns gt_tbl", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table_metrics(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_coefficients works for linear models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tbl <- tl_table_coefficients(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_coefficients works for regularised models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
  tbl <- tl_table_coefficients(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_coefficients errors for unsupported methods", {
  skip_if_not_installed("gt")

  model <- tl_model(iris, Species ~ ., method = "forest")
  expect_error(tl_table_coefficients(model), "importance")
})

test_that("tl_table_confusion works for classification", {
  skip_if_not_installed("gt")

  model <- tl_model(iris, Species ~ ., method = "forest")
  tbl <- tl_table_confusion(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_confusion errors for regression", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_table_confusion(model), "classification")
})

test_that("tl_table_importance works for tree-based models", {
  skip_if_not_installed("gt")

  model <- tl_model(iris, Species ~ ., method = "forest")
  tbl <- tl_table_importance(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("the importance table says when a tree has no splits", {
  skip_if_not_installed("gt")
  # A single-node tree has no variable importance. The table failed with
  # "Column `importance` not found in `.data`"
  stump <- tl_model(mtcars, mpg ~ wt + hp, method = "tree", cp = 1)
  expect_error(tl_table_importance(stump),
               "No feature has non-zero importance: the tree has no splits")
  # A tree that splits still gets its table
  tree <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")
  expect_s3_class(tl_table_importance(tree), "gt_tbl")
})

test_that("tl_table_variance works for PCA", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "pca")
  tbl <- tl_table_variance(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_variance errors for non-PCA models", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  expect_error(tl_table_variance(model), "PCA")
})

test_that("tl_table_loadings works for PCA", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "pca")
  tbl <- tl_table_loadings(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_clusters works for kmeans", {
  skip_if_not_installed("gt")

  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
  tbl <- tl_table_clusters(model)
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table_comparison requires at least 2 models", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_table_comparison(model), "at least 2")
})

test_that("tl_table_comparison works with multiple models", {
  skip_if_not_installed("gt")

  m1 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
  tbl <- tl_table_comparison(m1, m2, names = c("Linear", "Lasso"))
  expect_s3_class(tbl, "gt_tbl")
})

test_that("tl_table errors for unknown type", {
  skip_if_not_installed("gt")

  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  expect_error(tl_table(model, type = "nonexistent"), "Unknown table type")
})

test_that("two models of one method get separate comparison columns", {
  skip_if_not_installed("gt")

  # Both default to "linear (reg)", which pivoted into list cells
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "linear")
  data <- tl_table_comparison(m1, m2)[["_data"]]

  expect_equal(ncol(data), 3)
  expect_true(all(vapply(data[-1], is.numeric, logical(1))))

  expect_error(tl_table_comparison(m1, m2, names = "a"), "must match")
  expect_error(tl_table_comparison(m1, m2, names = c("a", "a")), "unique")
})

test_that("the comparison table needs new_data for models fitted apart", {
  skip_if_not_installed("gt")
  # With no new_data every model was scored on the first model's training
  # rows: a model fitted on other rows was scored partly on rows it never
  # saw, without a word
  early <- tl_model(mtcars[1:20, ], mpg ~ wt, method = "linear")
  late <- tl_model(mtcars[13:32, ], mpg ~ wt, method = "linear")
  expect_error(
    tl_table_comparison(early, late, names = c("early", "late")),
    "fitted on different data.*'late'.*Pass the rows to compare them on"
  )

  # Given the rows, both are scored on them
  data <- tl_table_comparison(early, late, names = c("early", "late"),
                              new_data = mtcars)[["_data"]]
  rmse <- function(m) sqrt(mean((mtcars$mpg - predict(m, mtcars)$.pred)^2))
  expect_equal(data$early[data$metric == "Rmse"], rmse(early))
  expect_equal(data$late[data$metric == "Rmse"], rmse(late))
})

test_that("the comparison table scores engineered features on the raw rows", {
  skip_if_not_installed("gt")
  # A model fitted on PCA scores, as tl_auto_ml() builds its candidates,
  # stores the scores, and predict() rebuilds them from raw rows. It was
  # refused beside a tree fitted on those raw rows, which 0.5.0 compared.
  # The raw rows the tree stored score both models, in either order.
  ib <- droplevels(iris[iris$Species != "setosa", ])
  reduced <- tl_reduce_dimensions(ib, response = "Species", method = "pca",
                                  n_components = 2)
  pca <- tl_model(reduced$data, Species ~ PC1 + PC2, method = "logistic")
  pca$feature_transform <- list(
    kind = "pca", reduction_model = reduced$reduction_model,
    response = "Species"
  )
  tree <- tl_model(ib, Species ~ ., method = "tree")
  accuracy <- function(m) {
    mean(predict(m, ib, type = "class")$.pred == ib$Species)
  }

  data <- tl_table_comparison(tree, pca, names = c("tree", "pca"))[["_data"]]
  expect_equal(data$tree, accuracy(tree))
  expect_equal(data$pca, accuracy(pca))
  data <- tl_table_comparison(pca, tree, names = c("pca", "tree"))[["_data"]]
  expect_equal(data$tree, accuracy(tree))
  expect_equal(data$pca, accuracy(pca))

  # A cluster candidate stores the rows it was fitted on beside its cluster
  # column, so it supplies the rows when no model was fitted on raw columns,
  # as it did in 0.5.0
  km <- tl_model(ib[, 1:4], method = "kmeans", k = 2)
  with_clusters <- ib
  with_clusters$cluster_kmeans <- factor(
    predict(km, new_data = ib[, 1:4])$cluster
  )
  clustered <- tl_model(with_clusters, Species ~ ., method = "tree")
  clustered$feature_transform <- list(
    kind = "cluster", cluster_model = km, column = "cluster_kmeans",
    levels = levels(with_clusters$cluster_kmeans), response = "Species"
  )
  data <- tl_table_comparison(clustered, pca,
                              names = c("clustered", "pca"))[["_data"]]
  expect_equal(data$clustered, accuracy(clustered))
  expect_equal(data$pca, accuracy(pca))

  # Those rows are checked like any model's
  late_km <- tl_model(ib[26:75, 1:4], method = "kmeans", k = 2)
  late <- with_clusters[26:75, ]
  late_clustered <- tl_model(late, Species ~ ., method = "tree")
  late_clustered$feature_transform <- list(
    kind = "cluster", cluster_model = late_km, column = "cluster_kmeans",
    levels = levels(late$cluster_kmeans), response = "Species"
  )
  expect_error(
    tl_table_comparison(tree, late_clustered, names = c("tree", "late")),
    "fitted on different data.*'late'"
  )

  # Models fitted on PCA scores alone store no rows to compare them on, so
  # the rows have to be passed
  pca_tree <- tl_model(reduced$data, Species ~ PC1 + PC2, method = "tree")
  pca_tree$feature_transform <- pca$feature_transform
  expect_error(
    tl_table_comparison(pca, pca_tree),
    "None of the models stores the rows it was fitted on.*'new_data'"
  )
  expect_s3_class(tl_table_comparison(pca, pca_tree, new_data = ib), "gt_tbl")
})

test_that("the importance note counts the rows xgboost, svm and nn fitted", {
  # nobs() has no method for these fits, so the count fell back to every
  # stored row: 153 for airquality, where xgboost trains on the 116 rows
  # with an Ozone value and svm and nn on the 111 complete ones
  vars <- c("Ozone", "Temp", "Wind", "Solar.R")
  complete <- sum(stats::complete.cases(airquality[, vars]))
  ozone <- Ozone ~ Temp + Wind + Solar.R

  svm <- tl_model(airquality, ozone, method = "svm")
  expect_identical(tl_fit_rows(svm), complete)
  set.seed(1)
  nn <- tl_model(airquality, ozone, method = "nn", size = 2, trace = FALSE,
                 linout = TRUE)
  expect_identical(tl_fit_rows(nn), complete)

  skip_if_not_installed("xgboost")
  xgb <- tl_model(airquality, ozone, method = "xgboost", nrounds = 5)
  expect_identical(tl_fit_rows(xgb), sum(!is.na(airquality$Ozone)))
  skip_if_not_installed("gt")
  expect_match(tl_table_importance(xgb)[["_source_notes"]][[1]], "n = 116$")
})

test_that("the row count sees missing weights and svm with no fitted values", {
  # An svm fitted with fitted = FALSE keeps no fitted values, and the count
  # fell back to all 153 stored rows, where svm used the 111 complete ones
  vars <- c("Ozone", "Temp", "Wind", "Solar.R")
  svm <- tl_model(airquality, Ozone ~ Temp + Wind + Solar.R, method = "svm",
                  fitted = FALSE)
  expect_length(svm$fit$fitted, 0L)
  expect_identical(tl_fit_rows(svm),
                   sum(stats::complete.cases(airquality[, vars])))
  expect_identical(tl_fit_rows(svm),
                   nrow(airquality) - length(svm$fit$na.action))

  # A column the formula subtracts is not one svm needed complete
  minus <- tl_model(airquality, Ozone ~ . - Solar.R, method = "svm",
                    fitted = FALSE)
  used <- setdiff(names(airquality), "Solar.R")
  expect_identical(tl_fit_rows(minus),
                   sum(stats::complete.cases(airquality[, used])))

  # xgboost leaves out a row missing its weight, which the count did not
  # see: 116 rows with an Ozone value, where 108 also have a weight
  skip_if_not_installed("xgboost")
  w <- rep(1, nrow(airquality))
  w[1:10] <- NA
  xgb <- tl_model(airquality, Ozone ~ Temp + Wind, method = "xgboost",
                  weights = w, nrounds = 5)
  expect_identical(tl_fit_rows(xgb), sum(!is.na(airquality$Ozone) & !is.na(w)))
  expect_identical(
    tl_fit_rows(xgb),
    length(tl_xgb_training_rows(Ozone ~ Temp + Wind, airquality, w)$y)
  )

  # The count is recorded on the booster, so a model read back from disk
  # keeps it
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(xgb, path)
  expect_identical(tl_fit_rows(readRDS(path)), tl_fit_rows(xgb))

  # A model saved without the count, as an earlier version saved it,
  # counts the rows with a response instead of failing
  unweighted <- tl_model(airquality, Ozone ~ Temp + Wind, method = "xgboost",
                         nrounds = 5)
  attr(unweighted$fit, "training_rows") <- NULL
  expect_identical(tl_fit_rows(unweighted), sum(!is.na(airquality$Ozone)))
})

test_that("models fitted on one frame still share it as the default", {
  skip_if_not_installed("gt")
  # tl_model() makes a text column a factor only where its formula uses
  # the column, so the two models store the frame with gear as a factor and
  # as text. They were fitted on the same rows, and are compared on them.
  cars <- transform(mtcars, gear = as.character(gear))
  with_gear <- tl_model(cars, mpg ~ wt + gear, method = "linear")
  without <- tl_model(cars, mpg ~ wt + hp, method = "linear")
  data <- tl_table_comparison(with_gear, without,
                              names = c("a", "b"))[["_data"]]
  rmse <- function(m) sqrt(mean((cars$mpg - predict(m, cars)$.pred)^2))
  expect_equal(data$a[data$metric == "Rmse"], rmse(with_gear))
  expect_equal(data$b[data$metric == "Rmse"], rmse(without))
})

test_that("comparison names must not be missing", {
  skip_if_not_installed("gt")
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ hp, method = "linear")
  expect_error(tl_table_comparison(m1, m2, names = c("a", NA)),
               "no missing values")
})

test_that("the confusion table says when rows are missing the response", {
  skip_if_not_installed("gt")
  model <- tl_model(iris, Species ~ ., method = "forest", ntree = 50)
  d <- iris
  d$Species[1:5] <- NA
  expect_warning(tl_table_confusion(model, new_data = d), "5 row")
})

test_that("the confusion table reads the classes the model was trained on", {
  skip_if_not_installed("gt")
  # A test split of a subset still declares the class the subset dropped,
  # which became a row of zeros in a two-class model's matrix
  ib <- iris[iris$Species != "setosa", ]
  split <- tl_split(ib, prop = 0.7, seed = 1)
  model <- tl_model(split$train, Species ~ Sepal.Width + Petal.Length,
                    method = "logistic")
  data <- tl_table_confusion(model, new_data = split$test)[["_data"]]
  classes <- c("versicolor", "virginica")
  expect_identical(data$Actual, classes)
  expect_identical(setdiff(names(data), "Actual"), classes)

  # The counts are table() on the model's classes
  predicted <- predict(model, split$test, type = "class")$.pred
  expected <- table(factor(as.character(split$test$Species), levels = classes),
                    predicted)
  for (cls in classes) {
    expect_equal(data[[cls]], as.vector(expected[, cls]))
  }

  # Rows follow the model's class order whatever the data's level order
  relevelled <- split$test
  relevelled$Species <- factor(as.character(relevelled$Species),
                               levels = rev(classes))
  expect_identical(
    tl_table_confusion(model, new_data = relevelled)[["_data"]]$Actual,
    classes
  )
})

test_that("the confusion table leaves out a class the model never saw", {
  skip_if_not_installed("gt")
  ib <- droplevels(iris[iris$Species != "setosa", ])
  model <- tl_model(ib, Species ~ Sepal.Width + Petal.Length,
                    method = "logistic")
  expect_warning(
    tbl <- tl_table_confusion(model, new_data = iris[c(1:5, 51:150), ]),
    "5 row\\(s\\) belong to a class the model was not trained on \\(setosa\\)"
  )
  data <- tbl[["_data"]]
  expect_identical(data$Actual, c("versicolor", "virginica"))
  expect_equal(sum(data$versicolor, data$virginica), 100)
})

test_that("the confusion table names a response column the data lacks", {
  skip_if_not_installed("gt")
  am <- transform(mtcars, am = factor(am))
  model <- tl_model(am, am ~ wt, method = "logistic")
  expect_error(tl_table_confusion(model, new_data = am[, c("wt", "hp")]),
               "Response variable 'am' not found in the evaluation data")
})

test_that("table source notes count the rows the table describes", {
  skip_if_not_installed("gt")
  # Every note reported nrow(model$data). Coefficients for Ozone ~ Temp +
  # Wind said n = 153 where lm used 116, and tables scored on a ten-row
  # test set said n = 22, the training rows.
  note <- function(tbl) tbl[["_source_notes"]][[1]]

  ozone <- tl_model(airquality, Ozone ~ Temp + Wind, method = "linear")
  expect_identical(stats::nobs(ozone$fit), 116L)
  expect_match(note(tl_table_coefficients(ozone)), "n = 116$")
  # Scoring the training rows leaves out the 37 with no Ozone
  expect_match(note(tl_table_metrics(ozone)), "n = 116$")

  lasso <- tl_model(airquality, Ozone ~ Temp + Wind, method = "lasso")
  expect_match(note(tl_table_coefficients(lasso)), "n = 116$")
  expect_match(note(tl_table_importance(lasso)), "n = 116$")

  # rpart keeps a row missing a predictor and drops one missing the response
  tree <- tl_model(airquality, Ozone ~ Temp + Wind + Solar.R, method = "tree")
  expect_identical(tree$fit$frame$n[1], 116L)
  expect_match(note(tl_table_importance(tree)), "n = 116$")

  split <- tl_split(mtcars, prop = 0.7, seed = 1)
  expect_identical(nrow(split$test), 10L)
  linear <- tl_model(split$train, mpg ~ wt + hp, method = "linear")
  expect_match(note(tl_table_metrics(linear, new_data = split$test)),
               "n = 10$")

  am <- transform(mtcars, am = factor(am))
  am_split <- tl_split(am, prop = 0.7, seed = 1)
  logistic <- tl_model(am_split$train, am ~ wt, method = "logistic")
  expect_match(note(tl_table_confusion(logistic, new_data = am_split$test)),
               paste0("n = ", nrow(am_split$test), "$"))

  # The comparison counts the rows each model scored
  m1 <- tl_model(airquality, Ozone ~ Temp, method = "linear")
  m2 <- tl_model(airquality, Ozone ~ Temp + Solar.R, method = "linear")
  expect_identical(
    note(tl_table_comparison(m1, m2, names = c("a", "b"))),
    "tidylearn | n = 116 (a), 111 (b)"
  )
  expect_identical(
    note(tl_table_comparison(ozone, m1, names = c("a", "b"))),
    "tidylearn | n = 116"
  )
})

test_that("forest importance works without permutation importance", {
  model <- tl_model(iris, Species ~ ., method = "forest", ntree = 50,
                    importance = FALSE)
  imp <- tl_extract_importance(model)
  expect_setequal(imp$feature, names(iris)[1:4])
  expect_equal(max(imp$importance), 100)
  reg <- tl_model(mtcars, mpg ~ ., method = "forest", ntree = 50,
                  importance = FALSE)
  expect_equal(max(tl_extract_importance(reg)$importance), 100)
})

test_that("the dbscan cluster table does not count noise as a cluster", {
  skip_if_not_installed("gt")
  model <- tl_model(iris[, 1:4], method = "dbscan", eps = 0.4, minPts = 5)
  subtitle <- tl_table_clusters(model)[["_heading"]]$subtitle
  expect_match(subtitle, paste(model$fit$n_clusters, "clusters"))
  expect_match(subtitle, "noise")
})

test_that("cluster tables summarise only the columns the fit used", {
  skip_if_not_installed("gt")
  # The dbscan and hclust tables averaged every numeric column of the data,
  # though the formula named two; the kmeans table showed only those two
  db <- tl_model(iris[, 1:4], ~ Sepal.Length + Sepal.Width,
                 method = "dbscan", eps = 0.3, minPts = 4)
  data <- tl_table_clusters(db)[["_data"]]
  expect_named(data, c("cluster", "size", "Sepal.Length", "Sepal.Width"))
  assigned <- db$fit$clusters$cluster
  by_cluster <- as.character(data$cluster)
  expect_equal(data$size, as.vector(table(assigned)[by_cluster]))
  expect_equal(
    data$Sepal.Width,
    as.vector(tapply(iris$Sepal.Width, assigned, mean)[by_cluster])
  )

  hc <- tl_model(iris[, 1:4], ~ Petal.Length + Petal.Width, method = "hclust")
  hdata <- tl_table_clusters(hc, k = 3)[["_data"]]
  expect_named(hdata, c("cluster", "size", "Petal.Length", "Petal.Width"))
  cut <- stats::cutree(hc$fit$model, k = 3)
  expect_equal(
    hdata$Petal.Length,
    as.vector(tapply(iris$Petal.Length, cut, mean)[as.character(hdata$cluster)])
  )

  # Without a formula the fit uses every numeric column, and so does the
  # table
  everything <- tl_table_clusters(tl_model(iris, method = "hclust"))
  expect_named(everything[["_data"]], c("cluster", "size", names(iris)[1:4]))
})

test_that("cluster tables average around a missing value", {
  skip_if_not_installed("gt")
  with_na <- iris[, 1:4]
  with_na[5, 1] <- NA
  tbl <- suppressWarnings(tl_table_clusters(tl_model(with_na,
                                                     method = "hclust")))
  expect_false(anyNA(tbl[["_data"]]$Sepal.Length))
})

test_that("a long formula gives one source note", {
  model <- tl_model(mtcars, mpg ~ cyl + disp + hp + drat + wt + qsec + vs +
                      am + gear + carb + I(wt^2) + I(hp^2), method = "linear")
  expect_length(tl_model_info(model), 1)
})

test_that("tl_table_importance supports xgboost, as documented", {
  skip_if_not_installed("xgboost")
  skip_if_not_installed("gt")
  model <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "xgboost",
                    nrounds = 10)
  data <- tl_table_importance(model)[["_data"]]
  expect_true(all(data$feature %in% c("wt", "hp", "qsec")))
  expect_equal(max(data$importance), 100)
})
