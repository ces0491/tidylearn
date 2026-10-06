# The method-specific wrappers: rpart, gbm, e1071, nnet, keras and xgboost.
# test-predict-contract.R drives every method through the same predict()
# grid; this file covers what each wrapper does on its own -- the arguments
# it forwards, the values it fixes, and the helpers built on one backend.

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

xgb_v3 <- function() {
  utils::packageVersion("xgboost") >= "3.0.0"
}

# The parameters tl_fit_xgboost() sets, so a direct xgb.train() call can
# reproduce a tidylearn fit
xgb_params <- function(objective, eval_metric, ...) {
  c(
    list(
      objective = objective, eval_metric = eval_metric, max_depth = 6,
      eta = 0.3, subsample = 1, colsample_bytree = 1, min_child_weight = 1,
      gamma = 0, alpha = 0, lambda = 1
    ),
    list(...)
  )
}

# More rows than gbm's defaults need: it refuses to fit unless half the
# rows exceed twice n.minobsinnode plus one
make_regression_data <- function(seed = 20261005, n = 120) {
  set.seed(seed)
  d <- data.frame(x1 = stats::runif(n), x2 = stats::runif(n))
  d$y <- 4 * d$x1 + d$x2 + stats::rnorm(n, sd = 0.2)
  d
}

geoms_of <- function(p) {
  vapply(p$layers, function(l) class(l$geom)[1], character(1))
}

# ---- every backend -----------------------------------------------------

test_that("the backends refuse an offset they cannot apply at predict()", {
  # Held at the same predictors, a prediction moved by 0 when the offset
  # moved by 100, for every method here: rpart and gbm fit an offset()
  # term as a shift of the response that their predict() never adds back,
  # and the rest left it out of the fit. Passed as an argument, it was
  # ignored by randomForest, e1071 and xgboost without a word.
  d <- make_regression_data()
  d$off <- rep(c(0, 100), length.out = nrow(d))
  methods <- c("tree", "forest", "boost", "svm", "nn", "deep")
  if (requireNamespace("xgboost", quietly = TRUE)) {
    methods <- c(methods, "xgboost")
  }

  for (method in methods) {
    expect_error(
      tl_model(d, y ~ x1 + offset(off), method = method),
      paste0("Method \"", method, "\" cannot use offset\\(off\\)"),
      info = method
    )
    expect_error(
      tl_model(d, y ~ x1, method = method, offset = d$off),
      paste0("Method \"", method, "\" cannot use the offset argument"),
      info = method
    )
  }

  # The offset column as an ordinary predictor still fits
  expect_s3_class(tl_model(d, y ~ x1 + off, method = "tree"),
                  "tidylearn_tree")
})

# ---- tree ------------------------------------------------------------

test_that("a tree refuses an argument rpart() and rpart.control() lack", {
  skip_if_not_installed("rpart")

  # Everything that was not one of rpart()'s own arguments went to
  # rpart.control(), which discards names it does not recognise. A
  # misspelt maxdeth = 1 fitted the default tree, and an offset fitted with
  # no offset, where rpart() itself stops on both. An offset is refused
  # with the other backends' in "the backends refuse an offset ...".
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp + qsec, method = "tree", maxdeth = 1),
    "Method \"tree\" has no argument 'maxdeth'"
  )
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "tree", offset = mtcars$hp),
    "Method \"tree\" cannot use the offset argument"
  )

  # rpart.control()'s own arguments still reach it, spelt correctly
  model <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "tree",
                    maxdepth = 1, xval = 0)
  direct <- rpart::rpart(mpg ~ wt + hp + qsec, data = mtcars,
                         control = rpart::rpart.control(maxdepth = 1,
                                                        xval = 0))
  expect_equal(model$fit$frame, direct$frame)
  expect_equal(nrow(model$fit$frame), 3L)
})

test_that("an explicit minsplit re-derives minbucket over a control list", {
  skip_if_not_installed("rpart")

  # rpart.control() derives minbucket from minsplit, so a list built with
  # rpart.control(cp = 0.001) carries minbucket = 7 from its default
  # minsplit = 20. Copying that over an explicit minsplit = 5 left a node
  # needing 14 rows to split, and minsplit did nothing: 7 nodes, not 19.
  model <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "tree",
                    minsplit = 5,
                    control = rpart::rpart.control(cp = 0.001))
  expect_equal(model$fit$control$minsplit, 5)
  expect_equal(model$fit$control$minbucket, round(5 / 3))

  direct <- rpart::rpart(
    mpg ~ wt + hp + qsec, data = mtcars,
    control = rpart::rpart.control(minsplit = 5, cp = 0.001)
  )
  expect_equal(model$fit$frame, direct$frame)

  # Without an explicit minsplit, the list's own pair is kept
  listed <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "tree",
                     control = rpart::rpart.control(minbucket = 4,
                                                    cp = 0.001))
  expect_equal(listed$fit$control$minbucket, 4)
  expect_equal(listed$fit$control$minsplit, 12)

  # And an explicit minbucket beats both
  explicit <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "tree",
                       minsplit = 5, minbucket = 3,
                       control = rpart::rpart.control(cp = 0.001))
  expect_equal(explicit$fit$control$minbucket, 3)
})

test_that("tl_plot_importance refuses a tree with no splits", {
  skip_if_not_installed("rpart")

  # tl_extract_importance() returns no rows for a tree with no splits,
  # and the plot drew an empty chart
  stump <- tl_model(mtcars, mpg ~ wt + hp, method = "tree", cp = 1)
  expect_equal(nrow(stump$fit$frame), 1L)
  expect_error(
    tl_plot_importance(stump),
    "No feature has non-zero importance: the tree has no splits"
  )

  # A tree with splits still plots its importance
  grown <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")
  p <- tl_plot_importance(grown)
  expect_true("GeomCol" %in% geoms_of(p))
  expect_setequal(as.character(p$data$feature),
                  names(grown$fit$variable.importance))
})

# ---- forest ------------------------------------------------------------

test_that("a forest fits transformed terms", {
  skip_if_not_installed("randomForest")

  # randomForest's formula interface re-evaluates each term on a copy of
  # its model frame whose columns data.frame() has renamed, so log(hp) or
  # factor(cyl) failed with "object 'hp' not found"
  set.seed(1)
  model <- tl_model(mtcars, mpg ~ log(hp) + wt, method = "forest",
                    ntree = 50)
  x <- data.frame(`log(hp)` = log(mtcars$hp), wt = mtcars$wt,
                  check.names = FALSE)
  set.seed(1)
  direct <- randomForest::randomForest(x, mtcars$mpg, ntree = 50,
                                       importance = TRUE)
  expect_equal(predict(model, mtcars)$.pred,
               unname(stats::predict(direct, x)))

  # New data needs only the raw columns, and a missing value leaves NA in
  # its own row
  nd <- mtcars[1:5, c("hp", "wt")]
  nd$hp[2] <- NA
  preds <- predict(model, nd)$.pred
  expect_true(is.na(preds[2]))
  expect_equal(preds[-2], predict(model, mtcars[c(1, 3, 4, 5), ])$.pred)

  # A computed factor, scored on new data holding one of its levels
  set.seed(1)
  cyl <- tl_model(mtcars, mpg ~ factor(cyl) + wt, method = "forest",
                  ntree = 50)
  expect_identical(rownames(randomForest::importance(cyl$fit)),
                   c("factor(cyl)", "wt"))
  expect_equal(nrow(predict(cyl, data.frame(cyl = 6, wt = 3))), 1L)

  # Classification probabilities through the same path
  set.seed(1)
  flowers <- tl_model(iris, Species ~ log(Petal.Length) + Petal.Width,
                      method = "forest", ntree = 50)
  probs <- predict(flowers, iris[c(1, 51, 101), ], type = "prob")
  expect_named(probs, levels(iris$Species))
  expect_equal(unname(rowSums(probs)), rep(1, 3))

  # A formula of plain columns keeps randomForest's own formula interface
  plain <- tl_model(mtcars, mpg ~ hp + wt, method = "forest", ntree = 10)
  expect_s3_class(plain$fit, "randomForest.formula")
})

# ---- partial dependence ----------------------------------------------

test_that("partial dependence averages the predictions at each grid value", {
  skip_if_not_installed("randomForest")
  skip_if_not_installed("gbm")

  # predict() returns a tibble, and mean() of a tibble is NA with a
  # warning, so every regression curve -- tree, forest and boost alike,
  # and the function's own example -- came back all NA.
  d <- make_regression_data()
  set.seed(1)
  models <- list(
    tree = tl_model(d, y ~ x1 + x2, method = "tree"),
    forest = tl_model(d, y ~ x1 + x2, method = "forest", ntree = 50),
    boost = tl_model(d, y ~ x1 + x2, method = "boost", n.trees = 50)
  )
  direct <- list(
    tree = function(fit, nd) stats::predict(fit, newdata = nd),
    forest = function(fit, nd) stats::predict(fit, newdata = nd),
    boost = function(fit, nd) {
      gbm::predict.gbm(fit, newdata = nd, n.trees = fit$n.trees)
    }
  )
  grid <- seq(min(d$x1), max(d$x1), length.out = 5)

  for (method in names(models)) {
    expect_no_warning(
      pd <- tl_plot_partial_dependence(models[[method]], var = "x1",
                                       n.pts = 5)
    )
    by_hand <- vapply(grid, function(value) {
      nd <- d
      nd$x1 <- value
      mean(direct[[method]](models[[method]]$fit, nd))
    }, numeric(1))
    expect_equal(pd$data$y, by_hand, info = method)
  }

  # No class is mapped, so no class label is set: ggplot2 reports a label
  # for an unmapped aesthetic each time the plot is drawn
  expect_no_message(print(pd))
})

test_that("partial dependence draws every class of a multiclass model", {
  skip_if_not_installed("rpart")

  # The curve was the mean probability of the second class alone, and the
  # class column held whichever class had the highest mean -- a different
  # class from the one drawn, and never shown.
  model <- tl_model(iris, Species ~ ., method = "tree")
  pd <- tl_plot_partial_dependence(model, var = "Petal.Length", n.pts = 4)

  expect_equal(nrow(pd$data), 4L * 3L)
  expect_identical(levels(pd$data$class), levels(iris$Species))

  grid <- seq(min(iris$Petal.Length), max(iris$Petal.Length),
              length.out = 4)
  by_hand <- t(vapply(grid, function(value) {
    nd <- iris
    nd$Petal.Length <- value
    colMeans(stats::predict(model$fit, newdata = nd, type = "prob"))
  }, numeric(3)))
  for (cl in levels(iris$Species)) {
    expect_equal(pd$data$y[pd$data$class == cl], unname(by_hand[, cl]),
                 info = cl)
  }

  # One line per class, told apart by colour
  geoms <- geoms_of(pd)
  built <- ggplot2::layer_data(pd, which(geoms == "GeomLine"))
  expect_length(unique(built$colour), 3L)

  # Two classes draw the positive class, the second level, and say so
  binary <- droplevels(iris[iris$Species != "setosa", ])
  pb <- tl_plot_partial_dependence(
    tl_model(binary, Species ~ ., method = "tree"),
    var = "Petal.Length", n.pts = 4
  )
  expect_identical(as.character(unique(pb$data$class)), "virginica")
  expect_match(pb$labels$y, "virginica")

  # A model that records no classes reads them off the response the
  # formula computes, not the raw column: factor(mpg > 20) has two classes,
  # where mpg has 25 values
  cars <- rbind(mtcars, mtcars)
  above <- tl_model(cars, factor(mpg > 20) ~ wt + hp, method = "tree")
  above$spec$response_levels <- NULL
  pa <- tl_plot_partial_dependence(above, var = "wt", n.pts = 3)
  expect_identical(levels(pa$data$class), c("FALSE", "TRUE"))
  expect_false(anyNA(pa$data$y))
})

# ---- boost -------------------------------------------------------------

test_that("boost applies case weights as gbm does", {
  skip_if_not_installed("gbm")

  # gbm() evaluates weights inside its own model frame, so forwarded
  # through ... it failed with "..1 used in an incorrect context".
  d <- make_regression_data()
  w <- stats::runif(nrow(d), 0.1, 5)

  set.seed(1)
  model <- tl_model(d, y ~ x1 + x2, method = "boost", weights = w,
                    n.trees = 50)
  set.seed(1)
  direct <- gbm::gbm(
    y ~ x1 + x2, data = d, weights = w, distribution = "gaussian",
    n.trees = 50, interaction.depth = 3, shrinkage = 0.1,
    n.minobsinnode = 10, cv.folds = 0, verbose = FALSE
  )
  expect_equal(predict(model, d)$.pred,
               gbm::predict.gbm(direct, newdata = d, n.trees = 50))

  set.seed(1)
  unweighted <- tl_model(d, y ~ x1 + x2, method = "boost", n.trees = 50)
  expect_false(isTRUE(all.equal(predict(model, d)$.pred,
                                predict(unweighted, d)$.pred)))

  # The stored call refers to the weights rather than spelling out 120
  # values, and still reaches them
  expect_false(is.numeric(model$fit$call$weights))
  expect_equal(eval(model$fit$call$weights), w)
})

test_that("boost classifies a response the formula computes", {
  skip_if_not_installed("gbm")

  # The 0/1 recoding was written over the raw column, and gbm evaluated
  # factor(am) afresh from it, got a factor and stopped: "Bernoulli
  # requires the response to be numeric in {0,1}"
  cars <- rbind(mtcars, mtcars)
  set.seed(1)
  model <- tl_model(cars, factor(am) ~ wt + hp, method = "boost",
                    n.trees = 30)
  expect_identical(model$spec$response_levels, c("0", "1"))

  coded <- transform(cars, am01 = am)
  set.seed(1)
  direct <- gbm::gbm(
    am01 ~ wt + hp, data = coded, distribution = "bernoulli",
    n.trees = 30, interaction.depth = 3, shrinkage = 0.1,
    n.minobsinnode = 10, cv.folds = 0, verbose = FALSE
  )
  probs <- predict(model, mtcars, type = "prob")
  expect_named(probs, c("0", "1"))
  expect_equal(probs[["1"]],
               gbm::predict.gbm(direct, newdata = mtcars, n.trees = 30,
                                type = "response"))

  # A dot still leaves the response's own column out of the predictors
  dotted <- tl_model(cars, factor(am) ~ ., method = "boost", n.trees = 5)
  expect_false(any(c("am", ".tl_response") %in% dotted$fit$var.names))

  # factor(am) has the classes of the column itself; factor(mpg > 20)
  # does not, and has to be read from the formula as well
  coded <- transform(cars, hi = as.integer(mpg > 20))
  set.seed(1)
  above <- tl_model(cars, factor(mpg > 20) ~ wt + hp, method = "boost",
                    n.trees = 30)
  set.seed(1)
  direct_hi <- gbm::gbm(
    hi ~ wt + hp, data = coded, distribution = "bernoulli",
    n.trees = 30, interaction.depth = 3, shrinkage = 0.1,
    n.minobsinnode = 10, cv.folds = 0, verbose = FALSE
  )
  expect_named(predict(above, cars, type = "prob"), c("FALSE", "TRUE"))
  expect_equal(predict(above, cars, type = "prob")[["TRUE"]],
               gbm::predict.gbm(direct_hi, newdata = cars, n.trees = 30,
                                type = "response"))

  # And a model that records no classes reads them off the same response
  above$spec$response_levels <- NULL
  expect_identical(levels(predict(above, cars[1:3, ])$.pred),
                   c("FALSE", "TRUE"))
})

test_that("boost takes the caller's verbose and regression distribution", {
  skip_if_not_installed("gbm")

  # Both were fixed in the gbm() call while ... went to the same call, so
  # naming either failed with "formal argument matched by multiple actual
  # arguments".
  d <- make_regression_data()
  set.seed(1)
  robust <- tl_model(d, y ~ x1 + x2, method = "boost", n.trees = 30,
                     distribution = "laplace")
  set.seed(1)
  direct <- gbm::gbm(
    y ~ x1 + x2, data = d, distribution = "laplace", n.trees = 30,
    interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10,
    cv.folds = 0, verbose = FALSE
  )
  expect_identical(robust$fit$distribution$name, "laplace")
  expect_equal(predict(robust, d)$.pred,
               gbm::predict.gbm(direct, newdata = d, n.trees = 30))

  expect_output(
    tl_model(d, y ~ x1 + x2, method = "boost", n.trees = 5, verbose = TRUE),
    "TrainDeviance"
  )

  # For classification the distribution follows the response, and the
  # predict path reads only those two
  binary <- droplevels(iris[iris$Species != "setosa", ])
  expect_error(
    tl_model(binary, Species ~ ., method = "boost", distribution = "adaboost"),
    "sets gbm's distribution from the response"
  )
})

# ---- svm -------------------------------------------------------------

test_that("svm predicts every row, whatever the unused columns hold", {
  skip_if_not_installed("e1071")

  # predict.svm() applies na.omit to the whole of newdata, response and
  # unused columns included. airquality's missing Ozone (the response) and
  # Solar.R (unused) dropped 42 rows, and what came back no longer lined
  # up with the input from row 5 on.
  model <- tl_model(airquality, Ozone ~ Temp + Wind, method = "svm")
  preds <- predict(model, airquality)
  expect_equal(nrow(preds), nrow(airquality))
  expect_false(anyNA(preds$.pred))
  direct <- stats::predict(model$fit,
                           newdata = airquality[, c("Temp", "Wind")])
  expect_equal(preds$.pred, unname(direct))

  # Classification, with NA only in a column the formula does not use
  flowers <- iris
  flowers$id <- seq_len(nrow(flowers))
  flowers$id[c(2, 5)] <- NA
  clf <- tl_model(flowers, Species ~ Sepal.Length + Petal.Length,
                  method = "svm")
  classes <- predict(clf, flowers, type = "class")
  expect_equal(nrow(classes), nrow(flowers))
  expect_identical(
    as.character(classes$.pred),
    as.character(stats::predict(
      clf$fit, newdata = flowers[, c("Sepal.Length", "Petal.Length")]
    ))
  )
  expect_equal(nrow(predict(clf, flowers, type = "prob")), nrow(flowers))

  # NA exactly where a predictor is missing, and nowhere else
  gappy <- flowers
  gappy$Sepal.Length[10] <- NA
  with_gap <- predict(clf, gappy, type = "class")
  expect_true(is.na(with_gap$.pred[10]))
  expect_identical(as.character(with_gap$.pred[-10]),
                   as.character(classes$.pred[-10]))
})

test_that("svm refuses case weights, which e1071 does not have", {
  skip_if_not_installed("e1071")

  # svm.default() swallows arguments it does not recognise, so a weighted
  # fit was identical to the unweighted one while the model recorded that
  # weights had been applied
  w <- c(rep(0.01, 16), rep(10, 16))
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "svm", weights = w),
    "e1071::svm\\(\\) has no case weights"
  )

  # Per-class weights are e1071's own argument and still reach it
  model <- tl_model(iris, Species ~ ., method = "svm",
                    class.weights = c(setosa = 1, versicolor = 5,
                                      virginica = 1))
  expect_s3_class(model, "tidylearn_svm")
})

test_that("svm takes the caller's probability and type", {
  skip_if_not_installed("e1071")

  # probability was fixed to the task and type to C-classification while
  # ... went to the same call, so naming either failed with "formal
  # argument matched by multiple actual arguments"
  model <- tl_model(iris, Species ~ ., method = "svm", probability = FALSE)
  expect_false(isTRUE(model$fit$compprob))
  expect_equal(nrow(predict(model, iris[1:3, ], type = "class")), 3L)
  expect_error(
    predict(model, iris[1:3, ], type = "prob"),
    "Refit the model with probability = TRUE"
  )

  nu <- tl_model(iris, Species ~ ., method = "svm",
                 type = "nu-classification")
  # e1071's code for nu-classification
  expect_equal(nu$fit$type, 1)
})

test_that("tl_plot_svm_boundary draws the formula's predictors", {
  skip_if_not_installed("e1071")

  # The default axes were the first two numeric columns of the data, so
  # Species ~ Petal.Length + Petal.Width was drawn over the sepal columns
  # the model never saw
  model <- tl_model(iris, Species ~ Petal.Length + Petal.Width,
                    method = "svm")
  p <- tl_plot_svm_boundary(model, grid_size = 20)
  expect_identical(c(p$labels$x, p$labels$y),
                   c("Petal.Length", "Petal.Width"))
  geoms <- geoms_of(p)
  regions <- p$layers[[which(geoms == "GeomRaster")]]$data
  expect_gt(length(unique(regions$pred_class)), 1L)

  # One axis named, the other taken from the formula
  one <- tl_plot_svm_boundary(model, x_var = "Petal.Width", grid_size = 10)
  expect_identical(c(one$labels$x, one$labels$y),
                   c("Petal.Width", "Petal.Length"))

  # No 0.5 contour for three classes
  expect_false("GeomContour" %in% geoms)
})

test_that("tl_plot_svm_boundary draws the 0.5 contour for two classes", {
  skip_if_not_installed("e1071")

  # The condition compared e1071's numeric type code with the string
  # "C-classification" and read $probability, which the fitted object does
  # not have, so the contour was never drawn
  binary <- droplevels(iris[iris$Species != "setosa", ])
  model <- tl_model(binary, Species ~ Sepal.Length + Sepal.Width,
                    method = "svm")
  p <- tl_plot_svm_boundary(model, grid_size = 20)
  geoms <- geoms_of(p)
  expect_true("GeomContour" %in% geoms)

  grid <- p$layers[[which(geoms == "GeomContour")]]$data
  direct <- attr(
    stats::predict(model$fit,
                   newdata = grid[, c("Sepal.Length", "Sepal.Width")],
                   probability = TRUE),
    "probabilities"
  )
  expect_equal(grid$pred_prob, unname(direct[, "virginica"]))
})

test_that("tl_plot_svm_boundary colours the points by the computed response", {
  skip_if_not_installed("e1071")

  # The points were coloured by the raw column the formula's response is
  # computed from: 25 shades of mpg for a model of factor(mpg > 20)
  cars <- rbind(mtcars, mtcars)
  model <- tl_model(cars, factor(mpg > 20) ~ wt + hp, method = "svm")
  p <- tl_plot_svm_boundary(model, grid_size = 10)
  points <- which(geoms_of(p) == "GeomPoint")
  built <- ggplot2::layer_data(p, points)
  expect_length(unique(built$colour), 2L)
  expect_equal(nrow(built), nrow(cars))
})

# ---- neural networks ---------------------------------------------------

test_that("a multiclass network predicts NA for a row missing a predictor", {
  skip_if_not_installed("nnet")

  # apply(probs, 1, which.max) returns integer(0) for the all-NA row
  # predict.nnet() leaves there, so the default type and "class" failed
  # with "invalid subscript type 'list'"
  set.seed(1)
  model <- tl_model(iris, Species ~ ., method = "nn", size = 3,
                    trace = FALSE)
  nd <- iris[c(1, 51, 101), ]
  nd$Petal.Length[2] <- NA

  for (type in c("response", "class")) {
    preds <- predict(model, nd, type = type)
    expect_equal(nrow(preds), 3L, info = type)
    expect_true(is.na(preds$.pred[2]), info = type)
    expect_identical(
      as.character(preds$.pred[c(1, 3)]),
      stats::predict(model$fit, newdata = nd[c(1, 3), ], type = "class"),
      info = type
    )
  }

  probs <- predict(model, nd, type = "prob")
  expect_true(all(is.na(unlist(probs[2, ]))))

  # A model that records no classes reads them off the response the
  # formula computes: the raw mpg column gave 25 "classes" for a
  # one-output network, and predict() failed
  cars <- rbind(mtcars, mtcars)
  set.seed(1)
  above <- tl_model(cars, factor(mpg > 20) ~ wt + hp, method = "nn",
                    size = 2, trace = FALSE)
  recorded <- predict(above, cars, type = "prob")
  above$spec$response_levels <- NULL
  expect_equal(predict(above, cars, type = "prob"), recorded)
})

test_that("a neural network applies case weights as nnet does", {
  skip_if_not_installed("nnet")

  # nnet() evaluates weights inside its own model frame, so forwarded
  # through ... it failed with "..1 used in an incorrect context"
  w <- c(rep(0.01, 16), rep(10, 16))
  set.seed(1)
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "nn", weights = w,
                    size = 2, trace = FALSE)
  set.seed(1)
  direct <- nnet::nnet(mpg ~ wt + hp, data = mtcars, weights = w, size = 2,
                       decay = 0, maxit = 100, trace = FALSE, linout = TRUE)
  expect_equal(predict(model, mtcars)$.pred,
               as.vector(stats::predict(direct, newdata = mtcars)))
  expect_false(is.numeric(model$fit$call$weights))

  # The caller's linout wins over the regression default: a logistic
  # output unit cannot leave [0, 1]
  set.seed(1)
  squashed <- tl_model(mtcars, mpg ~ wt + hp, method = "nn", size = 2,
                       trace = FALSE, linout = FALSE)
  expect_true(all(predict(squashed, mtcars)$.pred <= 1))
  expect_gt(max(predict(model, mtcars)$.pred), 1)
})

test_that("tl_tune_nn scores a two-class candidate by its own predictions", {
  skip_if_not_installed("nnet")
  skip_if_not_installed("rsample")

  # predict.nnet(type = "raw") is an n x 1 matrix for two classes, so the
  # is.vector() branch never ran and which.max() over one column always
  # chose the first class. Every candidate scored the share of the second
  # class -- 0.5 here -- and the first in the grid always won.
  binary <- droplevels(iris[iris$Species != "setosa", ])
  set.seed(1)
  tuned <- tl_tune_nn(binary, Species ~ ., is_classification = TRUE,
                      sizes = c(1, 3), decays = c(0, 0.1), folds = 4)

  # The same folds and fits, by hand
  set.seed(1)
  splits <- rsample::vfold_cv(binary, v = 4)
  grid <- expand.grid(size = c(1, 3), decay = c(0, 0.1))
  lv <- levels(binary$Species)
  by_hand <- vapply(seq_len(nrow(grid)), function(i) {
    mean(vapply(seq_len(4), function(j) {
      train <- rsample::analysis(splits$splits[[j]])
      test <- rsample::assessment(splits$splits[[j]])
      net <- nnet::nnet(Species ~ ., data = train, size = grid$size[i],
                        decay = grid$decay[i], maxit = 100, trace = FALSE)
      p <- stats::predict(net, newdata = test, type = "raw")[, 1]
      mean(ifelse(p > 0.5, lv[2], lv[1]) != test$Species)
    }, numeric(1)))
  }, numeric(1))

  expect_equal(tuned$tuning_results$error, by_hand)
  expect_false(all(tuned$tuning_results$error == 0.5))
})

test_that("tl_tune_nn reads the task from the response", {
  skip_if_not_installed("nnet")
  skip_if_not_installed("rsample")

  # is_classification defaulted to FALSE, so a factor response was fitted
  # as a regression on its codes and failed with "NA/NaN argument"
  binary <- droplevels(iris[iris$Species != "setosa", ])
  set.seed(1)
  inferred <- tl_tune_nn(binary, Species ~ ., sizes = 2, decays = 0,
                         folds = 2)
  set.seed(1)
  flagged <- tl_tune_nn(binary, Species ~ ., is_classification = TRUE,
                        sizes = 2, decays = 0, folds = 2)
  expect_equal(inferred$tuning_results, flagged$tuning_results)

  # A subset that still declares setosa is two classes, not three
  undropped <- iris[iris$Species != "setosa", ]
  set.seed(1)
  expect_no_warning(
    two <- tl_tune_nn(undropped, Species ~ ., sizes = 2, decays = 0,
                      folds = 2)
  )
  expect_identical(two$model$lev, c("versicolor", "virginica"))

  expect_error(
    tl_tune_nn(binary, Species ~ ., is_classification = FALSE, sizes = 2,
               decays = 0, folds = 2),
    "is_classification = FALSE, but 'Species' is a factor"
  )

  # A numeric response is still a regression
  set.seed(1)
  reg <- tl_tune_nn(mtcars, mpg ~ wt, sizes = 2, decays = 0, folds = 2)
  expect_true(all(is.finite(reg$tuning_results$error)))
})

test_that("tl_tune_nn scores the response the formula computes", {
  skip_if_not_installed("nnet")
  skip_if_not_installed("rsample")

  # Each fold was scored against the raw column, so log(y) ~ x was judged
  # by how far its log-scale predictions fell from y itself
  set.seed(2)
  d <- data.frame(x1 = stats::runif(100), x2 = stats::runif(100))
  d$y <- exp(3 * d$x1 + stats::rnorm(100, sd = 0.1))
  set.seed(1)
  tuned <- tl_tune_nn(d, log(y) ~ x1 + x2, sizes = 2, decays = 0, folds = 2)

  set.seed(1)
  splits <- rsample::vfold_cv(d, v = 2)
  by_hand <- mean(vapply(seq_len(2), function(j) {
    train <- rsample::analysis(splits$splits[[j]])
    test <- rsample::assessment(splits$splits[[j]])
    net <- nnet::nnet(log(y) ~ x1 + x2, data = train, size = 2, decay = 0,
                      maxit = 100, trace = FALSE, linout = TRUE)
    mean((as.vector(stats::predict(net, newdata = test)) - log(test$y))^2)
  }, numeric(1)))
  expect_equal(tuned$tuning_results$error, by_hand)

  # And the task is the computed response's: factor(am) is two classes,
  # where the numeric am made it a regression that nnet refused
  set.seed(1)
  two <- tl_tune_nn(rbind(mtcars, mtcars), factor(am) ~ wt + hp,
                    sizes = 2, decays = 0, folds = 2)
  expect_identical(two$model$lev, c("0", "1"))
})

test_that("tl_tune_nn takes the caller's maxit", {
  skip_if_not_installed("nnet")
  skip_if_not_installed("rsample")

  # maxit = 100 was fixed in both calls while ... went to them too, so
  # passing it failed with "formal argument matched by multiple actual
  # arguments"
  set.seed(1)
  tuned <- tl_tune_nn(iris, Species ~ ., is_classification = TRUE,
                      sizes = 2, decays = 0, folds = 2, maxit = 300)
  expect_equal(tuned$model$call$maxit, 300)
})

test_that("tl_tune_nn and tl_tune_deep refuse per-row arguments", {
  skip_if_not_installed("nnet")

  # A weight vector went whole into every fold, a subset of the rows:
  # nnet failed on the lengths, and keras on an argument it does not take
  w <- rep(c(1, 2), length.out = nrow(iris))
  expect_error(
    tl_tune_nn(iris, Species ~ ., sizes = 2, decays = 0, folds = 2,
               weights = w),
    "tl_tune_nn\\(\\) cannot re-split 'weights' across folds"
  )
  expect_error(
    tl_tune_deep(iris, Species ~ ., subset = w > 1),
    "tl_tune_deep\\(\\) cannot re-split 'subset' across folds"
  )
})

# ---- xgboost -----------------------------------------------------------

test_that("xgboost applies case weights through its DMatrix", {
  skip_if_not_installed("xgboost")

  # weights reached xgb.train() as an unknown argument: xgboost 3.x warned
  # that it did not recognise them and fitted without
  w <- c(rep(0.01, 16), rep(10, 16))
  expect_no_warning(
    model <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost",
                      nrounds = 20, weights = w)
  )
  x <- stats::model.matrix(mpg ~ wt + hp, mtcars)[, -1]
  direct <- xgboost::xgb.train(
    params = xgb_params("reg:squarederror", "rmse"),
    data = xgboost::xgb.DMatrix(x, label = mtcars$mpg, weight = w),
    nrounds = 20, verbose = 0
  )
  expect_equal(predict(model, mtcars)$.pred,
               stats::predict(direct, xgboost::xgb.DMatrix(x)))

  unweighted <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost",
                         nrounds = 20)
  expect_false(isTRUE(all.equal(predict(model, mtcars)$.pred,
                                predict(unweighted, mtcars)$.pred)))
})

test_that("xgboost settings passed to tl_model() reach params", {
  skip_if_not_installed("xgboost")

  # They reached xgb.train() as arguments it does not have. xgboost 3.x
  # moves them into params, but warns that doing so will become an error.
  expect_no_warning(
    model <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost",
                      nrounds = 20, max_leaves = 2, tree_method = "hist")
  )
  x <- stats::model.matrix(mpg ~ wt + hp, mtcars)[, -1]
  direct <- xgboost::xgb.train(
    params = xgb_params("reg:squarederror", "rmse", max_leaves = 2,
                        tree_method = "hist"),
    data = xgboost::xgb.DMatrix(x, label = mtcars$mpg),
    nrounds = 20, verbose = 0
  )
  expect_equal(predict(model, mtcars)$.pred,
               stats::predict(direct, xgboost::xgb.DMatrix(x)))

  # The caller's value wins over one tidylearn sets. xgboost 3.x refused
  # an objective passed alongside the one in params.
  expect_no_warning(
    absolute <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost",
                         nrounds = 5, objective = "reg:absoluteerror")
  )
  direct <- xgboost::xgb.train(
    params = xgb_params("reg:absoluteerror", "rmse"),
    data = xgboost::xgb.DMatrix(x, label = mtcars$mpg),
    nrounds = 5, verbose = 0
  )
  expect_equal(predict(absolute, mtcars)$.pred,
               stats::predict(direct, xgboost::xgb.DMatrix(x)))
})

test_that("early_stopping_rounds without data to stop on is refused by name", {
  skip_if_not_installed("xgboost")

  # tl_model() holds no rows out, so xgboost stopped with "For early
  # stopping, 'evals' must have at least one element"
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "xgboost", nrounds = 20,
             early_stopping_rounds = 3),
    "early_stopping_rounds needs data to stop on"
  )

  # With a validation set of the caller's own, it stops early
  train <- mtcars[1:24, ]
  held_out <- mtcars[25:32, ]
  validation <- xgboost::xgb.DMatrix(
    stats::model.matrix(mpg ~ wt + hp, held_out)[, -1],
    label = held_out$mpg
  )
  eval_arg <- if (xgb_v3()) "evals" else "watchlist"
  model <- do.call(tl_model, c(
    list(train, mpg ~ wt + hp, method = "xgboost", nrounds = 200,
         early_stopping_rounds = 3),
    stats::setNames(list(list(validation = validation)), eval_arg)
  ))
  rounds <- if (xgb_v3()) {
    xgboost::xgb.get.num.boosted.rounds(model$fit)
  } else {
    model$fit$niter
  }
  expect_lt(rounds, 200)
})

test_that("xgboost drops a missing response and keeps missing predictors", {
  skip_if_not_installed("xgboost")

  # model.matrix() dropped the row with the missing predictor while the
  # labels kept it: "The length of labels must equal to the number of rows
  # in the input data". xgboost routes missing predictors itself.
  gappy <- mtcars
  gappy$wt[3] <- NA
  gappy$mpg[5] <- NA
  model <- tl_model(gappy, mpg ~ wt + hp, method = "xgboost", nrounds = 10)

  kept <- !is.na(gappy$mpg)
  x <- cbind(wt = gappy$wt, hp = gappy$hp)
  direct <- xgboost::xgb.train(
    params = xgb_params("reg:squarederror", "rmse"),
    data = xgboost::xgb.DMatrix(x[kept, ], label = gappy$mpg[kept]),
    nrounds = 10, verbose = 0
  )
  preds <- predict(model, gappy)
  expect_equal(nrow(preds), nrow(gappy))
  expect_equal(preds$.pred, stats::predict(direct, xgboost::xgb.DMatrix(x)))
})

test_that("xgboost fits the response the formula computes", {
  skip_if_not_installed("xgboost")

  # The labels were read from the raw column, so log(mpg) ~ wt + hp was
  # fitted to mpg itself, while tl_evaluate() scores the log
  model <- tl_model(mtcars, log(mpg) ~ wt + hp, method = "xgboost",
                    nrounds = 20)
  x <- stats::model.matrix(mpg ~ wt + hp, mtcars)[, -1]
  direct <- xgboost::xgb.train(
    params = xgb_params("reg:squarederror", "rmse"),
    data = xgboost::xgb.DMatrix(x, label = log(mtcars$mpg)),
    nrounds = 20, verbose = 0
  )
  preds <- predict(model, mtcars)$.pred
  expect_equal(preds, stats::predict(direct, xgboost::xgb.DMatrix(x)))
  # log(mpg) on mtcars runs from 2.3 to 3.5
  expect_lt(max(preds), 4)

  # The tuner reads the same response, for its folds and its refit
  set.seed(1)
  tuned <- tl_tune_xgboost(mtcars, log(mpg) ~ wt + hp, cv_folds = 3,
                           nrounds = 20, param_grid = list(max_depth = 2),
                           verbose = FALSE)
  expect_lt(max(predict(tuned, mtcars)$.pred), 4)
  expect_lt(attr(tuned, "tuning_results")$best_score, 1)
})

test_that("predict()'s iterationrange follows the installed xgboost", {
  skip_if_not_installed("xgboost")

  # iterationrange is documented as inclusive of both ends, as xgboost 3.x
  # reads it, but went to predict() as given, and xgboost before 3.0 reads
  # the end as exclusive: c(1, 5) predicted from four rounds there
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost", nrounds = 20)
  five <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost", nrounds = 5)
  six <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost", nrounds = 6)
  expect_equal(predict(model, mtcars, iterationrange = c(1, 5))$.pred,
               predict(five, mtcars)$.pred)

  # Told it runs an xgboost before 3.0, it hands this one that version's
  # c(1, 6) -- which 3.x reads as six rounds
  if (xgb_v3()) {
    local_mocked_bindings(tl_xgb_v3 = function() FALSE)
    expect_equal(predict(model, mtcars, iterationrange = c(1, 5))$.pred,
                 predict(six, mtcars)$.pred)
  }

  expect_error(
    predict(model, mtcars, iterationrange = c(1, 25)),
    "'iterationrange' runs to round 25, but the model has 20 rounds"
  )
  expect_error(
    predict(model, mtcars, iterationrange = c(5, 1)),
    "'iterationrange' must be c\\(start, end\\)"
  )
})

test_that("the extra-columns warning in xgboost predict has no call", {
  skip_if_not_installed("xgboost")

  # Every other message in the file is raised with call. = FALSE
  model <- tl_model(mtcars[, 1:4], mpg ~ ., method = "xgboost", nrounds = 5)
  w <- expect_warning(
    tl_predict_xgboost(model, mtcars),
    "New data contains columns not in the training data"
  )
  expect_null(conditionCall(w))
})

test_that("a gblinear booster gets no tree parameters", {
  skip_if_not_installed("xgboost")

  # max_depth, subsample and the other tree parameters were always set,
  # so a linear booster printed "Parameters: { ... } are not used" on
  # every fit
  # gblinear's default updater runs its coordinate updates in parallel,
  # and on several threads the result depends on their timing, so both
  # fits use one thread and the same seed
  notes <- utils::capture.output(
    model <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost",
                      nrounds = 10, booster = "gblinear", seed = 1,
                      nthread = 1),
    type = "message"
  )
  expect_false(any(grepl("are not used", notes)))

  x <- stats::model.matrix(mpg ~ wt + hp, mtcars)[, -1]
  direct <- xgboost::xgb.train(
    params = list(objective = "reg:squarederror", eval_metric = "rmse",
                  eta = 0.3, alpha = 0, lambda = 1, booster = "gblinear",
                  seed = 1, nthread = 1),
    data = xgboost::xgb.DMatrix(x, label = mtcars$mpg),
    nrounds = 10, verbose = 0
  )
  expect_equal(predict(model, mtcars)$.pred,
               stats::predict(direct, xgboost::xgb.DMatrix(x)))
})

test_that("xgboost probabilities come back as a tibble", {
  skip_if_not_installed("xgboost")

  # Every other method returns a tibble, as predict()'s @return says;
  # xgboost returned a data.frame for both binary and multiclass
  multi <- tl_model(iris, Species ~ ., method = "xgboost", nrounds = 5)
  expect_s3_class(predict(multi, iris[1:3, ], type = "prob"), "tbl_df")

  binary <- droplevels(iris[iris$Species != "setosa", ])
  two <- tl_model(binary, Species ~ ., method = "xgboost", nrounds = 5)
  probs <- predict(two, binary[1:3, ], type = "prob")
  expect_s3_class(probs, "tbl_df")
  expect_named(probs, c("versicolor", "virginica"))
})

test_that("tl_plot_xgboost_importance is a ggplot of xgboost's importance", {
  skip_if_not_installed("xgboost")

  # It returned xgb.plot.importance()'s data.table and drew base graphics,
  # and importance_type was never read
  model <- tl_model(mtcars, mpg ~ ., method = "xgboost", nrounds = 20)
  importance <- as.data.frame(xgboost::xgb.importance(model = model$fit))

  p <- tl_plot_xgboost_importance(model, top_n = 4)
  expect_s3_class(p, "ggplot")
  expect_true("GeomCol" %in% geoms_of(p))
  gain <- importance[order(-importance$Gain), ][1:4, ]
  expect_identical(as.character(p$data$feature), gain$Feature)
  expect_equal(p$data$importance, gain$Gain / max(importance$Gain))

  by_cover <- tl_plot_xgboost_importance(model, top_n = 4,
                                         importance_type = "cover")
  cover <- importance[order(-importance$Cover), ][1:4, ]
  expect_identical(as.character(by_cover$data$feature), cover$Feature)

  expect_error(
    tl_plot_xgboost_importance(model, importance_type = "weight"),
    "'importance_type' must be one of"
  )
})

test_that("tl_plot_xgboost_tree draws the tree tree_index names", {
  skip_if_not_installed("xgboost")
  skip_if_not_installed("DiagrammeR")

  # xgboost 3.x renamed the argument to the one-based tree_idx, so
  # tree_index was dropped with a warning that it will become an error,
  # and the first tree was drawn every time
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "xgboost", nrounds = 5)
  expect_no_warning(first <- tl_plot_xgboost_tree(model, tree_index = 0))
  last <- tl_plot_xgboost_tree(model, tree_index = 4)
  expect_false(identical(first$x$diagram, last$x$diagram))

  direct <- if (xgb_v3()) {
    xgboost::xgb.plot.tree(model = model$fit, tree_idx = 5)
  } else {
    xgboost::xgb.plot.tree(model = model$fit, trees = 4)
  }
  expect_identical(last$x$diagram, direct$x$diagram)

  expect_error(
    tl_plot_xgboost_tree(model, tree_index = 5),
    "the model has 5 trees, numbered 0 to 4"
  )
})

test_that("tl_tune_xgboost reads the task from the response", {
  skip_if_not_installed("xgboost")

  # is_classification defaulted to FALSE, so Species ~ . was tuned as a
  # regression on the class codes, without a word
  tuned <- tl_tune_xgboost(iris, Species ~ ., cv_folds = 3, nrounds = 5,
                           param_grid = list(max_depth = 2, eta = 0.3),
                           verbose = FALSE)
  expect_true(tuned$spec$is_classification)
  expect_identical(attr(tuned, "tuning_results")$best_params$objective,
                   "multi:softprob")
  expect_true(is.factor(predict(tuned, iris[c(1, 51, 101), ])$.pred))

  # A subset that still declares setosa is two classes: it got
  # multi:softprob and a setosa probability column
  undropped <- iris[iris$Species != "setosa", ]
  two <- tl_tune_xgboost(undropped, Species ~ ., cv_folds = 3, nrounds = 5,
                         param_grid = list(max_depth = 2), verbose = FALSE)
  expect_identical(attr(two, "tuning_results")$best_params$objective,
                   "binary:logistic")
  expect_named(predict(two, undropped[1:2, ], type = "prob"),
               c("versicolor", "virginica"))

  expect_error(
    tl_tune_xgboost(iris, Species ~ ., is_classification = FALSE,
                    cv_folds = 3, nrounds = 5,
                    param_grid = list(max_depth = 2), verbose = FALSE),
    "is_classification = FALSE, but 'Species' is a factor"
  )

  # The task is the computed response's. Read off the numeric am, the
  # cross-validation ran as a regression on the class codes while the
  # refit, through tl_model(), classified.
  coded <- tl_tune_xgboost(rbind(mtcars, mtcars), factor(am) ~ wt + hp,
                           cv_folds = 3, nrounds = 5,
                           param_grid = list(max_depth = 2), verbose = FALSE)
  expect_identical(attr(coded, "tuning_results")$best_params$objective,
                   "binary:logistic")
  expect_true(coded$spec$is_classification)
})

test_that("tl_tune_xgboost's default grid is tl_default_param_grid()'s", {
  skip_if_not_installed("xgboost")

  # The tuner kept its own copy of the large grid tl_default_param_grid()
  # gives tl_tune_grid(), free to drift from it
  tuned <- tl_tune_xgboost(mtcars, mpg ~ wt + hp, cv_folds = 2, nrounds = 1,
                           early_stopping_rounds = NULL, verbose = FALSE,
                           nthread = 1)
  large <- tl_default_param_grid("xgboost", size = "large")
  large$nrounds <- NULL
  results <- attr(tuned, "tuning_results")
  expect_identical(results$param_grid, large)
  expect_length(results$results, prod(lengths(large)))

  # And it follows that grid when the grid changes
  local_mocked_bindings(
    tl_default_param_grid = function(method, size = "medium", ...) {
      list(nrounds = c(5, 10), max_depth = c(2, 3))
    }
  )
  followed <- tl_tune_xgboost(mtcars, mpg ~ wt + hp, cv_folds = 2,
                              nrounds = 1, early_stopping_rounds = NULL,
                              verbose = FALSE, nthread = 1)
  expect_identical(attr(followed, "tuning_results")$param_grid,
                   list(max_depth = c(2, 3)))
})

test_that("tl_tune_xgboost takes weights and refuses other per-row arguments", {
  skip_if_not_installed("xgboost")

  # A subset or foldid went into params, where xgboost ignored it with a
  # note, and the tuning ran on every row
  expect_error(
    tl_tune_xgboost(mtcars, mpg ~ wt + hp, cv_folds = 2, nrounds = 2,
                    param_grid = list(max_depth = 2), verbose = FALSE,
                    subset = mtcars$cyl != 8),
    "tl_tune_xgboost\\(\\) cannot re-split 'subset' across folds"
  )

  # Weights live on the DMatrix, whose folds xgb.cv() slices with them
  w <- rep(c(1, 2), length.out = nrow(mtcars))
  weighted <- tl_tune_xgboost(mtcars, mpg ~ wt + hp, cv_folds = 2,
                              nrounds = 2, param_grid = list(max_depth = 2),
                              verbose = FALSE, weights = w)
  expect_identical(weighted$spec$per_row_args, "weights")
})

test_that("tl_tune_xgboost sends xgb.cv() arguments to xgb.cv() alone", {
  skip_if_not_installed("xgboost")

  # All of ... went to xgb.cv() and to the final xgb.train(), which warned
  # "Passed unrecognized parameters: showsd". A booster parameter belongs
  # in params for both.
  set.seed(1)
  expect_no_warning(
    tuned <- tl_tune_xgboost(mtcars, mpg ~ ., cv_folds = 3, nrounds = 5,
                             early_stopping_rounds = NULL,
                             param_grid = list(max_depth = 2),
                             verbose = FALSE, showsd = FALSE,
                             max_leaves = 3)
  )
  expect_equal(attr(tuned, "tuning_results")$best_params$max_leaves, 3)

  same <- tl_model(mtcars, mpg ~ ., method = "xgboost", max_depth = 2,
                   max_leaves = 3, nrounds = 5)
  expect_equal(predict(tuned, mtcars)$.pred, predict(same, mtcars)$.pred)
})

test_that("tl_compare_cv refits a tuned xgboost model at its tuned settings", {
  skip_if_not_installed("xgboost")

  # The tuned model recorded no arguments, so each fold refitted it at
  # xgboost's defaults: its fold scores matched the default model's
  # exactly, although their in-sample rmse differed by three orders of
  # magnitude
  set.seed(1)
  tuned <- tl_tune_xgboost(mtcars, mpg ~ .,
                           param_grid = list(max_depth = 1, eta = 0.01),
                           cv_folds = 3, nrounds = 3,
                           early_stopping_rounds = NULL, verbose = FALSE)
  expect_equal(tuned$spec$args$max_depth, 1)
  expect_equal(tuned$spec$args$eta, 0.01)
  expect_equal(tuned$spec$args$nrounds, 3L)

  same <- tl_model(mtcars, mpg ~ ., method = "xgboost", max_depth = 1,
                   eta = 0.01, nrounds = 3)
  default <- tl_model(mtcars, mpg ~ ., method = "xgboost")
  set.seed(2)
  cv <- tl_compare_cv(mtcars,
                      list(tuned = tuned, same = same, default = default),
                      folds = 3, metrics = "rmse")
  rmse <- split(cv$fold_metrics$value, cv$fold_metrics$model)
  expect_equal(rmse$tuned, rmse$same)
  expect_false(isTRUE(all.equal(rmse$tuned, rmse$default)))
})

test_that("tl_xgboost_shap scores unlabelled data and honours trees_idx", {
  skip_if_not_installed("xgboost")

  # The design matrix came from the full two-sided formula, so data
  # without the response failed with "object 'mpg' not found"; trees_idx
  # is not an argument of xgboost's predict() and was dropped
  model <- tl_model(mtcars, mpg ~ ., method = "xgboost", nrounds = 20)
  features <- attr(model$fit, "feature_names")
  unlabelled <- tl_xgboost_shap(model, data = mtcars[1:5, -1],
                                n_samples = NULL)
  labelled <- tl_xgboost_shap(model, data = mtcars[1:5, ], n_samples = NULL)
  expect_equal(unlabelled[features], labelled[features])

  dm <- xgboost::xgb.DMatrix(as.matrix(mtcars[1:5, features]))
  direct <- stats::predict(model$fit, dm, predcontrib = TRUE)
  expect_equal(unname(as.matrix(unlabelled[features])),
               unname(direct[, features]))
  expect_equal(unlabelled$BIAS, unname(direct[, ncol(direct)]))

  # The first five rounds of a 20-round fit are a 5-round fit
  five <- tl_model(mtcars, mpg ~ ., method = "xgboost", nrounds = 5)
  partial <- tl_xgboost_shap(model, data = mtcars[1:5, ], n_samples = NULL,
                             trees_idx = 1:5)
  reference <- tl_xgboost_shap(five, data = mtcars[1:5, ], n_samples = NULL)
  expect_equal(partial[features], reference[features])
  expect_false(isTRUE(all.equal(partial[features], labelled[features])))

  expect_error(
    tl_xgboost_shap(model, data = mtcars[1:5, ], trees_idx = c(1, 3)),
    "'trees_idx' must be a run of consecutive rounds"
  )
})

test_that("multiclass SHAP values come per class, under the right names", {
  skip_if_not_installed("xgboost")

  # xgboost 3.x returns a row x class x feature array, and the code
  # labelled a flattened slice of it: the column called Petal.Length held
  # Sepal.Length's SHAP for virginica, and most of the 18 columns were
  # named NA
  model <- tl_model(iris, Species ~ ., method = "xgboost", nrounds = 10)
  features <- attr(model$fit, "feature_names")
  shap <- tl_xgboost_shap(model, data = iris[1:5, ], n_samples = NULL)

  expect_false(anyNA(names(shap)))
  expect_equal(nrow(shap), 5L * 3L)
  expect_identical(levels(shap$class), levels(iris$Species))

  direct <- stats::predict(
    model$fit, xgboost::xgb.DMatrix(as.matrix(iris[1:5, features])),
    predcontrib = TRUE
  )
  for (k in seq_along(levels(iris$Species))) {
    cl <- levels(iris$Species)[k]
    by_class <- if (is.list(direct)) direct[[k]] else direct[, k, ]
    rows <- shap$class == cl
    expect_equal(unname(as.matrix(shap[rows, features])),
                 unname(by_class[, features]), info = cl)
    expect_equal(shap$row_id[rows], 1:5, info = cl)
  }

  # The dependence plot draws one panel per class from those values
  p <- tl_plot_xgboost_shap_dependence(model, feature = "Petal.Length",
                                       data = iris[1:5, ], n_samples = NULL)
  expect_s3_class(p$facet, "FacetWrap")
  versicolor <- p$data[p$data$class == "versicolor", ]
  expect_equal(versicolor$shap_value,
               shap$Petal.Length[shap$class == "versicolor"])
  expect_equal(versicolor$feature_value, iris$Petal.Length[1:5])

  expect_s3_class(
    tl_plot_xgboost_shap_summary(model, data = iris[1:5, ],
                                 n_samples = NULL),
    "ggplot"
  )
})

test_that("the SHAP dependence plot pairs each SHAP value with its own row", {
  skip_if_not_installed("xgboost")

  # tl_xgboost_shap() sampled its rows and this function drew a second,
  # independent sample for the feature values, so for y = 10x the plotted
  # correlation was near zero rather than near 1
  set.seed(3)
  big <- data.frame(x = stats::runif(400), z = stats::runif(400))
  big$y <- 10 * big$x + stats::rnorm(400, sd = 0.1)
  model <- tl_model(big, y ~ x + z, method = "xgboost", nrounds = 30)

  set.seed(10)
  p <- tl_plot_xgboost_shap_dependence(model, feature = "x",
                                       interaction_feature = "z")
  rows <- match(p$data$feature_value, big$x)
  expect_false(anyNA(rows))
  direct <- stats::predict(
    model$fit, xgboost::xgb.DMatrix(as.matrix(big[rows, c("x", "z")])),
    predcontrib = TRUE
  )
  expect_equal(p$data$shap_value, unname(direct[, "x"]))
  expect_equal(p$data$interaction_value, big$z[rows])
  expect_gt(stats::cor(p$data$feature_value, p$data$shap_value), 0.9)
})

# ---- deep learning -------------------------------------------------------

test_that("deep refuses case weights, which it does not pass to keras", {
  # They reached keras's fit() as an argument it does not take, and the
  # fit failed with Python's TypeError. Refused before keras is needed.
  w <- rep(c(1, 2), length.out = nrow(mtcars))
  expect_error(
    tl_model(mtcars, mpg ~ wt + hp, method = "deep", weights = w),
    "Method \"deep\" cannot use case weights"
  )
})

test_that("deep predict scores unlabelled data, keeps NA rows, pins levels", {
  skip_on_cran()
  skip_if_no_tensorflow()

  # The design matrix came from the full formula, so data without the
  # response failed with "object 'Species' not found"; model.matrix()
  # dropped a row with a missing predictor, so three rows came back as
  # two; and new data with fewer factor levels changed the columns
  tensorflow::set_random_seed(1)
  model <- tl_model(iris, Species ~ ., method = "deep", epochs = 2,
                    hidden_layers = 8, verbose = 0)
  rows <- iris[c(1, 51, 101), ]
  labelled <- predict(model, rows, type = "prob")
  unlabelled <- predict(model, rows[, -5], type = "prob")
  expect_equal(unlabelled, labelled)

  x <- scale(stats::model.matrix(Species ~ ., rows)[, -1],
             center = model$fit$x_means, scale = model$fit$x_sds)
  direct <- stats::predict(model$fit$model, x, verbose = 0)
  expect_equal(unname(as.matrix(labelled)), unname(direct),
               tolerance = 1e-6)

  gappy <- rows
  gappy$Petal.Length[2] <- NA
  probs <- predict(model, gappy, type = "prob")
  expect_equal(nrow(probs), 3L)
  expect_true(all(is.na(unlist(probs[2, ]))))
  # keras rounds a two-row batch slightly differently from a three-row one
  expect_equal(probs[c(1, 3), ], labelled[c(1, 3), ], tolerance = 1e-6)
  expect_true(is.na(predict(model, gappy)$.pred[2]))

  set.seed(1)
  d <- data.frame(g = factor(rep(c("a", "b", "c"), 40)),
                  x = stats::rnorm(120))
  d$y <- d$x + c(a = 0, b = 5, c = -5)[as.character(d$g)]
  tensorflow::set_random_seed(1)
  grouped <- tl_model(d, y ~ g + x, method = "deep", epochs = 2,
                      hidden_layers = 4, verbose = 0)
  short <- data.frame(g = factor(c("b", "c")), x = c(0, 0), y = c(5, -5))
  full <- short
  full$g <- factor(full$g, levels = c("a", "b", "c"))
  expect_equal(predict(grouped, short), predict(grouped, full))
})

test_that("a deep fit drops an incomplete row from x and y together", {
  skip_on_cran()
  skip_if_no_tensorflow()

  # model.matrix() dropped the row with the missing predictor while the
  # response kept it, so keras trained on pairs shifted by one row
  set.seed(5)
  d <- data.frame(x = stats::rnorm(120))
  d$y <- 3 * d$x
  gappy <- d
  gappy$x[1] <- NA

  fit_on <- function(data) {
    tensorflow::set_random_seed(1)
    tl_model(data, y ~ x, method = "deep", epochs = 3, hidden_layers = 4,
             dropout = 0, verbose = 0)
  }
  with_gap <- fit_on(gappy)
  without <- fit_on(d[-1, ])
  expect_equal(predict(with_gap, d[2:6, ])$.pred,
               predict(without, d[2:6, ])$.pred)
})

test_that("a deep fit holds out a random set of rows for validation", {
  skip_on_cran()
  skip_if_no_tensorflow()

  # keras's validation_split takes the last rows as given, before any
  # shuffling. iris is sorted by species, so the default 0.2 held out
  # every virginica row: the model never trained on that class and was
  # validated on nothing else.
  tensorflow::set_random_seed(1)
  model <- tl_model(iris, Species ~ ., method = "deep", epochs = 1,
                    hidden_layers = 8, dropout = 0, verbose = 0)
  held_out <- model$fit$validation_rows
  expect_length(held_out, 30L)
  expect_gt(length(unique(iris$Species[held_out])), 1L)

  # The val_loss keras reports is the loss on exactly those rows
  x <- scale(stats::model.matrix(Species ~ ., iris)[, -1],
             center = model$fit$x_means, scale = model$fit$x_sds)
  y <- keras::to_categorical(as.integer(iris$Species) - 1, num_classes = 3)
  loss <- keras::evaluate(model$fit$model, x[held_out, ], y[held_out, ],
                          verbose = 0)
  expect_equal(utils::tail(model$fit$history$metrics$val_loss, 1),
               unname(loss[["loss"]]), tolerance = 1e-5)
})

test_that("a deep fit carries a constant predictor without NaN", {
  skip_on_cran()
  skip_if_no_tensorflow()

  # Its sd is 0, and scaling by it turned the column into NaN: every
  # prediction was the same value and the loss never moved
  mc <- mtcars
  mc$k <- 1
  tensorflow::set_random_seed(1)
  model <- tl_model(mc, mpg ~ wt + hp + k, method = "deep", epochs = 2,
                    verbose = 0)
  expect_equal(unname(model$fit$x_sds),
               c(stats::sd(mc$wt), stats::sd(mc$hp), 1))

  preds <- predict(model, mc)$.pred
  expect_true(all(is.finite(preds)))
  expect_gt(length(unique(round(preds, 6))), 1L)
  expect_true(all(is.finite(model$fit$history$metrics$loss)))
})

test_that("tl_tune_deep returns a model predict() can use", {
  skip_on_cran()
  skip_if_no_tensorflow()

  # $model was the bare list tl_fit_deep() builds, so predict() on it
  # failed with "no applicable method"; and is_classification defaulted to
  # FALSE, so a factor response was fitted as a regression
  tensorflow::set_random_seed(1)
  tuned <- tl_tune_deep(iris, Species ~ ., hidden_layers_options = list(8),
                        learning_rates = c(0.01, 0.001), batch_sizes = 32,
                        epochs = 2)
  expect_s3_class(tuned$model, "tidylearn_model")
  expect_true(tuned$model$spec$is_classification)

  preds <- predict(tuned$model, iris[c(1, 51, 101), ])
  expect_equal(nrow(preds), 3L)
  expect_identical(levels(preds$.pred), levels(iris$Species))
  expect_s3_class(tl_plot_deep_history(tuned$model), "ggplot")

  rate <- as.numeric(keras::k_get_value(
    tuned$model$fit$model$optimizer$learning_rate
  ))
  expect_equal(rate, tuned$best_learning_rate, tolerance = 1e-6)
})

test_that("tl_plot_deep_architecture hands the network to keras's plot()", {
  skip_on_cran()
  skip_if_no_tensorflow()

  # It called keras::plot_model(), which keras 2.x does not have, so every
  # call failed with "object 'plot_model' not found"
  tensorflow::set_random_seed(1)
  model <- tl_model(iris, Species ~ ., method = "deep", epochs = 1,
                    hidden_layers = 4, verbose = 0)
  result <- tryCatch(
    suppressMessages(tl_plot_deep_architecture(model)),
    error = function(e) e
  )
  if (inherits(result, "error")) {
    # keras draws through pydot, graphviz and png. Where one is missing,
    # the error is keras saying so, not a failed lookup in tidylearn
    expect_match(conditionMessage(result), "graphviz|pydot|png")
  } else {
    expect_null(result)
  }
})
