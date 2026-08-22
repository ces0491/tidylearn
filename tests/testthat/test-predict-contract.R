# Every supervised method has to satisfy the same predict() contract.
#
# Testing one representative method per feature is what let six methods
# ship with a broken predict(): gbm was only ever fitted on a two-class
# response, so its 3-D multinomial array was never seen; SVM appeared
# only as a tuning target, so predict() was never called on it at all;
# and xgboost was never asked to score data without the response column,
# which is the ordinary production case.
#
# So drive every method through the same grid instead: binary and
# multiclass and regression, one row and many, with and without the
# response column present, and with a missing predictor value.

# Methods backed by packages tidylearn imports, so they always run.
# "deep" is excluded deliberately -- it needs a keras backend, which is
# not available on CRAN or in CI. xgboost is in Suggests, so it joins the
# grid only when installed.
classification_methods <- c(
  "logistic", "tree", "forest", "boost",
  "ridge", "lasso", "elastic_net", "svm", "nn"
)
regression_methods <- c(
  "linear", "tree", "forest", "boost",
  "ridge", "lasso", "elastic_net", "svm", "nn"
)

if (requireNamespace("xgboost", quietly = TRUE)) {
  classification_methods <- c(classification_methods, "xgboost")
  regression_methods <- c(regression_methods, "xgboost")
}

# logistic is binary-only by design; the multiclass path is documented as
# not implemented, so it is not part of the multiclass grid.
multiclass_methods <- setdiff(classification_methods, "logistic")

fit_quietly <- function(...) {
  suppressWarnings(suppressMessages(tl_model(...)))
}

# Small, fast, well-separated data so no method has to work hard.
binary_data <- droplevels(iris[c(1:25, 51:75), ])
multiclass_data <- iris[c(1:20, 51:70, 101:120), ]

# gbm refuses to fit when nTrain * bag.fraction <= 2 * n.minobsinnode + 1,
# which rules out mtcars-sized frames at the default bag.fraction, so
# generate a regression frame comfortably above that floor.
set.seed(20260810)
regression_data <- data.frame(
  wt = stats::runif(120, 1.5, 5.5),
  hp = stats::runif(120, 50, 340)
)
regression_data$mpg <- 37 - 5 * regression_data$wt -
  0.02 * regression_data$hp + stats::rnorm(120, sd = 1.5)

method_args <- function(method) {
  switch(method,
    "boost"   = list(n.trees = 10),
    "forest"  = list(ntree = 10),
    "xgboost" = list(nrounds = 5),
    "nn"      = list(size = 2, trace = FALSE, maxit = 50),
    list()
  )
}

fit_model <- function(method, data, formula) {
  do.call(
    fit_quietly,
    c(list(data = data, formula = formula, method = method),
      method_args(method))
  )
}

# ---- classification: class predictions -------------------------------

for (task in c("binary", "multiclass")) {
  data <- if (task == "binary") binary_data else multiclass_data
  methods <- if (task == "binary") {
    classification_methods
  } else {
    multiclass_methods
  }

  for (method in methods) {
    label <- paste0(task, " '", method, "'")

    test_that(paste(label, "predicts classes for every row"), {
      model <- fit_model(method, data, Species ~ .)
      expected_levels <- levels(data$Species)

      for (n in c(1L, 5L, nrow(data))) {
        preds <- suppressWarnings(
          predict(model, new_data = data[seq_len(n), ], type = "class")
        )

        expect_equal(nrow(preds), n,
                     info = paste(method, "class, n =", n))
        expect_true(all(
          stats::na.omit(as.character(preds$.pred)) %in% expected_levels
        ))
      }
    })

    test_that(paste(label, "predicts probabilities that sum to 1"), {
      model <- fit_model(method, data, Species ~ .)
      expected_levels <- levels(data$Species)

      for (n in c(1L, 5L, nrow(data))) {
        probs <- suppressWarnings(
          predict(model, new_data = data[seq_len(n), ], type = "prob")
        )

        expect_equal(nrow(probs), n,
                     info = paste(method, "prob, n =", n))

        prob_cols <- setdiff(names(probs), ".pred")
        expect_setequal(prob_cols, expected_levels)
        expect_equal(
          unname(rowSums(probs[, expected_levels, drop = FALSE])),
          rep(1, n),
          tolerance = 1e-6
        )
      }
    })

    test_that(paste(label, "scores data with no response column"), {
      model <- fit_model(method, data, Species ~ .)
      unlabelled <- data[, setdiff(names(data), "Species"), drop = FALSE]

      preds <- suppressWarnings(
        predict(model, new_data = unlabelled, type = "class")
      )
      expect_equal(nrow(preds), nrow(unlabelled))
    })
  }
}

# ---- regression ------------------------------------------------------

for (method in regression_methods) {
  label <- paste0("regression '", method, "'")

  test_that(paste(label, "predicts one value per row"), {
    model <- fit_model(method, regression_data, mpg ~ wt + hp)

    for (n in c(1L, 5L, nrow(regression_data))) {
      preds <- suppressWarnings(
        predict(model, new_data = regression_data[seq_len(n), ])
      )

      expect_equal(nrow(preds), n, info = paste(method, "n =", n))
      expect_true(is.numeric(preds$.pred))
    }
  })

  test_that(paste(label, "scores data with no response column"), {
    model <- fit_model(method, regression_data, mpg ~ wt + hp)
    unlabelled <- regression_data[, c("wt", "hp"), drop = FALSE]

    preds <- suppressWarnings(predict(model, new_data = unlabelled))
    expect_equal(nrow(preds), nrow(unlabelled))
  })
}

# ---- missing values keep predictions aligned -------------------------

# The failure this guards against is silent: an upstream predict method
# that defaults to na.omit returns a shorter vector, so row i of the
# output stops describing row i of the input and every prediction after
# the missing row is attributed to the wrong observation.

for (method in regression_methods) {
  label <- paste0("'", method, "'")

  test_that(paste(label, "keeps predictions aligned across an NA row"), {
    model <- fit_model(method, regression_data, mpg ~ wt + hp)

    new_data <- regression_data[1:5, ]
    new_data$wt[2] <- NA

    preds <- suppressWarnings(predict(model, new_data = new_data))

    expect_equal(nrow(preds), 5L)

    # Rows without missing predictors must be unaffected by the NA row
    clean <- suppressWarnings(
      predict(model, new_data = regression_data[c(1, 3, 4, 5), ])
    )
    expect_equal(preds$.pred[c(1, 3, 4, 5)], clean$.pred, tolerance = 1e-6)
  })
}

# ---------------------------------------------------------------------
# A response that declares more levels than it uses.
#
# Subsetting keeps every factor level, so iris[iris$Species != "setosa", ]
# holds two classes and declares three. That frame used to break seven of
# the eight classification methods in seven different ways: randomForest
# and glmnet refused to fit, gbm and nnet failed at predict or evaluate,
# rpart returned a probability column for the absent class, and
# tl_event_level_args() read the declared count and so let yardstick score
# the first level as positive -- reopening the metric bug 0.4.0.9000 fixed.
#
# The model itself was never in question: glm() and friends drop the empty
# level internally, so the fit was always identical to the dropped frame's.
# Only tidylearn's description of it was wrong. These assert the two are
# now indistinguishable.

undropped_binary <- iris[iris$Species != "setosa", ][c(1:25, 51:75), ]
dropped_binary <- droplevels(undropped_binary)

test_that("the fixture really does declare a level it never uses", {
  expect_equal(nlevels(undropped_binary$Species), 3L)
  expect_equal(length(unique(as.character(undropped_binary$Species))), 2L)
})

for (method in classification_methods) {
  label <- paste0("'", method, "'")

  test_that(paste(label, "ignores a declared but unused response level"), {
    model <- fit_model(method, undropped_binary, Species ~ .)

    # The spec describes the classes present, not the levels declared
    expect_equal(model$spec$response_levels, levels(dropped_binary$Species))

    probs <- suppressWarnings(predict(model, type = "prob"))
    expect_equal(ncol(probs), 2L)
    expect_equal(names(probs), levels(dropped_binary$Species))
    expect_equal(nrow(probs), nrow(undropped_binary))

    classes <- suppressWarnings(predict(model, type = "class"))
    expect_equal(nrow(classes), nrow(undropped_binary))

    # Every metric must land on the same number as the dropped frame.
    # Asserting only that metrics are present is what let the
    # event_level defect survive a green suite once already.
    wanted <- c("accuracy", "precision", "recall", "specificity", "f1")
    score <- function(d) {
      m <- fit_model(method, d, Species ~ .)
      e <- suppressWarnings(suppressMessages(tl_evaluate(m, metrics = wanted)))
      stats::setNames(e$value, e$metric)
    }
    set.seed(1)
    from_undropped <- score(undropped_binary)
    set.seed(1)
    from_dropped <- score(dropped_binary)

    expect_equal(sort(names(from_undropped)), sort(names(from_dropped)))
    expect_equal(
      from_undropped[sort(names(from_undropped))],
      from_dropped[sort(names(from_dropped))],
      tolerance = 1e-8
    )
  })
}

test_that("method = 'logistic' scores as classification whatever the storage", {
  # A 0/1 integer response produced a binomial glm described by a spec
  # that said is_classification = FALSE. tl_evaluate() then returned
  # rmse/mae/rsq for it, and asking for accuracy returned an empty tibble
  # -- no error, no warning, no metrics.
  set.seed(1)
  d <- data.frame(x = stats::rnorm(60))
  d$y <- as.integer(d$x + stats::rnorm(60) > 0)

  model <- fit_quietly(d, y ~ x, method = "logistic")

  expect_true(model$spec$is_classification)
  expect_equal(model$spec$response_levels, c("0", "1"))

  scored <- suppressWarnings(suppressMessages(
    tl_evaluate(model, metrics = c("accuracy", "f1"))
  ))
  expect_setequal(scored$metric, c("accuracy", "f1"))
  expect_true(all(is.finite(scored$value)))
  expect_false(any(c("rmse", "mae", "rsq") %in% scored$metric))
})
