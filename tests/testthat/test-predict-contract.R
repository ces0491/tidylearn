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
