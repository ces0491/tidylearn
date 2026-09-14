# The interaction helpers take a formula or a fitted model and work out the
# predictors themselves. The ways that has gone wrong are reading predictors
# with all.vars(), which misses `.` and `- id`, and reporting a quantity on a
# scale that depends on an unrelated flag. These tests pin the predictor set,
# the scale, and the arguments that narrow what gets tested.

am_data <- transform(mtcars, am = factor(am))

# ---- tl_auto_interactions(): exclude_vars ----------------------------------

make_exclude_data <- function() {
  set.seed(1)
  n <- 300
  dd <- data.frame(a = rnorm(n), b = rnorm(n), z = rnorm(n))
  dd$y <- 3 * dd$a * dd$z + dd$a + dd$b + rnorm(n)
  dd
}

test_that("tl_auto_interactions keeps a strong interaction when not excluded", {
  dd <- make_exclude_data()
  model <- tl_auto_interactions(dd, y ~ a + b + z)

  labels <- attr(stats::terms(model$spec$formula), "term.labels")
  expect_true("a:z" %in% labels)

  dotted <- tl_auto_interactions(dd, y ~ .)
  labels <- attr(stats::terms(dotted$spec$formula, data = dd), "term.labels")
  expect_true(all(c("a", "b", "z", "a:z") %in% labels))
})

test_that("tl_auto_interactions never adds a pair with an excluded variable", {
  dd <- make_exclude_data()

  # Excluding z leaves only pairs with no real interaction behind them.
  expect_message(
    model <- tl_auto_interactions(dd, y ~ a + b + z, exclude_vars = "z"),
    "No significant interactions found",
    fixed = TRUE
  )
  labels <- attr(stats::terms(model$spec$formula), "term.labels")
  expect_identical(labels, c("a", "b", "z"))

  # Excluding b keeps a:z and removes b from the stored test results.
  model <- tl_auto_interactions(dd, y ~ a + b + z, exclude_vars = "b")
  labels <- attr(stats::terms(model$spec$formula), "term.labels")
  expect_true("a:z" %in% labels)
  tests <- attr(model, "interaction_tests")
  expect_identical(nrow(tests), 1L)
  expect_false(any(tests$var1 == "b" | tests$var2 == "b"))
})

test_that("tl_auto_interactions refuses exclude_vars not in the formula", {
  dd <- make_exclude_data()
  expect_error(
    tl_auto_interactions(dd, y ~ a + b + z, exclude_vars = "q"),
    "'exclude_vars' names variables that are not predictors in 'formula': q",
    fixed = TRUE
  )
  expect_error(
    tl_auto_interactions(dd, y ~ a + b + z, exclude_vars = 3),
    "'exclude_vars' must be a character vector of predictor names",
    fixed = TRUE
  )
})

# ---- tl_interaction_effects(): scale of fit and intervals ------------------

test_that("interval flag does not change the scale of a logistic fit", {
  model <- suppressWarnings(
    tl_model(am_data, am ~ wt * hp, method = "logistic")
  )

  with_ci <- suppressWarnings(tl_interaction_effects(model, "wt", "hp"))
  without_ci <- suppressWarnings(
    tl_interaction_effects(model, "wt", "hp", intervals = FALSE)
  )

  expect_equal(with_ci$effects$fit, without_ci$effects$fit)
  expect_equal(with_ci$slopes$slope, without_ci$slopes$slope)
  expect_true(all(with_ci$effects$fit >= 0 & with_ci$effects$fit <= 1))
  expect_true(all(with_ci$effects$lower >= 0 & with_ci$effects$upper <= 1))
  expect_true(all(with_ci$effects$lower <= with_ci$effects$fit &
                    with_ci$effects$fit <= with_ci$effects$upper))
})

test_that("the logistic interval is the back-transformed link interval", {
  model <- suppressWarnings(
    tl_model(am_data, am ~ wt * hp, method = "logistic")
  )
  effects <- suppressWarnings(tl_interaction_effects(model, "wt", "hp"))$effects

  grid <- effects[, c("wt", "hp")]
  link <- stats::predict(model$fit, newdata = grid, type = "link",
                         se.fit = TRUE)
  z <- stats::qnorm(0.975)
  expect_equal(effects$lower, stats::plogis(link$fit - z * link$se.fit),
               ignore_attr = TRUE)
  expect_equal(effects$upper, stats::plogis(link$fit + z * link$se.fit),
               ignore_attr = TRUE)
})

test_that("the linear interval is the one predict.lm gives", {
  model <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")
  effects <- tl_interaction_effects(model, "wt", "hp")$effects

  reference <- stats::predict(model$fit, newdata = effects[, c("wt", "hp")],
                              interval = "confidence")
  expect_equal(effects$fit, reference[, "fit"], ignore_attr = TRUE)
  expect_equal(effects$lower, reference[, "lwr"], ignore_attr = TRUE)
  expect_equal(effects$upper, reference[, "upr"], ignore_attr = TRUE)
})

test_that("a fit without standard errors falls back to point estimates", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")

  expect_message(
    effects <- suppressWarnings(tl_interaction_effects(model, "wt", "hp")),
    "Confidence intervals need standard errors from a linear or generalised",
    fixed = TRUE
  )
  expect_false("lower" %in% names(effects$effects))
  expect_true(all(is.finite(effects$effects$fit)))

  expect_no_message(
    suppressWarnings(
      tl_interaction_effects(model, "wt", "hp", intervals = FALSE)
    )
  )
})

# ---- formulas written with `.` ---------------------------------------------

test_that("tl_interaction_effects accepts a model fitted with y ~ .", {
  model <- tl_model(mtcars[, c("mpg", "wt", "hp")], mpg ~ ., method = "linear")

  expect_warning(
    effects <- tl_interaction_effects(model, "wt", "hp"),
    "Interaction term wt:hp not found in model formula",
    fixed = TRUE
  )
  expect_true(all(is.finite(effects$slopes$slope)))

  expect_error(
    tl_interaction_effects(model, "wt", "cyl"),
    "Variables not found in model formula",
    fixed = TRUE
  )
})

test_that("tl_plot_interaction accepts a model fitted with y ~ .", {
  model <- tl_model(am_data[, c("mpg", "wt", "am")], mpg ~ ., method = "linear")

  expect_s3_class(tl_plot_interaction(model, "wt", "am"), "ggplot")
  expect_error(
    tl_plot_interaction(model, "wt", "hp"),
    "Variables not found in model formula",
    fixed = TRUE
  )
})

test_that("a variable dropped with - is not treated as a predictor", {
  model <- tl_model(mtcars[, c("mpg", "wt", "hp", "qsec")], mpg ~ . - qsec,
                    method = "linear")
  expect_error(
    tl_plot_interaction(model, "wt", "qsec"),
    "Variables not found in model formula",
    fixed = TRUE
  )
  # ...while the columns the `.` does keep are found. Reading the formula
  # with all.vars() refused these too, which is why the refusal above
  # proves nothing on its own.
  expect_s3_class(tl_plot_interaction(model, "wt", "hp"), "ggplot")
})

# ---- tl_test_interactions(): predictors and pair filters -------------------

test_that("tl_test_interactions expands a `.` formula and accepts a string", {
  data <- mtcars[, c("mpg", "wt", "hp", "qsec")]

  dotted <- tl_test_interactions(data, mpg ~ ., all_pairs = TRUE)
  expect_identical(nrow(dotted), 3L)
  expect_setequal(c(dotted$var1, dotted$var2), c("wt", "hp", "qsec"))

  dropped <- tl_test_interactions(data, mpg ~ . - qsec, all_pairs = TRUE)
  expect_identical(nrow(dropped), 1L)

  from_string <- tl_test_interactions(data, "mpg ~ wt + hp", all_pairs = TRUE)
  from_formula <- tl_test_interactions(data, mpg ~ wt + hp, all_pairs = TRUE)
  expect_equal(from_string, from_formula)
})

test_that("tl_test_interactions stops clearly when no pairs remain", {
  expect_error(
    tl_test_interactions(mtcars, mpg ~ wt + hp, all_pairs = TRUE,
                         categorical_only = TRUE),
    "No variable pairs left to test after applying categorical_only = TRUE",
    fixed = TRUE
  )
  expect_error(
    tl_test_interactions(mtcars, mpg ~ wt, all_pairs = TRUE),
    "No variable pairs left to test. The predictors in 'formula' are: wt",
    fixed = TRUE
  )

  # The same filter still returns the pairs that do qualify.
  kept <- tl_test_interactions(am_data, mpg ~ wt + hp + am, all_pairs = TRUE,
                               numeric_only = TRUE)
  expect_identical(nrow(kept), 1L)
  mixed <- tl_test_interactions(am_data, mpg ~ wt + hp + am, all_pairs = TRUE,
                                mixed_only = TRUE)
  expect_identical(nrow(mixed), 2L)
})

# ---- tl_plot_interaction(): confidence band --------------------------------

has_ribbon <- function(plot) {
  any(vapply(plot$layers, function(layer) {
    inherits(layer$geom, "GeomRibbon")
  }, logical(1)))
}

test_that("tl_plot_interaction draws a confidence band for an lm fit", {
  model <- tl_model(am_data, mpg ~ wt * am, method = "linear")
  plot <- tl_plot_interaction(model, "wt", "am")

  expect_true(has_ribbon(plot))
  built <- ggplot2::ggplot_build(plot)
  ribbon <- built$data[[which(vapply(plot$layers, function(layer) {
    inherits(layer$geom, "GeomRibbon")
  }, logical(1)))]]
  expect_true(all(ribbon$ymin < ribbon$ymax))

  # The categorical-first ordering draws the band too.
  expect_true(has_ribbon(tl_plot_interaction(model, "am", "wt")))
  expect_false(has_ribbon(
    tl_plot_interaction(model, "wt", "am", confidence = FALSE)
  ))
})

test_that("a logistic band stays on the probability scale", {
  vs_data <- transform(mtcars, vs = factor(vs), am = factor(am))
  model <- suppressWarnings(
    tl_model(vs_data, vs ~ wt + am, method = "logistic")
  )
  plot <- tl_plot_interaction(model, "wt", "am")

  expect_true(has_ribbon(plot))
  expect_true(all(plot$data$.lower >= 0 & plot$data$.upper <= 1))
  expect_true(all(plot$data$.lower <= plot$data$prediction &
                    plot$data$prediction <= plot$data$.upper))
})

test_that("a fit with no standard errors says the band is not drawn", {
  model <- tl_model(am_data, mpg ~ wt + am, method = "tree")

  expect_message(
    plot <- tl_plot_interaction(model, "wt", "am"),
    "No confidence band drawn",
    fixed = TRUE
  )
  expect_false(has_ribbon(plot))
  expect_no_message(tl_plot_interaction(model, "wt", "am", confidence = FALSE))
})

# ---- tl_interaction_effects(): tied quartiles ------------------------------

test_that("tied quartiles of by_var give one slope row each", {
  model <- tl_model(mtcars, mpg ~ wt * cyl, method = "linear")
  effects <- tl_interaction_effects(model, "wt", "cyl")

  expect_identical(effects$slopes$by_value, c(4, 6, 8))
  expect_identical(effects$slopes$by_label, c("Q0/Q25", "Q50", "Q75/Q100"))
  expect_identical(nrow(effects$effects), 300L)

  # A continuous by_var keeps all five quartiles.
  continuous <- tl_interaction_effects(
    tl_model(mtcars, mpg ~ wt * hp, method = "linear"), "wt", "hp"
  )
  expect_identical(continuous$slopes$by_label,
                   c("Q0", "Q25", "Q50", "Q75", "Q100"))
})

test_that("the interaction functions keep an offset in the formula", {
  model <- tl_model(mtcars, mpg ~ wt * hp + offset(log(disp)),
                    method = "linear")
  effects <- tl_interaction_effects(model, "wt", "hp")
  expect_true(all(is.finite(effects$effects$fit)))
  expect_s3_class(tl_plot_interaction(model, "wt", "hp"), "ggplot")

  tested <- tl_test_interactions(mtcars, mpg ~ wt + hp + offset(log(disp)),
                                 all_pairs = TRUE)
  reference <- anova(lm(mpg ~ wt + hp + offset(log(disp)), mtcars),
                     lm(mpg ~ wt + hp + wt:hp + offset(log(disp)), mtcars))
  expect_equal(tested$p_value[[1]], reference$`Pr(>F)`[[2]])
  # the offset's variable is not a candidate for an interaction
  expect_identical(nrow(tested), 1L)
  expect_false("disp" %in% c(tested$var1, tested$var2))

  auto <- suppressMessages(
    tl_auto_interactions(mtcars, mpg ~ wt + hp + offset(log(disp)))
  )
  expect_match(paste(deparse(auto$spec$formula), collapse = " "),
               "offset(log(disp))", fixed = TRUE)
  expect_setequal(attr(terms(auto$spec$formula), "term.labels"),
                  c("wt", "hp", "wt:hp"))
})

test_that("tl_auto_interactions returns the model when nothing is left", {
  # Every pair already in the formula used to stop with "No variable pairs
  # left to test" from the testing step
  expect_message(
    model <- tl_auto_interactions(mtcars, mpg ~ wt * hp),
    "No interactions left to test"
  )
  expect_setequal(attr(terms(model$spec$formula), "term.labels"),
                  c("wt", "hp", "wt:hp"))
})

test_that("the interaction testers need a response", {
  expect_error(tl_test_interactions(mtcars, ~ wt + hp, all_pairs = TRUE),
               "needs a response")
  expect_error(tl_auto_interactions(mtcars, ~ wt + hp), "needs a response")
})

test_that("interaction effects of a constant variable are refused", {
  d <- transform(mtcars, k = 1)
  model <- suppressWarnings(tl_model(d, mpg ~ wt * hp + k, method = "linear"))
  expect_error(tl_interaction_effects(model, "k", "hp"),
               "'k' takes a single value")
})

test_that("tl_plot_interaction refuses a prediction type", {
  am <- transform(mtcars, am = factor(am), vs = factor(vs))
  model <- tl_model(am, am ~ wt * vs, method = "logistic")
  expect_error(tl_plot_interaction(model, "wt", "vs", type = "class"),
               "takes no 'type' argument")
  expect_s3_class(tl_plot_interaction(model, "wt", "vs"), "ggplot")
})

test_that("a pair already in the formula is not tested again", {
  tested <- tl_test_interactions(mtcars, mpg ~ wt * hp + qsec,
                                 all_pairs = TRUE)
  pairs <- paste(tested$var1, tested$var2)
  expect_false(any(pairs %in% c("wt hp", "hp wt")))
  expect_false(anyNA(tested$p_value))
  expect_error(tl_test_interactions(mtcars, mpg ~ wt * hp, all_pairs = TRUE),
               "already in the formula")
})
