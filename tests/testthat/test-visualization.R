# ---- Visualization functions ----

# -- Supervised visualization helpers --

test_that("tl_plot_actual_predicted returns ggplot for regression", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  p <- tl_plot_actual_predicted(model)

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_residuals returns ggplot for regression", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  p <- tl_plot_residuals(model, type = "fitted")

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_diagnostics returns a list of plots", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  plots <- tl_plot_diagnostics(model, which = 1:2)

  expect_type(plots, "list")
  expect_length(plots, 2)
})

test_that("tl_plot_confusion returns ggplot for binary classification", {
  data <- iris[iris$Species != "setosa", ]
  data$Species <- droplevels(data$Species)
  model <- tl_model(data, Species ~ ., method = "logistic")
  p <- tl_plot_confusion(model)

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_roc returns ggplot for binary classification", {
  # Create binary classification dataset
  data <- iris[iris$Species != "setosa", ]
  data$Species <- droplevels(data$Species)
  model <- tl_model(data, Species ~ ., method = "logistic")
  p <- tl_plot_roc(model)

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_importance works for tree-based models", {
  skip_if_not_installed("randomForest")

  model <- tl_model(mtcars, mpg ~ wt + hp + cyl, method = "forest")
  p <- tl_plot_importance(model)

  expect_s3_class(p, "ggplot")
})

# -- plot.tidylearn_model dispatch --

test_that("plot.tidylearn_model dispatches correctly for regression", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

  # Default type = "auto" should give actual_predicted for regression

  p <- plot(model)
  expect_s3_class(p, "ggplot")

  # Explicit type
  p2 <- plot(model, type = "residuals")
  expect_s3_class(p2, "ggplot")
})

test_that("plot.tidylearn_model dispatches correctly for classification", {
  data <- iris[iris$Species != "setosa", ]
  data$Species <- droplevels(data$Species)
  model <- tl_model(data, Species ~ ., method = "logistic")

  # Default type = "auto" should give confusion for classification
  p <- plot(model)
  expect_s3_class(p, "ggplot")
})

# -- Lift and gain charts --

test_that("tl_plot_lift works for binary classification", {
  data <- iris[iris$Species != "setosa", ]
  data$Species <- droplevels(data$Species)
  model <- tl_model(data, Species ~ ., method = "logistic")
  p <- tl_plot_lift(model, bins = 5)

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_gain works for binary classification", {
  data <- iris[iris$Species != "setosa", ]
  data$Species <- droplevels(data$Species)
  model <- tl_model(data, Species ~ ., method = "logistic")
  p <- tl_plot_gain(model, bins = 5)

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_lift errors for regression models", {
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")

  expect_error(tl_plot_lift(model), "classification")
})

test_that("tl_plot_gain errors for regression models", {
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")

  expect_error(tl_plot_gain(model), "classification")
})

# -- Model comparison --

test_that("tl_plot_model_comparison returns ggplot", {
  model1 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  model2 <- tl_model(mtcars, mpg ~ wt + hp, method = "polynomial", degree = 2)

  p <- tl_plot_model_comparison(model1, model2, names = c("Linear", "Poly"))

  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_model_comparison errors for mixed model types", {
  model_reg <- tl_model(mtcars, mpg ~ wt, method = "linear")
  model_cls <- tl_model(iris, Species ~ ., method = "tree")

  expect_error(
    tl_plot_model_comparison(model_reg, model_cls),
    "same type"
  )
})

# -- Importance comparison --

test_that("tl_plot_importance_comparison works for tree-based models", {
  skip_if_not_installed("randomForest")

  model1 <- tl_model(mtcars, mpg ~ wt + hp + cyl, method = "forest")
  model2 <- tl_model(mtcars, mpg ~ wt + hp + cyl, method = "tree")
  p <- tl_plot_importance_comparison(model1, model2,
                                     names = c("Forest", "Tree"))

  expect_s3_class(p, "ggplot")
})

# -- CV results plotting --

test_that("tl_plot_cv_results returns ggplot", {
  cv_res <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3)

  # tl_plot_cv_results expects fold_metrics and summary in a specific format

  # Build compatible structure
  fold_metrics <- do.call(rbind, lapply(seq_along(cv_res$folds), function(i) {
    df <- cv_res$folds[[i]]
    df$fold <- i
    df
  }))

  cv_input <- list(
    fold_metrics = fold_metrics,
    summary = dplyr::rename(cv_res$summary, mean_value = mean)
  )

  p <- tl_plot_cv_results(cv_input)
  expect_s3_class(p, "ggplot")
})

# -- Unsupervised visualization --

test_that("plot_clusters returns ggplot", {
  km <- tidy_kmeans(iris[, 1:4], k = 3)
  clustered_data <- augment_kmeans(km, iris[, 1:4])
  p <- plot_clusters(clustered_data)

  expect_s3_class(p, "ggplot")
})

test_that("plot_cluster_sizes returns ggplot", {
  clusters <- sample(1:3, 50, replace = TRUE)
  p <- plot_cluster_sizes(clusters)

  expect_s3_class(p, "ggplot")
})

test_that("plot_elbow returns ggplot", {
  wss <- calc_wss(iris[, 1:4], max_k = 5)
  p <- plot_elbow(wss)

  expect_s3_class(p, "ggplot")
})

test_that("plot_variance_explained returns ggplot", {
  pca_obj <- tidy_pca(iris[, 1:4])
  variance_tbl <- get_pca_variance(pca_obj)
  p <- plot_variance_explained(variance_tbl)

  expect_s3_class(p, "ggplot")
})

test_that("plot_dendrogram works", {
  hc <- tidy_hclust(iris[1:20, 1:4])
  # plot_dendrogram uses base graphics; just test it doesn't error
  expect_invisible(plot_dendrogram(hc, k = 3))
})

test_that("plot_distance_heatmap returns ggplot", {
  d <- dist(iris[1:15, 1:4])
  p <- plot_distance_heatmap(d)

  expect_s3_class(p, "ggplot")
})

# -- Dashboard (Shiny) --

test_that("tl_dashboard errors without shiny installed", {
  skip_if(requireNamespace("shiny", quietly = TRUE) &&
            requireNamespace("shinydashboard", quietly = TRUE) &&
            requireNamespace("DT", quietly = TRUE),
          "Shiny stack is installed, cannot test missing-package path")

  model <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_error(tl_dashboard(model))
})

test_that("tl_dashboard returns shiny.appobj when packages available", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("shinydashboard")
  skip_if_not_installed("DT")

  model <- tl_model(mtcars, mpg ~ wt, method = "linear")
  app <- tl_dashboard(model)

  expect_s3_class(app, "shiny.appobj")
})

test_that("the neural network architecture plot handles one output unit", {
  skip_if_not_installed("nnet")
  skip_if_not_installed("NeuralNetTools")

  # plotnet() reads mod_in$call$formula whenever the net has a single
  # output unit -- every regression fit and every two-class fit. nnet()
  # records its call verbatim, so without substituting the formula in that
  # evaluates the symbol `formula` to stats::formula and dies with "cannot
  # coerce type 'closure' to vector of type 'character'". Multiclass takes
  # a different branch, which is why the one Rd example passed.
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- droplevels(iris_binary$Species)

  set.seed(1)
  binary <- tl_model(iris_binary, Species ~ ., method = "nn",
                     size = 3, trace = FALSE)
  expect_no_error(tl_plot_nn_architecture(binary))

  set.seed(1)
  regression <- tl_model(mtcars, mpg ~ wt + hp, method = "nn",
                         size = 3, trace = FALSE)
  expect_no_error(tl_plot_nn_architecture(regression))

  set.seed(1)
  multiclass <- tl_model(iris, Species ~ ., method = "nn",
                         size = 3, trace = FALSE)
  expect_no_error(tl_plot_nn_architecture(multiclass))
})

test_that("a fitted neural network records a usable formula", {
  skip_if_not_installed("nnet")

  # The property the plot depends on, stated directly so it is not only
  # tested through a Suggests package.
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "nn",
                    size = 2, trace = FALSE)

  expect_s3_class(eval(model$fit$call$formula), "formula")
  expect_equal(
    deparse(eval(model$fit$call$formula)),
    deparse(mpg ~ wt + hp)
  )
})

test_that("tl_spread_labels separates values that would overlap", {
  # Two coefficients within 0.02 of each other printed one label on top
  # of the other on the mtcars lasso path
  y <- c(0.80, 0.82, 2.5, -3.4, -0.1)
  spread <- tl_spread_labels(y, min_gap = 0.5)

  expect_length(spread, length(y))
  expect_gte(min(diff(sort(spread))), 0.5 - 1e-9)

  # Order has to survive, or a label lands on the wrong path
  expect_equal(order(spread), order(y))

  # The block stays where it started rather than drifting upward
  expect_equal(mean(range(spread)), mean(range(y)))

  # Values already far enough apart are left alone
  far <- c(0, 5, 10)
  expect_equal(tl_spread_labels(far, min_gap = 0.5), far)

  # Degenerate inputs: nothing to separate, no gap to enforce
  expect_equal(tl_spread_labels(3, min_gap = 0.5), 3)
  expect_equal(tl_spread_labels(y, min_gap = 0), y)
  expect_equal(tl_spread_labels(c(1, 1, 1), min_gap = NA), c(1, 1, 1))
})

test_that("tl_plot_regularization_path labels every top feature legibly", {
  skip_if_not_installed("glmnet")

  set.seed(1)
  model <- tl_model(mtcars, mpg ~ ., method = "lasso")
  p <- tl_plot_regularization_path(model, label_n = 5)
  expect_s3_class(p, "ggplot")

  # The labels used to sit on the lines at the smallest lambda, sharing
  # their colour and each other's position. They are drawn to the left of
  # the paths now, spread apart, with a leader line back to each one.
  # Find the text layer by its geom rather than by position, so adding
  # a layer to the plot does not silently move the assertion elsewhere
  geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
  text_layer <- which(geoms == "GeomText")
  expect_length(text_layer, 1L)

  built <- ggplot2::layer_data(p, text_layer)
  expect_equal(nrow(built), 5L)

  line_layer <- which(geoms == "GeomLine")
  coef_range <- diff(range(ggplot2::layer_data(p, line_layer)$y))
  expect_gte(min(diff(sort(built$y))), 0.07 * coef_range * 0.99)

  expect_s3_class(tl_plot_regularization_path(model, label_n = 0), "ggplot")
  expect_s3_class(tl_plot_regularization_path(model, label_n = 1), "ggplot")
})

test_that("two models of one method get separate comparison bars", {
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "linear")
  plot <- suppressMessages(tl_plot_model_comparison(m1, m2))

  expect_length(unique(plot$data$model), 2)
  built <- ggplot2::ggplot_build(plot)$data[[1]]
  expect_length(unique(built$x), 2)

  expect_error(
    suppressMessages(tl_plot_model_comparison(m1, m2, names = c("a", "a"))),
    "unique"
  )
})

test_that("gain and lift do not depend on row order", {
  ib <- droplevels(iris[iris$Species != "setosa", ])
  tree <- tl_model(ib, Species ~ Sepal.Width, method = "tree")
  shuffled <- ib[rev(seq_len(nrow(ib))), ]

  curve_y <- function(plot) ggplot2::ggplot_build(plot)$data[[1]]$y
  gain <- function(d) curve_y(tl_plot_gain(tree, new_data = d))
  lift <- function(d) curve_y(tl_plot_lift(tree, new_data = d))
  expect_equal(gain(ib), gain(shuffled))
  expect_equal(lift(ib), lift(shuffled))
  # the curve still ends at every responder
  expect_equal(utils::tail(gain(ib), 1), 100)
})

test_that("gain and lift leave out rows missing the response, and say so", {
  ib <- droplevels(iris[iris$Species != "setosa", ])
  model <- tl_model(ib, Species ~ Sepal.Width + Petal.Length,
                    method = "logistic")
  d <- ib
  d$Species[c(3, 60)] <- NA
  expect_warning(p <- tl_plot_lift(model, new_data = d), "2 row")
  expect_false(anyNA(ggplot2::ggplot_build(p)$data[[1]]$y))
  expect_warning(p <- tl_plot_gain(model, new_data = d), "2 row")
  expect_false(anyNA(ggplot2::ggplot_build(p)$data[[1]]$y))
})

test_that("regularised importance does not depend on a predictor's units", {
  set.seed(1)
  r1 <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "ridge", lambda = 0.5)
  rescaled <- transform(mtcars, hp = hp / 100)
  r2 <- tl_model(rescaled, mpg ~ wt + hp + qsec, method = "ridge",
                 lambda = 0.5)
  i1 <- tl_get_importance_regularized(r1)
  i2 <- tl_get_importance_regularized(r2)
  expect_equal(i1$importance[match(c("wt", "hp", "qsec"), i1$feature)],
               i2$importance[match(c("wt", "hp", "qsec"), i2$feature)],
               tolerance = 1e-6)
})

test_that("regularised importance scales each coefficient by its own column", {
  # The design matrix's first column was dropped as the intercept. A formula
  # with - 1 has none, so the first predictor lost its standard deviation
  # and, with it, its importance.
  model <- tl_model(mtcars, mpg ~ wt + hp + disp - 1, method = "lasso",
                    lambda = 0.01)
  beta <- as.matrix(stats::coef(model$fit, s = 0.01))[, 1]
  beta <- beta[names(beta) != "(Intercept)" & beta != 0]
  design <- stats::model.matrix(mpg ~ wt + hp + disp - 1, mtcars)
  raw <- abs(beta) * apply(design[, names(beta), drop = FALSE], 2, stats::sd)

  imp <- tl_get_importance_regularized(model)
  expect_setequal(imp$feature, names(beta))
  expect_equal(imp$importance[match(names(raw), imp$feature)],
               unname(100 * raw / max(raw)))
})

test_that("a feature one model dropped counts as zero in the ranking", {
  lasso <- tl_model(mtcars, mpg ~ ., method = "lasso")
  tree <- tl_model(mtcars, mpg ~ ., method = "tree")
  p <- tl_plot_importance_comparison(lasso, tree, top_n = 3)
  # every plotted feature has a bar for both models
  counts <- table(as.character(p$data$feature))
  expect_true(all(counts == 2))
})

test_that("importance comparison names a factor once for every model", {
  # randomForest reports the factor Species; glmnet reports its design
  # columns Speciesversicolor and Speciesvirginica. Zero-filling across the
  # two namings said the lasso gave Species nothing and the forest gave the
  # dummies nothing, and halved both in the ranking.
  set.seed(1)
  forest <- tl_model(iris, Sepal.Length ~ ., method = "forest", ntree = 100)
  lasso <- tl_model(iris, Sepal.Length ~ ., method = "lasso")
  p <- tl_plot_importance_comparison(forest, lasso,
                                     names = c("Forest", "Lasso"))
  d <- as.data.frame(p$data)

  predictors <- c("Sepal.Width", "Petal.Length", "Petal.Width", "Species")
  expect_setequal(as.character(d$feature[d$model == "Forest"]), predictors)
  expect_setequal(as.character(d$feature[d$model == "Lasso"]), predictors)

  # A factor takes its largest design column, on the model's own 0-100
  # scale: |coefficient| x column SD, from glmnet directly
  design <- stats::model.matrix(Sepal.Length ~ ., iris)[, -1]
  beta <- as.matrix(
    stats::coef(lasso$fit, s = attr(lasso$fit, "lambda_1se"))
  )[colnames(design), 1]
  raw <- abs(beta) * apply(design, 2, stats::sd)
  expect_equal(
    d$importance[d$model == "Lasso" & d$feature == "Species"],
    100 * max(raw[c("Speciesversicolor", "Speciesvirginica")]) / max(raw)
  )

  forest_raw <- randomForest::importance(forest$fit)[, "%IncMSE"]
  expect_equal(
    d$importance[d$model == "Forest" & d$feature == "Species"],
    100 * forest_raw[["Species"]] / max(forest_raw)
  )
})

test_that("importance comparison zero-fills only a model's own predictors", {
  # The tree was given hp and cyl, the lasso qsec and am. Zero-filling
  # across all five gave each model bars for predictors it never saw and
  # halved those predictors' averages, so the tree's two strongest, cyl
  # and hp, ranked below wt.
  tree <- tl_model(mtcars, mpg ~ wt + hp + cyl, method = "tree")
  lasso <- tl_model(mtcars, mpg ~ wt + qsec + am, method = "lasso",
                    lambda = 1)
  p <- tl_plot_importance_comparison(tree, lasso, names = c("Tree", "Lasso"))
  d <- as.data.frame(p$data)
  expect_setequal(as.character(d$feature[d$model == "Tree"]),
                  c("wt", "hp", "cyl"))
  expect_setequal(as.character(d$feature[d$model == "Lasso"]),
                  c("wt", "qsec", "am"))

  # A predictor the lasso was given and dropped still scores zero for it
  lasso_imp <- tl_get_importance_regularized(lasso)
  expect_false("am" %in% lasso_imp$feature)
  expect_equal(d$importance[d$model == "Lasso" & d$feature == "am"], 0)

  # Each predictor ranks on its mean over the models that were given it
  tree_imp <- tl_extract_importance(tree)
  score <- function(imp, feature) {
    if (feature %in% imp$feature) imp$importance[imp$feature == feature] else 0
  }
  average <- c(
    wt = mean(c(score(tree_imp, "wt"), score(lasso_imp, "wt"))),
    hp = score(tree_imp, "hp"),
    cyl = score(tree_imp, "cyl"),
    qsec = score(lasso_imp, "qsec"),
    am = score(lasso_imp, "am")
  )
  top2 <- tl_plot_importance_comparison(tree, lasso, top_n = 2)$data
  expect_setequal(unique(as.character(top2$feature)),
                  names(sort(average, decreasing = TRUE))[1:2])
})

test_that("a model with no importance stays in the comparison", {
  # A lasso that penalised every predictor away has no importance rows,
  # so it dropped out of the comparison without a word
  tree <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")
  null_lasso <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso",
                         lambda = 100)
  expect_warning(
    p <- tl_plot_importance_comparison(tree, null_lasso,
                                       names = c("Tree", "Null")),
    "'Null' has no feature with non-zero importance"
  )
  d <- as.data.frame(p$data)
  expect_setequal(as.character(d$feature[d$model == "Null"]), c("wt", "hp"))
  expect_true(all(d$importance[d$model == "Null"] == 0))
})

test_that("a tree with no splits has empty importance, not an error", {
  # rpart leaves variable.importance NULL for a single-node tree, and the
  # rescaling failed with "Column `importance` not found in `.data`"; so
  # did any comparison that included the tree
  stump <- tl_model(mtcars, mpg ~ wt + hp, method = "tree", cp = 1)
  imp <- tl_extract_importance(stump)
  expect_identical(nrow(imp), 0L)
  expect_named(imp, c("feature", "importance"))

  tree <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")
  expect_warning(
    p <- tl_plot_importance_comparison(stump, tree,
                                       names = c("Stump", "Tree")),
    "'Stump' has no feature with non-zero importance: the tree has no splits"
  )
  expect_true(all(p$data$importance[p$data$model == "Stump"] == 0))
})

test_that("importance keeps the package's order when every value is negative", {
  # Permutation importance can be negative for every feature. Dividing by
  # the largest value, itself negative, flipped the ranking: %IncMSE of
  # -6.20, -0.87 and -6.61 became 709, 100 and 756.
  set.seed(7)
  d <- data.frame(y = rnorm(60), x1 = rnorm(60), x2 = rnorm(60),
                  x3 = rnorm(60))
  forest <- tl_model(d, y ~ ., method = "forest")
  raw <- randomForest::importance(forest$fit)[, "%IncMSE"]
  expect_true(all(raw < 0))
  imp <- tl_extract_importance(forest)
  expect_equal(imp$importance[match(names(raw), imp$feature)],
               unname(100 * raw / max(abs(raw))))
})

test_that("importance rescaling keeps the top value at 100", {
  # A positive maximum still scales to 100, whatever the other signs
  expect_equal(tl_rescale_importance(c(5, -8, 2.5)), c(100, -160, 50))
  # With nothing positive, the largest magnitude sets the scale
  expect_equal(tl_rescale_importance(c(-6.20, -0.87, -6.61)),
               100 * c(-6.20, -0.87, -6.61) / 6.61)
  # All zero has no scale; 0 / 0 gave NaN
  expect_equal(tl_rescale_importance(c(0, 0)), c(0, 0))
  expect_equal(tl_rescale_importance(numeric(0)), numeric(0))
})

test_that("importance comparison refuses repeated names", {
  # Two models under one name shared a bar position, so names = c("A", "A")
  # drew both models' bars in the same places
  tree <- tl_model(mtcars, mpg ~ wt + hp, method = "tree")
  lasso <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
  expect_error(
    tl_plot_importance_comparison(tree, lasso, names = c("A", "A")),
    "'names' must be unique"
  )
  expect_error(
    tl_plot_importance_comparison(tree, lasso, names = "A"),
    "Length of 'names' \\(1\\) must match the number of models \\(2\\)"
  )
  p <- tl_plot_importance_comparison(tree, lasso, names = c("A", "B"))
  expect_setequal(unique(p$data$model), c("A", "B"))
})

test_that("regularised importance works for a multiclass model", {
  model <- tl_model(iris, Species ~ ., method = "lasso")
  imp <- tl_get_importance_regularized(model)
  expect_type(imp$importance, "double")
  expect_true(all(imp$feature %in% names(iris)[1:4]))
  skip_if_not_installed("gt")
  expect_s3_class(tl_table_importance(model), "gt_tbl")
})

test_that("the dashboard importance panel dispatches regularised models", {
  expect_s3_class(tl_dashboard_importance_plot(
    tl_model(mtcars, mpg ~ ., method = "lasso")
  ), "ggplot")
  expect_s3_class(tl_dashboard_importance_plot(
    tl_model(mtcars, mpg ~ ., method = "tree")
  ), "ggplot")
})

test_that("importance comparison with no supported model says so", {
  m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
  m2 <- tl_model(mtcars, mpg ~ hp, method = "linear")
  expect_error(
    suppressWarnings(tl_plot_importance_comparison(m1, m2)),
    "None of the models"
  )
})

test_that("lift and gain draw the number of bins asked for", {
  am <- transform(mtcars, am = factor(am))
  model <- tl_model(am, am ~ wt, method = "logistic")
  lift <- ggplot2::ggplot_build(tl_plot_lift(model, bins = 10))$data[[1]]
  expect_equal(nrow(lift), 10)
  gain <- ggplot2::ggplot_build(tl_plot_gain(model, bins = 10))$data[[1]]
  expect_equal(nrow(gain), 11)
  expect_equal(utils::tail(gain$x, 1), 100)
})
# Tests for what the review of the medium fixes found.

test_that("the regularised importance plot shows the table's importance", {
  model <- tl_model(transform(mtcars, hp = hp / 100), mpg ~ wt + hp + qsec,
                    method = "ridge")
  plotted <- tl_plot_importance_regularized(model)$data
  tabled <- tl_get_importance_regularized(model)
  expect_equal(plotted$importance[match(tabled$feature, plotted$feature)],
               tabled$importance)
})

test_that("importance refuses a penalty outside the fitted path", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "ridge", lambda = 0.1)
  expect_error(tl_get_importance_regularized(model, lambda = 5),
               "outside the fitted path")
  expect_error(tl_plot_importance_regularized(model, lambda = 5),
               "outside the fitted path")
})

test_that("importance plots share the table's extraction", {
  forest <- tl_model(iris, Species ~ ., method = "forest", ntree = 50,
                     importance = FALSE)
  expect_s3_class(tl_plot_importance(forest), "ggplot")
  expect_s3_class(tl_dashboard_importance_plot(forest), "ggplot")
  skip_if_not_installed("xgboost")
  xgb <- tl_model(mtcars, mpg ~ wt + hp + qsec, method = "xgboost",
                  nrounds = 10)
  expect_s3_class(tl_plot_importance(xgb), "ggplot")
  expect_s3_class(tl_dashboard_importance_plot(xgb), "ggplot")
})

test_that("tl_table_coefficients reports every ignored argument", {
  skip_if_not_installed("gt")
  model <- tl_model(mtcars, mpg ~ wt, method = "linear")
  expect_warning(tl_table_coefficients(model, conf.int = TRUE),
                 "conf.int. Did you mean conf_int")
  # Every formal filled by position, so 99 lands in ... unnamed
  expect_warning(
    tl_table_coefficients(model, "1se", 4, FALSE, 0.95, FALSE, 99, foo = 1),
    "does not use: <unnamed>, foo"
  )
  expect_no_warning(tl_table_coefficients(model))
})

test_that("a model with every predictor penalised away says so", {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso", lambda = 100)
  expect_no_warning(imp <- tl_get_importance_regularized(model))
  expect_identical(nrow(imp), 0L)
  expect_error(tl_plot_importance_regularized(model),
               "dropped every predictor")
  skip_if_not_installed("gt")
  expect_error(tl_table_importance(model), "dropped every predictor")
})

test_that("lift and gain refuse a bin count that is not a whole number", {
  am <- transform(mtcars, am = factor(am))
  model <- tl_model(am, am ~ wt, method = "logistic")
  for (bins in list(0, 2.5, NA_real_, "10")) {
    expect_error(tl_plot_lift(model, bins = bins), "'bins' must be")
    expect_error(tl_plot_gain(model, bins = bins), "'bins' must be")
  }
})

# The y values of a chart's curve, found by geom rather than layer position
curve_values <- function(plot) {
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  ggplot2::layer_data(plot, which(geoms == "GeomLine"))$y
}

test_that("lift and gain read the classes the model was trained on", {
  # A test split of a subset still declares the class the subset dropped:
  # setosa made the response three-level, so both charts called the binary
  # model multiclass and refused it
  ib <- iris[iris$Species != "setosa", ]
  split <- tl_split(ib, prop = 0.7, seed = 1)
  model <- tl_model(split$train, Species ~ Sepal.Width + Petal.Length,
                    method = "logistic")
  # Rows with the same predictors tie, which the hand count below does not
  # model, so keep one of each
  test_rows <- split$test[
    !duplicated(split$test[, c("Sepal.Width", "Petal.Length")]),
  ]
  expect_identical(nlevels(test_rows$Species), 3L)

  gain <- tl_plot_gain(model, new_data = test_rows, bins = 4)
  lift <- tl_plot_lift(model, new_data = test_rows, bins = 4)

  # Cumulative share of virginica, the second class, ranked by its
  # probability, by hand
  probs <- predict(model, test_rows, type = "prob")$virginica
  expect_false(anyDuplicated(probs) > 0)
  ranked <- test_rows$Species[order(probs, decreasing = TRUE)] == "virginica"
  bin <- ceiling(seq_along(ranked) * 4 / length(ranked))
  responders <- cumsum(tapply(ranked, bin, sum))
  expect_equal(curve_values(gain),
               unname(c(0, 100 * responders / sum(ranked))))
  rows <- cumsum(tapply(ranked, bin, length))
  expect_equal(curve_values(lift),
               unname((responders / rows) / mean(ranked)))
})

test_that("the gain chart's positive class follows the model, not the data", {
  # Reordering the test factor's levels moved the positive class: with
  # levels c("1", "0") the chart ranked rows by P(am = 0) and counted the
  # zeros as responders
  am <- transform(mtcars, am = factor(am))
  model <- tl_model(am, am ~ wt, method = "logistic")
  relevelled <- am
  relevelled$am <- factor(relevelled$am, levels = c("1", "0"))
  expect_equal(curve_values(tl_plot_gain(model, new_data = relevelled)),
               curve_values(tl_plot_gain(model, new_data = am)))
  expect_equal(curve_values(tl_plot_lift(model, new_data = relevelled)),
               curve_values(tl_plot_lift(model, new_data = am)))
})

test_that("lift and gain leave out a class the model never saw, and say so", {
  ib <- droplevels(iris[iris$Species != "setosa", ])
  model <- tl_model(ib, Species ~ Sepal.Width + Petal.Length,
                    method = "logistic")
  with_setosa <- iris[c(1:5, 51:150), ]
  expect_warning(
    gain <- tl_plot_gain(model, new_data = with_setosa),
    "5 row\\(s\\) belong to a class the model was not trained on \\(setosa\\)"
  )
  expect_equal(curve_values(gain),
               curve_values(tl_plot_gain(model, new_data = iris[51:150, ])))
  expect_warning(
    tl_plot_lift(model, new_data = with_setosa),
    "not trained on \\(setosa\\)"
  )
})

test_that("lift and gain name a response column the data lacks", {
  # The missing column read as NULL, and the charts reported a model that
  # was not binary
  am <- transform(mtcars, am = factor(am))
  model <- tl_model(am, am ~ wt, method = "logistic")
  no_response <- am[, c("wt", "hp")]
  expect_error(tl_plot_lift(model, new_data = no_response),
               "Response variable 'am' not found in the evaluation data")
  expect_error(tl_plot_gain(model, new_data = no_response),
               "Response variable 'am' not found in the evaluation data")
})

test_that("lift and gain need a responder among the scored rows", {
  # With no row of the positive class every cumulative share is 0 / 0
  am <- transform(mtcars, am = factor(am))
  model <- tl_model(am, am ~ wt, method = "logistic")
  automatic <- am[am$am == "0", ]
  expect_error(tl_plot_gain(model, new_data = automatic),
               "no row of the positive class \\('1'\\)")
  expect_error(tl_plot_lift(model, new_data = automatic),
               "no row of the positive class \\('1'\\)")
})

test_that("multiclass models still get the binary-only message", {
  model <- tl_model(iris, Species ~ ., method = "tree")
  expect_error(tl_plot_lift(model),
               "only implemented for binary classification")
  expect_error(tl_plot_gain(model),
               "only implemented for binary classification")
})

# -- Dashboard panels, run through the server without a browser --
# testServer() attaches shiny and the diagnostics plots report their loess
# formula, so its calls run under suppressMessages()

test_that("the dashboard's residual panel draws for every regression method", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("shinydashboard")
  skip_if_not_installed("DT")
  # The panel passed the evaluation data to tl_plot_residuals() as its
  # plot type, and failed with "the condition has length > 1"
  models <- list(
    tl_model(mtcars, mpg ~ wt + hp, method = "linear"),
    tl_model(mtcars, mpg ~ wt + hp, method = "lasso"),
    tl_model(mtcars, mpg ~ wt + hp, method = "forest", ntree = 50)
  )
  for (model in models) {
    suppressMessages(shiny::testServer(tl_dashboard(model), {
      expect_no_error(output$residuals_plot)
    }))
  }

  # Residuals of the evaluation data, as the predictions table shows them
  test_rows <- mtcars[1:10, ]
  forest <- models[[3]]
  p <- tl_dashboard_residuals_plot(forest, test_rows)
  geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
  points <- ggplot2::layer_data(p, which(geoms == "GeomPoint"))
  predicted <- unname(predict(forest, test_rows)$.pred)
  expect_equal(points$x, predicted)
  expect_equal(points$y, test_rows$mpg - predicted)
})

test_that("the dashboard's predictions show the response the model fits", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("shinydashboard")
  skip_if_not_installed("DT")
  # The predictions table and the residual panel read the raw column, so a
  # log(mpg) ~ wt + hp model listed mpg beside log-scale predictions, with
  # residuals of mpg minus log(mpg)
  model <- tl_model(mtcars, log(mpg) ~ wt + hp, method = "linear")
  suppressMessages(shiny::testServer(tl_dashboard(model), {
    expect_no_error(output$predictions_table)
    expect_no_error(output$residuals_plot)
  }))

  table <- tl_dashboard_predictions(model, mtcars)
  expect_equal(table$actual, log(mtcars$mpg))
  expect_equal(table$predicted, unname(stats::fitted(model$fit)))
  expect_equal(table$residual, unname(stats::residuals(model$fit)))

  p <- tl_dashboard_residuals_plot(model, mtcars)
  geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
  points <- ggplot2::layer_data(p, which(geoms == "GeomPoint"))
  expect_equal(points$y, unname(stats::residuals(model$fit)))
})

test_that("the dashboard diagnostics panel shows four plots, or why not", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("shinydashboard")
  skip_if_not_installed("DT")
  skip_if_not_installed("gridExtra")
  # renderPlot() printed the list of four plots and each print replaced the
  # one before, so the panel showed only the last. Outside lm the panel
  # failed inside rstandard().
  linear <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  arranged <- suppressMessages(tl_dashboard_diagnostics_plot(linear))
  expect_s3_class(arranged, "gtable")
  expect_length(arranged$grobs, 4)
  suppressMessages(shiny::testServer(tl_dashboard(linear), {
    expect_no_error(output$diagnostics_plot)
  }))

  for (model in list(tl_model(mtcars, mpg ~ wt + hp, method = "lasso"),
                     tl_model(mtcars, mpg ~ wt + hp, method = "forest",
                              ntree = 50))) {
    suppressMessages(shiny::testServer(tl_dashboard(model), {
      expect_error(
        output$diagnostics_plot,
        "Diagnostic plots are available for linear and polynomial models only"
      )
    }))
  }
})

# -- Cross-validation plot --

test_that("tl_plot_cv_results(metrics =) draws only the metrics asked for", {
  # The fold lines were filtered but the mean lines were not, so
  # metrics = "rmse" still laid out mae, rmse and rsq panels
  set.seed(1)
  cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 5)
  p <- tl_plot_cv_results(cv, metrics = "rmse")
  expect_identical(nrow(ggplot2::ggplot_build(p)$layout$layout), 1L)

  geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
  mean_line <- ggplot2::layer_data(p, which(geoms == "GeomHline"))
  expect_equal(mean_line$yintercept,
               cv$summary$mean[cv$summary$metric == "rmse"])
  folds <- ggplot2::layer_data(p, which(geoms == "GeomLine"))
  expect_equal(
    folds$y,
    vapply(cv$folds, function(f) f$value[f$metric == "rmse"], numeric(1))
  )
})

test_that("tl_plot_cv_results says which requested metrics are missing", {
  set.seed(1)
  cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3)
  expect_warning(
    p <- tl_plot_cv_results(cv, metrics = c("rmse", "accuracy")),
    "not in the cross-validation results: accuracy"
  )
  expect_identical(nrow(ggplot2::ggplot_build(p)$layout$layout), 1L)
  expect_error(
    tl_plot_cv_results(cv, metrics = "accuracy"),
    "None of the requested metrics is in the cross-validation results"
  )
})

# -- Cluster helpers --

test_that("cluster plots keep the cluster column off the axes", {
  # A numeric cluster column listed first became the x axis
  set.seed(1)
  d <- data.frame(cluster = kmeans(iris[, 1:4], 3)$cluster, iris[, 1:4])
  point_data <- function(p) {
    geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
    ggplot2::layer_data(p, which(geoms == "GeomPoint")[1])
  }
  points <- point_data(plot_clusters(d))
  expect_equal(points$x, d$Sepal.Length)
  expect_equal(points$y, d$Sepal.Width)

  skip_if_not_installed("gridExtra")
  plots <- create_cluster_dashboard(d)
  points <- point_data(plots[[1]])
  expect_equal(points$x, d$Sepal.Length)
  expect_equal(points$y, d$Sepal.Width)
})

test_that("plot_clusters needs a numeric column besides the clusters", {
  d <- data.frame(group = rep(c("a", "b"), 10), cluster = rep(1:2, 10))
  expect_error(plot_clusters(d),
               "needs a numeric column other than the cluster column")
})

test_that("the cluster dashboard skips the panels it cannot draw", {
  skip_if_not_installed("gridExtra")
  # One numeric column left the scatter plot out and a NULL in its place,
  # which grid.arrange() could not draw
  set.seed(1)
  one_numeric <- data.frame(x = rnorm(20), cluster = factor(rep(1:2, 10)))
  expect_no_error(plots <- create_cluster_dashboard(one_numeric))
  expect_named(plots, "sizes")

  # Metrics computed without a distance matrix have no silhouette, and
  # reading the absent column warned
  d <- data.frame(iris[, 1:2], cluster = rep(1:3, 50))
  metrics <- tibble::tibble(k = 3L, min_size = 50L, max_size = 50L)
  expect_no_warning(
    plots <- create_cluster_dashboard(d, validation_metrics = metrics)
  )
  expect_named(plots, c("clusters", "sizes", "metrics"))
  geoms <- vapply(plots$metrics$layers, function(l) class(l$geom)[1],
                  character(1))
  label <- ggplot2::layer_data(plots$metrics, which(geoms == "GeomText"))$label
  expect_false(grepl("Silhouette", label))
  expect_match(label, "Number of Clusters: 3")
})

test_that("plot_dendrogram draws a tidylearn hclust model", {
  # plot() dispatched to plot.tidylearn_model, which has no main or xlab
  # argument, and failed with "unused arguments"
  model <- tl_model(USArrests, method = "hclust")
  drawn <- expect_invisible(plot_dendrogram(model, k = 3))
  expect_identical(drawn, model$fit$model)

  kmeans_model <- tl_model(USArrests, method = "kmeans", k = 3)
  expect_error(plot_dendrogram(kmeans_model),
               "needs a model fitted with method = \"hclust\"")
})

test_that("a single cluster is not called clusters", {
  skip_if_not_installed("gt")
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 1)
  expect_match(tl_table_clusters(model)[["_heading"]]$subtitle, "1 cluster$")
})
