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
