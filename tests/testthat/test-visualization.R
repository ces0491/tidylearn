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
