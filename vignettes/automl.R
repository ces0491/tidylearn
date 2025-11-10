## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 5
)


## ----setup--------------------------------------------------------------------
library(tidylearn)
library(dplyr)
library(ggplot2)


## ----eval=FALSE---------------------------------------------------------------
# # Run AutoML on iris dataset
# result <- tl_auto_ml(iris, Species ~ .,
#                     task = "classification",
#                     time_budget = 60)
# 
# # View best model
# print(result$best_model)


## ----eval=FALSE---------------------------------------------------------------
# # View all models tried
# names(result$models)


## ----eval=FALSE---------------------------------------------------------------
# # View leaderboard
# result$leaderboard


## ----eval=FALSE---------------------------------------------------------------
# # Run AutoML on regression problem
# result_reg <- tl_auto_ml(mtcars, mpg ~ .,
#                         task = "regression",
#                         time_budget = 60)
# 
# # Best model
# print(result_reg$best_model)


## ----eval=FALSE---------------------------------------------------------------
# # AutoML with all features enabled
# result_full <- tl_auto_ml(
#   data = iris,
#   formula = Species ~ .,
#   task = "auto",                    # Automatically detect task type
#   use_reduction = TRUE,             # Try PCA preprocessing
#   use_clustering = TRUE,            # Add cluster features
#   time_budget = 120,                # 2 minutes
#   cv_folds = 5,                     # Cross-validation folds
#   metric = NULL                     # Auto-select metric
# )


## ----eval=FALSE---------------------------------------------------------------
# # Task type is automatically detected
# result_auto <- tl_auto_ml(iris, Species ~ ., task = "auto")
# # Detects: Classification (factor response)
# 
# result_auto_reg <- tl_auto_ml(mtcars, mpg ~ ., task = "auto")
# # Detects: Regression (numeric response)


## ----eval=FALSE---------------------------------------------------------------
# # Quick search (30 seconds)
# quick_result <- tl_auto_ml(iris, Species ~ ., time_budget = 30)
# 
# # Thorough search (10 minutes)
# thorough_result <- tl_auto_ml(iris, Species ~ ., time_budget = 600)


## ----eval=FALSE---------------------------------------------------------------
# # Disable dimensionality reduction
# no_reduction <- tl_auto_ml(iris, Species ~ .,
#                           use_reduction = FALSE,
#                           time_budget = 60)
# 
# # Disable cluster features
# no_clustering <- tl_auto_ml(iris, Species ~ .,
#                            use_clustering = FALSE,
#                            time_budget = 60)
# 
# # Baseline models only
# baseline_only <- tl_auto_ml(iris, Species ~ .,
#                            use_reduction = FALSE,
#                            use_clustering = FALSE,
#                            time_budget = 30)


## ----eval=FALSE---------------------------------------------------------------
# # Adjust cross-validation folds
# result_cv <- tl_auto_ml(iris, Species ~ .,
#                        cv_folds = 10,    # More folds = better estimate, slower
#                        time_budget = 120)
# 
# # Fewer folds for faster evaluation
# result_fast <- tl_auto_ml(iris, Species ~ .,
#                          cv_folds = 3,
#                          time_budget = 60)


## ----eval=FALSE---------------------------------------------------------------
# result <- tl_auto_ml(iris, Species ~ ., time_budget = 60)
# 
# # Best performing model
# best_model <- result$best_model
# 
# # All models trained
# all_models <- result$models
# 
# # Specific model
# baseline_logistic <- result$models$baseline_logistic
# pca_forest <- result$models$pca_forest


## ----eval=FALSE---------------------------------------------------------------
# # View performance comparison
# leaderboard <- result$leaderboard
# 
# # Sort by performance
# leaderboard <- leaderboard %>%
#   arrange(desc(performance))
# 
# print(leaderboard)


## ----eval=FALSE---------------------------------------------------------------
# # Use best model for predictions
# predictions <- predict(result$best_model, new_data = new_data)
# 
# # Or use a specific model
# predictions_pca <- predict(result$models$pca_forest, new_data = new_data)


## ----eval=FALSE---------------------------------------------------------------
# # Split data for evaluation
# split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)
# 
# # Run AutoML on training data
# automl_iris <- tl_auto_ml(split$train, Species ~ .,
#                          time_budget = 90,
#                          cv_folds = 5)
# 
# # Evaluate on test set
# test_preds <- predict(automl_iris$best_model, new_data = split$test)
# test_accuracy <- mean(test_preds$.pred == split$test$Species)
# 
# cat("AutoML Test Accuracy:", round(test_accuracy * 100, 1), "%\n")


## ----eval=FALSE---------------------------------------------------------------
# # Compare models
# for (model_name in names(automl_iris$models)) {
#   model <- automl_iris$models[[model_name]]
#   preds <- predict(model, new_data = split$test)
#   acc <- mean(preds$.pred == split$test$Species)
#   cat(model_name, ":", round(acc * 100, 1), "%\n")
# }


## ----eval=FALSE---------------------------------------------------------------
# # Split mtcars data
# split_mtcars <- tl_split(mtcars, prop = 0.7, seed = 42)
# 
# # Run AutoML
# automl_mpg <- tl_auto_ml(split_mtcars$train, mpg ~ .,
#                         task = "regression",
#                         time_budget = 90)
# 
# # Evaluate
# test_preds_mpg <- predict(automl_mpg$best_model, new_data = split_mtcars$test)
# rmse <- sqrt(mean((test_preds_mpg$.pred - split_mtcars$test$mpg)^2))
# 
# cat("AutoML Test RMSE:", round(rmse, 2), "\n")


## ----eval=FALSE---------------------------------------------------------------
# # Preprocess data first
# processed <- tl_prepare_data(
#   split$train,
#   Species ~ .,
#   scale_method = "standardize",
#   remove_correlated = TRUE
# )
# 
# # Run AutoML on preprocessed data
# automl_processed <- tl_auto_ml(processed$data, Species ~ .,
#                               time_budget = 60)
# 
# # Note: Need to apply same preprocessing to test data
# test_processed <- tl_prepare_data(
#   split$test,
#   Species ~ .,
#   scale_method = "standardize"
# )
# 
# test_preds_proc <- predict(automl_processed$best_model,
#                            new_data = test_processed$data)


## ----eval=FALSE---------------------------------------------------------------
# # Manual approach: choose one model
# manual_model <- tl_model(split$train, Species ~ ., method = "forest")
# manual_preds <- predict(manual_model, new_data = split$test)
# manual_acc <- mean(manual_preds$.pred == split$test$Species)
# 
# # AutoML approach
# automl_model <- tl_auto_ml(split$train, Species ~ ., time_budget = 60)
# automl_preds <- predict(automl_model$best_model, new_data = split$test)
# automl_acc <- mean(automl_preds$.pred == split$test$Species)
# 
# cat("Manual Selection:", round(manual_acc * 100, 1), "%\n")
# cat("AutoML:", round(automl_acc * 100, 1), "%\n")


## ----eval=FALSE---------------------------------------------------------------
# # First pass: quick exploration
# quick_automl <- tl_auto_ml(split$train, Species ~ ~.,
#                           time_budget = 30,
#                           use_reduction = TRUE,
#                           use_clustering = FALSE)
# 
# # Analyze what worked
# best_approach <- quick_automl$best_model$spec$method
# 
# # Second pass: focus on promising approaches
# if (grepl("pca", names(quick_automl$best_model)[1])) {
#   # If PCA worked well, focus on dimensionality reduction
#   refined_automl <- tl_auto_ml(split$train, Species ~ .,
#                               time_budget = 60,
#                               use_reduction = TRUE,
#                               use_clustering = TRUE)
# }


## ----eval=FALSE---------------------------------------------------------------
# # Get top 3 models
# top_models <- automl_iris$leaderboard %>%
#   arrange(desc(performance)) %>%
#   head(3)
# 
# # Make predictions with each
# ensemble_preds <- list()
# for (i in 1:nrow(top_models)) {
#   model_name <- top_models$model[i]
#   model <- automl_iris$models[[model_name]]
#   ensemble_preds[[i]] <- predict(model, new_data = split$test)$.pred
# }
# 
# # Majority vote for classification
# final_pred <- apply(do.call(cbind, ensemble_preds), 1, function(x) {
#   names(which.max(table(x)))
# })
# 
# ensemble_acc <- mean(final_pred == split$test$Species)
# cat("Ensemble Accuracy:", round(ensemble_acc * 100, 1), "%\n")


## ----eval=FALSE---------------------------------------------------------------
# # AutoML automatically uses accuracy for classification
# result_class <- tl_auto_ml(iris, Species ~ .,
#                           metric = "accuracy",
#                           time_budget = 60)


## ----eval=FALSE---------------------------------------------------------------
# # AutoML automatically uses RMSE for regression
# result_reg <- tl_auto_ml(mtcars, mpg ~ .,
#                         metric = "rmse",
#                         time_budget = 60)


## ----eval=FALSE---------------------------------------------------------------
# # Reduce time budget
# quick_result <- tl_auto_ml(data, formula, time_budget = 30)
# 
# # Reduce CV folds
# fast_result <- tl_auto_ml(data, formula, cv_folds = 3)
# 
# # Disable feature engineering
# baseline_result <- tl_auto_ml(data, formula,
#                              use_reduction = FALSE,
#                              use_clustering = FALSE)


## ----eval=FALSE---------------------------------------------------------------
# # Increase time budget
# thorough_result <- tl_auto_ml(data, formula, time_budget = 300)
# 
# # Ensure feature engineering is enabled
# full_result <- tl_auto_ml(data, formula,
#                          use_reduction = TRUE,
#                          use_clustering = TRUE)


## ----eval=FALSE---------------------------------------------------------------
# # Complete AutoML workflow
# workflow_split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)
# 
# automl_result <- tl_auto_ml(
#   data = workflow_split$train,
#   formula = Species ~ .,
#   task = "auto",
#   use_reduction = TRUE,
#   use_clustering = TRUE,
#   time_budget = 120,
#   cv_folds = 5
# )
# 
# # Evaluate best model
# final_preds <- predict(automl_result$best_model, new_data = workflow_split$test)
# final_accuracy <- mean(final_preds$.pred == workflow_split$test$Species)
# 
# cat("Final AutoML Accuracy:", round(final_accuracy * 100, 1), "%\n")
# cat("Best approach:", automl_result$best_model$spec$method, "\n")

