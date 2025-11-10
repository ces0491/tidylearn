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


## -----------------------------------------------------------------------------
# Create binary classification dataset
iris_binary <- iris %>%
  filter(Species %in% c("setosa", "versicolor")) %>%
  mutate(Species = droplevels(Species))

# Split data
split <- tl_split(iris_binary, prop = 0.7, stratify = "Species", seed = 123)


## -----------------------------------------------------------------------------
# Train logistic regression
model_logistic <- tl_model(split$train, Species ~ ., method = "logistic")
print(model_logistic)


## -----------------------------------------------------------------------------
# Predictions
preds_logistic <- predict(model_logistic, new_data = split$test)
head(preds_logistic)


## -----------------------------------------------------------------------------
# Train decision tree
model_tree <- tl_model(split$train, Species ~ ., method = "tree")
print(model_tree)

# Predictions
preds_tree <- predict(model_tree, new_data = split$test)


## -----------------------------------------------------------------------------
# Split full iris dataset
split_multi <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)


## -----------------------------------------------------------------------------
# Train random forest
model_forest <- tl_model(split_multi$train, Species ~ ., method = "forest")
print(model_forest)


## -----------------------------------------------------------------------------
# Predictions
preds_forest <- predict(model_forest, new_data = split_multi$test)
head(preds_forest)


## -----------------------------------------------------------------------------
# Accuracy on test set
mean(preds_forest$.pred == split_multi$test$Species)


## ----eval=FALSE---------------------------------------------------------------
# # Train SVM
# model_svm <- tl_model(split_multi$train, Species ~ ., method = "svm")
# print(model_svm)
# 
# # Predictions
# preds_svm <- predict(model_svm, new_data = split_multi$test)


## -----------------------------------------------------------------------------
# Split mtcars data
split_reg <- tl_split(mtcars, prop = 0.7, seed = 123)

# Train linear model
model_lm <- tl_model(split_reg$train, mpg ~ wt + hp + disp, method = "linear")
print(model_lm)


## -----------------------------------------------------------------------------
# Predictions
preds_lm <- predict(model_lm, new_data = split_reg$test)
head(preds_lm)


## -----------------------------------------------------------------------------
# Calculate RMSE
rmse <- sqrt(mean((preds_lm$.pred - split_reg$test$mpg)^2))
cat("RMSE:", round(rmse, 2), "\n")


## -----------------------------------------------------------------------------
# Polynomial regression for non-linear relationships
model_poly <- tl_model(split_reg$train, mpg ~ wt, method = "polynomial", degree = 2)
print(model_poly)


## -----------------------------------------------------------------------------
# Predictions
preds_poly <- predict(model_poly, new_data = split_reg$test)

# RMSE
rmse_poly <- sqrt(mean((preds_poly$.pred - split_reg$test$mpg)^2))
cat("Polynomial RMSE:", round(rmse_poly, 2), "\n")


## -----------------------------------------------------------------------------
# Train random forest for regression
model_rf_reg <- tl_model(split_reg$train, mpg ~ ., method = "forest")
print(model_rf_reg)


## -----------------------------------------------------------------------------
# Predictions
preds_rf <- predict(model_rf_reg, new_data = split_reg$test)

# RMSE
rmse_rf <- sqrt(mean((preds_rf$.pred - split_reg$test$mpg)^2))
cat("Random Forest RMSE:", round(rmse_rf, 2), "\n")


## ----eval=FALSE---------------------------------------------------------------
# # Ridge regression (L2 regularization)
# model_ridge <- tl_model(split_reg$train, mpg ~ ., method = "ridge")
# print(model_ridge)
# 
# # Predictions
# preds_ridge <- predict(model_ridge, new_data = split_reg$test)


## ----eval=FALSE---------------------------------------------------------------
# # LASSO (L1 regularization) - performs feature selection
# model_lasso <- tl_model(split_reg$train, mpg ~ ., method = "lasso")
# print(model_lasso)
# 
# # Predictions
# preds_lasso <- predict(model_lasso, new_data = split_reg$test)


## ----eval=FALSE---------------------------------------------------------------
# # Elastic Net - combines L1 and L2 regularization
# model_enet <- tl_model(split_reg$train, mpg ~ ., method = "elastic_net", alpha = 0.5)
# print(model_enet)
# 
# # Predictions
# preds_enet <- predict(model_enet, new_data = split_reg$test)


## -----------------------------------------------------------------------------
# Compare multiple models
models <- list(
  linear = tl_model(split_reg$train, mpg ~ ., method = "linear"),
  tree = tl_model(split_reg$train, mpg ~ ., method = "tree"),
  forest = tl_model(split_reg$train, mpg ~ ., method = "forest")
)


## -----------------------------------------------------------------------------
# Calculate RMSE for each model
results <- data.frame(
  Model = character(),
  RMSE = numeric(),
  stringsAsFactors = FALSE
)

for (model_name in names(models)) {
  preds <- predict(models[[model_name]], new_data = split_reg$test)
  rmse <- sqrt(mean((preds$.pred - split_reg$test$mpg)^2))

  results <- rbind(results, data.frame(
    Model = model_name,
    RMSE = rmse
  ))
}

results <- results %>% arrange(RMSE)
print(results)


## -----------------------------------------------------------------------------
# Preprocess data
processed <- tl_prepare_data(
  split_reg$train,
  mpg ~ .,
  scale_method = "standardize",
  remove_correlated = TRUE,
  correlation_cutoff = 0.9
)


## -----------------------------------------------------------------------------
# Train on preprocessed data
model_processed <- tl_model(processed$data, mpg ~ ., method = "linear")
print(model_processed)


## -----------------------------------------------------------------------------
# Interaction terms
model_interact <- tl_model(split_reg$train, mpg ~ wt * hp, method = "linear")

# Polynomial terms using I()
model_poly_manual <- tl_model(split_reg$train, mpg ~ wt + I(wt^2), method = "linear")

# Subset of predictors
model_subset <- tl_model(split_reg$train, mpg ~ wt + hp + disp, method = "linear")


## -----------------------------------------------------------------------------
# Create dataset with categorical variables
mtcars_cat <- mtcars %>%
  mutate(
    cyl = as.factor(cyl),
    gear = as.factor(gear),
    am = as.factor(am)
  )

split_cat <- tl_split(mtcars_cat, prop = 0.7, seed = 123)

# Model with categorical predictors
model_cat <- tl_model(split_cat$train, mpg ~ ., method = "forest")
print(model_cat)


## -----------------------------------------------------------------------------
# Create data with missing values
mtcars_missing <- mtcars
mtcars_missing[sample(1:nrow(mtcars_missing), 5), "hp"] <- NA
mtcars_missing[sample(1:nrow(mtcars_missing), 3), "wt"] <- NA

# Preprocess to handle missing values
processed_missing <- tl_prepare_data(
  mtcars_missing,
  mpg ~ .,
  impute_method = "mean",
  scale_method = "standardize"
)

# Train model
model_imputed <- tl_model(processed_missing$data, mpg ~ ., method = "linear")


## -----------------------------------------------------------------------------
# Complete workflow example
final_split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)
final_prep <- tl_prepare_data(final_split$train, Species ~ ., scale_method = "standardize")
final_model <- tl_model(final_prep$data, Species ~ ., method = "forest")
final_preds <- predict(final_model, new_data = final_split$test)

# Evaluate
accuracy <- mean(final_preds$.pred == final_split$test$Species)
cat("Test Accuracy:", round(accuracy * 100, 1), "%\n")

