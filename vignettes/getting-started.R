## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 5
)


## ----eval = FALSE-------------------------------------------------------------
# # Install from GitHub
# # devtools::install_github("ces0491/tidylearn")
# 
# # Or install from source
# devtools::install_local("path/to/tidylearn")


## ----setup--------------------------------------------------------------------
library(tidylearn)
library(dplyr)


## -----------------------------------------------------------------------------
# Classification with logistic regression
model_logistic <- tl_model(iris, Species ~ ., method = "logistic")
print(model_logistic)


## -----------------------------------------------------------------------------
# Make predictions
predictions <- predict(model_logistic)
head(predictions)


## -----------------------------------------------------------------------------
# Regression with linear model
model_linear <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
print(model_linear)


## -----------------------------------------------------------------------------
# Predictions
predictions_reg <- predict(model_linear)
head(predictions_reg)


## -----------------------------------------------------------------------------
# Principal Component Analysis
model_pca <- tl_model(iris[, 1:4], method = "pca")
print(model_pca)


## -----------------------------------------------------------------------------
# Transform data
transformed <- predict(model_pca)
head(transformed)


## -----------------------------------------------------------------------------
# K-means clustering
model_kmeans <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
print(model_kmeans)


## -----------------------------------------------------------------------------
# Get cluster assignments
clusters <- model_kmeans$fit$clusters
head(clusters)


## -----------------------------------------------------------------------------
# Compare with actual species
table(clusters$cluster, iris$Species)


## -----------------------------------------------------------------------------
# Prepare data with multiple preprocessing steps
processed <- tl_prepare_data(
  iris,
  Species ~ .,
  impute_method = "mean",
  scale_method = "standardize",
  encode_categorical = FALSE
)


## -----------------------------------------------------------------------------
# Check preprocessing steps applied
names(processed$preprocessing_steps)


## -----------------------------------------------------------------------------
# Use processed data for modeling
model_processed <- tl_model(processed$data, Species ~ ., method = "forest")


## -----------------------------------------------------------------------------
# Simple random split
split <- tl_split(iris, prop = 0.7, seed = 123)

# Train model
model_train <- tl_model(split$train, Species ~ ., method = "logistic")

# Test predictions
predictions_test <- predict(model_train, new_data = split$test)
head(predictions_test)


## -----------------------------------------------------------------------------
# Stratified split (maintains class proportions)
split_strat <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)

# Check proportions are maintained
prop.table(table(split_strat$train$Species))
prop.table(table(split_strat$test$Species))
prop.table(table(iris$Species))


## -----------------------------------------------------------------------------
# Quick example combining everything
data_split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)
data_prep <- tl_prepare_data(data_split$train, Species ~ ., scale_method = "standardize")
model_final <- tl_model(data_prep$data, Species ~ ., method = "forest")
test_preds <- predict(model_final, new_data = data_split$test)

print(model_final)

