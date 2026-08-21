# Integration Workflows: Combining Supervised and Unsupervised Learning

``` r

library(tidylearn)
library(dplyr)
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
library(ggplot2)
```

## Introduction

This vignette covers the functions that combine supervised and
unsupervised learning into multi-step workflows.

**Important:** These integration functions orchestrate the wrapped
packages (stats, glmnet, randomForest, cluster, etc.) rather than
implementing new algorithms. tidylearn provides the workflow
coordination, while the underlying packages do the computational work.
Access raw model objects via `model$fit`.

## Dimensionality Reduction as Preprocessing

Use PCA or MDS to reduce feature space before supervised learning.

### Basic Usage

``` r

# Reduce dimensions before classification
reduced <- tl_reduce_dimensions(iris,
                                response = "Species",
                                method = "pca",
                                n_components = 3)

# Inspect reduced data
head(reduced$data)
#> # A tibble: 6 × 4
#>     PC1    PC2     PC3 Species
#>   <dbl>  <dbl>   <dbl> <fct>  
#> 1 -2.26 -0.478  0.127  setosa 
#> 2 -2.07  0.672  0.234  setosa 
#> 3 -2.36  0.341 -0.0441 setosa 
#> 4 -2.29  0.595 -0.0910 setosa 
#> 5 -2.38 -0.645 -0.0157 setosa 
#> 6 -2.07 -1.48  -0.0269 setosa
```

``` r

# Train classifier on the reduced features. iris has three species, so this
# needs a multiclass-capable method -- logistic regression is binary only.
model_reduced <- tl_model(reduced$data, Species ~ ., method = "forest")
print(model_reduced)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 150
```

``` r

# Training-set accuracy. Use tl_evaluate() rather than comparing $.pred by
# hand: what $.pred holds depends on the method and prediction type.
tl_evaluate(model_reduced)
#> # A tibble: 1 × 2
#>   metric   value
#>   <chr>    <dbl>
#> 1 accuracy     1
```

### Comparison: Original vs Reduced Features

``` r

# Split data for fair comparison
split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)

# Model with original features
model_original <- tl_model(split$train, Species ~ ., method = "forest")
eval_original <- tl_evaluate(model_original, new_data = split$test)

# Model with PCA features
reduced_train <- tl_reduce_dimensions(split$train,
                                      response = "Species",
                                      method = "pca",
                                      n_components = 3)

model_pca <- tl_model(reduced_train$data, Species ~ ., method = "forest")

# The test set must be projected through the PCA fitted on the training
# data -- refitting PCA on the test set would leak information
test_transformed <- predict(
  reduced_train$reduction_model,
  new_data = split$test %>% select(-Species)
)
test_transformed$Species <- split$test$Species

eval_pca <- tl_evaluate(model_pca, new_data = test_transformed)

# Compare results
acc <- function(x) round(x$value[x$metric == "accuracy"] * 100, 1)
n_original <- ncol(split$train) - 1
n_reduced <- sum(grepl("^PC", names(reduced_train$data)))
cat("Original features:", n_original, "->", acc(eval_original), "%\n")
#> Original features: 4 -> 93.3 %
cat("PCA features:", n_reduced, "->", acc(eval_pca), "%\n")
#> PCA features: 3 -> 91.1 %
cat("Feature reduction:",
    round((1 - n_reduced / n_original) * 100, 1), "%\n")
#> Feature reduction: 25 %
```

## Cluster-Based Feature Engineering

Add cluster assignments as a feature, so a model that cannot express
group structure directly gets a column that encodes it.

### Adding Cluster Features

``` r

# Add cluster features
data_clustered <- tl_add_cluster_features(iris,
                                          response = "Species",
                                          method = "kmeans",
                                          k = 3)

# Check new features
names(data_clustered)
#> [1] "Sepal.Length"   "Sepal.Width"    "Petal.Length"   "Petal.Width"   
#> [5] "Species"        "cluster_kmeans"
```

``` r

# Train model with cluster features
model_cluster <- tl_model(data_clustered, Species ~ ., method = "forest")
print(model_cluster)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 150
```

### Performance Comparison

``` r

# Compare models with and without cluster features
split_comp <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)

# Without cluster features
model_no_cluster <- tl_model(split_comp$train, Species ~ ., method = "forest")
preds_no_cluster <- predict(model_no_cluster, new_data = split_comp$test)
acc_no_cluster <- mean(preds_no_cluster$.pred == split_comp$test$Species)

# With cluster features
train_clustered <- tl_add_cluster_features(split_comp$train,
                                           response = "Species",
                                           method = "kmeans",
                                           k = 3)
model_with_cluster <- tl_model(train_clustered, Species ~ ., method = "forest")

# Need to get cluster model for test data
cluster_model <- attr(train_clustered, "cluster_model")
test_clusters <- predict(cluster_model, new_data = split_comp$test[, -5])
test_clustered <- split_comp$test
test_clustered$cluster_kmeans <- as.factor(test_clusters$cluster)

preds_with_cluster <- predict(model_with_cluster, new_data = test_clustered)
acc_with_cluster <- mean(preds_with_cluster$.pred == split_comp$test$Species)

cat("Without cluster features:", round(acc_no_cluster * 100, 1), "%\n")
#> Without cluster features: 93.3 %
cat("With cluster features:", round(acc_with_cluster * 100, 1), "%\n")
#> With cluster features: 93.3 %
```

## Semi-Supervised Learning

Train models with limited labels using cluster-based label propagation.

### Training with Limited Labels

``` r

# Use only 10% of labels
set.seed(123)
labeled_indices <- sample(nrow(iris), size = 15)  # Only 15 out of 150 labeled!

# Train semi-supervised model. supervised_method defaults to "logistic",
# which is binary only -- iris has three species, so name a multiclass
# method explicitly.
model_semi <- tl_semisupervised(iris, Species ~ .,
                                labeled_indices = labeled_indices,
                                cluster_method = "kmeans",
                                supervised_method = "forest")

print(model_semi)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 150
```

``` r

# Check how labels were propagated
label_mapping <- model_semi$semisupervised_info$label_mapping
print(label_mapping)
#> # A tibble: 3 × 2
#>   cluster cluster_label
#>     <int> <chr>        
#> 1       1 virginica    
#> 2       2 versicolor   
#> 3       3 setosa
```

``` r

# Evaluate against the true labels
preds_semi <- predict(model_semi, new_data = iris, type = "class")
accuracy_semi <- mean(preds_semi$.pred == iris$Species)

cat("Accuracy with only", length(labeled_indices), "labels:",
    round(accuracy_semi * 100, 1), "%\n")
#> Accuracy with only 15 labels: 90 %
labeled_pct <- round(
  length(labeled_indices) / nrow(iris) * 100, 1
)
cat("Proportion of data labeled:", labeled_pct, "%\n")
#> Proportion of data labeled: 10 %
```

### Comparison: Semi-Supervised vs Fully Supervised

``` r

# Fully supervised with same amount of data
labeled_data <- iris[labeled_indices, ]
model_full <- tl_model(labeled_data, Species ~ ., method = "forest")
preds_full <- predict(model_full, new_data = iris, type = "class")
accuracy_full <- mean(preds_full$.pred == iris$Species)

cat("Fully supervised (15 samples):", round(accuracy_full * 100, 1), "%\n")
#> Fully supervised (15 samples): 94.7 %
cat("Semi-supervised (15 labels + propagation):",
    round(accuracy_semi * 100, 1), "%\n")
#> Semi-supervised (15 labels + propagation): 90 %
```

## Anomaly-Aware Modeling

Detect and handle outliers before supervised learning.

### Flagging Anomalies

``` r

# Flag anomalies as a feature
model_anomaly_flag <- tl_anomaly_aware(iris, Species ~ .,
                                       response = "Species",
                                       anomaly_method = "dbscan",
                                       action = "flag",
                                       supervised_method = "forest")

# Check anomaly info
cat("Anomalies detected:", model_anomaly_flag$anomaly_info$n_anomalies, "\n")
```

### Removing Anomalies

``` r

# Remove anomalies before training
model_anomaly_remove <- tl_anomaly_aware(iris, Species ~ .,
                                         response = "Species",
                                         anomaly_method = "dbscan",
                                         action = "remove",
                                         supervised_method = "forest")

cat("Anomalies removed:", model_anomaly_remove$anomalies_removed, "\n")
```

## Stratified Models

Create cluster-specific models for heterogeneous data.

### Training Stratified Models

``` r

# Train separate models for different clusters
stratified_models <- tl_stratified_models(mtcars, mpg ~ .,
                                          cluster_method = "kmeans",
                                          k = 3,
                                          supervised_method = "linear")
#> Note: Response 'mpg' has 6 unique numeric values. Treating as regression. Convert to factor for classification.
#> Note: Response 'mpg' has 8 unique numeric values. Treating as regression. Convert to factor for classification.

# Check structure
names(stratified_models)
#> [1] "cluster_model"     "supervised_models" "formula"          
#> [4] "data"
length(stratified_models$supervised_models)
#> [1] 3
```

``` r

# Predictions using stratified models
preds_stratified <- predict(stratified_models)
head(preds_stratified)
#> # A tibble: 6 × 2
#>   .pred .cluster
#>   <dbl>    <int>
#> 1  20.7        3
#> 2  20.5        3
#> 3  24.7        3
#> 4  21.4        1
#> 5  19.2        2
#> 6  18.1        1
```

``` r

# Calculate RMSE
rmse_stratified <- sqrt(mean((preds_stratified$.pred - mtcars$mpg)^2))
cat("Stratified Model RMSE:", round(rmse_stratified, 2), "\n")
#> Stratified Model RMSE: 1.06

# Compare with single model
model_single <- tl_model(mtcars, mpg ~ ., method = "linear")
preds_single <- predict(model_single)
rmse_single <- sqrt(mean((preds_single$.pred - mtcars$mpg)^2))
cat("Single Model RMSE:", round(rmse_single, 2), "\n")
#> Single Model RMSE: 2.15
```

## Complete Integration Workflow

Combining multiple integration techniques:

``` r

# Step 1: Split data
workflow_split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)

# Step 2: Reduce dimensions
workflow_reduced <- tl_reduce_dimensions(workflow_split$train,
                                         response = "Species",
                                         method = "pca",
                                         n_components = 3)

# Step 3: Add cluster features to reduced data
workflow_clustered <- tl_add_cluster_features(workflow_reduced$data,
                                              response = "Species",
                                              method = "kmeans",
                                              k = 3)

# Step 4: Train final model
workflow_model <- tl_model(workflow_clustered, Species ~ ., method = "forest")

print(workflow_model)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 105
```

``` r

# Transform test data through same pipeline
# 1. Apply PCA transformation
test_pca <- predict(workflow_reduced$reduction_model,
                    new_data = workflow_split$test[, -5])
test_pca$Species <- workflow_split$test$Species

# 2. Get cluster assignments. The cluster model was fitted on the PC
# columns; predict() matches new_data to those columns by name and
# errors on a mismatch rather than assigning against the wrong ones.
cluster_model_wf <- attr(workflow_clustered, "cluster_model")
test_clusters_wf <- predict(cluster_model_wf, new_data = test_pca)
test_pca$cluster_kmeans <- as.factor(test_clusters_wf$cluster)

# 3. Predict
workflow_preds <- predict(workflow_model, new_data = test_pca)
workflow_accuracy <- mean(workflow_preds$.pred == workflow_split$test$Species)

cat("Complete Workflow Accuracy:", round(workflow_accuracy * 100, 1), "%\n")
#> Complete Workflow Accuracy: 88.9 %
```

## Practical Example: Credit Risk Assessment

``` r

# Simulate credit data
set.seed(42)
n <- 500
credit_data <- data.frame(
  age = rnorm(n, 40, 12),
  income = rnorm(n, 50000, 20000),
  debt_ratio = runif(n, 0, 0.5),
  credit_score = rnorm(n, 700, 100),
  years_employed = rpois(n, 5)
)

# Create target variable (default risk)
credit_data$default <- factor(
  ifelse(
    credit_data$debt_ratio > 0.4 & credit_data$credit_score < 650,
    "Yes", "No"
  )
)

# Split data
credit_split <- tl_split(
  credit_data, prop = 0.7, stratify = "default", seed = 123
)
```

``` r

# Strategy 1: Add customer segments as features
credit_clustered <- tl_add_cluster_features(credit_split$train,
                                            response = "default",
                                            method = "kmeans",
                                            k = 4)

model_credit <- tl_model(credit_clustered, default ~ ., method = "forest")

# Transform test data
cluster_model_credit <- attr(credit_clustered, "cluster_model")
test_clusters_credit <- predict(cluster_model_credit,
                                new_data = credit_split$test[, -6])
test_credit <- credit_split$test
test_credit$cluster_kmeans <- as.factor(test_clusters_credit$cluster)

preds_credit <- predict(model_credit, new_data = test_credit)
accuracy_credit <- mean(preds_credit$.pred == credit_split$test$default)

cat("Credit Risk Model Accuracy:", round(accuracy_credit * 100, 1), "%\n")
#> Credit Risk Model Accuracy: 98.7 %
```

## What Each Combination Buys You

- **Dimensionality reduction before fitting** trades some accuracy for a
  smaller feature space. Whether the trade is worth making is empirical
  – run the comparison above on your own data.
- **Cluster features** give a model a handle on group structure it
  cannot otherwise express. On data with no group structure they add
  noise.
- **Semi-supervised learning** is worth reaching for when labels are
  expensive and unlabelled observations are plentiful.
- **Anomaly-aware modelling** decides what happens to outliers
  explicitly rather than leaving it to the loss function.
- **Stratified models** fit one model per cluster, which reads more
  easily than a single model with many interaction terms, and needs
  enough observations in every cluster to be estimable.

## Best Practices

1.  **Always transform test data** using the same unsupervised models as
    training data
2.  **Experiment with different combinations** to find the best approach
3.  **Use semi-supervised learning** when labels are expensive to obtain
4.  **Consider stratified models** for heterogeneous datasets
5.  **Validate performance** on held-out test data

## Summary

tidylearn’s integration functions combine supervised and unsupervised
learning:

- **[`tl_reduce_dimensions()`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md)**:
  Use PCA/MDS as preprocessing
- **[`tl_add_cluster_features()`](https://tidylearn.sheetsolved.com/reference/tl_add_cluster_features.md)**:
  Engineer features from clusters
- **[`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)**:
  Train with limited labels
- **[`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md)**:
  Handle outliers intelligently
- **[`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)**:
  Create cluster-specific models

These functions orchestrate the underlying packages (stats, cluster,
glmnet, randomForest, etc.) to enable multi-step workflows with a
consistent interface.

``` r

# Final integrated example
final_data <- iris
final_split <- tl_split(
  final_data, prop = 0.7, stratify = "Species", seed = 999
)

# Combine PCA + clustering
final_reduced <- tl_reduce_dimensions(final_split$train,
                                      response = "Species",
                                      method = "pca",
                                      n_components = 3)
final_clustered <- tl_add_cluster_features(final_reduced$data,
                                           response = "Species",
                                           method = "kmeans",
                                           k = 3)
final_model <- tl_model(final_clustered, Species ~ ., method = "forest")

print(final_model)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 105
```
