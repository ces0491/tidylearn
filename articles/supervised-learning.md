# Supervised Learning with tidylearn

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
```

## Introduction

This vignette demonstrates supervised learning capabilities in
tidylearn. All methods shown here wrap established R packages - the
algorithms are unchanged, tidylearn simply provides a consistent
interface and tidy output.

**Wrapped packages include:**

- stats ([`lm()`](https://rdrr.io/r/stats/lm.html),
  [`glm()`](https://rdrr.io/r/stats/glm.html)) for linear and logistic
  regression
- rpart for decision trees
- randomForest for random forests
- gbm and xgboost for gradient boosting
- glmnet for regularization (ridge, lasso, elastic net)
- e1071 for support vector machines
- nnet for neural networks

Access raw model objects via `model$fit` for package-specific
functionality.

## Classification

### Binary Classification

Let’s create a binary classification problem from the iris dataset:

``` r

# Create binary classification dataset. setosa is linearly separable from
# the other two species, and logistic regression has no finite maximum
# likelihood estimate on separable data -- glm() fits, warns that the
# algorithm did not converge, and returns coefficients that diverged.
# versicolor and virginica overlap, so this is a real classification
# problem. Even here, a 70% split of 100 rows is separable at some seeds;
# this one is not.
iris_binary <- iris %>%
  filter(Species %in% c("versicolor", "virginica")) %>%
  mutate(Species = droplevels(Species))

# Split data
split <- tl_split(iris_binary, prop = 0.7, stratify = "Species", seed = 42)
```

#### Logistic Regression

``` r

# Train logistic regression
model_logistic <- tl_model(split$train, Species ~ ., method = "logistic")
print(model_logistic)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: logistic 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 70
```

The `type` argument controls what `.pred` holds. This matters: the
default `"response"` returns **probabilities** for logistic regression
but **class labels** for trees, forests and SVMs. Ask for
`type = "class"` when you want labels regardless of method.

``` r

# Predicted labels
preds_logistic <- predict(model_logistic, new_data = split$test, type = "class")
head(preds_logistic)
#> # A tibble: 6 × 1
#>   .pred     
#>   <fct>     
#> 1 versicolor
#> 2 versicolor
#> 3 versicolor
#> 4 versicolor
#> 5 versicolor
#> 6 versicolor
```

``` r

# Class probabilities
head(predict(model_logistic, new_data = split$test, type = "prob"))
#> # A tibble: 6 × 2
#>   versicolor  virginica
#>        <dbl>      <dbl>
#> 1      1.000 0.0000690 
#> 2      1.000 0.0000917 
#> 3      0.999 0.00101   
#> 4      1.000 0.00000874
#> 5      1.000 0.000208  
#> 6      0.952 0.0480
```

[`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
resolves this for you — it always scores against class labels, and
computes whichever metrics you ask for:

``` r

tl_evaluate(
  model_logistic,
  new_data = split$test,
  metrics = c("accuracy", "precision", "recall", "f1", "auc")
)
#> # A tibble: 5 × 2
#>   metric    value
#>   <chr>     <dbl>
#> 1 accuracy  0.867
#> 2 precision 0.789
#> 3 recall    1    
#> 4 f1        0.882
#> 5 auc       0.991
```

#### Decision Trees

``` r

# Train decision tree
model_tree <- tl_model(split$train, Species ~ ., method = "tree")
print(model_tree)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: tree 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 70

# Predictions
preds_tree <- predict(model_tree, new_data = split$test)
```

### Multi-class Classification

``` r

# Split full iris dataset
split_multi <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)
```

#### Random Forest

``` r

# Train random forest
model_forest <- tl_model(split_multi$train, Species ~ ., method = "forest")
print(model_forest)
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

# Predictions
preds_forest <- predict(model_forest, new_data = split_multi$test)
head(preds_forest)
#> # A tibble: 6 × 1
#>   .pred 
#>   <fct> 
#> 1 setosa
#> 2 setosa
#> 3 setosa
#> 4 setosa
#> 5 setosa
#> 6 setosa
```

``` r

# Accuracy on test set
mean(preds_forest$.pred == split_multi$test$Species)
#> [1] 0.9333333
```

#### Support Vector Machines

``` r

# Train SVM
model_svm <- tl_model(split_multi$train, Species ~ ., method = "svm")
print(model_svm)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: svm 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 105

# Predictions
preds_svm <- predict(model_svm, new_data = split_multi$test)
```

## Regression

### Linear Regression

``` r

# Split mtcars data
split_reg <- tl_split(mtcars, prop = 0.7, seed = 123)

# Train linear model
model_lm <- tl_model(split_reg$train, mpg ~ wt + hp + disp, method = "linear")
print(model_lm)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: linear 
#> Task: Regression 
#> Formula: mpg ~ wt + hp + disp 
#> 
#> Training observations: 22
```

``` r

# Predictions
preds_lm <- predict(model_lm, new_data = split_reg$test)
head(preds_lm)
#> # A tibble: 6 × 1
#>   .pred
#>   <dbl>
#> 1  24.0
#> 2  23.1
#> 3  21.2
#> 4  20.7
#> 5  16.0
#> 6  17.2
```

``` r

# Calculate RMSE
rmse <- sqrt(mean((preds_lm$.pred - split_reg$test$mpg)^2))
cat("RMSE:", round(rmse, 2), "\n")
#> RMSE: 2.16
```

### Polynomial Regression

``` r

# Polynomial regression for non-linear relationships
model_poly <- tl_model(
  split_reg$train, mpg ~ wt,
  method = "polynomial", degree = 2
)
print(model_poly)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: polynomial 
#> Task: Regression 
#> Formula: mpg ~ wt 
#> 
#> Training observations: 22
```

``` r

# Predictions
preds_poly <- predict(model_poly, new_data = split_reg$test)

# RMSE
rmse_poly <- sqrt(mean((preds_poly$.pred - split_reg$test$mpg)^2))
cat("Polynomial RMSE:", round(rmse_poly, 2), "\n")
#> Polynomial RMSE: 2.09
```

### Random Forest Regression

``` r

# Train random forest for regression
model_rf_reg <- tl_model(split_reg$train, mpg ~ ., method = "forest")
print(model_rf_reg)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Regression 
#> Formula: mpg ~ . 
#> 
#> Training observations: 22
```

``` r

# Predictions
preds_rf <- predict(model_rf_reg, new_data = split_reg$test)

# RMSE
rmse_rf <- sqrt(mean((preds_rf$.pred - split_reg$test$mpg)^2))
cat("Random Forest RMSE:", round(rmse_rf, 2), "\n")
#> Random Forest RMSE: 1.97
```

## Regularized Regression

Regularization helps prevent overfitting by adding penalties to model
complexity.

### Ridge Regression

``` r

# Ridge regression (L2 regularization)
model_ridge <- tl_model(split_reg$train, mpg ~ ., method = "ridge")
print(model_ridge)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: ridge 
#> Task: Regression 
#> Formula: mpg ~ . 
#> 
#> Training observations: 22

# Predictions
preds_ridge <- predict(model_ridge, new_data = split_reg$test)
```

### LASSO

``` r

# LASSO (L1 regularization) - performs feature selection
model_lasso <- tl_model(split_reg$train, mpg ~ ., method = "lasso")
print(model_lasso)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: lasso 
#> Task: Regression 
#> Formula: mpg ~ . 
#> 
#> Training observations: 22

# Predictions
preds_lasso <- predict(model_lasso, new_data = split_reg$test)
```

### Elastic Net

``` r

# Elastic Net - combines L1 and L2 regularization
model_enet <- tl_model(
  split_reg$train, mpg ~ .,
  method = "elastic_net", alpha = 0.5
)
print(model_enet)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: elastic_net 
#> Task: Regression 
#> Formula: mpg ~ . 
#> 
#> Training observations: 22

# Predictions
preds_enet <- predict(model_enet, new_data = split_reg$test)
```

## Model Comparison

``` r

# Compare multiple models
models <- list(
  linear = tl_model(split_reg$train, mpg ~ ., method = "linear"),
  tree = tl_model(split_reg$train, mpg ~ ., method = "tree"),
  forest = tl_model(split_reg$train, mpg ~ ., method = "forest")
)
```

``` r

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
#>    Model     RMSE
#> 1 forest 2.021532
#> 2 linear 2.281450
#> 3   tree 4.095888
```

## Advanced Features

### Using Preprocessed Data

``` r

# Preprocess data
processed <- tl_prepare_data(
  split_reg$train,
  mpg ~ .,
  scale_method = "standardize",
  remove_correlated = TRUE,
  correlation_cutoff = 0.9
)
#> Removing 1 highly correlated features
#> Scaling numeric features using method: standardize
```

``` r

# Train on preprocessed data
model_processed <- tl_model(processed$data, mpg ~ ., method = "linear")
print(model_processed)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: linear 
#> Task: Regression 
#> Formula: mpg ~ . 
#> 
#> Training observations: 22
```

### Formula Variations

``` r

# Interaction terms
model_interact <- tl_model(split_reg$train, mpg ~ wt * hp, method = "linear")

# Polynomial terms using I()
model_poly_manual <- tl_model(
  split_reg$train, mpg ~ wt + I(wt^2), method = "linear"
)

# Subset of predictors
model_subset <- tl_model(
  split_reg$train, mpg ~ wt + hp + disp, method = "linear"
)
```

## Handling Different Data Types

### Categorical Predictors

``` r

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
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: forest 
#> Task: Regression 
#> Formula: mpg ~ . 
#> 
#> Training observations: 22
```

### Missing Values

``` r

# Create data with missing values. Seeded so the vignette renders the
# same output on every build.
set.seed(123)
mtcars_missing <- mtcars
mtcars_missing[sample(seq_len(nrow(mtcars_missing)), 5), "hp"] <- NA
mtcars_missing[sample(seq_len(nrow(mtcars_missing)), 3), "wt"] <- NA

# Preprocess to handle missing values
processed_missing <- tl_prepare_data(
  mtcars_missing,
  mpg ~ .,
  impute_method = "mean",
  scale_method = "standardize"
)
#> Imputing missing values using method: mean
#> Scaling numeric features using method: standardize

# Train model
model_imputed <- tl_model(processed_missing$data, mpg ~ ., method = "linear")
```

## Best Practices

1.  **Split before training.** A metric computed on the rows the model
    was fitted to measures memorisation, not performance.
2.  **Stratify classification splits** so both sets carry the same class
    proportions as the source data.
3.  **Scale inputs for methods that need it** – regularised regression,
    SVM and neural networks – and replay the training transformation on
    the test set, as shown below. Trees and forests are scale-invariant.
4.  **Compare several models** on the same split before committing to
    one.
5.  **Reach for regularisation** when predictors outnumber observations,
    or when they are strongly correlated.
6.  **Match the metric to the task** – accuracy, F1 or AUC for
    classification; RMSE or MAE for regression.

## Summary

tidylearn provides a unified interface for supervised learning:

- **Classification**: Logistic regression, decision trees, random
  forests, SVM, etc.
- **Regression**: Linear, polynomial, random forests, regularized
  methods
- **Preprocessing**: Integrated data preparation tools
- **Consistent API**: Same function
  ([`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md))
  for all methods
- **Tidy Output**: Easy-to-use predictions and model objects

``` r

# Complete workflow example
final_split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)

final_prep <- tl_prepare_data(
  final_split$train, Species ~ .,
  scale_method = "standardize"
)
#> Scaling numeric features using method: standardize
final_model <- tl_model(final_prep$data, Species ~ ., method = "forest")
```

A model trained on scaled features expects scaled inputs at prediction
time.
[`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
keeps the parameters it learned, so the same transformation can be
replayed on the test set — applying the *training* centre and scale,
never recomputing them from the test data:

``` r

scaling <- final_prep$preprocessing_steps$scaling$scaling_params

final_test <- final_split$test
for (col in names(scaling)) {
  final_test[[col]] <-
    (final_test[[col]] - scaling[[col]]$mean) / scaling[[col]]$sd
}

final_preds <- predict(final_model, new_data = final_test)
accuracy <- mean(final_preds$.pred == final_split$test$Species)
cat("Test Accuracy:", round(accuracy * 100, 1), "%\n")
#> Test Accuracy: 95.6 %
```

Skipping that step and predicting on raw test data is a common and quiet
mistake — the model still returns predictions, they are just wrong. For
multi-step preprocessing,
[`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
handles this bookkeeping for you:
[`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
replays the training preprocessing automatically.
