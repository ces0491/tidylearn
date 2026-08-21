# Data Preprocessing for tidylearn

Unified preprocessing functions that work with both supervised and
unsupervised workflows Prepare Data for Machine Learning

## Usage

``` r
tl_prepare_data(
  data,
  formula = NULL,
  impute_method = "mean",
  scale_method = "standardize",
  encode_categorical = TRUE,
  remove_zero_variance = TRUE,
  remove_correlated = FALSE,
  correlation_cutoff = 0.95
)
```

## Arguments

- data:

  A data frame

- formula:

  Optional formula (for supervised learning)

- impute_method:

  Method for missing value imputation: "mean", "median", "mode", "knn"

- scale_method:

  Scaling method: "standardize", "normalize", "robust", "none"

- encode_categorical:

  Whether to encode categorical variables (default: TRUE)

- remove_zero_variance:

  Remove zero-variance features (default: TRUE)

- remove_correlated:

  Remove highly correlated features (default: FALSE)

- correlation_cutoff:

  Correlation threshold for removal (default: 0.95)

## Value

A list with components:

- `data`:

  The processed data frame.

- `original_data`:

  The original unprocessed data frame.

- `preprocessing_steps`:

  A list of metadata for each preprocessing step applied (imputation
  values, encoding maps, scaling parameters, etc.).

- `formula`:

  The formula passed in (or `NULL`).

## Details

Comprehensive preprocessing pipeline including imputation, scaling,
encoding, and feature engineering

## Examples

``` r
# \donttest{
processed <- tl_prepare_data(iris, Species ~ ., scale_method = "standardize")
#> Scaling numeric features using method: standardize
model <- tl_model(processed$data, Species ~ ., method = "tree")
# }
```
