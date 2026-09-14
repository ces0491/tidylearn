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

  Optional formula (for supervised learning). Only its predictors are
  processed; a column it excludes, such as `- id`, is returned
  unchanged.

- impute_method:

  Method for imputing a missing numeric value: "mean", "median" or
  "mode". A missing categorical value is always filled with the column's
  most frequent value.

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

  A record of each step applied (imputation values, encoding maps,
  scaling parameters, etc.). It is for inspection: no function applies
  it to new data.

- `formula`:

  The formula passed in (or `NULL`).

## Details

Comprehensive preprocessing pipeline including imputation, scaling,
encoding, and feature engineering

The statistics are learned from, and applied to, the data passed in.
Preparing a whole dataset and then splitting it lets the test rows shape
the imputation values and scaling their own scores are measured against.
To evaluate a model, split first, or use
[`tl_pipeline`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md),
which learns its preprocessing inside each resampling fold.

## Examples

``` r
# \donttest{
processed <- tl_prepare_data(iris, Species ~ ., scale_method = "standardize")
#> Scaling numeric features using method: standardize
model <- tl_model(processed$data, Species ~ ., method = "tree")
# }
```
