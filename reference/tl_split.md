# Split data into train and test sets

Split data into train and test sets

## Usage

``` r
tl_split(data, prop = 0.8, stratify = NULL, seed = NULL)
```

## Arguments

- data:

  A data frame

- prop:

  Proportion for training set (default: 0.8)

- stratify:

  Column name for stratified splitting

- seed:

  Random seed for reproducibility

## Value

A list with two elements:

- `$train`:

  A data frame containing the training subset.

- `$test`:

  A data frame containing the test subset.

## Examples

``` r
# \donttest{
split_data <- tl_split(iris, prop = 0.7, stratify = "Species")
train <- split_data$train
test <- split_data$test
# }
```
