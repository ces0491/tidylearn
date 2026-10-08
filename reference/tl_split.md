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

  Column name for stratified splitting. Each stratum is split at `prop`.
  A numeric column with more than five distinct values is stratified by
  its quartiles, and rows missing the value form a stratum of their own.
  A stratum of a single row goes to the training set, unless such strata
  together hold more than a tenth of the rows, as in an ID column; then
  they are pooled into one stratum and split together.

- seed:

  Random seed for reproducibility

## Value

A list with two elements:

- `$train`:

  A data frame containing the training subset.

- `$test`:

  A data frame containing the test subset.

A split that leaves the test set empty, as a single row does, warns.

## Examples

``` r
# \donttest{
split_data <- tl_split(iris, prop = 0.7, stratify = "Species")
train <- split_data$train
test <- split_data$test
# }
```
