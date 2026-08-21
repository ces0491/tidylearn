# Calculate classification metrics

Calculate classification metrics

## Usage

``` r
tl_calc_classification_metrics(
  actuals,
  predicted,
  predicted_probs = NULL,
  metrics = c("accuracy", "precision", "recall", "f1", "auc"),
  thresholds = NULL,
  ...
)
```

## Arguments

- actuals:

  Actual values (ground truth)

- predicted:

  Predicted class values

- predicted_probs:

  Predicted probabilities (for metrics like AUC)

- metrics:

  Character vector of metrics to compute

- thresholds:

  Optional vector of thresholds to evaluate for threshold-dependent
  metrics

- ...:

  Additional arguments

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with
columns `metric` (character) and `value` (numeric) containing the
requested classification metrics. When `thresholds` are supplied,
additional rows are appended with threshold-specific metric names.

## Examples

``` r
# \donttest{
model <- tl_model(iris, Species ~ ., method = "forest")
preds <- predict(model)
tl_calc_classification_metrics(iris$Species, preds$.pred)
#> # A tibble: 4 × 2
#>   metric    value
#>   <chr>     <dbl>
#> 1 accuracy      1
#> 2 precision     1
#> 3 recall        1
#> 4 f1            1
# }
```
