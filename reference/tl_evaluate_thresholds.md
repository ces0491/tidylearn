# Evaluate metrics at different thresholds

Evaluate metrics at different thresholds

## Usage

``` r
tl_evaluate_thresholds(actuals, probs, thresholds, pos_class)
```

## Arguments

- actuals:

  Actual values (ground truth)

- probs:

  Predicted probabilities

- thresholds:

  Vector of thresholds to evaluate

- pos_class:

  The positive class

## Value

A tibble of metrics at different thresholds
