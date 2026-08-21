# Anomaly-Aware Supervised Learning

Detect outliers using DBSCAN or other methods, then optionally remove
them or down-weight them before supervised learning.

## Usage

``` r
tl_anomaly_aware(
  data,
  formula,
  response,
  anomaly_method = "dbscan",
  action = "flag",
  supervised_method = "logistic",
  ...
)
```

## Arguments

- data:

  A data frame

- formula:

  Model formula

- response:

  Response variable name

- anomaly_method:

  Method for anomaly detection: "dbscan", "isolation_forest"

- action:

  Action to take: "remove", "flag", "downweight"

- supervised_method:

  Supervised learning method

- ...:

  Additional arguments

## Value

A tidylearn model object with additional class
`"tidylearn_anomaly_aware"`. The model includes an `anomaly_info`
element with `anomaly_model`, `is_anomaly` (logical vector),
`n_anomalies`, and `action`.

## Examples

``` r
# \donttest{
model <- tl_anomaly_aware(iris, Species ~ ., response = "Species",
                           anomaly_method = "dbscan", action = "flag")
#> Warning: glm.fit: algorithm did not converge
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
# }
```
