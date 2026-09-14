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
  supervised_method = "tree",
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

  Method for anomaly detection. Only "dbscan" is implemented; its noise
  points are the anomalies.

- action:

  Action to take: "remove", "flag", "downweight". `"downweight"` gives
  anomalies a case weight of 0.1, and needs a `supervised_method` that
  takes case weights: `"linear"`, `"polynomial"`, `"logistic"`,
  `"tree"`, `"ridge"`, `"lasso"`, `"elastic_net"` or `"forest"`. A
  forest reads them as sampling weights.

- supervised_method:

  Supervised learning method (default: `"tree"`, which handles both
  regression and classification with any number of classes).
  `"logistic"` is binary-only and errors on a response with more than
  two levels.

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
# }
```
