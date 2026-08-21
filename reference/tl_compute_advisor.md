# Advise on the best compute tier for a tidylearn fit

Estimates runtime and feasibility on local CPU, local GPU (when
available and applicable), and cloud GPU (stubbed until cloud
integration lands), then returns a structured recommendation. Useful
before kicking off a long fit — call this first to see whether the
problem is laptop-sized, GPU-sized, or cloud-sized.

## Usage

``` r
tl_compute_advisor(x, ...)

# S3 method for class 'character'
tl_compute_advisor(
  x,
  data,
  formula = NULL,
  hyperparams = list(),
  gpu_check = NULL,
  ...
)

# S3 method for class 'tidylearn_supervised'
tl_compute_advisor(
  x,
  data = NULL,
  formula = NULL,
  hyperparams = list(),
  gpu_check = NULL,
  ...
)

# Default S3 method
tl_compute_advisor(x, ...)
```

## Arguments

- x:

  Either a method name (character scalar — same names as accepted by
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md),
  supervised methods only) or a fitted `tidylearn_supervised` model.
  When given a fitted model, the advisor introspects its method and
  formula and advises on what a refit (or a fit on similar new data)
  would cost.

- ...:

  Unused, reserved for method-specific extensions.

- data:

  A data frame. Required when `x` is a method name; optional when `x` is
  a fitted model (defaults to the model's training data).

- formula:

  Optional formula. Used to determine the number of effective
  predictors. Ignored when `x` is a fitted model.

- hyperparams:

  Named list of hyperparameters that affect runtime (e.g.
  `list(nrounds = 1000)` for xgboost, `list(epochs = 50, units = 256)`
  for deep learning). Missing entries fall back to per-method defaults.

- gpu_check:

  Optional `tidylearn_gpu_check` object. If omitted,
  [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  is called once internally.

## Value

An object of class `tidylearn_compute_advice` containing `problem`,
`local_cpu`, `local_gpu`, `cloud`, `recommendation`, and `reasoning`. A
[`print()`](https://rdrr.io/r/base/print.html) method is provided.

## Details

Estimates are deliberately rough — order-of-magnitude, not bills.
Per-method scaling constants are calibrated against typical hardware and
will be off by 2-3x in either direction for any individual job. Treat
the recommendation as a starting point, not gospel.

## Examples

``` r
# Estimating from a method name needs neither a GPU nor the backend
# package -- it is arithmetic over the problem dimensions
advice <- tl_compute_advisor("xgboost", iris, Species ~ .,
                             hyperparams = list(nrounds = 1000))
advice$recommendation
#> [1] "cpu"
print(advice)
#> <tidylearn compute advice>
#> Problem:        xgboost on 150 rows x 4 cols (~0.0 MB)
#> 
#> Tier estimates (order-of-magnitude):
#>   Local CPU:    0.0s   (peak RAM ~0 MB, 4 cores)
#>   Local GPU:    --   [not applicable]
#>   Cloud:        45.0s   (~$0.01) [T4 (16 GB VRAM / 16 GB RAM)]   [not configured]
#> 
#> Recommendation: cpu
#> 
#> Reasoning:
#>   - Estimated local CPU runtime ~0.0s. Cloud cold-start (~45s) would dominate; just run it locally.
#> 
#> Notes:
#>   - Method 'xgboost' could use GPU, but no GPU-capable backend was detected. See ?tl_check_gpu.
#>   - Cloud integration is not yet configured in tidylearn. Estimates shown so users can see the tier's shape; actual submission is not yet supported.

# \donttest{
# Dispatching on a fitted model requires the backend to be installed
if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "xgboost")
  tl_compute_advisor(model)
}
#> <tidylearn compute advice>
#> Problem:        xgboost on 150 rows x 4 cols (~0.0 MB)
#> 
#> Tier estimates (order-of-magnitude):
#>   Local CPU:    0.0s   (peak RAM ~0 MB, 4 cores)
#>   Local GPU:    --   [not applicable]
#>   Cloud:        45.0s   (~$0.01) [T4 (16 GB VRAM / 16 GB RAM)]   [not configured]
#> 
#> Recommendation: cpu
#> 
#> Reasoning:
#>   - Estimated local CPU runtime ~0.0s. Cloud cold-start (~45s) would dominate; just run it locally.
#> 
#> Notes:
#>   - Method 'xgboost' could use GPU, but no GPU-capable backend was detected. See ?tl_check_gpu.
#>   - Cloud integration is not yet configured in tidylearn. Estimates shown so users can see the tier's shape; actual submission is not yet supported.
# }
```
