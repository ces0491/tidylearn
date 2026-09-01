# Print method for `tidylearn_compute_advice` objects

Print method for `tidylearn_compute_advice` objects

## Usage

``` r
# S3 method for class 'tidylearn_compute_advice'
print(x, ...)
```

## Arguments

- x:

  A `tidylearn_compute_advice` object.

- ...:

  Unused.

## Value

The input `x`, invisibly.

## Examples

``` r
# Runtime, peak memory and cost per tier, before committing to a fit
advice <- tl_compute_advisor("forest", data = iris, formula = Species ~ .)
print(advice)
#> <tidylearn compute advice>
#> Problem:        forest on 150 rows x 4 cols (~0.0 MB)
#> 
#> Tier estimates (order-of-magnitude):
#>   Local CPU:    0.0s   (peak RAM ~0 MB, 4 cores)
#>   Local GPU:    --   [not applicable]
#>   Cloud:        45.1s   (~$0.01) [cpu-small (2 CPU / 8 GB)]   [not configured]
#> 
#> Recommendation: cpu
#> 
#> Reasoning:
#>   - Estimated local CPU runtime ~0.0s. Cloud cold-start (~45s) would dominate; just run it locally.
#> 
#> Notes:
#>   - Method 'forest' has no upstream GPU path; local GPU does not apply.
#>   - Cloud integration is not yet configured in tidylearn. Estimates shown so users can see the tier's shape; actual submission is not yet supported.
```
