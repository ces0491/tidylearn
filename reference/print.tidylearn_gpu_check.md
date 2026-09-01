# Print method for `tidylearn_gpu_check` objects

Print method for `tidylearn_gpu_check` objects

## Usage

``` r
# S3 method for class 'tidylearn_gpu_check'
print(x, ...)
```

## Arguments

- x:

  A `tidylearn_gpu_check` object.

- ...:

  Unused.

## Value

The input `x`, invisibly.

## Examples

``` r
# Reports no GPU rather than failing on a machine without one
print(tl_check_gpu())
#> <tidylearn GPU detection>
#> Any GPU:        no
#> CUDA driver:    absent
#> 
#> Backends:
#>   xgboost     CPU only
#>   tensorflow  CPU only
#>   keras       CPU only
#>   torch       not installed
#> 
#> Notes:
#>   - No NVIDIA CUDA driver detected. All GPU paths fall back to CPU.
```
