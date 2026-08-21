# Detect local GPU availability for tidylearn methods

Reports whether the local machine has a CUDA-capable GPU and which
tidylearn backends (`xgboost`, `keras`, `tensorflow`, `torch`) are
positioned to use it. Detection is intentionally cheap: it parses
`nvidia-smi` output and checks which R packages are installed, but does
not load Python or fit a model. A backend reported as
`gpu_likely_works = TRUE` may still fall back to CPU if it was not
compiled or configured with CUDA support — confirm with a small real fit
before relying on it for production workloads.

## Usage

``` r
tl_check_gpu(verbose = FALSE)
```

## Arguments

- verbose:

  Logical. If `TRUE`, also prints the result. Default `FALSE`.

## Value

An object of class `tidylearn_gpu_check`: a list with components
`any_gpu` (logical), `cuda` (driver info list with `driver_present`,
`device_count`, `device_names`, `driver_version`), `backends`
(per-backend status list each containing `installed`,
`gpu_likely_works`, `notes`), and `messages` (character vector). A
[`print()`](https://rdrr.io/r/base/print.html) method is provided.

## Details

Apple MPS (Metal Performance Shaders) is intentionally not detected in
this iteration; see the issue tracker for the MPS feature request.

## Examples

``` r
# Safe to call anywhere: probes for nvidia-smi on the PATH and checks
# which backend packages are installed. Reports no GPU rather than
# failing when there isn't one.
gpu <- tl_check_gpu()
gpu$any_gpu
#> [1] FALSE

if (gpu$any_gpu && gpu$backends$xgboost$gpu_likely_works) {
  # xgboost with GPU is worth trying for this workload
}
```
