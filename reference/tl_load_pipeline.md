# Load a pipeline from disk

Load a pipeline from disk

## Usage

``` r
tl_load_pipeline(file)
```

## Arguments

- file:

  Path to the pipeline file

## Value

A `tidylearn_pipeline` object previously saved with
[`tl_save_pipeline`](https://tidylearn.sheetsolved.com/reference/tl_save_pipeline.md).

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .)
f <- tempfile(fileext = ".rds")
tl_save_pipeline(pipe, f)
pipe2 <- tl_load_pipeline(f)
# }
```
