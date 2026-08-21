# Save a pipeline to disk

Save a pipeline to disk

## Usage

``` r
tl_save_pipeline(pipeline, file)
```

## Arguments

- pipeline:

  A tidylearn pipeline object

- file:

  Path to save the pipeline

## Value

Called for its side effect of saving to disk; returns `NULL` invisibly.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .)
tl_save_pipeline(pipe, tempfile(fileext = ".rds"))
# }
```
