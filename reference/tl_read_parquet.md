# Read a Parquet file

Reads a Parquet file into a `tidylearn_data` object. Requires the
nanoparquet package.

## Usage

``` r
tl_read_parquet(path, ...)
```

## Arguments

- path:

  Path to a Parquet file.

- ...:

  Additional arguments passed to
  [`nanoparquet::read_parquet()`](https://nanoparquet.r-lib.org/reference/read_parquet.html).

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
# \donttest{
# data <- tl_read_parquet("path/to/data.parquet")
# }
```
