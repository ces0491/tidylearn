# Read an RDS file

Reads an RDS file into a `tidylearn_data` object. Uses base R
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) — no additional
packages required.

## Usage

``` r
tl_read_rds(path)
```

## Arguments

- path:

  Path to an RDS file.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
# \donttest{
# data <- tl_read_rds("path/to/data.rds")
# }
```
