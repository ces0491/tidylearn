# Read a CSV file

Reads a CSV file into a `tidylearn_data` object. Uses readr when
available for faster parsing, with a base R fallback.

## Usage

``` r
tl_read_csv(path, ...)
```

## Arguments

- path:

  Path to a CSV file.

- ...:

  Additional arguments passed to
  [`readr::read_csv()`](https://readr.tidyverse.org/reference/read_delim.html)
  or [`utils::read.csv()`](https://rdrr.io/r/utils/read.table.html).

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
# \donttest{
# data <- tl_read_csv("path/to/data.csv")
# }
```
