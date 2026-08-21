# Read a JSON file

Reads a JSON file into a `tidylearn_data` object. Expects the JSON to
represent tabular data (array of objects or similar). Requires the
jsonlite package.

## Usage

``` r
tl_read_json(path, flatten = TRUE, ...)
```

## Arguments

- path:

  Path to a JSON file.

- flatten:

  Logical. Automatically flatten nested data frames? Default is `TRUE`.

- ...:

  Additional arguments passed to
  [`jsonlite::fromJSON()`](https://jeroen.r-universe.dev/jsonlite/reference/fromJSON.html).

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
# \donttest{
# data <- tl_read_json("path/to/data.json")
# }
```
