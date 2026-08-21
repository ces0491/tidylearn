# Read an RData file

Reads an RData (`.rdata` or `.rda`) file into a `tidylearn_data` object.
Since RData files can contain multiple objects, use the `name` argument
to specify which object to extract. If `name` is `NULL` and the file
contains exactly one data frame, it is returned automatically.

## Usage

``` r
tl_read_rdata(path, name = NULL, ...)
```

## Arguments

- path:

  Path to an RData file.

- name:

  Optional name of the object to extract from the RData file. If `NULL`
  (default), the function returns the first data frame found, or errors
  if there are multiple data frames.

- ...:

  Currently unused.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
# \donttest{
# data <- tl_read_rdata("path/to/data.rdata")
# data <- tl_read_rdata("path/to/data.rdata", name = "my_data")
# }
```
