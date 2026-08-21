# Read an Excel file

Reads an Excel file (`.xls`, `.xlsx`, or `.xlsm`) into a
`tidylearn_data` object. Requires the readxl package.

## Usage

``` r
tl_read_excel(path, sheet = 1, ...)
```

## Arguments

- path:

  Path to an Excel file.

- sheet:

  Sheet to read. Either a string (the name of a sheet) or an integer
  (the position of the sheet). Defaults to the first sheet.

- ...:

  Additional arguments passed to
  [`readxl::read_excel()`](https://readxl.tidyverse.org/reference/read_excel.html).

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
# \donttest{
# data <- tl_read_excel("path/to/data.xlsx")
# data <- tl_read_excel("path/to/data.xlsx", sheet = "Sheet2")
# }
```
