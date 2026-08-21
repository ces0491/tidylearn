# Read all matching files from a directory

Scans a directory for files matching a pattern or format, reads each
one, and row-binds them into a single `tidylearn_data` object with a
`source_file` column identifying the origin of each row.

## Usage

``` r
tl_read_dir(
  path,
  pattern = NULL,
  format = NULL,
  recursive = FALSE,
  .quiet = FALSE,
  ...
)
```

## Arguments

- path:

  Path to a directory.

- pattern:

  Optional regex pattern to filter file names (e.g.,
  `"sales_.*\\.csv$"`). If `NULL`, files are filtered by `format`
  instead.

- format:

  File format to read. If `NULL` and `pattern` is `NULL`, all recognized
  data files are read. If specified, only files with matching extensions
  are read.

- recursive:

  Logical. Should subdirectories be scanned? Default is `FALSE`.

- .quiet:

  Suppress messages. Default is `FALSE`.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object with an additional `source_file` column
identifying the origin of each row.

## Examples

``` r
# \donttest{
# Read all CSVs from a directory
# data <- tl_read_dir("data/", format = "csv")

# Read with pattern matching
# data <- tl_read_dir("data/", pattern = "^sales_.*\.csv$")

# Read all recognized data files recursively
# data <- tl_read_dir("data/", recursive = TRUE)
# }
```
