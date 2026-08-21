# Read data from diverse sources

Auto-detects the data format from the file extension or source pattern
and dispatches to the appropriate reader. All readers return a
`tidylearn_data` object, which is a tibble subclass carrying metadata
about the data source.

## Usage

``` r
tl_read(source, ..., format = NULL, .quiet = FALSE)
```

## Arguments

- source:

  A file path, URL, connection string, directory path, or a character
  vector of multiple file paths.

- ...:

  Additional arguments passed to the format-specific reader.

- format:

  Optional explicit format override. One of `"csv"`, `"tsv"`, `"excel"`,
  `"parquet"`, `"json"`, `"rds"`, `"rdata"`, `"sqlite"`, `"postgres"`,
  `"mysql"`, `"bigquery"`, `"s3"`, `"github"`, `"kaggle"`. When `NULL`
  (default), the format is auto-detected from the file extension or
  source pattern. Note: `.txt` files default to CSV; use
  `format = "tsv"` to override.

- .quiet:

  Logical. If `TRUE`, suppresses informational messages. Default is
  `FALSE`.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Details

When `source` is a character vector of multiple paths, each file is read
and row-bound into a single result with a `source_file` column. When
`source` is a directory path, it is equivalent to calling
[`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md).
When `source` is a `.zip` file, it is equivalent to calling
[`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md).

## Examples

``` r
# \donttest{
# Read a single CSV file
# data <- tl_read("path/to/data.csv")

# Read multiple files and row-bind
# data <- tl_read(c("jan.csv", "feb.csv", "mar.csv"))

# Read all CSVs from a directory
# data <- tl_read("data/")

# Read from a zip archive
# data <- tl_read("data.zip")

# Explicit format override
# data <- tl_read("path/to/data.txt", format = "tsv")
# }
```
