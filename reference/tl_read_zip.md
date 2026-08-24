# Read data from a zip archive

Extracts a zip archive to a temporary directory and reads the contents.
If the archive contains a single data file, it is read directly. If
multiple data files are found, they are row-bound with a `source_file`
column. Use the `file` argument to select a specific file from the
archive.

## Usage

``` r
tl_read_zip(path, file = NULL, format = NULL, .quiet = FALSE, ...)
```

## Arguments

- path:

  Path to a zip file.

- file:

  Optional name of a specific file within the archive to read. Supports
  partial matching.

- format:

  Optional format override for the file(s) inside the archive. When the
  archive holds more than one kind of data file, this selects the
  members of that format rather than forcing it onto all of them.

- .quiet:

  Suppress messages. Default is `FALSE`.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`. The
archive is extracted to a temporary directory that is cleaned up
automatically. If multiple data files are found, a `source_file` column
identifies the origin of each row.

## Examples

``` r
# \donttest{
# Read from a zip archive
# data <- tl_read_zip("data.zip")

# Read a specific file from the archive
# data <- tl_read_zip("data.zip", file = "train.csv")
# }
```
