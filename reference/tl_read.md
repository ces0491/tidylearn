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

  A file path or `file://` URL, a GitHub or Kaggle URL, an `s3://` or
  `bigquery://` URI, a database connection string, a directory path, or
  a character vector of multiple file paths. Other web URLs, and URLs
  with any other scheme such as `ftp://`, are refused: download the file
  and read the local copy.

- ...:

  Additional arguments passed to the format-specific reader.

- format:

  Optional explicit format override. One of `"csv"`, `"tsv"`, `"excel"`,
  `"parquet"`, `"json"`, `"rds"`, `"rdata"`, `"sqlite"`, `"postgres"`,
  `"mysql"`, `"bigquery"`, `"s3"`, `"github"`, `"kaggle"`. When `NULL`
  (default), the format is auto-detected from the file extension or
  source pattern. Note: `.txt` files default to CSV; use
  `format = "tsv"` to override. A source with a URL scheme other than
  `file://` is read only with the format its scheme or host implies; to
  read an S3 object as a given file format, call
  [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
  with that format.

- .quiet:

  Logical. If `TRUE`, suppresses informational messages. Default is
  `FALSE`.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Details

When `source` is a character vector of multiple paths, each file is read
and row-bound into a single result with a `source_file` column giving
each file's path below the deepest folder they share. When `source` is a
directory path, it is equivalent to calling
[`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md).
When `source` is a local `.zip` file, it is equivalent to calling
[`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md).

## Examples

``` r
# The format is detected from the extension
csv <- tempfile(fileext = ".csv")
write.csv(mtcars, csv, row.names = FALSE)
tl_read(csv)
#> Reading csv data from: /tmp/Rtmp5pHJ7J/file1ab3363aa564.csv
#> Returned: 32 rows x 11 columns
#> -- tidylearn data ---------
#> Source: /tmp/Rtmp5pHJ7J/file1ab3363aa564.csv 
#> Format: csv 
#> Read at: 2026-10-09 07:12:06 
#> 
#> # A tibble: 32 × 11
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#>  * <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#>  1  21       6  160    110  3.9   2.62  16.5     0     1     4     4
#>  2  21       6  160    110  3.9   2.88  17.0     0     1     4     4
#>  3  22.8     4  108     93  3.85  2.32  18.6     1     1     4     1
#>  4  21.4     6  258    110  3.08  3.22  19.4     1     0     3     1
#>  5  18.7     8  360    175  3.15  3.44  17.0     0     0     3     2
#>  6  18.1     6  225    105  2.76  3.46  20.2     1     0     3     1
#>  7  14.3     8  360    245  3.21  3.57  15.8     0     0     3     4
#>  8  24.4     4  147.    62  3.69  3.19  20       1     0     4     2
#>  9  22.8     4  141.    95  3.92  3.15  22.9     1     0     4     2
#> 10  19.2     6  168.   123  3.92  3.44  18.3     1     0     4     4
#> # ℹ 22 more rows

# Several files are row-bound, with a source_file column naming each
jan <- tempfile(fileext = ".csv")
feb <- tempfile(fileext = ".csv")
write.csv(mtcars[1:16, ], jan, row.names = FALSE)
write.csv(mtcars[17:32, ], feb, row.names = FALSE)
both <- tl_read(c(jan, feb), .quiet = TRUE)
table(both$source_file)
#> 
#> file1ab32d687d1.csv file1ab34d2e0af.csv 
#>                  16                  16 

# A .txt file is read as CSV unless told otherwise
txt <- tempfile(fileext = ".txt")
write.table(mtcars, txt, sep = "\t", row.names = FALSE)
tl_read(txt, format = "tsv", .quiet = TRUE)
#> -- tidylearn data ---------
#> Source: /tmp/Rtmp5pHJ7J/file1ab31d0940fd.txt 
#> Format: tsv 
#> Read at: 2026-10-09 07:12:06 
#> 
#> # A tibble: 32 × 11
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#>  * <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#>  1  21       6  160    110  3.9   2.62  16.5     0     1     4     4
#>  2  21       6  160    110  3.9   2.88  17.0     0     1     4     4
#>  3  22.8     4  108     93  3.85  2.32  18.6     1     1     4     1
#>  4  21.4     6  258    110  3.08  3.22  19.4     1     0     3     1
#>  5  18.7     8  360    175  3.15  3.44  17.0     0     0     3     2
#>  6  18.1     6  225    105  2.76  3.46  20.2     1     0     3     1
#>  7  14.3     8  360    245  3.21  3.57  15.8     0     0     3     4
#>  8  24.4     4  147.    62  3.69  3.19  20       1     0     4     2
#>  9  22.8     4  141.    95  3.92  3.15  22.9     1     0     4     2
#> 10  19.2     6  168.   123  3.92  3.44  18.3     1     0     4     4
#> # ℹ 22 more rows

unlink(c(csv, jan, feb, txt))
```
