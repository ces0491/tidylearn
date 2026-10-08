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

  Optional name of a specific file within the archive to read: its path
  within the archive (`"2024/sales.csv"`), its file name, or part of its
  path. An exact path wins, then an exact file name, then a partial
  match; a name that matches more than one member at the first of those
  steps that matches anything is an error listing them.

- format:

  Optional format override for the file(s) inside the archive. Without
  `file`, this selects the members of that format rather than forcing it
  onto them, whether the archive holds one data file or several. With
  `file`, the named member is read as `format` whatever its extension.

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
gives each row's member as its path within the archive.

## Details

An archive with a member whose name could reach outside the extraction
directory – an absolute path, a drive letter on Windows, or any `..`
component – is refused before anything is extracted.

## Examples

``` r
# readr ships a zip archive holding one CSV
archive <- readr::readr_example("mtcars.csv.zip")
tl_read_zip(archive)
#> Reading csv data from: /tmp/RtmpK94WAU/tl_zip_1cbacd9f1f4/mtcars.csv
#> Returned: 32 rows x 11 columns
#> -- tidylearn data ---------
#> Source: /home/runner/work/_temp/Library/readr/extdata/mtcars.csv.zip//mtcars.csv 
#> Format: zip+csv 
#> Read at: 2026-10-08 17:59:47 
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

# Name a member to read just that one
tl_read_zip(archive, file = "mtcars.csv", .quiet = TRUE)
#> -- tidylearn data ---------
#> Source: /home/runner/work/_temp/Library/readr/extdata/mtcars.csv.zip//mtcars.csv 
#> Format: zip+csv 
#> Read at: 2026-10-08 17:59:47 
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
```
