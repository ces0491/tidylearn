# Read a JSON file

Reads a JSON file into a `tidylearn_data` object. Expects the JSON to
represent tabular data (array of objects or similar). A file with the
`.ndjson` extension is read as newline-delimited JSON, one record per
line, and so is a `.json` file that does not parse as a single document
but does as one record per line. Requires the jsonlite package.

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
  [`jsonlite::fromJSON()`](https://jeroen.r-universe.dev/jsonlite/reference/fromJSON.html),
  or to
  [`jsonlite::stream_in()`](https://jeroen.r-universe.dev/jsonlite/reference/stream_in.html)
  for an `.ndjson` file.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Examples

``` r
path <- tempfile(fileext = ".json")
jsonlite::write_json(mtcars, path)
tl_read_json(path)
#> -- tidylearn data ---------
#> Source: /tmp/RtmpK94WAU/file1cba159c712a.json 
#> Format: json 
#> Read at: 2026-10-08 17:59:44 
#> 
#> # A tibble: 32 × 11
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#>  * <dbl> <int> <dbl> <int> <dbl> <dbl> <dbl> <int> <int> <int> <int>
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

# Newline-delimited JSON: one record per line
lines <- tempfile(fileext = ".ndjson")
jsonlite::stream_out(mtcars, file(lines), verbose = FALSE)
tl_read_json(lines)
#> -- tidylearn data ---------
#> Source: /tmp/RtmpK94WAU/file1cba39f263a2.ndjson 
#> Format: json 
#> Read at: 2026-10-08 17:59:44 
#> 
#> # A tibble: 32 × 11
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#>  * <dbl> <int> <dbl> <int> <dbl> <dbl> <dbl> <int> <int> <int> <int>
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

unlink(c(path, lines))
```
