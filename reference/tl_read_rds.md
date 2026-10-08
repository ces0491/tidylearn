# Read an RDS file

Reads an RDS file into a `tidylearn_data` object. Uses base R
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) — no additional
packages required.

## Usage

``` r
tl_read_rds(path)
```

## Arguments

- path:

  Path to an RDS file.

## Value

A `tidylearn_data` object (a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) subclass)
with attributes `tl_source`, `tl_format`, and `tl_timestamp`.

## Reading files you did not create

[`readRDS()`](https://rdrr.io/r/base/readRDS.html) rebuilds whatever R
objects the file describes, so read only files from a source you trust.
On R before 4.4.0 a crafted file can run code as it is read
(CVE-2024-27322). The remote readers
[`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
and
[`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
refuse `.rds` and `.rdata` files on those versions unless
`trust_rds = TRUE`.

## Examples

``` r
path <- tempfile(fileext = ".rds")
saveRDS(mtcars, path)
tl_read_rds(path)
#> -- tidylearn data ---------
#> Source: /tmp/RtmpK94WAU/file1cba74907f7b.rds 
#> Format: rds 
#> Read at: 2026-10-08 17:59:46 
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
unlink(path)
```
