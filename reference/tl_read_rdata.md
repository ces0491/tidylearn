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

## Reading files you did not create

[`load()`](https://rdrr.io/r/base/load.html) rebuilds whatever R objects
the file describes, so read only files from a source you trust. On R
before 4.4.0 a crafted file can run code as it is read (CVE-2024-27322).
The remote readers
[`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
and
[`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
refuse `.rds` and `.rdata` files on those versions unless
`trust_rds = TRUE`.

## Examples

``` r
path <- tempfile(fileext = ".rdata")
cars <- mtcars
flowers <- iris
save(cars, flowers, file = path)

# With more than one data frame in the file, name the one to read
tl_read_rdata(path, name = "flowers")
#> -- tidylearn data ---------
#> Source: /tmp/Rtmp5pHJ7J/file1ab335409584.rdata 
#> Format: rdata 
#> Read at: 2026-10-09 07:12:11 
#> 
#> # A tibble: 150 × 5
#>    Sepal.Length Sepal.Width Petal.Length Petal.Width Species
#>  *        <dbl>       <dbl>        <dbl>       <dbl> <fct>  
#>  1          5.1         3.5          1.4         0.2 setosa 
#>  2          4.9         3            1.4         0.2 setosa 
#>  3          4.7         3.2          1.3         0.2 setosa 
#>  4          4.6         3.1          1.5         0.2 setosa 
#>  5          5           3.6          1.4         0.2 setosa 
#>  6          5.4         3.9          1.7         0.4 setosa 
#>  7          4.6         3.4          1.4         0.3 setosa 
#>  8          5           3.4          1.5         0.2 setosa 
#>  9          4.4         2.9          1.4         0.2 setosa 
#> 10          4.9         3.1          1.5         0.1 setosa 
#> # ℹ 140 more rows
unlink(path)
```
