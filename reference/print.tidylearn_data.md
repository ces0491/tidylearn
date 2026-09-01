# Print a tidylearn_data object

Print a tidylearn_data object

## Usage

``` r
# S3 method for class 'tidylearn_data'
print(x, ...)
```

## Arguments

- x:

  A `tidylearn_data` object.

- ...:

  Additional arguments passed to the tibble print method.

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
f <- tempfile(fileext = ".csv")
write.csv(iris, f, row.names = FALSE)
d <- tl_read(f)
#> Reading csv data from: /tmp/RtmpUcJe2E/file1e0f7040dbf8.csv
#> Returned: 150 rows x 5 columns
print(d)
#> -- tidylearn data ---------
#> Source: /tmp/RtmpUcJe2E/file1e0f7040dbf8.csv 
#> Format: csv 
#> Read at: 2026-09-01 18:03:55 
#> 
#> # A tibble: 150 × 5
#>    Sepal.Length Sepal.Width Petal.Length Petal.Width Species
#>  *        <dbl>       <dbl>        <dbl>       <dbl> <chr>  
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
# }
```
