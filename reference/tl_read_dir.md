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
  are read. `.txt` files are read only when selected with `pattern`.

- recursive:

  Logical. Should subdirectories be scanned? Default is `FALSE`.

- .quiet:

  Suppress messages. Default is `FALSE`.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object with an additional `source_file` column giving
each row's file as a path below `path`, such as `"2024/sales.csv"`.

## Examples

``` r
dir <- tempfile("sales_")
dir.create(file.path(dir, "2024"), recursive = TRUE)
write.csv(mtcars[1:16, ], file.path(dir, "jan.csv"), row.names = FALSE)
write.csv(mtcars[17:32, ], file.path(dir, "2024", "feb.csv"),
          row.names = FALSE)

# Only the top level unless asked to recurse
tl_read_dir(dir, format = "csv")
#> Found 1 file(s) in /tmp/RtmpK94WAU/sales_1cba5b620cba
#> Reading 1 files...
#> Combined: 16 rows x 12 columns from 1 files
#> -- tidylearn data ---------
#> Source: 1 files 
#> Format: csv 
#> Read at: 2026-10-08 17:59:43 
#> 
#> # A tibble: 16 × 12
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb source_file
#>  * <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <chr>      
#>  1  21       6  160    110  3.9   2.62  16.5     0     1     4     4 jan.csv    
#>  2  21       6  160    110  3.9   2.88  17.0     0     1     4     4 jan.csv    
#>  3  22.8     4  108     93  3.85  2.32  18.6     1     1     4     1 jan.csv    
#>  4  21.4     6  258    110  3.08  3.22  19.4     1     0     3     1 jan.csv    
#>  5  18.7     8  360    175  3.15  3.44  17.0     0     0     3     2 jan.csv    
#>  6  18.1     6  225    105  2.76  3.46  20.2     1     0     3     1 jan.csv    
#>  7  14.3     8  360    245  3.21  3.57  15.8     0     0     3     4 jan.csv    
#>  8  24.4     4  147.    62  3.69  3.19  20       1     0     4     2 jan.csv    
#>  9  22.8     4  141.    95  3.92  3.15  22.9     1     0     4     2 jan.csv    
#> 10  19.2     6  168.   123  3.92  3.44  18.3     1     0     4     4 jan.csv    
#> 11  17.8     6  168.   123  3.92  3.44  18.9     1     0     4     4 jan.csv    
#> 12  16.4     8  276.   180  3.07  4.07  17.4     0     0     3     3 jan.csv    
#> 13  17.3     8  276.   180  3.07  3.73  17.6     0     0     3     3 jan.csv    
#> 14  15.2     8  276.   180  3.07  3.78  18       0     0     3     3 jan.csv    
#> 15  10.4     8  472    205  2.93  5.25  18.0     0     0     3     4 jan.csv    
#> 16  10.4     8  460    215  3     5.42  17.8     0     0     3     4 jan.csv    

# Files in subfolders are labelled by their path below dir
all_months <- tl_read_dir(dir, recursive = TRUE, .quiet = TRUE)
table(all_months$source_file)
#> 
#> 2024/feb.csv      jan.csv 
#>           16           16 

# Or select files by a regular expression
tl_read_dir(dir, pattern = "^jan", .quiet = TRUE)
#> -- tidylearn data ---------
#> Source: 1 files 
#> Format: multi 
#> Read at: 2026-10-08 17:59:43 
#> 
#> # A tibble: 16 × 12
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb source_file
#>  * <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <chr>      
#>  1  21       6  160    110  3.9   2.62  16.5     0     1     4     4 jan.csv    
#>  2  21       6  160    110  3.9   2.88  17.0     0     1     4     4 jan.csv    
#>  3  22.8     4  108     93  3.85  2.32  18.6     1     1     4     1 jan.csv    
#>  4  21.4     6  258    110  3.08  3.22  19.4     1     0     3     1 jan.csv    
#>  5  18.7     8  360    175  3.15  3.44  17.0     0     0     3     2 jan.csv    
#>  6  18.1     6  225    105  2.76  3.46  20.2     1     0     3     1 jan.csv    
#>  7  14.3     8  360    245  3.21  3.57  15.8     0     0     3     4 jan.csv    
#>  8  24.4     4  147.    62  3.69  3.19  20       1     0     4     2 jan.csv    
#>  9  22.8     4  141.    95  3.92  3.15  22.9     1     0     4     2 jan.csv    
#> 10  19.2     6  168.   123  3.92  3.44  18.3     1     0     4     4 jan.csv    
#> 11  17.8     6  168.   123  3.92  3.44  18.9     1     0     4     4 jan.csv    
#> 12  16.4     8  276.   180  3.07  4.07  17.4     0     0     3     3 jan.csv    
#> 13  17.3     8  276.   180  3.07  3.73  17.6     0     0     3     3 jan.csv    
#> 14  15.2     8  276.   180  3.07  3.78  18       0     0     3     3 jan.csv    
#> 15  10.4     8  472    205  2.93  5.25  18.0     0     0     3     4 jan.csv    
#> 16  10.4     8  460    215  3     5.42  17.8     0     0     3     4 jan.csv    

unlink(dir, recursive = TRUE)
```
