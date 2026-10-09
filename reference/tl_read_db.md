# Read from a DBI database connection

Executes a SQL query against an existing DBI connection and returns the
result as a `tidylearn_data` object. The connection is not closed by
this function — the caller is responsible for managing the connection
lifecycle.

## Usage

``` r
tl_read_db(conn, query, ...)
```

## Arguments

- conn:

  A DBI connection object (e.g., from
  [`DBI::dbConnect()`](https://dbi.r-dbi.org/reference/dbConnect.html)).

- query:

  A SQL query string.

- ...:

  Additional arguments passed to
  [`DBI::dbGetQuery()`](https://dbi.r-dbi.org/reference/dbGetQuery.html).

## Value

A `tidylearn_data` object containing the query results.

## Examples

``` r
# RSQLite imports DBI, so both are available here
conn <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
DBI::dbWriteTable(conn, "cars", mtcars)
tl_read_db(conn, "SELECT mpg, cyl, hp FROM cars WHERE cyl = 6")
#> -- tidylearn data ---------
#> Source: SQLiteConnection: SELECT mpg, cyl, hp FROM cars WHERE cyl = 6 
#> Format: database 
#> Read at: 2026-10-09 07:12:08 
#> 
#> # A tibble: 7 × 3
#>     mpg   cyl    hp
#> * <dbl> <dbl> <dbl>
#> 1  21       6   110
#> 2  21       6   110
#> 3  21.4     6   110
#> 4  18.1     6   105
#> 5  19.2     6   123
#> 6  17.8     6   123
#> 7  19.7     6   175
DBI::dbDisconnect(conn)
```
