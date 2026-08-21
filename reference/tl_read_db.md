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
# \donttest{
# conn <- DBI::dbConnect(RSQLite::SQLite(), "my_database.sqlite")
# data <- tl_read_db(conn, "SELECT * FROM my_table")
# DBI::dbDisconnect(conn)
# }
```
