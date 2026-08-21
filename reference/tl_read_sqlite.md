# Read from a SQLite database

Opens a SQLite database file, executes a SQL query, and returns the
result as a `tidylearn_data` object. The connection is automatically
closed when done. Requires DBI and RSQLite.

## Usage

``` r
tl_read_sqlite(path, query, ...)
```

## Arguments

- path:

  Path to a SQLite database file (`.sqlite` or `.db`).

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
# data <- tl_read_sqlite("my_database.sqlite", "SELECT * FROM my_table")
# }
```
