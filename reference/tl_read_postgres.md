# Read from a PostgreSQL database

Connects to a PostgreSQL database, executes a SQL query, and returns the
result as a `tidylearn_data` object. Accepts either a connection string
or individual connection parameters. Requires DBI and RPostgres.

## Usage

``` r
tl_read_postgres(
  dsn,
  query,
  dbname = NULL,
  user = NULL,
  password = NULL,
  port = 5432,
  ...
)
```

## Arguments

- dsn:

  A PostgreSQL connection string (e.g.,
  `"postgres://user:pass@host:port/dbname"`), or the database host if
  using named parameters.

- query:

  A SQL query string.

- dbname:

  Database name (if not in `dsn`).

- user:

  Username (if not in `dsn`).

- password:

  Password (if not in `dsn`).

- port:

  Port number. Default is 5432.

- ...:

  Additional arguments passed to
  [`DBI::dbConnect()`](https://dbi.r-dbi.org/reference/dbConnect.html).

## Value

A `tidylearn_data` object containing the query results.

## Examples

``` r
# \donttest{
# data <- tl_read_postgres(
#   dsn = "localhost",
#   query = "SELECT * FROM my_table",
#   dbname = "mydb",
#   user = "myuser",
#   password = "mypass"
# )
# }
```
