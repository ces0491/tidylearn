# Read from Google BigQuery

Executes a SQL query against Google BigQuery and returns the result as a
`tidylearn_data` object. Requires the bigrquery package and valid Google
Cloud authentication.

## Usage

``` r
tl_read_bigquery(project, query, dataset = NULL, ...)
```

## Arguments

- project:

  Google Cloud project ID.

- query:

  A SQL query string (Standard SQL).

- dataset:

  Optional default dataset for unqualified table names.

- ...:

  Additional arguments passed to
  [`bigrquery::bq_project_query()`](https://bigrquery.r-dbi.org/reference/bq_query.html).

## Value

A `tidylearn_data` object containing the query results.

## Examples

``` r
# \donttest{
# data <- tl_read_bigquery(
#   project = "my-project",
#   query = "SELECT * FROM `my_dataset.my_table` LIMIT 1000"
# )
# }
```
