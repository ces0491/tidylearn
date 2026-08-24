# Data Ingestion with tidylearn

## Overview

Every machine learning workflow starts with data. tidylearn’s
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
family provides a single consistent interface for loading data from
files, databases, cloud storage, and APIs into tidy tibbles — ready for
[`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
and
[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md).

All readers return a `tidylearn_data` object, a tibble subclass that
carries metadata about the source, format, and read timestamp.

``` r

library(tidylearn)
library(dplyr)
```

------------------------------------------------------------------------

## The `tl_read()` Dispatcher

[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
auto-detects the data format and dispatches to the appropriate backend —
just like
[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
routes to the right algorithm:

``` r

# Format is auto-detected from the file extension
data <- tl_read("sales.csv")
data <- tl_read("results.xlsx", sheet = "Q1")
data <- tl_read("experiment.parquet")
data <- tl_read("config.json")
data <- tl_read("model_data.rds")

# Override format detection when the extension is ambiguous
data <- tl_read("export.txt", format = "tsv")
```

The result always prints with a metadata header:

``` r

tmp <- tempfile(fileext = ".csv")
write.csv(mtcars, tmp, row.names = FALSE)

data <- tl_read(tmp, .quiet = TRUE)
data
#> -- tidylearn data ---------
#> Source: /tmp/RtmpZMDR7O/file28e134888e95.csv 
#> Format: csv 
#> Read at: 2026-08-24 10:06:36 
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
```

------------------------------------------------------------------------

## File Formats

### CSV and TSV

Uses [readr](https://readr.tidyverse.org/) when available for fast,
column-type-aware parsing. Falls back to base R automatically if readr
is not installed.

``` r

# Create example files
tmp_csv <- tempfile(fileext = ".csv")
tmp_tsv <- tempfile(fileext = ".tsv")
write.csv(iris, tmp_csv, row.names = FALSE)
write.table(iris, tmp_tsv, sep = "\t", row.names = FALSE)

csv_data <- tl_read_csv(tmp_csv)
tsv_data <- tl_read_tsv(tmp_tsv)
nrow(csv_data)
#> [1] 150
```

### Excel

Reads `.xls`, `.xlsx`, and `.xlsm` files via
[readxl](https://readxl.tidyverse.org/). Select sheets by name or
position:

``` r

library(readxl)

path <- readxl_example("datasets.xlsx")
excel_data <- tl_read_excel(path, sheet = "mtcars")
head(excel_data, 3)
#> -- tidylearn data ---------
#> Source: /home/runner/work/_temp/Library/readxl/extdata/datasets.xlsx 
#> Format: excel 
#> Read at: 2026-08-24 10:06:37 
#> 
#> # A tibble: 3 × 11
#>     mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#>   <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1  21       6   160   110  3.9   2.62  16.5     0     1     4     4
#> 2  21       6   160   110  3.9   2.88  17.0     0     1     4     4
#> 3  22.8     4   108    93  3.85  2.32  18.6     1     1     4     1
```

### Parquet

Lightweight, columnar storage for large datasets. Uses
[nanoparquet](https://cran.r-project.org/package=nanoparquet) — a fast,
dependency-free reader:

``` r

library(nanoparquet)

tmp_pq <- tempfile(fileext = ".parquet")
write_parquet(iris, tmp_pq)

pq_data <- tl_read_parquet(tmp_pq)
nrow(pq_data)
#> [1] 150
```

### JSON

Reads tabular JSON (array of objects) via
[jsonlite](https://cran.r-project.org/package=jsonlite). Nested
structures are automatically flattened:

``` r

library(jsonlite)

tmp_json <- tempfile(fileext = ".json")
write_json(mtcars[1:5, ], tmp_json)

json_data <- tl_read_json(tmp_json)
json_data
#> -- tidylearn data ---------
#> Source: /tmp/RtmpZMDR7O/file28e12bd05f08.json 
#> Format: json 
#> Read at: 2026-08-24 10:06:37 
#> 
#> # A tibble: 5 × 11
#>     mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#> * <dbl> <int> <int> <int> <dbl> <dbl> <dbl> <int> <int> <int> <int>
#> 1  21       6   160   110  3.9   2.62  16.5     0     1     4     4
#> 2  21       6   160   110  3.9   2.88  17.0     0     1     4     4
#> 3  22.8     4   108    93  3.85  2.32  18.6     1     1     4     1
#> 4  21.4     6   258   110  3.08  3.22  19.4     1     0     3     1
#> 5  18.7     8   360   175  3.15  3.44  17.0     0     0     3     2
```

### RDS and RData

Native R serialisation formats — no extra packages needed:

``` r

tmp_rds <- tempfile(fileext = ".rds")
saveRDS(iris, tmp_rds)

rds_data <- tl_read_rds(tmp_rds)
nrow(rds_data)
#> [1] 150
```

``` r

tmp_rdata <- tempfile(fileext = ".rdata")
my_data <- mtcars
save(my_data, file = tmp_rdata)

# Name is auto-detected when there is a single data frame
rdata_data <- tl_read_rdata(tmp_rdata)
nrow(rdata_data)
#> [1] 32
```

------------------------------------------------------------------------

## Databases

All database readers use [DBI](https://dbi.r-dbi.org/) as the interface
layer. Each reader manages its own connection lifecycle — connect,
query, disconnect — so you only need to provide the path or credentials
and a SQL query.

### SQLite

The simplest database backend — no server required:

``` r

library(DBI)
library(RSQLite)

# Create an example database
tmp_db <- tempfile(fileext = ".sqlite")
conn <- dbConnect(SQLite(), tmp_db)
dbWriteTable(conn, "iris_tbl", iris)
dbDisconnect(conn)

# Read with tl_read_sqlite
db_data <- tl_read_sqlite(
  tmp_db,
  "SELECT * FROM iris_tbl WHERE Species = 'setosa'"
)
nrow(db_data)
#> [1] 50
```

### Using a Live Connection

If you already have a DBI connection, use
[`tl_read_db()`](https://tidylearn.sheetsolved.com/reference/tl_read_db.md)
directly — it will not close your connection:

``` r

conn <- dbConnect(SQLite(), ":memory:")
dbWriteTable(conn, "mtcars_tbl", mtcars)

sql <- "SELECT mpg, wt, hp FROM mtcars_tbl WHERE mpg > 20"
db_result <- tl_read_db(conn, sql)
db_result
#> -- tidylearn data ---------
#> Source: SQLiteConnection: SELECT mpg, wt, hp FROM mtcars_tbl WHERE mpg > 20 
#> Format: database 
#> Read at: 2026-08-24 10:06:38 
#> 
#> # A tibble: 14 × 3
#>      mpg    wt    hp
#>  * <dbl> <dbl> <dbl>
#>  1  21    2.62   110
#>  2  21    2.88   110
#>  3  22.8  2.32    93
#>  4  21.4  3.22   110
#>  5  24.4  3.19    62
#>  6  22.8  3.15    95
#>  7  32.4  2.2     66
#>  8  30.4  1.62    52
#>  9  33.9  1.84    65
#> 10  21.5  2.46    97
#> 11  27.3  1.94    66
#> 12  26    2.14    91
#> 13  30.4  1.51   113
#> 14  21.4  2.78   109

dbDisconnect(conn)
```

### PostgreSQL, MySQL, and BigQuery

These require a running database server or cloud service. The API is the
same — provide connection details and a SQL query:

``` r

# PostgreSQL
pg_data <- tl_read_postgres(
  dsn = "localhost",
  query = "SELECT * FROM sales WHERE year = 2025",
  dbname = "analytics",
  user = "myuser",
  password = "mypass"
)

# MySQL / MariaDB # nolint: commented_code_linter.
mysql_data <- tl_read_mysql(
  dsn = "mysql://user:pass@host:3306/mydb",
  query = "SELECT * FROM customers LIMIT 1000"
)

# BigQuery
bq_data <- tl_read_bigquery(
  project = "my-gcp-project",
  query = "SELECT * FROM `dataset.table` LIMIT 1000"
)
```

------------------------------------------------------------------------

## Cloud and API Sources

### Amazon S3

Downloads a file from S3 and auto-detects the format from the key’s
extension. Requires valid AWS credentials:

``` r

data <- tl_read_s3("s3://my-bucket/data/sales_2025.csv")
data <- tl_read_s3("s3://my-bucket/data/results.parquet", region = "eu-west-1")
```

### GitHub

Downloads raw files directly from a repository. Accepts full URLs or
`owner/repo` shorthand:

``` r

# Read a CSV from a public GitHub repository
data <- tl_read_github("tidyverse/dplyr",
  path = "data-raw/starwars.csv", ref = "main"
)
```

### Kaggle

Downloads datasets via the [Kaggle
CLI](https://github.com/Kaggle/kaggle-cli). Install with
`pip install kaggle` and configure your API credentials:

``` r

data <- tl_read_kaggle("zillow/zecon", file = "Zip_time_series.csv")
data <- tl_read_kaggle("titanic", file = "train.csv", type = "competition")
```

------------------------------------------------------------------------

## Multi-File Reading

Real-world data is often split across multiple files. tidylearn handles
three common patterns.

### Multiple Paths

Pass a character vector to
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md) —
each file is read and row-bound, with a `source_file` column tracking
origin:

``` r

dir <- tempdir()
write.csv(iris[1:50, ], file.path(dir, "batch1.csv"), row.names = FALSE)
write.csv(iris[51:100, ], file.path(dir, "batch2.csv"), row.names = FALSE)

paths <- file.path(dir, c("batch1.csv", "batch2.csv"))
combined <- tl_read(paths, .quiet = TRUE)
table(combined$source_file)
#> 
#> batch1.csv batch2.csv 
#>         50         50
```

### Directory Scanning

Point
[`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md)
at a directory. Filter by format, regex pattern, or scan recursively:

``` r

dir <- tempfile(pattern = "tl_vignette_")
dir.create(dir)
write.csv(iris[1:50, ], file.path(dir, "jan.csv"), row.names = FALSE)
write.csv(iris[51:100, ], file.path(dir, "feb.csv"), row.names = FALSE)
write.csv(iris[101:150, ], file.path(dir, "mar.csv"), row.names = FALSE)

# Read all CSVs from the directory
all_data <- tl_read_dir(dir, format = "csv", .quiet = TRUE)
nrow(all_data)
#> [1] 150
table(all_data$source_file)
#> 
#> feb.csv jan.csv mar.csv 
#>      50      50      50
```

``` r

# Filter with a regex pattern
subset <- tl_read_dir(dir, pattern = "^(jan|feb)", .quiet = TRUE)
nrow(subset)
#> [1] 100
```

Passing a directory path directly to
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
works too:

``` r

data <- tl_read("data/monthly_exports/")
```

### Zip Archives

[`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)
extracts the archive, auto-detects the file format, and reads the
contents. Select a specific file or let it discover data files
automatically:

``` r

# Create an example zip
dir <- tempfile(pattern = "tl_zip_src_")
dir.create(dir)
write.csv(iris, file.path(dir, "iris.csv"), row.names = FALSE)
zip_path <- tempfile(fileext = ".zip")
# -j stores the file without its directory path, so no setwd() is needed
utils::zip(zip_path, file.path(dir, "iris.csv"), flags = "-j9X")

zip_data <- tl_read_zip(zip_path, .quiet = TRUE)
nrow(zip_data)
#> [1] 150
attr(zip_data, "tl_format")
#> [1] "zip+csv"
```

Zip files are also auto-detected by
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md):

``` r

data <- tl_read("download.zip")
data <- tl_read("download.zip", file = "train.csv")
```

------------------------------------------------------------------------

## The `tidylearn_data` Class

Every reader returns a `tidylearn_data` object — a tibble subclass with
three metadata attributes:

| Attribute      | Description                                               |
|----------------|-----------------------------------------------------------|
| `tl_source`    | File path, URL, or description of the data source         |
| `tl_format`    | Detected or specified format (e.g., `"csv"`, `"zip+csv"`) |
| `tl_timestamp` | POSIXct timestamp of when the data was read               |

Because `tidylearn_data` inherits from `tbl_df`, all dplyr verbs,
ggplot2, and tidylearn functions work transparently:

``` r

tmp <- tempfile(fileext = ".csv")
write.csv(mtcars, tmp, row.names = FALSE)
data <- tl_read(tmp, .quiet = TRUE)

# Check metadata
attr(data, "tl_format")
#> [1] "csv"

# Works with dplyr
data %>%
  filter(mpg > 20) %>%
  select(mpg, wt, hp) %>%
  head(3)
#> -- tidylearn data ---------
#> Source: /tmp/RtmpZMDR7O/file28e13625660e.csv 
#> Format: csv 
#> Read at: 2026-08-24 10:06:39 
#> 
#> # A tibble: 3 × 3
#>     mpg    wt    hp
#>   <dbl> <dbl> <dbl>
#> 1  21    2.62   110
#> 2  21    2.88   110
#> 3  22.8  2.32    93
```

------------------------------------------------------------------------

## Full Pipeline

Combining
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
with the rest of tidylearn gives you a complete workflow from raw data
to published results:

``` r

# 1. Ingest
tmp <- tempfile(fileext = ".csv")
write.csv(iris, tmp, row.names = FALSE)
data <- tl_read(tmp, .quiet = TRUE)

# CSV files lose factor information, so convert character columns as needed
data <- data %>% mutate(Species = as.factor(Species))

# 2. Split
split <- tl_split(data, prop = 0.7, stratify = "Species", seed = 42)

# 3. Model
model <- tl_model(split$train, Species ~ ., method = "forest")

# 4. Evaluate
eval_result <- tl_evaluate(model, new_data = split$test)
eval_result
#> # A tibble: 1 × 2
#>   metric   value
#>   <chr>    <dbl>
#> 1 accuracy 0.956
```

------------------------------------------------------------------------

## Supported Formats Reference

| Format | Function | Backend | Dependency |
|----|----|----|----|
| CSV | [`tl_read_csv()`](https://tidylearn.sheetsolved.com/reference/tl_read_csv.md) | readr / base R | Suggests (readr) |
| TSV | [`tl_read_tsv()`](https://tidylearn.sheetsolved.com/reference/tl_read_tsv.md) | readr / base R | Suggests (readr) |
| Excel | [`tl_read_excel()`](https://tidylearn.sheetsolved.com/reference/tl_read_excel.md) | readxl | Suggests |
| Parquet | [`tl_read_parquet()`](https://tidylearn.sheetsolved.com/reference/tl_read_parquet.md) | nanoparquet | Suggests |
| JSON | [`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md) | jsonlite | Suggests |
| RDS | [`tl_read_rds()`](https://tidylearn.sheetsolved.com/reference/tl_read_rds.md) | base R | None |
| RData | [`tl_read_rdata()`](https://tidylearn.sheetsolved.com/reference/tl_read_rdata.md) | base R | None |
| SQLite | [`tl_read_sqlite()`](https://tidylearn.sheetsolved.com/reference/tl_read_sqlite.md) | DBI + RSQLite | Suggests |
| PostgreSQL | [`tl_read_postgres()`](https://tidylearn.sheetsolved.com/reference/tl_read_postgres.md) | DBI + RPostgres | Suggests |
| MySQL | [`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md) | DBI + RMariaDB | Suggests |
| BigQuery | [`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md) | bigrquery | Suggests |
| S3 | [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md) | paws.storage | Suggests |
| GitHub | [`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md) | base R | None |
| Kaggle | [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md) | Kaggle CLI | None (system) |
| Directory | [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md) | (dispatches) | — |
| Zip | [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md) | base R + (dispatches) | — |
| Multi-path | `tl_read(c(...))` | (dispatches) | — |
