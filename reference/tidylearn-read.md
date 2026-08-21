# Data Reading Functions for tidylearn

Functions for reading data from diverse sources into tidy
`tidylearn_data` objects. The main dispatcher
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
auto-detects the format from the file extension and routes to the
appropriate reader. All readers return a `tidylearn_data` object, which
is a tibble subclass carrying metadata about the data source.

## Details

Supported file formats:

- **CSV**: `.csv` files via readr (with base R fallback)

- **TSV**: `.tsv` files via readr (with base R fallback)

- **Excel**: `.xls`, `.xlsx`, `.xlsm` files via readxl

- **Parquet**: `.parquet` files via nanoparquet

- **JSON**: `.json` files via jsonlite

- **RDS**: `.rds` files via base
  [`readRDS()`](https://rdrr.io/r/base/readRDS.html)

- **RData**: `.rdata`, `.rda` files via base
  [`load()`](https://rdrr.io/r/base/load.html)

Supported databases (via DBI):

- **SQLite**: `.sqlite`, `.db` files via RSQLite

- **PostgreSQL**: via RPostgres

- **MySQL/MariaDB**: via RMariaDB

- **BigQuery**: via bigrquery

Supported cloud/API sources:

- **S3**: `s3://` URIs via paws.storage

- **GitHub**: raw file download from repositories

- **Kaggle**: dataset download via Kaggle CLI

Multi-file reading:

- **Multiple paths**: pass a character vector to
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)

- **Directories**:
  [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md)
  scans for data files with optional pattern/format filtering and
  recursive scanning

- **Zip archives**:
  [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)
  extracts and reads from `.zip` files

When combining multiple files, a `source_file` column is added to
identify the origin of each row.
