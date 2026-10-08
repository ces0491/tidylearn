# Data Reading Functions for tidylearn

Functions for reading data from diverse sources into tidy
`tidylearn_data` objects. The main dispatcher
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
auto-detects the format from the file extension and routes to the
appropriate reader. All readers return a `tidylearn_data` object, which
is a tibble subclass carrying metadata about the data source.

## Details

Supported file formats:

- **CSV**: `.csv` files via readr (with base R fallback), and `.txt`
  files named directly

- **TSV**: `.tsv` files via readr (with base R fallback)

- **Excel**: `.xls`, `.xlsx`, `.xlsm` files via readxl

- **Parquet**: `.parquet` files via nanoparquet

- **JSON**: `.json` files, and newline-delimited `.ndjson` files, via
  jsonlite

- **RDS**: `.rds` files via base
  [`readRDS()`](https://rdrr.io/r/base/readRDS.html)

- **RData**: `.rdata`, `.rda` files via base
  [`load()`](https://rdrr.io/r/base/load.html)

CSV and TSV files compressed with gzip, bzip2 or xz (`data.csv.gz`) are
recognised by the extension under the compression one.

Supported databases (via DBI):

- **SQLite**: `.sqlite`, `.db` files via RSQLite

- **PostgreSQL**: via RPostgres

- **MySQL/MariaDB**: via RMariaDB

- **BigQuery**: `bigquery://project/dataset` URIs via bigrquery

Supported cloud/API sources:

- **S3**: `s3://` URIs via paws.storage

- **GitHub**: raw file download from repositories

- **Kaggle**: dataset download via Kaggle CLI

A `file://` URL is read as the local path it names. Other web URLs, and
URLs with any other scheme such as `ftp://`, are not read; download the
file first.

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
identify the origin of each row: the file's path below the directory or
archive it came from, or, for paths given directly, below the deepest
folder they share. Files in one folder are labelled by their bare names.

Directory and archive scans read the extensions listed above except
`.txt`, which in a folder is as likely to hold notes as data. Name a
`.txt` file directly, or select it with `pattern`, to read it.
