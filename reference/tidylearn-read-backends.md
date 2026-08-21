# Data Reading Backends for tidylearn

Backend readers for databases and cloud/API sources. All backends are
optional dependencies checked at call time via `tl_check_packages()`.

## Details

Database backends (via DBI):

- **SQLite**: via RSQLite

- **PostgreSQL**: via RPostgres

- **MySQL/MariaDB**: via RMariaDB

- **BigQuery**: via bigrquery

Cloud/API backends:

- **S3**: via paws.storage

- **GitHub**: via base
  [`download.file()`](https://rdrr.io/r/utils/download.file.html)

- **Kaggle**: via Kaggle CLI
