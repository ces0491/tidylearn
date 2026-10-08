#' @title Data Reading Backends for tidylearn
#' @name tidylearn-read-backends
#' @description Backend readers for databases and cloud/API sources.
#'   All backends are optional dependencies checked at call time via
#'   \code{tl_check_packages()}.
#'
#' @details
#' Database backends (via \pkg{DBI}):
#' \itemize{
#'   \item \strong{SQLite}: via \pkg{RSQLite}
#'   \item \strong{PostgreSQL}: via \pkg{RPostgres}
#'   \item \strong{MySQL/MariaDB}: via \pkg{RMariaDB}
#'   \item \strong{BigQuery}: via \pkg{bigrquery}
#' }
#'
#' Cloud/API backends:
#' \itemize{
#'   \item \strong{S3}: via \pkg{paws.storage}
#'   \item \strong{GitHub}: via base \code{download.file()}
#'   \item \strong{Kaggle}: via Kaggle CLI
#' }
NULL

# ---- Database readers ----

#' Read from a DBI database connection
#'
#' Executes a SQL query against an existing \pkg{DBI} connection and returns
#' the result as a \code{tidylearn_data} object. The connection is not closed
#' by this function — the caller is responsible for managing the connection
#' lifecycle.
#'
#' @param conn A \pkg{DBI} connection object (e.g., from
#'   \code{DBI::dbConnect()}).
#' @param query A SQL query string.
#' @param ... Additional arguments passed to \code{DBI::dbGetQuery()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examplesIf requireNamespace("RSQLite", quietly = TRUE)
#' # RSQLite imports DBI, so both are available here
#' conn <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
#' DBI::dbWriteTable(conn, "cars", mtcars)
#' tl_read_db(conn, "SELECT mpg, cyl, hp FROM cars WHERE cyl = 6")
#' DBI::dbDisconnect(conn)
#'
#' @export
tl_read_db <- function(conn, query, ...) {
  tl_check_packages("DBI")

  if (!inherits(conn, "DBIConnection")) {
    stop("'conn' must be a DBI connection object. ",
         "Create one with DBI::dbConnect().",
         call. = FALSE)
  }

  if (!is.character(query) || length(query) != 1 || !nzchar(query)) {
    stop("'query' must be a non-empty SQL string.", call. = FALSE)
  }

  data <- DBI::dbGetQuery(conn, query, ...)

  if (nrow(data) == 0) {
    warning("Query returned 0 rows.", call. = FALSE)
  }

  source_desc <- paste0(class(conn)[1], ": ", substr(query, 1, 80))
  new_tidylearn_data(data, source = source_desc, format = "database")
}

#' Read from a SQLite database
#'
#' Opens a SQLite database file, executes a SQL query, and returns the result
#' as a \code{tidylearn_data} object. The connection is automatically closed
#' when done. Requires \pkg{DBI} and \pkg{RSQLite}.
#'
#' @param path Path to a SQLite database file (\code{.sqlite} or \code{.db}).
#' @param query A SQL query string.
#' @param ... Additional arguments passed to \code{DBI::dbGetQuery()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examplesIf requireNamespace("RSQLite", quietly = TRUE)
#' # RSQLite imports DBI, so both are available here
#' path <- tempfile(fileext = ".sqlite")
#' conn <- DBI::dbConnect(RSQLite::SQLite(), path)
#' DBI::dbWriteTable(conn, "cars", mtcars)
#' DBI::dbDisconnect(conn)
#'
#' tl_read_sqlite(path, "SELECT mpg, cyl, hp FROM cars WHERE cyl = 6")
#' unlink(path)
#'
#' @export
tl_read_sqlite <- function(path, query, ...) {
  tl_validate_file_path(path)
  tl_check_packages("DBI", "RSQLite")

  if (missing(query) || !is.character(query) || !nzchar(query)) {
    stop("'query' is required. Provide a SQL string, e.g., ",
         "'SELECT * FROM my_table'.",
         call. = FALSE)
  }

  conn <- DBI::dbConnect(RSQLite::SQLite(), path)
  on.exit(DBI::dbDisconnect(conn), add = TRUE)

  data <- DBI::dbGetQuery(conn, query, ...)

  if (nrow(data) == 0) {
    warning("Query returned 0 rows.", call. = FALSE)
  }

  new_tidylearn_data(data, source = path, format = "sqlite")
}

#' Read from a PostgreSQL database
#'
#' Connects to a PostgreSQL database, executes a SQL query, and returns the
#' result as a \code{tidylearn_data} object. Accepts either a connection string
#' or individual connection parameters. Requires \pkg{DBI} and \pkg{RPostgres}.
#'
#' @param dsn A PostgreSQL connection string (e.g.,
#'   \code{"postgres://user:pass@host:port/dbname"}), or the database host if
#'   using named parameters. Percent-encode special characters in the user
#'   name and password (\code{@} as \code{\%40}). Query parameters such as
#'   \code{?sslmode=require} are passed to libpq as connection keywords.
#' @param query A SQL query string.
#' @param dbname Database name (if not in \code{dsn}).
#' @param user Username (if not in \code{dsn}).
#' @param password Password (if not in \code{dsn}), which keeps it out of
#'   the connection string.
#' @param port Port number. Default is 5432.
#' @param ... Additional arguments passed to \code{DBI::dbConnect()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examples
#' \dontrun{
#' # Needs a running PostgreSQL server
#' tl_read_postgres(
#'   dsn = "localhost",
#'   query = "SELECT * FROM my_table",
#'   dbname = "mydb",
#'   user = "myuser",
#'   password = Sys.getenv("PGPASSWORD")
#' )
#'
#' # The same connection as a URL, with the password still kept out of it
#' tl_read_postgres(
#'   "postgres://myuser@localhost:5432/mydb?sslmode=require",
#'   query = "SELECT * FROM my_table",
#'   password = Sys.getenv("PGPASSWORD")
#' )
#' }
#'
#' @export
tl_read_postgres <- function(dsn, query, dbname = NULL, user = NULL,
                             password = NULL, port = 5432, ...) {
  tl_check_packages("DBI", "RPostgres")

  if (missing(query) || !is.character(query) || !nzchar(query)) {
    stop("'query' is required. Provide a SQL string.",
         call. = FALSE)
  }

  conn <- NULL
  on.exit({
    if (!is.null(conn)) DBI::dbDisconnect(conn)
  }, add = TRUE)

  # Parse connection string if provided. RPostgres hands its extra
  # arguments to libpq as connection keywords, and libpq has no "dsn"
  # keyword, so the URL is taken apart here as tl_read_mysql() does.
  if (grepl("^postgres(ql)?://", dsn, ignore.case = TRUE)) {
    parsed <- tl_parse_db_url(dsn)
    conn_args <- c(
      tl_db_url_args(parsed, dbname, user, password, port),
      parsed$params,
      list(...)
    )
    conn <- tryCatch(
      do.call(DBI::dbConnect, c(list(RPostgres::Postgres()), conn_args)),
      error = function(e) {
        stop(
          "Failed to connect to PostgreSQL: ",
          e$message,
          "\nCheck your connection string.",
          call. = FALSE
        )
      }
    )
  } else {
    conn <- tryCatch(
      DBI::dbConnect(
        RPostgres::Postgres(),
        host = dsn,
        dbname = dbname,
        user = user,
        password = password,
        port = port,
        ...
      ),
      error = function(e) {
        stop(
          "Failed to connect to PostgreSQL: ",
          e$message,
          "\nCheck your connection parameters.",
          call. = FALSE
        )
      }
    )
  }

  data <- DBI::dbGetQuery(conn, query)

  if (nrow(data) == 0) {
    warning("Query returned 0 rows.", call. = FALSE)
  }

  # Redact before the DSN becomes an attribute of the returned object:
  # print.tidylearn_data() shows it and saveRDS() persists it
  source_desc <- if (grepl("^postgres", dsn, ignore.case = TRUE)) {
    dsn
  } else {
    paste0("postgres://", dsn)
  }
  new_tidylearn_data(
    data, source = tl_redact_db_url(source_desc), format = "postgres"
  )
}

#' Read from a MySQL/MariaDB database
#'
#' Connects to a MySQL or MariaDB database, executes a SQL query, and returns
#' the result as a \code{tidylearn_data} object. Accepts either a connection
#' string or individual connection parameters. Requires \pkg{DBI} and
#' \pkg{RMariaDB}.
#'
#' @param dsn A MySQL connection string (e.g.,
#'   \code{"mysql://user:pass@host:port/dbname"}), or the database host if
#'   using named parameters. Percent-encode special characters in the user
#'   name and password (\code{@} as \code{\%40}). Query parameters are
#'   refused, because \pkg{RMariaDB} ignores arguments it does not know;
#'   pass its \code{dbConnect()} arguments, such as \code{ssl.ca},
#'   through \code{...}.
#' @param query A SQL query string.
#' @param dbname Database name (if not in \code{dsn}).
#' @param user Username (if not in \code{dsn}).
#' @param password Password (if not in \code{dsn}), which keeps it out of
#'   the connection string.
#' @param port Port number (if not in \code{dsn}). Default is 3306.
#' @param ... Additional arguments passed to \code{DBI::dbConnect()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examples
#' \dontrun{
#' # Needs a running MySQL or MariaDB server
#' tl_read_mysql(
#'   dsn = "localhost",
#'   query = "SELECT * FROM my_table",
#'   dbname = "mydb",
#'   user = "myuser",
#'   password = Sys.getenv("MYSQL_PWD")
#' )
#' }
#'
#' @export
tl_read_mysql <- function(dsn, query, dbname = NULL, user = NULL,
                          password = NULL, port = 3306, ...) {
  tl_check_packages("DBI", "RMariaDB")

  if (missing(query) || !is.character(query) || !nzchar(query)) {
    stop("'query' is required. Provide a SQL string.",
         call. = FALSE)
  }

  conn <- NULL
  on.exit({
    if (!is.null(conn)) DBI::dbDisconnect(conn)
  }, add = TRUE)

  # Parse connection string or use named params
  if (grepl("^mysql://", dsn, ignore.case = TRUE)) {
    parsed <- tl_parse_db_url(dsn)
    # Passed on, a parameter such as ssl-mode=REQUIRED would be ignored by
    # RMariaDB and the connection made without the TLS asked for
    if (length(parsed$params) > 0L) {
      stop(
        "MySQL connection strings cannot carry query parameters (",
        paste(names(parsed$params), collapse = ", "), "). RMariaDB ",
        "ignores arguments it does not know, so they would be dropped ",
        "without a word. Pass the matching RMariaDB::dbConnect() ",
        "arguments, such as ssl.ca, through '...' instead.",
        call. = FALSE
      )
    }
    conn_args <- c(
      tl_db_url_args(parsed, dbname, user, password, port),
      list(...)
    )
    conn <- tryCatch(
      do.call(DBI::dbConnect, c(list(RMariaDB::MariaDB()), conn_args)),
      error = function(e) {
        stop(
          "Failed to connect to MySQL: ",
          e$message,
          "\nCheck your connection string.",
          call. = FALSE
        )
      }
    )
  } else {
    conn <- tryCatch(
      DBI::dbConnect(
        RMariaDB::MariaDB(),
        host = dsn,
        dbname = dbname,
        user = user,
        password = password,
        port = port,
        ...
      ),
      error = function(e) {
        stop(
          "Failed to connect to MySQL: ",
          e$message,
          "\nCheck your connection parameters.",
          call. = FALSE
        )
      }
    )
  }

  data <- DBI::dbGetQuery(conn, query)

  if (nrow(data) == 0) {
    warning("Query returned 0 rows.", call. = FALSE)
  }

  source_desc <- if (grepl("^mysql", dsn, ignore.case = TRUE)) {
    dsn
  } else {
    paste0("mysql://", dsn)
  }
  new_tidylearn_data(
    data, source = tl_redact_db_url(source_desc), format = "mysql"
  )
}

#' Read from Google BigQuery
#'
#' Executes a SQL query against Google BigQuery and returns the result as a
#' \code{tidylearn_data} object. Requires the \pkg{bigrquery} package and
#' valid Google Cloud authentication.
#'
#' @param project Google Cloud project ID.
#' @param query A SQL query string (Standard SQL).
#' @param dataset Optional default dataset for unqualified table names,
#'   in \code{project}.
#' @param ... Additional arguments passed to
#'   \code{bigrquery::bq_project_query()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examples
#' \dontrun{
#' # Needs Google Cloud credentials
#' tl_read_bigquery(
#'   project = "my-project",
#'   query = "SELECT * FROM `my_dataset.my_table` LIMIT 1000"
#' )
#'
#' # Unqualified table names resolve against `dataset`
#' tl_read_bigquery(
#'   project = "my-project",
#'   query = "SELECT * FROM my_table LIMIT 1000",
#'   dataset = "my_dataset"
#' )
#' }
#'
#' @export
tl_read_bigquery <- function(project, query, dataset = NULL, ...) {
  tl_check_packages("bigrquery")

  if (missing(query) || !is.character(query) || !nzchar(query)) {
    stop("'query' is required. Provide a SQL string.", call. = FALSE)
  }

  # Handle bigquery:// URI format from dispatcher
  if (grepl("^bigquery://", project)) {
    parts <- strsplit(sub("^bigquery://", "", project), "/")[[1]]
    project <- parts[1]
    if (length(parts) > 1 && is.null(dataset)) {
      dataset <- parts[2]
    }
  }

  if (!is.null(dataset) && (!is.character(dataset) || length(dataset) != 1L ||
                              is.na(dataset) || !nzchar(dataset))) {
    stop("'dataset' must be a single dataset name.", call. = FALSE)
  }

  # bq_project_query() passes its dots to bq_perform_query(), whose
  # default_dataset is what unqualified table names resolve against
  query_args <- list(...)
  if (!is.null(dataset) && is.null(query_args[["default_dataset"]])) {
    query_args$default_dataset <- bigrquery::bq_dataset(project, dataset)
  }

  tb <- tryCatch(
    do.call(bigrquery::bq_project_query, c(list(project, query), query_args)),
    error = function(e) {
      stop("BigQuery query failed: ", e$message,
           "\nCheck your project ID, query, and authentication.",
           call. = FALSE)
    }
  )

  data <- bigrquery::bq_table_download(tb)

  if (nrow(data) == 0) {
    warning("Query returned 0 rows.", call. = FALSE)
  }

  source_desc <- paste0("bigquery://", project)
  if (!is.null(dataset)) {
    source_desc <- paste0(source_desc, "/", dataset)
  }
  new_tidylearn_data(data, source = source_desc, format = "bigquery")
}

# ---- Cloud/API readers ----

#' Read from Amazon S3
#'
#' Downloads a file from an S3 bucket and reads it into a \code{tidylearn_data}
#' object. The file format is auto-detected from the key's extension, or can be
#' specified explicitly. Requires the \pkg{paws.storage} package and valid AWS
#' credentials.
#'
#' @param source An S3 URI (e.g., \code{"s3://bucket/path/to/file.csv"}).
#'   Zip archives are not read from S3: download the object and read it
#'   with \code{tl_read_zip()}.
#' @param format Optional format override for the downloaded file. If
#'   \code{NULL}, auto-detected from the S3 key extension.
#' @param region AWS region. If \code{NULL}, uses the default from your AWS
#'   configuration.
#' @param ... Additional arguments passed to the format-specific reader.
#' @param trust_rds Logical. Read an \code{.rds}, \code{.rdata} or
#'   \code{.rda} object on R older than 4.4.0? Those versions can run code
#'   embedded in a crafted file as it is read (CVE-2024-27322), so such
#'   objects are refused there unless this is \code{TRUE}. It has no
#'   effect on R 4.4.0 or later. Default \code{FALSE}.
#'
#' @section Reading R serialisation files:
#' An \code{.rds} or \code{.rdata} object is rebuilt with \code{readRDS()}
#' or \code{load()}, which recreate whatever R objects the file describes.
#' Read them only from a bucket you trust, on any version of R.
#'
#' @return A \code{tidylearn_data} object containing the downloaded data.
#'
#' @examples
#' \dontrun{
#' # Needs AWS credentials
#' tl_read_s3("s3://my-bucket/data/sales.csv")
#' tl_read_s3("s3://my-bucket/data/results.parquet", region = "eu-west-1")
#' }
#'
#' @export
tl_read_s3 <- function(source, format = NULL, region = NULL, ...,
                       trust_rds = FALSE) {
  tl_check_packages("paws.storage")
  tl_check_trust_rds(trust_rds)

  # Parse s3:// URI. Guard the length first: strsplit(character(0), ...)
  # is an empty list, so [[1]] was "subscript out of bounds" rather than
  # the Invalid S3 URI message every other malformed input produces.
  if (!is.character(source)) {
    stop("'source' must be an S3 URI string, e.g. s3://bucket/key.csv; got ",
         paste(class(source), collapse = "/"), ".",
         call. = FALSE)
  }
  if (length(source) != 1L) {
    stop("'source' must be a single S3 URI, e.g. s3://bucket/key.csv; got ",
         "length ", length(source), ".",
         call. = FALSE)
  }

  s3_path <- sub("^s3://", "", source, ignore.case = TRUE)
  parts <- strsplit(s3_path, "/", fixed = TRUE)[[1]]

  if (length(parts) < 2) {
    stop("Invalid S3 URI: '", source, "'. ",
         "Expected format: s3://bucket/key",
         call. = FALSE)
  }

  bucket <- parts[1]
  key <- paste(parts[-1], collapse = "/")

  # Detect format from key extension
  if (is.null(format)) {
    if (tolower(tools::file_ext(key)) == "zip") {
      stop("tl_read_s3() cannot read a zip archive. Download it, then ",
           "read the local copy with tl_read_zip().",
           call. = FALSE)
    }
    format <- tryCatch(
      tl_detect_format(key),
      error = function(e) {
        stop("Cannot detect file format from S3 key '", key, "'. ",
             "Specify the 'format' argument.",
             call. = FALSE)
      }
    )
  }

  # Before anything is downloaded, and before credentials are needed
  tl_check_remote_rds(format, "S3", trust_rds)

  # Create S3 client
  config <- list()
  if (!is.null(region)) config$region <- region

  s3 <- tryCatch(
    paws.storage::s3(config = config),
    error = function(e) {
      stop("Failed to create S3 client: ", e$message,
           "\nCheck your AWS credentials and configuration.",
           call. = FALSE)
    }
  )

  # Download to temp file
  ext <- tools::file_ext(key)
  tmp <- tempfile(fileext = paste0(".", ext))
  on.exit(unlink(tmp), add = TRUE)

  resp <- tryCatch(
    s3$get_object(Bucket = bucket, Key = key),
    error = function(e) {
      stop("Failed to download s3://", bucket, "/", key, ": ", e$message,
           call. = FALSE)
    }
  )

  writeBin(resp$Body, tmp)

  # Read the downloaded file using the appropriate reader
  result <- switch(format,
    "csv"     = tl_read_csv(tmp, ...),
    "tsv"     = tl_read_tsv(tmp, ...),
    "excel"   = tl_read_excel(tmp, ...),
    "parquet" = tl_read_parquet(tmp, ...),
    "json"    = tl_read_json(tmp, ...),
    "rds"     = tl_read_rds(tmp),
    "rdata"   = tl_read_rdata(tmp, ...),
    stop("Unsupported format '", format, "' for S3 source.", call. = FALSE)
  )

  # Override source to show S3 URI instead of temp path
  attr(result, "tl_source") <- source
  attr(result, "tl_format") <- paste0("s3+", format)
  result
}

#' Read from GitHub
#'
#' Downloads a raw file from a GitHub repository and reads it into a
#' \code{tidylearn_data} object. Accepts either a full GitHub URL or a
#' \code{owner/repo} shorthand with a file path.
#'
#' @param source A GitHub URL or \code{"owner/repo"} string. A URL is either
#'   a file page (\code{https://github.com/<owner>/<repo>/blob/<ref>/<path>},
#'   with \code{raw} in place of \code{blob} or with neither, and with or
#'   without \code{www.} and a query such as \code{?raw=true}) or a
#'   raw file (\code{https://raw.githubusercontent.com/...}), whose query is
#'   kept for the download. Zip archives are not read from GitHub: download
#'   the file and read it with \code{tl_read_zip()}.
#' @param path Path to the file within the repository (required when
#'   \code{source} is \code{"owner/repo"} format).
#' @param ref Branch, tag, or commit SHA. Default is \code{"main"}.
#' @param ... Additional arguments passed to the format-specific reader.
#' @param trust_rds Logical. Read an \code{.rds}, \code{.rdata} or
#'   \code{.rda} file on R older than 4.4.0? Those versions can run code
#'   embedded in a crafted file as it is read (CVE-2024-27322), so such
#'   files are refused there unless this is \code{TRUE}. It has no effect
#'   on R 4.4.0 or later. Default \code{FALSE}.
#'
#' @section Reading R serialisation files:
#' An \code{.rds} or \code{.rdata} file is rebuilt with \code{readRDS()} or
#' \code{load()}, which recreate whatever R objects the file describes.
#' Read them only from a repository you trust, on any version of R.
#'
#' @return A \code{tidylearn_data} object containing the downloaded data.
#'
#' @examples
#' \dontrun{
#' # Downloads over the network
#' tl_read_github("user/repo", path = "data/file.csv")
#' tl_read_github("https://github.com/user/repo/blob/main/data/file.csv")
#' }
#'
#' @export
tl_read_github <- function(source, path = NULL, ref = "main", ...,
                           trust_rds = FALSE) {
  tl_check_trust_rds(trust_rds)
  raw_url <- tl_github_raw_url(source, path, ref)

  # The file name is the URL's path without its query, which a link
  # copied from GitHub often carries (?raw=true)
  file_name <- basename(sub("[?#].*$", "", raw_url))

  if (tolower(tools::file_ext(file_name)) == "zip") {
    stop("tl_read_github() cannot read a zip archive. Download it, then ",
         "read the local copy with tl_read_zip().",
         call. = FALSE)
  }

  # Detect format from the URL file extension
  format <- tryCatch(
    tl_detect_format(file_name),
    error = function(e) {
      stop("Cannot detect file format from GitHub URL. ",
           "The file must have a recognizable extension (csv, json, etc.).",
           call. = FALSE)
    }
  )

  tl_check_remote_rds(format, "GitHub", trust_rds)

  # Download to temp file
  ext <- tools::file_ext(file_name)
  tmp <- tempfile(fileext = paste0(".", ext))
  on.exit(unlink(tmp), add = TRUE)

  tryCatch(
    tl_download_file(raw_url, tmp),
    error = function(e) {
      stop("Failed to download from GitHub: ", e$message,
           "\nURL: ", raw_url,
           call. = FALSE)
    }
  )

  # Read the downloaded file
  result <- switch(format,
    "csv"     = tl_read_csv(tmp, ...),
    "tsv"     = tl_read_tsv(tmp, ...),
    "excel"   = tl_read_excel(tmp, ...),
    "parquet" = tl_read_parquet(tmp, ...),
    "json"    = tl_read_json(tmp, ...),
    "rds"     = tl_read_rds(tmp),
    "rdata"   = tl_read_rdata(tmp, ...),
    stop("Unsupported format '", format, "' for GitHub source.", call. = FALSE)
  )

  # Override source to show GitHub URL instead of temp path
  attr(result, "tl_source") <- source
  attr(result, "tl_format") <- paste0("github+", format)
  result
}

#' Read from Kaggle
#'
#' Downloads a dataset file from Kaggle using the Kaggle CLI and reads it into
#' a \code{tidylearn_data} object. Requires the Kaggle CLI to be installed and
#' configured (\code{pip install kaggle}).
#'
#' @param source A Kaggle dataset slug (e.g., \code{"user/dataset-name"}) or a
#'   Kaggle URL. A URL may point at any tab of the dataset or competition
#'   page (\code{/data}, \code{/code}, \code{/versions/2}) and may carry a
#'   query; a competition URL is read as a competition without
#'   \code{type} being set.
#' @param file The specific file to read from the dataset, as a path within
#'   it; a file the CLI saved under its base name is found too. If
#'   \code{NULL}, the download is searched for files these readers handle
#'   (CSV, TSV, Excel, Parquet and JSON, with compressed CSV/TSV and
#'   \code{.ndjson}); with several, the newest is read and a message names
#'   it.
#' @param dest Directory to keep the download in. The default is a fresh
#'   per-dataset directory under \code{tempdir()}. A supplied \code{dest}
#'   receives this download's files, replacing files of the same name;
#'   nothing else in it is read, unpacked or changed.
#' @param type Either \code{"dataset"} (default) or \code{"competition"}.
#'   Left unset, a competition URL in \code{source} sets it.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object containing the downloaded data.
#'
#' @examples
#' \dontrun{
#' # Needs the Kaggle CLI and Kaggle credentials
#' tl_read_kaggle("zillow/zecon", file = "Zip_time_series.csv")
#' tl_read_kaggle("titanic", file = "train.csv", type = "competition")
#' tl_read_kaggle("https://www.kaggle.com/competitions/titanic/data",
#'                file = "train.csv")
#' }
#'
#' @export
tl_read_kaggle <- function(source, file = NULL, dest = NULL,
                           type = "dataset", ...) {
  if (!is.character(type) || length(type) != 1L ||
        !type %in% c("dataset", "competition")) {
    stop("'type' must be \"dataset\" or \"competition\".", call. = FALSE)
  }

  # A pasted URL names its own kind, and its slug is taken from the
  # segments after /datasets/ or /competitions/: a link copied from a tab
  # such as /data ends in the tab's name.
  if (tl_is_kaggle_url(source)) {
    parsed <- tl_parse_kaggle_url(source)
    if (missing(type)) {
      type <- parsed$type
    } else if (!identical(type, parsed$type)) {
      stop("'", source, "' is a Kaggle ", parsed$type, " URL, but type = \"",
           type, "\". Leave 'type' unset or pass type = \"", parsed$type,
           "\".", call. = FALSE)
    }
    source <- parsed$slug
  }

  # Validate before anything is interpolated into a command line, and
  # before looking for the CLI: a malformed slug is the caller's mistake
  # whether or not the tool is installed, and saying so is more use than
  # reporting a missing dependency. The slug arrives from a pasted URL or
  # straight from the caller -- the URL branch above is skipped entirely
  # for a bare string -- and system2() applies shQuote() to the command
  # but not to the arguments, so a slug is pasted into the shell line as
  # written.
  source <- tl_check_kaggle_slug(source, type)
  if (!is.null(file)) {
    file <- tl_check_kaggle_filename(file)
  }

  # Check Kaggle CLI is installed
  tl_check_kaggle_cli()

  # Download into an empty directory of our own, even when the caller
  # names one. In a folder shared with anything else, the search below
  # would find files other downloads left there, and unpacking would
  # extract every zip in it, an unrelated archive's included. Without a
  # dest the download stays in a per-dataset folder for the session. With
  # one it is staged in a fresh tempfile(), which can never be dest itself,
  # whatever folder the caller names.
  if (is.null(dest)) {
    staging <- file.path(tempdir(), paste0("tl_kaggle_", tl_slug_key(source)))
    unlink(staging, recursive = TRUE, force = TRUE)
  } else {
    staging <- tempfile("tl_kaggle_")
    on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  }
  dir.create(staging, recursive = TRUE, showWarnings = FALSE)

  # Download the dataset. Quote the caller-derived values: a slug is
  # already restricted to safe characters by the checks above, but a
  # destination path may legitimately contain spaces.
  if (type == "competition") {
    args <- c("competitions", "download", "-c", shQuote(source),
              "-p", shQuote(staging))
    if (!is.null(file)) args <- c(args, "-f", shQuote(file))
  } else {
    args <- c("datasets", "download", "-d", shQuote(source),
              "-p", shQuote(staging), "--unzip")
    if (!is.null(file)) args <- c(args, "-f", shQuote(file))
  }

  result <- tryCatch(
    system2(tl_kaggle_command(), args, stdout = TRUE, stderr = TRUE),
    error = function(e) {
      stop("Kaggle CLI failed: ", e$message, call. = FALSE)
    }
  )

  status <- attr(result, "status")
  if (!is.null(status) && status != 0) {
    stop("Kaggle download failed:\n", paste(result, collapse = "\n"),
         call. = FALSE)
  }

  # Competition downloads arrive as a zip. The dataset endpoint takes
  # --unzip; the competition endpoint has no such flag, so unpack here or
  # the search below finds no data file at all.
  tl_unzip_kaggle_archives(staging)

  # Kept before reading, so the download survives a file that cannot be
  # read
  if (!is.null(dest)) {
    tl_keep_kaggle_download(staging, dest)
  }

  # The formats the switch below reads
  kaggle_formats <- c("csv", "tsv", "excel", "parquet", "json")

  # Find the downloaded file
  if (!is.null(file)) {
    # The CLI saves a single requested file under its base name, and an
    # unpacked archive keeps the path it was given in, so look in both
    # places: file = "data/train.csv" arrives as train.csv on its own
    downloaded <- file.path(staging, c(file, basename(file)))
    downloaded <- downloaded[file.exists(downloaded)][1]
    if (is.na(downloaded)) {
      stop("File '", file, "' is not in the Kaggle download. It holds: ",
           paste(list.files(staging, recursive = TRUE), collapse = ", "),
           call. = FALSE)
    }
  } else {
    # The extensions the readers below handle, from the table the
    # directory and archive scans use, so the two cannot disagree
    candidates <- list.files(
      staging, pattern = tl_scan_pattern(kaggle_formats), full.names = TRUE,
      recursive = TRUE, ignore.case = TRUE
    )
    candidates <- candidates[order(file.mtime(candidates), decreasing = TRUE)]

    if (length(candidates) == 0) {
      stop("No data files found in downloaded Kaggle dataset. ",
           "Specify the 'file' argument.",
           call. = FALSE)
    }

    if (length(candidates) > 1) {
      message("Multiple files found. Reading: ", basename(candidates[1]))
    }

    downloaded <- candidates[1]
  }

  # Detect format and read
  format <- tl_detect_format(downloaded)
  result <- switch(format,
    "csv"     = tl_read_csv(downloaded, ...),
    "tsv"     = tl_read_tsv(downloaded, ...),
    "excel"   = tl_read_excel(downloaded, ...),
    "parquet" = tl_read_parquet(downloaded, ...),
    "json"    = tl_read_json(downloaded, ...),
    stop("Unsupported format '", format, "' in Kaggle download.", call. = FALSE)
  )

  # Override source metadata
  attr(result, "tl_source") <- paste0("kaggle://", source)
  attr(result, "tl_format") <- paste0("kaggle+", format)
  result
}

# ---- Internal helpers ----

#' Download a URL to a local file
#'
#' The one place a reader fetches over HTTP, so tests can stand in for
#' the network.
#'
#' @param url The URL to fetch.
#' @param destfile Where to write it.
#' @return The status from \code{utils::download.file()}, invisibly.
#' @keywords internal
#' @noRd
tl_download_file <- function(url, destfile) {
  invisible(utils::download.file(url, destfile, mode = "wb", quiet = TRUE))
}

#' Can reading an R serialisation file run code embedded in it?
#'
#' Before R 4.4.0, \code{readRDS()} and \code{load()} rebuilt promises
#' stored in the file, so a crafted file ran code as soon as the object
#' was used (CVE-2024-27322).
#'
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_unserialize_runs_code <- function() {
  getRversion() < "4.4.0"
}

#' Validate a trust_rds argument
#' @param trust_rds The value supplied.
#' @keywords internal
#' @noRd
tl_check_trust_rds <- function(trust_rds) {
  if (!is.logical(trust_rds) || length(trust_rds) != 1L || is.na(trust_rds)) {
    stop("'trust_rds' must be TRUE or FALSE.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Refuse a downloaded R serialisation file where reading it can run code
#'
#' Called before the download, so nothing is fetched for a refused file.
#'
#' @param format The detected format.
#' @param where The source, for the message: "GitHub" or "S3".
#' @param trust_rds The caller's opt-in.
#' @keywords internal
#' @noRd
tl_check_remote_rds <- function(format, where, trust_rds) {
  if (format %in% c("rds", "rdata") && !isTRUE(trust_rds) &&
        tl_unserialize_runs_code()) {
    stop(
      "Reading an .rds or .rdata file from ", where, " hands downloaded ",
      "bytes to R's deserialiser, and on R ", getRversion(), " a crafted ",
      "file can run code as it is read (CVE-2024-27322, fixed in R ",
      "4.4.0). Pass trust_rds = TRUE if you trust the source, or upgrade R.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' The raw-file URL for a GitHub source
#'
#' The host is read from the parsed URL, and the owner and repository from
#' the path's segments. Substituting text instead would take
#' "www.github.com" for owner/repo shorthand, and remove an owner called
#' "blob" in place of the /blob/ segment.
#'
#' @param source A GitHub URL or "owner/repo" string.
#' @param path,ref As for tl_read_github().
#' @return The URL to download.
#' @keywords internal
#' @noRd
tl_github_raw_url <- function(source, path, ref) {
  if (!is.character(source) || length(source) != 1L || is.na(source)) {
    stop("'source' must be a single GitHub URL or \"owner/repo\" string.",
         call. = FALSE)
  }

  if (!tl_has_scheme(source)) {
    # owner/repo shorthand
    if (is.null(path)) {
      stop("'path' is required when 'source' is in 'owner/repo' format.",
           call. = FALSE)
    }
    return(paste0(
      "https://raw.githubusercontent.com/", source, "/", ref, "/", path
    ))
  }

  host <- tl_url_host(source)
  if (identical(host, "raw.githubusercontent.com")) {
    # A private file's access token is in the query, so it stays
    return(source)
  }
  if (!host %in% c("github.com", "www.github.com")) {
    stop("'", source, "' is not a GitHub URL. tl_read_github() reads ",
         "github.com file links and raw.githubusercontent.com files.",
         call. = FALSE)
  }

  # github.com/<owner>/<repo>/blob/<ref>/<path>, or raw in place of blob,
  # or the ref straight after the repository. The query (?raw=true,
  # ?plain=1) only steers GitHub's page.
  page <- sub("^[A-Za-z][A-Za-z0-9+.-]*://[^/]*", "", source)
  page <- sub("[?#].*$", "", page)
  segments <- strsplit(page, "/", fixed = TRUE)[[1]]
  segments <- segments[nzchar(segments)]

  raw_host <- "https://raw.githubusercontent.com/"
  if (length(segments) >= 5L && segments[3] %in% c("blob", "raw")) {
    return(paste0(raw_host, paste(segments[-3], collapse = "/")))
  }
  # A /tree/ link is a folder, so it is no file to read
  if (length(segments) >= 4L && !segments[3] %in% c("blob", "raw", "tree")) {
    return(paste0(raw_host, paste(segments, collapse = "/")))
  }

  stop("'", source, "' is not a link to a file on GitHub. Expected ",
       "https://github.com/<owner>/<repo>/blob/<ref>/<path>.",
       call. = FALSE)
}

#' Refuse a Kaggle slug that is not one
#'
#' The slug reaches \code{system2()}, which applies \code{shQuote()} to the
#' command and leaves the arguments as written, so whatever is in the slug
#' is pasted into a shell command line. A pasted dataset URL is the vector,
#' and \code{tl_read_kaggle()} skips URL parsing entirely when the caller
#' passes a bare string, so the slug is not necessarily anything Kaggle
#' produced. Kaggle's own identifiers are letters, digits, hyphens,
#' underscores and dots, so accept exactly that and nothing else.
#'
#' @param slug The dataset slug or competition name
#' @param type \code{"dataset"} (owner/name) or \code{"competition"} (name)
#' @return The slug, unchanged, when it is well formed
#' @keywords internal
#' @noRd
tl_check_kaggle_slug <- function(slug, type = "dataset") {
  if (!is.character(slug) || length(slug) != 1L || is.na(slug)) {
    stop("Kaggle source must be a single string.", call. = FALSE)
  }

  segment <- "[A-Za-z0-9][A-Za-z0-9._-]*"
  pattern <- if (identical(type, "competition")) {
    paste0("^", segment, "$")
  } else {
    paste0("^", segment, "/", segment, "$")
  }

  if (!grepl(pattern, slug)) {
    is_competition <- identical(type, "competition")
    noun <- if (is_competition) "competition name" else "dataset slug"
    shape <- if (is_competition) {
      "a name like \"titanic\""
    } else {
      "\"owner/dataset-name\""
    }
    stop(
      "'", slug, "' is not a valid Kaggle ", noun,
      ". Expected ", shape,
      ", using only letters, digits, dots, hyphens and underscores.",
      call. = FALSE
    )
  }

  slug
}

#' Refuse a Kaggle file name that could reach the shell or escape the
#' download directory
#'
#' The name is both interpolated into the CLI call and pasted onto
#' \code{dest} with \code{file.path()}, so it has to be a plain relative
#' path with no parent-directory steps.
#'
#' @param file The requested file name
#' @return The file name, unchanged, when it is safe
#' @keywords internal
#' @noRd
tl_check_kaggle_filename <- function(file) {
  if (!is.character(file) || length(file) != 1L || is.na(file)) {
    stop("Kaggle 'file' must be a single string.", call. = FALSE)
  }

  if (grepl("^([A-Za-z]:|[\\\\/])", file) ||
        any(strsplit(file, "[\\\\/]")[[1]] == "..")) {
    stop(
      "Kaggle 'file' must be a relative path inside the dataset, but got '",
      file, "'.",
      call. = FALSE
    )
  }

  if (!grepl("^[A-Za-z0-9._/-]+$", file)) {
    stop(
      "Kaggle 'file' may contain only letters, digits, dots, hyphens, ",
      "underscores and '/', but got '", file, "'.",
      call. = FALSE
    )
  }

  file
}

#' A filesystem-safe key for a slug, to name its download directory
#'
#' @param slug A validated Kaggle slug
#' @return The slug with its separator replaced
#' @keywords internal
#' @noRd
tl_slug_key <- function(slug) {
  gsub("[^A-Za-z0-9._-]", "_", slug)
}

#' Unpack any zip archives the Kaggle CLI left behind
#'
#' @param dest The download directory
#' @return `TRUE`, invisibly
#' @keywords internal
#' @noRd
tl_unzip_kaggle_archives <- function(dest) {
  archives <- list.files(
    dest, pattern = "\\.zip$", full.names = TRUE, ignore.case = TRUE
  )
  for (archive in archives) {
    members <- tryCatch(
      utils::unzip(archive, list = TRUE)$Name,
      error = function(e) {
        warning("Could not unpack '", basename(archive), "': ",
                conditionMessage(e), call. = FALSE)
        NULL
      }
    )
    if (is.null(members)) {
      next
    }

    # Outside the tryCatch below, which turns unzip()'s complaints into
    # warnings: a member that would land outside `dest` stops the read
    tl_refuse_unsafe_zip(archive, members)

    tryCatch(
      utils::unzip(archive, exdir = dest),
      warning = function(w) {
        warning("Could not unpack '", basename(archive), "': ",
                conditionMessage(w), call. = FALSE)
      },
      error = function(e) {
        warning("Could not unpack '", basename(archive), "': ",
                conditionMessage(e), call. = FALSE)
      }
    )
  }
  invisible(TRUE)
}

#' Copy a finished Kaggle download into the caller's folder
#'
#' Files of the same name are replaced, as a download into that folder
#' would replace them; nothing else there is touched.
#'
#' @param staging The directory the download was made in.
#' @param dest The caller's folder.
#' @return The copied paths, invisibly.
#' @keywords internal
#' @noRd
tl_keep_kaggle_download <- function(staging, dest) {
  files <- list.files(staging, recursive = TRUE)
  targets <- file.path(dest, files)
  for (folder in unique(dirname(targets))) {
    dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  }

  copied <- file.copy(file.path(staging, files), targets, overwrite = TRUE)
  if (!all(copied)) {
    warning("Could not copy ", paste(files[!copied], collapse = ", "),
            " into '", dest, "'.", call. = FALSE)
  }
  invisible(targets)
}

#' The command that runs the Kaggle CLI
#'
#' One place for the name, so tests can run a stand-in by its full path.
#' Putting the stand-in first on PATH is not enough on Windows, which
#' prefers kaggle.exe anywhere on PATH to a kaggle.bat ahead of it.
#'
#' @return The command to pass to \code{system2()}.
#' @keywords internal
#' @noRd
tl_kaggle_command <- function() {
  "kaggle"
}

#' Check that the Kaggle CLI is installed
#' @keywords internal
#' @noRd
tl_check_kaggle_cli <- function() {
  result <- tryCatch(
    system2(tl_kaggle_command(), "--version", stdout = TRUE, stderr = TRUE),
    error = function(e) NULL,
    warning = function(w) NULL
  )

  if (is.null(result)) {
    stop("Kaggle CLI not found. Install it with: pip install kaggle\n",
         "Then configure credentials: ",
         "https://github.com/Kaggle/kaggle-cli#api-credentials",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Is a source a Kaggle URL rather than a slug?
#'
#' A link pasted without its scheme ("www.kaggle.com/datasets/...") counts,
#' since a slug cannot contain "kaggle.com/".
#'
#' @param source The source as given.
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_is_kaggle_url <- function(source) {
  is.character(source) && length(source) == 1L && !is.na(source) &&
    (tl_has_scheme(source) ||
       grepl("^(www\\.)?kaggle\\.com/", source, ignore.case = TRUE))
}

#' Parse a Kaggle URL into a slug and a type
#'
#' The dataset is the two path segments after /datasets/, the competition
#' the one after /competitions/. Anything further is a tab of the page
#' (/data, /code, /versions/2), and a query or fragment is not part of the
#' path at all.
#'
#' @param url A Kaggle URL, with or without its scheme.
#' @return A list with \code{slug} and \code{type} (\code{"dataset"} or
#'   \code{"competition"}).
#' @keywords internal
#' @noRd
tl_parse_kaggle_url <- function(url) {
  with_scheme <- if (tl_has_scheme(url)) url else paste0("https://", url)
  if (!tl_is_kaggle_host(tl_url_host(with_scheme))) {
    stop("'", url, "' is not a Kaggle URL.", call. = FALSE)
  }

  path <- sub("[?#].*$", "",
              sub("^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]*", "", with_scheme))
  segments <- strsplit(path, "/", fixed = TRUE)[[1]]
  segments <- segments[nzchar(segments)]

  if (length(segments) >= 3L && segments[1] == "datasets") {
    return(list(slug = paste(segments[2:3], collapse = "/"),
                type = "dataset"))
  }
  if (length(segments) >= 2L && segments[1] == "competitions") {
    return(list(slug = segments[2], type = "competition"))
  }

  stop("Cannot parse Kaggle URL: '", url, "'. Expected a dataset link ",
       "such as https://www.kaggle.com/datasets/<owner>/<dataset> or a ",
       "competition link such as https://www.kaggle.com/competitions/<name>.",
       call. = FALSE)
}

#' Strip credentials out of a database connection string
#'
#' A DSN carries the password in the clear. It must never reach the
#' \code{tl_source} attribute (which \code{print.tidylearn_data()}
#' displays and \code{saveRDS()} persists), a progress message, or an
#' error string.
#'
#' @param url A connection string, or any other source description
#' @return The same string with the password in any
#'   \code{user:password@@} userinfo, \code{password=} query parameter or
#'   libpq keyword replaced by \code{***}; input with no password is
#'   returned unchanged
#' @keywords internal
#' @noRd
tl_redact_db_url <- function(url) {
  if (!is.character(url) || length(url) != 1 || is.na(url)) {
    return(url)
  }

  # scheme://user:password@rest  ->  scheme://user:***@rest. The user
  # name may be empty, the scheme is case-insensitive, and the password
  # runs to the last '@' before any query: an unencoded '@' is still part
  # of it.
  redacted <- sub(
    "^([A-Za-z][A-Za-z0-9+.-]*://)([^:@/]*):([^?#]*)@",
    "\\1\\2:***@",
    url,
    perl = TRUE
  )

  # A bare "user:password@host" with no scheme is also accepted by the
  # backends below, so cover that shape too
  redacted <- sub("^([^:@/]+):([^@/]*)@", "\\1:***@", redacted, perl = TRUE)

  # A password given as a query parameter, or as a keyword in a libpq
  # connection string made of space-separated key=value pairs
  gsub(
    "(^|[?&;[:space:]])(password|passwd|pwd)=[^&;#[:space:]]*",
    "\\1\\2=***",
    redacted,
    ignore.case = TRUE,
    perl = TRUE
  )
}

#' Parse a database connection URL
#'
#' The query string is split off first, since a parameter such as
#' \code{?ssl-mode=REQUIRED} would otherwise be read as part of the
#' database name. Credentials and the database name are percent-decoded.
#'
#' @param url A URL such as \code{postgres://user:password@@host:port/dbname}.
#' @return A list with \code{user}, \code{password}, \code{host}, \code{port}
#'   and \code{dbname} (each \code{NULL} when absent, bar the host), and
#'   \code{params}, a named list of the decoded query parameters.
#' @keywords internal
#' @noRd
tl_parse_db_url <- function(url) {
  query <- if (grepl("?", url, fixed = TRUE)) sub("^[^?]*\\?", "", url) else ""
  query <- sub("#.*$", "", query)
  main <- sub("[?#].*$", "", url)

  # mysql://user:password@host:port/dbname
  # postgres://user:password@host:port/dbname
  pattern <- paste0(
    "^[A-Za-z][A-Za-z0-9+.-]*://(?:([^:@/]*)(?::([^@]*))?@)?",
    "([^:/@]+)(?::([0-9]+))?(?:/(.+))?$"
  )
  m <- regmatches(main, regexec(pattern, main, perl = TRUE))[[1]]

  if (length(m) == 0) {
    stop("Cannot parse database URL: '", tl_redact_db_url(url), "'. ",
         "Expected scheme://user:password@host:port/dbname. ",
         "Percent-encode special characters in the user name and ",
         "password, such as '@' as %40.",
         call. = FALSE)
  }

  decoded <- function(x) if (nzchar(x)) utils::URLdecode(x) else NULL

  list(
    user = decoded(m[2]),
    password = decoded(m[3]),
    host = m[4],
    port = if (nzchar(m[5])) as.integer(m[5]) else NULL,
    dbname = decoded(m[6]),
    params = tl_parse_url_query(query)
  )
}

#' Split a URL query string into decoded parameters
#'
#' @param query The text after '?', without a fragment.
#' @return A named list of values; \code{list()} when there are none.
#' @keywords internal
#' @noRd
tl_parse_url_query <- function(query) {
  pairs <- strsplit(query, "&", fixed = TRUE)[[1]]
  pairs <- pairs[nzchar(pairs)]
  if (length(pairs) == 0L) {
    return(list())
  }

  keys <- utils::URLdecode(sub("=.*$", "", pairs))
  values <- ifelse(grepl("=", pairs, fixed = TRUE),
                   sub("^[^=]*=", "", pairs), "")
  stats::setNames(as.list(utils::URLdecode(values)), keys)
}

#' Connection arguments from a parsed URL, with named arguments as fallback
#'
#' What the URL carries wins; a named argument fills a part it leaves out,
#' which keeps a password out of the connection string.
#'
#' @param parsed The result of \code{tl_parse_db_url()}.
#' @param dbname,user,password,port The reader's named arguments.
#' @return A named list for \code{DBI::dbConnect()}, without empty entries.
#' @keywords internal
#' @noRd
tl_db_url_args <- function(parsed, dbname, user, password, port) {
  args <- list(
    host     = parsed$host,
    port     = parsed$port %||% port,
    dbname   = parsed$dbname %||% dbname,
    user     = parsed$user %||% user,
    password = parsed$password %||% password
  )
  Filter(Negate(is.null), args)
}
