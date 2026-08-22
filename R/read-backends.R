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
#' @examples
#' \donttest{
#' # conn <- DBI::dbConnect(RSQLite::SQLite(), "my_database.sqlite")
#' # data <- tl_read_db(conn, "SELECT * FROM my_table")
#' # DBI::dbDisconnect(conn)
#' }
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
#' @examples
#' \donttest{
#' # data <- tl_read_sqlite("my_database.sqlite", "SELECT * FROM my_table")
#' }
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
#'   using named parameters.
#' @param query A SQL query string.
#' @param dbname Database name (if not in \code{dsn}).
#' @param user Username (if not in \code{dsn}).
#' @param password Password (if not in \code{dsn}).
#' @param port Port number. Default is 5432.
#' @param ... Additional arguments passed to \code{DBI::dbConnect()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_postgres(
#' #   dsn = "localhost",
#' #   query = "SELECT * FROM my_table",
#' #   dbname = "mydb",
#' #   user = "myuser",
#' #   password = "mypass"
#' # )
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

  # Parse connection string if provided
  if (grepl("^postgres(ql)?://", dsn)) {
    conn <- tryCatch(
      DBI::dbConnect(
        RPostgres::Postgres(), dsn = dsn, ...
      ),
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
  source_desc <- if (grepl("^postgres", dsn)) {
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
#'   using named parameters.
#' @param query A SQL query string.
#' @param dbname Database name (if not in \code{dsn}).
#' @param user Username (if not in \code{dsn}).
#' @param password Password (if not in \code{dsn}).
#' @param port Port number. Default is 3306.
#' @param ... Additional arguments passed to \code{DBI::dbConnect()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_mysql(
#' #   dsn = "localhost",
#' #   query = "SELECT * FROM my_table",
#' #   dbname = "mydb",
#' #   user = "myuser",
#' #   password = "mypass"
#' # )
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
  if (grepl("^mysql://", dsn)) {
    parsed <- tl_parse_db_url(dsn)
    conn <- tryCatch(
      DBI::dbConnect(
        RMariaDB::MariaDB(),
        host = parsed$host,
        dbname = parsed$dbname,
        user = parsed$user,
        password = parsed$password,
        port = parsed$port %||% 3306L,
        ...
      ),
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

  source_desc <- if (grepl("^mysql", dsn)) dsn else paste0("mysql://", dsn)
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
#' @param dataset Optional default dataset for unqualified table names.
#' @param ... Additional arguments passed to
#'   \code{bigrquery::bq_project_query()}.
#'
#' @return A \code{tidylearn_data} object containing the query results.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_bigquery(
#' #   project = "my-project",
#' #   query = "SELECT * FROM `my_dataset.my_table` LIMIT 1000"
#' # )
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

  tb <- tryCatch(
    bigrquery::bq_project_query(project, query, ...),
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
#' @param format Optional format override for the downloaded file. If
#'   \code{NULL}, auto-detected from the S3 key extension.
#' @param region AWS region. If \code{NULL}, uses the default from your AWS
#'   configuration.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object containing the downloaded data.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_s3("s3://my-bucket/data/sales.csv")
#' # data <- tl_read_s3("s3://my-bucket/data/results.parquet")
#' }
#'
#' @export
tl_read_s3 <- function(source, format = NULL, region = NULL, ...) {
  tl_check_packages("paws.storage")

  # Parse s3:// URI
  s3_path <- sub("^s3://", "", source)
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
    format <- tryCatch(
      tl_detect_format(key),
      error = function(e) {
        stop("Cannot detect file format from S3 key '", key, "'. ",
             "Specify the 'format' argument.",
             call. = FALSE)
      }
    )
  }

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
#' @param source A GitHub URL or \code{"owner/repo"} string.
#' @param path Path to the file within the repository (required when
#'   \code{source} is \code{"owner/repo"} format).
#' @param ref Branch, tag, or commit SHA. Default is \code{"main"}.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object containing the downloaded data.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_github("user/repo", path = "data/file.csv")
#' # data <- tl_read_github(
#' #   "https://github.com/user/repo/blob/main/data/file.csv"
#' # )
#' }
#'
#' @export
tl_read_github <- function(source, path = NULL, ref = "main", ...) {
  # Build raw URL
  if (grepl("^https?://github\\.com", source)) {
    # Convert GitHub blob URL to raw URL
    raw_url <- sub("github\\.com", "raw.githubusercontent.com", source)
    raw_url <- sub("/blob/", "/", raw_url)
  } else if (grepl("^https?://raw\\.githubusercontent\\.com", source)) {
    raw_url <- source
  } else {
    # owner/repo shorthand
    if (is.null(path)) {
      stop("'path' is required when 'source' is in 'owner/repo' format.",
           call. = FALSE)
    }
    raw_url <- paste0(
      "https://raw.githubusercontent.com/",
      source, "/", ref, "/", path
    )
  }

  # Detect format from the URL file extension
  format <- tryCatch(
    tl_detect_format(basename(raw_url)),
    error = function(e) {
      stop("Cannot detect file format from GitHub URL. ",
           "The file must have a recognizable extension (csv, json, etc.).",
           call. = FALSE)
    }
  )

  # Download to temp file
  ext <- tools::file_ext(raw_url)
  tmp <- tempfile(fileext = paste0(".", ext))
  on.exit(unlink(tmp), add = TRUE)

  tryCatch(
    utils::download.file(raw_url, tmp, mode = "wb", quiet = TRUE),
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
#'   Kaggle URL.
#' @param file The specific file to read from the dataset. If \code{NULL} and
#'   the dataset contains exactly one file, it is read automatically.
#' @param dest Directory to download files to. The default is a fresh
#'   per-dataset directory under \code{tempdir()}; supply a path to keep
#'   the download.
#' @param type Either \code{"dataset"} (default) or \code{"competition"}.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object containing the downloaded data.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_kaggle("zillow/zecon", file = "Zip_time_series.csv")
#' # data <- tl_read_kaggle("titanic", file = "train.csv", type = "competition")
#' }
#'
#' @export
tl_read_kaggle <- function(source, file = NULL, dest = NULL,
                           type = "dataset", ...) {
  # Parse Kaggle URL to slug if needed
  if (grepl("kaggle\\.com", source)) {
    source <- tl_parse_kaggle_url(source)
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

  # Download into a directory of our own unless the caller named one.
  # Sharing tempdir() across calls meant list.files() below could match
  # a data file left by an earlier download and return it as this
  # dataset -- the wrong data, with no indication anything was amiss.
  if (is.null(dest)) {
    dest <- file.path(tempdir(), paste0("tl_kaggle_", tl_slug_key(source)))
    unlink(dest, recursive = TRUE, force = TRUE)
    dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  }

  # Download the dataset. Quote the caller-derived values: a slug is
  # already restricted to safe characters by the checks above, but a
  # destination path may legitimately contain spaces.
  if (type == "competition") {
    args <- c("competitions", "download", "-c", shQuote(source),
              "-p", shQuote(dest))
    if (!is.null(file)) args <- c(args, "-f", shQuote(file))
  } else {
    args <- c("datasets", "download", "-d", shQuote(source),
              "-p", shQuote(dest), "--unzip")
    if (!is.null(file)) args <- c(args, "-f", shQuote(file))
  }

  result <- tryCatch(
    system2("kaggle", args, stdout = TRUE, stderr = TRUE),
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
  tl_unzip_kaggle_archives(dest)

  # Find the downloaded file
  if (!is.null(file)) {
    downloaded <- file.path(dest, file)
  } else {
    # Find files in dest that match common data formats
    data_exts <- c("csv", "tsv", "json", "parquet", "xlsx", "xls")
    pattern <- paste0("\\.(", paste(data_exts, collapse = "|"), ")$")
    candidates <- list.files(
      dest, pattern = pattern, full.names = TRUE, recursive = TRUE
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

  if (!file.exists(downloaded)) {
    stop("Downloaded file not found: '", downloaded, "'.",
         call. = FALSE)
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

#' Check that the Kaggle CLI is installed
#' @keywords internal
#' @noRd
tl_check_kaggle_cli <- function() {
  result <- tryCatch(
    system2("kaggle", "--version", stdout = TRUE, stderr = TRUE),
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

#' Parse a Kaggle URL into a dataset slug
#' @keywords internal
#' @noRd
tl_parse_kaggle_url <- function(url) {
  # https://www.kaggle.com/datasets/user/dataset-name
  # https://www.kaggle.com/competitions/competition-name
  if (grepl("/datasets/", url)) {
    parts <- regmatches(url, regexpr("[^/]+/[^/]+$", url))
  } else if (grepl("/competitions/", url)) {
    parts <- regmatches(url, regexpr("[^/]+$", url))
  } else {
    stop("Cannot parse Kaggle URL: '", url, "'.", call. = FALSE)
  }
  parts
}

#' Strip credentials out of a database connection string
#'
#' A DSN carries the password in the clear. It must never reach the
#' \code{tl_source} attribute (which \code{print.tidylearn_data()}
#' displays and \code{saveRDS()} persists), a progress message, or an
#' error string.
#'
#' @param url A connection string, or any other source description
#' @return The same string with any \code{user:password@@} userinfo
#'   replaced by \code{user:***@@}; non-URL input is returned unchanged
#' @keywords internal
#' @noRd
tl_redact_db_url <- function(url) {
  if (!is.character(url) || length(url) != 1 || is.na(url)) {
    return(url)
  }

  # scheme://user:password@rest  ->  scheme://user:***@rest
  redacted <- sub(
    "^([a-z][a-z0-9+.-]*://)([^:@/]+):([^@/]*)@",
    "\\1\\2:***@",
    url,
    perl = TRUE
  )

  # A bare "user:password@host" with no scheme is also accepted by the
  # backends below, so cover that shape too
  sub("^([^:@/]+):([^@/]*)@", "\\1:***@", redacted, perl = TRUE)
}

#' Parse a database connection URL
#' @keywords internal
#' @noRd
tl_parse_db_url <- function(url) {
  # mysql://user:password@host:port/dbname
  # postgres://user:password@host:port/dbname
  pattern <- paste0(
    "^[a-z]+://(?:([^:@]+)(?::([^@]*))?@)?",
    "([^:/]+)(?::([0-9]+))?(?:/(.+))?$"
  )
  m <- regmatches(url, regexec(pattern, url))[[1]]

  if (length(m) == 0) {
    stop("Cannot parse database URL: '", tl_redact_db_url(url), "'.",
         call. = FALSE)
  }

  list(
    user = if (nzchar(m[2])) m[2] else NULL,
    password = if (nzchar(m[3])) m[3] else NULL,
    host = m[4],
    port = if (nzchar(m[5])) as.integer(m[5]) else NULL,
    dbname = if (nzchar(m[6])) m[6] else NULL
  )
}
