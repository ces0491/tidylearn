#' @title Data Reading Functions for tidylearn
#' @name tidylearn-read
#' @description Functions for reading data from diverse sources into tidy
#'   \code{tidylearn_data} objects. The main dispatcher
#'   \code{tl_read()} auto-detects the format from the file
#'   extension and routes to the appropriate reader.
#'   All readers return a \code{tidylearn_data} object,
#'   which is a tibble subclass carrying metadata about
#'   the data source.
#'
#' @details
#' Supported file formats:
#' \itemize{
#'   \item \strong{CSV}: \code{.csv} files via \pkg{readr}
#'     (with base R fallback), and \code{.txt} files named directly
#'   \item \strong{TSV}: \code{.tsv} files via \pkg{readr}
#'     (with base R fallback)
#'   \item \strong{Excel}: \code{.xls}, \code{.xlsx},
#'     \code{.xlsm} files via \pkg{readxl}
#'   \item \strong{Parquet}: \code{.parquet} files via \pkg{nanoparquet}
#'   \item \strong{JSON}: \code{.json} files, and newline-delimited
#'     \code{.ndjson} files, via \pkg{jsonlite}
#'   \item \strong{RDS}: \code{.rds} files via base \code{readRDS()}
#'   \item \strong{RData}: \code{.rdata}, \code{.rda}
#'     files via base \code{load()}
#' }
#' CSV and TSV files compressed with gzip, bzip2 or xz
#' (\code{data.csv.gz}) are recognised by the extension under the
#' compression one.
#'
#' Supported databases (via \pkg{DBI}):
#' \itemize{
#'   \item \strong{SQLite}: \code{.sqlite}, \code{.db} files via \pkg{RSQLite}
#'   \item \strong{PostgreSQL}: via \pkg{RPostgres}
#'   \item \strong{MySQL/MariaDB}: via \pkg{RMariaDB}
#'   \item \strong{BigQuery}: via \pkg{bigrquery}
#' }
#'
#' Supported cloud/API sources:
#' \itemize{
#'   \item \strong{S3}: \code{s3://} URIs via \pkg{paws.storage}
#'   \item \strong{GitHub}: raw file download from repositories
#'   \item \strong{Kaggle}: dataset download via Kaggle CLI
#' }
#' Other web URLs are not read; download the file first.
#'
#' Multi-file reading:
#' \itemize{
#'   \item \strong{Multiple paths}: pass a character vector to \code{tl_read()}
#'   \item \strong{Directories}: \code{tl_read_dir()} scans for data files with
#'     optional pattern/format filtering and recursive scanning
#'   \item \strong{Zip archives}: \code{tl_read_zip()} extracts and reads from
#'     \code{.zip} files
#' }
#' When combining multiple files, a \code{source_file} column is added to
#' identify the origin of each row: the file's path below the directory
#' or archive it came from, or, for paths given directly, below the
#' deepest folder they share. Files in one folder are labelled by their
#' bare names.
#'
#' Directory and archive scans read the extensions listed above except
#' \code{.txt}, which in a folder is as likely to hold notes as data.
#' Name a \code{.txt} file directly, or select it with \code{pattern},
#' to read it.
NULL

# ---- tidylearn_data class ----

#' Create a tidylearn_data object
#'
#' Constructor for the \code{tidylearn_data} class, a
#' tibble subclass that carries metadata about the data
#' source.
#'
#' @param data A data frame or tibble.
#' @param source Character string describing the data source (e.g., file path).
#' @param format Character string indicating the format (e.g., "csv", "excel").
#' @param timestamp POSIXct timestamp of when the data
#'   was read. Defaults to current time.
#'
#' @return A \code{tidylearn_data} object (tibble
#'   subclass with source metadata).
#' @keywords internal
#' @noRd
new_tidylearn_data <- function(data, source, format, timestamp = Sys.time()) {
  data <- tibble::as_tibble(data)
  structure(
    data,
    class = c("tidylearn_data", class(data)),
    tl_source = source,
    tl_format = format,
    tl_timestamp = timestamp
  )
}

#' Print a tidylearn_data object
#'
#' @param x A \code{tidylearn_data} object.
#' @param ... Additional arguments passed to the tibble print method.
#' @return The input object \code{x}, returned invisibly.
#' @examples
#' \donttest{
#' f <- tempfile(fileext = ".csv")
#' write.csv(iris, f, row.names = FALSE)
#' d <- tl_read(f)
#' print(d)
#' unlink(f)
#' }
#' @export
print.tidylearn_data <- function(x, ...) {
  cat("-- tidylearn data ---------\n")
  cat("Source:", attr(x, "tl_source"), "\n")
  cat("Format:", attr(x, "tl_format"), "\n")
  cat("Read at:", format(attr(x, "tl_timestamp")), "\n\n")
  NextMethod()
}

# ---- Format detection ----

# The extensions each file format is recognised by: in tl_detect_format(),
# and in the scans of tl_read_dir() and tl_read_zip(), so the two cannot
# disagree. .txt defaults to CSV; use format = "tsv" to override.
.tl_format_extensions <- list(
  csv     = c("csv", "txt"),
  tsv     = "tsv",
  excel   = c("xls", "xlsx", "xlsm"),
  rds     = "rds",
  rdata   = c("rdata", "rda"),
  parquet = "parquet",
  json    = c("json", "ndjson"),
  sqlite  = c("sqlite", "db")
)

# readr and base R both read compressed delimited files transparently, so
# data.csv.gz is detected by the extension under the compression one
.tl_compressible_formats <- c("csv", "tsv")
.tl_compression_extensions <- c("gz", "bz2", "xz")

# Hosts tl_read_github() rewrites to raw file downloads
.tl_github_hosts <- c(
  "github.com", "www.github.com", "raw.githubusercontent.com"
)

#' Does a source carry a URL scheme such as https:// or s3://?
#'
#' A source with a scheme is never a local file, so it is routed by its
#' protocol before any extension is looked at.
#'
#' @param source A single string.
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_has_scheme <- function(source) {
  grepl("^[A-Za-z][A-Za-z0-9+.-]*://", source)
}

#' Is a source an http(s) URL?
#' @param source A single string.
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_is_web_url <- function(source) {
  grepl("^https?://", source, ignore.case = TRUE)
}

#' The host a URL points at
#'
#' The host is parsed out of the URL rather than searched for in the
#' string, where "https://mirror.example.org/github.com/data.csv" would
#' pass for GitHub.
#'
#' @param url A single URL string.
#' @return The host, lower case, without user info or port; \code{""} when
#'   \code{url} has no scheme.
#' @keywords internal
#' @noRd
tl_url_host <- function(url) {
  m <- regmatches(url, regexec("^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]*)", url))
  authority <- m[[1]][2]
  if (is.na(authority)) {
    return("")
  }
  authority <- sub("^.*@", "", authority)
  tolower(sub(":[0-9]*$", "", authority))
}

#' Is a host Kaggle's?
#' @param host A host name, lower case.
#' @return A single logical.
#' @keywords internal
#' @noRd
tl_is_kaggle_host <- function(host) {
  identical(host, "kaggle.com") || endsWith(host, ".kaggle.com")
}

#' The error for a web URL tl_read() has no reader for
#' @param source The URL.
#' @keywords internal
#' @noRd
tl_stop_web_url <- function(source) {
  stop(
    "tl_read() reads web URLs only from GitHub and Kaggle, and '",
    tl_url_host(source), "' is neither. Download the file first, for ",
    "example with utils::download.file(), and read the local copy.",
    call. = FALSE
  )
}

#' Regular expression matching the files a directory or archive scan reads
#'
#' Scans leave out \code{.txt}: a folder's \code{.txt} is as likely to
#' hold notes as data, and reading notes as CSV either fails the row-bind
#' or adds junk rows. A database needs a query, so no scan reads one.
#'
#' @param formats The formats to match.
#' @return A single regular expression, to be used with
#'   \code{ignore.case = TRUE}.
#' @keywords internal
#' @noRd
tl_scan_pattern <- function(formats = c("csv", "tsv", "excel", "parquet",
                                        "json", "rds", "rdata")) {
  plain <- setdiff(unlist(.tl_format_extensions[formats]), "txt")
  compressible <- setdiff(
    unlist(.tl_format_extensions[intersect(formats, .tl_compressible_formats)]),
    "txt"
  )

  alternatives <- paste0("\\.(", paste(plain, collapse = "|"), ")$")
  if (length(compressible) > 0L) {
    alternatives <- c(alternatives, paste0(
      "\\.(", paste(compressible, collapse = "|"), ")\\.(",
      paste(.tl_compression_extensions, collapse = "|"), ")$"
    ))
  }
  paste(alternatives, collapse = "|")
}

#' Detect data format from source string
#'
#' Infers the data format from a file extension, URL pattern, or connection
#' string. Used internally by \code{tl_read()} when
#' \code{format} is not specified.
#'
#' @param source Character string: a file path, URL, or connection string.
#'
#' @return A character string indicating the detected format.
#' @keywords internal
#' @noRd
tl_detect_format <- function(source) {
  # Protocols take priority over extensions. Schemes are case-insensitive,
  # and a web URL is matched on its host.
  if (tl_is_web_url(source)) {
    host <- tl_url_host(source)
    if (host %in% .tl_github_hosts) return("github")
    if (tl_is_kaggle_host(host)) return("kaggle")
    tl_stop_web_url(source)
  }
  if (grepl("^s3://", source, ignore.case = TRUE)) return("s3")
  if (grepl("^postgres(ql)?://", source, ignore.case = TRUE)) {
    return("postgres")
  }
  if (grepl("^mysql://", source, ignore.case = TRUE)) return("mysql")

  ext <- tolower(tools::file_ext(source))
  if (ext %in% .tl_compression_extensions) {
    inner <- tolower(tools::file_ext(tools::file_path_sans_ext(source)))
    compressible <- unlist(.tl_format_extensions[.tl_compressible_formats])
    if (inner %in% compressible) ext <- inner
  }

  for (format in names(.tl_format_extensions)) {
    if (ext %in% .tl_format_extensions[[format]]) return(format)
  }

  stop("Cannot detect format from: '", tl_redact_db_url(source),
       "'. Please specify the 'format' argument.",
       call. = FALSE)
}

# ---- Main dispatcher ----

#' Read data from diverse sources
#'
#' Auto-detects the data format from the file extension or source pattern and
#' dispatches to the appropriate reader. All readers
#' return a \code{tidylearn_data} object, which is a
#' tibble subclass carrying metadata about the data
#' source.
#'
#' When \code{source} is a character vector of multiple paths, each file is read
#' and row-bound into a single result with a \code{source_file} column
#' giving each file's path below the deepest folder they share. When
#' \code{source} is a directory path, it is equivalent to calling
#' \code{tl_read_dir()}. When \code{source} is a local \code{.zip} file, it
#' is equivalent to calling \code{tl_read_zip()}.
#'
#' @param source A file path, a GitHub or Kaggle URL, an \code{s3://} URI,
#'   a database connection string, a directory path, or a character vector
#'   of multiple file paths. Other web URLs are refused: download the file
#'   and read the local copy.
#' @param ... Additional arguments passed to the format-specific reader.
#' @param format Optional explicit format override.
#'   One of \code{"csv"}, \code{"tsv"},
#'   \code{"excel"}, \code{"parquet"}, \code{"json"},
#'   \code{"rds"}, \code{"rdata"},
#'   \code{"sqlite"}, \code{"postgres"}, \code{"mysql"}, \code{"bigquery"},
#'   \code{"s3"}, \code{"github"}, \code{"kaggle"}. When \code{NULL} (default),
#'   the format is auto-detected from the file extension
#'   or source pattern. Note: \code{.txt} files default
#'   to CSV; use \code{format = "tsv"} to override.
#' @param .quiet Logical. If \code{TRUE}, suppresses
#'   informational messages. Default is \code{FALSE}.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examples
#' # The format is detected from the extension
#' csv <- tempfile(fileext = ".csv")
#' write.csv(mtcars, csv, row.names = FALSE)
#' tl_read(csv)
#'
#' # Several files are row-bound, with a source_file column naming each
#' jan <- tempfile(fileext = ".csv")
#' feb <- tempfile(fileext = ".csv")
#' write.csv(mtcars[1:16, ], jan, row.names = FALSE)
#' write.csv(mtcars[17:32, ], feb, row.names = FALSE)
#' both <- tl_read(c(jan, feb), .quiet = TRUE)
#' table(both$source_file)
#'
#' # A .txt file is read as CSV unless told otherwise
#' txt <- tempfile(fileext = ".txt")
#' write.table(mtcars, txt, sep = "\t", row.names = FALSE)
#' tl_read(txt, format = "tsv", .quiet = TRUE)
#'
#' unlink(c(csv, jan, feb, txt))
#'
#' @export
tl_read <- function(source, ..., format = NULL, .quiet = FALSE) {
  if (!is.character(source) || length(source) == 0) {
    stop("'source' must be a character string or vector of paths",
         call. = FALSE)
  }
  if (anyNA(source) || !all(nzchar(source))) {
    stop("'source' must not contain NA or empty strings.", call. = FALSE)
  }

  # Multi-path: read each and row-bind
  if (length(source) > 1) {
    return(tl_read_multi(source, ..., format = format, .quiet = .quiet))
  }

  # A source with a scheme is routed by its protocol: s3://bucket/a.zip
  # is an S3 object, not a zip file on this disk
  local <- !tl_has_scheme(source)

  # Directory: delegate to tl_read_dir
  if (local && dir.exists(source)) {
    return(tl_read_dir(source, ..., format = format, .quiet = .quiet))
  }

  # Zip file: delegate to tl_read_zip
  if (local && tolower(tools::file_ext(source)) == "zip") {
    return(tl_read_zip(source, ..., format = format, .quiet = .quiet))
  }

  if (is.null(format)) {
    format <- tl_detect_format(source)
  }

  # With a file format named, a web URL would reach a local-file reader
  # and be reported as a file that does not exist
  if (tl_is_web_url(source) && !format %in% c("github", "kaggle")) {
    tl_stop_web_url(source)
  }

  supported_formats <- c(
    "csv", "tsv", "excel", "parquet", "json", "rds", "rdata",
    "sqlite", "postgres", "mysql", "bigquery",
    "s3", "github", "kaggle"
  )

  if (!format %in% supported_formats) {
    stop("Unsupported format: '", format, "'.",
         "\nSupported formats: ", paste(supported_formats, collapse = ", "),
         call. = FALSE)
  }

  # A database source is a DSN carrying a password in the clear, so the
  # progress line reports the redacted form
  if (!.quiet) {
    message("Reading ", format, " data from: ", tl_redact_db_url(source))
  }

  result <- switch(format,
    "csv"      = tl_read_csv(source, ...),
    "tsv"      = tl_read_tsv(source, ...),
    "excel"    = tl_read_excel(source, ...),
    "parquet"  = tl_read_parquet(source, ...),
    "json"     = tl_read_json(source, ...),
    "rds"      = tl_read_rds(source),
    "rdata"    = tl_read_rdata(source, ...),
    "sqlite"   = tl_read_sqlite(source, ...),
    "postgres" = tl_read_postgres(source, ...),
    "mysql"    = tl_read_mysql(source, ...),
    "bigquery" = tl_read_bigquery(source, ...),
    "s3"       = tl_read_s3(source, ...),
    "github"   = tl_read_github(source, ...),
    "kaggle"   = tl_read_kaggle(source, ...)
  )

  if (!.quiet) {
    message("Returned: ", nrow(result), " rows x ", ncol(result), " columns")
  }

  result
}

# ---- Multi-file reading ----

#' Read multiple files and row-bind
#'
#' Internal helper that reads multiple file paths and combines them into a
#' single \code{tidylearn_data} object with a \code{source_file} column
#' identifying the origin of each row.
#'
#' @param paths Character vector of file paths.
#' @param ... Additional arguments passed to each reader.
#' @param format Optional format override applied to all files.
#' @param .quiet Suppress messages.
#' @param labels The label for each path. \code{NULL} labels each path
#'   relative to the deepest folder they all share.
#'
#' @return A \code{tidylearn_data} object with a \code{source_file} column.
#' @keywords internal
#' @noRd
tl_read_multi <- function(paths, ..., format = NULL, .quiet = FALSE,
                          labels = NULL) {
  if (!.quiet) {
    message("Reading ", length(paths), " files...")
  }

  if (is.null(labels)) {
    labels <- tl_relative_labels(paths)
  }

  results <- lapply(paths, function(p) {
    tl_read(p, ..., format = format, .quiet = TRUE)
  })

  # One column for every file's label, chosen once. Chosen per file, the
  # labels of a file holding its own source_file column would go to
  # tl_source_file and the rest into that file's column, among its data.
  col <- "source_file"
  if (any(vapply(results, function(df) col %in% names(df), logical(1)))) {
    col <- "tl_source_file"
    warning(
      "Column 'source_file' already exists in the data. ",
      "Using 'tl_source_file' for the origin of each row instead.",
      call. = FALSE
    )
  }
  for (i in seq_along(results)) {
    results[[i]][[col]] <- rep(labels[[i]], nrow(results[[i]]))
  }

  combined <- dplyr::bind_rows(results)

  if (!.quiet) {
    message("Combined: ", nrow(combined), " rows x ", ncol(combined),
            " columns from ", length(paths), " files")
  }

  new_tidylearn_data(
    combined,
    source = paste0(length(paths), " files"),
    format = if (!is.null(format)) format else "multi"
  )
}

#' Label paths read together by their part below a shared folder
#'
#' Base names alone would label 2023/sales.csv and 2024/sales.csv alike.
#' Each path is shown relative to the deepest folder all of them share, so
#' files in one folder keep their bare names.
#'
#' @param paths Character vector of paths or URIs.
#' @return A character vector the length of \code{paths}.
#' @keywords internal
#' @noRd
tl_relative_labels <- function(paths) {
  local <- !tl_has_scheme(paths)
  full <- paths
  full[local] <- normalizePath(paths[local], winslash = "/", mustWork = FALSE)

  parts <- strsplit(full, "/", fixed = TRUE)
  folders <- lapply(parts, function(p) p[-length(p)])

  shared <- 0L
  repeat {
    k <- shared + 1L
    if (any(lengths(folders) < k)) break
    if (length(unique(vapply(folders, `[`, character(1), k))) != 1L) break
    shared <- k
  }

  vapply(parts, function(p) {
    paste(p[-seq_len(shared)], collapse = "/")
  }, character(1))
}

#' Read all matching files from a directory
#'
#' Scans a directory for files matching a pattern or format, reads each one,
#' and row-binds them into a single \code{tidylearn_data} object with a
#' \code{source_file} column identifying the origin of each row.
#'
#' @param path Path to a directory.
#' @param pattern Optional regex pattern to filter file names (e.g.,
#'   \code{"sales_.*\\\\.csv$"}). If \code{NULL}, files are filtered by
#'   \code{format} instead.
#' @param format File format to read. If \code{NULL} and \code{pattern} is
#'   \code{NULL}, all recognized data files are read. If specified, only files
#'   with matching extensions are read. \code{.txt} files are read only
#'   when selected with \code{pattern}.
#' @param recursive Logical. Should subdirectories be scanned? Default is
#'   \code{FALSE}.
#' @param .quiet Suppress messages. Default is \code{FALSE}.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object with an additional
#'   \code{source_file} column giving each row's file as a path below
#'   \code{path}, such as \code{"2024/sales.csv"}.
#'
#' @examples
#' dir <- tempfile("sales_")
#' dir.create(file.path(dir, "2024"), recursive = TRUE)
#' write.csv(mtcars[1:16, ], file.path(dir, "jan.csv"), row.names = FALSE)
#' write.csv(mtcars[17:32, ], file.path(dir, "2024", "feb.csv"),
#'           row.names = FALSE)
#'
#' # Only the top level unless asked to recurse
#' tl_read_dir(dir, format = "csv")
#'
#' # Files in subfolders are labelled by their path below dir
#' all_months <- tl_read_dir(dir, recursive = TRUE, .quiet = TRUE)
#' table(all_months$source_file)
#'
#' # Or select files by a regular expression
#' tl_read_dir(dir, pattern = "^jan", .quiet = TRUE)
#'
#' unlink(dir, recursive = TRUE)
#'
#' @export
tl_read_dir <- function(path, pattern = NULL, format = NULL,
                        recursive = FALSE, .quiet = FALSE, ...) {
  if (!dir.exists(path)) {
    stop("Directory not found: '", path, "'", call. = FALSE)
  }

  # Build file list, relative to path: the relative path is each row's
  # source_file label
  if (!is.null(pattern)) {
    found <- list.files(path, pattern = pattern, recursive = recursive)
  } else if (!is.null(format)) {
    scannable <- c("csv", "tsv", "excel", "parquet", "json", "rds", "rdata")
    if (!format %in% scannable) {
      stop("Cannot scan directory for format '", format, "'. ",
           "Use 'pattern' argument instead.",
           call. = FALSE)
    }
    found <- list.files(path, pattern = tl_scan_pattern(format),
                        recursive = recursive, ignore.case = TRUE)
  } else {
    # All recognized data file extensions
    found <- list.files(path, pattern = tl_scan_pattern(),
                        recursive = recursive, ignore.case = TRUE)
  }
  # Without recursion list.files() returns folders too, and a folder named
  # like a data file (old.csv) would be read as a directory into the result
  found <- found[!dir.exists(file.path(path, found))]
  files <- file.path(path, found)

  if (length(files) == 0) {
    stop("No data files found in '", path, "'.",
         if (!is.null(pattern)) paste0("\nPattern: ", pattern),
         if (!is.null(format)) paste0("\nFormat: ", format),
         call. = FALSE)
  }

  if (!.quiet) {
    message("Found ", length(files), " file(s) in ", path)
  }

  tl_read_multi(files, ..., format = format, .quiet = .quiet, labels = found)
}

#' Read data from a zip archive
#'
#' Extracts a zip archive to a temporary directory and reads the contents.
#' If the archive contains a single data file, it is read directly. If
#' multiple data files are found, they are row-bound with a \code{source_file}
#' column. Use the \code{file} argument to select a specific file from
#' the archive.
#'
#' An archive with a member whose name could reach outside the extraction
#' directory -- an absolute path, a drive letter on Windows, or any
#' \code{..} component -- is refused before anything is extracted.
#'
#' @param path Path to a zip file.
#' @param file Optional name of a specific file within the archive to read:
#'   its path within the archive (\code{"2024/sales.csv"}), its file name,
#'   or part of its path. An exact path wins, then an exact file name, then
#'   a partial match; a name that matches more than one member at the
#'   first of those steps that matches anything is an error listing them.
#' @param format Optional format override for the file(s) inside the
#'   archive. Without \code{file}, this selects the members of that
#'   format rather than forcing it onto them, whether the archive holds
#'   one data file or several. With \code{file}, the named member is read
#'   as \code{format} whatever its extension.
#' @param .quiet Suppress messages. Default is \code{FALSE}.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}. The archive is extracted to a temporary directory
#'   that is cleaned up automatically. If multiple data files are found,
#'   a \code{source_file} column gives each row's member as its path within
#'   the archive.
#'
#' @examplesIf requireNamespace("readr", quietly = TRUE)
#' # readr ships a zip archive holding one CSV
#' archive <- readr::readr_example("mtcars.csv.zip")
#' tl_read_zip(archive)
#'
#' # Name a member to read just that one
#' tl_read_zip(archive, file = "mtcars.csv", .quiet = TRUE)
#'
#' @export
tl_read_zip <- function(path, file = NULL, format = NULL,
                        .quiet = FALSE, ...) {
  tl_validate_file_path(path)

  # Extract to temp directory
  dest <- tempfile(pattern = "tl_zip_")
  dir.create(dest)
  on.exit(unlink(dest, recursive = TRUE), add = TRUE)

  tl_unzip_checked(path, dest)
  members <- list.files(dest, recursive = TRUE)

  if (!is.null(file)) {
    target <- tl_match_zip_member(file, members)
    result <- tl_read(file.path(dest, target), ..., format = format,
                      .quiet = .quiet)
    attr(result, "tl_source") <- paste0(path, "//", target)
    attr(result, "tl_format") <- paste0("zip+", attr(result, "tl_format"))
    return(result)
  }

  # No specific file — read all data files
  data_files <- members[grepl(tl_scan_pattern(), members, ignore.case = TRUE)]

  if (length(data_files) == 0) {
    stop("No recognized data files found in archive.",
         "\nFiles in archive: ", paste(members, collapse = ", "),
         call. = FALSE)
  }

  # An archive can hold more than one kind of file. Forcing `format` onto
  # every member read a JSON as a CSV and row-bound the result, producing
  # a column named after the JSON's first line and no error at all. When
  # the caller names a format, take it to mean the members of that format
  # -- one member or many, so a single JSON is not read as a CSV either.
  if (!is.null(format)) {
    detected <- vapply(data_files, tl_detect_format, character(1),
                       USE.NAMES = FALSE)
    wanted <- detected == format

    if (!any(wanted)) {
      stop("No ", format, " files in archive.",
           "\nFound: ", paste(unique(detected), collapse = ", "),
           "\nTo read a member as ", format, " whatever its extension, ",
           "name it with 'file'.",
           call. = FALSE)
    }

    if (!all(wanted) && !.quiet) {
      message("Reading the ", sum(wanted), " ", format, " file(s); ",
              "ignoring ", paste(unique(detected[!wanted]), collapse = ", "),
              ".")
    }

    data_files <- data_files[wanted]
  }

  if (length(data_files) == 1) {
    result <- tl_read(file.path(dest, data_files), ..., format = format,
                      .quiet = .quiet)
    attr(result, "tl_source") <- paste0(path, "//", data_files)
    attr(result, "tl_format") <- paste0("zip+", attr(result, "tl_format"))
    return(result)
  }

  # Multiple files — row-bind
  if (!.quiet) {
    message("Found ", length(data_files), " data file(s) in ", basename(path))
  }

  result <- tl_read_multi(file.path(dest, data_files), ..., format = format,
                          .quiet = .quiet, labels = data_files)
  attr(result, "tl_source") <- path
  attr(result, "tl_format") <- "zip+multi"
  result
}

#' Pick the archive member a caller named
#'
#' An exact path wins, then an exact file name, then a part of the path.
#' Several matches at the step that decides are an error. Settled by file
#' order, "train" would read "full_train.csv" where the archive also holds
#' "train.csv", without a word under \code{.quiet = TRUE}.
#'
#' @param file The name the caller gave.
#' @param members Member paths, relative to the extraction directory.
#' @return The chosen member path.
#' @keywords internal
#' @noRd
tl_match_zip_member <- function(file, members) {
  wanted <- sub("^\\./", "", gsub("\\\\", "/", file))
  steps <- list(
    members == wanted,
    basename(members) == wanted,
    grepl(wanted, members, fixed = TRUE)
  )

  for (hit in steps) {
    if (sum(hit) == 1L) {
      return(members[hit])
    }
    if (sum(hit) > 1L) {
      stop("'", file, "' matches ", sum(hit), " files in the archive: ",
           paste(members[hit], collapse = ", "),
           ". Give the path of one of them within the archive.",
           call. = FALSE)
    }
  }

  stop("File '", file, "' not found in archive.",
       "\nAvailable files: ", paste(members, collapse = ", "),
       call. = FALSE)
}

#' Refuse a zip archive with members that would land outside the
#' directory it is unpacked into
#'
#' \code{unzip()} before R 4.5.1 extracts a member named \code{"../x"} or
#' \code{"/x"} as written, so a crafted archive could plant a file
#' anywhere the user can write -- an \code{.Rprofile}, say, which runs at
#' the next R start. Any \code{..} component is refused, including one
#' that resolves inside, so the check needs no path arithmetic. Names are
#' split on both separators, since archives made on Windows use either and
#' some extractors treat a backslash as one. A drive letter
#' (\code{"C:x"}) is refused only on Windows: elsewhere a colon is an
#' ordinary character in a file name.
#'
#' @param path The archive, named in the message.
#' @param members Member names, as listed by \code{unzip(list = TRUE)}.
#' @return \code{TRUE}, invisibly, when every member is safe.
#' @keywords internal
#' @noRd
tl_refuse_unsafe_zip <- function(path, members) {
  reason <- rep(NA_character_, length(members))

  climbing <- vapply(
    strsplit(members, "[/\\\\]"),
    function(parts) any(parts == ".."),
    logical(1)
  )
  reason[climbing] <- "has a '..' component"
  reason[grepl("^[/\\\\]", members)] <- "is an absolute path"
  if (.Platform$OS.type == "windows") {
    reason[grepl("^[A-Za-z]:", members)] <- "names a drive"
  }

  unsafe <- !is.na(reason)
  if (any(unsafe)) {
    stop(
      "Refusing to unpack '", basename(path), "': ",
      paste0("'", members[unsafe], "' ", reason[unsafe], collapse = "; "),
      ". tidylearn does not extract a member that is an absolute path, ",
      "names a drive or has a '..' component, since such a name can reach ",
      "outside the folder an archive is unpacked into. Extract it by hand ",
      "only if you trust where it came from.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Unzip an archive once its member names are known to be safe
#'
#' @param path The archive.
#' @param exdir The directory to extract into.
#' @return The extracted paths, invisibly.
#' @keywords internal
#' @noRd
tl_unzip_checked <- function(path, exdir) {
  members <- tryCatch(
    utils::unzip(path, list = TRUE)$Name,
    error = function(e) {
      stop("Cannot read zip archive '", path, "': ", conditionMessage(e),
           call. = FALSE)
    }
  )
  tl_refuse_unsafe_zip(path, members)
  invisible(utils::unzip(path, exdir = exdir))
}

# ---- File format readers ----

#' Read a CSV file
#'
#' Reads a CSV file into a \code{tidylearn_data} object. Uses \pkg{readr} when
#' available for faster parsing, with a base R fallback.
#'
#' @param path Path to a CSV file.
#' @param ... Additional arguments passed to \code{readr::read_csv()} or
#'   \code{utils::read.csv()}.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examples
#' path <- tempfile(fileext = ".csv")
#' write.csv(mtcars, path, row.names = FALSE)
#' tl_read_csv(path)
#' unlink(path)
#'
#' @export
tl_read_csv <- function(path, ...) {
  tl_validate_file_path(path)

  if (requireNamespace("readr", quietly = TRUE)) {
    data <- readr::read_csv(path, show_col_types = FALSE, ...)
  } else {
    message("Install 'readr' for faster CSV reading. Using base R.")
    data <- utils::read.csv(path, stringsAsFactors = FALSE, ...)
    data <- tibble::as_tibble(data)
  }

  new_tidylearn_data(data, source = path, format = "csv")
}

#' Read a TSV file
#'
#' Reads a tab-separated file into a
#' \code{tidylearn_data} object. Uses \pkg{readr} when
#' available for faster parsing, with a base R fallback.
#'
#' @param path Path to a TSV file.
#' @param ... Additional arguments passed to \code{readr::read_tsv()} or
#'   \code{utils::read.delim()}.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examples
#' path <- tempfile(fileext = ".tsv")
#' write.table(mtcars, path, sep = "\t", row.names = FALSE)
#' tl_read_tsv(path)
#' unlink(path)
#'
#' @export
tl_read_tsv <- function(path, ...) {
  tl_validate_file_path(path)

  if (requireNamespace("readr", quietly = TRUE)) {
    data <- readr::read_tsv(path, show_col_types = FALSE, ...)
  } else {
    message("Install 'readr' for faster TSV reading. Using base R.")
    data <- utils::read.delim(path, stringsAsFactors = FALSE, ...)
    data <- tibble::as_tibble(data)
  }

  new_tidylearn_data(data, source = path, format = "tsv")
}

#' Read an Excel file
#'
#' Reads an Excel file (\code{.xls}, \code{.xlsx}, or \code{.xlsm}) into a
#' \code{tidylearn_data} object. Requires the \pkg{readxl} package.
#'
#' @param path Path to an Excel file.
#' @param sheet Sheet to read. Either a string (the name of a sheet) or an
#'   integer (the position of the sheet). Defaults to the first sheet.
#' @param ... Additional arguments passed to \code{readxl::read_excel()}.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examplesIf requireNamespace("readxl", quietly = TRUE)
#' # readxl ships a workbook with one data set per sheet
#' path <- readxl::readxl_example("datasets.xlsx")
#' tl_read_excel(path)
#' tl_read_excel(path, sheet = "mtcars")
#'
#' @export
tl_read_excel <- function(path, sheet = 1, ...) {
  tl_validate_file_path(path)
  tl_check_packages("readxl")

  data <- readxl::read_excel(path, sheet = sheet, ...)

  new_tidylearn_data(data, source = path, format = "excel")
}

#' Read an RDS file
#'
#' Reads an RDS file into a \code{tidylearn_data} object. Uses base R
#' \code{readRDS()} — no additional packages required.
#'
#' @section Reading files you did not create:
#' \code{readRDS()} rebuilds whatever R objects the file describes, so
#' read only files from a source you trust. On R before 4.4.0 a crafted
#' file can run code as it is read (CVE-2024-27322). The remote readers
#' \code{tl_read_github()} and \code{tl_read_s3()} refuse \code{.rds}
#' and \code{.rdata} files on those versions unless
#' \code{trust_rds = TRUE}.
#'
#' @param path Path to an RDS file.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examples
#' path <- tempfile(fileext = ".rds")
#' saveRDS(mtcars, path)
#' tl_read_rds(path)
#' unlink(path)
#'
#' @export
tl_read_rds <- function(path) {
  tl_validate_file_path(path)

  data <- readRDS(path)

  if (!is.data.frame(data)) {
    stop("The RDS file does not contain a data frame. ",
         "tl_read_rds() expects tabular data.",
         call. = FALSE)
  }

  new_tidylearn_data(data, source = path, format = "rds")
}

#' Read an RData file
#'
#' Reads an RData (\code{.rdata} or \code{.rda}) file
#' into a \code{tidylearn_data} object. Since RData files
#' can contain multiple objects, use the \code{name}
#' argument to specify which object to extract.
#' If \code{name} is \code{NULL} and
#' the file contains exactly one data frame, it is returned automatically.
#'
#' @section Reading files you did not create:
#' \code{load()} rebuilds whatever R objects the file describes, so read
#' only files from a source you trust. On R before 4.4.0 a crafted file
#' can run code as it is read (CVE-2024-27322). The remote readers
#' \code{tl_read_github()} and \code{tl_read_s3()} refuse \code{.rds}
#' and \code{.rdata} files on those versions unless
#' \code{trust_rds = TRUE}.
#'
#' @param path Path to an RData file.
#' @param name Optional name of the object to extract from the RData file. If
#'   \code{NULL} (default), the function returns the first data frame found, or
#'   errors if there are multiple data frames.
#' @param ... Currently unused.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examples
#' path <- tempfile(fileext = ".rdata")
#' cars <- mtcars
#' flowers <- iris
#' save(cars, flowers, file = path)
#'
#' # With more than one data frame in the file, name the one to read
#' tl_read_rdata(path, name = "flowers")
#' unlink(path)
#'
#' @export
tl_read_rdata <- function(path, name = NULL, ...) {
  tl_validate_file_path(path)

  env <- new.env(parent = emptyenv())
  load(path, envir = env)

  objects <- ls(envir = env)

  if (length(objects) == 0) {
    stop("The RData file is empty.", call. = FALSE)
  }

  if (!is.null(name)) {
    if (!name %in% objects) {
      stop("Object '", name, "' not found in the RData file.",
           "\nAvailable objects: ", paste(objects, collapse = ", "),
           call. = FALSE)
    }
    data <- get(name, envir = env)
  } else {
    # Find data frames
    df_objects <- objects[vapply(objects, function(nm) {
      is.data.frame(get(nm, envir = env))
    }, logical(1))]

    if (length(df_objects) == 0) {
      stop("No data frames found in the RData file.",
           "\nAvailable objects: ", paste(objects, collapse = ", "),
           call. = FALSE)
    }

    if (length(df_objects) > 1) {
      stop("Multiple data frames found in the RData file: ",
           paste(df_objects, collapse = ", "),
           "\nPlease specify which one to load with the 'name' argument.",
           call. = FALSE)
    }

    data <- get(df_objects[[1]], envir = env)
  }

  if (!is.data.frame(data)) {
    stop("The selected object is not a data frame. ",
         "tl_read_rdata() expects tabular data.",
         call. = FALSE)
  }

  new_tidylearn_data(data, source = path, format = "rdata")
}

#' Read a Parquet file
#'
#' Reads a Parquet file into a \code{tidylearn_data} object. Requires the
#' \pkg{nanoparquet} package.
#'
#' @param path Path to a Parquet file.
#' @param ... Additional arguments passed to \code{nanoparquet::read_parquet()}.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examplesIf requireNamespace("nanoparquet", quietly = TRUE)
#' path <- tempfile(fileext = ".parquet")
#' nanoparquet::write_parquet(mtcars, path)
#' tl_read_parquet(path)
#' unlink(path)
#'
#' @export
tl_read_parquet <- function(path, ...) {
  tl_validate_file_path(path)
  tl_check_packages("nanoparquet")

  data <- nanoparquet::read_parquet(path, ...)

  new_tidylearn_data(data, source = path, format = "parquet")
}

#' Read a JSON file
#'
#' Reads a JSON file into a \code{tidylearn_data} object. Expects the JSON to
#' represent tabular data (array of objects or similar). A file with the
#' \code{.ndjson} extension is read as newline-delimited JSON, one record
#' per line, and so is a \code{.json} file that does not parse as a single
#' document but does as one record per line. Requires the \pkg{jsonlite}
#' package.
#'
#' @param path Path to a JSON file.
#' @param flatten Logical. Automatically flatten nested data frames? Default is
#'   \code{TRUE}.
#' @param ... Additional arguments passed to \code{jsonlite::fromJSON()}, or
#'   to \code{jsonlite::stream_in()} for an \code{.ndjson} file.
#'
#' @return A \code{tidylearn_data} object (a \link[tibble]{tibble} subclass)
#'   with attributes \code{tl_source}, \code{tl_format}, and
#'   \code{tl_timestamp}.
#'
#' @examplesIf requireNamespace("jsonlite", quietly = TRUE)
#' path <- tempfile(fileext = ".json")
#' jsonlite::write_json(mtcars, path)
#' tl_read_json(path)
#'
#' # Newline-delimited JSON: one record per line
#' lines <- tempfile(fileext = ".ndjson")
#' jsonlite::stream_out(mtcars, file(lines), verbose = FALSE)
#' tl_read_json(lines)
#'
#' unlink(c(path, lines))
#'
#' @export
tl_read_json <- function(path, flatten = TRUE, ...) {
  tl_validate_file_path(path)
  tl_check_packages("jsonlite")

  # fromJSON() reads a file as one document, and stops newline-delimited
  # JSON at the end of its first record ("trailing garbage")
  read_lines <- function() {
    lines <- jsonlite::stream_in(file(path), verbose = FALSE, ...)
    if (isTRUE(flatten)) jsonlite::flatten(lines) else lines
  }

  if (tolower(tools::file_ext(path)) == "ndjson") {
    data <- read_lines()
  } else {
    # The same records are often saved under a .json name, so a file that
    # fails as one document is tried line by line before giving up
    data <- tryCatch(
      jsonlite::fromJSON(path, flatten = flatten, ...),
      error = function(e) {
        tryCatch(read_lines(), error = function(e_lines) {
          stop("Cannot read '", path, "' as JSON or as newline-delimited ",
               "JSON: ", conditionMessage(e), call. = FALSE)
        })
      }
    )
  }

  if (!is.data.frame(data)) {
    stop("The JSON file does not contain tabular data. ",
         "tl_read_json() expects an array of objects.",
         call. = FALSE)
  }

  new_tidylearn_data(data, source = path, format = "json")
}
