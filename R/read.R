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
#'     (with base R fallback)
#'   \item \strong{TSV}: \code{.tsv} files via \pkg{readr}
#'     (with base R fallback)
#'   \item \strong{Excel}: \code{.xls}, \code{.xlsx},
#'     \code{.xlsm} files via \pkg{readxl}
#'   \item \strong{Parquet}: \code{.parquet} files via \pkg{nanoparquet}
#'   \item \strong{JSON}: \code{.json} files via \pkg{jsonlite}
#'   \item \strong{RDS}: \code{.rds} files via base \code{readRDS()}
#'   \item \strong{RData}: \code{.rdata}, \code{.rda}
#'     files via base \code{load()}
#' }
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
#' identify the origin of each row.
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
#'
#' @export
print.tidylearn_data <- function(x, ...) {
  cat("-- tidylearn data ---------\n")
  cat("Source:", attr(x, "tl_source"), "\n")
  cat("Format:", attr(x, "tl_format"), "\n")
  cat("Read at:", format(attr(x, "tl_timestamp")), "\n\n")
  NextMethod()
}

# ---- Format detection ----

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
  # Check URL/protocol patterns first (these take priority over extensions)
  if (grepl("^https?://", source)) {
    github_pat <- "github\\.com|raw\\.githubusercontent\\.com"
    if (grepl(github_pat, source)) return("github")
    if (grepl("kaggle\\.com", source)) return("kaggle")
  }
  if (grepl("^s3://", source)) return("s3")
  if (grepl("^postgres(ql)?://", source)) return("postgres")
  if (grepl("^mysql://", source)) return("mysql")


  # Fall back to file extension matching
  # Note: .txt defaults to CSV; use format = "tsv" to override
  ext <- tolower(tools::file_ext(source))

  ext_map <- c(
    csv = "csv", tsv = "tsv", txt = "csv",
    xls = "excel", xlsx = "excel", xlsm = "excel",
    rds = "rds", rdata = "rdata", rda = "rdata",
    parquet = "parquet",
    json = "json", ndjson = "json",
    sqlite = "sqlite", db = "sqlite"
  )

  if (ext %in% names(ext_map)) return(ext_map[[ext]])

  stop("Cannot detect format from: '", source,
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
#' and row-bound into a single result with a \code{source_file} column. When
#' \code{source} is a directory path, it is equivalent to calling
#' \code{tl_read_dir()}. When \code{source} is a \code{.zip} file, it is
#' equivalent to calling \code{tl_read_zip()}.
#'
#' @param source A file path, URL, connection string, directory path, or a
#'   character vector of multiple file paths.
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
#' @return A \code{tidylearn_data} object (tibble
#'   subclass with source metadata).
#'
#' @examples
#' \donttest{
#' # Read a single CSV file
#' # data <- tl_read("path/to/data.csv")
#'
#' # Read multiple files and row-bind
#' # data <- tl_read(c("jan.csv", "feb.csv", "mar.csv"))
#'
#' # Read all CSVs from a directory
#' # data <- tl_read("data/")
#'
#' # Read from a zip archive
#' # data <- tl_read("data.zip")
#'
#' # Explicit format override
#' # data <- tl_read("path/to/data.txt", format = "tsv")
#' }
#'
#' @export
tl_read <- function(source, ..., format = NULL, .quiet = FALSE) {
  if (!is.character(source) || length(source) == 0) {
    stop("'source' must be a character string or vector of paths",
         call. = FALSE)
  }

  # Multi-path: read each and row-bind
  if (length(source) > 1) {
    return(tl_read_multi(source, ..., format = format, .quiet = .quiet))
  }

  # Directory: delegate to tl_read_dir
  if (dir.exists(source)) {
    return(tl_read_dir(source, ..., format = format, .quiet = .quiet))
  }

  # Zip file: delegate to tl_read_zip
  if (tolower(tools::file_ext(source)) == "zip") {
    return(tl_read_zip(source, ..., format = format, .quiet = .quiet))
  }

  if (is.null(format)) {
    format <- tl_detect_format(source)
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

  if (!.quiet) message("Reading ", format, " data from: ", source)

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
#'
#' @return A \code{tidylearn_data} object with a \code{source_file} column.
#' @keywords internal
#' @noRd
tl_read_multi <- function(paths, ..., format = NULL, .quiet = FALSE) {
  if (!.quiet) {
    message("Reading ", length(paths), " files...")
  }

  results <- lapply(paths, function(p) {
    df <- tl_read(p, ..., format = format, .quiet = TRUE)
    col <- "source_file"
    if (col %in% names(df)) {
      col <- "tl_source_file"
      warning(
        "Column 'source_file' already exists in data. ",
        "Using 'tl_source_file' instead.",
        call. = FALSE
      )
    }
    df[[col]] <- basename(p)
    df
  })

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
#'   with matching extensions are read.
#' @param recursive Logical. Should subdirectories be scanned? Default is
#'   \code{FALSE}.
#' @param .quiet Suppress messages. Default is \code{FALSE}.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object with a \code{source_file} column.
#'
#' @examples
#' \donttest{
#' # Read all CSVs from a directory
#' # data <- tl_read_dir("data/", format = "csv")
#'
#' # Read with pattern matching
#' # data <- tl_read_dir("data/", pattern = "^sales_.*\\.csv$")
#'
#' # Read all recognized data files recursively
#' # data <- tl_read_dir("data/", recursive = TRUE)
#' }
#'
#' @export
tl_read_dir <- function(path, pattern = NULL, format = NULL,
                        recursive = FALSE, .quiet = FALSE, ...) {
  if (!dir.exists(path)) {
    stop("Directory not found: '", path, "'", call. = FALSE)
  }

  # Build file list
  if (!is.null(pattern)) {
    files <- list.files(path, pattern = pattern, full.names = TRUE,
                        recursive = recursive)
  } else if (!is.null(format)) {
    ext_patterns <- list(
      csv = "\\.csv$", tsv = "\\.tsv$",
      excel = "\\.(xlsx?|xlsm)$",
      parquet = "\\.parquet$", json = "\\.json$",
      rds = "\\.rds$", rdata = "\\.(rdata|rda)$"
    )
    pat <- ext_patterns[[format]]
    if (is.null(pat)) {
      stop("Cannot scan directory for format '", format, "'. ",
           "Use 'pattern' argument instead.",
           call. = FALSE)
    }
    files <- list.files(path, pattern = pat, full.names = TRUE,
                        recursive = recursive, ignore.case = TRUE)
  } else {
    # All recognized data file extensions
    pat <- "\\.(csv|tsv|xlsx?|xlsm|parquet|json|rds|rdata|rda)$"
    files <- list.files(path, pattern = pat, full.names = TRUE,
                        recursive = recursive, ignore.case = TRUE)
  }

  if (length(files) == 0) {
    stop("No data files found in '", path, "'.",
         if (!is.null(pattern)) paste0("\nPattern: ", pattern),
         if (!is.null(format)) paste0("\nFormat: ", format),
         call. = FALSE)
  }

  if (!.quiet) {
    message("Found ", length(files), " file(s) in ", path)
  }

  tl_read_multi(files, ..., format = format, .quiet = .quiet)
}

#' Read data from a zip archive
#'
#' Extracts a zip archive to a temporary directory and reads the contents.
#' If the archive contains a single data file, it is read directly. If
#' multiple data files are found, they are row-bound with a \code{source_file}
#' column. Use the \code{file} argument to select a specific file from
#' the archive.
#'
#' @param path Path to a zip file.
#' @param file Optional name of a specific file within the archive to read.
#'   Supports partial matching.
#' @param format Optional format override for the file(s) inside the archive.
#' @param .quiet Suppress messages. Default is \code{FALSE}.
#' @param ... Additional arguments passed to the format-specific reader.
#'
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # Read from a zip archive
#' # data <- tl_read_zip("data.zip")
#'
#' # Read a specific file from the archive
#' # data <- tl_read_zip("data.zip", file = "train.csv")
#' }
#'
#' @export
tl_read_zip <- function(path, file = NULL, format = NULL,
                        .quiet = FALSE, ...) {
  tl_validate_file_path(path)

  # Extract to temp directory
  dest <- tempfile(pattern = "tl_zip_")
  dir.create(dest)
  on.exit(unlink(dest, recursive = TRUE), add = TRUE)

  utils::unzip(path, exdir = dest)

  if (!is.null(file)) {
    # Find the specific file (supports partial matching)
    all_files <- list.files(dest, recursive = TRUE, full.names = TRUE)
    matches <- all_files[grepl(file, basename(all_files), fixed = TRUE)]

    if (length(matches) == 0) {
      available <- basename(all_files)
      stop("File '", file, "' not found in archive.",
           "\nAvailable files: ", paste(available, collapse = ", "),
           call. = FALSE)
    }

    target <- matches[1]
    if (length(matches) > 1 && !.quiet) {
      message("Multiple matches for '", file, "'. Reading: ",
              basename(target))
    }

    result <- tl_read(target, ..., format = format, .quiet = .quiet)
    attr(result, "tl_source") <- paste0(path, "//", basename(target))
    attr(result, "tl_format") <- paste0("zip+", attr(result, "tl_format"))
    return(result)
  }

  # No specific file — read all data files
  data_pattern <- "\\.(csv|tsv|xlsx?|xlsm|parquet|json|rds|rdata|rda)$"
  data_files <- list.files(dest, pattern = data_pattern,
                           full.names = TRUE, recursive = TRUE,
                           ignore.case = TRUE)

  if (length(data_files) == 0) {
    all_files <- list.files(dest, recursive = TRUE)
    stop("No recognized data files found in archive.",
         "\nFiles in archive: ", paste(all_files, collapse = ", "),
         call. = FALSE)
  }

  if (length(data_files) == 1) {
    result <- tl_read(data_files[1], ..., format = format, .quiet = .quiet)
    attr(result, "tl_source") <- paste0(path, "//", basename(data_files[1]))
    attr(result, "tl_format") <- paste0("zip+", attr(result, "tl_format"))
    return(result)
  }

  # Multiple files — row-bind
  if (!.quiet) {
    message("Found ", length(data_files), " data file(s) in ", basename(path))
  }

  result <- tl_read_multi(data_files, ..., format = format, .quiet = .quiet)
  attr(result, "tl_source") <- path
  attr(result, "tl_format") <- "zip+multi"
  result
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
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_csv("path/to/data.csv")
#' }
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
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_tsv("path/to/data.tsv")
#' }
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
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_excel("path/to/data.xlsx")
#' # data <- tl_read_excel("path/to/data.xlsx", sheet = "Sheet2")
#' }
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
#' @param path Path to an RDS file.
#'
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_rds("path/to/data.rds")
#' }
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
#' @param path Path to an RData file.
#' @param name Optional name of the object to extract from the RData file. If
#'   \code{NULL} (default), the function returns the first data frame found, or
#'   errors if there are multiple data frames.
#' @param ... Currently unused.
#'
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_rdata("path/to/data.rdata")
#' # data <- tl_read_rdata("path/to/data.rdata", name = "my_data")
#' }
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
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_parquet("path/to/data.parquet")
#' }
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
#' represent tabular data (array of objects or similar). Requires the
#' \pkg{jsonlite} package.
#'
#' @param path Path to a JSON file.
#' @param flatten Logical. Automatically flatten nested data frames? Default is
#'   \code{TRUE}.
#' @param ... Additional arguments passed to \code{jsonlite::fromJSON()}.
#'
#' @return A \code{tidylearn_data} object.
#'
#' @examples
#' \donttest{
#' # data <- tl_read_json("path/to/data.json")
#' }
#'
#' @export
tl_read_json <- function(path, flatten = TRUE, ...) {
  tl_validate_file_path(path)
  tl_check_packages("jsonlite")

  data <- jsonlite::fromJSON(path, flatten = flatten, ...)

  if (!is.data.frame(data)) {
    stop("The JSON file does not contain tabular data. ",
         "tl_read_json() expects an array of objects.",
         call. = FALSE)
  }

  new_tidylearn_data(data, source = path, format = "json")
}
