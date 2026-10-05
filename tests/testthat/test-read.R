# ---- Helpers ----

# A minimal zip writer: stored (uncompressed) entries only. It lets a test
# build an archive holding member names a zip program refuses to write,
# such as "../x.csv" or "/x.csv", and keeps these tests free of any
# external zip program. The CRC-32 each entry needs is read from the
# trailer of a gzip stream, which base R can write.
write_test_zip <- function(path, members) {
  le <- function(x, size) {
    writeBin(as.integer(x), raw(), size = size, endian = "little")
  }
  crc32 <- function(bytes) {
    gz <- tempfile()
    on.exit(unlink(gz))
    con <- gzfile(gz, "wb")
    writeBin(bytes, con)
    close(con)
    stream <- readBin(gz, "raw", file.size(gz))
    stream[(length(stream) - 7L):(length(stream) - 4L)]
  }

  body <- raw(0)
  central <- raw(0)
  for (name in names(members)) {
    bytes <- charToRaw(members[[name]])
    name_raw <- charToRaw(name)
    # version needed, flags, method 0 (stored), time, date 1980-01-01,
    # CRC, compressed and full size, name length, extra-field length
    header <- c(
      le(20, 2), le(0, 2), le(0, 2), le(0, 2), le(33, 2), crc32(bytes),
      le(length(bytes), 4), le(length(bytes), 4),
      le(length(name_raw), 2), le(0, 2)
    )
    offset <- length(body)
    body <- c(body, as.raw(c(0x50, 0x4b, 0x03, 0x04)), header, name_raw, bytes)
    central <- c(
      central, as.raw(c(0x50, 0x4b, 0x01, 0x02)), le(20, 2), header,
      le(0, 2), le(0, 2), le(0, 2), le(0, 4), le(offset, 4), name_raw
    )
  }

  n <- length(members)
  end_record <- c(
    as.raw(c(0x50, 0x4b, 0x05, 0x06)), le(0, 2), le(0, 2), le(n, 2),
    le(n, 2), le(length(central), 4), le(length(body), 4), le(0, 2)
  )
  writeBin(c(body, central, end_record), path)
  invisible(path)
}

# A stand-in for the Kaggle CLI, first on PATH for the rest of the calling
# test. It answers --version. A competition download copies `archive`
# into the -p directory as competition.zip; a dataset download writes
# downloaded.csv there, as the real CLI's --unzip would. Every call is
# appended to the log file it returns. It is an R script behind a
# one-line launcher, so it behaves the same on every platform.
local_fake_kaggle <- function(archive = NULL, env = parent.frame()) {
  bin <- withr::local_tempdir("fake_kaggle_", .local_envir = env)
  log <- file.path(bin, "calls.log")
  script <- file.path(bin, "kaggle.R")
  writeLines(c(
    "args <- commandArgs(trailingOnly = TRUE)",
    paste0("cat(args, '\\n', file = ", deparse(log), ", append = TRUE)"),
    "if (identical(args[1], '--version')) {",
    "  cat('Kaggle API 1.6.17\\n')",
    "  quit(status = 0)",
    "}",
    "dest <- args[which(args == '-p') + 1]",
    "if (identical(args[1], 'competitions')) {",
    paste0(
      "  file.copy(", deparse(archive %||% ""),
      ", file.path(dest, 'competition.zip'), overwrite = TRUE)"
    ),
    "} else {",
    "  writeLines(c('who', 'fresh_download'),",
    "             file.path(dest, 'downloaded.csv'))",
    "}"
  ), script)

  if (.Platform$OS.type == "windows") {
    rscript <- normalizePath(file.path(R.home("bin"), "Rscript.exe"))
    writeLines(
      c("@echo off", sprintf('"%s" --vanilla "%s" %%*', rscript, script)),
      file.path(bin, "kaggle.bat")
    )
  } else {
    launcher <- file.path(bin, "kaggle")
    writeLines(c(
      "#!/bin/sh",
      sprintf('exec "%s" --vanilla "%s" "$@"',
              file.path(R.home("bin"), "Rscript"), script)
    ), launcher)
    Sys.chmod(launcher, "0755")
  }

  withr::local_path(bin, action = "prefix", .local_envir = env)
  # R CMD check points R_TESTS at a startup file the child R must not run
  withr::local_envvar(R_TESTS = NA, .local_envir = env)
  log
}

# ---- Format detection ----

test_that("tl_detect_format identifies file extensions correctly", {
  expect_equal(tl_detect_format("data.csv"), "csv")
  expect_equal(tl_detect_format("data.tsv"), "tsv")
  expect_equal(tl_detect_format("data.txt"), "csv")
  expect_equal(tl_detect_format("data.xls"), "excel")
  expect_equal(tl_detect_format("data.xlsx"), "excel")
  expect_equal(tl_detect_format("data.xlsm"), "excel")
  expect_equal(tl_detect_format("data.rds"), "rds")
  expect_equal(tl_detect_format("data.rdata"), "rdata")
  expect_equal(tl_detect_format("data.rda"), "rdata")
  expect_equal(tl_detect_format("data.parquet"), "parquet")
  expect_equal(tl_detect_format("data.json"), "json")
  expect_equal(tl_detect_format("data.sqlite"), "sqlite")
  expect_equal(tl_detect_format("data.db"), "sqlite")
})

test_that("tl_detect_format is case-insensitive", {
  expect_equal(tl_detect_format("data.CSV"), "csv")
  expect_equal(tl_detect_format("data.XLSX"), "excel")
  expect_equal(tl_detect_format("data.RDS"), "rds")
})

test_that("tl_detect_format identifies URL patterns", {
  expect_equal(
    tl_detect_format("https://github.com/user/repo/blob/main/data.csv"),
    "github"
  )
  url <- "https://raw.githubusercontent.com/user/repo/main/data.csv"
  expect_equal(tl_detect_format(url), "github")
  expect_equal(
    tl_detect_format("https://www.kaggle.com/datasets/user/dataset"),
    "kaggle"
  )
  expect_equal(tl_detect_format("s3://my-bucket/data.csv"), "s3")
  expect_equal(tl_detect_format("postgres://localhost/mydb"), "postgres")
  expect_equal(tl_detect_format("postgresql://localhost/mydb"), "postgres")
  expect_equal(tl_detect_format("mysql://localhost/mydb"), "mysql")
})

test_that("tl_detect_format errors on unrecognizable sources", {
  expect_error(tl_detect_format("unknown_thing"), "Cannot detect format")
  expect_error(tl_detect_format("no_extension"), "Cannot detect format")
})

# ---- tidylearn_data class ----

test_that("new_tidylearn_data creates correct class", {
  data <- new_tidylearn_data(iris, source = "test.csv", format = "csv")
  expect_s3_class(data, "tidylearn_data")
  expect_s3_class(data, "tbl_df")
  expect_equal(attr(data, "tl_source"), "test.csv")
  expect_equal(attr(data, "tl_format"), "csv")
  expect_true(inherits(attr(data, "tl_timestamp"), "POSIXct"))
})

test_that("tidylearn_data works with dplyr verbs", {
  data <- new_tidylearn_data(iris, source = "test.csv", format = "csv")
  filtered <- dplyr::filter(data, Species == "setosa")
  expect_equal(nrow(filtered), 50)
  selected <- dplyr::select(data, Sepal.Length, Species)
  expect_equal(ncol(selected), 2)
})

test_that("print.tidylearn_data shows metadata", {
  data <- new_tidylearn_data(iris, source = "test.csv", format = "csv")
  output <- capture.output(print(data))
  expect_true(any(grepl("tidylearn data", output)))
  expect_true(any(grepl("Source:", output)))
  expect_true(any(grepl("Format:", output)))
  expect_true(any(grepl("Read at:", output)))
})

# ---- tl_read dispatcher ----

test_that("tl_read errors on non-character source", {
  expect_error(tl_read(42), "must be a character string")
  expect_error(tl_read(NULL), "must be a character string")
})

test_that("tl_read errors on missing file", {
  expect_error(tl_read("nonexistent.csv"), "File not found")
})

test_that("tl_read errors on unsupported format", {
  expect_error(tl_read("file.csv", format = "avro"), "Unsupported format")
})

test_that("tl_read auto-detects CSV and reads correctly", {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  write.csv(iris, tmp, row.names = FALSE)

  result <- tl_read(tmp, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(ncol(result), 5)
})

test_that("tl_read auto-detects RDS and reads correctly", {
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(mtcars, tmp)

  result <- tl_read(tmp, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
})

test_that("tl_read respects explicit format override", {
  tmp <- tempfile(fileext = ".txt")
  on.exit(unlink(tmp), add = TRUE)
  write.table(iris, tmp, sep = "\t", row.names = FALSE)

  result <- tl_read(tmp, format = "tsv", .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(attr(result, "tl_format"), "tsv")
})

test_that("tl_read messages can be suppressed", {
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(iris, tmp)

  expect_silent(tl_read(tmp, .quiet = TRUE))
})

test_that("tl_read names the cause for a URL it cannot read", {
  # A web URL fell through to the local CSV reader, which reported the
  # address as a file that does not exist
  expect_error(
    tl_read("https://example.com/data.csv", .quiet = TRUE),
    "reads web URLs only from GitHub and Kaggle"
  )
  expect_error(
    tl_read("https://example.com/data.csv", format = "csv", .quiet = TRUE),
    "reads web URLs only from GitHub and Kaggle"
  )
  # The host decides, not a "github.com" further along the path
  expect_error(
    tl_read("https://mirror.example.org/github.com/data.csv", .quiet = TRUE),
    "reads web URLs only from GitHub and Kaggle"
  )
  expect_equal(
    tl_detect_format("https://www.github.com/owner/repo/blob/main/a.csv"),
    "github"
  )
  expect_equal(tl_detect_format("S3://bucket/data.csv"), "s3")
})

test_that("tl_read refuses a missing or empty source by name", {
  # NA reached an if() and failed with "missing value where TRUE/FALSE
  # needed", alone or inside a vector of paths
  tmp <- withr::local_tempfile(fileext = ".csv")
  write.csv(iris[1:2, ], tmp, row.names = FALSE)

  expect_error(tl_read(NA_character_), "'source' must not contain NA")
  expect_error(tl_read(c(tmp, NA)), "'source' must not contain NA")
  expect_error(tl_read(""), "'source' must not contain NA or empty")
})

test_that("a remote archive is not mistaken for a local zip", {
  skip_if_not_installed("paws.storage")
  # The .zip extension was checked before the protocol, so an S3 key
  # ending in .zip was looked for on the local disk
  expect_error(
    tl_read("s3://bucket/archive.zip", .quiet = TRUE),
    "tl_read_s3\\(\\) cannot read a zip archive"
  )
})

# ---- tl_read_csv ----

test_that("tl_read_csv reads CSV files", {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  write.csv(iris, tmp, row.names = FALSE)

  result <- tl_read_csv(tmp)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(ncol(result), 5)
  expect_equal(attr(result, "tl_format"), "csv")
})

test_that("tl_read_csv errors on missing file", {
  expect_error(tl_read_csv("nonexistent.csv"), "File not found")
})

# ---- tl_read_tsv ----

test_that("tl_read_tsv reads TSV files", {
  tmp <- tempfile(fileext = ".tsv")
  on.exit(unlink(tmp), add = TRUE)
  write.table(iris, tmp, sep = "\t", row.names = FALSE)

  result <- tl_read_tsv(tmp)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(attr(result, "tl_format"), "tsv")
})

# ---- tl_read_excel ----

test_that("tl_read_excel requires readxl package", {
  skip_if_not_installed("readxl")

  path <- readxl::readxl_example("datasets.xlsx")
  result <- tl_read_excel(path, sheet = "mtcars")
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
  expect_equal(attr(result, "tl_format"), "excel")
})

test_that("tl_read_excel errors on missing file", {
  skip_if_not_installed("readxl")
  expect_error(tl_read_excel("nonexistent.xlsx"), "File not found")
})

# ---- tl_read_rds ----

test_that("tl_read_rds reads RDS files into tidylearn_data", {
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(mtcars, tmp)

  result <- tl_read_rds(tmp)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
  expect_equal(attr(result, "tl_format"), "rds")
})

test_that("tl_read_rds coerces data.frame to tibble", {
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(as.data.frame(iris), tmp)

  result <- tl_read_rds(tmp)
  expect_s3_class(result, "tbl_df")
})

test_that("tl_read_rds errors on non-data-frame content", {
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(list(a = 1, b = 2), tmp)

  expect_error(tl_read_rds(tmp), "does not contain a data frame")
})

# ---- tl_read_rdata ----

test_that("tl_read_rdata reads single-object RData files", {
  tmp <- tempfile(fileext = ".rdata")
  on.exit(unlink(tmp), add = TRUE)
  my_data <- iris
  save(my_data, file = tmp)

  result <- tl_read_rdata(tmp)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(attr(result, "tl_format"), "rdata")
})

test_that("tl_read_rdata extracts named object", {
  tmp <- tempfile(fileext = ".rdata")
  on.exit(unlink(tmp), add = TRUE)
  first_df <- iris
  second_df <- mtcars
  save(first_df, second_df, file = tmp)

  result <- tl_read_rdata(tmp, name = "second_df")
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
})

test_that("tl_read_rdata errors on multiple data frames without name", {
  tmp <- tempfile(fileext = ".rdata")
  on.exit(unlink(tmp), add = TRUE)
  first_df <- iris
  second_df <- mtcars
  save(first_df, second_df, file = tmp)

  expect_error(tl_read_rdata(tmp), "Multiple data frames found")
})

test_that("tl_read_rdata errors on missing named object", {
  tmp <- tempfile(fileext = ".rdata")
  on.exit(unlink(tmp), add = TRUE)
  my_data <- iris
  save(my_data, file = tmp)

  expect_error(tl_read_rdata(tmp, name = "nonexistent"), "not found")
})

test_that("tl_read_rdata errors on non-data-frame named object", {
  tmp <- tempfile(fileext = ".rdata")
  on.exit(unlink(tmp), add = TRUE)
  my_list <- list(a = 1, b = 2)
  save(my_list, file = tmp)

  expect_error(tl_read_rdata(tmp, name = "my_list"), "not a data frame")
})

# ---- Output consistency ----

test_that("all readers produce tidylearn_data output", {
  # CSV
  tmp_csv <- tempfile(fileext = ".csv")
  write.csv(iris, tmp_csv, row.names = FALSE)

  # TSV
  tmp_tsv <- tempfile(fileext = ".tsv")
  write.table(iris, tmp_tsv, sep = "\t", row.names = FALSE)

  # RDS
  tmp_rds <- tempfile(fileext = ".rds")
  saveRDS(iris, tmp_rds)

  # RData
  tmp_rdata <- tempfile(fileext = ".rdata")
  my_data <- iris
  save(my_data, file = tmp_rdata)

  on.exit(unlink(c(tmp_csv, tmp_tsv, tmp_rds, tmp_rdata)), add = TRUE)

  results <- list(
    tl_read_csv(tmp_csv),
    tl_read_tsv(tmp_tsv),
    tl_read_rds(tmp_rds),
    tl_read_rdata(tmp_rdata)
  )

  for (result in results) {
    expect_s3_class(result, "tidylearn_data")
    expect_s3_class(result, "tbl_df")
    expect_equal(nrow(result), 150)
    expect_false(is.null(attr(result, "tl_source")))
    expect_false(is.null(attr(result, "tl_format")))
    expect_false(is.null(attr(result, "tl_timestamp")))
  }
})

# ---- tl_read_parquet ----

test_that("tl_read_parquet reads parquet files", {
  skip_if_not_installed("nanoparquet")

  tmp <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp), add = TRUE)
  nanoparquet::write_parquet(iris, tmp)

  result <- tl_read_parquet(tmp)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(ncol(result), 5)
  expect_equal(attr(result, "tl_format"), "parquet")
})

test_that("tl_read auto-detects parquet format", {
  skip_if_not_installed("nanoparquet")

  tmp <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp), add = TRUE)
  nanoparquet::write_parquet(mtcars, tmp)

  result <- tl_read(tmp, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
})

test_that("tl_read_parquet errors on missing file", {
  skip_if_not_installed("nanoparquet")
  expect_error(tl_read_parquet("nonexistent.parquet"), "File not found")
})

# ---- tl_read_json ----

test_that("tl_read_json reads JSON files", {
  skip_if_not_installed("jsonlite")

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  jsonlite::write_json(iris, tmp)

  result <- tl_read_json(tmp)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(attr(result, "tl_format"), "json")
})

test_that("tl_read_json errors on non-tabular JSON", {
  skip_if_not_installed("jsonlite")

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  writeLines('{"key": "value"}', tmp)

  expect_error(tl_read_json(tmp), "does not contain tabular data")
})

test_that("tl_read auto-detects JSON format", {
  skip_if_not_installed("jsonlite")

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  jsonlite::write_json(mtcars, tmp)

  result <- tl_read(tmp, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
})

test_that("tl_read reads newline-delimited JSON", {
  skip_if_not_installed("jsonlite")
  # .ndjson was routed to fromJSON(), which stops at the end of the first
  # record with "parse error: trailing garbage"
  nd <- withr::local_tempfile(fileext = ".ndjson")
  writeLines(c('{"a":1,"b":"x"}', '{"a":2,"b":"y"}'), nd)

  result <- tl_read(nd, .quiet = TRUE)
  expect_equal(result$a, c(1L, 2L))
  expect_equal(result$b, c("x", "y"))
  expect_equal(attr(result, "tl_format"), "json")
})

test_that("scans find .ndjson and compressed CSV, and leave .txt alone", {
  skip_if_not_installed("jsonlite")
  dir <- withr::local_tempdir()
  con <- gzfile(file.path(dir, "a.csv.gz"), "w")
  write.csv(data.frame(a = 1:2), con, row.names = FALSE)
  close(con)
  writeLines('{"a":3}', file.path(dir, "b.ndjson"))
  # A .txt is read as CSV when named directly, but a folder's .txt is as
  # likely to be notes as data, so scans skip it
  writeLines("not data", file.path(dir, "notes.txt"))

  result <- tl_read_dir(dir, .quiet = TRUE)
  expect_setequal(result$a, 1:3)
  expect_setequal(result$source_file, c("a.csv.gz", "b.ndjson"))

  expect_equal(tl_read_dir(dir, format = "csv", .quiet = TRUE)$a, 1:2)
  expect_equal(tl_read_dir(dir, format = "json", .quiet = TRUE)$a, 3L)
  expect_equal(tl_detect_format("data.csv.gz"), "csv")
  expect_equal(tl_detect_format("data.tsv.bz2"), "tsv")
})

# ---- tl_read_sqlite ----

test_that("tl_read_sqlite queries SQLite databases", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")

  tmp_db <- tempfile(fileext = ".sqlite")
  on.exit(unlink(tmp_db), add = TRUE)

  conn <- DBI::dbConnect(RSQLite::SQLite(), tmp_db)
  DBI::dbWriteTable(conn, "iris_tbl", iris)
  DBI::dbDisconnect(conn)

  result <- tl_read_sqlite(tmp_db, "SELECT * FROM iris_tbl")
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_equal(attr(result, "tl_format"), "sqlite")
})

test_that("tl_read_sqlite errors without query", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")

  tmp_db <- tempfile(fileext = ".sqlite")
  on.exit(unlink(tmp_db), add = TRUE)
  conn <- DBI::dbConnect(RSQLite::SQLite(), tmp_db)
  DBI::dbWriteTable(conn, "t", iris)
  DBI::dbDisconnect(conn)

  expect_error(tl_read_sqlite(tmp_db), "query.*required")
})

test_that("tl_read auto-detects sqlite format", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")

  tmp_db <- tempfile(fileext = ".sqlite")
  on.exit(unlink(tmp_db), add = TRUE)

  conn <- DBI::dbConnect(RSQLite::SQLite(), tmp_db)
  DBI::dbWriteTable(conn, "mtcars_tbl", mtcars)
  DBI::dbDisconnect(conn)

  result <- tl_read(tmp_db, query = "SELECT * FROM mtcars_tbl", .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
})

test_that("tl_read_sqlite warns on empty result", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")

  tmp_db <- tempfile(fileext = ".sqlite")
  on.exit(unlink(tmp_db), add = TRUE)

  conn <- DBI::dbConnect(RSQLite::SQLite(), tmp_db)
  DBI::dbWriteTable(conn, "t", iris)
  DBI::dbDisconnect(conn)

  expect_warning(
    tl_read_sqlite(tmp_db, "SELECT * FROM t WHERE 1 = 0"),
    "0 rows"
  )
})

# ---- tl_read_db ----

test_that("tl_read_db reads from live DBI connection", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")

  conn <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
  on.exit(DBI::dbDisconnect(conn), add = TRUE)
  DBI::dbWriteTable(conn, "test_tbl", mtcars)

  result <- tl_read_db(conn, "SELECT * FROM test_tbl")
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
  expect_equal(attr(result, "tl_format"), "database")
})

test_that("tl_read_db errors on non-DBI connection", {
  skip_if_not_installed("DBI")
  expect_error(tl_read_db("not_a_connection", "SELECT 1"), "DBI connection")
})

test_that("tl_read_db errors on empty query", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")

  conn <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
  on.exit(DBI::dbDisconnect(conn), add = TRUE)

  expect_error(tl_read_db(conn, ""), "non-empty SQL")
})

# ---- tl_read_postgres (error path only) ----

test_that("tl_read_postgres requires query", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RPostgres")

  expect_error(tl_read_postgres("localhost"), "query.*required")
})

# The connection tests record what reaches DBI::dbConnect() and stop
# there, so no database server is needed.
local_recorded_dbconnect <- function(env = parent.frame()) {
  seen <- new.env()
  testthat::local_mocked_bindings(
    dbConnect = function(drv, ...) {
      seen$args <- list(...)
      stop("no database server in tests")
    },
    .package = "DBI",
    .env = env
  )
  seen
}

test_that("tl_read_postgres() connects with the parts of a connection string", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RPostgres")
  seen <- local_recorded_dbconnect()

  # RPostgres passes its extra arguments to libpq as connection keywords,
  # and libpq has no "dsn" keyword, so every connection string failed
  # with 'invalid connection option "dsn"'
  expect_error(
    tl_read_postgres(
      "postgres://ana:p%40ss@db.example.com:6543/sales?sslmode=require",
      query = "SELECT 1"
    ),
    "Failed to connect to PostgreSQL"
  )
  expect_mapequal(seen$args, list(
    host = "db.example.com", port = 6543L, dbname = "sales",
    user = "ana", password = "p@ss", sslmode = "require"
  ))
})

test_that("named arguments fill what a PostgreSQL URL leaves out", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RPostgres")
  seen <- local_recorded_dbconnect()

  expect_error(
    tl_read_postgres(
      "postgresql://db.example.com/sales", query = "SELECT 1",
      user = "ana", password = "pw"
    ),
    "Failed to connect to PostgreSQL"
  )
  expect_mapequal(seen$args, list(
    host = "db.example.com", port = 5432, dbname = "sales",
    user = "ana", password = "pw"
  ))
})

test_that("tl_read() never prints the password in a connection string", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RPostgres")
  local_recorded_dbconnect()

  printed <- character(0)
  withCallingHandlers(
    expect_error(
      tl_read("postgres://ana@db.example.com/sales?password=hunter2",
              query = "SELECT 1"),
      "Failed to connect to PostgreSQL"
    ),
    message = function(m) {
      printed <<- c(printed, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_match(printed[1], "password=***", fixed = TRUE)
  expect_false(any(grepl("hunter2", printed, fixed = TRUE)))
})

# ---- tl_read_mysql (error path only) ----

test_that("tl_read_mysql requires query", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RMariaDB")

  expect_error(tl_read_mysql("localhost"), "query.*required")
})

test_that("tl_read_mysql() decodes percent-encoded credentials", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RMariaDB")
  seen <- local_recorded_dbconnect()

  # The password reached the server still encoded, as p%40ss%2Fw0rd
  expect_error(
    tl_read_mysql("mysql://ana:p%40ss%2Fw0rd@db.example.com/sales",
                  query = "SELECT 1"),
    "Failed to connect to MySQL"
  )
  expect_mapequal(seen$args, list(
    host = "db.example.com", port = 3306, dbname = "sales",
    user = "ana", password = "p@ss/w0rd"
  ))
})

test_that("tl_read_mysql() refuses query parameters it would drop", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RMariaDB")
  seen <- local_recorded_dbconnect()

  # ?ssl-mode=REQUIRED landed in the database name. RMariaDB ignores
  # arguments it does not know, so passing it on instead would connect
  # without the TLS the caller asked for.
  expect_error(
    tl_read_mysql("mysql://ana:pw@db.example.com/sales?ssl-mode=REQUIRED",
                  query = "SELECT 1"),
    "MySQL connection strings cannot carry query parameters"
  )
  expect_null(seen$args)
})

# ---- tl_read_bigquery (error path only) ----

test_that("tl_read_bigquery requires query", {
  skip_if_not_installed("bigrquery")

  expect_error(tl_read_bigquery("my-project"), "query.*required")
})

test_that("tl_read_bigquery() gives the query its default dataset", {
  skip_if_not_installed("bigrquery")
  # The dataset only labelled the result; the query never received it,
  # so an unqualified table name could not resolve
  seen <- new.env()
  local_mocked_bindings(
    bq_project_query = function(x, query, ...) {
      seen$args <- list(x = x, query = query, ...)
      "job-table"
    },
    bq_table_download = function(x, ...) data.frame(n = 1L),
    .package = "bigrquery"
  )

  result <- tl_read_bigquery("my-project", "SELECT * FROM t",
                             dataset = "my_dataset")
  expect_equal(
    seen$args$default_dataset,
    bigrquery::bq_dataset("my-project", "my_dataset")
  )
  expect_equal(attr(result, "tl_source"), "bigquery://my-project/my_dataset")

  # The bigquery:// form carries the dataset too
  tl_read_bigquery("bigquery://my-project/other", "SELECT 1")
  expect_equal(
    seen$args$default_dataset,
    bigrquery::bq_dataset("my-project", "other")
  )

  # No dataset, no default
  tl_read_bigquery("my-project", "SELECT 1")
  expect_null(seen$args$default_dataset)
})

# ---- tl_read_github ----

test_that("tl_read_github requires path for owner/repo format", {
  expect_error(
    tl_read_github("user/repo"),
    "path.*required"
  )
})

# Stands in for the network: every download hands back `file` and is
# recorded, so a test can check the URL a reader would have fetched
local_recorded_download <- function(file, env = parent.frame()) {
  seen <- new.env()
  seen$urls <- character(0)
  testthat::local_mocked_bindings(
    tl_download_file = function(url, destfile) {
      seen$urls <- c(seen$urls, url)
      file.copy(file, destfile, overwrite = TRUE)
    },
    .env = env
  )
  seen
}

test_that("GitHub file links resolve to the raw file they name", {
  csv <- withr::local_tempfile(fileext = ".csv")
  write.csv(data.frame(a = 1:2), csv, row.names = FALSE)
  seen <- local_recorded_download(csv)

  # ?raw=true hid the extension, www. was taken for owner/repo shorthand,
  # and an owner called "blob" lost its name to the /blob/ rewrite
  first <- tl_read_github(
    "https://github.com/owner/repo/blob/main/data/file.csv?raw=true"
  )
  tl_read_github("https://www.github.com/owner/repo/blob/main/data/file.csv")
  tl_read_github("https://github.com/blob/repo/blob/main/file.csv")
  tl_read_github("https://github.com/owner/repo/raw/v1.0/file.csv")

  expect_equal(seen$urls, c(
    "https://raw.githubusercontent.com/owner/repo/main/data/file.csv",
    "https://raw.githubusercontent.com/owner/repo/main/data/file.csv",
    "https://raw.githubusercontent.com/blob/repo/main/file.csv",
    "https://raw.githubusercontent.com/owner/repo/v1.0/file.csv"
  ))
  expect_equal(first$a, 1:2)
  expect_equal(attr(first, "tl_format"), "github+csv")

  # A raw link keeps its query: a private file's token lives there
  tl_read_github(
    "https://raw.githubusercontent.com/owner/repo/main/f.csv?token=abc"
  )
  expect_equal(
    seen$urls[5],
    "https://raw.githubusercontent.com/owner/repo/main/f.csv?token=abc"
  )
})

test_that("tl_read_github() names what is wrong with a link it cannot use", {
  expect_error(
    tl_read_github("https://github.com/owner/repo/tree/main/data"),
    "is not a link to a file on GitHub"
  )
  expect_error(
    tl_read_github("https://gitlab.com/owner/repo/-/raw/main/a.csv"),
    "is not a GitHub URL"
  )
  expect_error(
    tl_read_github("owner/repo", path = "data/archive.zip"),
    "tl_read_github\\(\\) cannot read a zip archive"
  )
})

test_that("a remote .rds is refused where reading it can run code", {
  # Before R 4.4.0 a crafted .rds or .rdata runs code as it is read
  # (CVE-2024-27322); the downloaded bytes went straight to readRDS()
  rds <- withr::local_tempfile(fileext = ".rds")
  saveRDS(mtcars, rds)
  seen <- local_recorded_download(rds)
  local_mocked_bindings(tl_unserialize_runs_code = function() TRUE)

  expect_error(
    tl_read_github("owner/repo", path = "data/cars.rds"),
    "CVE-2024-27322"
  )
  expect_error(
    tl_read_github("owner/repo", path = "data/cars.rda"),
    "CVE-2024-27322"
  )
  # Refused before anything was fetched
  expect_length(seen$urls, 0L)

  # A caller who trusts the source can still read it
  result <- tl_read_github("owner/repo", path = "data/cars.rds",
                           trust_rds = TRUE)
  expect_equal(nrow(result), 32L)
})

test_that("a remote .rds is read without an opt-in on current R", {
  rds <- withr::local_tempfile(fileext = ".rds")
  saveRDS(mtcars, rds)
  local_recorded_download(rds)
  local_mocked_bindings(tl_unserialize_runs_code = function() FALSE)

  result <- tl_read_github("owner/repo", path = "data/cars.rds")
  expect_equal(nrow(result), 32L)
  expect_equal(attr(result, "tl_format"), "github+rds")
})

test_that("tl_unserialize_runs_code() tracks the R release with the fix", {
  expect_identical(tl_unserialize_runs_code(), getRversion() < "4.4.0")
})

# ---- tl_read_kaggle (error path only) ----

test_that("tl_read_kaggle errors when kaggle CLI not installed", {
  skip_on_cran()
  skip_if(nzchar(Sys.which("kaggle")), "kaggle CLI is installed")
  expect_error(tl_read_kaggle("user/dataset"), "Kaggle CLI not found")
})

# The slug is interpolated into a command line: system2() applies
# shQuote() to the command and leaves the arguments as written. A pasted
# dataset URL is the vector, and the URL branch is skipped entirely when
# the caller passes a bare string, so the slug is not necessarily
# anything Kaggle produced.
#
# These run without the Kaggle CLI on purpose -- validation happens
# before the CLI is looked for, so a malformed slug is reported as the
# caller's mistake rather than as a missing dependency.

test_that("tl_read_kaggle refuses a slug carrying shell metacharacters", {
  injections <- c(
    "owner/data;whoami",
    "owner/data|whoami",
    "owner/data`whoami`",
    "owner/data$(whoami)",
    "owner/data && rm -rf .",
    "owner/data
whoami"
  )
  for (slug in injections) {
    expect_error(
      tl_read_kaggle(slug), "not a valid Kaggle dataset slug", info = slug
    )
  }

  # Same check after URL parsing, which is where a pasted link lands
  expect_error(
    tl_read_kaggle("https://www.kaggle.com/datasets/owner/data;whoami"),
    "not a valid Kaggle dataset slug"
  )

  # Structurally wrong, if harmless
  expect_error(tl_read_kaggle("nosuchslug"), "not a valid Kaggle dataset slug")
  expect_error(tl_read_kaggle("a/b/c"), "not a valid Kaggle dataset slug")
  expect_error(
    tl_read_kaggle("titanic/x", type = "competition"),
    "not a valid Kaggle competition name"
  )
})

test_that("tl_read_kaggle refuses a file name that escapes the download", {
  expect_error(
    tl_read_kaggle("owner/data", file = "../../etc/passwd"),
    "must be a relative path"
  )
  expect_error(
    tl_read_kaggle("owner/data", file = "/etc/passwd"),
    "must be a relative path"
  )
  expect_error(
    tl_read_kaggle("owner/data", file = paste0("C:", "\\", "windows")),
    "must be a relative path"
  )
  expect_error(
    tl_read_kaggle("owner/data", file = "train;whoami.csv"),
    "may contain only"
  )
})

test_that("a well-formed Kaggle request gets past validation", {
  skip_on_cran()
  skip_if(nzchar(Sys.which("kaggle")), "kaggle CLI is installed")

  # Reaching the CLI check is how we know validation accepted the input;
  # going further would need the CLI and the network.
  for (slug in c("zillow/zecon", "owner/data-set_1.0", "a/b")) {
    expect_error(tl_read_kaggle(slug), "Kaggle CLI not found", info = slug)
  }
  expect_error(
    tl_read_kaggle("titanic", type = "competition"), "Kaggle CLI not found"
  )
  expect_error(
    tl_read_kaggle("owner/data", file = "data/train.csv"),
    "Kaggle CLI not found"
  )
})

test_that("a Kaggle URL resolves to what it names, whatever the tab", {
  # The slug was the last two path segments, so /data, /code and
  # /versions/2 named another owner's dataset; a trailing slash and a
  # ?select= query were refused with messages about something else
  dataset_urls <- c(
    "https://www.kaggle.com/datasets/uciml/iris",
    "https://www.kaggle.com/datasets/uciml/iris/",
    "https://www.kaggle.com/datasets/uciml/iris/data",
    "https://www.kaggle.com/datasets/uciml/iris/code",
    "https://www.kaggle.com/datasets/uciml/iris/versions/2",
    "https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv",
    "https://kaggle.com/datasets/uciml/iris#about"
  )
  for (url in dataset_urls) {
    expect_equal(
      tl_parse_kaggle_url(url),
      list(slug = "uciml/iris", type = "dataset"),
      info = url
    )
  }

  competition_urls <- c(
    "https://www.kaggle.com/competitions/titanic",
    "https://www.kaggle.com/competitions/titanic/",
    "https://www.kaggle.com/competitions/titanic/data",
    "https://www.kaggle.com/competitions/titanic/overview"
  )
  for (url in competition_urls) {
    expect_equal(
      tl_parse_kaggle_url(url),
      list(slug = "titanic", type = "competition"),
      info = url
    )
  }

  expect_error(
    tl_parse_kaggle_url("https://www.kaggle.com/uciml/iris"),
    "Cannot parse Kaggle URL"
  )
  expect_error(
    tl_parse_kaggle_url("https://example.com/datasets/uciml/iris"),
    "is not a Kaggle URL"
  )
})

test_that("a Kaggle URL downloads the dataset or competition it names", {
  skip_on_cran()
  competition <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(competition, list("train.csv" = "who\ntrain\n"))
  log <- local_fake_kaggle(competition)

  result <- suppressMessages(
    tl_read_kaggle("https://www.kaggle.com/datasets/uciml/iris/data")
  )
  expect_equal(result$who, "fresh_download")
  expect_equal(attr(result, "tl_source"), "kaggle://uciml/iris")

  # A competition link is read as a competition without saying so again
  result <- suppressMessages(tl_read_kaggle(
    "https://www.kaggle.com/competitions/titanic/data", file = "train.csv"
  ))
  expect_equal(result$who, "train")

  calls <- readLines(log)
  expect_true(any(grepl("^datasets download -d uciml/iris ", calls)))
  expect_true(any(grepl("^competitions download -c titanic ", calls)))

  expect_error(
    tl_read_kaggle("https://www.kaggle.com/competitions/titanic",
                   type = "dataset"),
    "is a Kaggle competition URL, but type = \"dataset\""
  )
  expect_error(
    tl_read_kaggle("owner/data", type = "competitions"),
    "'type' must be \"dataset\" or \"competition\""
  )
})

test_that("tl_read_kaggle(dest =) touches only this call's download", {
  skip_on_cran()
  competition <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(competition, list(
    "train.csv" = "who\ncompetition_train\n",
    "test.csv" = "who\ncompetition_test\n"
  ))
  local_fake_kaggle(competition)

  dest <- withr::local_tempdir()
  writeLines(c("who", "MY_EDITED_NOTES"), file.path(dest, "notes.csv"))
  write_test_zip(
    file.path(dest, "old_project.zip"),
    list("notes.csv" = "who\nOLD_PROJECT\n")
  )

  # Every zip in dest was unpacked, so an unrelated archive overwrote the
  # user's notes.csv
  result <- suppressMessages(tl_read_kaggle(
    "titanic", type = "competition", file = "train.csv", dest = dest
  ))
  expect_equal(result$who, "competition_train")
  expect_equal(readLines(file.path(dest, "notes.csv"))[2], "MY_EDITED_NOTES")
  # The download itself is kept in dest
  expect_true(all(file.exists(
    file.path(dest, c("competition.zip", "train.csv", "test.csv"))
  )))

  # A later dataset download into the same folder re-extracted the
  # competition's archive, whose members were then the newest files and
  # were returned in place of the dataset
  result <- suppressMessages(tl_read_kaggle("owner/data-set", dest = dest))
  expect_equal(result$who, "fresh_download")
  expect_equal(readLines(file.path(dest, "notes.csv"))[2], "MY_EDITED_NOTES")
  expect_true(file.exists(file.path(dest, "downloaded.csv")))
})

test_that("tl_read_kaggle refuses a download whose members escape", {
  skip_on_cran()
  withr::defer(unlink(file.path(tempdir(), "escaped_kaggle.csv")))
  evil <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(evil, list(
    "train.csv" = "a\n1\n",
    "../escaped_kaggle.csv" = "a\n2\n"
  ))
  local_fake_kaggle(evil)

  expect_error(
    tl_read_kaggle("titanic", type = "competition"),
    "outside the directory it is unpacked into"
  )
  expect_false(file.exists(file.path(tempdir(), "escaped_kaggle.csv")))
})

# ---- tl_read_s3 (error path only) ----

test_that("tl_read_s3 errors on invalid URI", {
  skip_if_not_installed("paws.storage")
  expect_error(tl_read_s3("s3://bucket-only"), "Invalid S3 URI")
})

test_that("tl_read_s3() refuses an .rds where reading it can run code", {
  skip_if_not_installed("paws.storage")
  local_mocked_bindings(tl_unserialize_runs_code = function() TRUE)
  # Refused before an S3 client is created, so no credentials are needed
  local_mocked_bindings(
    s3 = function(...) stop("the S3 client should not be created"),
    .package = "paws.storage"
  )

  expect_error(tl_read_s3("s3://bucket/data/cars.rds"), "CVE-2024-27322")
  expect_error(tl_read_s3("S3://bucket/data/cars.RData"), "CVE-2024-27322")
})

test_that("tl_read_s3() reads a downloaded object, with an opt-in for .rds", {
  skip_if_not_installed("paws.storage")
  rds <- withr::local_tempfile(fileext = ".rds")
  saveRDS(mtcars, rds)
  body <- readBin(rds, "raw", file.size(rds))
  local_mocked_bindings(tl_unserialize_runs_code = function() TRUE)
  local_mocked_bindings(
    s3 = function(...) {
      list(get_object = function(...) list(Body = body))
    },
    .package = "paws.storage"
  )

  result <- tl_read_s3("s3://bucket/data/cars.rds", trust_rds = TRUE)
  expect_equal(nrow(result), 32L)
  expect_equal(attr(result, "tl_source"), "s3://bucket/data/cars.rds")
})

# ---- Internal helpers ----

test_that("tl_parse_db_url parses connection strings", {
  result <- tl_parse_db_url("mysql://user:pass@localhost:3306/mydb")
  expect_equal(result$user, "user")
  expect_equal(result$password, "pass")
  expect_equal(result$host, "localhost")
  expect_equal(result$port, 3306L)
  expect_equal(result$dbname, "mydb")
})

test_that("tl_parse_db_url handles minimal URLs", {
  result <- tl_parse_db_url("postgres://localhost/mydb")
  expect_null(result$user)
  expect_null(result$password)
  expect_equal(result$host, "localhost")
  expect_null(result$port)
  expect_equal(result$dbname, "mydb")
})

test_that("tl_parse_db_url decodes credentials and splits off the query", {
  # Credentials stayed percent-encoded, and the query string became part
  # of the database name
  parsed <- tl_parse_db_url(
    "mysql://us%40er:p%40ss%2Fw0rd@db.example.com:3306/sales?ssl-mode=REQUIRED"
  )
  expect_equal(parsed$user, "us@er")
  expect_equal(parsed$password, "p@ss/w0rd")
  expect_equal(parsed$host, "db.example.com")
  expect_equal(parsed$port, 3306L)
  expect_equal(parsed$dbname, "sales")
  expect_equal(parsed$params, list(`ssl-mode` = "REQUIRED"))

  expect_equal(tl_parse_db_url("postgres://localhost/mydb")$params, list())
  expect_equal(tl_parse_db_url("postgres://:pw@localhost/db")$password, "pw")
  expect_null(tl_parse_db_url("postgres://:pw@localhost/db")$user)

  # An unencoded '@' in a password cannot be told from the host separator
  expect_error(
    tl_parse_db_url("postgres://ana:p@ss@localhost/db"),
    "Percent-encode"
  )
})

test_that("connection strings are redacted whatever form the password takes", {
  redacted <- c(
    "postgres://user:secret@host/db"       = "postgres://user:***@host/db",
    "postgres://:secret@host/db"           = "postgres://:***@host/db",
    "POSTGRES://user:secret@host/db"       = "POSTGRES://user:***@host/db",
    "postgres://user:p@ss@host/db"         = "postgres://user:***@host/db",
    "postgres://host/db?password=secret"   = "postgres://host/db?password=***",
    "postgres://host/db?sslmode=require&password=secret&x=1" =
      "postgres://host/db?sslmode=require&password=***&x=1",
    "host=localhost password=secret dbname=x" =
      "host=localhost password=*** dbname=x",
    "user:secret@host"                     = "user:***@host"
  )
  for (url in names(redacted)) {
    expect_equal(tl_redact_db_url(url), redacted[[url]], info = url)
  }

  # Nothing that carries no password changes, file paths included
  unchanged <- c(
    "postgres://user@host/db",
    "C:/Users/ana@corp/data.csv",
    "/home/ana/pass=1/data.csv"
  )
  for (x in unchanged) {
    expect_equal(tl_redact_db_url(x), x, info = x)
  }
})

test_that("tl_parse_kaggle_url extracts dataset slug", {
  parsed <- tl_parse_kaggle_url(
    "https://www.kaggle.com/datasets/zillow/zecon"
  )
  expect_equal(parsed$slug, "zillow/zecon")
  expect_equal(parsed$type, "dataset")
})

test_that("tl_parse_kaggle_url extracts competition name", {
  parsed <- tl_parse_kaggle_url(
    "https://www.kaggle.com/competitions/titanic"
  )
  expect_equal(parsed$slug, "titanic")
  expect_equal(parsed$type, "competition")
})

# ---- Multi-path reading ----

test_that("tl_read accepts multiple paths and row-binds", {
  tmp1 <- tempfile(fileext = ".csv")
  tmp2 <- tempfile(fileext = ".csv")
  on.exit(unlink(c(tmp1, tmp2)), add = TRUE)

  write.csv(iris[1:50, ], tmp1, row.names = FALSE)
  write.csv(iris[51:100, ], tmp2, row.names = FALSE)

  result <- tl_read(c(tmp1, tmp2), .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 100)
  expect_true("source_file" %in% names(result))
  expect_equal(length(unique(result$source_file)), 2)
})

test_that("tl_read multi-path works with explicit format", {
  tmp1 <- tempfile(fileext = ".txt")
  tmp2 <- tempfile(fileext = ".txt")
  on.exit(unlink(c(tmp1, tmp2)), add = TRUE)

  write.table(iris[1:50, ], tmp1, sep = "\t", row.names = FALSE)
  write.table(iris[51:100, ], tmp2, sep = "\t", row.names = FALSE)

  result <- tl_read(c(tmp1, tmp2), format = "tsv", .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 100)
})

# Two sales.csv files, one per year, in their own folders
local_sales_by_year <- function(env = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = env)
  for (year in c("2023", "2024")) {
    dir.create(file.path(dir, year))
    write.csv(data.frame(v = as.integer(year)),
              file.path(dir, year, "sales.csv"), row.names = FALSE)
  }
  dir
}

test_that("files named directly are labelled from their common folder", {
  # Labels were basenames, so the two years' files were both "sales.csv"
  dir <- local_sales_by_year()
  result <- tl_read(file.path(dir, c("2023", "2024"), "sales.csv"),
                    .quiet = TRUE)
  expect_equal(result$source_file, c("2023/sales.csv", "2024/sales.csv"))
  expect_equal(result$v, c(2023L, 2024L))
})

test_that("a clashing source_file column moves every label aside", {
  # Only the file that had the column was labelled in tl_source_file; the
  # rest were labelled in its source_file column, splitting the labels
  # across two columns and mixing them into the data
  dir <- withr::local_tempdir()
  write.csv(data.frame(v = 1:2, source_file = "crm_export"),
            file.path(dir, "a.csv"), row.names = FALSE)
  write.csv(data.frame(v = 3:4), file.path(dir, "b.csv"), row.names = FALSE)

  expect_warning(
    result <- tl_read(file.path(dir, c("a.csv", "b.csv")), .quiet = TRUE),
    "Column 'source_file' already exists in the data"
  )
  expect_equal(result$tl_source_file, c("a.csv", "a.csv", "b.csv", "b.csv"))
  expect_equal(result$source_file, c("crm_export", "crm_export", NA, NA))
})

# ---- tl_read_dir ----

test_that("tl_read_dir reads all CSVs from a directory", {
  dir <- tempfile(pattern = "tl_test_dir_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  write.csv(iris[1:50, ], file.path(dir, "part1.csv"), row.names = FALSE)
  write.csv(iris[51:100, ], file.path(dir, "part2.csv"), row.names = FALSE)

  result <- tl_read_dir(dir, format = "csv", .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 100)
  expect_true("source_file" %in% names(result))
  expect_equal(sort(unique(result$source_file)), c("part1.csv", "part2.csv"))
})

test_that("tl_read_dir works with pattern filter", {
  dir <- tempfile(pattern = "tl_test_dir_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  write.csv(iris[1:50, ], file.path(dir, "sales_jan.csv"), row.names = FALSE)
  write.csv(iris[51:100, ], file.path(dir, "sales_feb.csv"), row.names = FALSE)
  write.csv(iris[101:150, ], file.path(dir, "other.csv"), row.names = FALSE)

  result <- tl_read_dir(dir, pattern = "^sales_", .quiet = TRUE)
  expect_equal(nrow(result), 100)
  expect_equal(length(unique(result$source_file)), 2)
})

test_that("tl_read_dir scans recursively when asked", {
  dir <- tempfile(pattern = "tl_test_dir_")
  dir.create(dir)
  subdir <- file.path(dir, "sub")
  dir.create(subdir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  write.csv(iris[1:50, ], file.path(dir, "a.csv"), row.names = FALSE)
  write.csv(iris[51:100, ], file.path(subdir, "b.csv"), row.names = FALSE)

  # Non-recursive misses subdir file
  result_flat <- tl_read_dir(dir, format = "csv", .quiet = TRUE)
  expect_equal(nrow(result_flat), 50)

  # Recursive finds both
  result_deep <- tl_read_dir(dir, format = "csv", recursive = TRUE,
                             .quiet = TRUE)
  expect_equal(nrow(result_deep), 100)
})

test_that("a recursive scan labels each file by its path below the folder", {
  # Both years' files were labelled "sales.csv", and a partition key held
  # in a folder name was lost
  dir <- local_sales_by_year()
  result <- tl_read_dir(dir, recursive = TRUE, .quiet = TRUE)
  expect_equal(
    result$source_file[order(result$v)],
    c("2023/sales.csv", "2024/sales.csv")
  )
})

test_that("tl_read_dir errors on empty directory", {
  dir <- tempfile(pattern = "tl_test_dir_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  expect_error(tl_read_dir(dir, .quiet = TRUE), "No data files found")
})

test_that("tl_read_dir errors on non-existent directory", {
  expect_error(tl_read_dir("/fake/dir"), "Directory not found")
})

test_that("tl_read dispatches to tl_read_dir for directories", {
  dir <- tempfile(pattern = "tl_test_dir_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  write.csv(iris, file.path(dir, "data.csv"), row.names = FALSE)

  result <- tl_read(dir, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
})

# ---- tl_read_zip ----

test_that("tl_read_zip reads single file from archive", {
  dir <- tempfile(pattern = "tl_zip_src_")
  dir.create(dir)
  zip_path <- tempfile(fileext = ".zip")
  on.exit(unlink(c(dir, zip_path), recursive = TRUE), add = TRUE)

  csv_path <- file.path(dir, "data.csv")
  write.csv(iris, csv_path, row.names = FALSE)

  # Create zip - use full path for the file inside
  withr::with_dir(dir, utils::zip(zip_path, "data.csv"))

  result <- tl_read_zip(zip_path, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
  expect_true(grepl("zip\\+", attr(result, "tl_format")))
})

test_that("tl_read_zip reads specific file from archive", {
  dir <- tempfile(pattern = "tl_zip_src_")
  dir.create(dir)
  zip_path <- tempfile(fileext = ".zip")
  on.exit(unlink(c(dir, zip_path), recursive = TRUE), add = TRUE)

  write.csv(iris, file.path(dir, "iris.csv"), row.names = FALSE)
  write.csv(mtcars, file.path(dir, "mtcars.csv"), row.names = FALSE)

  withr::with_dir(dir, utils::zip(zip_path, c("iris.csv", "mtcars.csv")))

  result <- tl_read_zip(zip_path, file = "mtcars.csv", .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 32)
})

test_that("tl_read_zip row-binds multiple files", {
  dir <- tempfile(pattern = "tl_zip_src_")
  dir.create(dir)
  zip_path <- tempfile(fileext = ".zip")
  on.exit(unlink(c(dir, zip_path), recursive = TRUE), add = TRUE)

  write.csv(iris[1:50, ], file.path(dir, "part1.csv"), row.names = FALSE)
  write.csv(iris[51:100, ], file.path(dir, "part2.csv"), row.names = FALSE)

  withr::with_dir(dir, utils::zip(zip_path, c("part1.csv", "part2.csv")))

  result <- tl_read_zip(zip_path, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 100)
  expect_true("source_file" %in% names(result))
})

test_that("tl_read_zip errors on missing file in archive", {
  dir <- tempfile(pattern = "tl_zip_src_")
  dir.create(dir)
  zip_path <- tempfile(fileext = ".zip")
  on.exit(unlink(c(dir, zip_path), recursive = TRUE), add = TRUE)

  write.csv(iris, file.path(dir, "data.csv"), row.names = FALSE)

  withr::with_dir(dir, utils::zip(zip_path, "data.csv"))

  expect_error(
    tl_read_zip(zip_path, file = "nonexistent.csv", .quiet = TRUE),
    "not found in archive"
  )
})

test_that("tl_read dispatches to tl_read_zip for .zip files", {
  dir <- tempfile(pattern = "tl_zip_src_")
  dir.create(dir)
  zip_path <- tempfile(fileext = ".zip")
  on.exit(unlink(c(dir, zip_path), recursive = TRUE), add = TRUE)

  write.csv(iris, file.path(dir, "data.csv"), row.names = FALSE)

  withr::with_dir(dir, utils::zip(zip_path, "data.csv"))

  result <- tl_read(zip_path, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
})

# ---- tl_read_zip format selection ----

test_that("tl_read_zip(format=) selects members rather than forcing them", {
  d <- file.path(tempdir(), "tl_mixed_zip")
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)

  write.csv(data.frame(a = 1:2), file.path(d, "a.csv"), row.names = FALSE)
  write.csv(data.frame(a = 3:4), file.path(d, "b.csv"), row.names = FALSE)
  writeLines('[{"a":1}]', file.path(d, "c.json"))

  archive <- file.path(d, "mixed.zip")
  old <- setwd(d)
  on.exit(setwd(old), add = TRUE)
  utils::zip(archive, c("a.csv", "b.csv", "c.json"), flags = "-q")
  setwd(old)

  # Forcing csv onto the json read its first line as a header, so the
  # result carried a column literally named [{"a":1}] and the row-bind
  # succeeded without complaint.
  result <- suppressMessages(tl_read_zip(archive, format = "csv"))
  expect_equal(nrow(result), 4L)
  expect_setequal(names(result), c("a", "source_file"))
  expect_false(any(grepl("[{]", names(result))))
  expect_setequal(result$a, 1:4)

  # A format nothing matches is an error naming what is there
  expect_error(
    tl_read_zip(archive, format = "parquet"),
    "No parquet files in archive"
  )
})

test_that("tl_read_zip(format =) selects from one member as from several", {
  skip_if_not_installed("jsonlite")
  # With a single member the format was forced onto it, so a JSON read as
  # CSV came back with no rows and JSON fragments for column names
  archive <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(archive, list("data.json" = '[{"a":1},{"a":2}]'))

  expect_error(
    tl_read_zip(archive, format = "csv", .quiet = TRUE),
    "No csv files in archive"
  )
  expect_equal(tl_read_zip(archive, format = "json", .quiet = TRUE)$a, 1:2)

  # Naming the member still forces the format onto it
  tab_separated <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(tab_separated, list("data.csv" = "a\tb\n1\t2\n"))
  forced <- tl_read_zip(tab_separated, file = "data.csv", format = "tsv",
                        .quiet = TRUE)
  expect_equal(names(forced), c("a", "b"))
})

# ---- tl_read_zip member selection ----

test_that("tl_read_zip(file =) prefers an exact name to a partial match", {
  # "train.csv" also matched "full_train.csv", which sorted first and was
  # read instead -- silently with .quiet = TRUE
  archive <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(archive, list(
    "full_train.csv" = "who\nfull_train\n",
    "train.csv" = "who\ntrain\n",
    "test.csv" = "who\ntest\n"
  ))

  expect_equal(tl_read_zip(archive, file = "train.csv", .quiet = TRUE)$who,
               "train")
  # A partial match is still found when it is the only one
  expect_equal(tl_read_zip(archive, file = "full_", .quiet = TRUE)$who,
               "full_train")
  # An ambiguous one is refused rather than settled by file order
  expect_error(
    tl_read_zip(archive, file = "train", .quiet = TRUE),
    "'train' matches 2 files in the archive"
  )
})

test_that("tl_read_zip(file =) selects a member by its path", {
  # Only base names were matched, so a member in a folder could not be
  # named and the two sales.csv files were indistinguishable
  archive <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(archive, list(
    "2023/sales.csv" = "yr\n2023\n",
    "2024/sales.csv" = "yr\n2024\n"
  ))

  result <- tl_read_zip(archive, file = "2024/sales.csv", .quiet = TRUE)
  expect_equal(result$yr, 2024)
  expect_equal(attr(result, "tl_source"), paste0(archive, "//2024/sales.csv"))
  expect_error(
    tl_read_zip(archive, file = "sales.csv", .quiet = TRUE),
    "'sales.csv' matches 2 files in the archive"
  )

  # Read together, each row is labelled by the member's path
  both <- tl_read_zip(archive, .quiet = TRUE)
  expect_equal(both$source_file[order(both$yr)],
               c("2023/sales.csv", "2024/sales.csv"))
})

test_that("tl_read_zip refuses members that would land outside its folder", {
  # R before 4.5.1 extracts "../" and absolute member names as written, so
  # a crafted archive could plant a file anywhere the user can write
  withr::defer(unlink(file.path(tempdir(), "escaped.csv")))
  unsafe <- c("../escaped.csv", "sub/../../escaped.csv", "/escaped.csv",
              "..\\escaped.csv", "C:/escaped.csv")
  for (name in unsafe) {
    archive <- withr::local_tempfile(fileext = ".zip")
    members <- list("data.csv" = "a\n1\n", "a\n2\n")
    names(members)[2] <- name
    write_test_zip(archive, members)

    expect_error(
      tl_read_zip(archive, .quiet = TRUE),
      "outside the directory it is unpacked into",
      info = name
    )
  }
  expect_false(file.exists(file.path(tempdir(), "escaped.csv")))
})

test_that("tl_read_zip still reads members with dots and folders", {
  archive <- withr::local_tempfile(fileext = ".zip")
  write_test_zip(archive, list(
    "v1..v2/data.csv" = "a\n1\n",
    "./more..csv" = "a\n2\n"
  ))
  expect_setequal(tl_read_zip(archive, .quiet = TRUE)$a, c(1, 2))
})
