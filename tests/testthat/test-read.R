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

# ---- tl_read_mysql (error path only) ----

test_that("tl_read_mysql requires query", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RMariaDB")

  expect_error(tl_read_mysql("localhost"), "query.*required")
})

# ---- tl_read_bigquery (error path only) ----

test_that("tl_read_bigquery requires query", {
  skip_if_not_installed("bigrquery")

  expect_error(tl_read_bigquery("my-project"), "query.*required")
})

# ---- tl_read_github ----

test_that("tl_read_github requires path for owner/repo format", {
  expect_error(
    tl_read_github("user/repo"),
    "path.*required"
  )
})

# ---- tl_read_kaggle (error path only) ----

test_that("tl_read_kaggle errors when kaggle CLI not installed", {
  skip_on_cran()
  skip_if(nzchar(Sys.which("kaggle")), "kaggle CLI is installed")
  expect_error(tl_read_kaggle("user/dataset"), "Kaggle CLI not found")
})

# ---- tl_read_s3 (error path only) ----

test_that("tl_read_s3 errors on invalid URI", {
  skip_if_not_installed("paws.storage")
  expect_error(tl_read_s3("s3://bucket-only"), "Invalid S3 URI")
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

test_that("tl_parse_kaggle_url extracts dataset slug", {
  slug <- tl_parse_kaggle_url(
    "https://www.kaggle.com/datasets/zillow/zecon"
  )
  expect_equal(slug, "zillow/zecon")
})

test_that("tl_parse_kaggle_url extracts competition name", {
  slug <- tl_parse_kaggle_url(
    "https://www.kaggle.com/competitions/titanic"
  )
  expect_equal(slug, "titanic")
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
  old_wd <- getwd()
  setwd(dir)
  utils::zip(zip_path, "data.csv")
  setwd(old_wd)

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

  old_wd <- getwd()
  setwd(dir)
  utils::zip(zip_path, c("iris.csv", "mtcars.csv"))
  setwd(old_wd)

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

  old_wd <- getwd()
  setwd(dir)
  utils::zip(zip_path, c("part1.csv", "part2.csv"))
  setwd(old_wd)

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

  old_wd <- getwd()
  setwd(dir)
  utils::zip(zip_path, "data.csv")
  setwd(old_wd)

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

  old_wd <- getwd()
  setwd(dir)
  utils::zip(zip_path, "data.csv")
  setwd(old_wd)

  result <- tl_read(zip_path, .quiet = TRUE)
  expect_s3_class(result, "tidylearn_data")
  expect_equal(nrow(result), 150)
})
