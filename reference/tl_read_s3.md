# Read from Amazon S3

Downloads a file from an S3 bucket and reads it into a `tidylearn_data`
object. The file format is auto-detected from the key's extension, or
can be specified explicitly. Requires the paws.storage package and valid
AWS credentials.

## Usage

``` r
tl_read_s3(source, format = NULL, region = NULL, ..., trust_rds = FALSE)
```

## Arguments

- source:

  An S3 URI (e.g., `"s3://bucket/path/to/file.csv"`). Zip archives are
  not read from S3: download the object and read it with
  [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md).

- format:

  Optional format override for the downloaded file. If `NULL`,
  auto-detected from the S3 key extension.

- region:

  AWS region. If `NULL`, uses the default from your AWS configuration.

- ...:

  Additional arguments passed to the format-specific reader.

- trust_rds:

  Logical. Read an `.rds`, `.rdata` or `.rda` object on R older than
  4.4.0? Those versions can run code embedded in a crafted file as it is
  read (CVE-2024-27322), so such objects are refused there unless this
  is `TRUE`. It has no effect on R 4.4.0 or later. Default `FALSE`.

## Value

A `tidylearn_data` object containing the downloaded data.

## Reading R serialisation files

An `.rds` or `.rdata` object is rebuilt with
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) or
[`load()`](https://rdrr.io/r/base/load.html), which recreate whatever R
objects the file describes. Read them only from a bucket you trust, on
any version of R.

## Examples

``` r
if (FALSE) { # \dontrun{
# Needs AWS credentials
tl_read_s3("s3://my-bucket/data/sales.csv")
tl_read_s3("s3://my-bucket/data/results.parquet", region = "eu-west-1")
} # }
```
