# Read from Amazon S3

Downloads a file from an S3 bucket and reads it into a `tidylearn_data`
object. The file format is auto-detected from the key's extension, or
can be specified explicitly. Requires the paws.storage package and valid
AWS credentials.

## Usage

``` r
tl_read_s3(source, format = NULL, region = NULL, ...)
```

## Arguments

- source:

  An S3 URI (e.g., `"s3://bucket/path/to/file.csv"`).

- format:

  Optional format override for the downloaded file. If `NULL`,
  auto-detected from the S3 key extension.

- region:

  AWS region. If `NULL`, uses the default from your AWS configuration.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object containing the downloaded data.

## Examples

``` r
# \donttest{
# data <- tl_read_s3("s3://my-bucket/data/sales.csv")
# data <- tl_read_s3("s3://my-bucket/data/results.parquet")
# }
```
