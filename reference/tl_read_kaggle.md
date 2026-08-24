# Read from Kaggle

Downloads a dataset file from Kaggle using the Kaggle CLI and reads it
into a `tidylearn_data` object. Requires the Kaggle CLI to be installed
and configured (`pip install kaggle`).

## Usage

``` r
tl_read_kaggle(source, file = NULL, dest = NULL, type = "dataset", ...)
```

## Arguments

- source:

  A Kaggle dataset slug (e.g., `"user/dataset-name"`) or a Kaggle URL.

- file:

  The specific file to read from the dataset. If `NULL` and the dataset
  contains exactly one file, it is read automatically.

- dest:

  Directory to download files to. The default is a fresh per-dataset
  directory under [`tempdir()`](https://rdrr.io/r/base/tempfile.html);
  supply a path to keep the download.

- type:

  Either `"dataset"` (default) or `"competition"`.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object containing the downloaded data.

## Examples

``` r
# \donttest{
# data <- tl_read_kaggle("zillow/zecon", file = "Zip_time_series.csv")
# data <- tl_read_kaggle("titanic", file = "train.csv", type = "competition")
# }
```
