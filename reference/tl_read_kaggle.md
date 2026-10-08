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

  A Kaggle dataset slug (e.g., `"user/dataset-name"`) or a Kaggle URL. A
  URL may point at any tab of the dataset or competition page (`/data`,
  `/code`, `/versions/2`) and may carry a query; a competition URL is
  read as a competition without `type` being set.

- file:

  The specific file to read from the dataset, as a path within it; a
  file the CLI saved under its base name is found too. If `NULL`, the
  download is searched for files these readers handle (CSV, TSV, Excel,
  Parquet and JSON, with compressed CSV/TSV and `.ndjson`); with
  several, the newest is read and a message names it.

- dest:

  Directory to keep the download in. The default is a fresh per-dataset
  directory under [`tempdir()`](https://rdrr.io/r/base/tempfile.html). A
  supplied `dest` receives this download's files, replacing files of the
  same name; nothing else in it is read, unpacked or changed.

- type:

  Either `"dataset"` (default) or `"competition"`. Left unset, a
  competition URL in `source` sets it.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object containing the downloaded data.

## Examples

``` r
if (FALSE) { # \dontrun{
# Needs the Kaggle CLI and Kaggle credentials
tl_read_kaggle("zillow/zecon", file = "Zip_time_series.csv")
tl_read_kaggle("titanic", file = "train.csv", type = "competition")
tl_read_kaggle("https://www.kaggle.com/competitions/titanic/data",
               file = "train.csv")
} # }
```
