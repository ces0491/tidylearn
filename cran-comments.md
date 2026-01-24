# tidylearn CRAN Submission

## Resubmission

This is a resubmission. In this version I have:

* Added method references with DOIs to the Description field in DESCRIPTION
* Added \value tags to all exported print, summary, plot, and predict methods
* Added \arguments tags for the pipe operator documentation
* Replaced \dontrun{} with \donttest{} in all examples
* Removed set.seed() with hardcoded value from tl_run_pipeline()

## R CMD check results

0 errors | 0 warnings | 2 notes

### NOTEs

1. **New submission**

   This is the first submission of tidylearn to CRAN.

2. **Suggested packages not available**

   ```
   Packages suggested but not available for checking:
     'arules', 'arulesViz', 'mclust'
   ```

   These packages are optional dependencies used only for specific features
   (market basket analysis and model-based clustering). The package functions
   correctly without them.

## Package Description

tidylearn provides a unified tidyverse-compatible interface to R's machine
learning ecosystem. It wraps established packages like glmnet, randomForest,
xgboost, e1071, cluster, and dbscan - providing consistent function signatures,
tidy tibble output, and unified visualization.

**Important**: tidylearn does NOT reimplement any algorithms. All computations
are performed by the underlying packages. tidylearn provides a consistent
interface layer.

## Test environments

* Windows 11 x64, R 4.4.3
* GitHub Actions: ubuntu-latest (R-devel, R-release), windows-latest (R-release), macos-latest (R-release)

## Downstream dependencies

This is a new package with no reverse dependencies.
