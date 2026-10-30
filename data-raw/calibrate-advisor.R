# Measures the seconds per complexity unit tl_compute_advisor() uses to
# estimate a fit's runtime. The cpu_const values in .tl_method_profiles are the
# constants this prints, and data-raw/advisor-calibration.csv holds the timings
# behind them. cpu_sec against elapsed_sec shows how many cores a fit used.
#
# Run from the package root on an otherwise idle machine, with the package
# installed. "deep" needs keras3 and a Python backend. TL_MACHINE names the
# hardware in the output.
#
#   TL_MACHINE="<cpu>" Rscript data-raw/calibrate-advisor.R

library(tidylearn)

complexity <- getFromNamespace("tl_method_complexity_internal", "tidylearn")

# Rows by predictors for each fit. The three sizes vary rows and predictors
# separately, so a complexity formula that scales wrongly shows up as a spread
# in seconds per unit. Each is large enough that the fit takes well over the
# timer's resolution.
sizes <- list(
  linear      = list(c(1e5, 10), c(5e5, 10), c(1e5, 40)),
  polynomial  = list(c(1e5, 10), c(5e5, 10), c(1e5, 20)),
  logistic    = list(c(5e4, 10), c(2e5, 10), c(5e4, 40)),
  ridge       = list(c(2e4, 10), c(2e5, 10), c(2e4, 40)),
  lasso       = list(c(2e4, 10), c(2e5, 10), c(2e4, 40)),
  elastic_net = list(c(2e4, 10), c(2e5, 10), c(2e4, 40)),
  tree        = list(c(2e4, 10), c(2e5, 10), c(2e4, 40)),
  forest      = list(c(2e3, 10), c(1e4, 10), c(2e3, 40)),
  boost       = list(c(2e4, 10), c(2e5, 10), c(2e4, 40)),
  xgboost     = list(c(2e4, 10), c(2e5, 10), c(2e4, 40)),
  nn          = list(c(5e3, 10), c(2e4, 10), c(5e3, 40)),
  svm         = list(c(2e3, 10), c(8e3, 10), c(2e3, 40)),
  deep        = list(c(1e3, 10), c(5e3, 10), c(1e3, 40))
)

# xgboost is also timed on one thread, to see what its extra cores buy. The
# constant comes from the default, which is the fit a user gets.
extra_args <- list(xgboost = list(list(nthread = 1L), list()))

make_data <- function(rows, cols, binary) {
  x <- matrix(stats::rnorm(rows * cols), rows)
  colnames(x) <- paste0("x", seq_len(cols))
  signal <- 2 * x[, 1] - x[, 2]
  y <- if (binary) {
    factor(stats::rbinom(rows, 1, stats::plogis(signal)))
  } else {
    signal + stats::rnorm(rows)
  }
  data.frame(y = y, x)
}

time_fit <- function(method, rows, cols, args) {
  data <- make_data(rows, cols, method == "logistic")
  gc(FALSE)
  timing <- system.time(suppressWarnings(suppressMessages(
    do.call(tl_model, c(list(data, y ~ ., method = method), args))
  )))
  units <- complexity(method, rows, cols, list())

  data.frame(
    method = method,
    rows = rows,
    cols = cols,
    nthread = if (is.null(args$nthread)) NA_integer_ else args$nthread,
    complexity = units,
    elapsed_sec = timing[["elapsed"]],
    cpu_sec = timing[["user.self"]] + timing[["sys.self"]],
    sec_per_unit = timing[["elapsed"]] / units
  )
}

set.seed(1)
results <- list()
for (method in names(sizes)) {
  if (method == "deep") {
    if (!requireNamespace("keras3", quietly = TRUE)) {
      next
    }
    # The first fit starts Python and TensorFlow, which is not fitting time
    time_fit(method, 100, 2, list())
  }
  arg_sets <- extra_args[[method]]
  if (is.null(arg_sets)) {
    arg_sets <- list(list())
  }
  for (size in sizes[[method]]) {
    for (args in arg_sets) {
      row <- time_fit(method, size[1], size[2], args)
      print(row, row.names = FALSE)
      results[[length(results) + 1L]] <- row
    }
  }
}

results <- do.call(rbind, results)
results$machine <- Sys.getenv("TL_MACHINE", unset = "unrecorded")
results$cores <- parallel::detectCores(logical = FALSE)
results$r_version <- as.character(getRversion())
results$date <- as.character(Sys.Date())

utils::write.csv(results, "data-raw/advisor-calibration.csv", row.names = FALSE)

# Each constant is the geometric midpoint of the method's seconds per unit, so
# a fit timed here is at most sqrt(spread) from its estimate
defaults <- results[is.na(results$nthread), ]
per_method <- split(defaults$sec_per_unit, defaults$method)
midpoint <- function(x) signif(sqrt(min(x) * max(x)), 2)
worst_factor <- function(x) round(sqrt(max(x) / min(x)), 1)
summary <- data.frame(
  method = names(per_method),
  cpu_const = vapply(per_method, midpoint, numeric(1)),
  worst_factor = vapply(per_method, worst_factor, numeric(1))
)
print(summary[match(names(sizes), summary$method), ], row.names = FALSE)
