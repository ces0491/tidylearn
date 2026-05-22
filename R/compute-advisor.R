#' Advise on the best compute tier for a tidylearn fit
#'
#' Estimates runtime and feasibility on local CPU, local GPU (when
#' available and applicable), and cloud GPU (stubbed until cloud
#' integration lands), then returns a structured recommendation. Useful
#' before kicking off a long fit — call this first to see whether the
#' problem is laptop-sized, GPU-sized, or cloud-sized.
#'
#' Estimates are deliberately rough — order-of-magnitude, not bills.
#' Per-method scaling constants are calibrated against typical hardware
#' and will be off by 2-3x in either direction for any individual job.
#' Treat the recommendation as a starting point, not gospel.
#'
#' @param x Either a method name (character scalar — same names as
#'   accepted by [tl_model()], supervised methods only) or a fitted
#'   `tidylearn_supervised` model. When given a fitted model, the
#'   advisor introspects its method and formula and advises on what a
#'   refit (or a fit on similar new data) would cost.
#' @param data A data frame. Required when `x` is a method name;
#'   optional when `x` is a fitted model (defaults to the model's
#'   training data).
#' @param formula Optional formula. Used to determine the number of
#'   effective predictors. Ignored when `x` is a fitted model.
#' @param hyperparams Named list of hyperparameters that affect runtime
#'   (e.g. `list(nrounds = 1000)` for xgboost, `list(epochs = 50,
#'   units = 256)` for deep learning). Missing entries fall back to
#'   per-method defaults.
#' @param gpu_check Optional `tidylearn_gpu_check` object. If omitted,
#'   `tl_check_gpu()` is called once internally.
#' @param ... Unused, reserved for method-specific extensions.
#'
#' @return An object of class `tidylearn_compute_advice` containing
#'   `problem`, `local_cpu`, `local_gpu`, `cloud`, `recommendation`,
#'   and `reasoning`. A `print()` method is provided.
#'
#' @examples
#' \dontrun{
#' advice <- tl_compute_advisor("xgboost", iris, Species ~ .,
#'                              hyperparams = list(nrounds = 1000))
#' print(advice)
#'
#' model <- tl_model(iris, Species ~ ., method = "xgboost")
#' tl_compute_advisor(model)
#' }
#' @export
tl_compute_advisor <- function(x, ...) {
  UseMethod("tl_compute_advisor")
}

#' @rdname tl_compute_advisor
#' @export
tl_compute_advisor.character <- function(x,
                                         data,
                                         formula = NULL,
                                         hyperparams = list(),
                                         gpu_check = NULL,
                                         ...) {
  method <- x
  if (length(method) != 1L || !nzchar(method)) {
    stop("'x' must be a single non-empty method name.", call. = FALSE)
  }
  if (missing(data) || !is.data.frame(data)) {
    stop("'data' must be a data frame.", call. = FALSE)
  }
  if (!(method %in% names(.tl_method_profiles))) {
    stop(
      "Method '", method, "' is not supported by the advisor.\n",
      "Supported methods: ",
      paste(names(.tl_method_profiles), collapse = ", "),
      call. = FALSE
    )
  }
  if (!is.list(hyperparams)) {
    stop("'hyperparams' must be a list.", call. = FALSE)
  }
  if (is.null(gpu_check)) {
    gpu_check <- tl_check_gpu()
  } else if (!inherits(gpu_check, "tidylearn_gpu_check")) {
    stop(
      "'gpu_check' must be a tidylearn_gpu_check object.",
      call. = FALSE
    )
  }

  n_rows <- nrow(data)
  n_cols <- tl_effective_p_internal(data, formula)
  est_size_mb <- (n_rows * n_cols * 8) / 1e6

  problem <- list(
    method      = method,
    rows        = n_rows,
    cols        = n_cols,
    est_size_mb = est_size_mb
  )

  local_cpu <- tl_estimate_local_cpu_internal(
    method, n_rows, n_cols, hyperparams
  )
  local_gpu <- tl_estimate_local_gpu_internal(
    method, n_rows, n_cols, hyperparams, gpu_check
  )
  cloud <- tl_estimate_cloud_internal(
    method, n_rows, n_cols, hyperparams, est_size_mb
  )

  rec <- tl_recommend_internal(local_cpu, local_gpu, cloud)

  structure(
    list(
      problem        = problem,
      local_cpu      = local_cpu,
      local_gpu      = local_gpu,
      cloud          = cloud,
      recommendation = rec$recommendation,
      reasoning      = rec$reasoning
    ),
    class = "tidylearn_compute_advice"
  )
}

#' @rdname tl_compute_advisor
#' @export
tl_compute_advisor.tidylearn_supervised <- function(x,
                                                    data = NULL,
                                                    hyperparams = list(),
                                                    gpu_check = NULL,
                                                    ...) {
  if (is.null(data)) {
    data <- x$data
  }
  tl_compute_advisor(
    x           = x$spec$method,
    data        = data,
    formula     = x$spec$formula,
    hyperparams = hyperparams,
    gpu_check   = gpu_check,
    ...
  )
}

#' @rdname tl_compute_advisor
#' @export
tl_compute_advisor.default <- function(x, ...) {
  stop(
    "tl_compute_advisor() expects either a method name (character) ",
    "or a fitted tidylearn_supervised model. Got an object of class: ",
    paste(class(x), collapse = "/"),
    ".",
    call. = FALSE
  )
}

# ---- internal helpers -------------------------------------------------

# Per-method runtime profile. cpu_const is seconds-per-complexity-unit,
# calibrated for a roughly 4-core modern laptop CPU. ram_mult is the
# expected peak memory as a multiple of the raw dataset size.
# gpu_speedup is the typical local-GPU vs CPU speedup for the method;
# 1 means no upstream GPU path.
.tl_method_profiles <- list(
  linear      = list(cpu_const = 5e-9,  ram_mult = 2, gpu_speedup = 1),
  polynomial  = list(cpu_const = 8e-9,  ram_mult = 2, gpu_speedup = 1),
  logistic    = list(cpu_const = 3e-8,  ram_mult = 2, gpu_speedup = 1),
  ridge       = list(cpu_const = 2e-8,  ram_mult = 3, gpu_speedup = 1),
  lasso       = list(cpu_const = 2e-8,  ram_mult = 3, gpu_speedup = 1),
  elastic_net = list(cpu_const = 2e-8,  ram_mult = 3, gpu_speedup = 1),
  tree        = list(cpu_const = 5e-8,  ram_mult = 3, gpu_speedup = 1),
  forest      = list(cpu_const = 1e-7,  ram_mult = 4, gpu_speedup = 1),
  boost       = list(cpu_const = 2e-7,  ram_mult = 4, gpu_speedup = 1),
  xgboost     = list(cpu_const = 1e-7,  ram_mult = 4, gpu_speedup = 5),
  nn          = list(cpu_const = 5e-7,  ram_mult = 3, gpu_speedup = 1),
  deep        = list(cpu_const = 2e-6,  ram_mult = 5, gpu_speedup = 15),
  svm         = list(cpu_const = 5e-7,  ram_mult = 6, gpu_speedup = 1)
)

# Returns the number of complexity units for a given method, given
# data dimensions and (optional) hyperparameters. Sensible defaults
# match the underlying package defaults where possible.
tl_method_complexity_internal <- function(method, n_rows, n_cols, hyp) {
  pull <- function(name, default) {
    val <- hyp[[name]]
    if (is.null(val) || !is.numeric(val) || length(val) != 1L) default else val
  }

  switch(
    method,
    "linear"      = n_rows * n_cols^2,
    "polynomial"  = n_rows * n_cols^2 * pull("degree", 2)^2,
    "logistic"    = n_rows * n_cols^2 * 5,
    "ridge"       = n_rows * n_cols * 100,
    "lasso"       = n_rows * n_cols * 100,
    "elastic_net" = n_rows * n_cols * 100,
    "tree"        = n_rows * log2(n_rows + 1) * n_cols,
    "forest"      = n_rows * sqrt(n_cols) * pull("ntree", 500) *
      log2(n_rows + 1),
    "boost"       = n_rows * n_cols * pull("n.trees", 100),
    "xgboost"     = n_rows * n_cols * pull("nrounds", 100),
    "nn"          = n_rows * n_cols * pull("size", 10) * pull("maxit", 100),
    "deep"        = n_rows * n_cols * pull("epochs", 10) * pull("units", 128),
    "svm"         = n_rows^2 * n_cols,
    n_rows * n_cols
  )
}

# Number of effective predictors for runtime estimation.
tl_effective_p_internal <- function(data, formula) {
  if (is.null(formula)) {
    return(max(ncol(data) - 1L, 1L))
  }
  if (!inherits(formula, "formula")) {
    formula <- stats::as.formula(formula)
  }

  rhs <- if (length(formula) == 3L) formula[[3]] else formula[[2]]
  rhs_vars <- all.vars(rhs)

  if (length(rhs_vars) == 0L || identical(rhs_vars, ".")) {
    return(max(ncol(data) - 1L, 1L))
  }
  length(rhs_vars)
}

# Local CPU runtime + RAM + feasibility heuristic.
tl_estimate_local_cpu_internal <- function(method, n_rows, n_cols, hyp) {
  profile <- .tl_method_profiles[[method]]
  complexity <- tl_method_complexity_internal(method, n_rows, n_cols, hyp)
  cores_raw <- parallel::detectCores(logical = FALSE)
  cores <- if (is.null(cores_raw) || is.na(cores_raw)) {
    1L
  } else {
    max(cores_raw, 1L)
  }
  est_seconds <- profile$cpu_const * complexity / cores
  est_peak_ram_mb <- (n_rows * n_cols * 8 / 1e6) * profile$ram_mult

  feasible <- est_peak_ram_mb < 16384  # heuristic ceiling: 16 GB laptop
  notes <- if (!feasible) {
    paste0(
      "Estimated peak RAM (~",
      format(round(est_peak_ram_mb), big.mark = ","),
      " MB) exceeds the 16 GB heuristic ceiling. May not fit on a ",
      "typical laptop."
    )
  } else {
    character(0)
  }

  list(
    est_seconds     = est_seconds,
    est_peak_ram_mb = est_peak_ram_mb,
    cores_used      = cores,
    feasible        = feasible,
    notes           = notes
  )
}

# Local GPU estimate. Only meaningful for methods with an upstream
# GPU path AND a detected GPU-capable backend.
tl_estimate_local_gpu_internal <- function(method, n_rows, n_cols, hyp,
                                           gpu_check) {
  profile <- .tl_method_profiles[[method]]
  has_gpu_path <- isTRUE(profile$gpu_speedup > 1)

  if (!has_gpu_path) {
    return(list(
      est_seconds = NA_real_,
      available   = FALSE,
      feasible    = FALSE,
      notes       = paste0(
        "Method '", method, "' has no upstream GPU path; local GPU does ",
        "not apply."
      )
    ))
  }

  capable <- tl_gpu_capable_methods(gpu_check)
  if (!(method %in% capable)) {
    return(list(
      est_seconds = NA_real_,
      available   = FALSE,
      feasible    = FALSE,
      notes       = paste0(
        "Method '", method, "' could use GPU, but no GPU-capable backend ",
        "was detected. See ?tl_check_gpu."
      )
    ))
  }

  cpu_est <- tl_estimate_local_cpu_internal(method, n_rows, n_cols, hyp)
  est_seconds <- cpu_est$est_seconds / profile$gpu_speedup

  list(
    est_seconds = est_seconds,
    available   = TRUE,
    feasible    = TRUE,
    notes       = paste0(
      "Assumes ", profile$gpu_speedup, "x speedup over local CPU. ",
      "Actual speedup varies with batch size and GPU model."
    )
  )
}

# Cloud (Modal) estimate. Stubbed in this slice — Modal integration
# is not yet wired up. Returns shape-correct values with configured
# = FALSE so the print method and recommendation logic can read it
# uniformly.
tl_estimate_cloud_internal <- function(method, n_rows, n_cols, hyp,
                                       est_size_mb) {
  profile <- .tl_method_profiles[[method]]
  has_gpu_path <- isTRUE(profile$gpu_speedup > 1)

  if (!has_gpu_path) {
    return(list(
      est_seconds    = NA_real_,
      est_cost_usd   = NA_real_,
      upload_seconds = NA_real_,
      configured     = FALSE,
      notes          = paste0(
        "Method '", method, "' has no upstream GPU path; cloud GPU ",
        "would not help."
      )
    ))
  }

  cpu_est <- tl_estimate_local_cpu_internal(method, n_rows, n_cols, hyp)
  # T4 ~ similar in throughput to a mid-range desktop GPU; assume 0.6x
  # of the locally-estimated GPU speedup to be conservative.
  cloud_gpu_seconds <- cpu_est$est_seconds / (profile$gpu_speedup * 0.6)
  cold_start_seconds <- 45
  upload_mbps <- 100 / 8  # 100 Mbps in MB/s
  upload_seconds <- est_size_mb / upload_mbps

  total_seconds <- cold_start_seconds + upload_seconds + cloud_gpu_seconds
  # Modal T4 list price ~$0.000164/sec at time of writing; cold-start
  # and upload time are roughly CPU-billed, treat as a flat $0.01.
  est_cost_usd <- cloud_gpu_seconds * 0.000164 + 0.01

  list(
    est_seconds    = total_seconds,
    est_cost_usd   = est_cost_usd,
    upload_seconds = upload_seconds,
    configured     = FALSE,
    notes          = paste(
      "Cloud integration is not yet configured in tidylearn. Estimates",
      "shown so users can see the tier's shape; actual submission is",
      "not yet supported."
    )
  )
}

# Pick a recommendation given the three tier estimates.
tl_recommend_internal <- function(cpu, gpu, cloud) {
  reasoning <- character(0)

  if (!isTRUE(cpu$feasible)) {
    reasoning <- c(reasoning, "Local CPU is infeasible by RAM heuristic.")
    if (isTRUE(gpu$available) && is_finite_num(gpu$est_seconds)) {
      return(list(
        recommendation = "gpu",
        reasoning = c(
          reasoning,
          "Local GPU path available; recommend GPU."
        )
      ))
    }
    if (isTRUE(cloud$configured)) {
      return(list(
        recommendation = "cloud",
        reasoning = c(reasoning, "Recommend cloud GPU.")
      ))
    }
    return(list(
      recommendation = "infeasible",
      reasoning = c(
        reasoning,
        "No GPU-capable backend detected and cloud is not configured."
      )
    ))
  }

  # CPU is feasible. Decide between cpu / gpu / cloud.
  cpu_seconds <- cpu$est_seconds

  if (cpu_seconds < 60) {
    return(list(
      recommendation = "cpu",
      reasoning = paste0(
        "Estimated local CPU runtime ~",
        format(round(cpu_seconds, 1), nsmall = 1),
        "s. Cloud cold-start alone would dominate; just run it locally."
      )
    ))
  }

  if (isTRUE(gpu$available) && is_finite_num(gpu$est_seconds)) {
    speedup <- cpu_seconds / gpu$est_seconds
    if (speedup >= 3 && gpu$est_seconds >= 5) {
      return(list(
        recommendation = "gpu",
        reasoning = paste0(
          "Local GPU is ~",
          format(round(speedup, 1), nsmall = 1),
          "x faster than CPU here and is already available."
        )
      ))
    }
  }

  if (isTRUE(cloud$configured) && is_finite_num(cloud$est_seconds)) {
    if (cloud$est_seconds < cpu_seconds / 3) {
      return(list(
        recommendation = "cloud",
        reasoning = paste0(
          "Cloud GPU saves an estimated ",
          format(round(cpu_seconds - cloud$est_seconds), big.mark = ","),
          "s vs. local CPU. Estimated cost: $",
          format(round(cloud$est_cost_usd, 2), nsmall = 2),
          "."
        )
      ))
    }
  }

  list(
    recommendation = "cpu",
    reasoning = paste0(
      "Estimated local CPU runtime ~",
      format(round(cpu_seconds), big.mark = ","),
      "s. No meaningfully faster tier available."
    )
  )
}

# Local helpers ---------------------------------------------------------

is_finite_num <- function(x) {
  is.numeric(x) && length(x) == 1L && is.finite(x)
}

#' Print method for `tidylearn_compute_advice` objects
#'
#' @param x A `tidylearn_compute_advice` object.
#' @param ... Unused.
#' @return The input `x`, invisibly.
#' @export
print.tidylearn_compute_advice <- function(x, ...) {
  cat("<tidylearn compute advice>\n")
  cat(
    "Problem:        ", x$problem$method,
    " on ", format(x$problem$rows, big.mark = ","), " rows x ",
    x$problem$cols, " cols ",
    "(~", format(round(x$problem$est_size_mb, 1), nsmall = 1), " MB)\n",
    sep = ""
  )
  cat("\n")
  cat("Tier estimates (order-of-magnitude):\n")

  fmt_seconds <- function(s) {
    if (!is_finite_num(s)) return("--")
    if (s < 60)   return(paste0(format(round(s, 1), nsmall = 1), "s"))
    if (s < 3600) return(paste0(format(round(s / 60, 1), nsmall = 1), "m"))
    paste0(format(round(s / 3600, 1), nsmall = 1), "h")
  }

  cat(sprintf(
    "  Local CPU:    %s   (peak RAM ~%s MB, %d cores)%s\n",
    fmt_seconds(x$local_cpu$est_seconds),
    format(round(x$local_cpu$est_peak_ram_mb), big.mark = ","),
    x$local_cpu$cores_used,
    if (!isTRUE(x$local_cpu$feasible)) "  [infeasible]" else ""
  ))
  cat(sprintf(
    "  Local GPU:    %s   %s\n",
    fmt_seconds(x$local_gpu$est_seconds),
    if (isTRUE(x$local_gpu$available)) "[available]" else "[not applicable]"
  ))
  cat(sprintf(
    "  Cloud GPU:    %s   (~$%s)   %s\n",
    fmt_seconds(x$cloud$est_seconds),
    if (is_finite_num(x$cloud$est_cost_usd)) {
      format(round(x$cloud$est_cost_usd, 2), nsmall = 2)
    } else {
      "--"
    },
    if (isTRUE(x$cloud$configured)) "[configured]" else "[not configured]"
  ))

  cat("\nRecommendation: ", x$recommendation, "\n", sep = "")
  if (length(x$reasoning) > 0L) {
    cat("\nReasoning:\n")
    cat(paste0("  - ", x$reasoning, "\n"), sep = "")
  }

  caveat_notes <- c(
    x$local_cpu$notes,
    x$local_gpu$notes,
    x$cloud$notes
  )
  caveat_notes <- caveat_notes[nzchar(caveat_notes)]
  if (length(caveat_notes) > 0L) {
    cat("\nNotes:\n")
    cat(paste0("  - ", caveat_notes, "\n"), sep = "")
  }

  invisible(x)
}
