#' Detect local GPU availability for tidylearn methods
#'
#' Reports whether the local machine has a CUDA-capable GPU and which
#' tidylearn backends (`xgboost`, `keras`, `tensorflow`, `torch`) are
#' positioned to use it. Detection is intentionally cheap: it parses
#' `nvidia-smi` output and checks which R packages are installed, but
#' does not load Python or fit a model. A backend reported as
#' `gpu_likely_works = TRUE` may still fall back to CPU if it was not
#' compiled or configured with CUDA support — confirm with a small
#' real fit before relying on it for production workloads.
#'
#' Apple MPS (Metal Performance Shaders) is intentionally not detected
#' in this iteration; see the issue tracker for the MPS feature request.
#'
#' @param verbose Logical. If `TRUE`, also prints the result. Default `FALSE`.
#'
#' @return An object of class `tidylearn_gpu_check`: a list with components
#'   `any_gpu` (logical), `cuda` (driver info list with `driver_present`,
#'   `device_count`, `device_names`, `driver_version`), `backends` (per-backend
#'   status list each containing `installed`, `gpu_likely_works`, `notes`),
#'   and `messages` (character vector). A `print()` method is provided.
#'
#' @examples
#' \dontrun{
#' gpu <- tl_check_gpu()
#' print(gpu)
#' if (gpu$any_gpu && gpu$backends$xgboost$gpu_likely_works) {
#'   # xgboost with GPU is worth trying for this workload
#' }
#' }
#' @export
tl_check_gpu <- function(verbose = FALSE) {
  cuda <- tl_detect_cuda_internal()
  backends <- list(
    xgboost    = tl_check_backend_gpu("xgboost", cuda),
    tensorflow = tl_check_backend_gpu("tensorflow", cuda),
    keras      = tl_check_backend_gpu("keras", cuda),
    torch      = tl_check_backend_gpu("torch", cuda)
  )

  any_gpu <- isTRUE(cuda$driver_present) &&
    any(vapply(backends, function(b) isTRUE(b$gpu_likely_works), logical(1)))

  messages <- character(0)
  if (!isTRUE(cuda$driver_present)) {
    messages <- c(
      messages,
      "No NVIDIA CUDA driver detected. All GPU paths fall back to CPU."
    )
  } else if (!any_gpu) {
    messages <- c(
      messages,
      "CUDA driver present but no GPU-capable R backend is installed."
    )
  } else {
    messages <- c(
      messages,
      paste(
        "Detection is cheap: backends marked gpu_likely_works = TRUE may",
        "still fall back to CPU if not compiled with CUDA. Verify with a",
        "tiny real fit before relying on the GPU path."
      )
    )
  }

  result <- structure(
    list(
      any_gpu  = any_gpu,
      cuda     = cuda,
      backends = backends,
      messages = messages
    ),
    class = "tidylearn_gpu_check"
  )

  if (isTRUE(verbose)) {
    print(result)
  }
  invisible(result)
}

#' Parse nvidia-smi output into a CUDA driver descriptor
#'
#' Cheap probe — runs `nvidia-smi` if it exists on PATH, parses the device
#' name and driver version. Returns a sentinel list when `nvidia-smi` is
#' absent, errors, or produces no output. Does not load Python.
#'
#' @keywords internal
#' @noRd
tl_detect_cuda_internal <- function() {
  na_result <- list(
    driver_present = FALSE,
    device_count   = 0L,
    device_names   = character(0),
    driver_version = NA_character_
  )

  smi_path <- Sys.which("nvidia-smi")
  if (!nzchar(smi_path)) {
    return(na_result)
  }

  out <- tryCatch(
    suppressWarnings(
      system2(
        "nvidia-smi",
        args = c(
          "--query-gpu=name,driver_version",
          "--format=csv,noheader"
        ),
        stdout = TRUE, stderr = FALSE
      )
    ),
    error = function(e) NULL
  )

  if (is.null(out) || length(out) == 0L) {
    return(na_result)
  }

  parsed <- strsplit(out, ",\\s*")
  device_names <- vapply(parsed, function(p) p[1], character(1))
  driver_v <- if (length(parsed[[1]]) >= 2) parsed[[1]][2] else NA_character_

  list(
    driver_present = TRUE,
    device_count   = length(out),
    device_names   = trimws(device_names),
    driver_version = trimws(driver_v)
  )
}

#' Heuristic per-backend GPU check
#'
#' Returns `installed` + `gpu_likely_works` for a single backend package.
#' "Likely" because we only verify that the package is installed and a
#' CUDA driver is present — we do not try a real fit. Each backend has
#' its own configuration caveat which is captured in `notes`.
#'
#' @keywords internal
#' @noRd
tl_check_backend_gpu <- function(pkg, cuda) {
  installed <- requireNamespace(pkg, quietly = TRUE)
  if (!installed) {
    return(list(
      installed        = FALSE,
      gpu_likely_works = FALSE,
      notes            = paste0("'", pkg, "' is not installed.")
    ))
  }
  if (!isTRUE(cuda$driver_present)) {
    return(list(
      installed        = TRUE,
      gpu_likely_works = FALSE,
      notes            = "Installed, but no CUDA driver detected."
    ))
  }

  config_note <- switch(
    pkg,
    "xgboost"    = "GPU support requires xgboost compiled with CUDA.",
    "tensorflow" = paste(
      "GPU support requires a tensorflow build with CUDA + matching",
      "toolkit."
    ),
    "keras"      = paste(
      "GPU support requires the underlying TensorFlow backend with",
      "CUDA enabled."
    ),
    "torch"      = paste(
      "GPU support requires the torch R package to bundle CUDA libs."
    ),
    "GPU support depends on backend-specific configuration."
  )

  list(
    installed        = TRUE,
    gpu_likely_works = TRUE,
    notes            = paste("Installed and CUDA driver present.", config_note)
  )
}

#' Map tidylearn methods to backends that can offer a GPU path
#'
#' Used by the advisor and (eventually) by `tl_fit` to decide whether a
#' method has any GPU code path upstream. Methods absent from this map
#' are CPU-only by design (no upstream GPU support in their R package).
#'
#' If `gpu_check` is supplied, returns only methods whose backend reports
#' `gpu_likely_works = TRUE` in that check. Otherwise returns all methods
#' that have an upstream GPU path.
#'
#' @keywords internal
#' @noRd
tl_gpu_capable_methods <- function(gpu_check = NULL) {
  method_backend <- c(
    "xgboost" = "xgboost",
    "deep"    = "keras"
  )

  if (is.null(gpu_check)) {
    return(unname(names(method_backend)))
  }

  if (!inherits(gpu_check, "tidylearn_gpu_check")) {
    stop(
      "'gpu_check' must be a tidylearn_gpu_check object.",
      call. = FALSE
    )
  }

  is_capable <- vapply(names(method_backend), function(m) {
    b <- gpu_check$backends[[method_backend[[m]]]]
    isTRUE(b$gpu_likely_works)
  }, logical(1))

  unname(names(method_backend)[is_capable])
}

#' Print method for `tidylearn_gpu_check` objects
#'
#' @param x A `tidylearn_gpu_check` object.
#' @param ... Unused.
#' @return The input `x`, invisibly.
#' @export
print.tidylearn_gpu_check <- function(x, ...) {
  cat("<tidylearn GPU detection>\n")
  cat(
    "Any GPU:        ",
    if (isTRUE(x$any_gpu)) "yes" else "no",
    "\n",
    sep = ""
  )
  cat(
    "CUDA driver:    ",
    if (isTRUE(x$cuda$driver_present)) "present" else "absent",
    "\n",
    sep = ""
  )
  if (isTRUE(x$cuda$driver_present)) {
    cat("  Driver:       ", x$cuda$driver_version, "\n", sep = "")
    cat(
      "  Devices (",
      x$cuda$device_count,
      "): ",
      paste(x$cuda$device_names, collapse = ", "),
      "\n",
      sep = ""
    )
  }

  cat("\nBackends:\n")
  for (name in names(x$backends)) {
    b <- x$backends[[name]]
    status <- if (isTRUE(b$gpu_likely_works)) {
      "GPU likely"
    } else if (isTRUE(b$installed)) {
      "CPU only"
    } else {
      "not installed"
    }
    cat(sprintf("  %-11s %s\n", name, status))
  }

  if (length(x$messages) > 0L) {
    cat("\nNotes:\n")
    cat(paste0("  - ", x$messages, "\n"), sep = "")
  }

  invisible(x)
}
