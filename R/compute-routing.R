#' Resolve the effective compute tier for a tidylearn fit
#'
#' Internal plumbing called from `tl_model_supervised()` (and any future
#' direct fit entry points) to validate the user's `compute` argument,
#' consult the advisor when `compute = "auto"`, and apply the documented
#' fallback rules:
#'
#' * `compute = "gpu"` for a method without an upstream GPU path -> warn
#'   and fall back to `"cpu"`.
#' * `compute = "gpu"` when no GPU-capable backend is detected -> warn
#'   and fall back to `"cpu"`.
#' * `compute = "cloud"` -> error (not yet wired up).
#'
#' Returns one of `"cpu"` or `"gpu"`.
#'
#' @keywords internal
#' @noRd
tl_resolve_compute <- function(method, data, formula, compute = "cpu",
                               hyperparams = list()) {
  allowed <- c("auto", "cpu", "gpu", "cloud")
  if (length(compute) != 1L || !is.character(compute) ||
        !(compute %in% allowed)) {
    stop(
      "'compute' must be one of: ",
      paste0("'", allowed, "'", collapse = ", "),
      ". Got: ", deparse(compute),
      call. = FALSE
    )
  }

  if (compute == "cloud") {
    stop(
      "Cloud compute is not yet wired up in this version of tidylearn. ",
      "Use compute = 'auto', 'cpu', or 'gpu'. The cloud tier will land ",
      "in a follow-up release.",
      call. = FALSE
    )
  }

  if (compute == "cpu") {
    return("cpu")
  }

  # Both "auto" and "gpu" need to know whether the method even has a
  # GPU path upstream.
  method_has_gpu_path <- method %in% tl_gpu_capable_methods(NULL)

  if (compute == "gpu" && !method_has_gpu_path) {
    warning(
      "Method '", method, "' has no upstream GPU path. ",
      "Falling back to CPU.",
      call. = FALSE
    )
    return("cpu")
  }

  # For "auto" on a CPU-only method, skip the (expensive) advisor call
  # and stay on CPU. No GPU detection needed.
  if (compute == "auto" && !method_has_gpu_path) {
    return("cpu")
  }

  # Method has a GPU path. Check if a GPU is actually available.
  gpu_check <- tl_check_gpu()
  capable_now <- tl_gpu_capable_methods(gpu_check)
  gpu_available <- method %in% capable_now

  if (compute == "gpu" && !gpu_available) {
    warning(
      "compute = 'gpu' requested but no GPU-capable backend was ",
      "detected for method '", method, "'. Falling back to CPU. ",
      "See ?tl_check_gpu.",
      call. = FALSE
    )
    return("cpu")
  }

  if (compute == "gpu") {
    return("gpu")
  }

  # compute == "auto" with a GPU-capable method. Defer to the advisor.
  advice <- tl_compute_advisor(
    method,
    data        = data,
    formula     = formula,
    hyperparams = hyperparams,
    gpu_check   = gpu_check
  )

  if (advice$recommendation == "gpu" && gpu_available) {
    message(
      "compute = 'auto' chose 'gpu' for method '", method, "'."
    )
    return("gpu")
  }

  message(
    "compute = 'auto' chose 'cpu' for method '", method, "'."
  )
  "cpu"
}
