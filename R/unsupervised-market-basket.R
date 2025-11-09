#' Tidy Apriori Algorithm
#'
#' Mine association rules using the Apriori algorithm with tidy output
#'
#' @param transactions A transactions object or data frame
#' @param support Minimum support (default: 0.01)
#' @param confidence Minimum confidence (default: 0.5)
#' @param minlen Minimum rule length (default: 2)
#' @param maxlen Maximum rule length (default: 10)
#' @param target Type of association mined: "rules" (default), "frequent itemsets", "maximally frequent itemsets"
#'
#' @return A list of class "tidy_rules" containing:
#' \itemize{
#'   \item rules_tbl: tibble of rules with lhs, rhs, and quality measures
#'   \item rules: original rules object
#'   \item parameters: parameters used
#' }
#'
#' @examples
#' \dontrun{
#' data("Groceries", package = "arules")
#'
#' # Basic apriori
#' rules <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'
#' # Access rules
#' rules$rules_tbl
#' }
#'
#' @export
tidy_apriori <- function(transactions, support = 0.01, confidence = 0.5,
                         minlen = 2, maxlen = 10, target = "rules") {

  # Set up parameters
  params <- list(
    supp = support,
    conf = confidence,
    minlen = minlen,
    maxlen = maxlen,
    target = target
  )

  # Run Apriori
  rules_obj <- arules::apriori(transactions, parameter = params)

  # Convert to tidy format if rules
  if (target == "rules") {
    rules_tbl <- tidy_rules(rules_obj)
  } else {
    rules_tbl <- NULL
  }

  result <- list(
    rules_tbl = rules_tbl,
    rules = rules_obj,
    parameters = params,
    n_rules = length(rules_obj)
  )

  class(result) <- c("tidy_apriori", "list")
  result
}


#' Convert Association Rules to Tidy Tibble
#'
#' @param rules A rules object from arules
#'
#' @return A tibble with one row per rule
#' @export
tidy_rules <- function(rules) {

  if (length(rules) == 0) {
    return(tibble::tibble())
  }

  # Extract LHS and RHS as character
  lhs <- arules::labels(arules::lhs(rules))
  rhs <- arules::labels(arules::rhs(rules))

  # Get quality measures
  quality_df <- arules::quality(rules)

  # Combine into tibble
  rules_tbl <- tibble::tibble(
    rule_id = seq_along(lhs),
    lhs = lhs,
    rhs = rhs
  ) %>%
    dplyr::bind_cols(tibble::as_tibble(quality_df))

  rules_tbl
}


#' Inspect Association Rules
#'
#' View rules sorted by various quality measures
#'
#' @param rules_obj A tidy_apriori object or rules object
#' @param by Sort by: "support", "confidence", "lift" (default), "count"
#' @param n Number of rules to display (default: 10)
#' @param decreasing Sort in decreasing order? (default: TRUE)
#'
#' @return A tibble of top rules
#' @export
inspect_rules <- function(rules_obj, by = "lift", n = 10, decreasing = TRUE) {

  # Handle different input types
  if (inherits(rules_obj, "tidy_apriori")) {
    rules_tbl <- rules_obj$rules_tbl
  } else if (inherits(rules_obj, "rules")) {
    rules_tbl <- tidy_rules(rules_obj)
  } else if (is.data.frame(rules_obj)) {
    rules_tbl <- rules_obj
  } else {
    stop("rules_obj must be a tidy_apriori object, rules object, or tibble")
  }

  # Sort and select top n
  if (by %in% names(rules_tbl)) {
    rules_tbl <- rules_tbl %>%
      dplyr::arrange(dplyr::desc(!!rlang::sym(by))) %>%
      dplyr::slice(1:min(n, nrow(rules_tbl)))

    if (!decreasing) {
      rules_tbl <- rules_tbl %>%
        dplyr::arrange(!!rlang::sym(by))
    }
  } else {
    warning("Sorting column not found, returning first n rules")
    rules_tbl <- rules_tbl %>% dplyr::slice(1:min(n, nrow(rules_tbl)))
  }

  rules_tbl
}


#' Filter Rules by Item
#'
#' Subset rules containing specific items
#'
#' @param rules_obj A tidy_apriori object or tibble of rules
#' @param item Character; item to filter by
#' @param where Character; "lhs", "rhs", or "both" (default: "both")
#'
#' @return A tibble of filtered rules
#' @export
filter_rules_by_item <- function(rules_obj, item, where = "both") {

  # Get rules tibble
  if (inherits(rules_obj, "tidy_apriori")) {
    rules_tbl <- rules_obj$rules_tbl
  } else {
    rules_tbl <- rules_obj
  }

  # Filter based on location
  if (where == "lhs") {
    filtered <- rules_tbl %>%
      dplyr::filter(grepl(item, lhs, fixed = TRUE))
  } else if (where == "rhs") {
    filtered <- rules_tbl %>%
      dplyr::filter(grepl(item, rhs, fixed = TRUE))
  } else {
    filtered <- rules_tbl %>%
      dplyr::filter(grepl(item, lhs, fixed = TRUE) | grepl(item, rhs, fixed = TRUE))
  }

  filtered
}


#' Find Related Items
#'
#' Find items frequently purchased with a given item
#'
#' @param rules_obj A tidy_apriori object
#' @param item Character; item to find associations for
#' @param min_lift Minimum lift threshold (default: 1.5)
#' @param top_n Number of top associations to return (default: 10)
#'
#' @return A tibble of related items with association metrics
#' @export
find_related_items <- function(rules_obj, item, min_lift = 1.5, top_n = 10) {

  # Get rules tibble
  if (inherits(rules_obj, "tidy_apriori")) {
    rules_tbl <- rules_obj$rules_tbl
  } else {
    rules_tbl <- rules_obj
  }

  # Filter rules containing the item
  related <- rules_tbl %>%
    dplyr::filter(grepl(item, lhs, fixed = TRUE) | grepl(item, rhs, fixed = TRUE)) %>%
    dplyr::filter(lift >= min_lift) %>%
    dplyr::arrange(dplyr::desc(lift)) %>%
    dplyr::slice(1:min(top_n, dplyr::n()))

  related
}


#' Summarize Association Rules
#'
#' Get summary statistics about rules
#'
#' @param rules_obj A tidy_apriori object or rules tibble
#'
#' @return A list with summary statistics
#' @export
summarize_rules <- function(rules_obj) {

  # Get rules tibble
  if (inherits(rules_obj, "tidy_apriori")) {
    rules_tbl <- rules_obj$rules_tbl
    n_rules <- rules_obj$n_rules
  } else {
    rules_tbl <- rules_obj
    n_rules <- nrow(rules_tbl)
  }

  if (n_rules == 0) {
    return(list(n_rules = 0, message = "No rules found"))
  }

  summary_list <- list(
    n_rules = n_rules,
    support = list(
      min = min(rules_tbl$support),
      max = max(rules_tbl$support),
      mean = mean(rules_tbl$support),
      median = stats::median(rules_tbl$support)
    ),
    confidence = list(
      min = min(rules_tbl$confidence),
      max = max(rules_tbl$confidence),
      mean = mean(rules_tbl$confidence),
      median = stats::median(rules_tbl$confidence)
    ),
    lift = list(
      min = min(rules_tbl$lift),
      max = max(rules_tbl$lift),
      mean = mean(rules_tbl$lift),
      median = stats::median(rules_tbl$lift)
    )
  )

  summary_list
}


#' Visualize Association Rules
#'
#' Create visualizations of association rules
#'
#' @param rules_obj A tidy_apriori object, rules object, or rules tibble
#' @param method Visualization method: "scatter" (default), "graph", "grouped", "paracoord"
#' @param top_n Number of top rules to visualize (default: 50)
#' @param ... Additional arguments passed to arulesViz::plot
#'
#' @return Visualization (side effect) or ggplot object
#' @export
visualize_rules <- function(rules_obj, method = "scatter", top_n = 50, ...) {

  # Get rules object
  if (inherits(rules_obj, "tidy_apriori")) {
    rules <- rules_obj$rules
  } else if (inherits(rules_obj, "rules")) {
    rules <- rules_obj
  } else if (is.data.frame(rules_obj)) {
    stop("Cannot visualize tibble directly; provide tidy_apriori or rules object")
  } else {
    stop("rules_obj must be a tidy_apriori or rules object")
  }

  # Subset to top n rules
  if (length(rules) > top_n) {
    rules <- head(rules, top_n)
  }

  # Create visualization based on method
  if (method == "scatter") {
    # Scatter plot with ggplot2
    rules_tbl <- tidy_rules(rules)

    p <- ggplot2::ggplot(rules_tbl, ggplot2::aes(x = support, y = confidence, color = lift, size = lift)) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::scale_color_gradient(low = "lightblue", high = "red") +
      ggplot2::labs(
        title = "Association Rules - Support vs Confidence",
        subtitle = sprintf("Top %d rules (colored by lift)", nrow(rules_tbl)),
        x = "Support",
        y = "Confidence"
      ) +
      ggplot2::theme_minimal()

    return(p)

  } else {
    # Use arulesViz for other methods
    arulesViz::plot(rules, method = method, ...)
  }
}


#' Generate Product Recommendations
#'
#' Get product recommendations based on basket contents
#'
#' @param rules_obj A tidy_apriori object
#' @param basket Character vector of items in current basket
#' @param top_n Number of recommendations to return (default: 5)
#' @param min_confidence Minimum confidence threshold (default: 0.5)
#'
#' @return A tibble with recommended items and metrics
#' @export
recommend_products <- function(rules_obj, basket, top_n = 5, min_confidence = 0.5) {

  # Get rules tibble
  if (inherits(rules_obj, "tidy_apriori")) {
    rules_tbl <- rules_obj$rules_tbl
  } else {
    rules_tbl <- rules_obj
  }

  # Find rules where LHS matches basket items
  recommendations <- rules_tbl %>%
    dplyr::filter(confidence >= min_confidence) %>%
    dplyr::filter(sapply(lhs, function(rule_lhs) {
      # Check if all items in LHS are in basket
      rule_items <- strsplit(gsub("[{}]", "", rule_lhs), ",")[[1]]
      rule_items <- trimws(rule_items)
      all(rule_items %in% basket)
    })) %>%
    dplyr::arrange(dplyr::desc(lift)) %>%
    dplyr::select(rhs, confidence, lift, support) %>%
    dplyr::slice(1:min(top_n, dplyr::n()))

  recommendations
}


#' Print Method for tidy_apriori
#'
#' @param x A tidy_apriori object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.tidy_apriori <- function(x, ...) {
  cat("Tidy Apriori Results\n")
  cat("====================\n\n")
  cat("Parameters:\n")
  cat("  Minimum support:   ", x$parameters$supp, "\n")
  cat("  Minimum confidence:", x$parameters$conf, "\n")
  cat("  Rule length:       ", x$parameters$minlen, "-", x$parameters$maxlen, "\n\n")

  cat("Results:\n")
  cat("  Number of rules:", x$n_rules, "\n\n")

  if (x$n_rules > 0) {
    summary <- summarize_rules(x)

    cat("Quality Measure Summary:\n")
    cat("  Support:    ", sprintf("%.4f - %.4f (mean: %.4f)",
                                  summary$support$min, summary$support$max, summary$support$mean), "\n")
    cat("  Confidence: ", sprintf("%.4f - %.4f (mean: %.4f)",
                                  summary$confidence$min, summary$confidence$max, summary$confidence$mean), "\n")
    cat("  Lift:       ", sprintf("%.2f - %.2f (mean: %.2f)",
                                  summary$lift$min, summary$lift$max, summary$lift$mean), "\n\n")

    cat("Top 5 rules by lift:\n")
    print(inspect_rules(x, by = "lift", n = 5))
  }

  cat("\nUse inspect_rules() to view more rules\n")
  cat("Use visualize_rules() to create visualizations\n")

  invisible(x)
}
