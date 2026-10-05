#' Tidy Apriori Algorithm
#'
#' Mine association rules using the Apriori algorithm with tidy output
#'
#' @param transactions A transactions object or data frame
#' @param support Minimum support (default: 0.01)
#' @param confidence Minimum confidence (default: 0.5)
#' @param minlen Minimum rule length (default: 2)
#' @param maxlen Maximum rule length (default: 10)
#' @param target Type of association mined: "rules"
#'   (default), "frequent itemsets",
#'   "maximally frequent itemsets"
#' @param control A list of algorithmic controls for
#'   \code{\link[arules]{apriori}}. The mining trace is off unless the list
#'   sets \code{verbose = TRUE}; any other entries are passed as given.
#' @param ... Further arguments passed to \code{\link[arules]{apriori}}:
#'   \code{appearance}, to restrict where items may appear, or more mining
#'   parameters, such as \code{smax} or \code{maxtime}, which it adds to
#'   the ones above.
#'
#' @return A list of class "tidy_apriori" containing:
#' \itemize{
#'   \item rules_tbl: tibble of rules, as \code{\link{tidy_rules}} returns
#'     it, or \code{NULL} for an itemset target
#'   \item rules: original arules object, rules or itemsets
#'   \item parameters: parameters used
#'   \item n_rules: number of rules, or of itemsets for an itemset target
#'   \item itemsets_tbl: for an itemset target only, a tibble with
#'     \code{itemset_id}, \code{itemset} (its label), \code{size}, the
#'     quality measures, and \code{items}, a list column holding each
#'     itemset's items
#' }
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#' data("Groceries", package = "arules")
#'
#' # Basic apriori
#' rules <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'
#' # Access rules
#' rules$rules_tbl
#' }
#' }
#'
#' @export
tidy_apriori <- function(transactions, support = 0.01, confidence = 0.5,
                         minlen = 2, maxlen = 10, target = "rules",
                         control = list(verbose = FALSE), ...) {


  # Check if arules is installed
  tl_check_packages("arules")

  # Set up parameters
  params <- list(
    supp = support,
    conf = confidence,
    minlen = minlen,
    maxlen = maxlen,
    target = target
  )

  # arules prints about 20 lines of trace on every call by default. A
  # control list that leaves verbose out keeps it off here.
  if (is.list(control)) {
    control <- utils::modifyList(list(verbose = FALSE), control)
  }

  # Run Apriori
  rules_obj <- arules::apriori(
    transactions, parameter = params, control = control, ...
  )

  # Itemsets have no sides to tidy. rules_tbl stays NULL for them, and the
  # helpers that read it say so rather than fail on the NULL.
  is_rules <- inherits(rules_obj, "rules")

  result <- list(
    rules_tbl = if (is_rules) tidy_rules(rules_obj) else NULL,
    rules = rules_obj,
    parameters = params,
    n_rules = length(rules_obj)
  )
  if (!is_rules) {
    result$itemsets_tbl <- tl_tidy_itemsets(rules_obj)
  }

  class(result) <- c("tidy_apriori", "list")
  result
}


#' Convert Association Rules to Tidy Tibble
#'
#' @param rules A rules object from arules
#'
#' @return A tibble with columns \code{rule_id}, \code{lhs}, \code{rhs},
#'   the quality measures (e.g., \code{support}, \code{confidence},
#'   \code{lift}), and the list columns \code{lhs_items} and
#'   \code{rhs_items}, each rule's items on that side as a character
#'   vector. The \code{lhs} and \code{rhs} labels are for reading; the
#'   helpers that match items read the lists, since an item name can hold
#'   the comma that separates items in a label. An empty rule set gives a
#'   zero-row tibble with the same columns.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   rules_obj <- arules::apriori(Groceries,
#'     parameter = list(supp = 0.001, conf = 0.5))
#'   rules_tbl <- tidy_rules(rules_obj)
#' }
#' }
#'
#' @export
tidy_rules <- function(rules) {

  # Check if arules is installed
  tl_check_packages("arules")

  if (!inherits(rules, "rules")) {
    stop(
      "'rules' must be an arules rules object, such as the $rules element ",
      "of a tidy_apriori() result mined with target = \"rules\". Got an ",
      "object of class ", paste(class(rules), collapse = "/"), ".",
      call. = FALSE
    )
  }

  n_rules <- length(rules)

  # labels() of an empty rule set is "{}", one label for no rules, so the
  # sides are read only when there are rules. An empty set used to return
  # a tibble with no columns at all, which every helper downstream failed
  # on.
  if (n_rules == 0) {
    lhs <- character(0)
    rhs <- character(0)
    lhs_items <- list()
    rhs_items <- list()
  } else {
    lhs_side <- arules::lhs(rules)
    rhs_side <- arules::rhs(rules)
    lhs <- arules::labels(lhs_side)
    rhs <- arules::labels(rhs_side)
    lhs_items <- arules::LIST(lhs_side)
    rhs_items <- arules::LIST(rhs_side)
  }

  # Get quality measures
  quality_df <- arules::quality(rules)

  # Combine into tibble. The item lists go last, so a printed table still
  # leads with the labels and measures.
  rules_tbl <- tibble::tibble(
    rule_id = seq_len(n_rules),
    lhs = lhs,
    rhs = rhs
  ) |>
    dplyr::bind_cols(tibble::as_tibble(quality_df))
  rules_tbl$lhs_items <- lhs_items
  rules_tbl$rhs_items <- rhs_items

  rules_tbl
}


#' Inspect Association Rules
#'
#' View rules sorted by various quality measures
#'
#' @param rules_obj A tidy_apriori object, an arules rules or itemsets
#'   object, or a tibble of rules
#' @param by Sort by: "support", "confidence", "lift" (default), "count".
#'   Itemsets have no lift, so for them the default sorts by support.
#' @param n Number of rules to display (default: 10)
#' @param decreasing If TRUE (default), the \code{n} rules with the
#'   highest values of \code{by}, highest first; if FALSE, the \code{n}
#'   with the lowest, lowest first, as in arules' \code{head(by = )}.
#'
#' @return A tibble of the \code{n} rules ranked highest (or, with
#'   \code{decreasing = FALSE}, lowest) by the quality measure \code{by}.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   inspect_rules(res, by = "lift", n = 5)
#' }
#' }
#'
#' @export
inspect_rules <- function(rules_obj, by = "lift", n = 10, decreasing = TRUE) {

  # Handle different input types
  if (inherits(rules_obj, "tidy_apriori")) {
    rules_tbl <- if (inherits(rules_obj$rules, "rules")) {
      rules_obj$rules_tbl
    } else {
      rules_obj$itemsets_tbl %||% tl_tidy_itemsets(rules_obj$rules)
    }
  } else if (inherits(rules_obj, "rules")) {
    rules_tbl <- tidy_rules(rules_obj)
  } else if (inherits(rules_obj, "itemsets")) {
    rules_tbl <- tl_tidy_itemsets(rules_obj)
  } else if (is.data.frame(rules_obj)) {
    rules_tbl <- tl_ungroup(rules_obj)
  } else {
    stop("rules_obj must be a tidy_apriori object, rules object, or tibble")
  }

  if (missing(by) && !"lift" %in% names(rules_tbl)) {
    by <- "support"
  }

  # Sort in the requested direction before taking n, as arules' head(by = )
  # does. Sorting descending first and reversing afterwards returned the n
  # highest rules in ascending order.
  if (by %in% names(rules_tbl)) {
    rules_tbl <- if (decreasing) {
      dplyr::arrange(rules_tbl, dplyr::desc(.data[[by]]))
    } else {
      dplyr::arrange(rules_tbl, .data[[by]])
    }
  } else {
    warning("Sorting column not found, returning first n rules")
  }

  utils::head(rules_tbl, n)
}


#' Filter Rules by Item
#'
#' Subset rules containing specific items
#'
#' @param rules_obj A tidy_apriori object, an arules rules object, or a
#'   tibble of rules from \code{\link{tidy_rules}}, which must keep its
#'   \code{lhs_items} and \code{rhs_items} columns
#' @param item Character; one item name, matched against whole items, so
#'   "coffee" does not match "instant coffee"
#' @param where Character; "lhs", "rhs", or "both" (default: "both")
#'
#' @return A tibble of rules containing the specified \code{item} in the
#'   requested position.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   filter_rules_by_item(res, "whole milk", where = "rhs")
#' }
#' }
#'
#' @export
filter_rules_by_item <- function(rules_obj, item, where = "both") {

  rules_tbl <- tl_rules_table(rules_obj)
  item <- tl_check_item(item)

  in_lhs <- tl_rules_holding(rules_tbl, item, "lhs")
  in_rhs <- tl_rules_holding(rules_tbl, item, "rhs")

  # Filter based on location
  keep <- switch(where,
    lhs = in_lhs,
    rhs = in_rhs,
    in_lhs | in_rhs
  )

  rules_tbl[keep, , drop = FALSE]
}


#' Find Related Items
#'
#' Find items frequently purchased with a given item
#'
#' @param rules_obj A tidy_apriori object, an arules rules object, or a
#'   tibble of rules from \code{\link{tidy_rules}}, which must keep its
#'   \code{lhs_items} and \code{rhs_items} columns
#' @param item Character; one item name to find associations for, matched
#'   against whole items
#' @param min_lift Minimum lift threshold (default: 1.5)
#' @param top_n Number of top associations to return (default: 10)
#'
#' @return A tibble of rules involving the specified \code{item}, filtered by
#'   \code{min_lift} and sorted by lift in descending order.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   find_related_items(res, "whole milk", min_lift = 1.5)
#' }
#' }
#'
#' @export
find_related_items <- function(rules_obj, item, min_lift = 1.5, top_n = 10) {

  rules_tbl <- tl_rules_table(rules_obj)
  item <- tl_check_item(item)

  # Filter rules containing the item
  involves <- tl_rules_holding(rules_tbl, item, "lhs") |
    tl_rules_holding(rules_tbl, item, "rhs")

  related <- rules_tbl[
    involves & rules_tbl$lift >= min_lift, , drop = FALSE
  ] |>
    dplyr::arrange(dplyr::desc(.data$lift))

  utils::head(related, top_n)
}


#' Summarize Association Rules
#'
#' Get summary statistics about rules
#'
#' @param rules_obj A tidy_apriori object, an arules rules object, or a
#'   rules tibble
#'
#' @return A list with \code{n_rules} and summary statistics (\code{min},
#'   \code{max}, \code{mean}, \code{median}) for \code{support},
#'   \code{confidence}, and \code{lift}.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   summarize_rules(res)
#' }
#' }
#'
#' @export
summarize_rules <- function(rules_obj) {

  rules_tbl <- tl_rules_table(rules_obj)
  n_rules <- nrow(rules_tbl)

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
#' @param rules_obj A tidy_apriori object or an arules rules object. A
#'   table of rules is refused: the plots need the rules object.
#' @param method Visualization method: "scatter" (default), drawn by
#'   tidylearn, or a method of \pkg{arulesViz}'s \code{plot()}, such as
#'   "graph", "grouped", "matrix" or "paracoord"
#' @param top_n Number of rules to visualize, those with the highest lift
#'   (default: 50)
#' @param ... Additional arguments passed to plot() for rules visualization
#'
#' @return For \code{method = "scatter"}, a \code{\link[ggplot2]{ggplot}}
#'   object. Other methods return what \pkg{arulesViz}'s \code{plot()}
#'   returns: a ggplot object for "graph", "grouped" and "matrix", and for
#'   "paracoord", which draws with grid, a grid \code{vpPath}.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   visualize_rules(res, method = "scatter")
#' }
#' }
#'
#' @export
visualize_rules <- function(rules_obj, method = "scatter", top_n = 50, ...) {

  # Get rules object
  if (inherits(rules_obj, "tidy_apriori")) {
    rules <- rules_obj$rules
    if (!inherits(rules, "rules")) {
      tl_stop_itemsets(rules_obj)
    }
  } else if (inherits(rules_obj, "rules")) {
    rules <- rules_obj
  } else if (is.data.frame(rules_obj)) {
    stop(
      "Cannot visualize tibble directly; ",
      "provide tidy_apriori or rules object"
    )
  } else {
    stop("rules_obj must be a tidy_apriori or rules object")
  }

  # Keep the top_n rules by lift. utils::head() falls through to its
  # default method on an S4 rule set, which kept the first top_n in mining
  # order while the subtitle called them the top ones.
  by_lift <- order(arules::quality(rules)$lift, decreasing = TRUE)
  rules <- rules[utils::head(by_lift, top_n)]

  # Create visualization based on method
  if (method == "scatter") {
    # Scatter plot with ggplot2
    rules_tbl <- tidy_rules(rules)

    p <- ggplot2::ggplot(
      rules_tbl,
      ggplot2::aes(
        x = support, y = confidence,
        color = lift, size = lift
      )
    ) +
      ggplot2::geom_point(alpha = 0.6) +
      ggplot2::scale_color_gradient(low = "lightblue", high = "red") +
      ggplot2::labs(
        title = "Association Rules - Support vs Confidence",
        subtitle = if (nrow(rules_tbl) == 0) {
          "No rules to plot"
        } else {
          sprintf("Top %d rules by lift", nrow(rules_tbl))
        },
        x = "Support",
        y = "Confidence"
      ) +
      ggplot2::theme_minimal()

    p

  } else {
    # Use arulesViz for other methods
    # Check if arulesViz is available
    if (!requireNamespace("arulesViz", quietly = TRUE)) {
      stop(
        "Package 'arulesViz' is required for ",
        "this visualization method.",
        call. = FALSE
      )
    }
    plot(rules, method = method, ...)
  }
}


#' Generate Product Recommendations
#'
#' Get product recommendations based on basket contents
#'
#' @param rules_obj A tidy_apriori object, an arules rules object, or a
#'   tibble of rules from \code{\link{tidy_rules}}, which must keep its
#'   \code{lhs_items} and \code{rhs_items} columns
#' @param basket Character vector of items in current basket
#' @param top_n Number of recommendations to return (default: 5)
#' @param min_confidence Minimum confidence threshold (default: 0.5)
#'
#' @return A tibble with columns \code{rhs} (recommended item),
#'   \code{confidence}, \code{lift}, and \code{support}, sorted by lift in
#'   descending order. A rule is used when the basket holds its whole
#'   left-hand side and none of its right-hand side, and each product is
#'   listed once, from its highest-lift rule.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   # The basket has to cover the whole left-hand side of a rule, so a
#'   # basket of very common items usually matches nothing above the
#'   # confidence floor
#'   recommend_products(res, basket = c("flour", "baking powder"))
#' }
#' }
#'
#' @export
recommend_products <- function(rules_obj, basket,
                               top_n = 5,
                               min_confidence = 0.5) {

  rules_tbl <- tl_rules_table(rules_obj)
  basket <- as.character(basket)

  # A rule fires when the basket holds its whole left-hand side, and is
  # worth suggesting only for what the basket lacks: rules recommending an
  # item already in it used to be returned, one of them four times over
  fires <- vapply(
    tl_rule_items(rules_tbl, "lhs"),
    function(items) all(items %in% basket),
    logical(1), USE.NAMES = FALSE
  )
  adds <- vapply(
    tl_rule_items(rules_tbl, "rhs"),
    function(items) !any(items %in% basket),
    logical(1), USE.NAMES = FALSE
  )

  recommendations <- rules_tbl[
    fires & adds & rules_tbl$confidence >= min_confidence, , drop = FALSE
  ] |>
    dplyr::arrange(dplyr::desc(.data$lift))

  # Several rules can suggest the same product; keep its highest-lift rule
  recommendations <- recommendations[
    !duplicated(recommendations$rhs_items), , drop = FALSE
  ]

  recommendations |>
    dplyr::select("rhs", "confidence", "lift", "support") |>
    utils::head(top_n)
}


#' Print Method for tidy_apriori
#'
#' @param x A tidy_apriori object
#' @param ... Additional arguments (ignored)
#'
#' @return The input object \code{x}, returned invisibly.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("Groceries", package = "arules")
#'   res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
#'   print(res)
#' }
#' }
#'
#' @export
print.tidy_apriori <- function(x, ...) {
  # An itemset result has no rules table, confidence or lift. Printing it
  # as rules gave nine min/max warnings, "Support: Inf - -Inf", and then
  # failed on the missing table.
  is_rules <- inherits(x$rules, "rules")

  cat("Tidy Apriori Results\n")
  cat("====================\n\n")
  cat("Parameters:\n")
  cat("  Minimum support:   ", x$parameters$supp, "\n")
  if (is_rules) {
    cat("  Minimum confidence:", x$parameters$conf, "\n")
    cat("  Rule length:       ",
        x$parameters$minlen, "-",
        x$parameters$maxlen, "\n\n")
  } else {
    cat("  Target:            ", x$parameters$target, "\n")
    cat("  Itemset size:      ",
        x$parameters$minlen, "-",
        x$parameters$maxlen, "\n\n")
  }

  cat("Results:\n")

  if (!is_rules) {
    cat("  Number of itemsets:", x$n_rules, "\n\n")

    if (x$n_rules > 0) {
      itemsets_tbl <- x$itemsets_tbl %||% tl_tidy_itemsets(x$rules)
      cat("Support:",
          sprintf("%.4f - %.4f (mean: %.4f)",
                  min(itemsets_tbl$support),
                  max(itemsets_tbl$support),
                  mean(itemsets_tbl$support)), "\n\n")

      cat("Top 5 itemsets by support:\n")
      top <- inspect_rules(x, by = "support", n = 5)
      print(top[setdiff(names(top), "items")])
    }

    cat("\nUse inspect_rules() to view more itemsets\n")
    return(invisible(x))
  }

  cat("  Number of rules:", x$n_rules, "\n\n")

  if (x$n_rules > 0) {
    summary <- summarize_rules(x)

    cat("Quality Measure Summary:\n")
    cat("  Support:    ",
        sprintf("%.4f - %.4f (mean: %.4f)",
                summary$support$min,
                summary$support$max,
                summary$support$mean), "\n")
    cat("  Confidence: ",
        sprintf("%.4f - %.4f (mean: %.4f)",
                summary$confidence$min,
                summary$confidence$max,
                summary$confidence$mean), "\n")
    cat("  Lift:       ",
        sprintf("%.2f - %.2f (mean: %.2f)",
                summary$lift$min,
                summary$lift$max,
                summary$lift$mean), "\n\n")

    cat("Top 5 rules by lift:\n")
    top <- inspect_rules(x, by = "lift", n = 5)
    # The labels say what the item lists hold; the lists are for matching
    print(top[setdiff(names(top), c("lhs_items", "rhs_items"))])
  }

  cat("\nUse inspect_rules() to view more rules\n")
  cat("Use visualize_rules() to create visualizations\n")

  invisible(x)
}


# ---- helpers ---------------------------------------------------------

#' Tidy a set of frequent itemsets
#'
#' @param itemsets An arules itemsets object
#' @return A tibble with \code{itemset_id}, \code{itemset} (the label),
#'   \code{size}, the quality measures, and the list column \code{items}
#' @keywords internal
#' @noRd
tl_tidy_itemsets <- function(itemsets) {
  n_itemsets <- length(itemsets)

  # As for rules, labels() of an empty set is a single "{}"
  if (n_itemsets == 0) {
    labels <- character(0)
    items <- list()
  } else {
    labels <- arules::labels(itemsets)
    items <- arules::LIST(arules::items(itemsets))
  }

  itemsets_tbl <- tibble::tibble(
    itemset_id = seq_len(n_itemsets),
    itemset = labels,
    size = lengths(items)
  ) |>
    dplyr::bind_cols(tibble::as_tibble(arules::quality(itemsets)))
  itemsets_tbl$items <- items

  itemsets_tbl
}

#' The rules table a market-basket helper works on
#'
#' The helpers take a tidy_apriori() result, an arules rules object, or a
#' table of rules. A result saved before tidy_rules() added the item lists
#' has them rebuilt from the rules it carries.
#'
#' @param rules_obj What the caller passed
#' @return A tibble of rules
#' @keywords internal
#' @noRd
tl_rules_table <- function(rules_obj) {
  if (inherits(rules_obj, "tidy_apriori")) {
    if (!inherits(rules_obj$rules, "rules")) {
      tl_stop_itemsets(rules_obj)
    }
    rules_tbl <- rules_obj$rules_tbl
    if (!all(c("lhs_items", "rhs_items") %in% names(rules_tbl))) {
      rules_tbl <- tidy_rules(rules_obj$rules)
    }
    return(rules_tbl)
  }

  if (inherits(rules_obj, "rules")) {
    return(tidy_rules(rules_obj))
  }

  if (is.data.frame(rules_obj)) {
    return(tl_ungroup(rules_obj))
  }

  stop(
    "'rules_obj' must be a tidy_apriori() result, an arules rules object, ",
    "or a table of rules from tidy_rules(). Got an object of class ",
    paste(class(rules_obj), collapse = "/"), ".",
    call. = FALSE
  )
}

#' Refuse an itemset result where rules are needed
#'
#' An itemset result's rules table is NULL, which the rule helpers used to
#' fail on with messages that named neither the table nor the cause.
#'
#' @param rules_obj A tidy_apriori result mined for itemsets
#' @keywords internal
#' @noRd
tl_stop_itemsets <- function(rules_obj) {
  stop(
    "'rules_obj' holds itemsets (target = \"", rules_obj$parameters$target,
    "\"), not rules. Mine with target = \"rules\" for this; the itemsets ",
    "are in $itemsets_tbl, and inspect_rules() reads them.",
    call. = FALSE
  )
}

#' One side's item lists, for matching
#'
#' Matching reads the item lists tidy_rules() adds. grepl() on the labels
#' matched "coffee" inside "instant coffee", and splitting a label on ","
#' cut an item such as "salt, iodised" in two.
#'
#' @param rules_tbl A rules table
#' @param side "lhs" or "rhs"
#' @return A list of character vectors, one per rule
#' @keywords internal
#' @noRd
tl_rule_items <- function(rules_tbl, side) {
  if (!all(c("lhs_items", "rhs_items") %in% names(rules_tbl))) {
    stop(
      "Matching items needs the lhs_items and rhs_items columns that ",
      "tidy_rules() adds, and this table of rules lacks them. Pass the ",
      "tidy_apriori() result itself, or keep those columns when filtering ",
      "its rules_tbl.",
      call. = FALSE
    )
  }

  rules_tbl[[paste0(side, "_items")]]
}

#' Which rules hold an item on one side
#'
#' @param rules_tbl A rules table
#' @param item A single item name
#' @param side "lhs" or "rhs"
#' @return A logical vector, one per rule
#' @keywords internal
#' @noRd
tl_rules_holding <- function(rules_tbl, item, side) {
  vapply(
    tl_rule_items(rules_tbl, side),
    function(items) item %in% items,
    logical(1), USE.NAMES = FALSE
  )
}

#' Refuse anything but a single item name
#'
#' grepl() used only the first element of a vector, with a warning; item
#' matching on a vector would quietly mean "any of these" instead.
#'
#' @param item The caller's item
#' @return The item, as a character string
#' @keywords internal
#' @noRd
tl_check_item <- function(item) {
  if (is.factor(item)) {
    item <- as.character(item)
  }

  if (!is.character(item) || length(item) != 1 || is.na(item)) {
    got <- if (length(item) == 0) {
      "nothing"
    } else {
      paste(utils::head(as.character(item), 5), collapse = ", ")
    }
    stop(
      "'item' must be a single item name, such as \"whole milk\". Got: ",
      got, ".",
      call. = FALSE
    )
  }

  item
}
