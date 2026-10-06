#' @title Regularization Functions for tidylearn
#' @name tidylearn-regularization
#' @description Ridge, Lasso, and Elastic Net regularization
#'   functionality
#' @importFrom glmnet glmnet cv.glmnet predict.glmnet
#' @importFrom stats model.matrix as.formula
#' @importFrom tibble tibble
NULL

#' Fit a Ridge regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso,
#'   between 0-1 for Elastic Net)
#' @param lambda Regularization parameter: a single penalty, or NULL or a
#'   sequence of penalties for cross-validation to choose from
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet() or cv.glmnet()
#' @return A fitted Ridge regression model
#' @keywords internal
tl_fit_ridge <- function(data, formula,
                         is_classification = FALSE,
                         alpha = 0, lambda = NULL,
                         cv_folds = 5, ...) {
  tl_fit_regularized(
    data, formula, is_classification,
    alpha, lambda, cv_folds, ...
  )
}

#' Fit a Lasso regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso,
#'   between 0-1 for Elastic Net)
#' @param lambda Regularization parameter: a single penalty, or NULL or a
#'   sequence of penalties for cross-validation to choose from
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet() or cv.glmnet()
#' @return A fitted Lasso regression model
#' @keywords internal
tl_fit_lasso <- function(data, formula,
                         is_classification = FALSE,
                         alpha = 1, lambda = NULL,
                         cv_folds = 5, ...) {
  tl_fit_regularized(
    data, formula, is_classification,
    alpha, lambda, cv_folds, ...
  )
}

#' Fit an Elastic Net regression model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (default: 0.5 for
#'   Elastic Net)
#' @param lambda Regularization parameter: a single penalty, or NULL or a
#'   sequence of penalties for cross-validation to choose from
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet() or cv.glmnet()
#' @return A fitted Elastic Net regression model
#' @keywords internal
tl_fit_elastic_net <- function(data, formula,
                               is_classification = FALSE,
                               alpha = 0.5,
                               lambda = NULL,
                               cv_folds = 5, ...) {
  tl_fit_regularized(
    data, formula, is_classification,
    alpha, lambda, cv_folds, ...
  )
}

#' Fit a regularized regression model
#'
#' Fits Ridge, Lasso, or Elastic Net regularization.
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param is_classification Logical indicating if this is a
#'   classification problem
#' @param alpha Mixing parameter (0 for Ridge, 1 for Lasso,
#'   between 0-1 for Elastic Net)
#' @param lambda Regularization parameter: a single penalty to fit at, or
#'   \code{NULL} (the default) or a sequence of penalties, from which
#'   cross-validation chooses one
#' @param cv_folds Number of folds for cross-validation
#'   (default: 5)
#' @param ... Additional arguments to pass to glmnet() or cv.glmnet(). A
#'   name neither function takes is an error, as are \code{x}, \code{y},
#'   \code{family} and \code{nfolds}, which tidylearn sets itself.
#' @param weights Optional case weights, one per row of \code{data}
#' @param foldid Optional fold for each row of \code{data}, for the
#'   cross-validation that chooses lambda
#' @param subset Optional rows of \code{data} to fit on
#' @param offset Not supported: glmnet would need the offset again at
#'   every prediction. An error if supplied.
#' @return A fitted regularized regression model
#' @keywords internal
tl_fit_regularized <- function(data, formula,
                               is_classification = FALSE,
                               alpha = 0, lambda = NULL,
                               cv_folds = 5, ...,
                               weights = NULL, foldid = NULL,
                               subset = NULL, offset = NULL) {
  # Check if glmnet is installed
  tl_check_packages("glmnet")

  # One penalty is fitted as given. The default path, or a sequence of
  # penalties, is cross-validated so that a single one is chosen: fitted
  # along a sequence with none chosen, the model left predict() nothing to
  # use.
  cross_validate <- is.null(lambda) || length(lambda) > 1L

  tl_check_glmnet_args(names2(list(...)), cross_validate, !is.null(foldid))

  if (!is.null(offset)) {
    stop(
      "ridge, lasso and elastic_net do not take an offset: glmnet needs it\n",
      "again at every prediction, and predict() has no way to supply it. Fit\n",
      "method = \"linear\" or \"logistic\" with offset(<column>) in the ",
      "formula for a\nmodel with an offset.",
      call. = FALSE
    )
  }

  # Parse the formula
  response_var <- all.vars(formula)[1]

  # The per-row arguments go into the model frame, so the rows na.omit and
  # subset remove are removed from them as well. Kept beside it, they still
  # had every row, and one missing predictor left glmnet with 32 weights for
  # 31 rows. do.call() puts the values themselves in the call: model.frame()
  # looks a name up in the data and the formula's environment, neither of
  # which holds this function's variables.
  frame_args <- list(formula = formula, data = data)
  frame_args$weights <- weights
  frame_args$foldid <- foldid
  frame_args$subset <- subset
  model_frame <- do.call(stats::model.frame, frame_args)
  design_terms <- stats::terms(model_frame)

  offset_term <- attr(design_terms, "offset")
  if (!is.null(offset_term)) {
    stop(
      "ridge, lasso and elastic_net cannot use the formula's ",
      deparse1(attr(design_terms, "variables")[[offset_term[1] + 1L]]),
      ".\nglmnet would need it again at every prediction, and fitting ",
      "without it would\nignore part of the formula. Fit method = \"linear\" ",
      "or \"logistic\" for a model\nwith an offset, or make the variable a ",
      "predictor.",
      call. = FALSE
    )
  }

  # glmnet fits its own intercept, so drop R's intercept column -- by name.
  # Dropping the first column took a predictor instead when the formula has
  # none, as in y ~ x1 + x2 - 1.
  design <- stats::model.matrix(design_terms, model_frame)
  x_mat <- design[, colnames(design) != "(Intercept)", drop = FALSE]

  # glmnet's own message, "x should be a matrix with 2 or more columns",
  # names an argument the caller never passed
  if (ncol(x_mat) < 2L) {
    stop(
      "ridge, lasso and elastic_net need at least two predictor columns, ",
      "but\n", deparse1(formula), " gives ",
      if (ncol(x_mat) == 0L) {
        "none"
      } else {
        paste0("one (", colnames(x_mat), ")")
      },
      ". Add a predictor, or fit an unpenalised model:\nmethod = \"linear\" ",
      "for a numeric response, \"logistic\" for two classes.",
      call. = FALSE
    )
  }

  # Take the response from the model frame rather than from `data`.
  # model.frame() applies na.omit, so one missing predictor drops that row
  # from x_mat while data[[response_var]] still holds every row -- and
  # glmnet then reports "number of observations in y (60) not equal to the
  # number of rows of x (59)", which names neither missing values nor the
  # column responsible. lm(), rpart(), nnet() and svm() all drop the row
  # and carry on; this now does the same.
  y <- stats::model.response(model_frame)
  case_weights <- stats::model.weights(model_frame)
  fold_ids <- model_frame[["(foldid)"]]

  # Retain the terms and factor levels so prediction can rebuild an
  # identically-coded design matrix on new data
  design_xlevels <- stats::.getXlevels(design_terms, model_frame)

  # Determine the appropriate family based on problem type
  if (is_classification) {
    # The classes of the rows being fitted. A class whose every row has a
    # missing value is gone from the model frame but stays a level, and
    # glmnet stops on "one multinomial or binomial class has 1 or 0
    # observations; not allowed".
    y <- tl_normalise_response(y)
    if (nlevels(y) < 2L) {
      stop(
        "ridge, lasso and elastic_net need rows of at least two classes, ",
        "but\nthe rows left once those with missing values are dropped are ",
        "all\n'", levels(y), "'.",
        call. = FALSE
      )
    }

    if (length(levels(y)) == 2) {
      # Binary classification
      family <- "binomial"
    } else {
      # Multiclass classification
      family <- "multinomial"
    }
  } else {
    # Regression
    family <- "gaussian"
  }

  # Fit the model
  if (cross_validate) {
    # Use cross-validation to select optimal lambda. weights and foldid are
    # NULL when not given, which is glmnet's own default.
    cv_model <- glmnet::cv.glmnet(
      x = x_mat,
      y = y,
      weights = case_weights,
      foldid = fold_ids,
      alpha = alpha,
      family = family,
      nfolds = cv_folds,
      lambda = lambda,
      ...
    )

    # Extract optimal lambda
    lambda_min <- cv_model$lambda.min
    lambda_1se <- cv_model$lambda.1se

    # cv.glmnet() has already fitted the full data along this path, and
    # coef() and predict() on cv_model read that fit. A second glmnet() run
    # at lambda = cv_model$lambda agreed with it only to the solver's
    # tolerance, and at the largest penalty -- where glmnet's own path has
    # every slope exactly zero -- it left a floating-point residue such as
    # -6e-17 on one, which importance reported as the most important
    # predictor.
    model <- cv_model$glmnet.fit

    # Store cross-validation results
    attr(model, "cv_results") <- cv_model
    attr(model, "lambda_min") <- lambda_min
    attr(model, "lambda_1se") <- lambda_1se
  } else {
    # Fit model with user-specified lambda
    model <- glmnet::glmnet(
      x = x_mat,
      y = y,
      weights = case_weights,
      alpha = alpha,
      family = family,
      lambda = lambda,
      ...
    )

    # Store lambda
    attr(model, "lambda_min") <- lambda
    attr(model, "lambda_1se") <- lambda
  }

  # Store formula, variables, and alpha
  attr(model, "formula") <- formula
  attr(model, "response_var") <- response_var
  attr(model, "alpha") <- alpha
  attr(model, "is_classification") <- is_classification

  # Design specification used at fit time
  attr(model, "tl_terms") <- design_terms
  attr(model, "tl_xlevels") <- design_xlevels
  attr(model, "tl_colnames") <- colnames(x_mat)
  # Each design column's standard deviation over the rows fitted, in design
  # order, for importance to scale the coefficients by. Recomputed from
  # model$data it would include the rows dropped for a missing value.
  attr(model, "tl_x_sd") <- apply(x_mat, 2, stats::sd)
  if (is_classification) {
    attr(model, "response_levels") <- levels(y)
  }

  model
}

#' Refuse arguments that glmnet would ignore
#'
#' \code{glmnet()} and \code{cv.glmnet()} both take \code{...} and discard
#' names they do not know, so a misspelt \code{standardise = FALSE} changed
#' nothing while the model's \code{$spec$args} recorded it as used. A
#' cross-validation argument at a single penalty reached \code{glmnet()}
#' alone and went the same way. The arguments tidylearn sets itself are
#' refused as well, with what sets them.
#'
#' @param arg_names Names of the arguments in \code{...}.
#' @param cross_validate Whether \code{cv.glmnet()} will run.
#' @param foldid_given Whether \code{foldid}, a formal of the caller, was
#'   supplied.
#' @return \code{TRUE}, invisibly, when every argument will be used.
#' @keywords internal
#' @noRd
tl_check_glmnet_args <- function(arg_names, cross_validate, foldid_given) {
  arg_names <- arg_names[arg_names != ""]
  fit_args <- setdiff(names(formals(glmnet::glmnet)), "...")
  cv_args <- setdiff(names(formals(glmnet::cv.glmnet)), "...")

  # Passed again beside tidylearn's own value, these failed with R's
  # "formal argument matched by multiple actual arguments"
  set_here <- c(
    x = "tidylearn sets 'x' and 'y' from the formula and data",
    y = "tidylearn sets 'x' and 'y' from the formula and data",
    family = paste0(
      "tidylearn sets 'family' from the response: gaussian for a numeric ",
      "response,\nbinomial for two classes, multinomial for more"
    ),
    nfolds = "tidylearn sets 'nfolds' from cv_folds; pass cv_folds instead"
  )
  owned <- intersect(arg_names, names(set_here))
  if (length(owned) > 0) {
    stop(
      paste(unique(set_here[owned]), collapse = ".\n"), ".",
      call. = FALSE
    )
  }

  unknown <- setdiff(arg_names, c(fit_args, cv_args))
  if (length(unknown) > 0) {
    stop(
      "glmnet::glmnet() and glmnet::cv.glmnet() have no ",
      if (length(unknown) == 1L) "argument" else "arguments", " named ",
      paste0("'", unknown, "'", collapse = ", "),
      ".\nridge, lasso and elastic_net would ignore ",
      if (length(unknown) == 1L) "it" else "them",
      " without saying so. Check the spelling\nagainst ?glmnet::glmnet and ",
      "?glmnet::cv.glmnet.",
      call. = FALSE
    )
  }

  if (!cross_validate) {
    cv_only <- intersect(
      c(arg_names, if (foldid_given) "foldid"),
      setdiff(cv_args, fit_args)
    )
    if (length(cv_only) > 0) {
      stop(
        paste0("'", cv_only, "'", collapse = ", "),
        if (length(cv_only) == 1L) " only applies" else " only apply",
        " to the cross-validation that chooses lambda, and none\nruns when ",
        "a single lambda is given. Leave lambda = NULL, or give a sequence ",
        "of\npenalties, to have one chosen by cross-validation.",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}

#' Plot regularization path for a regularized model
#'
#' @param model A tidylearn regularized model object
#' @param label_n Number of top features to label
#'   (default: 5)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "lasso")
#' tl_plot_regularization_path(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_line
#'   scale_x_log10 labs theme_minimal
#' @export
tl_plot_regularization_path <- function(model,
                                        label_n = 5,
                                        ...) {
  # Extract the glmnet model
  fit <- model$fit

  # A path runs through the penalties fitted. At a single penalty each term
  # was one point, so no line was drawn.
  if (length(fit$lambda) < 2L) {
    stop(
      "The regularization path needs a model fitted along several ",
      "penalties,\nbut this one was fitted at the single penalty lambda = ",
      signif(fit$lambda, 4), ".\nFit it with lambda = NULL, or a sequence ",
      "of penalties, to draw a path;\ntl_coefficients() gives the ",
      "coefficients at this one.",
      call. = FALSE
    )
  }
  # Every model fitted along several penalties is cross-validated now. One
  # fitted by an earlier version may not be, and has no lambda.min or
  # lambda.1se to mark.
  cross_validated <- !is.null(attr(fit, "cv_results"))

  # One row per term and penalty. A multinomial fit has a path per class,
  # and its coef() is a list of matrices that as.matrix() could not use.
  coef_df <- tl_glmnet_path_tbl(fit)
  by_class <- "class" %in% names(coef_df)

  # Identify the top features (by max absolute coef), within each class.
  # Paths are told apart by term_id rather than by name: a factor a with
  # level b and a numeric column ab both make a term called ab.
  top <- coef_df |>
    dplyr::group_by(
      dplyr::across(dplyr::any_of(c("class", "term_id", "feature")))
    ) |>
    dplyr::summarize(
      max_abs_coef = max(abs(.data$coefficient)),
      .groups = "drop"
    ) |>
    dplyr::arrange(dplyr::desc(.data$max_abs_coef), .data$feature) |>
    dplyr::group_by(dplyr::across(dplyr::any_of("class"))) |>
    dplyr::slice_head(n = label_n) |>
    dplyr::ungroup()

  # Mark top features for labeling
  path_key <- function(df) {
    paste(if (by_class) df$class else "", df$term_id)
  }
  coef_df$is_top <- path_key(coef_df) %in% path_key(top)

  # Get optimal lambda values
  lambda_min <- attr(fit, "lambda_min")
  lambda_1se <- attr(fit, "lambda_1se")

  smallest_lambda <- min(fit$lambda)
  label_data <- coef_df[coef_df$is_top & coef_df$lambda == smallest_lambda, ,
                        drop = FALSE]

  # Room on the left for the labels, which sit inside the panel left of the
  # smallest penalty. A fixed left expansion of 0.16 clipped long names --
  # Speciesversicolor lost its first letters at 7 x 5 inches -- so the room
  # grows with the longest label: about 0.07 in per character at the label
  # size, on a plot about 7 in wide, whose panel is about 6.3 in. A left
  # expansion of a gives the labels a / (1 + a + 0.02) of the panel. The
  # 0.16 floor is what names as short as mtcars' need, and keeps their
  # layout.
  label_share <- (0.07 * max(nchar(label_data$feature), 0L) + 0.05) / 6.3
  left_expand <- if (label_share >= 0.6) {
    1.5
  } else {
    min(1.5, max(0.16, label_share * 1.02 / (1 - label_share)))
  }

  # Create the plot
  p <- ggplot2::ggplot(
    coef_df,
    ggplot2::aes(
      x = .data$lambda,
      y = .data$coefficient,
      group = .data$term_id,
      color = .data$is_top
    )
  ) +
    ggplot2::geom_line(
      # `size` on a line is deprecated since ggplot2 3.4.0
      ggplot2::aes(alpha = .data$is_top, linewidth = .data$is_top)
    ) +
    ggplot2::scale_x_log10(
      expand = ggplot2::expansion(mult = c(left_expand, 0.02))
    ) +
    # Named, so each path takes the values for whether it is labelled. By
    # position, TRUE took the first value whenever every path was labelled,
    # so a model with label_n or fewer predictors was drawn grey and thin.
    ggplot2::scale_alpha_manual(
      values = c("FALSE" = 0.3, "TRUE" = 1)
    ) +
    ggplot2::scale_linewidth_manual(
      values = c("FALSE" = 0.5, "TRUE" = 1.2)
    ) +
    ggplot2::scale_color_manual(
      values = c("FALSE" = "gray", "TRUE" = "steelblue")
    ) +
    # On a plot narrower than the room was sized for, a long label runs
    # into the axis area rather than losing letters at the panel edge
    ggplot2::coord_cartesian(clip = "off") +
    ggplot2::labs(
      title = "Regularization Path",
      subtitle = if (cross_validated) {
        "Blue: lambda.min, Red: lambda.1se"
      } else {
        "No cross-validation ran, so no lambda.min or lambda.1se is marked"
      },
      x = "Lambda (log scale)",
      y = "Coefficients"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none")

  if (cross_validated) {
    p <- p +
      ggplot2::geom_vline(
        xintercept = lambda_min,
        linetype = "dashed",
        color = "blue"
      ) +
      ggplot2::geom_vline(
        xintercept = lambda_1se,
        linetype = "dashed",
        color = "red"
      )
  }

  if (by_class) {
    # A panel per class, each on its own scale
    p <- p + ggplot2::facet_wrap(
      ggplot2::vars(.data$class), ncol = 1, scales = "free_y"
    )
  }

  # Label each path where it starts, at the smallest lambda -- the left
  # of a log axis, not the right, which the previous comment here had
  # backwards along with the `hjust` that followed from it.
  if (label_n > 0) {
    # Two coefficients can start arbitrarily close together -- on mtcars
    # two of the five are within 0.02 -- so the labels printed on top of
    # each other. Draw them at spread positions and connect each to its
    # own line, so a moved label still says which path it belongs to. Each
    # class's labels are spread within its own panel.
    panel_of <- function(df) {
      if (by_class) as.character(df$class) else rep("", nrow(df))
    }
    label_data$label_y <- label_data$coefficient
    for (panel in unique(panel_of(label_data))) {
      rows <- panel_of(label_data) == panel
      span <- diff(range(coef_df$coefficient[panel_of(coef_df) == panel]))
      label_data$label_y[rows] <- tl_spread_labels(
        label_data$coefficient[rows],
        min_gap = 0.07 * span
      )
    }

    # Sitting on the lines they were also unreadable, inheriting the same
    # colour as the path behind them. Put them outside the leftmost point
    # and give the panel room to hold them.
    p <- p +
      ggplot2::geom_segment(
        data = label_data,
        ggplot2::aes(
          x = smallest_lambda * 0.82,
          xend = smallest_lambda,
          y = .data$label_y,
          yend = .data$coefficient
        ),
        colour = "grey70", linewidth = 0.3, inherit.aes = FALSE
      ) +
      ggplot2::geom_text(
        data = label_data,
        ggplot2::aes(
          label = .data$feature,
          x = smallest_lambda * 0.78,
          y = .data$label_y
        ),
        hjust = 1, vjust = 0.5, size = 3, colour = "grey25",
        inherit.aes = FALSE
      )
  }

  p
}

#' A glmnet fit's coefficient paths, as a tibble
#'
#' @param fit A glmnet fit.
#' @return A tibble with one row per term and penalty: \code{term_id} (the
#'   term's position, which a repeated name cannot confuse), \code{feature},
#'   \code{lambda} and \code{coefficient}, led by a \code{class} factor for
#'   a multinomial fit. The intercept is left out.
#' @keywords internal
#' @noRd
tl_glmnet_path_tbl <- function(fit) {
  one_class <- function(coefs) {
    coefs <- as.matrix(coefs)
    coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
    tibble::tibble(
      term_id = rep(seq_len(nrow(coefs)), times = ncol(coefs)),
      feature = rep(rownames(coefs), times = ncol(coefs)),
      lambda = rep(fit$lambda, each = nrow(coefs)),
      coefficient = as.vector(coefs)
    )
  }

  coefs <- stats::coef(fit)
  if (!is.list(coefs)) {
    return(one_class(coefs))
  }

  per_class <- lapply(names(coefs), function(class_name) {
    tibble::add_column(one_class(coefs[[class_name]]),
                       class = class_name, .before = 1)
  })
  paths <- dplyr::bind_rows(per_class)
  # In the order of the classes, so the panels are too
  paths$class <- factor(paths$class, levels = names(coefs))
  paths
}

#' Plot cross-validation results for a regularized model
#'
#' Shows the cross-validation error as a function of
#' lambda for ridge, lasso, or elastic net models fitted
#' with cv.glmnet.
#'
#' @param model A tidylearn regularized model object
#'   (ridge, lasso, or elastic_net)
#' @param ... Additional arguments (currently unused)
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "ridge")
#' tl_plot_regularization_cv(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_point geom_line
#'   geom_ribbon scale_x_log10 labs theme_minimal
#' @export
tl_plot_regularization_cv <- function(model, ...) {
  # Extract the glmnet model
  fit <- model$fit

  # Check if cross-validation results are available
  cv_results <- attr(fit, "cv_results")
  if (is.null(cv_results)) {
    stop(
      "Cross-validation results not available. ",
      "Model must be fitted with lambda = NULL or a sequence of penalties.",
      call. = FALSE
    )
  }

  # Get lambda values and mean cross-validated error
  lambda <- cv_results$lambda
  cvm <- cv_results$cvm
  cvsd <- cv_results$cvsd

  # Create a data frame for plotting
  cv_df <- tibble::tibble(
    lambda = lambda,
    error = cvm,
    error_upper = cvm + cvsd,
    error_lower = cvm - cvsd
  )

  # Get optimal lambda values
  lambda_min <- cv_results$lambda.min
  lambda_1se <- cv_results$lambda.1se

  # The measure cross-validation used, as glmnet names it. Read from the
  # model type, the label said "Binomial Deviance" for every classifier, a
  # multinomial one included, and "Mean Squared Error" whatever type.measure
  # chose.
  y_label <- unname(cv_results$name) %||% "Cross-validated error"

  # Create the plot
  p <- ggplot2::ggplot(
    cv_df,
    ggplot2::aes(x = lambda, y = error)
  ) +
    ggplot2::geom_ribbon(
      ggplot2::aes(
        ymin = error_lower,
        ymax = error_upper
      ),
      alpha = 0.2
    ) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::geom_vline(
      xintercept = lambda_min,
      linetype = "dashed",
      color = "blue"
    ) +
    ggplot2::geom_vline(
      xintercept = lambda_1se,
      linetype = "dashed",
      color = "red"
    ) +
    ggplot2::scale_x_log10() +
    ggplot2::labs(
      title = "Cross-Validation Results",
      subtitle = paste0(
        "Blue: lambda.min, Red: lambda.1se"
      ),
      x = "Lambda (log scale)",
      y = y_label
    ) +
    ggplot2::theme_minimal()

  p
}

#' Plot variable importance for a regularized model
#'
#' @param model A tidylearn regularized model object
#' @param lambda Which lambda to use: "1se" (default), "min", or a numeric
#'   penalty within the fitted path
#' @param top_n Number of top features to display
#'   (default: 20)
#' @param ... Additional arguments
#' @return A \code{\link[ggplot2]{ggplot}} object.
#' @examples
#' \donttest{
#' model <- tl_model(mtcars, mpg ~ ., method = "lasso")
#' tl_plot_importance_regularized(model)
#' }
#' @importFrom ggplot2 ggplot aes geom_col coord_flip
#'   labs theme_minimal
#' @export
tl_plot_importance_regularized <- function(model,
                                           lambda = "1se",
                                           top_n = 20,
                                           ...) {
  lambda_val <- tl_resolve_lambda(model$fit, lambda)

  # The same importance tl_table_importance() reports: raw |coefficient|
  # ranked predictors by their units, and failed on a multiclass fit
  importance_df <- tl_get_importance_regularized(model, lambda = lambda) |>
    dplyr::arrange(dplyr::desc(.data$importance)) |>
    dplyr::slice_head(n = top_n)
  if (nrow(importance_df) == 0) {
    stop("No feature has non-zero importance: the penalty dropped every ",
         "predictor from this model.", call. = FALSE)
  }

  # Create the plot
  p <- ggplot2::ggplot(
    importance_df,
    ggplot2::aes(
      x = stats::reorder(feature, importance),
      y = importance
    )
  ) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Feature Importance",
      subtitle = paste0(
        "|coefficient| x predictor SD at ",
        "lambda = ", signif(lambda_val, 4)
      ),
      x = NULL,
      y = "Importance (largest = 100)"
    ) +
    ggplot2::theme_minimal()

  p
}


#' Predict using a glmnet model
#' @keywords internal
#' @noRd
tl_predict_glmnet <- function(model, new_data,
                              type = "response",
                              ...) {
  fit <- model$fit
  is_classification <- model$spec$is_classification

  formula <- model$spec$formula

  # Rebuild the design matrix exactly as the fit did. Deriving it from a
  # "~ predictors - 1" formula instead would one-hot encode the first
  # factor, giving one more column than the model was trained on.
  design_terms <- attr(fit, "tl_terms")
  design_xlevels <- attr(fit, "tl_xlevels")

  if (is.null(design_terms)) {
    # Model fitted by an earlier version: recover the design from the
    # training data, which the model object carries
    training_frame <- stats::model.frame(formula, data = model$data)
    design_terms <- stats::terms(training_frame)
    design_xlevels <- stats::.getXlevels(design_terms, training_frame)
  }

  predictor_terms <- stats::delete.response(design_terms)
  new_frame <- stats::model.frame(
    predictor_terms, new_data,
    xlev = design_xlevels,
    na.action = stats::na.pass
  )
  # By name, as at fit time: without an intercept the first column is a
  # predictor
  x_new <- stats::model.matrix(predictor_terms, new_frame)
  x_new <- x_new[, colnames(x_new) != "(Intercept)", drop = FALSE]

  # Predict at the same penalty the coefficients, the coefficient table and
  # importance report by default. Predicting at lambda_min while those
  # showed lambda_1se meant the coefficients on display were not the ones
  # behind the predictions.
  lambda_val <- if (is.null(attr(fit, "lambda_1se"))) {
    # A fit carrying no selected penalty: use the first on the path
    fit$lambda[1]
  } else {
    tl_resolve_lambda(fit, "1se")
  }

  if (!is_classification) {
    return(as.vector(stats::predict(
      fit, newx = x_new, type = "response", s = lambda_val
    )))
  }

  # The classes fitted, or for a fit that predates storing them, the
  # model's. The training column is the raw one for a computed response,
  # so cut(mpg, ...) ~ . had one class per distinct mpg.
  class_levels <- attr(fit, "response_levels") %||% tl_model_classes(model)

  if (type == "prob") {
    probs <- stats::predict(
      fit, newx = x_new, type = "response", s = lambda_val
    )

    if (length(class_levels) == 2) {
      # glmnet returns the probability of the second level
      positive <- as.vector(probs)
      tibble::tibble(
        !!class_levels[1] := 1 - positive,
        !!class_levels[2] := positive
      )
    } else {
      # Multinomial: an [observation, class, lambda] array. drop = TRUE
      # would flatten a single-row prediction to a bare vector, so pin
      # the shape explicitly instead.
      prob_mat <- matrix(
        probs[, , 1], nrow = nrow(x_new), ncol = length(class_levels)
      )
      colnames(prob_mat) <- class_levels
      tibble::as_tibble(as.data.frame(prob_mat))
    }
  } else if (type == "class" || type == "response") {
    preds <- stats::predict(
      fit, newx = x_new, type = "class", s = lambda_val
    )
    factor(as.vector(preds), levels = class_levels)
  } else {
    stop(
      "Invalid prediction type for regularized classification. ",
      "Use 'prob', 'class', or 'response'.",
      call. = FALSE
    )
  }
}
