% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model_selection.R
\name{tl_test_model_difference}
\alias{tl_test_model_difference}
\title{Perform statistical comparison of models using cross-validation}
\usage{
tl_test_model_difference(
  cv_results,
  baseline_model = NULL,
  test = "t.test",
  metric = NULL
)
}
\arguments{
\item{cv_results}{Results from tl_compare_cv function}

\item{baseline_model}{Name of the model to use as baseline for comparison}

\item{test}{Type of statistical test: "t.test" or "wilcox"}

\item{metric}{Name of the metric to compare}
}
\value{
A data frame with statistical test results
}
\description{
Perform statistical comparison of models using cross-validation
}
