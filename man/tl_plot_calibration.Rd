% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classification.R
\name{tl_plot_calibration}
\alias{tl_plot_calibration}
\title{Plot calibration curve for a classification model}
\usage{
tl_plot_calibration(model, new_data = NULL, bins = 10, ...)
}
\arguments{
\item{model}{A tidylearn classification model object}

\item{new_data}{Optional data frame for evaluation (if NULL, uses training data)}

\item{bins}{Number of bins for grouping predictions (default: 10)}

\item{...}{Additional arguments}
}
\value{
A ggplot object with calibration curve
}
\description{
Plot calibration curve for a classification model
}
\keyword{internal}
