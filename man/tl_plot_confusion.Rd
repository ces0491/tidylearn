% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classification.R
\name{tl_plot_confusion}
\alias{tl_plot_confusion}
\title{Plot confusion matrix for a classification model}
\usage{
tl_plot_confusion(model, new_data = NULL, ...)
}
\arguments{
\item{model}{A tidylearn classification model object}

\item{new_data}{Optional data frame for evaluation (if NULL, uses training data)}

\item{...}{Additional arguments}
}
\value{
A ggplot object with confusion matrix
}
\description{
Plot confusion matrix for a classification model
}
\keyword{internal}
