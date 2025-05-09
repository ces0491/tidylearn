% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/trees.R
\name{tl_predict_tree}
\alias{tl_predict_tree}
\title{Predict using a decision tree model}
\usage{
tl_predict_tree(model, new_data, type = "response", ...)
}
\arguments{
\item{model}{A tidylearn tree model object}

\item{new_data}{A data frame containing the new data}

\item{type}{Type of prediction: "response" (default), "prob" (for classification), "class" (for classification)}

\item{...}{Additional arguments}
}
\value{
Predictions
}
\description{
Predict using a decision tree model
}
\keyword{internal}
