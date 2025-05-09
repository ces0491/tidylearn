% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/trees.R
\name{tl_fit_tree}
\alias{tl_fit_tree}
\title{Fit a decision tree model}
\usage{
tl_fit_tree(
  data,
  formula,
  is_classification = FALSE,
  cp = 0.01,
  minsplit = 20,
  maxdepth = 30,
  ...
)
}
\arguments{
\item{data}{A data frame containing the training data}

\item{formula}{A formula specifying the model}

\item{is_classification}{Logical indicating if this is a classification problem}

\item{cp}{Complexity parameter (default: 0.01)}

\item{minsplit}{Minimum number of observations in a node for a split}

\item{maxdepth}{Maximum depth of the tree}

\item{...}{Additional arguments to pass to rpart()}
}
\value{
A fitted decision tree model
}
\description{
Fit a decision tree model
}
\keyword{internal}
