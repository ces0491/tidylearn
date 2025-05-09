% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/integration.R
\name{tl_export_model}
\alias{tl_export_model}
\title{Export a tidylearn model to an external format}
\usage{
tl_export_model(model, format = "rds", file = NULL, ...)
}
\arguments{
\item{model}{A tidylearn model object}

\item{format}{Export format: "rds", "onnx", "pmml", "json"}

\item{file}{Path to save the exported model (if NULL, returns the model object)}

\item{...}{Additional arguments for the specific export format}
}
\value{
The exported model or NULL if saved to file
}
\description{
Export a tidylearn model to an external format
}
