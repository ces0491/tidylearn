% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tuning.R
\name{tl_plot_tuning_results}
\alias{tl_plot_tuning_results}
\title{Plot hyperparameter tuning results}
\usage{
tl_plot_tuning_results(
  model,
  top_n = 5,
  param1 = NULL,
  param2 = NULL,
  plot_type = "scatter"
)
}
\arguments{
\item{model}{A tidylearn model object with tuning results}

\item{top_n}{Number of top parameter sets to highlight}

\item{param1}{First parameter to plot (for 2D grid or scatter plots)}

\item{param2}{Second parameter to plot (for 2D grid or scatter plots)}

\item{plot_type}{Type of plot: "scatter", "grid", "parallel", "importance"}
}
\value{
A ggplot object
}
\description{
Plot hyperparameter tuning results
}
