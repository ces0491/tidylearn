# Table Functions for tidylearn

Functions for producing formatted gt tables from tidylearn models.
Provides a parallel interface to the plot functions:
`tl_table(model, type)` dispatches to the appropriate table formatter
based on model type. Requires the gt package (suggested dependency).
