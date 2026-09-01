# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

# xgboost and the BLAS both parallelise, and on a many-core check
# machine that put the suite well past the CPU-to-elapsed limit of 2.5.
#
# Setting OMP_NUM_THREADS here does not work, which an earlier version
# of this file got wrong. libgomp reads that variable once, when it is
# loaded -- and R has already loaded it through the BLAS before this
# script runs, so Sys.setenv() arrives too late and is ignored.
# Measured on a 16-core Linux box: an xgboost fit runs at a ratio of
# 15.8 with no cap, 15.8 with Sys.setenv() here, and 1.9 with the
# variable exported before R starts. The middle case is this file.
#
# The runtime API does work, because it changes the pool rather than
# the variable it was built from. Same fit, same machine: 1.97. The
# BLAS goes 14.8 to 1.94 alongside it.
#
# The package's own default is untouched -- users still get every core.
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(2)
  RhpcBLASctl::omp_set_num_threads(2)
}

library(testthat)
library(tidylearn)

test_check("tidylearn")
