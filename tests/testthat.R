# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

# xgboost parallelises with OpenMP and takes every core it is offered.
# On the CRAN Debian pre-test that put the suite at 90s CPU against 24s
# elapsed -- a ratio of 3.8, where the limit is 2.5 -- while the Windows
# check of the same tarball passed, because the ratio is a property of
# the build rather than of the tests. Capping the pool at two keeps the
# ratio under the limit wherever the check runs.
#
# This has to happen before anything loads xgboost, which is why it is
# here rather than in setup.R: the OpenMP runtime reads the variable when
# it initialises and ignores later changes. Nothing else in the suite is
# threaded, and the package's own default is untouched -- users still get
# every core.
Sys.setenv(OMP_NUM_THREADS = "2")
Sys.setenv(OMP_THREAD_LIMIT = "2")

library(testthat)
library(tidylearn)

test_check("tidylearn")
