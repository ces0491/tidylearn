# Tests that exercise base-graphics plotting (dendrograms, diagnostic
# panels) draw to the default device, which writes an Rplots.pdf into
# tests/testthat/ and would then be picked up by R CMD build. Send those
# draws to a null device instead. The device is closed when the test
# session exits.
grDevices::pdf(NULL)

# Fitting a keras model makes TensorFlow write autograph tracing files
# and a __pycache__ directory, and leaves them there. On a machine with
# a TensorFlow backend installed that is upwards of a hundred files per
# run, which R CMD check reports as "detritus in the temp directory".
# CRAN never sees it -- there is no backend there, so those tests skip
# -- but a maintainer running a local check does. Our tests create the
# files, so our tests remove them.
#
# They land in Python's tempfile.gettempdir(), which is the TMPDIR root
# -- NOT R's session tempdir(), which is a subdirectory of it that R
# removes on exit anyway. Cleaning the wrong one silently does nothing.
#
# Only files that appear during this run are removed, so a concurrent
# process's files are left alone.
tl_tf_tmp_dir <- dirname(tempdir())
tl_tf_tmp_pattern <- "^__autograph_generated_file.*\\.py$"
tl_tf_before <- list.files(tl_tf_tmp_dir, pattern = tl_tf_tmp_pattern)
tl_tf_pycache_existed <- dir.exists(file.path(tl_tf_tmp_dir, "__pycache__"))

withr::defer(
  {
    new_files <- setdiff(
      list.files(tl_tf_tmp_dir, pattern = tl_tf_tmp_pattern),
      tl_tf_before
    )
    if (length(new_files)) {
      unlink(file.path(tl_tf_tmp_dir, new_files), force = TRUE)
    }

    pycache <- file.path(tl_tf_tmp_dir, "__pycache__")
    if (!tl_tf_pycache_existed && dir.exists(pycache)) {
      unlink(pycache, recursive = TRUE, force = TRUE)
    }
  },
  teardown_env()
)
