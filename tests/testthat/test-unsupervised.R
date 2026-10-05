test_that("PCA models work", {
  model <- tl_model(iris[, 1:4], method = "pca")

  expect_s3_class(model, "tidylearn_pca")
  expect_equal(model$spec$paradigm, "unsupervised")

  # Check PCA components
  expect_true("scores" %in% names(model$fit))
  expect_true("loadings" %in% names(model$fit))
  expect_true("variance_explained" %in% names(model$fit))

  # Transform data
  transformed <- predict(model)
  expect_s3_class(transformed, "tbl_df")
  expect_true(
    all(grepl("PC", names(transformed)) |
          names(transformed) == ".obs_id")
  )
})

test_that("K-means clustering works", {
  model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)

  expect_s3_class(model, "tidylearn_kmeans")

  # Check cluster assignments
  expect_true("clusters" %in% names(model$fit))
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(iris))
  expect_true("cluster" %in% names(clusters))

  # Clusters should be 1 to k
  expect_true(all(clusters$cluster %in% 1:3))
})

test_that("PAM (K-medoids) clustering works", {
  skip_if_not_installed("cluster")

  model <- tl_model(iris[, 1:4], method = "pam", k = 3)

  expect_s3_class(model, "tidylearn_pam")

  # Check cluster assignments
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(iris))
  expect_true(all(clusters$cluster %in% 1:3))
})

test_that("CLARA clustering works", {
  skip_if_not_installed("cluster")

  # Create larger dataset for CLARA
  large_data <- iris[rep(seq_len(nrow(iris)), 10), 1:4]

  model <- tl_model(large_data, method = "clara", k = 3, samples = 5)

  expect_s3_class(model, "tidylearn_clara")

  # Check cluster assignments
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(large_data))
})

test_that("Hierarchical clustering works", {
  model <- tl_model(iris[, 1:4], method = "hclust")

  expect_s3_class(model, "tidylearn_hclust")

  # Check dendrogram exists
  expect_true("model" %in% names(model$fit))
  expect_s3_class(model$fit$model, "hclust")
})

test_that("DBSCAN clustering works", {
  skip_if_not_installed("dbscan")

  model <- tl_model(iris[, 1:4], method = "dbscan", eps = 0.5, minPts = 5)

  expect_s3_class(model, "tidylearn_dbscan")

  # Check cluster assignments (including noise points as 0)
  clusters <- model$fit$clusters
  expect_equal(nrow(clusters), nrow(iris))
  expect_true("cluster" %in% names(clusters))
})

test_that("MDS works", {
  model <- tl_model(iris[, 1:4], method = "mds", k = 2)

  expect_s3_class(model, "tidylearn_mds")

  # Check MDS points
  expect_true("points" %in% names(model$fit))
  points <- model$fit$points
  expect_equal(nrow(points), nrow(iris))
})

test_that("clustering models predict on new data", {
  # Train clustering model
  model <- tl_model(iris[1:100, 1:4], method = "kmeans", k = 3)

  # Predict on new data
  new_data <- iris[101:150, 1:4]
  predictions <- predict(model, new_data = new_data)

  expect_equal(nrow(predictions), nrow(new_data))
  expect_true("cluster" %in% names(predictions))
})

test_that("PCA retains specified number of components", {
  model <- tl_model(iris[, 1:4], method = "pca")

  # Default should retain all components
  transformed <- predict(model)
  pc_cols <- sum(grepl("^PC", names(transformed)))
  expect_equal(pc_cols, 4)
})

test_that("unsupervised methods handle different data sizes", {
  # Small dataset
  small_data <- iris[1:20, 1:4]
  model_small <- tl_model(small_data, method = "kmeans", k = 2)
  expect_s3_class(model_small, "tidylearn_kmeans")

  # Large dataset
  large_data <- iris[rep(seq_len(nrow(iris)), 5), 1:4]
  model_large <- tl_model(large_data, method = "kmeans", k = 3)
  expect_s3_class(model_large, "tidylearn_kmeans")
})

test_that("clustering validates k parameter", {
  # k should be reasonable - expect an error for invalid k
  expect_error(
    tl_model(iris[, 1:4], method = "kmeans", k = nrow(iris) + 1)
  )

  # Valid k should work
  expect_s3_class(
    tl_model(iris[, 1:4], method = "kmeans", k = 3),
    "tidylearn_kmeans"
  )
})

test_that("tidy_gower returns a dist object with correct metadata", {
  d <- tidy_gower(iris[1:10, 1:4])

  expect_s3_class(d, "dist")
  expect_equal(attr(d, "Size"), 10L)
  expect_equal(attr(d, "method"), "gower")
})

test_that("tidy_gower handles single-row input without erroring", {
  d <- tidy_gower(iris[1, 1:4])

  expect_s3_class(d, "dist")
  expect_equal(length(d), 0L)          # no pairs to compute
  expect_equal(attr(d, "method"), "gower")
})

test_that("tidy_gower distances are non-negative and symmetric", {
  d <- tidy_gower(iris[1:20, 1:4])
  m <- as.matrix(d)

  expect_true(all(m >= 0))
  expect_equal(m, t(m))               # symmetric
  expect_true(all(diag(m) == 0))       # self-distance is 0
})

test_that("tidy_gower gives distance 0 for identical rows", {
  dup <- iris[c(1, 1, 2), 1:4]
  d <- as.matrix(tidy_gower(dup))

  expect_equal(d[1, 2], 0)   # row 1 vs its duplicate
  expect_gt(d[1, 3], 0)      # genuinely different rows are non-zero
})

test_that("tidy_gower numeric-only distances are correct", {
  # Two observations: x in {0, 1}.  range = 1, so d = |0-1|/1 = 1.
  df <- data.frame(x = c(0, 1))
  d  <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 1)

  # Three observations: x in {0, 0.5, 1}.
  # d(1,2) = 0.5, d(1,3) = 1, d(2,3) = 0.5
  df3 <- data.frame(x = c(0, 0.5, 1))
  d3  <- as.matrix(tidy_gower(df3))

  expect_equal(d3[1, 2], 0.5)
  expect_equal(d3[1, 3], 1.0)
  expect_equal(d3[2, 3], 0.5)
})

test_that("tidy_gower categorical distances are correct", {
  # Same category → 0; different category → 1
  df <- data.frame(color = factor(c("red", "red", "blue")))
  d  <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 0)   # red vs red
  expect_equal(d[1, 3], 1)   # red vs blue
  expect_equal(d[2, 3], 1)   # red vs blue
})

test_that("tidy_gower ordered distances are correct", {
  # Ranks: low=1, medium=2, high=3; range = 2
  # d(low, medium) = 1/2, d(low, high) = 2/2 = 1, d(medium, high) = 1/2
  df <- data.frame(
    rating = ordered(c("low", "medium", "high"),
                     levels = c("low", "medium", "high"))
  )
  d <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 0.5)
  expect_equal(d[1, 3], 1.0)
  expect_equal(d[2, 3], 0.5)
})

test_that("tidy_gower handles mixed numeric and categorical columns", {
  # x in {0, 1, 0.5}, color in {red, red, blue}; equal weights, so each
  # pairwise distance is the average of the two d_k.
  # d(1,2): d_x = 1, d_color = 0  → 0.5
  # d(1,3): d_x = 0.5, d_color = 1  → 0.75
  # d(2,3): d_x = 0.5, d_color = 1  → 0.75
  df <- data.frame(
    x     = c(0, 1, 0.5),
    color = factor(c("red", "red", "blue"))
  )
  d <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 0.50)
  expect_equal(d[1, 3], 0.75)
  expect_equal(d[2, 3], 0.75)
})

test_that("tidy_gower skips NA values when computing distances", {
  # x: {0, 1}, y: {1, NA}.  For pair (1,2) only x is valid.
  # d = |0-1| / (1-0) = 1
  df <- data.frame(x = c(0, 1), y = c(1, NA_real_))
  d  <- as.matrix(tidy_gower(df))

  expect_equal(d[1, 2], 1)   # y skipped; only x contributes
})

test_that("tidy_gower gives NA for a pair with no variable in common", {
  # The distance matrix started at zero and was only filled when some
  # variable was observed in both rows, so rows 1 and 2 came out identical
  # (distance 0) where cluster::daisy() gives NA
  df <- data.frame(x = c(1, NA, 3), y = c(NA, 2, 5))
  ours <- as.matrix(tidy_gower(df))
  theirs <- as.matrix(cluster::daisy(df, metric = "gower"))

  expect_true(is.na(ours[1, 2]))
  expect_equal(unname(ours), unname(theirs))
})

test_that("PAM and hclust refuse undefined distances by naming the rows", {
  # PAM used to group the two rows as identical. With the distance now
  # NA, pam() and hclust() reject it in words that name neither the rows
  # nor the cause ("NA/NaN/Inf in foreign function call (arg 10)")
  df <- data.frame(x = c(1, NA, 3, 4), y = c(NA, 2, 5, 6))

  expect_error(
    tidy_pam(df, k = 2, metric = "gower"),
    "no variable observed in both \\(rows 1 and 2\\)"
  )
  expect_error(
    tidy_hclust(df, distance = "gower"),
    "no variable observed in both \\(rows 1 and 2\\)"
  )

  # A missing value that leaves every pair a shared variable is still fine
  shared <- data.frame(x = c(1, NA, 3, 4), y = c(1, 2, 5, 6))
  expect_s3_class(tidy_pam(shared, k = 2, metric = "gower"), "tidy_pam")
  expect_s3_class(tidy_hclust(shared, distance = "gower"), "tidy_hclust")
})

test_that("tidy_gower respects custom weights", {
  # Upweight x so the numeric dimension dominates
  df <- data.frame(
    x     = c(0, 1),
    color = factor(c("red", "blue"))
  )
  d_equal    <- as.matrix(tidy_gower(df))
  d_weighted <- as.matrix(tidy_gower(df, weights = c(x = 10, color = 1)))

  # Need ≥3 rows so the column range is wider than the pair (1,2) difference.
  # With only 2 rows, range == difference, so d_x = 1 always — weights can't
  # move a result that is already at its maximum.
  #
  # x: range = 1, d(1,2)_x = |0 - 0.1| / 1 = 0.1
  # color: d(1,2)_color = 1  (red vs blue)
  # equal weights: (0.1 + 1) / 2 = 0.55
  # heavy color (1, 100): (1*0.1 + 100*1) / 101 ≈ 0.991
  df2 <- data.frame(
    x     = c(0, 0.1, 1),
    color = factor(c("red", "blue", "red"))
  )
  d_eq2 <- as.matrix(tidy_gower(df2))
  d_wt2 <- as.matrix(tidy_gower(df2, weights = c(x = 1, color = 100)))

  expect_gt(d_wt2[1, 2], d_eq2[1, 2])  # heavy color weight → larger distance
})

test_that("tidy_gower constant column adds to denominator not numerator", {
  # A constant column (range = 0) has d_k = 0 for every pair, so it adds
  # nothing to the numerator.  However, both observations are non-NA, so the
  # column IS counted in valid_vars (the denominator).  The net effect is that
  # the overall distance is exactly halved compared to y alone.
  #
  # pair (1,2): d_x = 0, d_y = |0-1|/1 = 1  → (0 + 1) / 2 = 0.5
  # y alone:    d_y = 1/1 = 1
  df   <- data.frame(x = c(5, 5, 5), y = c(0, 1, 0.5))
  d    <- as.matrix(tidy_gower(df))
  d_y  <- as.matrix(tidy_gower(data.frame(y = c(0, 1, 0.5))))

  expect_equal(d, d_y / 2)
})


test_that("tidy_dist dispatches to tidy_gower for method = 'gower'", {
  df <- data.frame(
    x     = c(1, 2, 3),
    color = factor(c("a", "b", "a"))
  )
  d1 <- tidy_dist(df, method = "gower")
  d2 <- tidy_gower(df)

  expect_equal(as.vector(d1), as.vector(d2))
  expect_equal(attr(d1, "method"), "gower")
})

test_that("standardize_data standardises a rowwise tibble over its columns", {
  # mutate() on a rowwise tibble works one row at a time, and a single
  # value has no spread, so every standardised value came back NaN
  rows <- dplyr::rowwise(tibble::as_tibble(iris[1:10, 1:4]))
  std <- standardize_data(rows)
  expect_equal(std$Sepal.Length, as.numeric(scale(iris$Sepal.Length[1:10])))
  expect_equal(std$Petal.Width, as.numeric(scale(iris$Petal.Width[1:10])))
  expect_s3_class(std, "rowwise_df")

  # Centring alone gave 0 for every row, and scaling alone NA
  centred <- standardize_data(rows, scale = FALSE)
  expect_equal(
    centred$Sepal.Width,
    iris$Sepal.Width[1:10] - mean(iris$Sepal.Width[1:10])
  )

  # A rowwise identifier keeps its values, as it does under mutate()
  with_id <- dplyr::rowwise(
    tibble::tibble(id = 1:5, x = c(1, 2, 3, 4, 10)), id
  )
  std_id <- standardize_data(with_id)
  expect_equal(std_id$id, 1:5)
  expect_equal(std_id$x, as.numeric(scale(c(1, 2, 3, 4, 10))))
  expect_equal(dplyr::group_vars(std_id), "id")

  # A grouped tibble is still standardised within each group
  grouped <- standardize_data(dplyr::group_by(iris, Species))
  expect_equal(
    grouped$Sepal.Length[iris$Species == "setosa"],
    as.numeric(scale(iris$Sepal.Length[iris$Species == "setosa"]))
  )
})

test_that("unsupervised methods work with formula", {
  # PCA with formula
  model <- tl_model(
    iris,
    ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    method = "pca"
  )

  expect_s3_class(model, "tidylearn_pca")

  # Clustering with formula
  model2 <- tl_model(iris, ~ Sepal.Length + Sepal.Width,
                     method = "kmeans", k = 3)
  expect_s3_class(model2, "tidylearn_kmeans")
})

test_that("a non-numeric column named in the formula is reported", {
  # ~ Sepal.Length + Species clustered on Sepal.Length alone and PCA
  # returned two components for three named columns, without a word
  numeric_only <- list(
    kmeans = list(k = 2),
    pca = list(),
    mds = list(),
    clara = list(k = 2),
    dbscan = list(eps = 0.5, minPts = 5),
    hclust = list(),
    pam = list(k = 2)
  )

  for (method in names(numeric_only)) {
    expect_warning(
      do.call(tl_model, c(
        list(iris, ~ Sepal.Length + Sepal.Width + Species, method = method),
        numeric_only[[method]]
      )),
      "ignored the formula's non-numeric column: Species",
      info = method
    )
  }

  set.seed(1)
  km <- suppressWarnings(
    tl_model(iris, ~ Sepal.Length + Species, method = "kmeans", k = 2)
  )
  expect_named(km$fit$centers, c("cluster", "Sepal.Length"))

  # `.` means the numeric columns for these methods, so it stays quiet
  expect_no_warning(tl_model(iris, ~ ., method = "kmeans", k = 3))
  expect_no_warning(tl_model(iris, ~ ., method = "pca"))
})

test_that("Gower distance uses the factor columns the caller selected", {
  # tl_fit_hclust() passed the formula's columns to tidy_hclust(), which
  # then kept only the numeric ones, so the heights were Gower on `num`
  # alone. For PAM, `~ .` expanded to the numeric columns and dropped
  # `fac`, so it disagreed with the no-formula fit on 8 of 20 rows.
  set.seed(9)
  mixed <- data.frame(
    num = c(stats::rnorm(10), stats::rnorm(10, 0.2)),
    fac = factor(rep(c("a", "b"), each = 10))
  )
  gower <- cluster::daisy(mixed, metric = "gower")
  heights <- stats::hclust(gower, method = "average")$height

  named <- tl_model(mixed, ~ num + fac, method = "hclust", distance = "gower")
  expect_equal(named$fit$model$height, heights)
  unnamed <- tl_model(mixed, method = "hclust", distance = "gower")
  expect_equal(unnamed$fit$model$height, heights)
  expect_equal(tidy_hclust(mixed, distance = "gower")$model$height, heights)

  medoid_clusters <- unname(cluster::pam(gower, k = 2, diss = TRUE)$clustering)
  for (f in list(NULL, ~ ., ~ num + fac)) {
    pam_fit <- tl_model(mixed, f, method = "pam", k = 2, metric = "gower")
    expect_equal(
      pam_fit$fit$clusters$cluster, medoid_clusters,
      info = deparse(f)
    )
  }

  # `. - x` still excludes x when the dot is expanded over every column
  without_num <- tl_model(
    mixed, ~ . - num, method = "hclust", distance = "gower"
  )
  expect_equal(
    without_num$fit$model$height,
    stats::hclust(cluster::daisy(mixed["fac"], metric = "gower"),
                  method = "average")$height
  )

  # The arithmetic distances still use the numeric columns only
  expect_equal(
    tl_model(mixed, method = "hclust")$fit$model$height,
    stats::hclust(stats::dist(mixed["num"]), method = "average")$height
  )
})

# ---- princomp loadings -----------------------------------------------

test_that("tidy_pca(method = 'princomp') produces usable loadings", {
  # princomp() returns a "loadings" object rather than a plain matrix, and
  # as_tibble() read that as one long vector -- 16 values against 4 row
  # names -- so get_pca_loadings() failed outright with
  # "Can't recycle ..1 (size 16) to match ..3 (size 4)".
  pca <- tidy_pca(iris[, 1:4], method = "princomp")

  expect_equal(nrow(pca$loadings), 16L)
  expect_setequal(names(pca$loadings), c("variable", "component", "loading"))
  expect_setequal(unique(pca$loadings$variable), names(iris)[1:4])
  expect_length(unique(pca$loadings$component), 4L)

  wide <- get_pca_loadings(pca)
  expect_equal(nrow(wide), 4L)
  expect_setequal(wide$variable, names(iris)[1:4])
  expect_true(all(vapply(wide[-1], is.numeric, logical(1))))
})

test_that("princomp and prcomp agree on the loadings, up to sign", {
  # The two routines are the same decomposition by different algorithms,
  # so a component may come back negated but not different. Comparing the
  # numbers is what catches a reshape that silently transposes or
  # recycles; checking only that columns exist would not.
  wide_p <- get_pca_loadings(tidy_pca(iris[, 1:4], method = "prcomp"))
  wide_c <- get_pca_loadings(tidy_pca(iris[, 1:4], method = "princomp"))

  expect_equal(wide_p$variable, wide_c$variable)

  for (i in seq_len(4)) {
    from_prcomp <- wide_p[[i + 1L]]
    from_princomp <- wide_c[[i + 1L]]
    agree <- isTRUE(all.equal(from_prcomp, from_princomp, tolerance = 1e-6)) ||
      isTRUE(all.equal(from_prcomp, -from_princomp, tolerance = 1e-6))
    expect_true(agree, info = paste("component", i))
  }
})

test_that("princomp says it centres, and the settings record that it did", {
  # princomp() has no way to skip centring, so center = FALSE returned
  # centred scores while $settings$center claimed FALSE
  expect_warning(
    pca <- tidy_pca(USArrests, method = "princomp",
                    center = FALSE, scale = FALSE),
    "always centres the data, so center = FALSE was ignored"
  )
  expect_true(pca$settings$center)
  expect_equal(
    unname(as.matrix(pca$scores[, -1])),
    unname(unclass(stats::princomp(USArrests)$scores))
  )

  # The default call has nothing to warn about
  expect_no_warning(tidy_pca(USArrests, method = "princomp"))
  expect_false(tidy_pca(USArrests, center = FALSE)$settings$center)
})

# ---- gap statistic for hierarchical clustering -----------------------

test_that("optimal_hclust_k(method = 'gap') runs at all", {
  # clusGap() requires FUN to return a list with a `cluster` element and
  # cutree() returns a bare integer vector, so this branch died on its
  # first call with "$ operator is invalid for atomic vectors" -- which
  # kept two further faults below from ever showing themselves.
  hc <- tidy_hclust(USArrests[1:25, ], method = "average")
  result <- optimal_hclust_k(hc, method = "gap", max_k = 4)

  expect_true(is.list(result))
  expect_true("gap_data" %in% names(result))
  expect_equal(nrow(result$gap_data), 4L)
  expect_true(all(is.finite(result$gap_data$gap)))
})

test_that("the gap refit uses the model's own distance", {
  # The refit called stats::dist(), whose default is Euclidean, so a model
  # built with any other distance was scored against clusterings it would
  # never produce. Identical numbers here would mean the fallback is back.
  d <- USArrests[1:25, ]

  set.seed(1)
  euclidean <- optimal_hclust_k(
    tidy_hclust(d, method = "average", distance = "euclidean"),
    method = "gap", max_k = 4
  )
  set.seed(1)
  manhattan <- optimal_hclust_k(
    tidy_hclust(d, method = "average", distance = "manhattan"),
    method = "gap", max_k = 4
  )

  expect_false(isTRUE(all.equal(
    euclidean$gap_data$gap, manhattan$gap_data$gap
  )))
})

test_that("the gap statistic says why it cannot run from a distance alone", {
  # clusGap() resamples observations; a model built from a dist has none.
  # This used to surface as "no applicable method for 'select' applied to
  # an object of class NULL".
  hc <- tidy_hclust(tidy_dist(USArrests[1:25, ]), method = "average")
  expect_null(hc$data)
  expect_error(
    optimal_hclust_k(hc, method = "gap", max_k = 4),
    "needs the observations"
  )

  # Silhouette works from the distances, and still does
  expect_type(
    optimal_hclust_k(hc, method = "silhouette", max_k = 4)$optimal_k, "double"
  )
})

test_that("the gap statistic refuses a Gower tree built on factor columns", {
  # clusGap() draws its reference data uniformly over each numeric column,
  # so the refit dropped the factors and scored clusterings the tree never
  # made, without a word
  set.seed(9)
  mixed <- data.frame(
    num = stats::rnorm(20),
    fac = factor(rep(c("a", "b"), each = 10))
  )
  hc <- tidy_hclust(mixed, distance = "gower")
  expect_error(
    optimal_hclust_k(hc, method = "gap", max_k = 4),
    "on the non-numeric column: fac. Use method = \"silhouette\""
  )

  # Silhouette works from the tree's own distances
  expect_equal(optimal_hclust_k(hc, max_k = 4)$k_range, 2:4)

  # A Gower tree on numeric columns, and a Euclidean tree whose data holds a
  # factor it never used, can still use the gap statistic
  numeric_gower <- tidy_hclust(USArrests[1:25, ], distance = "gower")
  expect_equal(
    nrow(optimal_hclust_k(numeric_gower, method = "gap", max_k = 3)$gap_data),
    3L
  )
  euclidean <- tidy_hclust(mixed, cols = c(num, fac))
  expect_equal(
    nrow(optimal_hclust_k(euclidean, method = "gap", max_k = 3)$gap_data),
    3L
  )
})

test_that("tidy_gap_stat's firstmax is the first local maximum", {
  # which.max() is maxSE()'s "globalmax", so k_firstmax always equalled
  # k_globalmax: 8 for this seed, where maxSE(method = "firstmax") gives 5
  set.seed(1)
  gap <- tidy_gap_stat(iris[, 1:4], max_k = 8, B = 10, nstart = 5)
  tab <- gap$model$Tab

  expect_equal(
    gap$k_firstmax,
    cluster::maxSE(tab[, "gap"], tab[, "SE.sim"], method = "firstmax")
  )
  expect_equal(
    gap$k_globalmax,
    cluster::maxSE(tab[, "gap"], tab[, "SE.sim"], method = "globalmax")
  )
  # This seed separates the two rules, so the test can tell them apart
  expect_false(gap$k_firstmax == gap$k_globalmax)

  # The rules nest: firstSEmax <= firstmax <= globalmax. The labels follow
  printed <- capture.output(print(gap))
  expect_match(grep("firstSEmax:", printed, value = TRUE), "most conservative")
  expect_match(grep("  firstmax:", printed, value = TRUE), "middle ground")
  expect_match(grep("globalmax:", printed, value = TRUE), "most liberal")
})

test_that("cluster-count arguments are checked by name", {
  # 2:max_k with max_k = 1 is c(2, 1), so k = 1 reached silhouette(),
  # which returns NA, and sil[, 3] failed: "incorrect number of dimensions"
  hc <- tidy_hclust(USArrests)
  expect_error(
    optimal_hclust_k(hc, max_k = 1),
    "'max_k' must be a single whole number of at least 2 and at most 49"
  )
  expect_error(
    tidy_silhouette_analysis(iris[, 1:4], max_k = 1),
    "'max_k' must be a single whole number of at least 2 and at most 149"
  )
  expect_error(
    tidy_gap_stat(iris[, 1:4], max_k = 1),
    "'max_k' must be a single whole number of at least 2"
  )
  # 1:0 is c(1, 0), which asked kmeans() for zero centres
  expect_error(
    calc_wss(iris[, 1:4], max_k = 0),
    "'max_k' must be a single whole number of at least 1"
  )

  # cutree() returns a matrix for a vector k, and flattening it gave 100
  # assignments for 50 observations
  expect_error(tidy_cutree(hc, k = 2:3), "'k' must be a single whole number")
  expect_error(augment_hclust(hc, USArrests, k = 2:3), "'k' must be a single")
  expect_error(tidy_cutree(hc, h = c(50, 100)), "'h' must be a single number")

  # The values these checks have to let through
  expect_equal(nrow(tidy_cutree(hc, k = 3)), 50L)
  expect_equal(nrow(tidy_cutree(hc, h = 100)), 50L)
  expect_equal(
    optimal_hclust_k(hc, max_k = 49)$k_range, 2:49
  )
  expect_equal(nrow(tidy_silhouette_analysis(iris[, 1:4], max_k = 2)), 1L)
  expect_equal(nrow(calc_wss(iris[, 1:4], max_k = 1)), 1L)
})

# ---- missing values at the entry point -------------------------------

# Missing values are the most ordinary thing that can be wrong with a data
# set, and these routines rejected them from inside C code: kmeans() gave
# "NA/NaN/Inf in foreign function call (arg 1)", and anything looping over
# k with purrr wrapped that again into "In index: 2. Caused by error in
# `do_one()`". Neither names the column, the problem, or a way forward.

test_that("NA-intolerant entry points name the columns and a way forward", {
  na_data <- iris[, 1:4]
  na_data[1, 1] <- NA
  na_data[5, 3] <- NA
  na_data[9, 3] <- NA

  guarded <- list(
    tidy_kmeans = function(d) tidy_kmeans(d, k = 3),
    tidy_pca = function(d) tidy_pca(d),
    calc_wss = function(d) calc_wss(d, max_k = 3),
    optimal_clusters = function(d) optimal_clusters(d, max_k = 3),
    tidy_silhouette_analysis = function(d) {
      tidy_silhouette_analysis(d, max_k = 3)
    },
    tidy_gap_stat = function(d) tidy_gap_stat(d, max_k = 3)
  )

  for (nm in names(guarded)) {
    message_seen <- tryCatch(
      suppressWarnings(suppressMessages(guarded[[nm]](na_data))),
      error = function(e) conditionMessage(e)
    )
    expect_true(is.character(message_seen), info = nm)
    expect_match(message_seen, "missing or infinite values", info = nm)
    # Naming the columns is the point; a bare refusal would not help
    expect_match(message_seen, "Sepal.Length", info = nm)
    expect_match(message_seen, "Petal.Length", info = nm)
    # And it must not leak the machinery it used to
    expect_false(grepl("do_one|foreign function", message_seen), info = nm)
  }
})

test_that("the check catches infinities and an empty numeric selection", {
  infinite <- iris[, 1:4]
  infinite$Sepal.Width[2] <- Inf
  expect_error(tidy_kmeans(infinite, k = 3), "missing or infinite values")

  expect_error(
    tidy_kmeans(data.frame(a = letters[1:5]), k = 2),
    "needs at least one numeric column"
  )
})

test_that("methods that accept missing values are left alone", {
  # pam(), clara(), dist() and daisy() handle NA themselves. Guarding them
  # would remove working behaviour rather than improve a message.
  na_data <- iris[, 1:4]
  na_data[1, 1] <- NA

  expect_s3_class(suppressWarnings(tidy_pam(na_data, k = 3)), "tidy_pam")
  expect_s3_class(suppressWarnings(tidy_clara(na_data, k = 3)), "tidy_clara")
  expect_s3_class(suppressWarnings(tidy_hclust(na_data)), "tidy_hclust")
})

test_that("clean data is unaffected by the check", {
  expect_s3_class(tidy_kmeans(iris[, 1:4], k = 3), "tidy_kmeans")
  expect_s3_class(tidy_pca(iris[, 1:4]), "tidy_pca")
  expect_equal(nrow(calc_wss(iris[, 1:4], max_k = 3)), 3L)
})

# ---- grouped input ---------------------------------------------------

# dplyr re-adds a grouped tibble's grouping variables to any column
# selection ("Adding missing grouping variables"), so Species came back as
# a feature: clara clustered on its factor codes, the distance-based
# routines coerced it to NA, and kmeans failed outright.

test_that("grouped tibbles are fitted on their columns, not their groups", {
  plain <- tibble::as_tibble(iris)
  grouped <- dplyr::group_by(plain, Species)
  d <- stats::dist(iris[, 1:4])

  fits <- list(
    tidy_kmeans = function(x) {
      set.seed(1)
      tidy_kmeans(x, k = 3)$centers
    },
    tidy_kmeans_cols = function(x) {
      set.seed(1)
      tidy_kmeans(x, k = 3, cols = c("Sepal.Length", "Sepal.Width"))$centers
    },
    tidy_pam = function(x) tidy_pam(x, k = 3)$clusters,
    tidy_clara = function(x) tidy_clara(x, k = 3)$clusters,
    calc_wss = function(x) {
      set.seed(1)
      calc_wss(x, max_k = 3)
    },
    tidy_dbscan = function(x) tidy_dbscan(x, eps = 0.5, minPts = 5)$clusters,
    tidy_knn_dist = function(x) tidy_knn_dist(x),
    explore_dbscan_params = function(x) explore_dbscan_params(x, 0.5, 5),
    tidy_dist = function(x) as.vector(tidy_dist(x)),
    compare_distances = function(x) lapply(compare_distances(x), as.vector),
    tidy_hclust = function(x) tidy_hclust(x)$model$height,
    tidy_mds = function(x) tidy_mds(x)$config,
    tidy_silhouette_analysis = function(x) {
      set.seed(1)
      tidy_silhouette_analysis(x, max_k = 3)
    },
    tidy_gap_stat = function(x) {
      set.seed(1)
      tidy_gap_stat(x, max_k = 3, B = 5)$gap_data
    },
    calc_validation_metrics = function(x) {
      calc_validation_metrics(rep(1:3, each = 50), x, d)
    },
    compare_clusterings = function(x) {
      compare_clusterings(list(by_row = rep(1:3, each = 50)), x)
    }
  )

  for (nm in names(fits)) {
    expected <- fits[[nm]](plain)
    actual <- tryCatch(
      fits[[nm]](grouped),
      error = function(e) paste("error:", conditionMessage(e)),
      message = function(m) paste("message:", conditionMessage(m))
    )
    expect_equal(actual, expected, info = nm)
  }

  for (method in c("pca", "mds", "kmeans", "pam", "clara", "hclust",
                   "dbscan")) {
    args <- switch(
      method,
      kmeans = list(k = 3), pam = list(k = 3), clara = list(k = 3),
      dbscan = list(eps = 0.5, minPts = 5),
      list()
    )
    set.seed(1)
    expected <- do.call(tl_model, c(list(plain, method = method), args))
    set.seed(1)
    actual <- tryCatch(
      do.call(tl_model, c(list(grouped, method = method), args)),
      error = function(e) paste("error:", conditionMessage(e)),
      message = function(m) paste("message:", conditionMessage(m))
    )
    expect_equal(actual$fit$model, expected$fit$model, info = method)
  }
})

test_that("augment_hclust attaches clusters by position, grouped or not", {
  # row_number() inside mutate() counts within each group, so the join
  # attached other rows' clusters to a grouped tibble
  d <- tibble::as_tibble(USArrests, rownames = "state")
  d$grp <- rep(c("a", "b"), length.out = 50)
  hc <- tidy_hclust(d, method = "ward.D2")
  reference <- as.integer(stats::cutree(hc$model, k = 3))

  grouped <- augment_hclust(hc, dplyr::group_by(d, grp), k = 3)
  expect_equal(grouped$cluster, reference)
  expect_equal(augment_hclust(hc, d, k = 3)$cluster, reference)
  # The caller's grouping stays on the data they get back
  expect_equal(dplyr::group_vars(grouped), "grp")

  # The join used to fill unmatched rows with NA, or drop assignments, when
  # the data was not the data the tree was built on
  expect_error(
    augment_hclust(hc, d[1:10, ], k = 3),
    "the tree has 50, but 'data' has 10 rows"
  )
})

# ---- column selection ------------------------------------------------

test_that("cols takes tidy-select expressions, not only strings", {
  # if (!is.null(cols)) evaluated the argument before enquo() captured it,
  # so a bare name failed with "object 'Sepal.Length' not found"
  set.seed(1)
  bare <- tidy_kmeans(iris, k = 2, cols = c(Sepal.Length, Sepal.Width))
  set.seed(1)
  quoted <- tidy_kmeans(iris, k = 2, cols = c("Sepal.Length", "Sepal.Width"))
  expect_equal(bare$centers, quoted$centers)
  expect_named(bare$centers, c("cluster", "Sepal.Length", "Sepal.Width"))

  petals <- c("Petal.Length", "Petal.Width")
  expect_named(
    tidy_pca(iris, cols = dplyr::starts_with("Petal"))$model$center, petals
  )
  pam_fit <- tidy_pam(iris, k = 2, cols = c(Petal.Length, Petal.Width))
  expect_equal(
    unname(pam_fit$model$clustering),
    unname(cluster::pam(stats::dist(iris[petals]), k = 2,
                        diss = TRUE)$clustering)
  )
  expect_named(
    tidy_hclust(iris, cols = dplyr::starts_with("Petal"))$data, petals
  )
  expect_equal(
    as.vector(tidy_dist(iris, cols = c(Petal.Length, Petal.Width))),
    as.vector(stats::dist(iris[petals]))
  )
  expect_equal(
    tidy_dbscan(iris, eps = 0.3, minPts = 5,
                cols = c(Petal.Length, Petal.Width))$clusters$cluster,
    as.integer(dbscan::dbscan(iris[petals], eps = 0.3, minPts = 5)$cluster)
  )
  expect_equal(
    tidy_knn_dist(iris, k = 4, cols = c(Petal.Length, Petal.Width))$knn_dist,
    as.numeric(dbscan::kNNdist(iris[petals], k = 4))
  )
})

# ---- DBSCAN ----------------------------------------------------------

test_that("tidy_dbscan clusters on the distance it is given", {
  # distance = "manhattan" was ignored: 2 clusters and 17 noise points,
  # the Euclidean answer, where dbscan on Manhattan distances finds 3 and 91
  x <- iris[, 1:4]
  manhattan <- stats::dist(x, method = "manhattan")
  theirs <- dbscan::dbscan(manhattan, eps = 0.5, minPts = 5)

  ours <- tidy_dbscan(x, eps = 0.5, minPts = 5, distance = "manhattan")
  expect_equal(ours$clusters$cluster, as.integer(theirs$cluster))
  expect_equal(c(ours$n_clusters, ours$n_noise), c(3, 91))
  # Core points are judged on the same distances
  expect_equal(
    ours$clusters$is_core,
    dbscan::is.corepoint(manhattan, eps = 0.5, minPts = 5)
  )

  via_model <- tl_model(x, method = "dbscan", eps = 0.5, minPts = 5,
                        distance = "manhattan")
  expect_equal(via_model$fit$clusters$cluster, as.integer(theirs$cluster))

  # Euclidean is still the default
  expect_equal(
    tidy_dbscan(x, eps = 0.5, minPts = 5)$clusters$cluster,
    as.integer(dbscan::dbscan(x, eps = 0.5, minPts = 5)$cluster)
  )
})

test_that("the k-NN and DBSCAN helpers accept a numeric matrix", {
  # Each of them piped the matrix straight into dplyr::select(): "no
  # applicable method for 'select' applied to an object of class matrix"
  as_matrix <- as.matrix(iris[, 1:4])
  as_frame <- iris[, 1:4]

  expect_equal(tidy_knn_dist(as_matrix), tidy_knn_dist(as_frame))
  expect_equal(
    suggest_eps(as_matrix, minPts = 5)$eps,
    suggest_eps(as_frame, minPts = 5)$eps
  )
  expect_equal(
    explore_dbscan_params(as_matrix, eps_values = 0.5, minPts_values = 5),
    explore_dbscan_params(as_frame, eps_values = 0.5, minPts_values = 5)
  )
  expect_equal(
    tidy_dbscan(as_matrix, eps = 0.5, minPts = 5)$clusters$cluster,
    as.integer(dbscan::dbscan(as_matrix, eps = 0.5, minPts = 5)$cluster)
  )
})

test_that("suggest_eps reads the k-NN distance at k = minPts - 1", {
  # It used k = minPts, one neighbour more than dbscan's own convention
  # (kNNdistplot(minPts = ) sets k = minPts - 1), and more than the k = 4
  # that tidy_knn_dist() and plot_knn_dist() default to for minPts = 5
  x <- iris[, 1:4]
  expect_equal(
    suggest_eps(x, minPts = 5)$eps,
    unname(stats::quantile(dbscan::kNNdist(x, k = 4), 0.95))
  )

  # With one point per neighbourhood every point is core, so there is no
  # radius to suggest
  expect_error(
    suggest_eps(x, minPts = 1),
    "'minPts' must be a single whole number of at least 2"
  )
  expect_equal(
    suggest_eps(x, minPts = 2)$eps,
    unname(stats::quantile(dbscan::kNNdist(x, k = 1), 0.95))
  )
})

test_that("plot_knn_dist labels a percentile that is not a whole percent", {
  # sprintf("%d") refuses a double, and 0.975 * 100 is not a whole number
  # -- nor is 0.57 * 100, which is 56.99999999999999
  for (percentile in c(0.975, 0.57, 0.95)) {
    p <- plot_knn_dist(iris[, 1:4], k = 4, percentile = percentile)
    is_text <- vapply(
      p$layers, function(l) inherits(l$geom, "GeomText"), logical(1)
    )
    label <- ggplot2::layer_data(p, which(is_text))$label
    expect_match(
      label, paste0("(", 100 * percentile, "% percentile)"),
      fixed = TRUE, info = percentile
    )
  }
})

# ---- PAM and CLARA ---------------------------------------------------

test_that("tidy_pam reports medoids by integer position", {
  # pam() on a labelled dist returns the medoids' labels, so mtcars gave
  # "Toyota Corona" and the same data as a tibble gave 21
  reference <- cluster::pam(stats::dist(mtcars), k = 3, diss = TRUE)$id.med

  named <- tidy_pam(mtcars, k = 3)
  unnamed <- tidy_pam(tibble::as_tibble(mtcars), k = 3)
  expect_identical(named$medoids$medoid_index, reference)
  expect_identical(unnamed$medoids$medoid_index, reference)
  expect_equal(named$medoids$mpg, mtcars$mpg[reference])
})

test_that("tidy_clara and tidy_pam pass further options to cluster", {
  # tidy_clara() had no `...`, so correct.d = TRUE was an unused argument;
  # without it, clara() warns about its pre-2016 distance formula on NA data
  na_data <- iris[, 1:4]
  na_data[1, 1] <- NA
  expect_no_warning(clara <- tidy_clara(na_data, k = 3, correct.d = TRUE))
  expect_equal(
    clara$model$clustering,
    cluster::clara(na_data, k = 3, samples = 50, sampsize = 46,
                   correct.d = TRUE)$clustering
  )

  # Starting medoids with no swap phase must come back unchanged
  pam <- tidy_pam(iris[, 1:4], k = 3, medoids = c(1, 51, 101),
                  do.swap = FALSE)
  expect_identical(pam$medoids$medoid_index, c(1L, 51L, 101L))
})

test_that("tidy_clara refuses a distance matrix and points to PAM", {
  # A branch passed dist objects straight to clara(), which samples
  # observations and takes no distances, so it failed inside cluster
  expect_error(
    tidy_clara(stats::dist(iris[, 1:4]), k = 3),
    "Use tidy_pam\\(\\), which takes a dist object"
  )
  expect_s3_class(tidy_clara(iris[, 1:4], k = 3), "tidy_clara")
})

# ---- validation ------------------------------------------------------

test_that("validation metrics leave DBSCAN noise out of every measure", {
  # k excluded noise, but the sizes, silhouette and WSS counted it as a
  # third cluster: min_size 17 was the noise count and avg_size 150 / 3
  x <- iris[, 1:4]
  clusters <- tidy_dbscan(x, eps = 0.5, minPts = 5)$clusters$cluster
  kept <- clusters != 0
  sil <- cluster::silhouette(clusters[kept], stats::dist(x[kept, ]))
  wss <- sum(vapply(
    split(x[kept, ], clusters[kept]),
    function(g) sum(scale(as.matrix(g), scale = FALSE)^2),
    numeric(1)
  ))

  metrics <- calc_validation_metrics(clusters, x, stats::dist(x))
  expect_equal(metrics$k, 2)
  expect_equal(metrics$n_noise, 17)
  expect_equal(
    c(metrics$min_size, metrics$max_size, metrics$avg_size),
    c(49, 84, 66.5)
  )
  expect_equal(metrics$avg_silhouette, mean(sil[, 3]))
  expect_equal(metrics$min_silhouette, min(sil[, 3]))
  expect_equal(metrics$total_wss, wss)

  compared <- compare_clusterings(list(dbscan = clusters), x)
  expect_equal(compared$avg_silhouette, mean(sil[, 3]))

  # A clustering without noise reports none
  set.seed(1)
  km <- stats::kmeans(x, 3, nstart = 5)$cluster
  no_noise <- calc_validation_metrics(km, x, stats::dist(x))
  expect_equal(no_noise$n_noise, 0)
  expect_equal(
    no_noise$avg_silhouette,
    mean(cluster::silhouette(km, stats::dist(x))[, 3])
  )
})

test_that("the validators accept factor and character cluster labels", {
  # silhouette() calls round() on the labels: "'round' not meaningful for
  # factors", which refused augment_kmeans()'s own output
  x <- iris[, 1:4]
  d <- stats::dist(x)
  km <- tidy_kmeans(x, k = 3)
  as_integer <- km$clusters$cluster
  as_factor <- augment_kmeans(km, x)$cluster
  as_letters <- c("a", "b", "c")[as_integer]

  expect_equal(
    calc_validation_metrics(as_factor, x, d),
    calc_validation_metrics(as_integer, x, d)
  )
  expect_equal(
    calc_validation_metrics(as_letters, x, d),
    calc_validation_metrics(as_integer, x, d)
  )
  expect_equal(
    tidy_silhouette(as_factor, d)$silhouette_data,
    tidy_silhouette(as_integer, d)$silhouette_data
  )
  by_letter <- tidy_silhouette(as_letters, d)
  expect_equal(by_letter$avg_width, tidy_silhouette(as_integer, d)$avg_width)
  expect_equal(by_letter$cluster_avg$cluster, c("a", "b", "c"))

  # augment_dbscan()'s factor keeps "0" as noise
  db <- tidy_dbscan(x, eps = 0.5, minPts = 5)
  expect_equal(
    calc_validation_metrics(augment_dbscan(db, x)$cluster, x, d),
    calc_validation_metrics(db$clusters$cluster, x, d)
  )

  # One cluster has no silhouette to report
  expect_error(
    tidy_silhouette(rep(1, 150), d),
    "Silhouette widths need at least 2 clusters"
  )
  expect_true(is.na(calc_validation_metrics(rep(1, 150), x, d)$avg_silhouette))
})

test_that("compare_clusterings names the entries it is not given names for", {
  # map_dfr() over names(NULL) returned a 0 x 0 tibble for an unnamed list
  set.seed(1)
  k2 <- stats::kmeans(iris[, 1:4], 2, nstart = 5)$cluster
  k3 <- stats::kmeans(iris[, 1:4], 3, nstart = 5)$cluster

  unnamed <- compare_clusterings(list(k2, k3), iris[, 1:4])
  expect_equal(unnamed$method, c("clustering_1", "clustering_2"))
  expect_equal(unnamed$k, c(2, 3))

  partly <- compare_clusterings(list(two = k2, k3), iris[, 1:4])
  expect_equal(partly$method, c("two", "clustering_2"))

  expect_error(
    compare_clusterings(k2, iris[, 1:4]),
    "'cluster_list' must be a list of cluster assignment vectors"
  )
})

# ---- MDS and PCA arguments -------------------------------------------

test_that("tl_model(method = 'mds') reaches every variant and ndim", {
  # tl_model()'s own `method` took the name, so sammon, kruskal and smacof
  # were unreachable, and ndim = 3 was "matched by multiple actual arguments"
  d <- stats::dist(USArrests)

  three <- tl_model(USArrests, method = "mds", ndim = 3)
  expect_equal(
    unname(as.matrix(three$fit$points[, paste0("Dim", 1:3)])),
    unname(stats::cmdscale(d, k = 3))
  )

  sammon <- tl_model(USArrests, method = "mds", mds_method = "sammon")
  expect_equal(
    unname(as.matrix(sammon$fit$points[, c("Dim1", "Dim2")])),
    unname(MASS::sammon(d, k = 2, trace = FALSE)$points)
  )
  metric <- tl_model(USArrests, method = "mds", mds_method = "metric", k = 3)
  expect_equal(
    metric$fit$stress,
    smacof::mds(d, ndim = 3, type = "ratio")$stress
  )

  expect_error(
    tl_model(USArrests, method = "mds", k = 2, ndim = 3),
    "'k' and 'ndim' both set the number of MDS dimensions"
  )
  expect_error(
    tl_model(USArrests, method = "mds", mds_method = "isomap"),
    "'mds_method' must be one of"
  )
  # k alone, as before
  expect_equal(
    ncol(tl_model(USArrests, method = "mds", k = 2)$fit$points), 3L
  )
})

test_that("MDS validates ndim and reports the dimensions it returned", {
  # print counted every column but one as a dimension, assuming an .obs_id
  # column, so a tibble fit (no row names) printed "Dimensions: 1"
  printed <- capture.output(print(tidy_mds(tibble::as_tibble(USArrests))))
  expect_true("Dimensions: 2 " %in% printed)

  expect_error(tidy_mds(USArrests, ndim = 0), "'ndim' must be a single whole")
  expect_error(
    tidy_mds_classical(stats::dist(USArrests), ndim = 1.5),
    "'ndim' must be a single whole"
  )

  # cmdscale() returns fewer columns than asked for when the distances
  # support fewer dimensions; naming ndim columns failed on the mismatch
  flat <- data.frame(x = 1:4, y = 2 * (1:4))
  mds <- suppressWarnings(tidy_mds(flat, ndim = 3))
  expect_equal(
    unname(as.matrix(mds$config)),
    unname(suppressWarnings(stats::cmdscale(stats::dist(flat), k = 3)))
  )
  expect_equal(
    mds$gof,
    suppressWarnings(
      stats::cmdscale(stats::dist(flat), k = 3, eig = TRUE)$GOF[1]
    )
  )
})

test_that("tidy_mds takes a Gower distance", {
  # `distance` went straight to stats::dist(), which has no "gower":
  # "invalid distance method"
  set.seed(9)
  mixed <- data.frame(
    num = stats::rnorm(20),
    fac = factor(rep(c("a", "b"), each = 10))
  )
  reference <- stats::cmdscale(cluster::daisy(mixed, metric = "gower"), k = 2)

  # Compared through the configuration's own distances, which a flipped
  # axis does not change
  mds <- tidy_mds(mixed, distance = "gower")
  expect_equal(
    as.vector(stats::dist(mds$config[c("Dim1", "Dim2")])),
    as.vector(stats::dist(reference))
  )

  # Through tl_model() the formula's factor is used rather than reported
  expect_no_warning(
    fit <- tl_model(mixed, ~ num + fac, method = "mds", distance = "gower")
  )
  expect_equal(
    as.vector(stats::dist(fit$fit$points[c("Dim1", "Dim2")])),
    as.vector(stats::dist(reference))
  )

  # Undefined distances are refused by name, where cmdscale() said only
  # "NA values not allowed in 'd'"
  expect_error(
    tidy_mds(data.frame(x = c(1, NA, 3, 4), y = c(NA, 2, 5, 6)),
             distance = "gower"),
    "MDS cannot use undefined distances: 1 pair of rows has no variable"
  )
})

test_that("component counts are checked by name", {
  # 1:n_components with n_components = 0 is c(1, 0), which kept PC1
  pca <- tidy_pca(iris[, 1:4])
  expect_error(
    get_pca_loadings(pca, n_components = 0),
    "'n_components' must be a single whole number of at least 1 and at most 4"
  )
  expect_error(
    augment_pca(pca, iris[, 1:4], n_components = 0),
    "'n_components' must be a single whole number of at least 1 and at most 4"
  )
  expect_error(
    get_pca_loadings(pca, n_components = 5),
    "at most 4"
  )

  expect_named(
    get_pca_loadings(pca, n_components = 2), c("variable", "PC1", "PC2")
  )
  expect_named(
    augment_pca(pca, iris[, 1:4], n_components = 4),
    c(names(iris)[1:4], paste0("PC", 1:4))
  )
})

# ---- market basket ---------------------------------------------------

groceries <- function() {
  skip_if_not_installed("arules")
  env <- new.env()
  utils::data("Groceries", package = "arules", envir = env)
  env$Groceries
}

# Whether each rule holds `item` on the given side, read straight from arules
rule_side_holds <- function(rules, side, item) {
  vapply(arules::LIST(side(rules)), function(items) item %in% items,
         logical(1))
}

test_that("recommend_products suggests only what the basket lacks, once each", {
  # Rules whose right-hand side was already in the basket were returned:
  # {yogurt}, then {other vegetables} four times, for this basket. Every
  # rule that fires for it suggests something it holds, so none is left.
  trans <- groceries()
  rules <- tidy_apriori(trans, support = 0.001, confidence = 0.5)
  full_basket <- c("whole milk", "other vegetables", "yogurt",
                   "root vegetables", "tropical fruit")
  expect_equal(nrow(recommend_products(rules, basket = full_basket)), 0L)

  # A broader rule set, against the rules read straight from arules
  broad <- tidy_apriori(trans, support = 0.001, confidence = 0.15)
  basket <- c("whole milk", "butter")
  lhs <- arules::LIST(arules::lhs(broad$rules))
  rhs <- arules::LIST(arules::rhs(broad$rules))
  quality <- arules::quality(broad$rules)
  fires <- vapply(lhs, function(items) all(items %in% basket), logical(1)) &
    vapply(rhs, function(items) !any(items %in% basket), logical(1)) &
    quality$confidence >= 0.15
  candidates <- data.frame(
    rhs = arules::labels(arules::rhs(broad$rules))[fires],
    lift = quality$lift[fires]
  )
  candidates <- candidates[order(-candidates$lift), ]
  expected <- utils::head(candidates[!duplicated(candidates$rhs), ], 50)

  rec <- recommend_products(broad, basket = basket,
                            min_confidence = 0.15, top_n = 50)
  expect_named(rec, c("rhs", "confidence", "lift", "support"))
  expect_equal(rec$rhs, expected$rhs)
  expect_equal(rec$lift, expected$lift)

  # The item lists behind the matching are arules' own
  expect_equal(broad$rules_tbl$lhs_items, lhs)
  expect_equal(broad$rules_tbl$rhs_items, rhs)
})

test_that("recommend_products matches item names that contain a comma", {
  skip_if_not_installed("arules")
  # The left-hand side's label was split on "," to recover its items, which
  # cut "salt, iodised" in two, so no rule on it could fire
  baskets <- list(
    c("salt, iodised", "bread"), c("salt, iodised", "bread"),
    c("salt, iodised", "bread"), c("salt, iodised", "bread", "milk"),
    c("bread", "milk"), c("salt, iodised", "milk")
  )
  rules <- tidy_apriori(as(baskets, "transactions"),
                        support = 0.3, confidence = 0.5)

  rec <- recommend_products(rules, basket = "salt, iodised",
                            min_confidence = 0.5)
  expect_true("{bread}" %in% rec$rhs)
})

test_that("filter_rules_by_item and find_related_items match whole items", {
  # grepl() matched "coffee" inside "instant coffee": 84 rules, of which 80
  # hold coffee. 19 Groceries items occur inside other item names.
  trans <- groceries()
  rules <- tidy_apriori(trans, support = 0.001, confidence = 0.5)

  # arules' own test of whether a rule's items include one
  expect_equal(
    nrow(filter_rules_by_item(rules, "coffee")),
    sum(arules::`%in%`(arules::items(rules$rules), "coffee"))
  )

  for (item in c("coffee", "ham", "oil")) {
    in_lhs <- rule_side_holds(rules$rules, arules::lhs, item)
    in_rhs <- rule_side_holds(rules$rules, arules::rhs, item)
    expect_equal(
      filter_rules_by_item(rules, item)$rule_id, which(in_lhs | in_rhs),
      info = item
    )
    expect_equal(
      filter_rules_by_item(rules, item, where = "lhs")$rule_id,
      which(in_lhs),
      info = item
    )
    expect_equal(
      filter_rules_by_item(rules, item, where = "rhs")$rule_id,
      which(in_rhs),
      info = item
    )
  }

  related <- find_related_items(rules, "coffee", min_lift = 1, top_n = 1000)
  expect_setequal(
    related$rule_id,
    which(rule_side_holds(rules$rules, arules::lhs, "coffee") |
            rule_side_holds(rules$rules, arules::rhs, "coffee"))
  )

  # One item at a time: a vector used to match on its first element only
  expect_error(
    filter_rules_by_item(rules, c("coffee", "ham")),
    "'item' must be a single item name"
  )
})

test_that("item matching reads the item lists tidy_rules() adds", {
  trans <- groceries()
  rules <- tidy_apriori(trans, support = 0.001, confidence = 0.5)

  # A filtered table keeps the lists, so it still matches
  strong <- dplyr::filter(rules$rules_tbl, .data$lift > 5)
  in_strong <- vapply(
    seq_len(nrow(strong)),
    function(i) "yogurt" %in% c(strong$lhs_items[[i]], strong$rhs_items[[i]]),
    logical(1)
  )
  expect_equal(
    filter_rules_by_item(strong, "yogurt")$rule_id, strong$rule_id[in_strong]
  )

  # A result without them is rebuilt from the rules it carries
  legacy <- rules
  legacy$rules_tbl <- dplyr::select(rules$rules_tbl, -"lhs_items", -"rhs_items")
  expect_equal(
    filter_rules_by_item(legacy, "coffee"),
    filter_rules_by_item(rules, "coffee")
  )

  # A bare table without them leaves nothing reliable to match on
  expect_error(
    filter_rules_by_item(legacy$rules_tbl, "yogurt"),
    "needs the lhs_items and rhs_items columns"
  )
})

test_that("an empty rule set gives empty results with the usual columns", {
  # tidy_rules() returned a zero-column tibble, so recommend_products()
  # failed with "object 'confidence' not found" and the item filters
  # reached arules::lhs() where they looked for a column
  trans <- groceries()
  rules <- tidy_apriori(trans, support = 0.001, confidence = 0.5)
  empty <- tidy_apriori(trans, support = 0.5, confidence = 0.9)
  no_rules <- rules$rules_tbl[0, ]

  expect_equal(empty$n_rules, 0L)
  expect_equal(empty$rules_tbl, no_rules)
  expect_equal(filter_rules_by_item(empty, "whole milk"), no_rules)
  expect_equal(find_related_items(empty, "whole milk"), no_rules)
  expect_equal(inspect_rules(empty), no_rules)
  expect_equal(
    recommend_products(empty, basket = "whole milk"),
    tibble::tibble(rhs = character(), confidence = numeric(),
                   lift = numeric(), support = numeric())
  )
  expect_no_error(ggplot2::ggplot_build(visualize_rules(empty)))
})

test_that("a frequent-itemsets result prints and inspects as itemsets", {
  # The print read the rules table, which itemsets do not have: nine
  # min/max warnings, "Support: Inf - -Inf", then "no applicable method
  # for 'slice' applied to an object of class NULL"
  trans <- groceries()
  itemsets <- tidy_apriori(trans, support = 0.05,
                           target = "frequent itemsets")

  expect_no_warning(printed <- capture.output(print(itemsets)))
  expect_true(any(grepl("Number of itemsets: 3", printed, fixed = TRUE)))

  support <- arules::quality(itemsets$rules)$support
  ranked <- order(support, decreasing = TRUE)
  top <- inspect_rules(itemsets, by = "support", n = 2)
  expect_equal(top$support, support[ranked[1:2]])
  expect_equal(top$itemset, arules::labels(itemsets$rules)[ranked[1:2]])
  expect_equal(
    itemsets$itemsets_tbl$items,
    arules::LIST(arules::items(itemsets$rules))
  )

  # Itemsets have no lift, so the default sort falls back to support
  expect_no_warning(by_default <- inspect_rules(itemsets))
  expect_equal(by_default$support, support[ranked])

  # The rule helpers say what they were given rather than failing on NULL
  expect_error(
    recommend_products(itemsets, basket = "whole milk"), "holds itemsets"
  )
  expect_error(summarize_rules(itemsets), "holds itemsets")
})

test_that("visualize_rules plots the top_n rules by lift", {
  # head() on an arules rule set kept the first N in mining order, so "Top
  # 50 rules" plotted lifts 2.04 to 16.7 where the highest 50 run 8.08 to 19.0
  trans <- groceries()
  rules <- tidy_apriori(trans, support = 0.001, confidence = 0.5)
  lifts <- arules::quality(rules$rules)$lift

  p <- visualize_rules(rules, top_n = 50)
  expect_equal(
    sort(p$data$lift, decreasing = TRUE),
    utils::head(sort(lifts, decreasing = TRUE), 50)
  )
})

test_that("inspect_rules(decreasing = FALSE) returns the lowest-ranked rules", {
  # It took the n highest and only then reversed their order, returning
  # lifts 16.4, 16.7 and 19.0 where the three lowest are 1.96
  trans <- groceries()
  rules <- tidy_apriori(trans, support = 0.001, confidence = 0.5)
  quality <- arules::quality(rules$rules)

  low <- inspect_rules(rules, by = "lift", n = 3, decreasing = FALSE)
  expect_equal(low$lift, utils::head(sort(quality$lift), 3))
  rare <- inspect_rules(rules, by = "support", n = 5, decreasing = FALSE)
  expect_equal(rare$support, utils::head(sort(quality$support), 5))

  high <- inspect_rules(rules, by = "lift", n = 3)
  expect_equal(
    high$lift, utils::head(sort(quality$lift, decreasing = TRUE), 3)
  )
})

test_that("tidy_apriori is quiet by default and passes options through", {
  # arules' trace printed about 20 lines on every call, with no way off
  trans <- groceries()
  expect_silent(tidy_apriori(trans, support = 0.01))
  expect_silent(tidy_apriori(trans, support = 0.01, control = list(sort = -1)))
  expect_output(
    tidy_apriori(trans, support = 0.01, control = list(verbose = TRUE)),
    "Apriori"
  )

  # appearance, and mining parameters beyond the named ones, reach apriori()
  milk <- list(rhs = "whole milk", default = "lhs")
  ours <- tidy_apriori(trans, support = 0.001, confidence = 0.5,
                       appearance = milk, smax = 0.002)
  theirs <- arules::apriori(
    trans,
    parameter = list(supp = 0.001, conf = 0.5, minlen = 2, maxlen = 10,
                     target = "rules", smax = 0.002),
    appearance = milk, control = list(verbose = FALSE)
  )
  expect_equal(ours$n_rules, length(theirs))
  expect_true(all(ours$rules_tbl$rhs == "{whole milk}"))
  expect_true(all(ours$rules_tbl$support <= 0.002))
})
