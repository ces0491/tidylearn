# Create a tidylearn model

Unified interface for creating machine learning models by wrapping
established R packages. This function dispatches to the appropriate
underlying package based on the method.

## Usage

``` r
tl_model(data, formula = NULL, method = "linear", ..., compute = "cpu")
```

## Arguments

- data:

  A data frame containing the training data

- formula:

  A formula specifying the model. For unsupervised methods, use `~ vars`
  or NULL.

- method:

  The modeling method. Supervised: "linear" (stats::lm), "logistic"
  (stats::glm), "tree" (rpart), "forest" (randomForest), "boost" (gbm),
  "ridge"/"lasso"/"elastic_net" (glmnet), "svm" (e1071), "nn" (nnet),
  "deep" (keras), "xgboost" (xgboost). `"logistic"` requires a two-class
  response and errors on anything else; every other classification
  method handles more than two classes. Unsupervised: "pca"
  (stats::prcomp), "mds" (stats/MASS/smacof), "kmeans" (stats::kmeans),
  "pam"/"clara" (cluster), "hclust" (stats::hclust), "dbscan" (dbscan).

- ...:

  Additional arguments passed to the underlying model function

- compute:

  Compute tier for the fit. One of `"cpu"` (default, existing
  behaviour), `"gpu"` (route to local CUDA when the method has an
  upstream GPU path – xgboost and deep learning today), `"auto"`
  (consult
  [`tl_compute_advisor`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  and pick per call), or `"cloud"` (reserved – not yet wired up). When
  `"gpu"` is requested for a method without an upstream GPU path or on a
  machine without a detected GPU, the call falls back to CPU with a
  warning.

## Value

A `tidylearn_model` object (S3) containing the fitted model (`$fit`),
model specification (`$spec`), and training data (`$data`). The object
also inherits from a method-specific class (e.g., `tidylearn_linear`)
and a paradigm class (`tidylearn_supervised` or
`tidylearn_unsupervised`).

## Details

The wrapped packages include: stats (lm, glm, prcomp, kmeans, hclust),
glmnet, randomForest, xgboost, gbm, e1071, nnet, rpart, cluster, and
dbscan. The underlying algorithms are unchanged - this function provides
a consistent interface and returns tidy output.

Access the raw model object from the underlying package via `model$fit`.

For classification, the response is reduced to the classes it actually
contains: subsetting a data frame keeps every factor level, and a level
no row uses would otherwise be reported as a class, given its own (zero)
probability column, and counted when deciding whether the problem is
binary. The fit is unaffected.

## Examples

``` r
# \donttest{
# Classification -> wraps randomForest::randomForest()
model <- tl_model(iris, Species ~ ., method = "forest")
model$fit  # Access the raw randomForest object
#> 
#> Call:
#>  randomForest(formula = formula, data = data, ntree = ntree, mtry = mtry,      importance = importance) 
#>                Type of random forest: classification
#>                      Number of trees: 500
#> No. of variables tried at each split: 2
#> 
#>         OOB estimate of  error rate: 4.67%
#> Confusion matrix:
#>            setosa versicolor virginica class.error
#> setosa         50          0         0        0.00
#> versicolor      0         47         3        0.06
#> virginica       0          4        46        0.08

# Regression -> wraps stats::lm()
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
model$fit  # Access the raw lm object
#> 
#> Call:
#> stats::lm(formula = formula, data = data)
#> 
#> Coefficients:
#> (Intercept)           wt           hp  
#>    37.22727     -3.87783     -0.03177  
#> 

# PCA -> wraps stats::prcomp()
model <- tl_model(iris, ~ ., method = "pca")
model$fit  # Access the raw prcomp object
#> $scores
#> # A tibble: 150 × 5
#>    .obs_id   PC1     PC2     PC3      PC4
#>    <chr>   <dbl>   <dbl>   <dbl>    <dbl>
#>  1 1       -2.26 -0.478   0.127   0.0241 
#>  2 2       -2.07  0.672   0.234   0.103  
#>  3 3       -2.36  0.341  -0.0441  0.0283 
#>  4 4       -2.29  0.595  -0.0910 -0.0657 
#>  5 5       -2.38 -0.645  -0.0157 -0.0358 
#>  6 6       -2.07 -1.48   -0.0269  0.00659
#>  7 7       -2.44 -0.0475 -0.334  -0.0367 
#>  8 8       -2.23 -0.222   0.0884 -0.0245 
#>  9 9       -2.33  1.11   -0.145  -0.0268 
#> 10 10      -2.18  0.467   0.253  -0.0398 
#> # ℹ 140 more rows
#> 
#> $loadings
#> # A tibble: 4 × 5
#>   variable        PC1     PC2    PC3    PC4
#>   <chr>         <dbl>   <dbl>  <dbl>  <dbl>
#> 1 Sepal.Length  0.521 -0.377   0.720  0.261
#> 2 Sepal.Width  -0.269 -0.923  -0.244 -0.124
#> 3 Petal.Length  0.580 -0.0245 -0.142 -0.801
#> 4 Petal.Width   0.565 -0.0669 -0.634  0.524
#> 
#> $variance_explained
#> # A tibble: 4 × 5
#>   component  sdev variance prop_variance cum_variance
#>   <chr>     <dbl>    <dbl>         <dbl>        <dbl>
#> 1 PC1       1.71    2.92         0.730          0.730
#> 2 PC2       0.956   0.914        0.229          0.958
#> 3 PC3       0.383   0.147        0.0367         0.995
#> 4 PC4       0.144   0.0207       0.00518        1    
#> 
#> $model
#> Standard deviations (1, .., p=4):
#> [1] 1.7083611 0.9560494 0.3830886 0.1439265
#> 
#> Rotation (n x k) = (4 x 4):
#>                     PC1         PC2        PC3        PC4
#> Sepal.Length  0.5210659 -0.37741762  0.7195664  0.2612863
#> Sepal.Width  -0.2693474 -0.92329566 -0.2443818 -0.1235096
#> Petal.Length  0.5804131 -0.02449161 -0.1421264 -0.8014492
#> Petal.Width   0.5648565 -0.06694199 -0.6342727  0.5235971
#> 
#> $settings
#> $settings$scale
#> [1] TRUE
#> 
#> $settings$center
#> [1] TRUE
#> 
#> $settings$method
#> [1] "prcomp"
#> 
#> 

# Clustering -> wraps stats::kmeans()
model <- tl_model(iris, method = "kmeans", k = 3)
model$fit  # Access the raw kmeans object
#> $clusters
#> # A tibble: 150 × 2
#>    .obs_id cluster
#>    <chr>     <int>
#>  1 1             2
#>  2 2             2
#>  3 3             2
#>  4 4             2
#>  5 5             2
#>  6 6             2
#>  7 7             2
#>  8 8             2
#>  9 9             2
#> 10 10            2
#> # ℹ 140 more rows
#> 
#> $centers
#> # A tibble: 3 × 5
#>   cluster Sepal.Length Sepal.Width Petal.Length Petal.Width
#>     <int>        <dbl>       <dbl>        <dbl>       <dbl>
#> 1       1         5.90        2.75         4.39       1.43 
#> 2       2         5.01        3.43         1.46       0.246
#> 3       3         6.85        3.07         5.74       2.07 
#> 
#> $metrics
#> # A tibble: 1 × 6
#>       k tot_withinss betweenss tot_ss  iter converged
#>   <dbl>        <dbl>     <dbl>  <dbl> <int> <lgl>    
#> 1     3         78.9      603.   681.     2 TRUE     
#> 
#> $model
#> K-means clustering with 3 clusters of sizes 62, 50, 38
#> 
#> Cluster means:
#>   Sepal.Length Sepal.Width Petal.Length Petal.Width
#> 1     5.901613    2.748387     4.393548    1.433871
#> 2     5.006000    3.428000     1.462000    0.246000
#> 3     6.850000    3.073684     5.742105    2.071053
#> 
#> Clustering vector:
#>   [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#>  [38] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#>  [75] 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 1 3 3 3 3 1 3 3 3 3
#> [112] 3 3 1 1 3 3 3 3 1 3 1 3 1 3 3 1 1 3 3 3 3 3 1 3 3 3 3 1 3 3 3 1 3 3 3 1 3
#> [149] 3 1
#> 
#> Within cluster sum of squares by cluster:
#> [1] 39.82097 15.15100 23.87947
#>  (between_SS / total_SS =  88.4 %)
#> 
#> Available components:
#> 
#> [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
#> [6] "betweenss"    "size"         "iter"         "ifault"      
#> 
# }
```
