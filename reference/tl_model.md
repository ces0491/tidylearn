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
  "deep" (keras), "xgboost" (xgboost). The method and the response have
  to agree, and a mismatch is an error rather than a meaningless fit:
  `"linear"` and `"polynomial"` need a numeric response, `"logistic"`
  needs exactly two classes, and every other supervised method takes
  either. Unsupervised: "pca" (stats::prcomp), "mds"
  (stats/MASS/smacof), "kmeans" (stats::kmeans), "pam"/"clara"
  (cluster), "hclust" (stats::hclust), "dbscan" (dbscan).

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

A `tidylearn_model` object (S3) containing the fitted model (`$fit`, or
`$fit$model` for an unsupervised method), model specification (`$spec`),
and training data (`$data`). The object also inherits from a
method-specific class (e.g., `tidylearn_linear`) and a paradigm class
(`tidylearn_supervised` or `tidylearn_unsupervised`).

## Details

The wrapped packages include: stats (lm, glm, prcomp, kmeans, hclust),
glmnet, randomForest, xgboost, gbm, e1071, nnet, rpart, cluster, and
dbscan. The underlying algorithms are unchanged - this function provides
a consistent interface and returns tidy output.

For a supervised method, `model$fit` is the object the wrapped function
returned. An unsupervised method returns tidied components as well, so
the wrapped object sits at `model$fit$model` and `model$fit` is the list
holding both.

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
#>  randomForest(formula = Species ~ ., data = data, ntree = 500,      importance = TRUE) 
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
model$fit$model  # The raw prcomp object, alongside tidied components
#> Standard deviations (1, .., p=4):
#> [1] 1.7083611 0.9560494 0.3830886 0.1439265
#> 
#> Rotation (n x k) = (4 x 4):
#>                     PC1         PC2        PC3        PC4
#> Sepal.Length  0.5210659 -0.37741762  0.7195664  0.2612863
#> Sepal.Width  -0.2693474 -0.92329566 -0.2443818 -0.1235096
#> Petal.Length  0.5804131 -0.02449161 -0.1421264 -0.8014492
#> Petal.Width   0.5648565 -0.06694199 -0.6342727  0.5235971

# Clustering -> wraps stats::kmeans()
model <- tl_model(iris, method = "kmeans", k = 3)
model$fit$model  # The raw kmeans object
#> K-means clustering with 3 clusters of sizes 50, 62, 38
#> 
#> Cluster means:
#>   Sepal.Length Sepal.Width Petal.Length Petal.Width
#> 1     5.006000    3.428000     1.462000    0.246000
#> 2     5.901613    2.748387     4.393548    1.433871
#> 3     6.850000    3.073684     5.742105    2.071053
#> 
#> Clustering vector:
#>   [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#>  [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#>  [75] 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 3 3 3 3 2 3 3 3 3
#> [112] 3 3 2 2 3 3 3 3 2 3 2 3 2 3 3 2 2 3 3 3 3 3 2 3 3 3 3 2 3 3 3 2 3 3 3 2 3
#> [149] 3 2
#> 
#> Within cluster sum of squares by cluster:
#> [1] 15.15100 39.82097 23.87947
#>  (between_SS / total_SS =  88.4 %)
#> 
#> Available components:
#> 
#> [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
#> [6] "betweenss"    "size"         "iter"         "ifault"      
# }
```
