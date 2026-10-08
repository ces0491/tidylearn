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
  or NULL; a one-sided formula names columns, and `~ . - x` means every
  numeric column but `x`. Supervised methods need a two-sided formula.

- method:

  The modeling method. Supervised: "linear" (stats::lm), "polynomial"
  (stats::lm on polynomial terms), "logistic" (stats::glm), "tree"
  (rpart), "forest" (randomForest), "boost" (gbm),
  "ridge"/"lasso"/"elastic_net" (glmnet), "svm" (e1071), "nn" (nnet),
  "deep" (keras), "xgboost" (xgboost). The method and the response have
  to agree, and a mismatch is an error rather than a meaningless fit:
  `"linear"` and `"polynomial"` need a numeric response, `"logistic"`
  needs exactly two classes, and every other supervised method takes
  either. Unsupervised: "pca" (stats::prcomp), "mds" (stats::cmdscale,
  or smacof or MASS through `mds_method`), "kmeans" (stats::kmeans),
  "pam"/"clara" (cluster), "hclust" (stats::hclust), "dbscan" (dbscan).

- ...:

  Arguments for the method: see the Method arguments section. Anything
  else is passed to the underlying model function.

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
and training data (`$data`).
[`update()`](https://rdrr.io/r/stats/update.html) and
[`step()`](https://rdrr.io/r/stats/step.html) on `$fit` refit on the
training rows and weights, whatever is in the calling environment. The
object also inherits from a method-specific class (e.g.,
`tidylearn_linear`) and a paradigm class (`tidylearn_supervised` or
`tidylearn_unsupervised`).

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

Whether a supervised model is a classification or a regression is
decided by the response the formula computes, so `factor(cyl) ~ wt` is a
classification even though `cyl` is numeric. A factor or text response
is a classification. A logical response is a regression for every method
but `"logistic"` – with `"linear"`, a linear probability model – so
write `factor(y) ~ ...` to classify it.

A categorical predictor stored as text, as
[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
returns it, is made a factor before the fit and stored as one in
`$data`, so every method treats it as a category. At
[`predict()`](https://rdrr.io/r/stats/predict.html), categorical columns
in new data are read against the levels seen in training: new data may
hold only some of them, and a level the model was not trained on is an
error that names the column.

## Method arguments

Arguments in `...` are passed to the function the method wraps, except
for these, which tidylearn takes itself:

- `"polynomial"`:

  `degree` (default 2). Each numeric main effect is replaced by
  `poly(term, degree, raw = TRUE)` and the result fitted with
  [`lm()`](https://rdrr.io/r/stats/lm.html). A numeric term is one that
  computes a numeric vector, such as `wt` or `log(wt)`, or a one-column
  matrix, such as `scale(wt)`. One that is also part of an interaction
  keeps its own term and gains `I(x^2)` up to `I(x^degree)`, so the
  interaction is coded as written. Factor and other non-numeric terms,
  interactions, [`I()`](https://rdrr.io/r/base/AsIs.html) terms, bases
  such as [`poly()`](https://rdrr.io/r/stats/poly.html) or a spline's,
  the response as written, an
  [`offset()`](https://rdrr.io/r/stats/offset.html) and a removed
  intercept are kept as they are.

- `"ridge"`, `"lasso"`, `"elastic_net"`:

  `alpha`, glmnet's mixing parameter (by default 0, 1 and 0.5);
  `lambda`, a single penalty to fit at, a sequence of penalties for
  [`glmnet::cv.glmnet()`](https://glmnet.stanford.edu/reference/cv.glmnet.html)
  to choose from, or `NULL` (the default) to let it choose its own; and
  `cv_folds` (default 5), the number of folds for that cross-validation,
  which takes the place of glmnet's `nfolds`.
  [`predict()`](https://rdrr.io/r/stats/predict.html) uses the
  `lambda.1se` penalty. tidylearn sets `x`, `y`, `family` and `nfolds`
  itself and refuses them, along with any argument glmnet does not take.

- `"svm"`:

  `tune` (default `FALSE`) and `tune_folds` (default 5), to choose
  `cost` by cross-validation before the fit, with `gamma` for a
  non-linear kernel and `degree` for a polynomial one.

- `"deep"`:

  `hidden_layers`, `activation`, `dropout`, `epochs`, `batch_size`,
  `validation_split` and `learning_rate`.

- `"pca"`:

  `scale` and `center` (both `TRUE`).

- `"mds"`:

  `mds_method`, the variant: `"classical"` (the default,
  [`stats::cmdscale()`](https://rdrr.io/r/stats/cmdscale.html)),
  `"metric"` or `"nonmetric"` (smacof), or `"sammon"` or `"kruskal"`
  (MASS). `k`, or its alias `ndim`, is the number of dimensions (default
  2).

- `"kmeans"`, `"pam"`, `"clara"`:

  `k`, the number of clusters (default 3); for `"pam"`, `metric` as well
  (default `"euclidean"`).

- `"hclust"`:

  `hclust_method`, the linkage: `"average"` (the default), `"ward.D"`,
  `"ward.D2"`, `"single"`, `"complete"`, `"mcquitty"`, `"median"` or
  `"centroid"`; and `distance` (default `"euclidean"`).

- `"dbscan"`:

  `eps` (default 0.5), `minPts` (default 5) and `distance` (default
  `"euclidean"`).

Some defaults differ from the wrapped function's: `"forest"` computes
importance (`importance = TRUE`), `"boost"` grows trees of
`interaction.depth = 3`, and `"nn"` fits `size = 5` hidden units with
`trace = FALSE`.

`weights` and `subset` take values, one per row of `data` (such as
`weights = data$w`), not column names. A `subset` is applied before the
fit, and `$data` holds only the rows it selects. Case weights are
applied by every supervised method except `"svm"` and `"deep"`, which
refuse them. An offset, written as
[`offset()`](https://rdrr.io/r/stats/offset.html) in the formula, is
applied by `"linear"`, `"polynomial"` and `"logistic"`; the other
methods refuse one, because their
[`predict()`](https://rdrr.io/r/stats/predict.html) would not add it
back.

## Examples

``` r
# \donttest{
# Classification -> wraps randomForest::randomForest()
model <- tl_model(iris, Species ~ ., method = "forest")
model$fit  # Access the raw randomForest object
#> 
#> Call:
#>  randomForest::randomForest(formula = Species ~ ., data = <environment>$data,      ntree = 500, importance = TRUE) 
#>                Type of random forest: classification
#>                      Number of trees: 500
#> No. of variables tried at each split: 2
#> 
#>         OOB estimate of  error rate: 4%
#> Confusion matrix:
#>            setosa versicolor virginica class.error
#> setosa         50          0         0        0.00
#> versicolor      0         47         3        0.06
#> virginica       0          3        47        0.06

# Regression -> wraps stats::lm()
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
model$fit  # Access the raw lm object
#> 
#> Call:
#> lm(formula = mpg ~ wt + hp, data = <environment>$data)
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
#> K-means clustering with 3 clusters of sizes 38, 62, 50
#> 
#> Cluster means:
#>   Sepal.Length Sepal.Width Petal.Length Petal.Width
#> 1     6.850000    3.073684     5.742105    2.071053
#> 2     5.901613    2.748387     4.393548    1.433871
#> 3     5.006000    3.428000     1.462000    0.246000
#> 
#> Clustering vector:
#>   [1] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#>  [38] 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#>  [75] 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 1 1 1 1 2 1 1 1 1
#> [112] 1 1 2 2 1 1 1 1 2 1 2 1 2 1 1 2 2 1 1 1 1 1 2 1 1 1 1 2 1 1 1 2 1 1 1 2 1
#> [149] 1 2
#> 
#> Within cluster sum of squares by cluster:
#> [1] 23.87947 39.82097 15.15100
#>  (between_SS / total_SS =  88.4 %)
#> 
#> Available components:
#> 
#> [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
#> [6] "betweenss"    "size"         "iter"         "ifault"      
# }
```
