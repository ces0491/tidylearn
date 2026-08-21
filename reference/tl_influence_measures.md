# Calculate influence measures for a linear model

Calculate influence measures for a linear model

## Usage

``` r
tl_influence_measures(
  model,
  threshold_cook = NULL,
  threshold_leverage = NULL,
  threshold_dffits = NULL
)
```

## Arguments

- model:

  A tidylearn model object

- threshold_cook:

  Cook's distance threshold (default: 4/n)

- threshold_leverage:

  Leverage threshold (default: 2\*(p+1)/n)

- threshold_dffits:

  DFFITS threshold (default: 2\*sqrt((p+1)/n))

## Value

A data frame with one row per observation containing influence measures:
`cooks_distance`, `leverage`, `dffits`, `std_residual`, `stud_residual`,
boolean flags for each threshold (`is_cook_influential`,
`is_leverage_influential`, `is_dffits_influential`, `is_outlier`),
per-coefficient `dfbetas_*` columns, and an overall `is_influential`
flag. Threshold values are stored as attributes.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_influence_measures(model)
#>                     observation cooks_distance   leverage       dffits
#> Mazda RX4                     1   1.589652e-02 0.04427691 -0.218494101
#> Mazda RX4 Wag                 2   5.464779e-03 0.04048669 -0.126664789
#> Datsun 710                    3   2.070651e-02 0.06020097 -0.249103400
#> Hornet 4 Drive                4   4.724822e-05 0.04747494  0.011699160
#> Hornet Sportabout             5   2.736184e-04 0.03686288  0.028162679
#> Valiant                       6   2.155064e-02 0.06715104 -0.253806124
#> Duster 360                    7   1.255218e-02 0.11701542 -0.191618944
#> Merc 240D                     8   1.677650e-02 0.11565615  0.221917842
#> Merc 230                      9   2.188702e-03 0.06001631  0.079763706
#> Merc 280                     10   1.554996e-03 0.04691083 -0.067222732
#> Merc 280C                    11   1.215737e-02 0.04691083 -0.190099538
#> Merc 450SE                   12   1.423008e-03 0.05619559  0.064280875
#> Merc 450SL                   13   1.458960e-04 0.04123610  0.020560728
#> Merc 450SLC                  14   6.266049e-03 0.04257292 -0.135714533
#> Cadillac Fleetwood           15   2.786686e-05 0.18577112  0.008984366
#> Lincoln Continental          16   1.780910e-02 0.20897838  0.227919348
#> Chrysler Imperial            17   4.236109e-01 0.18648721  1.231668760
#> Fiat 128                     18   1.574263e-01 0.07990991  0.749153703
#> Honda Civic                  19   9.371446e-03 0.12295814  0.165329646
#> Toyota Corolla               20   2.083933e-01 0.09950335  0.865985851
#> Toyota Corona                21   2.791982e-02 0.05303126 -0.292008465
#> Dodge Challenger             22   2.087419e-02 0.03571654 -0.253389811
#> AMC Javelin                  23   2.751510e-02 0.03339813 -0.294709853
#> Camaro Z28                   24   9.943527e-03 0.10298538 -0.170476763
#> Pontiac Firebird             25   1.443199e-02 0.04453321  0.207813200
#> Fiat X1-9                    26   5.920440e-04 0.09225188 -0.041423665
#> Porsche 914-2                27   5.674986e-06 0.07078181 -0.004054382
#> Lotus Europa                 28   7.353985e-02 0.15364135  0.471518032
#> Ford Pantera L               29   8.919701e-03 0.20442386 -0.161026362
#> Ferrari Dino                 30   5.732672e-03 0.06704651 -0.129395315
#> Maserati Bora                31   2.720397e-01 0.39420816  0.907521354
#> Volvo 142E                   32   5.600804e-03 0.04140623 -0.128232538
#>                     std_residual stud_residual is_cook_influential
#> Mazda RX4            -1.01458647   -1.01511928               FALSE
#> Mazda RX4 Wag        -0.62332752   -0.61663091               FALSE
#> Datsun 710           -0.98475880   -0.98422728               FALSE
#> Hornet 4 Drive        0.05332850    0.05240355               FALSE
#> Hornet Sportabout     0.14644776    0.14395389               FALSE
#> Valiant              -0.94769800   -0.94597876               FALSE
#> Duster 360           -0.53305899   -0.52637283               FALSE
#> Merc 240D             0.62035108    0.61364674               FALSE
#> Merc 230              0.32068555    0.31566819               FALSE
#> Merc 280             -0.30786160   -0.30300262               FALSE
#> Merc 280C            -0.86081660   -0.85686281               FALSE
#> Merc 450SE            0.26776519    0.26343390               FALSE
#> Merc 450SL            0.10087865    0.09914150               FALSE
#> Merc 450SLC          -0.65019507   -0.64359476               FALSE
#> Cadillac Fleetwood    0.01914207    0.01880925               FALSE
#> Lincoln Continental   0.44970228    0.44342961               FALSE
#> Chrysler Imperial     2.35451716    2.57247756                TRUE
#> Fiat 128              2.33192251    2.54205995                TRUE
#> Honda Civic           0.44781172    0.44155241               FALSE
#> Toyota Corolla        2.37861784    2.60515163                TRUE
#> Toyota Corona        -1.22297829   -1.23394918               FALSE
#> Dodge Challenger     -1.30026858   -1.31660890               FALSE
#> AMC Javelin          -1.54564189   -1.58546759               FALSE
#> Camaro Z28           -0.50973308   -0.50312646               FALSE
#> Pontiac Firebird      0.96380569    0.96258469               FALSE
#> Fiat X1-9            -0.13220037   -0.12994022               FALSE
#> Porsche 914-2        -0.01494999   -0.01469003               FALSE
#> Lotus Europa          1.10241512    1.10667850               FALSE
#> Ford Pantera L       -0.32270884   -0.31766698               FALSE
#> Ferrari Dino         -0.48919397   -0.48268129               FALSE
#> Maserati Bora         1.11989090    1.12500838                TRUE
#> Volvo 142E           -0.62369214   -0.61699651               FALSE
#>                     is_leverage_influential is_dffits_influential is_outlier
#> Mazda RX4                             FALSE                 FALSE      FALSE
#> Mazda RX4 Wag                         FALSE                 FALSE      FALSE
#> Datsun 710                            FALSE                 FALSE      FALSE
#> Hornet 4 Drive                        FALSE                 FALSE      FALSE
#> Hornet Sportabout                     FALSE                 FALSE      FALSE
#> Valiant                               FALSE                 FALSE      FALSE
#> Duster 360                            FALSE                 FALSE      FALSE
#> Merc 240D                             FALSE                 FALSE      FALSE
#> Merc 230                              FALSE                 FALSE      FALSE
#> Merc 280                              FALSE                 FALSE      FALSE
#> Merc 280C                             FALSE                 FALSE      FALSE
#> Merc 450SE                            FALSE                 FALSE      FALSE
#> Merc 450SL                            FALSE                 FALSE      FALSE
#> Merc 450SLC                           FALSE                 FALSE      FALSE
#> Cadillac Fleetwood                    FALSE                 FALSE      FALSE
#> Lincoln Continental                    TRUE                 FALSE      FALSE
#> Chrysler Imperial                     FALSE                  TRUE      FALSE
#> Fiat 128                              FALSE                  TRUE      FALSE
#> Honda Civic                           FALSE                 FALSE      FALSE
#> Toyota Corolla                        FALSE                  TRUE      FALSE
#> Toyota Corona                         FALSE                 FALSE      FALSE
#> Dodge Challenger                      FALSE                 FALSE      FALSE
#> AMC Javelin                           FALSE                 FALSE      FALSE
#> Camaro Z28                            FALSE                 FALSE      FALSE
#> Pontiac Firebird                      FALSE                 FALSE      FALSE
#> Fiat X1-9                             FALSE                 FALSE      FALSE
#> Porsche 914-2                         FALSE                 FALSE      FALSE
#> Lotus Europa                          FALSE                 FALSE      FALSE
#> Ford Pantera L                         TRUE                 FALSE      FALSE
#> Ferrari Dino                          FALSE                 FALSE      FALSE
#> Maserati Bora                          TRUE                  TRUE      FALSE
#> Volvo 142E                            FALSE                 FALSE      FALSE
#>                     dfbetas__Intercept_    dfbetas_wt   dfbetas_hp
#> Mazda RX4                  -0.161347204  0.0639304305  0.032966471
#> Mazda RX4 Wag              -0.069324050 -0.0004066495  0.045785122
#> Datsun 710                 -0.211199646  0.0972314374  0.043374926
#> Hornet 4 Drive              0.002672687  0.0044886906 -0.006839301
#> Hornet Sportabout           0.001784844 -0.0015536931  0.009208434
#> Valiant                    -0.005985946 -0.1516565139  0.180374447
#> Duster 360                  0.004705177  0.0781031774 -0.159988770
#> Merc 240D                   0.034255292  0.1224118752 -0.189552940
#> Merc 230                    0.019788247  0.0332570461 -0.055075623
#> Merc 280                   -0.003198686 -0.0337297820  0.036709039
#> Merc 280C                  -0.009045583 -0.0953846390  0.103809696
#> Merc 450SE                 -0.026973686  0.0356973740 -0.005712458
#> Merc 450SL                 -0.003961562  0.0049302300  0.003399822
#> Merc 450SLC                 0.031572445 -0.0400515832 -0.016800308
#> Cadillac Fleetwood         -0.006420656  0.0075499557 -0.002577897
#> Lincoln Continental        -0.168791258  0.1903129995 -0.058242601
#> Chrysler Imperial          -0.924056752  0.9355996760 -0.148009806
#> Fiat 128                    0.605181396 -0.1672758566 -0.311246566
#> Honda Civic                 0.156388333 -0.0819144214 -0.034026915
#> Toyota Corolla              0.804669969 -0.4114605894 -0.170934240
#> Toyota Corona              -0.231328587  0.0882138248  0.066064464
#> Dodge Challenger            0.003923967 -0.0888481611  0.049775308
#> AMC Javelin                -0.019610048 -0.0734203131  0.037837437
#> Camaro Z28                  0.029920076  0.0390740055 -0.128670440
#> Pontiac Firebird           -0.058806962  0.0868742949 -0.002278294
#> Fiat X1-9                  -0.037559007  0.0174261386  0.010208853
#> Porsche 914-2              -0.003655931  0.0020588013  0.000316321
#> Lotus Europa                0.423409344 -0.4072338373  0.188396749
#> Ford Pantera L             -0.022536462  0.0999346699 -0.148176049
#> Ferrari Dino               -0.065508308  0.0869804902 -0.085182962
#> Maserati Bora              -0.007482815 -0.4999048760  0.865763737
#> Volvo 142E                 -0.080001907  0.0127537553  0.038406565
#>                     is_influential
#> Mazda RX4                    FALSE
#> Mazda RX4 Wag                FALSE
#> Datsun 710                   FALSE
#> Hornet 4 Drive               FALSE
#> Hornet Sportabout            FALSE
#> Valiant                      FALSE
#> Duster 360                   FALSE
#> Merc 240D                    FALSE
#> Merc 230                     FALSE
#> Merc 280                     FALSE
#> Merc 280C                    FALSE
#> Merc 450SE                   FALSE
#> Merc 450SL                   FALSE
#> Merc 450SLC                  FALSE
#> Cadillac Fleetwood           FALSE
#> Lincoln Continental           TRUE
#> Chrysler Imperial             TRUE
#> Fiat 128                      TRUE
#> Honda Civic                  FALSE
#> Toyota Corolla                TRUE
#> Toyota Corona                FALSE
#> Dodge Challenger             FALSE
#> AMC Javelin                  FALSE
#> Camaro Z28                   FALSE
#> Pontiac Firebird             FALSE
#> Fiat X1-9                    FALSE
#> Porsche 914-2                FALSE
#> Lotus Europa                 FALSE
#> Ford Pantera L                TRUE
#> Ferrari Dino                 FALSE
#> Maserati Bora                 TRUE
#> Volvo 142E                   FALSE
# }
```
