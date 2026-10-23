# Data
* `data/boston_housing.csv` — 506 tracts × 14 columns (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B,
  LSTAT, MEDV); no missing values or duplicates. A header row was added to the supplied file.
* `data/boston_housing_3features.csv` — 489 rows: RM, LSTAT, PTRATIO, MEDV (MEDV in dollars, rescaled; censored and some
  outlying tracts removed). Used only as a secondary check.
* `MEDV` = median value of owner-occupied homes in $1000s (1970); 16 tracts are capped at 50.
* Origin: Harrison & Rubinfeld (1978). Both files were supplied by the author from commonly circulated copies.
