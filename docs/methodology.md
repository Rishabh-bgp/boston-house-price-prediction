# Methodology
* **Features:** 12 of the 13 inputs (B excluded). Linear models additionally see log1p(CRIM, LSTAT, DIS) in one variant.
* **Validation:** `RepeatedKFold(5, 5, random_state=42)`; metrics on the original scale.
* **Models:** mean baseline, linear, ridge, ridge with log target and log features, random forest (500 trees) ± log target,
  HistGradientBoosting (log target), GradientBoostingRegressor (300 trees, depth 3, lr 0.05, min leaf 5, subsample 0.8)
  with squared and Huber loss on log target.
* **Checks:** censored tracts not scored; with/without B (measurement only); quantile coverage; 80/20 hold-out; permutation
  importance (50 repeats); 3-feature subset.
* **Deployed:** GBR Huber on log target plus 5%/95% quantile GBRs, refit on all 506 tracts by `src/train.py`.
