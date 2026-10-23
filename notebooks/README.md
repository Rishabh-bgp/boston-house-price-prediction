# Notebooks — Boston house price prediction

`boston_house_prices.ipynb` is **self-contained**: it imports nothing from `src/` or `app/` and reads its own copies of
the data in `notebooks/data/` (`boston_housing.csv`, 506 tracts × 14 columns; `boston_housing_3features.csv`, 489 rows).

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook boston_house_prices.ipynb
```
Contents: EDA and known dataset issues → repeated 5-fold CV of 9 models → censoring check → cost of excluding `B`
(measurement only) → quantile-range coverage → hold-out diagnostics → permutation importance → 3-feature subset → conclusions.
The `B` column is never used as a model input except in the single, clearly labelled comparison in section 3b.
