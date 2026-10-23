# Ethics and dataset issues
* **B** = 1000(Bk − 0.63)², Bk = proportion of Black residents by town. Excluded from all deployed artefacts
  (`src/features.py`, API, `web/model.json`; a test asserts this). Measured cost of exclusion: RMSE 3.157 vs 3.130.
* **LSTAT** — "% lower status of the population"; kept with a caveat: a dated, coarse socioeconomic proxy.
* **Censoring** at MEDV = 50; **1970 data**; **towns, not individual houses**.
* Use for teaching only. Do not use for lending, pricing, insurance or any decision about people.
