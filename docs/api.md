# API
`uvicorn app.main:app` → `/docs`. `GET /health`; `POST /predict` with CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,
PTRATIO, LSTAT (validated ranges; out of range → 422). Response: `medv_thousands_usd`, `quantile_range_5_95`, `unit`.
