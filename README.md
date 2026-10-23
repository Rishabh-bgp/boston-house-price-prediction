<div align="center">

# 🏡 Boston House Price Prediction — Machine Learning Regression in Python (scikit-learn, FastAPI) with a Live In-Browser App

[![Live app](https://img.shields.io/badge/live%20app-GitHub%20Pages-8b1e1e?logo=github)](https://rishabh-bgp.github.io/boston-house-price-prediction/)
[![CI](https://github.com/Rishabh-bgp/boston-house-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishabh-bgp/boston-house-price-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.161-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Fairness](https://img.shields.io/badge/race%20feature%20B-excluded-1f2f4a)
![License](https://img.shields.io/badge/license-MIT-e8a33d)

**[▶ Open the live app](https://rishabh-bgp.github.io/boston-house-price-prediction/)**

<img src="docs/img/demo-desktop.png" alt="Screenshot of the Boston house price estimator: sunset header with an original colonial house, tree and striped flag illustration; neighbourhood sliders; an estimate of $20,669 in 1970 dollars with a quantile range and a histogram of all 506 tracts" width="760"/>

</div>

An end-to-end **machine learning regression project** that predicts the **median home value of Boston census tracts**
from the classic **Boston housing dataset**: EDA, **repeated K-fold cross-validation** of linear, ridge, random-forest
and gradient-boosting models, **log-target** and outlier-robust (**Huber**) training, censoring and fairness checks,
**permutation importance**, quantile prediction ranges, a **FastAPI** REST API, **Docker**, **pytest**, **GitHub Actions
CI** and a web app on **GitHub Pages** where the gradient-boosted trees run in your browser.

## 🎓 About this project
Developed as part of my training with **KF Property Network Pte Ltd (KFPN)**, a real estate agency associate business in
Singapore operating as a subsidiary of Knight Frank Singapore — a machine learning model for Boston, USA — and submitted as
my capstone for the **6-Month Joint Certification Program in AI & ML by [NIELIT Ropar](https://www.nielit.gov.in/ropar/)
with [IIT Ropar](https://www.iitrpr.ac.in/)**:

* Live, interactive sessions with NIELIT faculty
* Expert guest lectures from IIT Ropar professors on Natural Language Processing, Deep Learning and Neural Networks
* Hands-on lab access and project-based teaching
* A campus immersion visit to IIT Ropar

> **Disclaimer:** this is a personal portfolio project using a public dataset. It is not an official product of, and is
> not endorsed by, KFPN, Knight Frank or any other company, and it is not a property valuation service.

## ⚖️ Ethics: why `B` is excluded, and other known issues
* **`B` encodes race.** It is defined as 1000(Bk − 0.63)², where Bk is the proportion of Black residents by town. Using it
  would make predictions depend on a neighbourhood's racial make-up, so it is **removed from the deployed model, the API
  and the web app**. The notebook measures what that costs (section 3b): RMSE **3.157** without B vs **3.130** with it —
  within fold-to-fold noise.
* **`LSTAT`** (% "lower status" population) is kept: it is the strongest predictor and a neighbourhood socioeconomic
  measure, but it is a proxy that can carry historical inequality, and its definition is dated. Read its effect as a
  description of 1970 data, not a judgement about people.
* **Censoring:** `MEDV` is capped at 50 ($50,000) for 16 tracts, so the highest values are unknown.
* **Age of the data:** 1970 census tracts; values are in 1970 dollars and do not describe today's market.
* The dataset was removed from scikit-learn's built-in loaders because of these issues; it is used here for teaching,
  with the issues stated.

## 📈 Results
Repeated 5-fold CV (5 repeats = 25 folds), 506 tracts, 12 features (B excluded), metrics in $1000s, **mean ± std** —
copied from the executed notebook:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| GBR, squared loss, log target | 3.109 ± 0.423 | 2.116 ± 0.221 | 0.880 ± 0.030 |
| **GBR, Huber loss, log target (deployed)** | **3.157 ± 0.414** | **2.119 ± 0.202** | **0.877 ± 0.030** |
| HistGradientBoosting, log target | 3.330 ± 0.501 | 2.208 ± 0.240 | 0.863 ± 0.035 |
| Random forest | 3.359 ± 0.455 | 2.255 ± 0.249 | 0.860 ± 0.040 |
| Random forest, log target | 3.514 ± 0.453 | 2.290 ± 0.239 | 0.848 ± 0.037 |
| Ridge, log target + log features | 4.248 ± 0.509 | 2.978 ± 0.291 | 0.778 ± 0.052 |
| Ridge | 4.869 ± 0.540 | 3.454 ± 0.321 | 0.709 ± 0.059 |
| Linear regression | 4.870 ± 0.538 | 3.459 ± 0.320 | 0.709 ± 0.059 |
| Mean baseline | 9.174 ± 0.850 | 6.668 ± 0.646 | −0.015 ± 0.021 |

* **Honest reading:** the two GBR variants are tied within one standard deviation; the Huber version is deployed for
  robustness to outliers, not because it scored best.
* **Censored tracts not scored:** deployed model RMSE 2.801 ± 0.348 (R² 0.866).
* **Quantile range:** the 5%/95% models cover **76.9% ± 4.5%** of held-out tracts — below the nominal 90%, so the app
  calls it an indicative range. Median width $6,970.
* **Single 80/20 hold-out:** RMSE 2.932, MAE 1.918, R² 0.883.
* **Permutation importance** (increase in RMSE, $1000s): LSTAT 4.144, RM 2.797, DIS 0.864, AGE 0.447, CRIM 0.439,
  NOX 0.403, PTRATIO 0.300, TAX 0.168, RAD 0.064, INDUS 0.032, ZN 0.017, CHAS 0.003.
* **3-feature subset** (second file, 489 rows, MEDV in dollars): deployed-style model R² 0.828 ± 0.031 vs ridge 0.704 ± 0.041.
* **Browser ↔ Python parity:** on the live site, JavaScript predictions match scikit-learn for all 506 tracts
  (max |Δ| ≈ 4 × 10⁻¹³, in $1000s, for the point and both quantile models).

## 🖼️ Screenshots
| Desktop | Preset: highest-LSTAT tract | Mobile |
|---|---|---|
| <img src="docs/img/demo-desktop.png" alt="Desktop view" width="300"/> | <img src="docs/img/demo-desktop-2.png" alt="Estimate for the highest-LSTAT tract" width="300"/> | <img src="docs/img/demo-mobile.png" alt="Mobile view" width="140"/> |

## 🏗️ Architecture
```mermaid
flowchart LR
  A[data/boston_housing.csv] --> B[src/features.py<br/>12 features, B excluded]
  B --> C[src/train.py<br/>GBR Huber + 5%/95% quantile GBR on log MEDV]
  C --> D[models/model.joblib]
  C --> E[web/model.json<br/>300 trees × 3 models]
  D --> F[app/main.py · FastAPI /predict]
  E --> G[web/app.js · tree traversal in JS]
  G --> H[GitHub Pages]
  F --> I[Docker / compose]
```

## 📂 Repository structure
```
notebooks/            self-contained notebook (own README, requirements, data copies)
src/                  features.py, train.py (training + JSON export), predict.py (pure-Python JSON inference)
app/main.py           FastAPI: GET /health, POST /predict
tests/                pytest: API, validation, B exclusion, monotonic check, JSON-vs-sklearn parity
web/                  static app (index.html, style.css, app.js, model.json, data.json)
data/                 boston_housing.csv (506 × 14) and boston_housing_3features.csv (489 × 4)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml · gh-pages branch = published copy of web/
docs/                 extended documentation
```

## 🚀 Run locally
```bash
git clone https://github.com/Rishabh-bgp/boston-house-price-prediction.git && cd boston-house-price-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train && pytest -q
uvicorn app.main:app --reload      # http://localhost:8000/docs
python -m http.server -d web 8080  # http://localhost:8080
```

## 🐳 Docker
`docker compose up --build` → API on :8000, web app on :8080.

## 🔌 API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"CRIM":0.25651,"ZN":0,"INDUS":9.69,"CHAS":0,"NOX":0.538,"RM":6.2085,"AGE":77.5,"DIS":3.20745,"RAD":5,"TAX":330,"PTRATIO":19.05,"LSTAT":11.36}'
```
Returns `medv_thousands_usd`, `quantile_range_5_95` and the unit. A `B` field, if sent, is ignored. See [docs/api.md](docs/api.md).

## 📚 Documentation
[data](docs/data.md) · [ethics](docs/ethics.md) · [methodology](docs/methodology.md) · [results](docs/results.md) ·
[API](docs/api.md) · [web app](docs/web-app.md) · [deployment](docs/deployment.md) · [FAQ](docs/faq.md)

## 📝 Data citation
Harrison, D. & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of Environmental
Economics and Management*, 5, 81–102. `boston_housing_3features.csv` is a commonly circulated teaching subset
(RM, LSTAT, PTRATIO; MEDV rescaled).

## ⚖️ Licence
Code and docs: MIT © 2026 Rishabh Aryan. Data: see above (not covered by the MIT licence).

## 🏠 Real Estate ML Trilogy

Three capstone projects from my training with KF Property Network Pte Ltd (KFPN), each predicting home values in a different market with the same honest workflow: repeated K-fold CV, a log target, permutation importance, FastAPI, Docker, CI and an in-browser app.

| Part | Repository | Market and data | Live app | Deployed model, repeated K-fold CV |
|---|---|---|---|---|
| I | [real-estate-valuation-taiwan](https://github.com/Rishabh-bgp/real-estate-valuation-taiwan) | Xindian, New Taipei: 414 sales, 2012–13 | [▶ open](https://rishabh-bgp.github.io/real-estate-valuation-taiwan/) | R² 0.715, RMSE 7.136, MAE 4.551 (10k TWD per ping) |
| II | [boston-house-price-prediction](https://github.com/Rishabh-bgp/boston-house-price-prediction) | Boston: 506 census tracts, 1970 (race feature B excluded) | [▶ open](https://rishabh-bgp.github.io/boston-house-price-prediction/) | R² 0.877, RMSE 3.157, MAE 2.119 ($1000s) |
| III | [california-house-price-prediction](https://github.com/Rishabh-bgp/california-house-price-prediction) | California: 20,640 block groups, 1990 census | [▶ open](https://rishabh-bgp.github.io/california-house-price-prediction/) | R² 0.840, RMSE 0.461, MAE 0.291 ($100k) |

## 🔗 Other repositories by Rishabh Aryan

| Repository | Description | Live |
|---|---|---|
| [100-days-of-python](https://github.com/Rishabh-bgp/100-days-of-python) | My original, offline, no-API-key solutions to all 100 projects of the 100 Days of Code Python bootcamp curriculum… | — |
| [Agentic-Engineering-Ebook](https://github.com/Rishabh-bgp/Agentic-Engineering-Ebook) | Agentic Engineering is a practical notebook-book on specifying, generating, verifying, and refining multi-tenant… | — |
| [beed-ml-study](https://github.com/Rishabh-bgp/beed-ml-study) | Provenance audit and leakage-aware benchmark of seizure-type classifiers on the Bangalore EEG Epilepsy Dataset — IIT… | — |
| [Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction](https://github.com/Rishabh-bgp/Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction) | Brain–computer interfaces (BCIs) link neural activity to external devices, translating brain signals via algorithms… | — |
| [building-a-virtual-cell-in-python](https://github.com/Rishabh-bgp/building-a-virtual-cell-in-python) | A cell has a normal gene-activity pattern. Knock out one gene and that pattern changes. This project trains a small… | — |
| [california-house-price-prediction](https://github.com/Rishabh-bgp/california-house-price-prediction) | California house price prediction with machine learning: scikit-learn gradient boosting, repeated K-fold and spatial… | [open](https://rishabh-bgp.github.io/california-house-price-prediction/) |
| [Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML](https://github.com/Rishabh-bgp/Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML) | ML is transforming Chronic Kidney Disease (CKD) detection in HIV patients, catching early asymptomatic stages to… | — |
| [cobratate](https://github.com/Rishabh-bgp/cobratate) | Cobratate, an esoteric imperative programming language. Tate-flavored syntax, ordinary semantics, and a Python… | [open](https://rishabh-bgp.github.io/cobratate/) |
| [coda-synth](https://github.com/Rishabh-bgp/coda-synth) | CoDA-Synth: stylometric detection of LLM-rewritten dark-web documents with sparse TF-IDF + LinearSVC. Research lab,… | [open](https://shale-ember-delta-kite.grok.me/) |
| [IIT-ROPAR-GENAI-FOUNDATIONS](https://github.com/Rishabh-bgp/IIT-ROPAR-GENAI-FOUNDATIONS) | Free open-source Generative AI foundations ebooks (2026) covering Claude, Grok, Gemini, Cursor, OpenAI and more —… | — |
| [iris-classification](https://github.com/Rishabh-bgp/iris-classification) | Iris flower classification in Python: scikit-learn model comparison, tuning and evaluation, FastAPI /predict API,… | [open](https://rishabh-bgp.github.io/iris-classification/) |
| [kbc-game](https://github.com/Rishabh-bgp/kbc-game) | Kaun Banega Crorepati (KBC) quiz game for the terminal in Python: 16-question prize ladder, safe levels and 4… | — |
| [knn-classifier](https://github.com/Rishabh-bgp/knn-classifier) | Simple K-Nearest Neighbors classifier on the classic Iris dataset using scikit-learn. Loads data, trains the model,… | — |
| [linear-regression-model-](https://github.com/Rishabh-bgp/linear-regression-model-) | Supervised linear regression on the scikit-learn diabetes dataset. A single feature (BMI) is used to predict… | — |
| [logistic-regression](https://github.com/Rishabh-bgp/logistic-regression) | This graduate capstone applies multi-class logistic regression to the Iris dataset. It includes EDA, stratified… | — |
| [logistics-website-capstone](https://github.com/Rishabh-bgp/logistics-website-capstone) | Logistics website capstone: Kargonaut Logistics, a premium freight site mixing claymorphism, glassmorphism and… | [open](https://rishabh-bgp.github.io/logistics-website-capstone/) |
| [lovelight-leap](https://github.com/Rishabh-bgp/lovelight-leap) | 💖 Lovelight Leap: a love-themed 2D platformer in vanilla JS canvas. 4 worlds, a boss you cheer up, original… | [open](https://rishabh-bgp.github.io/lovelight-leap/) |
| [machine-learning-part-1](https://github.com/Rishabh-bgp/machine-learning-part-1) | Machine learning in Python with scikit-learn: heart disease classification and medical insurance cost regression… | — |
| [Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail](https://github.com/Rishabh-bgp/Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail) | Agentic commerce lets shopping and seller agents place orders without showing operational truth. This IS case argues… | — |
| [morocco-city-aware-load-forecasting](https://github.com/Rishabh-bgp/morocco-city-aware-load-forecasting) | City-aware short-term load forecasting and peak detection on high-resolution Moroccan smart-meter data (5 zones).… | — |
| [NIELIT-DATA-SCIENCE](https://github.com/Rishabh-bgp/NIELIT-DATA-SCIENCE) | Hands-on NumPy, Pandas, and Matplotlib Jupyter notebooks for the NIELIT Data Science curriculum — practical,… | — |
| [NIELIT-MACHINE-LEARNING-PROJECTS](https://github.com/Rishabh-bgp/NIELIT-MACHINE-LEARNING-PROJECTS) | Nineteen end-to-end machine learning projects: classification, regression, clustering, and recommendation, with… | — |
| [NIELIT-PYTHON-](https://github.com/Rishabh-bgp/NIELIT-PYTHON-) | Open-source Python teaching notebooks: a language reference and five capstone programs, with detailed documentation.… | — |
| [pan-cancer-classification](https://github.com/Rishabh-bgp/pan-cancer-classification) | Leakage, equivalence, conformal reliability and data efficiency on the UCI TCGA Pan-Cancer RNA-Seq dataset — IIT… | — |
| [Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks](https://github.com/Rishabh-bgp/Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks) | This notebook details the implementation and evaluation of a Bayesian perspective Early stage Event Prediction… | — |
| [prediction-as-power](https://github.com/Rishabh-bgp/prediction-as-power) | Laboratory and working paper on prediction as power in European electricity markets. A DE-like hourly simulator… | — |
| [real-estate-valuation-taiwan](https://github.com/Rishabh-bgp/real-estate-valuation-taiwan) | Taiwan real estate valuation (Xindian, New Taipei): house price per ping prediction with scikit-learn gradient… | [open](https://rishabh-bgp.github.io/real-estate-valuation-taiwan/) |
| [snakes-and-ladders](https://github.com/Rishabh-bgp/snakes-and-ladders) | 🐍🪜 Original Snakes & Ladders browser game: hand-drawn SVG snakes and ladders, 3D dice, 2-4 humans or bots, sound,… | [open](https://rishabh-bgp.github.io/snakes-and-ladders/) |
| [spam-scan](https://github.com/Rishabh-bgp/spam-scan) | Explainable spam, phishing & Indian-scam detector for SMS/email (English, Hindi, Hinglish): TF-IDF + Logistic… | [open](https://spam-scan.onrender.com) |
| [vedic-math-hybrid-kt](https://github.com/Rishabh-bgp/vedic-math-hybrid-kt) | Hybrid Vedic Math AI: rule-based sutras (Urdhva-Tiryagbhyam, Nikhilam, Yavadunam) plus RF classifier that cleans… | — |
| [wind-turbine-power-model](https://github.com/Rishabh-bgp/wind-turbine-power-model) | Single-turbine power curves + wind-farm simulation. Weibull wind stats, hourly NASA-style speed data, and… | — |
| [zoo-website-capstone](https://github.com/Rishabh-bgp/zoo-website-capstone) | Capstone project: a fictional zoo website built twice — a beginner-style simple version and a cinematic advanced… | [open](https://rishabh-bgp.github.io/zoo-website-capstone/) |
