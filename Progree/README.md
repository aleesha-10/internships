# Progree Data Science Internship — Tasks 2, 3 & 4

> **Author:** aleesha-10  
> **Program:** Progree Data Science Internship  
> **Repo:** https://github.com/aleesha-10/internships/tree/main/Progree

Three end-to-end data science projects spanning data engineering, unsupervised machine learning, and time-series forecasting with model interpretability. Each task is a self-contained, fully-executed Jupyter notebook — run top to bottom in Google Colab with zero manual setup (except uploading `Mall_Customers.csv` for Task 3).

---

## Contents

| File | Task | Description |
|---|---|---|
| `Task2_Web_Scraping_Pipeline.ipynb` | Task 2 | Fault-tolerant web scraping pipeline, two live sites |
| `Task3_Customer_Segmentation_Clustering.ipynb` | Task 3 | K-Means + Hierarchical customer segmentation |
| `Task4_TimeSeries_Forecasting_SHAP.ipynb` | Task 4 | Prophet forecasting + SHAP interpretability |
| `README.md` | — | This file |
| `Progree_DataScience_Report.pdf` | — | Full written report covering all three tasks |

---

## Task 2 — Automated Web Scraping & Data Extraction Pipeline

### Objective
Build an automated system pipeline that harvests data records from remote web environments, extracts target parameters, cleans the data, and writes it to a database file or structured CSV.

### Approach
- **Two independent live sources:** quotes.toscrape.com (quote listings) and books.toscrape.com (product listings) — different markup, different parsers, same pipeline shape. Proving generalization, not just hardcoded selectors.
- **Fault-tolerant crawler:** every HTTP request is wrapped in retry logic with exponential backoff + jitter (`max_retries=3, base_delay=1.0s`), so a single dropped connection never kills the whole crawl.
- **Automated pagination:** follows "Next" links automatically until no more pages exist.
- **Cleaning:** strips smart-quote characters (`\u201c/\u201d`) and normalizes internal whitespace across all text fields.
- **Dual persistence:** quotes dataset saved to both CSV and SQLite (table: `quotes`) and validated with a programmatic row-count assertion.
- **Interactive analytics:** Plotly dashboard built inline — top-author bar chart, tag-frequency treemap, price-by-rating box plot, pipeline output donut chart.
- **Run log:** structured JSON telemetry printed at the end of every run (pages crawled, records extracted, duplicates removed, output files).

### Results

| Metric | Value |
|---|---|
| Pages crawled | 10 |
| Records extracted | 100 |
| Duplicates removed | 0 |
| Sources validated | 2 |

**Top authors:** Albert Einstein (10), J.K. Rowling (9), Marilyn Monroe (7), Dr. Seuss (6), Mark Twain (6)

### Outputs
- `quotes_dataset.csv`
- `quotes_dataset.db` (SQLite, table: `quotes`)
- `books_dataset.csv`

### Dependencies
```
requests beautifulsoup4 pandas plotly
```

---

## Task 3 — Unsupervised Customer Segmentation (Clustering)

### Objective
Apply multi-variable clustering models to group customer data by behavior, using PCA, the Elbow Method, and K-Means.

### Dataset
**Mall Customer Segmentation Data** (Kaggle, by Vijay Choudhary) — 200 real customers, features: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`.

> Download from Kaggle (search "Mall Customer Segmentation Data") and upload the CSV to Colab before running the data-loading cell.

### Approach
- Dropped `CustomerID` (non-informative identifier); clustered on **Age, Annual Income, Spending Score**.
- `StandardScaler` normalization before any distance-based computation.
- **PCA** (2 components) for visualization.
- **Elbow Method + silhouette score** swept across k=2–10; both metrics independently converged on **k=4**.
- **K-Means** (n_init=10, random_state=42) final fit.
- **Cross-validation against Hierarchical (Ward-linkage) Clustering** — an independent algorithm — with Adjusted Rand Index measuring inter-algorithm agreement. Dendrogram generated to visually confirm 4 natural splits.
- **Interactive Plotly visuals:** rotatable 3D segment scatter, PCA projection with marginal box distributions, normalized radar profile, segment-size bar chart.
- Cluster profiles translated into **named business personas** with concrete marketing/retention recommendations.

### Results — Segment Profiles

| Persona | Age | Income (k$) | Spend Score | n | Key Action |
|---|---|---|---|---|---|
| Premium High-Value Shoppers | ~33 | $86k | 81.5 | 40 | VIP loyalty, early access |
| Affluent Low Engagers | ~39 | $87k | 19.6 | 38 | Exclusivity re-engagement (biggest opportunity) |
| Young Value Seekers | ~25 | $40k | 60.3 | 57 | Flash sales, referral incentives |
| Mature Steady Spenders | ~54 | $48k | 40.0 | 65 | Value bundles, protect the base |

**K-Means silhouette score (k=4):** confirmed by both Elbow Method and silhouette peak.  
**Adjusted Rand Index:** K-Means and Hierarchical clustering produced highly consistent groupings.

### Outputs
- `customer_segments.csv` — 200 labeled customer records with assigned cluster and persona

### Dependencies
```
scikit-learn scipy pandas numpy matplotlib seaborn plotly
```

---

## Task 4 — Time-Series Forecasting & Model Interpretability Dashboard

### Objective
Fit a time-series forecasting model to historical metrics, evaluate it rigorously, and measure variable impacts using a SHAP interpretability layer.

### Dataset
**AirPassengers** — monthly totals of international airline passengers (thousands), Jan 1949–Dec 1960. Loaded automatically from a public CSV mirror — no manual download needed.

```
https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

### Approach

**Stationarity:**
- ADF test confirmed the raw series is non-stationary (p > 0.05).
- First-order differencing (lag=1) removes trend; seasonal differencing (lag=12) removes the yearly seasonal component — producing a stationary series. These findings inform the choice of multiplicative seasonality mode in Prophet.

**Forecasting:**
- Prophet model fit on all but the final 24 months (`seasonality_mode="multiplicative"`, `interval_width=0.95`).
- Benchmarked against a **naive seasonal baseline** (this month = same month, one year ago) to quantify real model lift.
- Prophet's built-in **rolling-origin cross-validation** (`initial="730 days"`, `period="180 days"`, `horizon="365 days"`) for a robust accuracy estimate beyond a single split.
- **24-month forward forecast** (Jan 1961–Dec 1962) generated by refitting on the full dataset.

**Interpretability:**
Since Prophet is a curve-decomposition model (not feature-based), SHAP can't attach directly. Standard workaround: an **XGBoost surrogate model** is trained on 7 engineered time features that predict the same target; SHAP's TreeExplainer is applied to that surrogate.

Engineered features: `month_num`, `year`, `time_index`, `lag_1`, `lag_12`, `rolling_mean_3`, `rolling_std_3`

### Results

| Metric | Prophet | Naive Baseline |
|---|---|---|
| MAE | 25.33 | (baseline) |
| RMSE | 30.39 | — |
| MAPE | 5.35% | — |
| CI Coverage (95%) | 37.5% | — |

> **Note on CI coverage:** Only 37.5% of holdout actuals fell inside Prophet's 95% CI — an honest finding reported transparently. The post-1958 passenger growth acceleration exceeded the model's uncertainty bounds; widening `interval_width` or adding an explicit changepoint would address this.

### SHAP Feature Importance

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `lag_12` | **133.63** | Dominant — strict year-over-year seasonality |
| 2 | `lag_1` | 7.83 | Short-term momentum |
| 3 | `month_num` | 5.03 | Intra-year seasonal pattern |
| 4 | `time_index` | 2.17 | Long-term trend position |
| 5 | `rolling_std_3` | 1.46 | Recent volatility |
| 6 | `rolling_mean_3` | 1.32 | Short-term smoothed level |
| 7 | `year` | 0.49 | Macro year-level trend |

`lag_12` dominates by a factor of ~17x over the next feature — strong quantitative confirmation that prior-year same-month value, more than recent momentum or trend, drives this series.

### Outputs
- `airpassengers_prophet_forecast.csv`
- `airpassengers_shap_feature_importance.csv`

### Dependencies
```
prophet xgboost shap statsmodels pandas numpy matplotlib plotly
```

---

## How to Run

1. Open any notebook in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Run all** — packages install automatically in the first cell.
3. **Task 3 only:** upload `Mall_Customers.csv` to the Colab session's file browser before running the data-loading cell.
4. Output files (CSV, SQLite, etc.) appear in the Colab file browser and can be downloaded from there.

No local Python environment needed. All notebooks tested on Colab's standard CPU runtime.

---

## Package Summary

| Package | Used In |
|---|---|
| `requests`, `beautifulsoup4` | Task 2 |
| `pandas`, `numpy` | All |
| `matplotlib`, `seaborn` | Tasks 3, 4 |
| `plotly` | All (interactive visuals) |
| `scikit-learn`, `scipy` | Task 3 |
| `prophet` | Task 4 |
| `xgboost`, `shap` | Task 4 |
| `statsmodels` | Task 4 |
