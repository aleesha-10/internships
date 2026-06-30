# Progree Data Science Internship Tasks — Tasks 2, 3 & 4

This repository contains three self-contained Google Colab notebooks completed for the Progree Data Science internship tasks.

## Contents

| File | Task | Description |
|---|---|---|
| `Task2_Web_Scraping_Pipeline.ipynb` | Task 2 | Automated unstructured web scraping pipeline |
| `Task3_Customer_Segmentation_Clustering.ipynb` | Task 3 | Unsupervised K-Means customer segmentation |
| `Task4_TimeSeries_Forecasting_SHAP.ipynb` | Task 4 | Prophet time-series forecasting + SHAP interpretability |

Each notebook is self-contained: it installs its own dependencies in the first cell and can be run top to bottom independently in Google Colab.

---

## Task 2: Automated Unstructured Web Scraping & Data Extraction Pipeline

**Source:** [quotes.toscrape.com](http://quotes.toscrape.com) — a public sandbox site built for scraping practice.

**Approach:**
- Used `requests` + `BeautifulSoup` to automatically crawl every paginated page (following the "Next" link) with a polite delay between requests.
- Extracted core parameters per listing: quote text, author, author URL, and tags.
- Cleaned whitespace artifacts and smart-quote characters from the raw text.
- Structured the cleaned records into a pandas DataFrame and removed duplicates.

**Outputs:**
- `quotes_dataset.csv` — structured CSV of all scraped quotes
- `quotes_dataset.db` — same data written to a SQLite database (table: `quotes`)

---

## Task 3: Unsupervised Machine Learning Customer Segmentation (Clustering)

**Dataset:** Mall Customer Segmentation dataset (Kaggle, 200 customers) — `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`.

**Approach:**
- Dropped the non-informative `CustomerID` column; clustered on Age, Annual Income, and Spending Score.
- Standardized features with `StandardScaler` and applied PCA for 2D visualization.
- Used the Elbow Method and silhouette scores to select the optimal number of clusters (**k = 5**).
- Fit a final K-Means model and visualized the resulting customer segments in PCA space.

**Segments identified:**
- High income, low spending — affluent but price-conscious
- High income, high spending — premium target segment
- Low income, high spending — impulse-driven shoppers
- Low income, low spending — budget-constrained, low engagement
- Average income, average spending — mainstream "typical" customer

**Output:** `customer_segments.csv` — labeled customer records with assigned cluster.

> To reproduce: download the Mall Customer Segmentation CSV from Kaggle (search "Mall Customer Segmentation Data" by Vijay Choudhary) and upload it to the Colab session before running the data-loading cell.

---

## Task 4: Time-Series Forecasting Model & Model Interpretability Dashboard

**Dataset:** AirPassengers — monthly international airline passenger totals, Jan 1949–Dec 1960 (loaded directly from a public CSV mirror, no manual download needed).

**Approach:**
- Confirmed non-stationarity of the raw series via the Augmented Dickey-Fuller test.
- Applied first-order differencing and seasonal differencing (lag = 12) to achieve stationarity.
- Trained a **Prophet** model (multiplicative seasonality) on all but the final 24 months, forecasting forward with 95% confidence intervals.
- Evaluated forecast accuracy on the holdout period using MAE, RMSE, MAPE, and CI coverage.
- Since Prophet isn't a feature-based model, built an **XGBoost surrogate model** on engineered time features (lag values, rolling stats, calendar month, trend position) predicting the same target, then applied **SHAP** to that surrogate for feature-level interpretability.

**Outputs:**
- `airpassengers_prophet_forecast.csv` — Prophet forecast with confidence intervals
- `airpassengers_shap_feature_importance.csv` — SHAP feature importance ranking

**Deliverable note:** Task 4 also requires a separate 5–6 page whitepaper summarizing methodology, stationarity/differencing findings, forecast performance, and SHAP interpretability results — this is a separate document, not part of the notebook itself.

---

## How to Run

1. Open the desired notebook in Google Colab.
2. Run all cells top to bottom — each notebook installs its required packages in the first cell.
3. For Task 3, upload the Mall Customers CSV when prompted by the data-loading cell.
4. Generated CSV/DB output files will appear in the Colab session's file browser and can be downloaded from there.

## Requirements

- Python 3.x (Colab default runtime)
- Internet access (for package installs and Tasks 2 & 4 data sources)
- Packages auto-installed per notebook: `beautifulsoup4`, `requests`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `prophet`, `xgboost`, `shap`, `statsmodels`
