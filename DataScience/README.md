# python-data-projects

A collection of self-contained Python data science projects I built on the side — covering web scraping, unsupervised ML, and time-series forecasting. Each one is a standalone Jupyter notebook, runs fully in Google Colab, and goes a bit beyond the obvious approach.

---

## Projects

###  QuoteCrawler — Automated Web Scraping Pipeline
*Also could be called: WebHarvest · DataMiner · ScrapeSuite · PipelineCrawler · RawDataEngine*

Automated pipeline that crawls two live public sites end-to-end, extracts structured records from unstructured HTML, cleans the raw text, and persists the output to both CSV and SQLite. Built with retry logic and exponential backoff so it handles dropped connections without dying.

**Stack:** `requests` · `BeautifulSoup` · `pandas` · `sqlite3` · `plotly`

**Highlights:**
- Fault-tolerant crawler with retry + backoff on every request
- Validated against two structurally different sites (quotes + books) to prove the architecture generalizes
- Interactive Plotly analytics layer (tag treemap, author rankings, price distributions)
- Structured JSON run log printed at completion — pages crawled, records extracted, duplicates removed

<img width="857" height="204" alt="image" src="https://github.com/user-attachments/assets/6732aca4-3e82-41da-8dca-c3249e01ab17" />
<img width="826" height="212" alt="image" src="https://github.com/user-attachments/assets/a6578b61-edf4-4c5e-b41d-c9096595e3b8" />
<img width="779" height="234" alt="image" src="https://github.com/user-attachments/assets/389f2bc6-3704-4252-99be-8797d91f6a0c" />
<img width="752" height="222" alt="image" src="https://github.com/user-attachments/assets/3ef93223-2b47-4a23-9075-5eb0f92003f1" />

---

###  CustomerLens — Unsupervised Customer Segmentation
*Also could be called: SegmentIQ · ClusterVision · BehaviorMapper · ShopperDNA · MarketSlice*

Clusters real customer data into behaviorally distinct segments using K-Means, cross-validated against Hierarchical clustering. Translates the math into named business personas with concrete marketing recommendations — not just cluster labels.

**Stack:** `scikit-learn` · `scipy` · `plotly` · `seaborn`

**Dataset:** Mall Customer Segmentation (Kaggle, 200 customers)

**Highlights:**
- Elbow Method + silhouette score to find optimal k (converged on k=4)
- Cross-validated K-Means against Agglomerative (Ward) clustering via Adjusted Rand Index
- Interactive 3D segment explorer, radar profile chart, PCA projection with marginals
- 4 named personas with segment-specific retention/marketing strategy per cluster

---

### FlightCast — Time-Series Forecasting & Interpretability
*Also could be called: TrendSense · SeriesLab · ForecastEngine · TemporalIQ · ProphetBoard*

Fits a Prophet forecasting model to historical airline passenger data, evaluates it properly (not just against itself — against a naive baseline too), and explains the forecast using a SHAP interpretability layer via an XGBoost surrogate.

**Stack:** `prophet` · `xgboost` · `shap` · `statsmodels` · `plotly`

<img width="1489" height="440" alt="image" src="https://github.com/user-attachments/assets/acdc2a26-d32b-421e-a181-6f14b54a7ffc" />
<img width="606" height="412" alt="image" src="https://github.com/user-attachments/assets/145d66f3-29fe-4d39-bbda-233ea18ace06" />
<img width="1289" height="440" alt="image" src="https://github.com/user-attachments/assets/dc3c83f3-7d9c-4a47-a5da-0524b79e09c6" />
<img width="1010" height="427" alt="image" src="https://github.com/user-attachments/assets/283a45a1-cdf5-4fdb-882f-267cee0c21c1" />
<img width="629" height="268" alt="image" src="https://github.com/user-attachments/assets/2a95ec86-eb59-4b19-b4bd-8d22d44d7aa8" />
<img width="845" height="229" alt="image" src="https://github.com/user-attachments/assets/aa30e205-a5e3-45ef-880c-7df257267b76" />
<img width="575" height="247" alt="image" src="https://github.com/user-attachments/assets/db9a2280-cdee-4db9-a890-a383085fb536" />
<img width="808" height="242" alt="image" src="https://github.com/user-attachments/assets/9dc63ecd-7549-4474-9921-be6b61a60d43" />

**Dataset:** AirPassengers — monthly international airline passengers, 1949–1960

**Highlights:**
- ADF stationarity test + seasonal differencing pipeline before model fitting
- Prophet benchmarked against a naive seasonal baseline to quantify real model lift
- Rolling-origin cross-validation for a robust accuracy estimate (not just one train/test split)
- 24-month forward forecast beyond the dataset's actual end date
- SHAP TreeExplainer on XGBoost surrogate — `lag_12` found to dominate by 17x over any other feature

  <img width="1005" height="470" alt="image" src="https://github.com/user-attachments/assets/c8b47f7b-1c36-4bf0-b861-88a1a2b5d09c" />
  <img width="1189" height="890" alt="image" src="https://github.com/user-attachments/assets/3168f1d0-5f20-4da4-a525-f68ef9d850e4" />
  <img width="854" height="230" alt="image" src="https://github.com/user-attachments/assets/0b0083e3-ede2-4ae7-9ac5-e85f131ce2da" />
  <img width="437" height="297" alt="image" src="https://github.com/user-attachments/assets/706fc6d6-3dcc-4fe6-a0f4-970e8a5fd86d" />
  <img width="762" height="420" alt="image" src="https://github.com/user-attachments/assets/4a57bf6d-8d2b-452f-85dd-45183880772a" />





---

## How to Run

All notebooks run top to bottom in [Google Colab](https://colab.research.google.com/) — dependencies install in the first cell.

| Notebook | External file needed? |
|---|---|
| `Web_Scraping_Pipeline.ipynb` | No — data fetched live |
| `Customer_Segmentation_Clustering.ipynb` | Yes — upload `Mall_Customers.csv` ([Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)) |
| `TimeSeries_Forecasting_SHAP.ipynb` | No — data fetched live |

---

## Stack Overview

```
python · pandas · numpy · matplotlib · seaborn · plotly
scikit-learn · scipy · statsmodels
prophet · xgboost · shap
requests · beautifulsoup4 · sqlite3
```
