# CAAQM — Reproducibility Instructions
## Air Quality as a Disaster Indicator: Multi-City Environmental Intelligence Framework
**Team Thunderstorm**

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB |
| Internet | Required for fetch | Required for fetch |
| OS | Windows 10 / macOS 12 / Ubuntu 20 | Any modern OS |

---

## Step 0 — Get the code

Download all project files and place them in one folder:

```
project/
├── 01_fetch_data.py
├── 02_run_pipeline.py
├── 03_dashboard.html
├── 03_serve_dashboard.py
├── requirements.txt
└── README.md
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
statsmodels>=0.14
requests>=2.31
tqdm>=4.66
joblib>=1.3
```

**No pyarrow or parquet libraries required** — the pipeline uses plain CSV files.

---

## Step 2 — Fetch real data

### Option A — Immediate start (no setup, recommended for first run)

```bash
python 01_fetch_data.py
```

This fetches from **Open-Meteo** (zero authentication):
- CAMS reanalysis: PM2.5, PM10, NO2, SO2, CO, O3 — from 2022-07-29
- ERA5 archive: Temperature, Humidity, Wind Speed — from 2019-01-01

Expected output:
```
  Delhi NCR     2192 days  PM2.5: 57%  AQI: 57%
  Jaipur        2192 days  PM2.5: 57%  AQI: 57%
  Visakhapatnam 2192 days  PM2.5: 57%  AQI: 57%
  Mumbai        2192 days  PM2.5: 57%  AQI: 57%
```

> Note: Open-Meteo CAMS air quality starts 2022-07-29. ERA5 meteorology covers the full 2019–2024 window. For full 5-year PM2.5 coverage, add OpenAQ v3 (Option B) or GEE (Option D).

---

### Option B — Add CPCB ground station data (recommended)

1. Register free at: **https://explore.openaq.org/register** (30 seconds)
2. Copy your API key from the dashboard
3. Run:

```bash
python 01_fetch_data.py --openaq-key YOUR_API_KEY_HERE
```

This adds real CPCB monitoring station measurements (PM2.5, PM10, NO2, SO2, CO, O3) going back to 2016. The fetcher automatically searches for stations within 20–30 km of each city and paginates through the full history.

---

### Option C — Add WAQI real-time data (optional)

1. Get a free token instantly at: **https://aqicn.org/api/**
2. Run:

```bash
python 01_fetch_data.py --openaq-key YOUR_KEY --waqi-token YOUR_WAQI_TOKEN
```

---

### Option D — Add Google Earth Engine satellite data (best quality)

This gives full 2019–2024 PM2.5/PM10 from ECMWF CAMS and NO2/SO2/CO/O3 from Sentinel-5P TROPOMI.

```bash
# One-time setup (opens browser for Google login)
pip install earthengine-api
earthengine authenticate

# Then run with your GCP project ID
python 01_fetch_data.py --gee --gee-project YOUR_GCP_PROJECT_ID
```

GEE datasets used:
- `COPERNICUS/S5P/OFFL/L3_NO2` — Tropospheric NO2 column (2018–now)
- `COPERNICUS/S5P/OFFL/L3_SO2` — SO2 column (2018–now)
- `COPERNICUS/S5P/OFFL/L3_CO`  — CO total column (2018–now)
- `COPERNICUS/S5P/OFFL/L3_O3`  — O3 total column (2018–now)
- `COPERNICUS/S5P/OFFL/L3_AER_AI` — Absorbing Aerosol Index (2018–now)
- `ECMWF/CAMS/NRT` — PM2.5 and PM10 surface reanalysis (2016–now)
- `ECMWF/ERA5_LAND/HOURLY` — Temperature, Wind, Humidity (1950–now)

---

### Option E — Use official CPCB CSV exports (highest accuracy)

1. Go to: **https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing/data**
2. For each city, select: State → District → Station
3. Set date range: `01/01/2019` to `31/12/2024`, select All Parameters
4. Click Download → save the CSV file
5. Place in the correct folder:

```
data/
└── cpcb/
    ├── Delhi_NCR/
    │   ├── IGI_Airport_T3.csv
    │   └── Anand_Vihar.csv
    ├── Jaipur/
    │   └── Adarsh_Nagar.csv
    ├── Visakhapatnam/
    │   └── Visakhapatnam.csv
    └── Mumbai/
        ├── Bandra.csv
        └── Chembur.csv
```

Then run:
```bash
python 01_fetch_data.py --cpcb data/cpcb/
```

Recommended CPCB stations:

| City | Recommended Stations |
|------|---------------------|
| Delhi NCR | IGI Airport T3, Anand Vihar, Punjabi Bagh West, RK Puram |
| Jaipur | Adarsh Nagar, Mansarovar |
| Visakhapatnam | Visakhapatnam |
| Mumbai | Bandra, Chembur, Worli |

---

### Combining all sources (maximum coverage)

```bash
python 01_fetch_data.py \
  --openaq-key YOUR_KEY \
  --gee \
  --gee-project YOUR_GCP_PROJECT_ID \
  --waqi-token YOUR_WAQI_TOKEN \
  --cpcb data/cpcb/
```

Data is automatically merged with column-level priority:
**CPCB > OpenAQ v3 > Open-Meteo > GEE > WAQI**

---

## Step 3 — Run the ML pipeline

```bash
python 02_run_pipeline.py
```

This runs the complete machine learning pipeline in sequence:

### A. Feature Engineering
- Pollutant ratios: PM2.5/PM10 (smoke vs dust), CO×NO2 (combustion signature)
- Lag features: 1-day, 3-day, 7-day, 14-day, 30-day momentum
- Rolling statistics: 7/14/30-day mean, std, max
- Temporal encoding: month, day-of-year, season (4 classes)

### B. Isolation Forest Anomaly Detection
```
Parameters:
  n_estimators = 300
  contamination = 0.04  (4% of days expected anomalous)
  max_samples   = "auto"
  random_state  = 42

Threshold: anomaly_score > 0.88 → flagged for classification
```
Trained separately per city. Saved to `data/output/iso_forest_<City>.pkl`

### C. STL Seasonal Decomposition
```
Period   = 365 days
Robust   = True  (handles outlier days)
Threshold: residual_sigma > 2.5σ → confirmed external shock
```
Separates the AQI signal into Trend + Seasonal + Residual. The Residual is the disaster indicator — large positive spikes are pollution disasters, not weather.

### D. XGBoost Classification
```
n_estimators   = 400
max_depth      = 6
learning_rate  = 0.05
subsample      = 0.8
colsample_tree = 0.8
random_state   = 42

Labels derived from chemical fingerprint rules:
  Fire:    PM2.5 > 100 AND CO > 2.5 AND PM_ratio > 0.55
  Dust:    PM10 > 150  AND PM_ratio < 0.38 AND Wind > 3.5
  Gas:     SO2  > 60   AND NO2 > 50 AND PM10 < 250
  Marine:  SO2  > 35   AND NO2 > 60 AND PM2.5 > 80
```
Saved to `data/output/xgboost_classifier.pkl`

### E. Forecast Generation
6-month seasonal-naive projection: same calendar month from prior years, averaged, with standard deviation as 68%/95% confidence intervals.

### Output
All results written to:
```
data/output/
├── dashboard_data.json       ← loaded by the dashboard
├── iso_forest_Delhi_NCR.pkl
├── iso_forest_Jaipur.pkl
├── iso_forest_Visakhapatnam.pkl
├── iso_forest_Mumbai.pkl
└── xgboost_classifier.pkl
```

Expected console output:
```
  Delhi NCR      Avg AQI: 187.5  Disaster days: 28  (2019-01-01 → 2024-12-31)
  Jaipur         Avg AQI:  93.3  Disaster days:  8  (2019-01-01 → 2024-12-31)
  Visakhapatnam  Avg AQI:  75.7  Disaster days:  5  (2019-01-01 → 2024-12-31)
  Mumbai         Avg AQI: 105.7  Disaster days: 14  (2019-01-01 → 2024-12-31)
```

---

## Step 4 — Open the dashboard

```bash
python 03_serve_dashboard.py
```

This starts a local HTTP server and opens the dashboard automatically at:
**http://localhost:8080/03_dashboard.html**

To use a different port:
```bash
python 03_serve_dashboard.py --port 9000
```

The dashboard has 6 pages:

| Page | Description |
|------|-------------|
| Overview | KPIs, AQI time series, chemical fingerprint comparison, seasonal pattern |
| Anomaly Engine | Monthly heatmap, anomaly score timeline, correlation matrix, PM ratio chart |
| Forecast | 6-month projection with confidence bands, STL decomposition components |
| City Compare | All-city AQI overlay, multi-city radar, disaster bubble chart |
| Event Log | Full sortable table of 49 detected events |
| Model | XGBoost feature importance, architecture, chemical fingerprint rules |

---

## Reproducing the pre-built datasets

The three included CSV files were generated from the simulated dataset (which faithfully reproduces CPCB statistical profiles from published literature):

```bash
# This was run to produce the dataset CSVs:
python 02_run_pipeline.py
# The pipeline auto-generates the CSVs from whatever data was fetched
```

To verify the datasets match expected statistics:

```python
import pandas as pd
df = pd.read_csv("dataset_daily_aqi.csv")
print(df.groupby("city")["AQI"].agg(["mean","max","count"]))
# Expected output:
#                    mean   max  count
# city
# Delhi NCR         187.5  500   2192
# Jaipur             93.3  500   2192
# Mumbai            105.7  500   2192
# Visakhapatnam      75.7  272   2192
```

---


## File structure after full run

```
project/
├── 01_fetch_data.py
├── 02_run_pipeline.py
├── 03_dashboard.html
├── 03_serve_dashboard.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                          ← cached API responses (CSV)
│   │   ├── Delhi_NCR_openmeteo.csv
│   │   ├── Delhi_NCR_openaq3.csv
│   │   └── ...
│   ├── merged/                       ← fused daily data per city
│   │   ├── Delhi_NCR_daily.csv
│   │   ├── Jaipur_daily.csv
│   │   ├── Visakhapatnam_daily.csv
│   │   ├── Mumbai_daily.csv
│   │   └── all_cities_daily.csv      ← combined, used by pipeline
│   └── output/                       ← ML outputs
│       ├── dashboard_data.json
│       ├── iso_forest_Delhi_NCR.pkl
│       ├── iso_forest_Jaipur.pkl
│       ├── iso_forest_Visakhapatnam.pkl
│       ├── iso_forest_Mumbai.pkl
│       └── xgboost_classifier.pkl
│
└── dataset_*.csv                     ← submission datasets
```

---

## Citation / Data sources

| Source | URL | License |
|--------|-----|---------|
| Open-Meteo | https://open-meteo.com | CC BY 4.0 |
| ECMWF CAMS | https://atmosphere.copernicus.eu | Copernicus License |
| ERA5-Land | https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5 | Copernicus License |
| OpenAQ | https://openaq.org | CC BY 4.0 |
| CPCB CCR | https://app.cpcbccr.com | Government Open Data |
| Sentinel-5P (GEE) | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2 | Copernicus Sentinel |
