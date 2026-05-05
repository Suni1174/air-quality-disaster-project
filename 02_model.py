"""
=============================================================
STEP 2 — Anomaly Detection + Event Classification + Forecasting
Team Thunderstorm | AQI Disaster Intelligence

Input : data/processed/features.parquet
Output: data/processed/results.parquet
        models/isolation_forests.pkl
        models/xgb_classifier.pkl
        models/xgb_forecast.pkl
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.tsa.seasonal import STL
import xgboost as xgb

PROCESSED = Path("data/processed")
MODELS    = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "Normal", 1: "Fire", 2: "Dust", 3: "Gas", 4: "Industrial"}

# Features available from your actual GEE data
ANOMALY_FEATS = [
    "NO2", "SO2", "CO", "O3", "PM2_5", "PM10", "AOD",
    "ratio_pm25_pm10", "ratio_co_no2", "ratio_so2_no2",
    "co_x_no2", "so2_x_no2", "aod_x_wind", "fire_x_no2",
    "PM2_5_roll6_std", "NO2_roll6_std", "SO2_roll6_std",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "fire_count",
]

CLASSIFIER_FEATS = ANOMALY_FEATS + [
    "PM2_5_lag1", "PM2_5_lag6", "PM2_5_lag24",
    "NO2_lag1",   "NO2_lag6",   "NO2_lag24",
    "SO2_lag1",   "SO2_lag6",
    "CO_lag1",    "CO_lag6",
    "AOD_lag1",   "AOD_lag24",
    "AQI_roll6_mean", "AQI_roll24_mean",
    "temp_c", "humidity", "wind_speed",
    "wind_u", "wind_v", "precip", "pressure",
    "is_weekend",
]

FORECAST_FEATS = [
    "PM2_5_lag1", "PM2_5_lag6", "PM2_5_lag24",
    "PM2_5_roll6_mean", "PM2_5_roll24_mean", "PM2_5_roll6_std",
    "AOD_lag1", "AOD_lag24",
    "NO2_lag1", "SO2_lag1", "CO_lag1",
    "fire_count",
    "temp_c", "humidity", "wind_speed", "wind_u", "wind_v",
    "precip", "pressure", "cloud_pct",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend",
]


# ═══════════════════════════════════════════════════════════════
# TASK 1: Isolation Forest
# ═══════════════════════════════════════════════════════════════

def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    print("  → Isolation Forest …")
    df = df.copy()
    df["anomaly_score"] = 0.0
    models = {}

    for city, grp in df.groupby("city"):
        feats = [f for f in ANOMALY_FEATS if f in grp.columns]
        X = grp[feats].fillna(0).values

        clf = IsolationForest(
            n_estimators=300,
            contamination=0.025,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X)
        raw   = clf.decision_function(X)
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        df.loc[grp.index, "anomaly_score"] = score
        models[city] = clf

        flagged = (score > 0.88).sum()
        print(f"     {city}: {flagged:,} timestamps flagged "
              f"({flagged/len(score)*100:.1f}%)")

    joblib.dump(models, MODELS / "isolation_forests.pkl")
    df["is_anomaly"] = df["anomaly_score"] > 0.88
    return df


# ═══════════════════════════════════════════════════════════════
# TASK 2: STL Residual Spike Detection
# ═══════════════════════════════════════════════════════════════

def stl_residual(series: pd.Series, period=24*7) -> pd.Series:
    s = series.ffill().bfill().fillna(0)
    if len(s) < 2 * period:
        return pd.Series(np.zeros(len(s)), index=series.index)
    try:
        res = STL(s, period=period, robust=True).fit().resid
        return pd.Series(res.values, index=series.index)
    except Exception:
        return s - s.rolling(24, center=True, min_periods=1).mean()


def run_stl(df: pd.DataFrame, sigma=3.0) -> pd.DataFrame:
    print("  → STL residual detection (daily resampling for speed) …")
    frames = []

    for city, grp in df.groupby("city"):
        grp = grp.sort_values("datetime").copy()

        for pol in ["PM2_5", "NO2", "SO2", "AOD"]:
            if pol not in grp.columns:
                continue

            # ── Downsample hourly → daily mean (fast STL) ──
            daily = (grp.set_index("datetime")[pol]
                       .resample("D").mean()
                       .ffill().bfill().fillna(0))

            if len(daily) < 60:
                grp[f"{pol}_residual"] = 0.0
                continue

            # STL on ~2190 daily points is 10-20× faster than hourly
            try:
                result   = STL(daily, period=7, robust=True).fit()
                residual = pd.Series(result.resid.values, index=daily.index)
            except Exception:
                residual = daily - daily.rolling(7, min_periods=1).mean()

            # ── Map daily residual back to hourly rows ──
            grp["_date"] = grp["datetime"].dt.normalize()
            res_map      = residual.rename("_res")
            grp          = grp.merge(res_map, left_on="_date",
                                     right_index=True, how="left")
            grp[f"{pol}_residual"] = grp["_res"].fillna(0)
            grp.drop(columns=["_date", "_res"], inplace=True)

        # Spike flag: residual > sigma × rolling std
        spike_flags = pd.DataFrame(index=grp.index)
        for pol in ["PM2_5", "NO2", "SO2", "AOD"]:
            col_r = f"{pol}_residual"
            if col_r not in grp.columns:
                continue
            rstd = grp[col_r].rolling(72, min_periods=12).std().fillna(1)
            spike_flags[pol] = (grp[col_r] > sigma * rstd).values

        grp["stl_spike"] = spike_flags.any(axis=1)
        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    print(f"     STL spikes: {result['stl_spike'].sum():,} timestamps")
    return result

def combine_evidence(df: pd.DataFrame) -> pd.DataFrame:
    df["confirmed_event"] = (df["anomaly_score"] > 0.82) & df["stl_spike"]
    n = df["confirmed_event"].sum()
    print(f"     Confirmed events: {n:,} ({n/len(df)*100:.2f}%)")
    return df


# ═══════════════════════════════════════════════════════════════
# TASK 3: XGBoost Event Classifier
# ═══════════════════════════════════════════════════════════════

def train_classifier(df: pd.DataFrame):
    print("  → Training XGBoost classifier …")

    feats = [f for f in CLASSIFIER_FEATS if f in df.columns]

    # Balance: all event rows + 8% normal sample
    ev   = df[df["event_code"] > 0]
    norm = df[df["event_code"] == 0].sample(frac=0.08, random_state=42)
    td   = pd.concat([ev, norm], ignore_index=True)

    X = td[feats].fillna(0)
    y = td["event_code"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Inverse-frequency class weights
    from collections import Counter
    cnt = Counter(y_tr)
    sw  = y_tr.map({c: max(cnt.values()) / v for c, v in cnt.items()})

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        eval_metric="mlogloss", n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr, sample_weight=sw,
              eval_set=[(X_te, y_te)], verbose=False)

    preds = model.predict(X_te)
    names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    print("\n" + classification_report(y_te, preds,
          target_names=names, zero_division=0))

    joblib.dump((model, feats), MODELS / "xgb_classifier.pkl")
    return model, feats


def apply_classifier(df: pd.DataFrame, model, feats):
    X     = df[feats].fillna(0)
    proba = model.predict_proba(X)

    df["pred_code"]  = model.predict(X)
    df["pred_label"] = df["pred_code"].map(LABEL_MAP)

    for i, lbl in LABEL_MAP.items():
        df[f"prob_{lbl}"] = proba[:, i]

    # Non-anomalous rows → Normal
    df.loc[~df["confirmed_event"], "pred_label"] = "Normal"
    df.loc[~df["confirmed_event"], "pred_code"]  = 0
    return df


# ═══════════════════════════════════════════════════════════════
# TASK 4: 24-hour PM2.5 Forecast
# ═══════════════════════════════════════════════════════════════

def train_forecast(df: pd.DataFrame):
    print("  → Training 24h PM2.5 forecast model …")
    feats = [f for f in FORECAST_FEATS if f in df.columns]

    df = df.copy()
    for city, grp in df.groupby("city"):
        df.loc[grp.index, "target_24h"] = grp["PM2_5"].shift(-24).values

    td = df.dropna(subset=["target_24h"] + feats)
    X  = td[feats].fillna(0)
    y  = td["target_24h"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    from sklearn.metrics import mean_absolute_error, r2_score
    yp  = model.predict(X_te)
    mae = mean_absolute_error(y_te, yp)
    r2  = r2_score(y_te, yp)
    print(f"     Forecast  MAE={mae:.2f} µg/m³   R²={r2:.3f}")

    joblib.dump((model, feats), MODELS / "xgb_forecast.pkl")
    return model, feats


def apply_forecast(df: pd.DataFrame, model, feats):
    X = df[feats].fillna(0)
    df["forecast_pm25_24h"] = np.clip(model.predict(X), 0, 1500)
    return df


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run(df=None):
    print("\n══ STEP 2: ANOMALY DETECTION + MODELLING ══\n")

    if df is None:
        fp = PROCESSED / "features.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Run 01_features.py first! Expected: {fp}")
        df = pd.read_parquet(fp)
        df["datetime"] = pd.to_datetime(df["datetime"])

    df = run_isolation_forest(df)
    df = run_stl(df)
    df = combine_evidence(df)

    clf, clf_feats  = train_classifier(df)
    df              = apply_classifier(df, clf, clf_feats)

    fc, fc_feats    = train_forecast(df)
    df              = apply_forecast(df, fc, fc_feats)

    out = PROCESSED / "results.parquet"
    df.to_parquet(out, index=False)
    print(f"\nResults saved → {out}")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"\n    Predicted events:\n"
          f"{df[df['pred_label']!='Normal']['pred_label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    run()
