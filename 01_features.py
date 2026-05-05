"""
=============================================================
STEP 1 — Feature Engineering
Team Thunderstorm | AQI Disaster Intelligence

Input : data/processed/downloaded_raw.parquet
         Columns: temp_c, humidity, wind_speed, wind_dir, precip,
                  pressure, dewpoint, cloud_pct, visibility_m,
                  datetime, city, NO2, SO2, CO, O3, AOD, fire_count

Output: data/processed/features.parquet
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Your actual columns from GEE ─────────────────────────────
GEE_POLLUTANTS = ["NO2", "SO2", "CO", "O3"]   # µg/m³ after conversion
MET_COLS       = ["temp_c", "humidity", "wind_speed", "wind_dir",
                   "precip", "pressure", "dewpoint", "cloud_pct"]


# ═══════════════════════════════════════════════════════════════
# 1. Load + unit-fix
# ═══════════════════════════════════════════════════════════════

def load_and_fix_units(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

    # ── AOD → PM2.5 / PM10 proxies ────────────────────────────
    # WHO formula: PM2.5 ≈ AOD × 120 (India urban, surface-level)
    if "AOD" in df.columns:
        df["AOD"]  = df["AOD"].clip(lower=0, upper=5)
        df["PM2_5"] = (df["AOD"] * 120).clip(0, 1500)
        df["PM10"]  = (df["AOD"] * 195).clip(0, 3000)
    else:
        df["PM2_5"] = np.nan
        df["PM10"]  = np.nan

    # ── S5P column densities already scaled to µg/m³ in downloader ──
    # Clip extreme outliers (0.1% tails)
    for pol in GEE_POLLUTANTS:
        if pol in df.columns:
            lo = 0
            hi = df[pol].quantile(0.999)
            df[pol] = df[pol].clip(lo, hi)

    print(f"  Loaded {len(df):,} rows | cities: {df['city'].unique()}")
    print(f"  Date range: {df['datetime'].min().date()} → "
          f"{df['datetime'].max().date()}")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Interpolate NaNs
# ═══════════════════════════════════════════════════════════════

def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    all_pols = ["PM2_5", "PM10"] + GEE_POLLUTANTS

    for city, grp in df.groupby("city"):
        grp = grp.copy().sort_values("datetime")

        for col in all_pols + MET_COLS:
            if col not in grp.columns:
                continue
            # Linear for gaps ≤ 6 h
            grp[col] = grp[col].interpolate(
                method="linear", limit=6, limit_direction="both"
            )
            # Seasonal mean for longer gaps
            grp["_h"] = grp["datetime"].dt.hour
            grp["_m"] = grp["datetime"].dt.month
            seasonal  = grp.groupby(["_m", "_h"])[col].transform("mean")
            grp[col]  = grp[col].fillna(seasonal).fillna(0)

        grp.drop(columns=["_h", "_m"], errors="ignore", inplace=True)
        frames.append(grp)

    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# 3. Smooth (Savitzky-Golay) — keeps spike peaks intact
# ═══════════════════════════════════════════════════════════════

def smooth(series: pd.Series, window=11, poly=2) -> pd.Series:
    v = series.ffill().bfill().fillna(0).values.astype(float)
    try:
        return pd.Series(
            savgol_filter(v, window_length=window, polyorder=poly, mode="interp"),
            index=series.index
        )
    except Exception:
        return series.rolling(5, center=True, min_periods=1).mean()


def apply_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for city, grp in df.groupby("city"):
        grp = grp.copy()
        for pol in ["PM2_5", "PM10"] + GEE_POLLUTANTS:
            if pol in grp.columns:
                grp[pol + "_smooth"] = smooth(grp[pol]).values
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# 4. AQI calculation (India CPCB sub-index)
# ═══════════════════════════════════════════════════════════════

def sub_idx(v, blo, bhi, ilo, ihi):
    return ((ihi - ilo) / (bhi - blo + 1e-9)) * (v - blo) + ilo

def calc_aqi(row):
    subs = []
    p = row.get("PM2_5", np.nan)
    if pd.notna(p) and p > 0:
        if   p <= 30:  subs.append(sub_idx(p,   0,  30,   0,  50))
        elif p <= 60:  subs.append(sub_idx(p,  31,  60,  51, 100))
        elif p <= 90:  subs.append(sub_idx(p,  61,  90, 101, 200))
        elif p <= 120: subs.append(sub_idx(p,  91, 120, 201, 300))
        else:          subs.append(min(sub_idx(p, 121, 250, 301, 400), 500))
    p = row.get("PM10", np.nan)
    if pd.notna(p) and p > 0:
        if   p <= 50:  subs.append(sub_idx(p,   0,  50,   0,  50))
        elif p <= 100: subs.append(sub_idx(p,  51, 100,  51, 100))
        elif p <= 250: subs.append(sub_idx(p, 101, 250, 101, 200))
        elif p <= 350: subs.append(sub_idx(p, 251, 350, 201, 300))
        else:          subs.append(min(sub_idx(p, 351, 430, 301, 400), 500))
    return max(subs) if subs else np.nan


# ═══════════════════════════════════════════════════════════════
# 5. Feature engineering
# ═══════════════════════════════════════════════════════════════

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── AQI ───────────────────────────────────────────────────
    df["AQI"] = df.apply(calc_aqi, axis=1)
    # Fallback AQI from AOD if PM2.5 unavailable
    if df["AQI"].isna().mean() > 0.5 and "AOD" in df.columns:
        df["AQI"] = df["AQI"].fillna(df["AOD"] * 200)

    # ── Pollutant ratios ───────────────────────────────────────
    df["ratio_pm25_pm10"] = df["PM2_5"] / (df["PM10"]  + 1e-6)
    df["ratio_co_no2"]    = df["CO"]    / (df["NO2"]   + 1e-6)
    df["ratio_so2_no2"]   = df["SO2"]   / (df["NO2"]   + 1e-6)

    # ── Interaction terms ──────────────────────────────────────
    df["co_x_no2"]    = df["CO"]   * df["NO2"]
    df["so2_x_no2"]   = df["SO2"]  * df["NO2"]
    df["aod_x_wind"]  = df["AOD"].fillna(0) * df["wind_speed"].fillna(0)
    df["fire_x_no2"]  = df["fire_count"].fillna(0) * df["NO2"]

    # ── Temporal encodings ─────────────────────────────────────
    df["hour"]       = df["datetime"].dt.hour
    df["dayofweek"]  = df["datetime"].dt.dayofweek
    df["month"]      = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lag features (per city) ────────────────────────────────
    lag_cols = ["PM2_5", "PM10", "NO2", "SO2", "CO", "AOD"]
    for city, grp in df.groupby("city"):
        idx = grp.index
        for col in lag_cols:
            if col not in grp.columns:
                continue
            for lag in [1, 6, 24]:
                df.loc[idx, f"{col}_lag{lag}"] = grp[col].shift(lag).values

    # ── Rolling stats ──────────────────────────────────────────
    roll_cols = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    for city, grp in df.groupby("city"):
        idx = grp.index
        for col in roll_cols:
            if col not in grp.columns:
                continue
            df.loc[idx, f"{col}_roll6_mean"] = (
                grp[col].rolling(6,  min_periods=1).mean().values)
            df.loc[idx, f"{col}_roll24_mean"] = (
                grp[col].rolling(24, min_periods=1).mean().values)
            df.loc[idx, f"{col}_roll6_std"] = (
                grp[col].rolling(6,  min_periods=1).std().fillna(0).values)

    # ── Wind vector components ─────────────────────────────────
    angle        = np.deg2rad(df["wind_dir"].fillna(0))
    df["wind_u"] = df["wind_speed"].fillna(0) * np.sin(angle)
    df["wind_v"] = df["wind_speed"].fillna(0) * np.cos(angle)

    return df


# ═══════════════════════════════════════════════════════════════
# 6. Known event labels (for XGBoost training)
# ═══════════════════════════════════════════════════════════════

EVENTS = [
    ("Delhi",  "2019-10-25", "2019-11-15", "Fire"),
    ("Delhi",  "2020-11-01", "2020-11-20", "Fire"),
    ("Delhi",  "2021-10-28", "2021-11-12", "Fire"),
    ("Delhi",  "2022-10-30", "2022-11-14", "Fire"),
    ("Delhi",  "2023-10-25", "2023-11-10", "Fire"),
    ("Jaipur", "2019-05-18", "2019-05-22", "Dust"),
    ("Jaipur", "2020-05-25", "2020-05-28", "Dust"),
    ("Jaipur", "2021-06-02", "2021-06-05", "Dust"),
    ("Jaipur", "2022-05-14", "2022-05-17", "Dust"),
    ("Jaipur", "2023-05-20", "2023-05-24", "Dust"),
    ("Vizag",  "2020-05-07", "2020-05-09", "Gas"),
    ("Vizag",  "2021-06-10", "2021-06-12", "Industrial"),
    ("Vizag",  "2023-03-15", "2023-03-17", "Industrial"),
    ("Vizag",  "2024-08-21", "2024-08-23", "Industrial"),
]
LABEL_MAP = {"Normal": 0, "Fire": 1, "Dust": 2, "Gas": 3, "Industrial": 4}

def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["event_label"] = "Normal"
    for city, s, e, etype in EVENTS:
        mask = ((df["city"] == city) &
                (df["datetime"] >= s) &
                (df["datetime"] <= e))
        df.loc[mask, "event_label"] = etype
    df["event_code"] = df["event_label"].map(LABEL_MAP).fillna(0).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run(df=None):
    print("\n══ STEP 1: FEATURE ENGINEERING ══\n")

    raw_path = PROCESSED / "downloaded_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Run data.py first! Expected: {raw_path}"
        )

    df = load_and_fix_units(raw_path)

    print("  → Interpolating gaps …")
    df = interpolate(df)

    print("  → Smoothing signals …")
    df = apply_smoothing(df)

    print("  → Engineering features …")
    df = engineer(df)

    print("  → Attaching event labels …")
    df = attach_labels(df)

    out = PROCESSED / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✅  Features saved → {out}")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"    Event breakdown:\n{df['event_label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    run()
