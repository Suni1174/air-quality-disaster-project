"""
=============================================================
GEE Python Downloader — AQI Disaster Intelligence
Team Thunderstorm

Downloads for Delhi, Jaipur, Vizag (2019–2024):
  • Sentinel-5P  → NO2, SO2, CO, O3
  • MODIS MAIAC  → AOD (PM2.5 proxy)
  • MODIS Fire   → Fire hotspot counts
  • Open-Meteo   → Temperature, humidity, wind (free, no GEE needed)

One-time setup:
    pip install earthengine-api geemap pandas requests
    earthengine authenticate
=============================================================
"""

import ee
import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# ── Output directory ──────────────────────────────────────────
OUT = Path("data/raw/gee")
OUT.mkdir(parents=True, exist_ok=True)
(Path("data/raw/met")).mkdir(parents=True, exist_ok=True)

# ── Study area ────────────────────────────────────────────────
CITIES = {
    "Delhi":  {"lat": 28.7041, "lon": 77.1025, "buffer_km": 50},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "buffer_km": 50},
    "Vizag":  {"lat": 17.6868, "lon": 83.2185, "buffer_km": 50},
}

START = "2019-01-01"
END   = "2024-12-31"

# ── GEE Collections (all verified working) ───────────────────
COLLECTIONS = {
    "NO2": {
        "id":   "COPERNICUS/S5P/NRTI/L3_NO2",
        "band": "NO2_column_number_density",
        "scale_to_ugm3": 1e6 * 46.0055,   # mol/m² → µg/m³ approx via col height
        "unit": "mol/m2",
    },
    "SO2": {
        "id":   "COPERNICUS/S5P/NRTI/L3_SO2",
        "band": "SO2_column_number_density",
        "scale_to_ugm3": 1e6 * 64.066,
        "unit": "mol/m2",
    },
    "CO": {
        "id":   "COPERNICUS/S5P/NRTI/L3_CO",
        "band": "CO_column_number_density",
        "scale_to_ugm3": 1e3 * 28.01,
        "unit": "mol/m2",
    },
    "O3": {
        "id":   "COPERNICUS/S5P/NRTI/L3_O3",
        "band": "O3_column_number_density",
        "scale_to_ugm3": 1e6 * 48.0,
        "unit": "mol/m2",
    },
    "AOD": {
        "id":   "MODIS/061/MCD19A2_GRANULES",   # MAIAC AOD, 1km
        "band": "Optical_Depth_047",
        "scale_to_ugm3": None,   # AOD is unitless, kept as-is
        "unit": "unitless",
    },
}


# ═══════════════════════════════════════════════════════════════
# Initialise GEE
# ═══════════════════════════════════════════════════════════════

def init_gee(project: str = None):
    """Initialize GEE. Tries to auto-detect project, or pass it explicitly."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            # Try reading project from gcloud config automatically
            import subprocess
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True
            )
            detected = result.stdout.strip()
            if detected and detected != "(unset)":
                print(f"  ℹ️  Auto-detected project: {detected}")
                ee.Initialize(project=detected)
            else:
                # Last resort — prompts you to paste it
                project = input("  Enter your Google Cloud Project ID: ").strip()
                ee.Initialize(project=project)
        print("✅ GEE initialised")
    except Exception as e:
        print(f"⚠️  GEE init failed: {e}")
        print("   Fix: replace init_gee() call with init_gee(project='your-project-id')")
        raise


# ═══════════════════════════════════════════════════════════════
# Core extractor — pulls daily mean over city buffer
# ═══════════════════════════════════════════════════════════════

def extract_daily_series(collection_id: str,
                          band: str,
                          roi: ee.Geometry,
                          start: str,
                          end: str,
                          scale: int = 10000) -> pd.DataFrame:
    col = (ee.ImageCollection(collection_id)
             .filterDate(start, end)
             .filterBounds(roi)
             .select(band))

    # ── For granule-based collections (e.g. MAIAC AOD), mosaic by day first ──
    def mosaic_by_day(date_str):
        date   = ee.Date(date_str)
        daily  = col.filterDate(date, date.advance(1, 'day'))
        mosaic = daily.mean()   # spatial+temporal mean for that day
        mean_val = mosaic.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        )
        return ee.Feature(None, mean_val).set('date', date.format('YYYY-MM-dd'))

    # Generate list of dates in range
    n_days   = ee.Date(end).difference(ee.Date(start), 'day').int()
    date_list = ee.List.sequence(0, n_days.subtract(1)).map(
        lambda offset: ee.Date(start).advance(offset, 'day').format('YYYY-MM-dd')
    )

    fc    = ee.FeatureCollection(date_list.map(mosaic_by_day))
    feats = fc.getInfo()['features']

    rows = []
    for f in feats:
        props = f['properties']
        val   = props.get(band)
        if val is not None:
            rows.append({'date': props.get('date'), band: val})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    return df
# ═══════════════════════════════════════════════════════════════
# Fire hotspot counter (FIRMS)
# ═══════════════════════════════════════════════════════════════

def extract_fire_series(roi: ee.Geometry,
                         start: str, end: str) -> pd.DataFrame:
    """
    Count MODIS fire pixels per day within the ROI.
    FIRMS collection gives one point per active fire detection.
    """
    firms = (ee.FeatureCollection("ESA/GLOBFIRE/V2/FinalDB")
               .filterDate(start, end)
               .filterBounds(roi))

    # Alternatively use MODIS Terra fire radiative power raster
    fire_col = (ee.ImageCollection("MODIS/061/MOD14A1")
                  .filterDate(start, end)
                  .filterBounds(roi)
                  .select("FireMask"))

    def count_fire(img):
        # Fire pixels: FireMask >= 7 = high confidence fire
        fire_pixels = img.gte(7)
        count = fire_pixels.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=1000,
            maxPixels=1e10,
            bestEffort=True,
        )
        return (ee.Feature(None, count)
                  .set("date", img.date().format("YYYY-MM-dd")))

    fc    = ee.FeatureCollection(fire_col.map(count_fire))
    feats = fc.getInfo()["features"]

    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({"date": p.get("date"), "fire_count": p.get("FireMask", 0)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["fire_count"] = pd.to_numeric(df["fire_count"], errors="coerce").fillna(0)
    return df.sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Chunked downloader — avoids GEE 5000-element limit
# ═══════════════════════════════════════════════════════════════

def chunked_extract(collection_id: str, band: str,
                     roi: ee.Geometry, scale: int,
                     start: str, end: str,
                     chunk_months: int = 3) -> pd.DataFrame:
    """
    Break the full date range into chunks to stay under GEE limits.
    Sentinel-5P has ~1 image/day → 6 years = ~2190 images (fine).
    MODIS MAIAC is granule-based (many per day) → needs chunking.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")

    all_dfs = []
    cursor  = s
    chunk   = 0

    while cursor < e:
        next_cursor = min(cursor + timedelta(days=30 * chunk_months), e)
        cs = cursor.strftime("%Y-%m-%d")
        ce = next_cursor.strftime("%Y-%m-%d")
        chunk += 1

        print(f"      chunk {chunk}: {cs} → {ce}", end="\r")
        try:
            df = extract_daily_series(collection_id, band, roi, cs, ce, scale)
            all_dfs.append(df)
            time.sleep(0.5)   # be polite to GEE
        except Exception as ex:
            print(f"\n      ⚠️  chunk {chunk} failed: {ex} — skipping")

        cursor = next_cursor

    print()
    if not all_dfs:
        return pd.DataFrame()
    merged = (pd.concat(all_dfs, ignore_index=True)
                .drop_duplicates("date")
                .sort_values("date")
                .reset_index(drop=True))
    return merged


# ═══════════════════════════════════════════════════════════════
# Download all pollutants for one city
# ═══════════════════════════════════════════════════════════════

def download_city(city: str, lat: float, lon: float,
                   buffer_km: int = 50) -> pd.DataFrame:
    """Download all GEE layers for one city and merge into single DataFrame."""

    cache_file = OUT / f"gee_{city.lower()}.csv"
    if cache_file.exists():
        print(f"  ♻  Cache hit — loading {cache_file.name}")
        return pd.read_csv(cache_file, parse_dates=["date"])

    roi = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)
    print(f"\n  📡 Downloading GEE data for {city} …")

    master = None  # will hold merged df

    # ── Sentinel-5P pollutants ─────────────────────────────
    for pol, cfg in COLLECTIONS.items():
        print(f"    → {pol} ({cfg['id'].split('/')[-1]}) …")

        # MAIAC AOD needs smaller chunks (granule-based)
        chunk_m = 2 if pol == "AOD" else 6
        scale   = 1000 if pol == "AOD" else 7000

        df = chunked_extract(
            collection_id=cfg["id"],
            band=cfg["band"],
            roi=roi,
            scale=scale,
            start=START,
            end=END,
            chunk_months=chunk_m,
        )

        if df.empty:
            print(f"      ⚠️  No data returned for {pol}")
            continue

        df.rename(columns={cfg["band"]: pol}, inplace=True)

        # Convert raw units → µg/m³ where applicable
        if cfg["scale_to_ugm3"] is not None:
            df[pol] = df[pol] * cfg["scale_to_ugm3"]

        print(f"      ✅ {len(df):,} daily rows")

        if master is None:
            master = df
        else:
            master = pd.merge(master, df, on="date", how="outer")

    # ── Fire hotspots ──────────────────────────────────────
    print(f"    → Fire hotspots (MODIS MOD14A1) …")
    try:
        fire_df = extract_fire_series(roi, START, END)
        if not fire_df.empty:
            master = pd.merge(master, fire_df, on="date", how="left")
            master["fire_count"] = master["fire_count"].fillna(0)
            print(f"      ✅ {len(fire_df):,} days with fire data")
    except Exception as ex:
        print(f"      ⚠️  Fire extraction failed: {ex}")
        master["fire_count"] = 0

    master["city"] = city
    master.to_csv(cache_file, index=False)
    print(f"  💾 Saved → {cache_file}")
    return master


# ═══════════════════════════════════════════════════════════════
# Open-Meteo: hourly met data (no GEE, no API key)
# ═══════════════════════════════════════════════════════════════

def download_met_openmeteo(city: str, lat: float, lon: float) -> pd.DataFrame:
    """Hourly weather from Open-Meteo archive (completely free)."""
    cache = Path("data/raw/met") / f"met_{city.lower()}.csv"
    if cache.exists():
        print(f"  ♻  Met cache hit: {city}")
        return pd.read_csv(cache, parse_dates=["datetime"])

    print(f"  ↓ Open-Meteo met data: {city} …")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START}&end_date={END}"
        "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,"
        "winddirection_10m,precipitation,surface_pressure,dewpoint_2m,"
        "cloudcover,visibility"
        "&timezone=Asia%2FKolkata"
        "&wind_speed_unit=ms"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    df.rename(columns={
        "temperature_2m":      "temp_c",
        "relativehumidity_2m": "humidity",
        "windspeed_10m":       "wind_speed",
        "winddirection_10m":   "wind_dir",
        "precipitation":       "precip",
        "surface_pressure":    "pressure",
        "dewpoint_2m":         "dewpoint",
        "cloudcover":          "cloud_pct",
        "visibility":          "visibility_m",
    }, inplace=True)
    df["city"] = city
    df.to_csv(cache, index=False)
    print(f"  ✅ {len(df):,} hourly rows → {cache}")
    return df


# ═══════════════════════════════════════════════════════════════
# Merge GEE daily → hourly (forward-fill to match met frequency)
# ═══════════════════════════════════════════════════════════════

def merge_gee_with_met(gee_df: pd.DataFrame,
                        met_df: pd.DataFrame) -> pd.DataFrame:
    """
    GEE gives daily values. Met gives hourly.
    Strategy: repeat each daily GEE row for all 24 hours of that day,
    then merge on datetime + city.
    """
    gee_df = gee_df.copy()
    gee_df["date"] = pd.to_datetime(gee_df["date"])

    # Expand daily → hourly by joining on date
    met_df = met_df.copy()
    met_df["datetime"] = pd.to_datetime(met_df["datetime"])
    met_df["date"]     = met_df["datetime"].dt.normalize()

    merged = pd.merge(met_df, gee_df, on=["date", "city"], how="left")
    merged.drop(columns=["date"], inplace=True)
    return merged


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run():
    print("""
╔══════════════════════════════════════════════════════╗
║  GEE + Open-Meteo Data Downloader  |  Team ⚡        ║
║  Cities: Delhi | Jaipur | Vizag    |  2019–2024      ║
╚══════════════════════════════════════════════════════╝
""")
    init_gee(project="color-composites-ors")

    all_frames = []

    for city, coords in CITIES.items():
        print(f"\n{'═'*54}")
        print(f"  CITY: {city}")
        print(f"{'═'*54}")

        # 1. GEE pollutant data (daily)
        gee_df = download_city(city, coords["lat"], coords["lon"],
                                coords["buffer_km"])

        # 2. Met data (hourly)
        met_df = download_met_openmeteo(city, coords["lat"], coords["lon"])

        # 3. Merge → hourly dataframe with both pollutants + met
        merged = merge_gee_with_met(gee_df, met_df)
        all_frames.append(merged)
        print(f"  ✅ {city}: {len(merged):,} hourly rows, "
              f"{merged.shape[1]} columns")

    # Stack all cities
    final = pd.concat(all_frames, ignore_index=True)

    # Save
    out_path = Path("data/processed/downloaded_raw.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path, index=False)
    print(f"\n{'═'*54}")
    print(f"✅  ALL DONE")
    print(f"   Total rows : {len(final):,}")
    print(f"   Columns    : {list(final.columns)}")
    print(f"   Saved      → {out_path}")
    print(f"{'═'*54}\n")
    return final


if __name__ == "__main__":
    run()