"""
=============================================================
LIVE PREDICTOR — AQI Disaster Intelligence
Team Thunderstorm

Usage:  python predict.py
        → Enter pollutant values interactively
        → Get disaster type prediction + confidence

Also callable from code:
    from predict import predict_single
    result = predict_single(NO2=45, SO2=800, CO=0.9, O3=38,
                            PM2_5=120, PM10=180, AOD=1.0,
                            fire_count=0, temp_c=32,
                            humidity=65, wind_speed=3.2)
=============================================================
"""

import numpy as np
import joblib
from pathlib import Path

MODELS = Path("models")

LABEL_MAP  = {0:"Normal", 1:"Fire", 2:"Dust", 3:"Gas", 4:"Industrial"}
LABEL_CLR  = {
    "Normal":     "",
    "Fire":       "",
    "Dust":       "",
    "Gas":        "",
    "Industrial": "",
}
# Typical safe reference ranges (annual mean background for India)
SAFE_REF = {
    "NO2":   40,    # µg/m³  WHO annual guideline
    "SO2":   20,    # µg/m³  India NAAQS
    "CO":    2.0,   # mg/m³  (ppm-equivalent simplified)
    "O3":    100,   # µg/m³  India 8h standard
    "PM2_5": 60,    # µg/m³  India annual standard
    "PM10":  100,   # µg/m³  India annual standard
    "AOD":   0.3,   # unitless
}

CLASSIFIER_FEATS = [
    "NO2", "SO2", "CO", "O3", "PM2_5", "PM10", "AOD",
    "ratio_pm25_pm10", "ratio_co_no2", "ratio_so2_no2",
    "co_x_no2", "so2_x_no2", "aod_x_wind", "fire_x_no2",
    "PM2_5_roll6_std", "NO2_roll6_std", "SO2_roll6_std",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "fire_count",
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


def load_model():
    p = MODELS / "xgb_classifier.pkl"
    if not p.exists():
        raise FileNotFoundError(
            "Run 02_model.py first to train the classifier!"
        )
    model, feats = joblib.load(p)
    return model, feats


def build_feature_vector(NO2, SO2, CO, O3, PM2_5, PM10, AOD,
                          fire_count=0, temp_c=30, humidity=60,
                          wind_speed=3.0, wind_dir=180,
                          precip=0, pressure=1013,
                          hour=12, month=6, is_weekend=0,
                          is_spike=True):   # ← NEW parameter
    """
    is_spike=True:  lags use background levels (sudden onset scenario)
    is_spike=False: lags use current value (steady state)
    """
    # Background/baseline levels (what it was before the event)
    bg_NO2   = min(NO2   * 0.15, 50)
    bg_SO2   = min(SO2   * 0.05, 20)
    bg_CO    = min(CO    * 0.12, 1.2)
    bg_PM25  = min(PM2_5 * 0.18, 70)
    bg_PM10  = min(PM10  * 0.12, 100)
    bg_AOD   = min(AOD   * 0.20, 0.4)

    lag1_NO2  = bg_NO2  if is_spike else NO2
    lag6_NO2  = bg_NO2  if is_spike else NO2
    lag24_NO2 = bg_NO2  if is_spike else NO2
    lag1_SO2  = bg_SO2  if is_spike else SO2
    lag6_SO2  = bg_SO2  if is_spike else SO2
    lag1_CO   = bg_CO   if is_spike else CO
    lag6_CO   = bg_CO   if is_spike else CO
    lag1_PM25 = bg_PM25 if is_spike else PM2_5
    lag6_PM25 = bg_PM25 if is_spike else PM2_5
    lag24_PM25= bg_PM25 if is_spike else PM2_5
    lag1_AOD  = bg_AOD  if is_spike else AOD
    lag24_AOD = bg_AOD  if is_spike else AOD

    # Rolling stats — spike gives high std, steady state gives 0
    PM2_5_roll6_std = (PM2_5 - bg_PM25) * 0.4 if is_spike else 0.0
    NO2_roll6_std   = (NO2   - bg_NO2)  * 0.4 if is_spike else 0.0
    SO2_roll6_std   = (SO2   - bg_SO2)  * 0.4 if is_spike else 0.0

    # Rolling means — midpoint between background and current
    roll_PM25 = (PM2_5 + bg_PM25) / 2 if is_spike else PM2_5
    roll_AOD  = (AOD   + bg_AOD)  / 2 if is_spike else AOD

    # Ratios
    ratio_pm25_pm10 = PM2_5 / (PM10 + 1e-6)
    ratio_co_no2    = CO    / (NO2  + 1e-6)
    ratio_so2_no2   = SO2   / (NO2  + 1e-6)

    co_x_no2   = CO  * NO2
    so2_x_no2  = SO2 * NO2
    aod_x_wind = AOD * wind_speed
    fire_x_no2 = fire_count * NO2

    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    angle  = np.deg2rad(wind_dir)
    wind_u = wind_speed * np.sin(angle)
    wind_v = wind_speed * np.cos(angle)

    if   PM2_5 <= 30:  AQI = PM2_5 / 30 * 50
    elif PM2_5 <= 60:  AQI = 50  + (PM2_5-30)  / 30  * 50
    elif PM2_5 <= 90:  AQI = 100 + (PM2_5-60)  / 30  * 100
    elif PM2_5 <= 120: AQI = 200 + (PM2_5-90)  / 30  * 100
    else:              AQI = min(300 + (PM2_5-120) / 130 * 100, 500)

    row = {
        "NO2": NO2, "SO2": SO2, "CO": CO, "O3": O3,
        "PM2_5": PM2_5, "PM10": PM10, "AOD": AOD,
        "ratio_pm25_pm10": ratio_pm25_pm10,
        "ratio_co_no2":    ratio_co_no2,
        "ratio_so2_no2":   ratio_so2_no2,
        "co_x_no2": co_x_no2, "so2_x_no2": so2_x_no2,
        "aod_x_wind": aod_x_wind, "fire_x_no2": fire_x_no2,
        "PM2_5_roll6_std": PM2_5_roll6_std,
        "NO2_roll6_std":   NO2_roll6_std,
        "SO2_roll6_std":   SO2_roll6_std,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "fire_count": fire_count,
        "PM2_5_lag1": lag1_PM25, "PM2_5_lag6": lag6_PM25, "PM2_5_lag24": lag24_PM25,
        "NO2_lag1":   lag1_NO2,  "NO2_lag6":   lag6_NO2,  "NO2_lag24":   lag24_NO2,
        "SO2_lag1":   lag1_SO2,  "SO2_lag6":   lag6_SO2,
        "CO_lag1":    lag1_CO,   "CO_lag6":    lag6_CO,
        "AOD_lag1":   lag1_AOD,  "AOD_lag24":  lag24_AOD,
        "AQI_roll6_mean":  roll_PM25 * 1.5,
        "AQI_roll24_mean": roll_PM25 * 1.2,
        "temp_c": temp_c, "humidity": humidity,
        "wind_speed": wind_speed, "wind_u": wind_u, "wind_v": wind_v,
        "precip": precip, "pressure": pressure,
        "is_weekend": is_weekend,
    }
    return row, AQI

def predict_single(NO2, SO2, CO, O3, PM2_5, PM10, AOD,
                   fire_count=0, temp_c=30, humidity=60,
                   wind_speed=3.0, wind_dir=180,
                   precip=0, pressure=1013,
                   hour=12, month=6, is_weekend=0, is_spike=True):

    model, feats = load_model()
    row, AQI = build_feature_vector(
        NO2, SO2, CO, O3, PM2_5, PM10, AOD,
        fire_count, temp_c, humidity, wind_speed,
        wind_dir, precip, pressure, hour, month, is_weekend,
        is_spike=is_spike
    )

    import pandas as pd
    X = pd.DataFrame([row])[feats].fillna(0)
    pred_code = int(model.predict(X)[0])
    proba     = model.predict_proba(X)[0]
    all_probs = {LABEL_MAP[i]: round(proba[i]*100, 1) for i in range(len(proba))}

    # ── Rule-based override using chemical fingerprints ──────────
    # These rules match exactly what we know from the training data
    ratio = PM2_5 / (PM10 + 1e-6)

    if SO2 > 200:
        # SO2 dominates → Gas or Industrial
        rule_code = 3 if SO2 > 500 else 4
    elif PM10 > 400 and ratio < 0.25 and CO < 2.0:
        # Huge PM10, low ratio, quiet CO/NO2 → Dust
        rule_code = 2
    elif PM2_5 > 200 and CO > 3.0:
        # High PM2.5 + high CO → Fire
        rule_code = 1
    elif PM2_5 > 300 or (PM2_5 > 150 and fire_count > 5):
        # Very high PM2.5 with fire pixels → Fire
        rule_code = 1
    elif PM10 > 300 and ratio < 0.35:
        # High PM10, low ratio → Dust
        rule_code = 2
    else:
        rule_code = pred_code   # trust the model for ambiguous cases

    # Blend: if model and rules agree → use that; if not → use rules
    final_code = rule_code
    final_lbl  = LABEL_MAP[final_code]

    # Recalculate confidence using rule strength
    if rule_code != pred_code:
        # Rules overrode model — set confidence based on how extreme the values are
        if   rule_code == 3 and SO2 > 500:   confidence = min(95, 60 + SO2/20)
        elif rule_code == 2 and PM10 > 400:  confidence = min(95, 55 + PM10/30)
        elif rule_code == 1 and CO > 3:      confidence = min(95, 55 + CO*5)
        else:                                confidence = 70.0
    else:
        confidence = proba[pred_code] * 100

    # Danger level
    if final_lbl == "Normal":
        danger = "LOW"
    elif final_lbl in ("Gas",) or (final_lbl == "Industrial" and SO2 > 300):
        danger = "HIGH"
    elif final_lbl in ("Fire", "Industrial") or confidence > 80:
        danger = "MEDIUM" if final_lbl == "Fire" and PM2_5 < 300 else "HIGH"
    else:
        danger = "MEDIUM"

    # Pollutants above safe limit
    vals = {"NO2":NO2,"SO2":SO2,"CO":CO,"O3":O3,"PM2_5":PM2_5,"PM10":PM10,"AOD":AOD}
    above_safe = [
        f"{pol}={v:.1f} ({v/SAFE_REF[pol]:.1f}× safe limit)"
        for pol, v in vals.items() if v > SAFE_REF[pol] * 2
    ]

    explanations = {
        "Fire":       f"PM2.5={PM2_5:.0f} + CO={CO:.2f} together signal active combustion. PM2.5/PM10 ratio={ratio:.2f} (high = fine particles from fire).",
        "Dust":       f"PM10={PM10:.0f} dominates. PM2.5/PM10 ratio={ratio:.2f} (low = coarse mineral dust, not combustion). CO/NO2 are quiet.",
        "Gas":        f"SO2={SO2:.0f} µg/m³ is {SO2/SAFE_REF['SO2']:.0f}× the safe limit — sudden chemical/industrial release signature.",
        "Industrial": f"SO2={SO2:.0f} + NO2={NO2:.0f} elevated together — characteristic of industrial stack emissions sustained over time.",
        "Normal":     f"All pollutants within expected urban ranges. AQI≈{AQI:.0f}.",
    }

    return {
        "label":       final_lbl,
        "icon":        LABEL_CLR[final_lbl],
        "confidence":  round(confidence, 1),
        "danger":      danger,
        "AQI":         round(AQI, 1),
        "all_probs":   all_probs,
        "above_safe":  above_safe,
        "explanation": explanations[final_lbl],
    }

def print_result(res):
    print("\n" + "═"*54)
    print(f"  {res['icon']}  PREDICTION:  {res['label']}")
    print(f"  Confidence : {res['confidence']}%")
    print(f"  Danger     : {res['danger']}")
    print(f"  Est. AQI   : {res['AQI']}")
    print("─"*54)
    print("  All class probabilities:")
    for lbl, pct in sorted(res['all_probs'].items(),
                            key=lambda x: -x[1]):
        bar = "█" * int(pct // 5)
        print(f"    {lbl:<12} {pct:5.1f}%  {bar}")
    print("─"*54)
    print(f"  Why: {res['explanation']}")
    if res['above_safe']:
        print("\n  Pollutants above 2× safe limit:")
        for a in res['above_safe']:
            print(f"     • {a}")
    print("═"*54 + "\n")


def interactive_mode():
    print("""
╔══════════════════════════════════════════════════════╗
║  AQI DISASTER PREDICTOR  |  Team Thunderstorm  ⚡    ║
║  Enter pollutant readings → get disaster prediction  ║
╚══════════════════════════════════════════════════════╝

  Units:  NO2, SO2, O3, PM2.5, PM10 in µg/m³
          CO in mg/m³   |   AOD unitless (0-5)
  Press Enter to use [default] value shown.
""")

    def ask(prompt, default):
        try:
            val = input(f"  {prompt} [{default}]: ").strip()
            return float(val) if val else float(default)
        except ValueError:
            return float(default)

    NO2        = ask("NO2   µg/m³",   45)
    SO2        = ask("SO2   µg/m³",   20)
    CO         = ask("CO    mg/m³",   1.2)
    O3         = ask("O3    µg/m³",   40)
    PM2_5      = ask("PM2.5 µg/m³",   80)
    PM10       = ask("PM10  µg/m³",  140)
    AOD        = ask("AOD   (0-5)",  0.6)
    fire_count = ask("Fire pixel count (0 if none)", 0)
    temp_c     = ask("Temperature °C",  30)
    humidity   = ask("Humidity %",      60)
    wind_speed = ask("Wind speed m/s",   3)

    res = predict_single(
        NO2=NO2, SO2=SO2, CO=CO, O3=O3,
        PM2_5=PM2_5, PM10=PM10, AOD=AOD,
        fire_count=fire_count, temp_c=temp_c,
        humidity=humidity, wind_speed=wind_speed
    )
    print_result(res)

    # Test with known disaster scenarios
    again = input("  Test with a known disaster scenario? (y/n): ").strip().lower()
    if again == "y":
        print("""
  Choose scenario:
  1 → Delhi Stubble Fire (Nov 2021)
  2 → Jaipur Dust Storm (May 2021)
  3 → Vizag LG Gas Leak (May 2020)
  4 → Normal day (Delhi, August)
""")
        choice = input("  Enter 1-4: ").strip()
        scenarios = {
            "1": dict(NO2=180, SO2=35, CO=8.5, O3=42, PM2_5=420, PM10=680, AOD=3.2, fire_count=45, temp_c=24, humidity=72, wind_speed=1.8),
            "2": dict(NO2=28,  SO2=12, CO=0.7, O3=55, PM2_5=95,  PM10=820, AOD=2.8, fire_count=0,  temp_c=43, humidity=18, wind_speed=9.5),
            "3": dict(NO2=85,  SO2=920,CO=0.9, O3=38, PM2_5=48,  PM10=65,  AOD=0.4, fire_count=0,  temp_c=30, humidity=78, wind_speed=2.1),
            "4": dict(NO2=42,  SO2=15, CO=1.1, O3=35, PM2_5=55,  PM10=90,  AOD=0.45,fire_count=0,  temp_c=34, humidity=55, wind_speed=3.5),
        }
        if choice in scenarios:
            res2 = predict_single(**scenarios[choice])
            print_result(res2)


if __name__ == "__main__":
    interactive_mode()
