"""
=============================================================
MASTER RUNNER — AQI Disaster Intelligence
Team Thunderstorm

Assumes data.py has already been run and produced:
    data/processed/downloaded_raw.parquet

Usage:
    python run.py              # features → model → dashboard
    python run.py --no-dash    # features + model only
=============================================================
"""
import sys, time, subprocess
from pathlib import Path

def step(name, module_path):
    import importlib.util
    print(f"\n{'─'*54}")
    print(f"  ▶  {name}")
    print(f"{'─'*54}")
    t0  = time.time()
    spec = importlib.util.spec_from_file_location("mod", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df   = mod.run()
    print(f"  ✔  Done in {time.time()-t0:.1f}s")
    return df

if __name__ == "__main__":
    if not Path("data/processed/downloaded_raw.parquet").exists():
        print("❌  Run data.py first to download GEE data!")
        sys.exit(1)

    print("""
╔══════════════════════════════════════════════════════════╗
║   AQI DISASTER INTELLIGENCE  |  Team Thunderstorm  ⚡    ║
║   Delhi  ·  Jaipur  ·  Vizag  |  2019 – 2024            ║
╚══════════════════════════════════════════════════════════╝
""")
    step("Feature Engineering",            "01_features.py")
    step("Anomaly Detection + Modelling",  "02_model.py")

    # NEW (safe — imports directly, no subprocess):
    if "--no-dash" not in sys.argv:
        print("\n  Launching dashboard → http://127.0.0.1:8050")
        import importlib.util
        spec = importlib.util.spec_from_file_location("dashboard", "03_dashboard.py")
        dash_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash_mod)
        dash_mod.app.run(debug=False, host="0.0.0.0", port=8050)
