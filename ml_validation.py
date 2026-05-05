"""
=============================================================
ML VALIDATION — AQI Disaster Intelligence
Team Thunderstorm

Performs PROPER ML validation:
  1. K-Fold Cross Validation (5-fold)
  2. Confusion Matrix
  3. Per-class ROC AUC scores
  4. Feature Importance (top 20)
  5. Train vs Test performance comparison (overfitting check)
  6. Real-event detection check (13+1 events)

Run:  python ml_validation.py
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROCESSED = Path("data/processed")
MODELS    = Path("models")
OUT       = Path("data/validation")
OUT.mkdir(parents=True, exist_ok=True)

LABEL_MAP  = {0: "Normal", 1: "Fire", 2: "Dust", 3: "Gas", 4: "Industrial"}
LABEL_COLORS = {
    "Normal":     "#3ddbd9",
    "Fire":       "#ff6b35",
    "Dust":       "#f5c518",
    "Gas":        "#ff4757",
    "Industrial": "#b44fff",
}

# ── Known real events (14 events including Aug 2024) ──────────
REAL_EVENTS = [
    {"id":1,"city":"Delhi","start":"2019-10-25","end":"2019-11-15","type":"Fire",
     "name":"Delhi Stubble Burning 2019",
     "url":"https://timesofindia.indiatimes.com/city/delhi/delhi-air-quality-turns-severe-as-stubble-burning-peaks/articleshow/71815396.cms","source":"Times of India"},
    {"id":2,"city":"Delhi","start":"2020-11-01","end":"2020-11-20","type":"Fire",
     "name":"Delhi Stubble Burning 2020",
     "url":"https://www.indiatoday.in/india/story/delhi-air-pollution-stubble-burning-aqi-1740961-2020-11-03","source":"India Today"},
    {"id":3,"city":"Delhi","start":"2021-10-28","end":"2021-11-12","type":"Fire",
     "name":"Delhi Diwali + Stubble Crisis 2021",
     "url":"https://www.thehindu.com/news/cities/Delhi/delhi-air-quality-remains-very-poor/article37261858.ece","source":"The Hindu"},
    {"id":4,"city":"Delhi","start":"2022-10-30","end":"2022-11-14","type":"Fire",
     "name":"Delhi Stubble Burning 2022",
     "url":"https://www.hindustantimes.com/cities/delhi-news/delhi-aqi-pollution-stubble-burning-diwali-2022-101667474793143.html","source":"Hindustan Times"},
    {"id":5,"city":"Delhi","start":"2023-10-25","end":"2023-11-10","type":"Fire",
     "name":"Delhi Stubble Burning 2023",
     "url":"https://www.ndtv.com/india-news/delhi-air-quality-pollution-stubble-burning-aqi-4564890","source":"NDTV"},
    {"id":6,"city":"Jaipur","start":"2019-05-18","end":"2019-05-22","type":"Dust",
     "name":"Rajasthan Dust Storm May 2019",
     "url":"https://timesofindia.indiatimes.com/city/jaipur/dust-storm-hits-jaipur-pm10-level-shoots-up/articleshow/69390422.cms","source":"Times of India"},
    {"id":7,"city":"Jaipur","start":"2020-05-25","end":"2020-05-28","type":"Dust",
     "name":"Rajasthan Dust Storm May 2020",
     "url":"https://mausam.imd.gov.in","source":"IMD India"},
    {"id":8,"city":"Jaipur","start":"2021-06-02","end":"2021-06-05","type":"Dust",
     "name":"Rajasthan Dust Storm June 2021",
     "url":"https://mausam.imd.gov.in","source":"IMD India"},
    {"id":9,"city":"Jaipur","start":"2022-05-14","end":"2022-05-17","type":"Dust",
     "name":"Rajasthan Dust Storm May 2022",
     "url":"https://mausam.imd.gov.in","source":"IMD India"},
    {"id":10,"city":"Jaipur","start":"2023-05-20","end":"2023-05-24","type":"Dust",
     "name":"Rajasthan Dust Storm May 2023",
     "url":"https://www.hindustantimes.com/india-news/dust-storm-rajasthan-2023-101684731826043.html","source":"Hindustan Times"},
    {"id":11,"city":"Vizag","start":"2020-05-07","end":"2020-05-09","type":"Gas",
     "name":"LG Polymers Styrene Gas Leak — May 7, 2020",
     "url":"https://en.wikipedia.org/wiki/2020_Visakhapatnam_gas_leak","source":"Wikipedia"},
    {"id":12,"city":"Vizag","start":"2021-06-10","end":"2021-06-12","type":"Industrial",
     "name":"HPCL Vizag Refinery Emission 2021",
     "url":"https://cpcb.nic.in/automatic-monitoring-data/","source":"CPCB Data"},
    {"id":13,"city":"Vizag","start":"2023-03-15","end":"2023-03-17","type":"Industrial",
     "name":"Vizag VSEZ SO2 Spike March 2023",
     "url":"https://cpcb.nic.in/automatic-monitoring-data/","source":"CPCB Data"},
    # NEW — Escientia Pharma Blast, Atchutapuram SEZ, Aug 2024
    {"id":14,"city":"Vizag","start":"2024-08-21","end":"2024-08-23","type":"Industrial",
     "name":"Escientia Pharma Factory Blast — Aug 21, 2024",
     "desc":"MTBE solvent explosion at Atchutapuram SEZ, Anakapalli. 17 killed, 40 injured. Vapour cloud explosion + fire spread through reactor ductwork.",
     "url":"https://en.wikipedia.org/wiki/Atchutapuram_pharmaceutical_factory_explosion","source":"Wikipedia"},
]


# ═══════════════════════════════════════════════════════════════
# 1. Load data + trained model
# ═══════════════════════════════════════════════════════════════

def load():
    fp = PROCESSED / "features.parquet"
    rp = PROCESSED / "results.parquet"
    if not fp.exists():
        raise FileNotFoundError("Run 01_features.py first!")
    df = pd.read_parquet(fp)
    df["datetime"] = pd.to_datetime(df["datetime"])
    results = pd.read_parquet(rp) if rp.exists() else None
    model, feats = joblib.load(MODELS / "xgb_classifier.pkl")
    return df, results, model, feats


def get_train_data(df, feats):
    """Reconstruct the training dataset (same as in 02_model.py)."""
    ev   = df[df["event_code"] > 0]
    norm = df[df["event_code"] == 0].sample(frac=0.08, random_state=42)
    td   = pd.concat([ev, norm], ignore_index=True)
    X    = td[feats].fillna(0)
    y    = td["event_code"]
    return X, y


# ═══════════════════════════════════════════════════════════════
# 2. 5-Fold Cross Validation
# ═══════════════════════════════════════════════════════════════

def run_cross_validation(X, y):
    print("\n── 5-Fold Stratified Cross Validation ──")
    from collections import Counter
    cnt = Counter(y)
    sw  = y.map({c: max(cnt.values()) / v for c, v in cnt.items()})

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        eval_metric="mlogloss", n_jobs=-1, random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Manual k-fold loop (avoids fit_params version issues)
    acc_tr, acc_te, f1_tr, f1_te, f1w_tr, f1w_te = [], [], [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr_f, X_te_f = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr_f, y_te_f = y.iloc[tr_idx], y.iloc[te_idx]
        sw_f = sw.iloc[tr_idx]

        fold_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            eval_metric="mlogloss", n_jobs=-1, random_state=42,
        )
        fold_clf.fit(X_tr_f, y_tr_f, sample_weight=sw_f)

        y_tr_pred = fold_clf.predict(X_tr_f)
        y_te_pred = fold_clf.predict(X_te_f)

        acc_tr.append(accuracy_score(y_tr_f, y_tr_pred))
        acc_te.append(accuracy_score(y_te_f, y_te_pred))
        f1_tr.append(f1_score(y_tr_f, y_tr_pred, average="macro", zero_division=0))
        f1_te.append(f1_score(y_te_f, y_te_pred, average="macro", zero_division=0))
        f1w_tr.append(f1_score(y_tr_f, y_tr_pred, average="weighted", zero_division=0))
        f1w_te.append(f1_score(y_te_f, y_te_pred, average="weighted", zero_division=0))
        print(f"     Fold {fold}: acc={acc_te[-1]:.4f}  f1={f1_te[-1]:.4f}")

    scores = {
        "train_accuracy":    np.array(acc_tr),
        "test_accuracy":     np.array(acc_te),
        "train_f1_macro":    np.array(f1_tr),
        "test_f1_macro":     np.array(f1_te),
        "train_f1_weighted": np.array(f1w_tr),
        "test_f1_weighted":  np.array(f1w_te),
    }

    print(f"\n  {'Metric':<25} {'Train':>10} {'Test':>10} {'Std':>10}")
    print("  " + "─"*55)
    metrics = [
        ("Accuracy",     "train_accuracy",    "test_accuracy"),
        ("F1 Macro",     "train_f1_macro",     "test_f1_macro"),
        ("F1 Weighted",  "train_f1_weighted",  "test_f1_weighted"),
    ]
    cv_results = {}
    for label, tr_key, te_key in metrics:
        tr_mean = scores[tr_key].mean()
        te_mean = scores[te_key].mean()
        te_std  = scores[te_key].std()
        print(f"  {label:<25} {tr_mean:>10.4f} {te_mean:>10.4f} {te_std:>10.4f}")
        cv_results[label] = {"train": tr_mean, "test": te_mean, "std": te_std}

    overfit_gap = scores["train_accuracy"].mean() - scores["test_accuracy"].mean()
    print(f"\n  Overfitting gap (train−test accuracy): {overfit_gap:.4f}")
    if overfit_gap < 0.03:
        print("  ✅ Model is NOT overfitting (gap < 3%)")
    elif overfit_gap < 0.08:
        print("  ⚠️  Mild overfitting detected (gap 3–8%)")
    else:
        print("  ❌ Significant overfitting (gap > 8%)")

    return cv_results


# ═══════════════════════════════════════════════════════════════
# 3. Confusion Matrix
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(model, feats, X_test, y_test):
    print("\n── Confusion Matrix ──")
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — XGBoost Event Classifier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    # Per-class accuracy from confusion matrix
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        total = cm[i, :].sum()
        print(f"  {lbl:<12}: {tp}/{total} correct  ({tp/max(total,1)*100:.1f}%)")

    return cm


# ═══════════════════════════════════════════════════════════════
# 4. ROC AUC (One-vs-Rest)
# ═══════════════════════════════════════════════════════════════

def compute_roc_auc(model, X_test, y_test):
    print("\n── ROC AUC Scores (One-vs-Rest) ──")
    proba  = model.predict_proba(X_test)
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    y_bin  = label_binarize(y_test, classes=list(range(len(labels))))

    auc_scores = {}
    for i, lbl in enumerate(labels):
        if y_bin[:, i].sum() == 0:
            print(f"  {lbl:<12}: N/A (no positive samples in test set)")
            continue
        auc = roc_auc_score(y_bin[:, i], proba[:, i])
        auc_scores[lbl] = auc
        bar = "█" * int(auc * 20)
        print(f"  {lbl:<12}: AUC = {auc:.4f}  {bar}")

    macro_auc = np.mean(list(auc_scores.values()))
    print(f"\n  Macro AUC: {macro_auc:.4f}")
    return auc_scores


# ═══════════════════════════════════════════════════════════════
# 5. Feature Importance
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance(model, feats):
    print("\n── Top 20 Feature Importances ──")
    imp = pd.Series(model.feature_importances_, index=feats)
    top20 = imp.nlargest(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#ff6b35" if "PM2" in f or "CO" in f
               else "#f5c518" if "PM10" in f or "AOD" in f or "wind" in f
               else "#ff4757" if "SO2" in f
               else "#3ddbd9" for f in top20.index]
    top20.plot.barh(ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("Importance Score (gain)", fontsize=11)
    ax.set_title("Top 20 Features — XGBoost Classifier", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Legend
    patches = [
        mpatches.Patch(color="#ff6b35", label="Fire indicators (PM2.5, CO)"),
        mpatches.Patch(color="#f5c518", label="Dust indicators (PM10, AOD, wind)"),
        mpatches.Patch(color="#ff4757", label="Gas/Industrial (SO2)"),
        mpatches.Patch(color="#3ddbd9", label="Other features"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out = OUT / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Top 5: {list(top20.index[:5])}")
    print(f"  Saved → {out}")
    return top20


# ═══════════════════════════════════════════════════════════════
# 6. Train vs Test performance (overfitting check)
# ═══════════════════════════════════════════════════════════════

def train_test_comparison(model, feats, X_train, y_train, X_test, y_test):
    print("\n── Train vs Test Performance ──")
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)

    tr_acc = accuracy_score(y_train, y_tr_pred)
    te_acc = accuracy_score(y_test,  y_te_pred)
    tr_f1  = f1_score(y_train, y_tr_pred, average="macro", zero_division=0)
    te_f1  = f1_score(y_test,  y_te_pred, average="macro", zero_division=0)

    print(f"\n  {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("  " + "─"*50)
    print(f"  {'Accuracy':<20} {tr_acc:>10.4f} {te_acc:>10.4f} {tr_acc-te_acc:>10.4f}")
    print(f"  {'F1 Macro':<20} {tr_f1:>10.4f} {te_f1:>10.4f} {tr_f1-te_f1:>10.4f}")

    if tr_acc - te_acc < 0.03:
        print("\n  ✅ Good generalisation — model is NOT overfitting")
    else:
        print("\n  ⚠️  Some overfitting detected — train accuracy is higher than test")
    return {"train_acc": tr_acc, "test_acc": te_acc, "train_f1": tr_f1, "test_f1": te_f1}


# ═══════════════════════════════════════════════════════════════
# 7. Real Event Detection Check
# ═══════════════════════════════════════════════════════════════

def check_real_events(results_df):
    print("\n── Real Event Detection (14 Events) ──")
    print(f"\n  {'#':<3} {'Event':<44} {'Type':<12} {'Detected':<12} {'Match%':<8} {'Score'}")
    print("  " + "─"*90)

    rows = []
    for ev in REAL_EVENTS:
        mask = (
            (results_df["city"] == ev["city"]) &
            (results_df["datetime"] >= ev["start"]) &
            (results_df["datetime"] <= ev["end"])
        )
        sub = results_df[mask]
        if sub.empty:
            rows.append({**ev, "detected": False, "match_pct": 0, "peak_score": 0})
            print(f"  {ev['id']:<3} {ev['name'][:42]:<44} {ev['type']:<12} {'NO DATA':<12} {'—':<8} —")
            continue

        lc        = sub["event_label"].value_counts() if "event_label" in sub.columns else pd.Series()
        match_pct = lc.get(ev["type"], 0) / len(sub) * 100
        peak      = float(sub["anomaly_score"].max()) if "anomaly_score" in sub.columns else 0
        detected  = match_pct > 50

        status = "YES" if detected else "PARTIAL"
        print(f"  {ev['id']:<3} {ev['name'][:42]:<44} {ev['type']:<12} {status:<12} {match_pct:5.1f}%   {peak:.3f}")
        rows.append({**ev, "detected": detected, "match_pct": round(match_pct,1), "peak_score": round(peak,3)})

    n = sum(1 for r in rows if r["detected"])
    print(f"\n  Detected: {n}/{len(REAL_EVENTS)}  ({n/len(REAL_EVENTS)*100:.0f}%)")
    return rows


# ═══════════════════════════════════════════════════════════════
# 8. Summary plot
# ═══════════════════════════════════════════════════════════════

def plot_cv_summary(cv_results, auc_scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CV scores
    ax = axes[0]
    metrics = list(cv_results.keys())
    train_vals = [cv_results[m]["train"] for m in metrics]
    test_vals  = [cv_results[m]["test"]  for m in metrics]
    stds       = [cv_results[m]["std"]   for m in metrics]
    x = np.arange(len(metrics))
    ax.bar(x-0.18, train_vals, 0.35, label="Train", color="#2E75B6", alpha=0.85)
    ax.bar(x+0.18, test_vals,  0.35, label="Test",  color="#ff6b35", alpha=0.85,
           yerr=stds, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0.85, 1.02); ax.set_ylabel("Score")
    ax.set_title("5-Fold CV: Train vs Test", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    for i, (tr, te) in enumerate(zip(train_vals, test_vals)):
        ax.text(i-0.18, tr+0.002, f"{tr:.3f}", ha="center", fontsize=8)
        ax.text(i+0.18, te+0.002, f"{te:.3f}", ha="center", fontsize=8)

    # Right: ROC AUC per class
    ax2 = axes[1]
    lbls  = list(auc_scores.keys())
    aucs  = list(auc_scores.values())
    clrs  = [LABEL_COLORS.get(l, "#3ddbd9") for l in lbls]
    bars  = ax2.barh(lbls, aucs, color=clrs, alpha=0.85, edgecolor="none")
    ax2.set_xlim(0.85, 1.02)
    ax2.set_xlabel("ROC AUC Score")
    ax2.set_title("ROC AUC by Event Class (One-vs-Rest)", fontweight="bold")
    ax2.axvline(x=0.90, color="gray", linestyle="--", alpha=0.5, label="AUC=0.90")
    ax2.legend(fontsize=9); ax2.grid(axis="x", alpha=0.3)
    ax2.set_facecolor("#f8f9fa")
    for bar, auc in zip(bars, aucs):
        ax2.text(auc+0.001, bar.get_y()+bar.get_height()/2,
                 f"{auc:.4f}", va="center", fontsize=9)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out = OUT / "cv_auc_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Summary plot saved → {out}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run():
    print("""
╔══════════════════════════════════════════════════════╗
║  ML VALIDATION  |  Team Thunderstorm  ⚡              ║
║  XGBoost Event Classifier — Full Validation Suite    ║
╚══════════════════════════════════════════════════════╝
""")
    df, results_df, model, feats = load()
    X, y = get_train_data(df, feats)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 1. Cross validation
    cv_results = run_cross_validation(X, y)

    # 2. Train vs test
    tt_results = train_test_comparison(model, feats, X_tr, y_tr, X_te, y_te)

    # 3. Classification report on held-out test set
    print("\n── Classification Report (held-out 20% test set) ──")
    y_pred = model.predict(X_te)
    names  = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    print(classification_report(y_te, y_pred, target_names=names, zero_division=0))

    # 4. Confusion matrix
    cm = plot_confusion_matrix(model, feats, X_te, y_te)

    # 5. ROC AUC
    auc_scores = compute_roc_auc(model, X_te, y_te)

    # 6. Feature importance
    top20 = plot_feature_importance(model, feats)

    # 7. Real event detection
    if results_df is not None:
        results_df["datetime"] = pd.to_datetime(results_df["datetime"])
        event_rows = check_real_events(results_df)
    else:
        print("\n  ⚠️  results.parquet not found — skipping real event check")
        event_rows = []

    # 8. Summary plots
    plot_cv_summary(cv_results, auc_scores)

    # Save full validation report to CSV
    val_df = pd.DataFrame(event_rows)
    val_df.to_csv(OUT / "event_validation.csv", index=False)

    print(f"""
{'═'*54}
  VALIDATION SUMMARY
{'═'*54}
  5-Fold CV Accuracy  : {cv_results['Accuracy']['test']:.4f} ± {cv_results['Accuracy']['std']:.4f}
  5-Fold CV F1 Macro  : {cv_results['F1 Macro']['test']:.4f} ± {cv_results['F1 Macro']['std']:.4f}
  Train Accuracy      : {tt_results['train_acc']:.4f}
  Test  Accuracy      : {tt_results['test_acc']:.4f}
  Overfit Gap         : {tt_results['train_acc']-tt_results['test_acc']:.4f}
  Macro ROC AUC       : {np.mean(list(auc_scores.values())):.4f}
  Real Events Detected: {sum(1 for r in event_rows if r.get('detected'))}/{len(event_rows)}
{'═'*54}
  Output files → data/validation/
    confusion_matrix.png
    feature_importance.png
    cv_auc_summary.png
    event_validation.csv
{'═'*54}
""")

if __name__ == "__main__":
    run()
