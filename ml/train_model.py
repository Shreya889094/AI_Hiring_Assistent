"""
AI Hiring Assistant - ML Training Pipeline
Trains Random Forest + XGBoost ensemble to shortlist candidates from CSV
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found - using GradientBoosting instead")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../backend/data/candidates_dataset.csv")

# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Education encoding
    edu_map = {"B.Sc CS": 1, "BCA": 2, "B.Tech": 3, "MCA": 4,
               "MBA": 5, "M.Tech": 6, "PhD": 7}
    df["education_level"] = df["education"].map(edu_map).fillna(3)

    # Experience buckets
    df["exp_bucket"] = pd.cut(df["experience_years"],
                               bins=[-1, 0, 2, 5, 8, 100],
                               labels=[0, 1, 2, 3, 4]).astype(int)

    # Composite scores
    df["overall_score"] = (
        0.30 * df["skill_score"] +
        0.20 * df["education_score"] +
        0.15 * df["certification_score"] +
        0.15 * df["project_score"] +
        0.10 * df["communication_score"] +
        0.10 * df["problem_solving_score"]
    )

    df["exp_skill_interaction"] = df["experience_years"] * df["num_skills"]
    df["cert_per_year"]         = df["certifications"] / (df["experience_years"] + 1)
    df["projects_per_year"]     = df["projects_count"] / (df["experience_years"] + 1)
    df["gpa_normalized"]        = (df["gpa"] - 5.0) / 5.0 * 100   # scale to 0-100

    return df


FEATURES = [
    "experience_years", "num_skills", "certifications", "projects_count",
    "gpa", "skill_score", "education_score", "certification_score",
    "project_score", "communication_score", "problem_solving_score",
    "education_level", "exp_bucket", "overall_score",
    "exp_skill_interaction", "cert_per_year", "projects_per_year",
    "gpa_normalized", "ats_score"
]
TARGET = "shortlisted"


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  AI Hiring Assistant – ML Training Pipeline")
    print("=" * 60)

    # Load & prepare
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dataset loaded: {len(df)} candidates")
    print(f"   Shortlisted  : {df[TARGET].sum()}")
    print(f"   Not selected : {(df[TARGET] == 0).sum()}")

    df = engineer_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}
    models  = {}

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n🌲 Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)
    rf_pred  = rf.predict(X_test_sc)
    rf_proba = rf.predict_proba(X_test_sc)[:, 1]
    cv_rf    = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring="f1").mean()

    results["Random Forest"] = {
        "accuracy" : round(accuracy_score(y_test, rf_pred) * 100, 2),
        "f1"       : round(f1_score(y_test, rf_pred) * 100, 2),
        "roc_auc"  : round(roc_auc_score(y_test, rf_proba) * 100, 2),
        "cv_f1"    : round(cv_rf * 100, 2),
    }
    models["rf"] = rf
    print(f"   Accuracy: {results['Random Forest']['accuracy']}%  "
          f"F1: {results['Random Forest']['f1']}%  "
          f"AUC: {results['Random Forest']['roc_auc']}%")

    # ── XGBoost / GradientBoosting ───────────────────────────────────────────
    if HAS_XGB:
        print("\n⚡ Training XGBoost …")
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42, eval_metric="logloss", verbosity=0
        )
    else:
        print("\n⚡ Training GradientBoosting …")
        xgb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

    xgb.fit(X_train_sc, y_train)
    xgb_pred  = xgb.predict(X_test_sc)
    xgb_proba = xgb.predict_proba(X_test_sc)[:, 1]
    cv_xgb    = cross_val_score(xgb, X_train_sc, y_train, cv=5, scoring="f1").mean()
    model_label = "XGBoost" if HAS_XGB else "GradientBoosting"

    results[model_label] = {
        "accuracy" : round(accuracy_score(y_test, xgb_pred) * 100, 2),
        "f1"       : round(f1_score(y_test, xgb_pred) * 100, 2),
        "roc_auc"  : round(roc_auc_score(y_test, xgb_proba) * 100, 2),
        "cv_f1"    : round(cv_xgb * 100, 2),
    }
    models["xgb"] = xgb
    print(f"   Accuracy: {results[model_label]['accuracy']}%  "
          f"F1: {results[model_label]['f1']}%  "
          f"AUC: {results[model_label]['roc_auc']}%")

    # ── Ensemble (soft voting) ───────────────────────────────────────────────
    print("\n🤝 Computing Ensemble (soft voting) …")
    ens_proba = (rf_proba + xgb_proba) / 2
    ens_pred  = (ens_proba >= 0.50).astype(int)

    results["Ensemble"] = {
        "accuracy" : round(accuracy_score(y_test, ens_pred) * 100, 2),
        "f1"       : round(f1_score(y_test, ens_pred) * 100, 2),
        "roc_auc"  : round(roc_auc_score(y_test, ens_proba) * 100, 2),
        "cv_f1"    : round((cv_rf + cv_xgb) / 2 * 100, 2),
    }
    print(f"   Accuracy: {results['Ensemble']['accuracy']}%  "
          f"F1: {results['Ensemble']['f1']}%  "
          f"AUC: {results['Ensemble']['roc_auc']}%")

    # ── Feature importance ───────────────────────────────────────────────────
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    feature_importance = {k: round(float(v) * 100, 2) for k, v in fi.items()}

    # ── Save artefacts ───────────────────────────────────────────────────────
    joblib.dump(rf,     os.path.join(OUTPUT_DIR, "random_forest.pkl"))
    joblib.dump(xgb,    os.path.join(OUTPUT_DIR, "xgboost_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    meta = {
        "features"          : FEATURES,
        "results"           : results,
        "feature_importance": feature_importance,
        "threshold"         : 0.50,
        "model_label"       : model_label,
    }
    with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Models saved to ml/trained_models/")
    print("\n📊 Final Comparison:")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10} {'AUC-ROC':>10}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>9}%  {m['f1']:>9}%  {m['roc_auc']:>9}%")

    print("\n🔑 Top 5 Feature Importances:")
    for feat, imp in list(feature_importance.items())[:5]:
        bar = "█" * int(imp / 3)
        print(f"  {feat:<28} {imp:5.1f}%  {bar}")

    return results, models, scaler


if __name__ == "__main__":
    train()
