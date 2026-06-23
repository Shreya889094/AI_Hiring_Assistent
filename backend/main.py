"""
AI Hiring Assistant – FastAPI Backend
Serves ML predictions, CSV upload, candidate ranking & dashboard data
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, json, os, io
from typing import Optional, List

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
ML_DIR  = os.path.join(BASE, "../ml/trained_models")
DATA_PATH = os.path.join(BASE, "data/candidates_dataset.csv")

# ── Load models ──────────────────────────────────────────────────────────────
rf_model  = joblib.load(os.path.join(ML_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(ML_DIR, "xgboost_model.pkl"))
scaler    = joblib.load(os.path.join(ML_DIR, "scaler.pkl"))
with open(os.path.join(ML_DIR, "model_meta.json")) as f:
    meta = json.load(f)

FEATURES = meta["features"]

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Hiring Assistant API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Feature engineering (must match training) ────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    edu_map = {"B.Sc CS":1,"BCA":2,"B.Tech":3,"MCA":4,"MBA":5,"M.Tech":6,"PhD":7}
    df["education_level"]       = df["education"].map(edu_map).fillna(3)
    df["exp_bucket"]            = pd.cut(df["experience_years"],
                                          bins=[-1,0,2,5,8,100],
                                          labels=[0,1,2,3,4]).astype(int)
    df["overall_score"]         = (0.30*df["skill_score"]
                                  +0.20*df["education_score"]
                                  +0.15*df["certification_score"]
                                  +0.15*df["project_score"]
                                  +0.10*df["communication_score"]
                                  +0.10*df["problem_solving_score"])
    df["exp_skill_interaction"] = df["experience_years"] * df["num_skills"]
    df["cert_per_year"]         = df["certifications"] / (df["experience_years"]+1)
    df["projects_per_year"]     = df["projects_count"]  / (df["experience_years"]+1)
    df["gpa_normalized"]        = (df["gpa"] - 5.0) / 5.0 * 100
    return df

def predict_df(df: pd.DataFrame):
    df = engineer_features(df)
    X  = df[FEATURES].fillna(0)
    Xs = scaler.transform(X)
    rf_prob  = rf_model.predict_proba(Xs)[:,1]
    xgb_prob = xgb_model.predict_proba(Xs)[:,1]
    ens_prob = (rf_prob + xgb_prob) / 2
    pred     = (ens_prob >= meta["threshold"]).astype(int)
    return ens_prob, pred

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AI Hiring Assistant API", "status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

# ── Dashboard stats ──────────────────────────────────────────────────────────
@app.get("/api/dashboard/stats")
def dashboard_stats():
    df = pd.read_csv(DATA_PATH)
    proba, pred = predict_df(df)
    df["ml_shortlisted"]  = pred
    df["ml_score"]        = (proba * 100).round(1)

    shortlisted = int(pred.sum())
    total       = len(df)

    ats_dist = pd.cut(df["ats_score"], bins=[0,40,55,70,85,100],
                      labels=["0-40","41-55","56-70","71-85","86-100"]).value_counts().to_dict()
    edu_dist  = df["education"].value_counts().to_dict()
    dept_dist = df["department"].value_counts().to_dict()
    skill_dist= {}
    for s in df["skills"].dropna():
        for sk in s.split(", "):
            skill_dist[sk] = skill_dist.get(sk, 0) + 1
    top_skills = dict(sorted(skill_dist.items(), key=lambda x: -x[1])[:12])

    top10 = (df[df["ml_shortlisted"]==1]
             .sort_values("ml_score", ascending=False)
             .head(10)[["candidate_id","name","email","education",
                         "experience_years","num_skills","ats_score","ml_score",
                         "job_title_applied","department","skills"]]
             .to_dict(orient="records"))

    return {
        "total_candidates"  : total,
        "shortlisted"       : shortlisted,
        "rejected"          : total - shortlisted,
        "shortlist_rate"    : round(shortlisted/total*100, 1),
        "avg_ats_score"     : round(float(df["ats_score"].mean()), 1),
        "avg_experience"    : round(float(df["experience_years"].mean()), 1),
        "avg_skills"        : round(float(df["num_skills"].mean()), 1),
        "ats_distribution"  : {str(k): int(v) for k, v in ats_dist.items()},
        "education_dist"    : edu_dist,
        "department_dist"   : dept_dist,
        "top_skills"        : top_skills,
        "top_candidates"    : top10,
        "model_accuracy"    : meta["results"]["Ensemble"]["accuracy"],
        "model_f1"          : meta["results"]["Ensemble"]["f1"],
        "model_auc"         : meta["results"]["Ensemble"]["roc_auc"],
        "feature_importance": meta["feature_importance"],
        "hiring_funnel"     : {
            "Applications Received": total,
            "Resumes Screened"     : total,
            "AI Shortlisted"       : shortlisted,
            "Interview Stage"      : int(shortlisted * 0.70),
            "Offer Extended"       : int(shortlisted * 0.40),
            "Hired"                : int(shortlisted * 0.25),
        }
    }

# ── Upload & predict CSV ─────────────────────────────────────────────────────
@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {str(e)}")

    required = ["experience_years","num_skills","certifications","projects_count",
                "gpa","skill_score","education_score","certification_score",
                "project_score","communication_score","problem_solving_score","ats_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    proba, pred = predict_df(df)
    df["ml_shortlisted"] = pred
    df["ml_score"]       = (proba * 100).round(1)
    df["rank"]           = df["ml_score"].rank(ascending=False, method="min").astype(int)

    shortlisted = df[df["ml_shortlisted"]==1].sort_values("ml_score", ascending=False)
    rejected    = df[df["ml_shortlisted"]==0].sort_values("ml_score", ascending=False)

    return {
        "total"          : len(df),
        "shortlisted"    : int(pred.sum()),
        "rejected"       : int((pred==0).sum()),
        "shortlist_rate" : round(pred.mean()*100, 1),
        "candidates"     : df.sort_values("ml_score", ascending=False)
                             .head(50)
                             .fillna("")
                             .to_dict(orient="records"),
        "shortlisted_list": shortlisted.head(30).fillna("").to_dict(orient="records"),
        "rejected_list"   : rejected.head(20).fillna("").to_dict(orient="records"),
    }

# ── Predict single candidate ─────────────────────────────────────────────────
class CandidateIn(BaseModel):
    name                  : str = "Candidate"
    education             : str = "B.Tech"
    gpa                   : float = 7.5
    experience_years      : float = 2.0
    num_skills            : int = 5
    certifications        : int = 2
    projects_count        : int = 3
    skill_score           : float = 70
    education_score       : float = 70
    certification_score   : float = 60
    project_score         : float = 65
    communication_score   : float = 70
    problem_solving_score : float = 70
    ats_score             : float = 68

@app.post("/api/predict/single")
def predict_single(c: CandidateIn):
    df = pd.DataFrame([c.dict()])
    proba, pred = predict_df(df)
    score = float(proba[0]) * 100
    shortlisted = bool(pred[0])
    status = "SHORTLISTED ✅" if shortlisted else "NOT SELECTED ❌"
    strength = ("Excellent" if score>=85 else "Good" if score>=70
                else "Average" if score>=55 else "Below Average")
    return {
        "name"        : c.name,
        "ml_score"    : round(score, 1),
        "shortlisted" : shortlisted,
        "status"      : status,
        "strength"    : strength,
        "rf_score"    : round(float(rf_model.predict_proba(
                           scaler.transform(engineer_features(df)[FEATURES]))[0,1])*100,1),
        "xgb_score"   : round(float(xgb_model.predict_proba(
                           scaler.transform(engineer_features(df)[FEATURES]))[0,1])*100,1),
        "suggestions" : _suggestions(c, score),
    }

def _suggestions(c: CandidateIn, score: float) -> list:
    tips = []
    if c.ats_score < 70:
        tips.append("Improve ATS score by adding relevant keywords")
    if c.certifications < 2:
        tips.append("Add more professional certifications")
    if c.num_skills < 6:
        tips.append("Expand technical skill set")
    if c.experience_years < 2:
        tips.append("Gain more work experience or internships")
    if c.projects_count < 3:
        tips.append("Build more portfolio projects")
    if not tips:
        tips.append("Strong profile – keep updating skills!")
    return tips

# ── All candidates list ──────────────────────────────────────────────────────
@app.get("/api/candidates")
def get_candidates(limit: int = 50, shortlisted_only: bool = False):
    df = pd.read_csv(DATA_PATH)
    proba, pred = predict_df(df)
    df["ml_shortlisted"] = pred
    df["ml_score"]       = (proba * 100).round(1)
    df["rank"]           = df["ml_score"].rank(ascending=False, method="min").astype(int)
    if shortlisted_only:
        df = df[df["ml_shortlisted"]==1]
    df = df.sort_values("ml_score", ascending=False).head(limit)
    return {"candidates": df.fillna("").to_dict(orient="records")}

# ── Model performance ────────────────────────────────────────────────────────
@app.get("/api/model/performance")
def model_performance():
    return {
        "results"           : meta["results"],
        "feature_importance": meta["feature_importance"],
        "features"          : FEATURES,
        "threshold"         : meta["threshold"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
