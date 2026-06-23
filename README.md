# 🤖 NexusHire AI – AI Hiring Assistant
### Final Year B.Tech Major Project | Production-Ready ML Platform

---

## 🚀 QUICK START (3 Steps)

### Step 1 — Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

### Step 2 — Train ML Models
```bash
# Generate dataset + train Random Forest & XGBoost
python backend/data/generate_dataset.py
python ml/train_model.py
```

> ✅ Expected output: **98% Accuracy | 96.97% F1 | 99.82% AUC-ROC**

### Step 3 — Start Backend API
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4 — Open Frontend
Open `frontend/index.html` in browser, OR use VS Code **Live Server** extension:
- Right-click `frontend/index.html` → **Open with Live Server**

---

## 📁 Project Structure

```
ai-hiring-assistant/
│
├── 📂 backend/
│   ├── main.py                    # FastAPI REST API (8 endpoints)
│   ├── data/
│   │   ├── generate_dataset.py    # Creates 500-candidate CSV dataset
│   │   └── candidates_dataset.csv # Auto-generated dataset
│
├── 📂 ml/
│   ├── train_model.py             # ML Pipeline: RF + XGBoost + Ensemble
│   └── trained_models/
│       ├── random_forest.pkl      # Trained Random Forest model
│       ├── xgboost_model.pkl      # Trained XGBoost model
│       ├── scaler.pkl             # Feature scaler
│       └── model_meta.json        # Metrics & feature importance
│
├── 📂 frontend/
│   └── index.html                 # Full dashboard (7 pages, no build step!)
│
├── 📂 .vscode/
│   ├── settings.json              # Python interpreter & Live Server config
│   ├── launch.json                # Run configs (F5 to start backend)
│   └── extensions.json            # Recommended extensions
│
├── requirements.txt               # All Python dependencies
└── README.md                      # This file
```

---

## 🧠 ML Models

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Random Forest (200 trees) | 98.0% | 96.88% | 99.91% |
| XGBoost | 98.0% | 96.97% | 99.82% |
| **Ensemble (Soft Voting)** | **98.0%** | **96.97%** | **99.82%** |

### Features Used (19 total)
- `ats_score` (42% importance)
- `overall_score` (17.2%)
- `skill_score` (8.7%)
- `exp_skill_interaction` (8.2%)
- `num_skills`, `experience_years`, `education_score`, `project_score` ...

### Shortlisting Formula
```
Shortlisted IF:
  ML Ensemble Score >= 0.50 (50%)
  AND experience_years >= 1
  AND num_skills >= 4

Overall Score = 0.30×skill + 0.20×education + 0.15×certification
             + 0.15×project + 0.10×communication + 0.10×problem_solving
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/api/dashboard/stats` | Full dashboard data |
| POST | `/api/predict/csv` | Upload CSV → get shortlisting |
| POST | `/api/predict/single` | Predict single candidate |
| GET | `/api/candidates` | All candidates with ML scores |
| GET | `/api/model/performance` | Model metrics & feature importance |

### Test API
```bash
# Dashboard
curl http://localhost:8000/api/dashboard/stats

# Single prediction
curl -X POST http://localhost:8000/api/predict/single \
  -H "Content-Type: application/json" \
  -d '{"name":"Priya","education":"M.Tech","gpa":8.5,"experience_years":4,"num_skills":8,"certifications":3,"projects_count":6,"skill_score":82,"education_score":85,"certification_score":55,"project_score":70,"communication_score":80,"problem_solving_score":78,"ats_score":81}'
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | Stats, charts, top candidate ranking |
| 👥 Candidates | Full searchable candidate database |
| 📂 Upload CSV | Drag & drop CSV → instant ML prediction |
| 🎯 Quick Predict | Enter any candidate's details → instant decision |
| 🔻 Hiring Funnel | Visual recruitment pipeline |
| 🧠 ML Model | Feature importance, model comparison radar chart |
| ⚡ Skill Insights | Top skills demand bar chart + skill cloud |

---

## 📊 Upload Your Own CSV

CSV must have these columns:
```
experience_years, num_skills, certifications, projects_count, gpa,
skill_score, education_score, certification_score, project_score,
communication_score, problem_solving_score, ats_score
```
Optional: `candidate_id, name, email, education, department, job_title_applied`

---

## 🛠️ VS Code Shortcuts

- **F5** → Starts FastAPI backend
- **Right-click index.html → Open with Live Server** → Opens dashboard
- **Ctrl+Shift+P → Python: Select Interpreter** → choose `venv`

---

## 🎓 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JS, Chart.js |
| Backend | FastAPI, Python 3.10+ |
| ML Models | Random Forest, XGBoost, Scikit-learn |
| Feature Eng | Pandas, NumPy |
| Model Storage | Joblib |
| API Server | Uvicorn |

---

## 📝 Project Info

- **Project Name**: NexusHire AI — AI Hiring Assistant
- **Type**: Final Year B.Tech Major Project
- **Domain**: AI/ML, Recruitment Automation, HR Tech
- **Dataset**: 500 synthetic candidates (auto-generated)
- **Models Trained**: Random Forest + XGBoost Ensemble
- **Model Accuracy**: 98% | AUC-ROC: 99.82%
