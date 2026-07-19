# 🛡️ Attack on Phishing

A machine learning-powered phishing URL detection system — a two-stream stacked ensemble model served through a Flask backend, with a full analytics dashboard, batch scanning, and explainability built on top.

**🔴 Live Demo:** [https://phishing-website-detection-8v50.onrender.com](https://phishing-website-detection-8v50.onrender.com)

> Hosted on Render's free tier — the first request after a period of inactivity may take 30-60 seconds to wake up. This is expected.

---

## ✨ Features

- **🔍 Scanner** — paste any URL and get an instant phishing/legitimate verdict, with a confidence score, threat meter, and a breakdown of the individual signals that drove the decision
- **🧠 Explainability** — a "Why This Was Flagged" panel showing the top features pushing a specific prediction, powered by SHAP
- **📊 Analytics Dashboard** — verdict split, suspicion score trend over time, and the most common red flags among flagged URLs — all rendered as dependency-free inline SVG charts
- **📋 Batch Scan** — upload a CSV or paste a list of URLs to scan many at once, with results exportable back to CSV
- **🔗 Redirect chain tracking** — see every hop a URL takes before landing
- **👍 Feedback loop** — mark whether a verdict was correct, logged for future model improvement
- **❓ How It Works** — an in-app explainer modal describing the full detection pipeline

## 🧠 How the Model Works

The detection model is a **two-stream stacked ensemble**:

```
71 features → LightGBM ─┐
                          ├─→ Logistic Regression (meta-learner) → phishing probability
71 features → Random Forest ─┘
```

- **~47 URL-level features** (entropy, IP usage, fake TLDs, brand spoofing, punycode, etc.) — computed without needing to fetch the page
- **~23 HTML/content features** (forms, external links, hidden elements, inline scripts, etc.) — computed by fetching and parsing the live page
- A hand-crafted **suspicion score** feeds in as an additional signal

A verdict isn't just "trust the model" — it goes through a layered decision process:
1. **Hard overrides** — raw IP hosts, known URL shorteners, and disposable/abuse-prone TLDs are flagged immediately, bypassing the model
2. **Dual-threshold rule** — everything else needs *both* the model's probability *and* the suspicion score to clear their thresholds before being flagged, cutting down on false positives
3. **Known-legitimate allowlist** — a short list of well-known domains gets a stricter bar before being flagged at all

See the in-app **"How It Works"** modal for the full breakdown.

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Model | scikit-learn (StackingClassifier), LightGBM, SHAP |
| Backend | Flask, gunicorn/waitress |
| Storage | SQLite (scan history, feedback) |
| Frontend | Vanilla HTML/CSS/JS — no framework, no external CDN dependencies |
| Deployment | Render |

## 📁 Repository Structure

```
├── app.py                    # Flask backend — API, feature extraction, decision logic
├── index.html                # Frontend — Scanner, Dashboard, Batch Scan, all in one file
├── train_and_save.py         # Model training script
├── requirements.txt          # Pinned dependencies
├── Procfile                  # Production start command (for Render/Heroku-style hosts)
├── model/
│   └── model_artifacts.pkl   # Trained model + imputer + scaler + feature lists
├── notebook/                 # Model development notebook (EDA, feature engineering, training)
├── docs/
│   └── paper.pdf             # Project write-up / report
└── sample_batch_urls.csv     # Example file for testing the Batch Scan feature
```

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/Phishing-Website-Detection.git
cd Phishing-Website-Detection
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000/** — the backend serves the frontend directly, so there's nothing else to configure.

For a production-style local run instead of Flask's dev server:
```bash
python -m waitress --host=127.0.0.1 --port=5000 app:app
```

## ⚠️ Known Limitations

- SQLite isn't safe for high-concurrency writes — fine for a demo/single-user scan history, not a multi-user production deployment
- The free hosting tier caps memory at 512MB, which is tight for this model stack (scikit-learn + LightGBM + pandas); SHAP is disabled in the deployed version specifically for this reason, but still available when running locally
- The model reflects a training-time snapshot — phishing patterns evolve, so periodic retraining (using the feedback log this app already collects) would be needed for long-term accuracy

## 📄 License

*(add your license here, e.g. MIT)*

## 👥 Team

*(add team member names here)*
