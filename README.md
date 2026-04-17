# 🌱 CropAI Platform — SKB_P3
### Autonomous Multi-Omics Fusion for Climate-Resilient Crop Breeding

---

## 📁 Files in This Project

```
crop_resilience_platform/
├── app.py                ← Main Streamlit dashboard (RUN THIS)
├── data_generator.py     ← Generates omics + climate data
├── model.py              ← AI models: Random Forest, XGBoost, Gradient Boost
├── scenario_planner.py   ← Climate scenario simulator
├── requirements.txt      ← All dependencies
└── README.md             ← This file
```

---

## 🚀 How to Run in VS Code (Step by Step)

### Step 1 — Open folder in VS Code
- File → Open Folder → select `crop_resilience_platform`

### Step 2 — Open Terminal in VS Code
- Terminal → New Terminal  (or press Ctrl + `)

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

Your browser will open at → **http://localhost:8501**

---

## 🤖 AI Models Included

| Model | Best For | Speed |
|---|---|---|
| **Random Forest** | Noisy omics data, beginners (DEFAULT) | Fast |
| **XGBoost** | Highest accuracy | Medium |
| **Gradient Boost** | Sklearn baseline comparison | Fast |

Switch between them in the **sidebar** — no code changes needed!

---

## 📊 Platform Features

| Page | What it does |
|---|---|
| 🏠 Overview | Platform summary, key metrics |
| 🔬 Data Explorer | View omics data, correlation matrix |
| 🤖 Model & Train | Feature importance, custom prediction, model comparison |
| 🌦️ Scenario Planner | Simulate 54 climate scenarios |
| 🧬 Top Genotypes | Rank best parents, generate breeding blueprint |

---

## ☁️ Deploy for Free (Hackathon Submission)

1. Create a free account at **github.com**
2. Upload all these files to a new GitHub repository
3. Go to **share.streamlit.io** → Connect GitHub
4. Select your repo → Click **Deploy**
5. Share your public URL!

---

## 🔧 Troubleshooting

**`pip` not recognized?**
```bash
python -m pip install -r requirements.txt
```

**`streamlit` not recognized?**
```bash
python -m streamlit run app.py
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```
