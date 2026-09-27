# RailSync AI 🚆
**AI-Powered Automatic Block Planning for Indian Railways**
Smart India Hackathon 2026 Prototype

---

## What It Does
RailSync AI recommends the best time window to perform railway track/asset maintenance
while causing minimum disruption to train operations.

---

## Project Structure
```
rail_sync_ai/
├── data/
│   ├── trains.csv          ← auto-generated synthetic train schedules
│   └── maintenance.csv     ← auto-generated maintenance tasks
├── src/
│   ├── generate_data.py    ← creates synthetic CSV datasets
│   ├── preprocess.py       ← cleans and validates data
│   ├── traffic_prediction.py ← Random Forest traffic model
│   ├── conflict_detection.py ← finds trains affected by a window
│   ├── optimizer.py        ← scores and ranks maintenance windows
│   └── what_if.py          ← what-if simulation + re-planning
├── app.py                  ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run (Windows / VS Code)

### 1. Install Python 3.10+
Download from https://python.org and ensure "Add to PATH" is checked.

### 2. Open Terminal in VS Code
Press `` Ctrl+` `` to open the integrated terminal.

### 3. Navigate to project folder
```bash
cd c:\Users\Admin\Downloads\hackathon\rail_sync_ai
```

### 4. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Generate synthetic data (one-time)
```bash
python src/generate_data.py
```

### 7. Run the dashboard
```bash
streamlit run app.py
```

The browser will open automatically at http://localhost:8501

---

## File-by-File Explanation

| File | Purpose |
|------|---------|
| `generate_data.py` | Creates 120 train records and 30 maintenance records as CSV files |
| `preprocess.py` | Loads CSVs, fixes missing values, converts times to minutes |
| `traffic_prediction.py` | Trains a Random Forest to predict train count in any time window |
| `conflict_detection.py` | Checks which trains overlap a given maintenance window |
| `optimizer.py` | Scores all windows using a weighted formula, returns top 3 |
| `what_if.py` | Lets user test custom windows and re-plan with updated schedules |
| `app.py` | Streamlit dashboard tying everything together |

---

## Disclaimer
This is a **prototype decision-support system only**.
It does NOT connect to real Indian Railways systems and must NOT be used
to directly control signals, switches, or train operations.
