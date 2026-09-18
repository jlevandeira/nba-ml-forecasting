# NBA Machine Learning Game Predictor & Champion Simulator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red?logo=xgboost)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

A complete end-to-end Machine Learning and Monte Carlo simulation pipeline designed to predict NBA game outcomes and forecast the next NBA Champion.

Built using official NBA historical box scores, rolling feature engineering, gradient boosting, and probabilistic tournament simulations.

---

## Architecture & Pipeline Workflow

* **1. Data Collection (`src/data_collection.py`)**
  * Fetches over 9,500 games (2018–2026) using the official `nba_api` (`LeagueGameFinder`).
  * Stores raw team game logs locally in CSV format to prevent API rate limits.

* **2. Data Preprocessing & Feature Engineering (`src/data_processing.py`)**
  * **Zero Data Leakage:** Uses 10-game rolling averages with `.shift(1)` so the model only accesses historical performance prior to tip-off.
  * **Relative Differentials:** Computes direct head-to-head differences (`Home Stat - Away Stat`).
  * **Target Definition:** Binary classification label (`HOME_WIN = 1` or `0`).

* **3. Model Training & Evaluation (`src/train_model.py`)**
  * **Temporal Split:** Strict chronological 80% train / 20% test split (never shuffle time-series data).
  * **Standardization:** Scales features using `StandardScaler`.
  * **Models Trained:** Logistic Regression baseline and XGBoost Classifier (`max_depth=3` to avoid overfitting).
  * **Model Persistence:** Serializes trained models and scalers to disk using `joblib`.

* **4. Monte Carlo Simulation Engine (`src/simulation.py`)**
  * **Probabilistic Game Engine:** Converts model probabilities into Bernoulli trials (`random.random() < p`).
  * **Dynamic Regular Season:** Simulates conference games to establish the Top 8 seeds for the East and West.
  * **Best-of-7 Playoffs:** Simulates 4 rounds of playoff series following the official NBA 2-2-1-1-1 home-court format.
  * **NBA Finals:** Decides home-court advantage based on superior regular season win record.
  * **Monte Carlo Loop:** Repeats 1,000+ full seasons to compute empirical championship probabilities.

---

## Project Structure

```text
ML-NBA/
├── src/
│   ├── data_collection.py     # Ingests box scores from official nba_api
│   ├── data_processing.py     # Computes rolling metrics and matchup differentials
│   ├── train_model.py         # Trains and evaluates ML models (XGBoost)
│   └── simulation.py          # Monte Carlo engine for season & playoff simulation
├── data/
│   ├── raw/                   # Raw historical game logs
│   └── processed/             # Cleaned feature tables ready for modeling
├── models/                    # Serialized models and scalers (.joblib)
├── THEORY.md                  # Comprehensive mathematical and theoretical documentation
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview and instructions
```

---

## Key Data Science Principles Applied

1. **Zero Data Leakage:** All rolling averages strictly use `.shift(1)` so the model only has access to stats strictly prior to tip-off.
2. **Temporal Validation:** Time-series split (first 80% older games for training, most recent 20% for testing) instead of random shuffling.
3. **Contextual Features:**
   - **Form (Rolling 10):** Points, FG%, 3P%, FT%, Rebounds, Assists, Steals, Blocks, Turnovers.
   - **Relative Differentials:** Direct head-to-head advantage (`Home Stat - Away Stat`).

---

## Model Performance

In professional sports forecasting, random noise and in-game variance make 70%+ accuracy unrealistic. The naive baseline of picking the home team yields ~54.9%.

| Model | Test Accuracy | Precision (Home / Away) | Notes |
| :--- | :---: | :---: | :--- |
| **Naive Home Baseline** | 54.90% | - | Always picking the home team |
| **Logistic Regression** | **62.86%** | 63% / 62% | Fast, well-calibrated baseline |
| **XGBoost Classifier** | **62.39%** | 63% / 62% | Non-linear tree ensemble |

Both models achieve **~63% out-of-sample accuracy**, beating the naive baseline by **+8%**.

---

## Monte Carlo Simulation Results

Empirical championship probabilities based on **1,000 complete simulated NBA seasons** (Regular season round-robin + Playoff best-of-7 series):

| Rank | Team | Probability | Titles (out of 1,000) |
| :---: | :--- | :---: | :---: |
| 1 | **San Antonio Spurs (SAS)** | **20.9%** | 209 |
| 2 | **Denver Nuggets (DEN)** | **15.9%** | 159 |
| 3 | **Oklahoma City Thunder (OKC)** | **12.7%** | 127 |
| 4 | **Houston Rockets (HOU)** | **9.3%** | 93 |
| 5 | **Boston Celtics (BOS)** | **8.9%** | 89 |
| 6 | **New York Knicks (NYK)** | **6.0%** | 60 |
| 7 | **Toronto Raptors (TOR)** | **4.5%** | 45 |
| 8 | **Atlanta Hawks (ATL)** | **3.8%** | 38 |
| 9 | **Detroit Pistons (DET)** | **3.3%** | 33 |
| 10 | **Cleveland Cavaliers (CLE)** | **3.0%** | 30 |
| 11 | **Charlotte Hornets (CHA)** | **2.8%** | 28 |
| 12 | **Los Angeles Clippers (LAC)** | **2.7%** | 27 |
| 13 | **Philadelphia 76ers (PHI)** | **2.4%** | 24 |
| 14 | **Miami Heat (MIA)** | **1.2%** | 12 |
| 15 | **Los Angeles Lakers (LAL)** | **1.1%** | 11 |
| 16 | **Indiana Pacers (IND)** | **0.7%** | 7 |
| 17 | **Portland Trail Blazers (POR)** | **0.2%** | 2 |
| 18 | **Chicago Bulls (CHI)** | **0.2%** | 2 |
| 19 | **Washington Wizards (WAS)** | **0.1%** | 1 |
| 20 | **Utah Jazz (UTA)** | **0.1%** | 1 |
| 21 | **Phoenix Suns (PHX)** | **0.1%** | 1 |
| 22 | **Minnesota Timberwolves (MIN)** | **0.1%** | 1 |

---

##  How to Run Locally

### 1. Clone & Setup Environment

```powershell
# Clone the repository
git clone https://github.com/jlevandeira/nba-ml-forecasting.git
cd ML-NBA

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Pipeline Steps

Execute the pipeline sequentially from the project root:

```powershell
# Step 1: Fetch raw data from NBA API (historical box scores)
python src/data_collection.py

# Step 2: Preprocess data and generate rolling features
python src/data_processing.py

# Step 3: Train models (XGBoost) and export artifacts (.joblib)
python src/train_model.py

# Step 4: Run the Monte Carlo Champion Simulation (1,000 seasons)
python src/simulation.py
```

---

## Roadmap / Upcoming Features

- [ ] **NBA MVP Predictor:** Supervised classification & ranking model to forecast individual MVP award winners.
- [ ] **Interactive CLI (`main.py`):** Unified terminal menu to select between Champion Simulation and MVP prediction.