# NBA Machine Learning Game Predictor & Champion Simulator (WORK IN PROGRESS)

A complete end-to-end Machine Learning and Monte Carlo cicle simulation pipeline designed to predict NBA game outcomes and forecast the next NBA Champion.

Built using official NBA historical box scores, rolling feature engineering, gradient boosting, and probabilistic tournament simulations.

---

##  Architecture & Pipeline Workflow

* **1. Data Collection (`src/data_collection.py`)**
  * Fetches over 9,500 games (2018–2026) using the official `nba_api` (`LeagueGameFinder`).
  * Stores raw team game logs locally in CSV format to prevent API rate limits.

* **2. Data Preprocessing & Feature Engineering (`src/data_processing.py`)**
  * **Zero Data Leakage:** Uses 10-game rolling averages with `.shift(1)` so the model only accesses historical performance prior to tip-off.
  * **Rest Days & Fatigue:** Calculates days since the previous game, identifying Back-to-Back situations (capped at 5 days).
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

## 🛠️ Key Data Science Principles Applied

1. **Zero Data Leakage:** All rolling averages strictly use `.shift(1)` so the model only has access to stats strictly prior to tip-off.
2. **Temporal Validation:** Time-series split (first 80% older games for training, most recent 20% for testing) instead of random shuffling.
3. **Contextual Features:**
   - **Form (Rolling 10):** Points, FG%, 3P%, FT%, Rebounds, Assists, Steals, Blocks, Turnovers.
   - **Fatigue:** Rest days calculation capped at 5 days to model back-to-back fatigue.
   - **Relative Differentials:** Home team metrics minus Away team metrics.

---

##  Model Performance

In professional sports forecasting, random noise and in-game variance make 70%+ accuracy unrealistic. The naive baseline of picking the home team yields ~54.9%.

| Model | Test Accuracy | Precision (Home / Away) | Notes |
| :--- | :---: | :---: | :--- |
| **Naive Home Baseline** | 54.90% | - | Always picking the home team |
| **Logistic Regression** | **62.86%** | 63% / 62% | Fast, well-calibrated baseline |
| **XGBoost Classifier** | **62.39%** | 63% / 62% | Non-linear tree ensemble |

Both models achieve **~63% out-of-sample accuracy**, beating the naive baseline by **+8%**.

---

##  Monte Carlo Simulation Results

By converting model predictions into Bernoulli trials ($P(\text{Home Win}) = p$) and simulating thousands of complete NBA seasons:

| Rank | Team | Probability | Notes |
| :---: | :--- | :---: | :--- |
| 1 | **Denver Nuggets (DEN)** | **34.0%** | Defending contender / High offensive efficiency |
| 2 | **Toronto Raptors (TOR)** | **20.0%** | High regular season win pace in simulation |
| 3 | **San Antonio Spurs (SAS)** | **12.0%** | Western Conference finalist |
| 4 | **New York Knicks (NYK)** | **8.0%** | Eastern Conference contender |
| 5 | **Oklahoma City Thunder (OKC)** | **8.0%** | Young core, top Western seed |
| 6 | **Houston Rockets (HOU)** | **6.0%** | Playoff contender |
| 7 | **Los Angeles Lakers (LAL)** | **4.0%** | Playoff qualifier |
| 8 | **Detroit Pistons (DET)** | **4.0%** | Dark horse surprise |
| 9 | **Atlanta Hawks (ATL)** | **2.0%** | Playoff qualifier |
| 10 | **Cleveland Cavaliers (CLE)** | **2.0%** | Playoff qualifier |


---

## How to Run Locally

### 1. Clone & Setup Environment
```powershell
git clone <YOUR-REPO-URL>
cd ML-NBA
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Pipeline Steps
#### 1. Fetch raw data from NBA API
python src/data_collection.py

#### 2. Preprocess data and generate features
python src/data_processing.py

#### 3. Train the model and save artifacts
python src/train_model.py

#### 4. Run the Monte Carlo Champion Simulation
python src/simulation.py


---

### Vê como este README valoriza o teu esforço:
* Mostra que sabes **porquê** as coisas foram feitas (explica o *Data Leakage*, o *Temporal Split* e o *Baseline* de 54%).
* Explica com clareza as regras da NBA aplicadas na simulação.
* Qualquer recrutador ou colega que visite o teu GitHub consegue entender o projeto em 60 segundos e reproduzi-lo no computador dele!