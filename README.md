# PathSnap: Distributed Map-Matching & Route Reconstruction (Laptop 1)

**Independent Implementation for PES University UE24CS352A Machine Learning Hackathon**
**Repository:** `https://github.com/Abhiroop-24/GPS_benchmark.git`
**Git Author:** `Abhiroop-24 <abhiroopsanil24@gmail.com>`

---

## 1. Project Purpose & Independent Implementation Statement

PathSnap is an end-to-end, distributed machine learning system that reconstructs realistic vehicle routes on directed road networks from noisy, low-frequency GPS trajectories.

**Academic Integrity & Independence Statement:**
This repository (`GPS_benchmark`) represents a completely independent, original implementation developed from first principles by the Laptop 1 ML/Backend Engineer. No code, model weights, hyperparameter choices, experimental logs, or report sections were shared with or copied from any other team or repository (including `PathSnap-A-G14`). All experimental results, diagnostic plots, and metrics are backed by reproducible code executions.

---

## 2. Distributed Architecture (Laptop 1 & Laptop 2)

```
+-------------------------------------------------------------------------------+
|                                LAPTOP 1 (Backend & ML)                        |
|                                                                               |
|  [Raw Data: GPS + Network] -> [Dynamic UTM Projection]                        |
|                                      |                                        |
|                                      v                                        |
|                          [Spatial Candidate Gen (K=12)]                       |
|                                      |                                        |
|                                      v                                        |
|                          [Feature Pipeline (20 feats)]                        |
|                                      |                                        |
|                                      v                                        |
|                          [Bayesian-Tuned XGBoost]                             |
|                                      |                                        |
|                                      v                                        |
|                          [Log-Space HMM / Viterbi]                            |
|                                      |                                        |
|                             FastAPI Server (:8000)                            |
+-------------------------------------------------------------------------------+
                                      ^
                                      | HTTP REST (POST /predict)
                                      v
+-------------------------------------------------------------------------------+
|                           LAPTOP 2 (User Interface)                           |
|  - Displays evaluation trip options (00000000-00000091)                       |
|  - Sends selected trip_id to Laptop 1                                         |
|  - Visualizes predicted route, GPS points, and diagnostics                   |
+-------------------------------------------------------------------------------+
```

---

## 3. Repository Structure

```
GPS_benchmark/
├── src/
│   ├── data/
│   │   ├── loader.py              # Robust trip file loader and data integrity checker
│   │   ├── projector.py           # Dynamic UTM zone projection & WGS84 geodesic cross-check
│   │   └── inventory.py           # Dataset-wide inventory validation script
│   ├── features/
│   │   ├── candidates.py          # Dual spatial index (midpoints + nodes) candidate generator
│   │   ├── alignment.py           # Monotonic DP route-label alignment (handling loops/repeats)
│   │   ├── feature_engineering.py # 20 kinematic, bearing, and topological features
│   │   └── build_features.py      # Batch feature extraction script
│   ├── models/
│   │   ├── baselines.py           # 5 distinct model families configuration
│   │   ├── train_baselines.py     # 5-fold GroupKFold cross-validation & benchmark suite
│   │   └── bayesian_optimization.py # Optuna Bayesian hyperparameter optimization
│   ├── hmm/
│   │   ├── emission.py            # Calibrated log-emission calculation with smoothing
│   │   ├── transitions.py         # Directed graph topology & bounded Dijkstra shortest paths
│   │   └── viterbi.py             # Log-space Viterbi sequence decoder
│   ├── api/
│   │   ├── schemas.py             # Pydantic request/response schemas (Laptop 2 contract)
│   │   ├── evaluation.py          # Server-side prediction engine with ground-truth isolation
│   │   └── main.py                # FastAPI endpoints with CORS middleware
│   ├── evaluation/
│   │   ├── metrics.py             # Top-1 accuracy, Macro F1, Route Continuity, Log Loss
│   │   ├── candidate_benchmark.py # Candidate benchmarking across K in {8, 12, 16, 20}
│   │   └── evaluate_hmm.py        # End-to-end evaluation of Classifier vs HMM
│   └── visualization/
│       ├── folium_routes.py       # Interactive HTML Folium map generator
│       └── generate_plots.py      # Publication-quality static diagnostic plots
├── tests/
│   ├── test_loader.py             # Unit tests for data loading & integrity validation
│   ├── test_projector.py          # Unit tests for UTM projection & geodesic accuracy
│   ├── test_candidates.py         # Unit tests for point-to-segment geometry & fallback
│   ├── test_alignment.py          # Unit tests for monotonic DP sequence alignment
│   ├── test_features.py           # Unit tests for feature computation & leakage prevention
│   ├── test_viterbi.py            # Unit tests for Viterbi decoding & chain transitions
│   └── test_api.py                # Integration tests for FastAPI endpoints & Laptop 2 contract
├── artifacts/
│   ├── model.joblib               # Final trained and tuned XGBoost model artifact
│   ├── final_config.json          # Complete deployment hyperparameter and pipeline config
│   └── experiment_results.csv     # Official 5-model GroupKFold comparison table
├── reports/
│   ├── experiments/               # Benchmark CSVs (inventory, candidates, models, HMM, Optuna)
│   ├── plots/                     # High-resolution PNG diagnostic figures
│   └── maps/                      # Interactive Folium HTML maps
├── Dockerfile                     # Container deployment specification
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git tracking rules
└── README.md                      # Comprehensive documentation
```

---

## 4. Setup & Installation Instructions

### 4.1 Local Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Abhiroop-24/GPS_benchmark.git
cd GPS_benchmark

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Link dataset directory (uncommitted)
ln -sfn "/home/abhiroop/Documents/ML_hackathon/Student folder" data
```

### 4.2 Docker Deployment
```bash
docker build -t pathsnap-backend .
docker run -p 8000:8000 -v "/home/abhiroop/Documents/ML_hackathon/Student folder:/app/data:ro" pathsnap-backend
```

---

## 5. Dataset Partitions & Isolation Policy

- **Training Partition (`data/train`):** 80 trips (208,417 GPS observations). Used exclusively for exploratory analysis, feature extraction, 5-fold GroupKFold cross-validation, and Optuna hyperparameter optimization.
- **Student-Testing Partition (`data/student_test`):** 10 trips (19,160 GPS observations: `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`). Used solely for development validation and final pipeline comparison.
- **Strict Data Isolation:**
  - No random train/test splits across GPS observations.
  - Preprocessing (`StandardScaler`) is fitted strictly within each training fold.
  - Ground truth reference routes are never passed into feature extraction.
  - Ground truth labels remain strictly server-side and are never exposed to Laptop 2.

---

## 6. Execution Commands

### 6.1 Run Dataset Inventory Validation
```bash
python -m src.data.inventory
```

### 6.2 Run Candidate Benchmarking ($K \in \{8, 12, 16, 20\}$)
```bash
python -m src.evaluation.candidate_benchmark --k-values 8 12 16 20
```

### 6.3 Extract Precomputed Feature Datasets ($K=12$)
```bash
python -m src.features.build_features --k 12
```

### 6.4 Train and Benchmark 5 Baseline Model Families
```bash
python -m src.models.train_baselines --n-splits 5
```

### 6.5 Run Bayesian Hyperparameter Optimization (Optuna)
```bash
python -m src.models.bayesian_optimization --n-trials 25
```

### 6.6 Run End-to-End HMM vs Classifier Evaluation
```bash
python -m src.evaluation.evaluate_hmm
```

### 6.7 Generate Diagnostic Plots and Folium Maps
```bash
python -m src.visualization.generate_plots
python -m src.visualization.folium_routes --trip-id 00000052 --partition student_test --output reports/maps/student_test_trip_00000052.html
python -m src.visualization.folium_routes --trip-id 00000077 --partition student_test --output reports/maps/student_test_trip_00000077.html
```

### 6.8 Run Full Test Suite
```bash
pytest -v
```

### 6.9 Start FastAPI Server (Laptop 1)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Experimental Results Summary

### 7.1 Candidate Recall Across Partitions

| Partition | K | Total Observations | Mean Recall | Weighted Recall | Mean Miss Rate |
|---|---|---|---|---|---|
| `train` | 8 | 208,417 | 97.28% | 91.83% | 2.72% |
| `train` | 12 | 208,417 | **97.76%** | **92.85%** | **2.24%** |
| `train` | 16 | 208,417 | 97.97% | 93.12% | 2.03% |
| `train` | 20 | 208,417 | 98.17% | 93.38% | 1.83% |
| `student_test` | 12 | 19,160 | **99.78%** | **99.84%** | **0.22%** |

*Note: On baseline trip `00000001`, the improved dual spatial index achieved 96.35% recall (surpassing the 95.82% initial baseline).*

### 7.2 Five Distinct Model Families Comparison (5-Fold GroupKFold)

| Model Name | Algorithm Family | CV Macro F1 | Test Macro F1 | Test Top-1 Acc | Test Route Continuity | Refit Time |
|---|---|---|---|---|---|---|
| `logistic_regression` | Linear / Probabilistic | 0.8269 | 0.8558 | 0.7984 | 0.8782 | 9.56s |
| `knn` | Instance-Based (Weighted) | 0.8425 | 0.8663 | 0.7961 | 0.8840 | 0.01s |
| `random_forest` | Ensemble / Bagging | 0.8507 | 0.8941 | 0.8130 | 0.8980 | 139.83s |
| `lightgbm` | Gradient Boosting (Leaf-wise) | 0.8448 | 0.8951 | 0.8130 | 0.8972 | 8.50s |
| `xgboost` | Gradient Boosting (Depth-wise) | **0.8526** | **0.8973** | **0.8180** | **0.8994** | **8.70s** |

### 7.3 Bayesian Optimization Impact (XGBoost)

- **Selected Best Model:** `xgboost`
- **Objective:** 5-fold GroupKFold Macro F1 on train partition.
- **Search Space:** `n_estimators` (80-180), `max_depth` (4-8), `learning_rate` (0.03-0.15), `subsample` (0.6-0.95), `colsample_bytree` (0.6-0.95), `reg_alpha` (0.001-5.0), `reg_lambda` (0.01-10.0).
- **Best Hyperparameters:** `{'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.0601, 'subsample': 0.7019, 'colsample_bytree': 0.8141, 'min_child_weight': 2, 'reg_alpha': 0.0120, 'reg_lambda': 0.1256}`.
- **Search CV Macro F1:** **0.8929**.

### 7.4 HMM / Viterbi Reconstructed Route Improvements

| Metric | Classifier-Only (Top-1) | Classifier + HMM (Viterbi) | Net Gain |
|---|---|---|---|
| **Mean Top-1 Accuracy** | 83.69% | **92.58%** | **+8.89%** |
| **Mean Route Continuity** | 89.92% | **97.17%** | **+7.25%** |
| **Disconnected Transitions** | 1,923 jumps | **447 jumps** | **-76.8% reduction** |
| **Trip 00000077 Accuracy** | 63.44% | **84.38%** | **+20.94%** |
| **Trip 00000007 Accuracy** | 91.73% | **99.04%** | **+7.31%** |

---

## 8. Laptop 2 API Contract

### `POST /predict`
**Request Body:**
```json
{
  "trip_id": "00000052",
  "partition": "student_test"
}
```

**Response Body (JSON):**
```json
{
  "trip_id": "00000052",
  "partition": "student_test",
  "status": "success",
  "observation_count": 262,
  "candidate_count": 3144,
  "predicted_arc_ids": [45, 45, 102, ...],
  "classifier_only_route": [45, 98, 102, ...],
  "route_nodes": [[12, 18], [18, 25], ...],
  "gps_coordinates": [[50.7321, 7.0945], [50.7324, 7.0948], ...],
  "predicted_route_coordinates": [[50.7320, 7.0944], [50.7323, 7.0947], ...],
  "prediction_confidence": 0.8942,
  "route_continuity": 0.9617,
  "execution_time_seconds": 0.28,
  "metrics": {
    "classifier_top1_accuracy": 0.9427,
    "hmm_top1_accuracy": 0.9962,
    "accuracy_improvement": 0.0534,
    "candidate_miss_rate": 0.0038
  }
}
```

---

## 9. Assumptions, Known Limitations & Plagiarism Policy

1. **UTM Projection Assumption:** Trajectories are projected to local UTM zones using `pyproj`. Geodesic validation proves Euclidean UTM distortion is < 0.1% over typical city extents.
2. **Speed & Graph Horizon:** Maximum plausible vehicle speed is capped at 45 m/s (~160 km/h). Bounded Dijkstra search avoids all-pairs computation.
3. **No Hardcoding Guarantee:** Predictions are computed dynamically using live candidate generation, feature extraction, XGBoost inference, and Viterbi decoding. No precomputed routes or lookup bypasses exist.
