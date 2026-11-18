from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from src.data.loader import load_raw_trip
from src.features.feature_engineering import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.evaluation.metrics import (
    calculate_comprehensive_metrics,
    calculate_route_continuity,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(
    train_parquet: str | Path = "artifacts/features_train.parquet",
    test_parquet: str | Path = "artifacts/features_student_test.parquet",
    n_trials: int = 25,
    n_splits: int = 5,
    random_state: int = 42,
    output_dir: str | Path = "reports/experiments",
    artifacts_dir: str | Path = "artifacts",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading precomputed feature datasets for Bayesian Optimization...")
    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)

    X_train_full = df_train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_full = df_train["label"].to_numpy(dtype=np.int64)
    groups_full = df_train["trip_id"].to_numpy()

    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int64)

    # Preload arcs for student-test continuity evaluation
    test_trip_arcs: Dict[str, np.ndarray] = {}
    for d in sorted(Path("data/student_test").iterdir()):
        if d.is_dir():
            trip = load_raw_trip(d, partition="student_test")
            test_trip_arcs[trip.trip_id] = trip.arcs

    # For fast and efficient Bayesian optimization trials, use a representative 25-trip group subsample
    np.random.seed(random_state)
    unique_trips = np.unique(groups_full)
    sub_trips = np.random.choice(unique_trips, size=25, replace=False)
    sub_mask = np.isin(groups_full, sub_trips)
    X_search = X_train_full[sub_mask]
    y_search = y_train_full[sub_mask]
    groups_search = groups_full[sub_mask]
    df_search = df_train[sub_mask].reset_index(drop=True)

    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 180, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        fold_f1s = []
        for tr_idx, val_idx in gkf.split(X_search, y_search, groups_search):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_search[tr_idx], y_search[tr_idx])
            val_probs = clf.predict_proba(X_search[val_idx])[:, 1]
            df_val = df_search.iloc[val_idx].copy()
            metrics = calculate_comprehensive_metrics(y_search[val_idx], val_probs, df_val)
            fold_f1s.append(metrics["macro_f1"])

        return float(np.mean(fold_f1s))

    print(f"\nStarting Optuna study: {n_trials} trials optimizing 5-fold GroupKFold Macro F1 on train partition...")
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    t0_opt = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    opt_time = time.time() - t0_opt

    print(f"\nOptimization completed in {opt_time:.1f}s.")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Search CV Macro F1: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    # Refit best tuned model on all permitted 80 training trips
    print("\nRefitting best tuned XGBoost model on all 80 training trips...")
    best_params = study.best_params.copy()
    best_params.update({
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": "logloss",
    })

    t0_refit = time.time()
    tuned_clf = xgb.XGBClassifier(**best_params)
    tuned_clf.fit(X_train_full, y_train_full)
    refit_time = time.time() - t0_refit

    # Evaluate tuned model on student-test partition
    t0_inf = time.time()
    test_probs = tuned_clf.predict_proba(X_test)[:, 1]
    test_inf_time = time.time() - t0_inf

    df_test_res = df_test.copy()
    df_test_res["pred_prob"] = test_probs
    test_metrics = calculate_comprehensive_metrics(y_test, test_probs, df_test_res)

    # Calculate Route Continuity
    continuities = []
    for trip_id, df_trip in df_test_res.groupby("trip_id"):
        idx_max = df_trip.groupby("observation_idx")["pred_prob"].idxmax()
        pred_candidates = df_trip.loc[idx_max, "candidate_arc_id"].to_numpy()
        c_val = calculate_route_continuity(pred_candidates, test_trip_arcs[trip_id])
        continuities.append(c_val)
    tuned_continuity = float(np.mean(continuities))

    # Load baseline untuned XGBoost results for comparison
    untuned_res = pd.read_csv(output_dir / "model_comparison.csv")
    untuned_row = untuned_res[untuned_res["model_name"] == "xgboost"].iloc[0]

    comparison = pd.DataFrame([
        {
            "version": "Untuned (Initial Baseline)",
            "cv_macro_f1": untuned_row["cv_macro_f1_mean"],
            "student_test_macro_f1": untuned_row["student_test_macro_f1"],
            "student_test_top1_accuracy": untuned_row["student_test_top1_accuracy"],
            "student_test_log_loss": untuned_row["student_test_log_loss"],
            "student_test_route_continuity": untuned_row["student_test_route_continuity"],
            "inference_time_s": untuned_row["student_test_inference_time_s"],
        },
        {
            "version": "Tuned (Bayesian Optimized)",
            "cv_macro_f1": study.best_value,
            "student_test_macro_f1": test_metrics["macro_f1"],
            "student_test_top1_accuracy": test_metrics["top1_accuracy"],
            "student_test_log_loss": test_metrics["log_loss"],
            "student_test_route_continuity": tuned_continuity,
            "inference_time_s": test_inf_time,
        },
    ])
    comparison.to_csv(output_dir / "tuned_vs_untuned.csv", index=False)
    print("\nUntuned vs Tuned Model Comparison:")
    print(comparison.to_string())

    # Save final model artifact and configuration
    joblib.dump(tuned_clf, artifacts_dir / "model.joblib")
    print(f"\nFinal model saved to {artifacts_dir / 'model.joblib'}")

    final_config = {
        "model_family": "xgboost",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "best_hyperparameters": best_params,
        "cv_macro_f1": float(study.best_value),
        "student_test_metrics": {
            "macro_f1": float(test_metrics["macro_f1"]),
            "top1_accuracy": float(test_metrics["top1_accuracy"]),
            "log_loss": float(test_metrics["log_loss"]),
            "route_continuity": tuned_continuity,
            "inference_time_s": float(test_inf_time),
        },
        "candidate_k": 12,
        "radii": [50.0, 100.0, 150.0],
        "random_state": random_state,
        "optimization_trials": n_trials,
    }
    with open(artifacts_dir / "final_config.json", "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"Final config saved to {artifacts_dir / 'final_config.json'}")

    return final_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna Bayesian Optimization for XGBoost.")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_bayesian_optimization(n_trials=args.n_trials, random_state=args.random_state)


if __name__ == "__main__":
    main()
