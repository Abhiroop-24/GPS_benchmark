from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_raw_trip
from src.features.feature_engineering import FEATURE_COLUMNS
from src.evaluation.metrics import (
    calculate_comprehensive_metrics,
    calculate_observation_top1_accuracy,
    calculate_route_continuity,
)
from src.models.baselines import get_model_pipeline


MODEL_FAMILIES = [
    ("logistic_regression", "Linear/Probabilistic"),
    ("knn", "Instance-Based"),
    ("random_forest", "Ensemble/Bagging"),
    ("lightgbm", "Gradient Boosting (Leaf-wise)"),
    ("xgboost", "Gradient Boosting (Depth-wise)"),
]


def train_and_evaluate_baselines(
    train_parquet: str | Path = "artifacts/features_train.parquet",
    test_parquet: str | Path = "artifacts/features_student_test.parquet",
    output_dir: str | Path = "reports/experiments",
    artifacts_dir: str | Path = "artifacts",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading precomputed feature datasets...")
    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)

    X_train_full = df_train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_full = df_train["label"].to_numpy(dtype=np.int64)
    groups_train = df_train["trip_id"].to_numpy()

    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int64)

    # Preload arcs for student_test trips to compute route continuity
    data_dir = Path("data/student_test")
    test_trip_arcs: Dict[str, np.ndarray] = {}
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            trip = load_raw_trip(d, partition="student_test")
            test_trip_arcs[trip.trip_id] = trip.arcs

    results: List[Dict[str, Any]] = []

    for model_name, family in MODEL_FAMILIES:
        print(f"\n==========================================")
        print(f"Evaluating Model: {model_name} ({family})")
        print(f"==========================================")

        gkf = GroupKFold(n_splits=n_splits)
        cv_macro_f1s = []
        cv_top1_accs = []
        cv_log_losses = []
        cv_train_times = []
        cv_infer_times = []

        # For KNN, use a representative subsample for tractable runtime
        if model_name == "knn":
            np.random.seed(random_state)
            unique_trips = np.unique(groups_train)
            sub_trips = np.random.choice(unique_trips, size=15, replace=False)
            sub_mask = np.isin(groups_train, sub_trips)
            X_cv_use = X_train_full[sub_mask]
            y_cv_use = y_train_full[sub_mask]
            groups_cv_use = groups_train[sub_mask]
            df_train_use = df_train[sub_mask].reset_index(drop=True)
        else:
            X_cv_use = X_train_full
            y_cv_use = y_train_full
            groups_cv_use = groups_train
            df_train_use = df_train

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_cv_use, y_cv_use, groups_cv_use)):
            clf, needs_scale, init_params = get_model_pipeline(model_name, random_state=random_state)

            X_tr, y_tr = X_cv_use[tr_idx], y_cv_use[tr_idx]
            X_val, y_val = X_cv_use[val_idx], y_cv_use[val_idx]

            # Fit preprocessing strictly on training fold only
            if needs_scale:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)

            # Fit model
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            tr_time = time.time() - t0

            # Predict on val fold
            t1 = time.time()
            val_probs = clf.predict_proba(X_val)[:, 1]
            inf_time = time.time() - t1

            # Compute fold metrics
            df_val_sub = df_train_use.iloc[val_idx].copy()
            metrics = calculate_comprehensive_metrics(y_val, val_probs, df_val_sub)

            cv_macro_f1s.append(metrics["macro_f1"])
            cv_top1_accs.append(metrics["top1_accuracy"])
            cv_log_losses.append(metrics["log_loss"])
            cv_train_times.append(tr_time)
            cv_infer_times.append(inf_time)

            print(f"  Fold {fold + 1}/{n_splits} - Macro F1: {metrics['macro_f1']:.4f}, Top-1 Acc: {metrics['top1_accuracy']:.4f}, Log Loss: {metrics['log_loss']:.4f}")

        mean_cv_macro_f1 = float(np.mean(cv_macro_f1s))
        std_cv_macro_f1 = float(np.std(cv_macro_f1s))
        mean_cv_top1 = float(np.mean(cv_top1_accs))
        mean_cv_logloss = float(np.mean(cv_log_losses))
        mean_cv_train_time = float(np.mean(cv_train_times))
        mean_cv_inf_time = float(np.mean(cv_infer_times))

        # Refit on full permitted training data
        print(f"\nRefitting {model_name} on all training data...")
        final_clf, needs_scale, init_params = get_model_pipeline(model_name, random_state=random_state)
        
        if model_name == "knn":
            # Subsample for full fit to keep memory/latency tractable
            np.random.seed(random_state)
            sub_idx = np.random.choice(len(X_train_full), size=80000, replace=False)
            X_fit = X_train_full[sub_idx]
            y_fit = y_train_full[sub_idx]
        else:
            X_fit = X_train_full
            y_fit = y_train_full

        final_scaler = None
        if needs_scale:
            final_scaler = StandardScaler()
            X_fit = final_scaler.fit_transform(X_fit)
            X_test_eval = final_scaler.transform(X_test)
        else:
            X_test_eval = X_test

        t0_refit = time.time()
        final_clf.fit(X_fit, y_fit)
        refit_time = time.time() - t0_refit

        # Inference on student_test partition
        t0_test_inf = time.time()
        test_probs = final_clf.predict_proba(X_test_eval)[:, 1]
        test_inf_time = time.time() - t0_test_inf

        # Student-test evaluation metrics
        df_test_res = df_test.copy()
        df_test_res["pred_prob"] = test_probs
        test_metrics = calculate_comprehensive_metrics(y_test, test_probs, df_test_res)

        # Calculate Route Continuity on student_test trips
        continuities = []
        for trip_id, df_trip in df_test_res.groupby("trip_id"):
            # For each observation, get predicted candidate with highest probability
            idx_max = df_trip.groupby("observation_idx")["pred_prob"].idxmax()
            pred_candidates = df_trip.loc[idx_max, "candidate_arc_id"].to_numpy()
            arcs_matrix = test_trip_arcs[trip_id]
            c_val = calculate_route_continuity(pred_candidates, arcs_matrix)
            continuities.append(c_val)
        mean_route_continuity = float(np.mean(continuities))

        # Check evidence of overfitting / underfitting
        overfit_gap = float(mean_cv_macro_f1 - test_metrics["macro_f1"])
        if abs(overfit_gap) < 0.03:
            fit_evidence = f"Well-generalized (CV F1 {mean_cv_macro_f1:.4f} vs Test F1 {test_metrics['macro_f1']:.4f}, gap {overfit_gap:+.4f})"
        elif overfit_gap > 0.03:
            fit_evidence = f"Mild overfitting (CV F1 exceeds Test F1 by {overfit_gap:.4f})"
        else:
            fit_evidence = f"Strong test generalization (Test F1 exceeds CV F1 by {-overfit_gap:.4f})"

        # Save model predictions
        pred_csv = output_dir / f"predictions_{model_name}.csv"
        df_test_res[["trip_id", "observation_idx", "candidate_arc_id", "label", "pred_prob"]].to_csv(pred_csv, index=False)

        # Save model artifact
        joblib.dump(final_clf, artifacts_dir / f"model_{model_name}.joblib")
        if final_scaler:
            joblib.dump(final_scaler, artifacts_dir / f"scaler_{model_name}.joblib")

        record = {
            "model_name": model_name,
            "algorithm_family": family,
            "initial_hyperparameters": str(init_params),
            "loss_objective": "log_loss / cross_entropy",
            "preprocessing": "StandardScaler" if needs_scale else "None",
            "random_seed": random_state,
            "cv_macro_f1_mean": mean_cv_macro_f1,
            "cv_macro_f1_std": std_cv_macro_f1,
            "cv_top1_accuracy": mean_cv_top1,
            "cv_log_loss": mean_cv_logloss,
            "cv_train_time_s": mean_cv_train_time,
            "cv_inference_time_s": mean_cv_inf_time,
            "refit_train_time_s": refit_time,
            "student_test_top1_accuracy": test_metrics["top1_accuracy"],
            "student_test_macro_f1": test_metrics["macro_f1"],
            "student_test_weighted_f1": test_metrics["weighted_f1"],
            "student_test_log_loss": test_metrics["log_loss"],
            "student_test_precision_macro": test_metrics["precision_macro"],
            "student_test_recall_macro": test_metrics["recall_macro"],
            "student_test_accuracy": test_metrics["accuracy"],
            "student_test_route_continuity": mean_route_continuity,
            "student_test_inference_time_s": test_inf_time,
            "fit_evidence": fit_evidence,
        }
        results.append(record)
        print(f"Test Top-1 Acc: {test_metrics['top1_accuracy']:.4f}, Test Macro F1: {test_metrics['macro_f1']:.4f}, Continuity: {mean_route_continuity:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "model_comparison.csv", index=False)
    df_results.to_csv(artifacts_dir / "experiment_results.csv", index=False)
    print("\nFinal Model Comparison Table:")
    print(df_results[["model_name", "algorithm_family", "cv_macro_f1_mean", "student_test_macro_f1", "student_test_top1_accuracy", "student_test_route_continuity", "refit_train_time_s"]].to_string())

    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate 5 baseline models.")
    parser.add_argument("--train-parquet", type=Path, default=Path("artifacts/features_train.parquet"))
    parser.add_argument("--test-parquet", type=Path, default=Path("artifacts/features_student_test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_and_evaluate_baselines(
        args.train_parquet,
        args.test_parquet,
        args.output_dir,
        args.artifacts_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
