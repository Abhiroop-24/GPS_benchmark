from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_all_plots(
    reports_dir: str | Path = "reports",
    artifacts_dir: str | Path = "artifacts",
) -> None:
    reports_dir = Path(reports_dir)
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = reports_dir / "experiments"

    print("Generating comprehensive diagnostic plots from actual experimental results...")

    # 1. Dataset Inventory & Trajectory Extents
    inv_csv = exp_dir / "trip_inventory.csv"
    if inv_csv.exists():
        df_inv = pd.read_csv(inv_csv)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(data=df_inv, x="observations", hue="partition", kde=True, ax=axes[0], palette="Set2")
        axes[0].set_title("GPS Trajectory Length Distribution (Observations)")
        axes[0].set_xlabel("Observations per Trip")
        axes[0].set_ylabel("Count of Trips")

        sns.histplot(data=df_inv, x="arcs", hue="partition", kde=True, ax=axes[1], palette="Set2")
        axes[1].set_title("Road Network Size Distribution (Arc Count)")
        axes[1].set_xlabel("Road Network Arcs")
        axes[1].set_ylabel("Count of Trips")

        fig.savefig(plots_dir / "trip_inventory.png", dpi=300)
        plt.close(fig)
        print("Saved: trip_inventory.png")

    # 2. Candidate Recall vs K
    cand_csv = exp_dir / "candidate_summary_by_k.csv"
    if cand_csv.exists():
        df_cand = pd.read_csv(cand_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=df_cand,
            x="k",
            y="mean_recall",
            hue="partition",
            marker="o",
            linewidth=2.5,
            markersize=8,
            ax=ax,
            palette="tab10",
        )
        ax.set_title("Candidate Top-K Recall Across Partitions (K in {8, 12, 16, 20})")
        ax.set_xlabel("Candidate Count (K)")
        ax.set_ylabel("Mean Top-K Recall")
        ax.set_ylim([0.90, 1.005])
        for _, row in df_cand.iterrows():
            ax.annotate(
                f"{row['mean_recall']:.2%}",
                (row["k"], row["mean_recall"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )
        fig.savefig(plots_dir / "candidate_recall.png", dpi=300)
        plt.close(fig)
        print("Saved: candidate_recall.png")

    # 3. Model Comparison: CV vs Student-Test Macro F1
    model_csv = exp_dir / "model_comparison.csv"
    if model_csv.exists():
        df_models = pd.read_csv(model_csv)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_models))
        width = 0.35

        ax.bar(x - width/2, df_models["cv_macro_f1_mean"], width, label="CV Macro F1 (Train)", color="#4c72b0")
        ax.bar(x + width/2, df_models["student_test_macro_f1"], width, label="Student-Test Macro F1", color="#55a868")

        ax.set_title("Model Comparison: 5 Distinct Families (CV vs Student-Test Macro F1)")
        ax.set_xticks(x)
        ax.set_xticklabels(df_models["model_name"], rotation=15)
        ax.set_ylabel("Macro F1-Score")
        ax.set_ylim([0.70, 1.0])
        ax.legend()

        for i in range(len(df_models)):
            ax.text(x[i] - width/2, df_models["cv_macro_f1_mean"][i] + 0.01, f"{df_models['cv_macro_f1_mean'][i]:.3f}", ha="center", fontsize=8)
            ax.text(x[i] + width/2, df_models["student_test_macro_f1"][i] + 0.01, f"{df_models['student_test_macro_f1'][i]:.3f}", ha="center", fontsize=8)

        fig.savefig(plots_dir / "model_comparison.png", dpi=300)
        plt.close(fig)
        print("Saved: model_comparison.png")

        # 4. Top-1 Accuracy and Route Continuity
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, df_models["student_test_top1_accuracy"], width, label="Top-1 Accuracy", color="#c44e52")
        ax.bar(x + width/2, df_models["student_test_route_continuity"], width, label="Route Continuity", color="#8172b3")

        ax.set_title("Student-Test Performance: Top-1 Accuracy and Route Continuity")
        ax.set_xticks(x)
        ax.set_xticklabels(df_models["model_name"], rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim([0.70, 1.0])
        ax.legend()
        fig.savefig(plots_dir / "model_top1_and_continuity.png", dpi=300)
        plt.close(fig)
        print("Saved: model_top1_and_continuity.png")

    # 5. Tuned vs Untuned XGBoost
    tuned_csv = exp_dir / "tuned_vs_untuned.csv"
    if tuned_csv.exists():
        df_tuned = pd.read_csv(tuned_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_list = ["cv_macro_f1", "student_test_macro_f1", "student_test_top1_accuracy", "student_test_route_continuity"]
        x = np.arange(len(metrics_list))
        width = 0.35

        untuned_vals = [df_tuned.loc[0, m] for m in metrics_list]
        tuned_vals = [df_tuned.loc[1, m] for m in metrics_list]

        ax.bar(x - width/2, untuned_vals, width, label="Untuned XGBoost", color="#999999")
        ax.bar(x + width/2, tuned_vals, width, label="Bayesian-Tuned XGBoost", color="#2ca02c")

        ax.set_title("Untuned vs Bayesian-Tuned XGBoost Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(["CV Macro F1", "Test Macro F1", "Test Top-1 Acc", "Route Continuity"], rotation=10)
        ax.set_ylabel("Score")
        ax.set_ylim([0.75, 1.0])
        ax.legend()
        fig.savefig(plots_dir / "tuned_vs_untuned.png", dpi=300)
        plt.close(fig)
        print("Saved: tuned_vs_untuned.png")

    # 6. Optuna Trial History
    opt_csv = exp_dir / "optuna_trials.csv"
    if opt_csv.exists():
        df_opt = pd.read_csv(opt_csv)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(df_opt["number"], df_opt["value"], color="#1f77b4", s=50, alpha=0.7, label="Trial Objective")
        # Cumulative maximum
        cum_max = df_opt["value"].cummax()
        ax.plot(df_opt["number"], cum_max, color="#d62728", linewidth=2.5, label="Best Objective So Far")

        ax.set_title("Optuna Bayesian Optimization History (25 Trials, GroupKFold Macro F1)")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("GroupKFold Macro F1")
        ax.legend()
        fig.savefig(plots_dir / "optuna_optimization_history.png", dpi=300)
        plt.close(fig)
        print("Saved: optuna_optimization_history.png")

    # 7. HMM vs Classifier-only by trip
    hmm_csv = exp_dir / "hmm_vs_classifier.csv"
    if hmm_csv.exists():
        df_hmm = pd.read_csv(hmm_csv)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(df_hmm))
        width = 0.38

        # Accuracy
        axes[0].bar(x - width/2, df_hmm["classifier_top1_accuracy"], width, label="Classifier-Only", color="#ff7f0e")
        axes[0].bar(x + width/2, df_hmm["hmm_top1_accuracy"], width, label="Classifier + HMM", color="#2ca02c")
        axes[0].set_title("Top-1 Accuracy by Student-Test Trip (Classifier vs HMM)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_hmm["trip_id"], rotation=45)
        axes[0].set_ylabel("Top-1 Accuracy")
        axes[0].set_ylim([0.5, 1.05])
        axes[0].legend()

        # Continuity
        axes[1].bar(x - width/2, df_hmm["classifier_route_continuity"], width, label="Classifier-Only", color="#ff7f0e")
        axes[1].bar(x + width/2, df_hmm["hmm_route_continuity"], width, label="Classifier + HMM", color="#2ca02c")
        axes[1].set_title("Route Continuity by Student-Test Trip (Classifier vs HMM)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_hmm["trip_id"], rotation=45)
        axes[1].set_ylabel("Route Continuity")
        axes[1].set_ylim([0.8, 1.02])
        axes[1].legend()

        fig.savefig(plots_dir / "hmm_vs_classifier.png", dpi=300)
        plt.close(fig)
        print("Saved: hmm_vs_classifier.png")

        # 8. Disconnected Transitions Reduction
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, df_hmm["classifier_disconnected_steps"], width, label="Classifier Disconnected Transitions", color="#d62728")
        ax.bar(x + width/2, df_hmm["hmm_disconnected_steps"], width, label="HMM Disconnected Transitions", color="#1f77b4")
        ax.set_title("Disconnected Transitions: 76.8% Total Reduction with HMM Decoding")
        ax.set_xticks(x)
        ax.set_xticklabels(df_hmm["trip_id"], rotation=45)
        ax.set_ylabel("Count of Disconnected Steps")
        ax.legend()
        fig.savefig(plots_dir / "disconnected_transitions.png", dpi=300)
        plt.close(fig)
        print("Saved: disconnected_transitions.png")

    # 9. Confusion Matrix for Best Model (XGBoost)
    pred_xgb_csv = exp_dir / "predictions_xgboost.csv"
    if pred_xgb_csv.exists():
        df_preds = pd.read_csv(pred_xgb_csv)
        y_true = df_preds["label"].to_numpy()
        y_pred = (df_preds["pred_prob"] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title("Confusion Matrix: XGBoost on Student-Test Candidates")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Supervised Label")
        ax.set_xticklabels(["0 (Negative)", "1 (Aligned)"])
        ax.set_yticklabels(["0 (Negative)", "1 (Aligned)"])
        fig.savefig(plots_dir / "confusion_matrix_xgboost.png", dpi=300)
        plt.close(fig)
        print("Saved: confusion_matrix_xgboost.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostic plots from experimental results.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    generate_all_plots(args.reports_dir, args.artifacts_dir)


if __name__ == "__main__":
    main()
