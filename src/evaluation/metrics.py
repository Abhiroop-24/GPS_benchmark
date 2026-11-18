from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    log_loss,
    confusion_matrix,
)


def calculate_observation_top1_accuracy(
    df_eval: pd.DataFrame,
    pred_col: str = "pred_prob",
    label_col: str = "label",
) -> float:
    """
    For each observation (grouped by trip_id, observation_idx),
    select the candidate with the highest predicted score.
    Returns the proportion where the top candidate has label == 1.
    """
    # Group by trip and observation
    # Sort by predicted probability descending within each group
    idx_max = df_eval.groupby(["trip_id", "observation_idx"])[pred_col].idxmax()
    top_candidates = df_eval.loc[idx_max]
    top1_acc = float((top_candidates[label_col] == 1).mean())
    return top1_acc


def calculate_route_continuity(
    predicted_arcs: np.ndarray,
    arcs: np.ndarray,
) -> float:
    """
    Measures the proportion of consecutive predicted arcs that are topologically
    connected in the directed road graph (either identical or arc[i].to_node == arc[i+1].from_node).
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    consecutive_pairs = len(predicted_arcs) - 1
    connected_count = 0

    for i in range(consecutive_pairs):
        a1 = int(predicted_arcs[i])
        a2 = int(predicted_arcs[i + 1])
        if a1 == -1 or a2 == -1:
            continue
        if a1 == a2:
            connected_count += 1
        elif arcs[a1, 1] == arcs[a2, 0]:
            connected_count += 1

    return float(connected_count / max(consecutive_pairs, 1))


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_eval: pd.DataFrame,
    pred_col: str = "pred_prob",
) -> Dict[str, float]:
    """Compute all standard evaluation metrics for candidate classification."""
    y_pred = (y_prob >= 0.5).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    try:
        eps = 1e-15
        clipped_prob = np.clip(y_prob, eps, 1.0 - eps)
        ll = float(log_loss(y_true, clipped_prob))
    except Exception:
        ll = float("nan")

    df_eval = df_eval.copy()
    df_eval[pred_col] = y_prob
    top1_acc = calculate_observation_top1_accuracy(df_eval, pred_col=pred_col)

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "log_loss": ll,
        "top1_accuracy": top1_acc,
    }
