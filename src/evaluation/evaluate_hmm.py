from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loader import load_raw_trip, TripData
from src.features.candidates import SpatialCandidateGenerator
from src.features.alignment import align_trip_candidates
from src.features.feature_engineering import extract_trip_features, FEATURE_COLUMNS
from src.hmm.viterbi import ViterbiDecoder
from src.evaluation.metrics import (
    calculate_route_continuity,
    calculate_comprehensive_metrics,
)


def evaluate_hmm_pipeline(
    data_dir: str | Path = "data/student_test",
    model_path: str | Path = "artifacts/model.joblib",
    output_csv: str | Path = "reports/experiments/hmm_vs_classifier.csv",
    k: int = 12,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    trip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Evaluating HMM vs Classifier-only across {len(trip_dirs)} student-test trips...")

    trip_records: List[Dict[str, Any]] = []

    for trip_dir in tqdm(trip_dirs, desc="Evaluating HMM"):
        trip = load_raw_trip(trip_dir, partition="student_test")
        t0 = time.time()

        # 1. Candidate generation
        gen = SpatialCandidateGenerator(trip.nodes_xy, trip.arcs)
        cands = gen.generate_candidates(trip.track_xy, k=k)
        cands.trip_id = trip.trip_id

        # 2. Alignment for ground truth comparison
        align = None
        if trip.route is not None:
            align = align_trip_candidates(trip.route, cands, miss_penalty=150.0)

        # 3. Feature extraction
        df_feat = extract_trip_features(trip, cands, align)
        X_feat = df_feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        # 4. Model probability predictions
        probs = model.predict_proba(X_feat)[:, 1]
        df_feat["pred_prob"] = probs

        # Reshape probabilities per observation
        cand_probs_list: List[np.ndarray] = []
        for i in range(len(trip.track)):
            obs_probs = df_feat.loc[df_feat["observation_idx"] == i, "pred_prob"].to_numpy()
            cand_probs_list.append(obs_probs)

        # 5. Viterbi decoding
        decoder = ViterbiDecoder(trip.nodes_xy, trip.arcs, beta=15.0, disconnected_penalty=-45.0)
        v_res = decoder.decode(
            trip.trip_id,
            trip.track_xy,
            trip.track[:, 2],
            cands,
            cand_probs_list,
        )

        elapsed = time.time() - t0

        # Calculate observation accuracies if alignment is available
        clf_acc = 0.0
        hmm_acc = 0.0
        if align is not None:
            # True arc ID per observation
            true_arcs = align.aligned_arc_ids
            valid_mask = true_arcs >= 0
            if valid_mask.sum() > 0:
                clf_acc = float((v_res.classifier_arcs[valid_mask] == true_arcs[valid_mask]).mean())
                hmm_acc = float((v_res.predicted_arcs[valid_mask] == true_arcs[valid_mask]).mean())

        acc_improvement = hmm_acc - clf_acc
        cont_improvement = v_res.route_continuity_hmm - v_res.route_continuity_classifier

        trip_records.append({
            "trip_id": trip.trip_id,
            "observations": len(trip.track),
            "candidate_miss_rate": align.candidate_miss_rate if align else 0.0,
            "classifier_top1_accuracy": clf_acc,
            "hmm_top1_accuracy": hmm_acc,
            "accuracy_improvement": acc_improvement,
            "classifier_route_continuity": v_res.route_continuity_classifier,
            "hmm_route_continuity": v_res.route_continuity_hmm,
            "continuity_improvement": cont_improvement,
            "classifier_disconnected_steps": v_res.disconnected_steps_classifier,
            "hmm_disconnected_steps": v_res.disconnected_steps_hmm,
            "execution_time_s": elapsed,
        })

    df_out = pd.DataFrame(trip_records)
    df_out.to_csv(output_csv, index=False)

    print("\nHMM vs Classifier-Only Comparison by Trip:")
    print(df_out[["trip_id", "classifier_top1_accuracy", "hmm_top1_accuracy", "accuracy_improvement", "classifier_route_continuity", "hmm_route_continuity", "classifier_disconnected_steps", "hmm_disconnected_steps"]].to_string())

    print("\nOverall Summary:")
    print(f"Mean Classifier Top-1 Accuracy:   {df_out['classifier_top1_accuracy'].mean():.4f}")
    print(f"Mean HMM Top-1 Accuracy:          {df_out['hmm_top1_accuracy'].mean():.4f}")
    print(f"Mean Classifier Route Continuity: {df_out['classifier_route_continuity'].mean():.4f}")
    print(f"Mean HMM Route Continuity:        {df_out['hmm_route_continuity'].mean():.4f}")
    print(f"Total Disconnected Transitions:   {df_out['classifier_disconnected_steps'].sum()} (Classifier) -> {df_out['hmm_disconnected_steps'].sum()} (HMM)")

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HMM vs Classifier-only on student-test.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/student_test"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/experiments/hmm_vs_classifier.csv"))
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()

    evaluate_hmm_pipeline(args.data_dir, args.model_path, args.output_csv, k=args.k)


if __name__ == "__main__":
    main()
