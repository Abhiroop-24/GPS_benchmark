from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loader import load_raw_trip, TripData
from src.features.candidates import SpatialCandidateGenerator
from src.features.alignment import align_trip_candidates


def run_candidate_benchmark(
    data_dir: str | Path,
    k_values: List[int] = [8, 12, 16, 20],
    output_dir: str | Path = "reports/experiments",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions = [
        ("train", sorted([d for d in (data_dir / "train").iterdir() if d.is_dir()])),
        ("student_test", sorted([d for d in (data_dir / "student_test").iterdir() if d.is_dir()])),
    ]

    trip_level_records: List[Dict[str, Any]] = []

    for k in k_values:
        print(f"\n--- Benchmarking K = {k} ---")
        t0_k = time.time()

        for partition_name, trip_dirs in partitions:
            for trip_dir in tqdm(trip_dirs, desc=f"K={k} [{partition_name}]"):
                t_start = time.time()
                trip = load_raw_trip(trip_dir, partition=partition_name)
                gen = SpatialCandidateGenerator(trip.nodes_xy, trip.arcs)
                cands = gen.generate_candidates(trip.track_xy, k=k)
                cands.trip_id = trip.trip_id

                if trip.route is not None:
                    align = align_trip_candidates(trip.route, cands, miss_penalty=150.0)
                    topk_recall = float(align.topk_recall)
                    miss_rate = float(align.candidate_miss_rate)
                else:
                    topk_recall = np.nan
                    miss_rate = np.nan

                elapsed = time.time() - t_start

                trip_level_records.append({
                    "k": k,
                    "partition": partition_name,
                    "trip_id": trip.trip_id,
                    "observations": len(trip.track),
                    "mean_candidates": float(np.mean(cands.candidate_counts)),
                    "median_candidates": float(np.median(cands.candidate_counts)),
                    "topk_recall": topk_recall,
                    "candidate_miss_rate": miss_rate,
                    "runtime_seconds": float(elapsed),
                })

    df_trips = pd.DataFrame(trip_level_records)
    df_trips.to_csv(output_dir / "candidate_benchmarks_all_trips.csv", index=False)

    # Summary by K and partition
    summary = df_trips.groupby(["partition", "k"]).agg(
        trips=("trip_id", "count"),
        total_obs=("observations", "sum"),
        mean_recall=("topk_recall", "mean"),
        weighted_recall=("topk_recall", lambda s: np.average(s, weights=df_trips.loc[s.index, "observations"])),
        mean_miss_rate=("candidate_miss_rate", "mean"),
        weighted_miss_rate=("candidate_miss_rate", lambda s: np.average(s, weights=df_trips.loc[s.index, "observations"])),
        total_runtime_s=("runtime_seconds", "sum"),
        mean_time_per_trip_s=("runtime_seconds", "mean"),
    ).reset_index()

    summary.to_csv(output_dir / "candidate_summary_by_k.csv", index=False)
    print("\nCandidate Benchmark Summary:")
    print(summary.to_string())

    return df_trips, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate generation across K values.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--k-values", nargs="+", type=int, default=[8, 12, 16, 20])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"))
    args = parser.parse_args()

    run_candidate_benchmark(args.data_dir, args.k_values, args.output_dir)


if __name__ == "__main__":
    main()
