from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm

from src.data.loader import load_raw_trip
from src.features.candidates import SpatialCandidateGenerator
from src.features.alignment import align_trip_candidates
from src.features.feature_engineering import extract_trip_features


def build_and_save_features(
    data_dir: str | Path,
    output_dir: str | Path = "artifacts",
    k: int = 12,
) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions = ["train", "student_test"]

    for partition in partitions:
        part_dir = data_dir / partition
        if not part_dir.exists():
            continue

        trip_dirs = sorted([d for d in part_dir.iterdir() if d.is_dir()])
        print(f"\nBuilding features for partition '{partition}' ({len(trip_dirs)} trips, K={k})...")
        t0 = time.time()

        dfs: List[pd.DataFrame] = []
        for trip_dir in tqdm(trip_dirs, desc=f"Extracting {partition}"):
            trip = load_raw_trip(trip_dir, partition=partition)
            gen = SpatialCandidateGenerator(trip.nodes_xy, trip.arcs)
            cands = gen.generate_candidates(trip.track_xy, k=k)
            cands.trip_id = trip.trip_id

            align = None
            if trip.route is not None:
                align = align_trip_candidates(trip.route, cands, miss_penalty=150.0)

            df_trip = extract_trip_features(trip, cands, align)
            dfs.append(df_trip)

        df_all = pd.concat(dfs, ignore_index=True)
        out_file = output_dir / f"features_{partition}.parquet"
        df_all.to_parquet(out_file, index=False)
        elapsed = time.time() - t0
        print(f"Saved {len(df_all):,} candidate rows to {out_file} in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and persist feature datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()

    build_and_save_features(args.data_dir, args.output_dir, k=args.k)


if __name__ == "__main__":
    main()
