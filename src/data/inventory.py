from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from src.data.loader import load_raw_trip


def build_dataset_inventory(
    data_dir: str | Path,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    records: List[Dict[str, Any]] = []

    partitions = [
        ("train", data_dir / "train"),
        ("student_test", data_dir / "student_test"),
    ]

    for partition_name, partition_dir in partitions:
        if not partition_dir.exists():
            continue

        trip_dirs = sorted([d for d in partition_dir.iterdir() if d.is_dir()])
        print(f"Scanning partition {partition_name}: {len(trip_dirs)} trips found.")

        for trip_dir in tqdm(trip_dirs, desc=f"Validating {partition_name}"):
            trip_data = load_raw_trip(trip_dir, partition=partition_name)
            v = trip_data.validation_report

            record = {
                "partition": partition_name,
                "trip_id": trip_data.trip_id,
                "epsg": trip_data.epsg,
                "observations": v["num_observations"],
                "nodes": v["num_nodes"],
                "arcs": v["num_arcs"],
                "route_length": v.get("route_length", 0),
                "route_unique_arcs": v.get("route_unique_arcs", 0),
                "route_repeated_arcs": v.get("route_repeated_arcs", 0),
                "disconnected_route_steps": v.get("disconnected_route_steps", 0),
                "timestamps_ordered": v["timestamps_ordered"],
                "duplicate_timestamps": v["duplicate_timestamps"],
                "irregular_timestamps": v["irregular_timestamps"],
                "large_gaps_gt_5s": v["large_gaps_gt_5s"],
                "max_gap_seconds": v["max_gap_seconds"],
                "zero_length_arcs": v["zero_length_arcs"],
                "isolated_nodes": v["isolated_nodes"],
                "source_nodes": v["source_nodes"],
                "sink_nodes": v["sink_nodes"],
                "weak_components": v["weak_components"],
                "issues": "; ".join(v["issues"]) if v["issues"] else "None",
            }
            records.append(record)

    df = pd.DataFrame(records)
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Inventory saved to {output_csv}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset inventory and validate all trips.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/trip_inventory.csv"))
    args = parser.parse_args()

    df = build_dataset_inventory(args.data_dir, args.output)
    print("\nDataset Summary by Partition:")
    summary = df.groupby("partition").agg(
        trips=("trip_id", "count"),
        total_obs=("observations", "sum"),
        mean_obs=("observations", "mean"),
        min_obs=("observations", "min"),
        max_obs=("observations", "max"),
        mean_arcs=("arcs", "mean"),
        trips_with_issues=("issues", lambda s: (s != "None").sum()),
    )
    print(summary)


if __name__ == "__main__":
    main()
