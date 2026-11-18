from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List
import folium
from folium import plugins
import joblib
import numpy as np

from src.data.loader import load_raw_trip, TripData
from src.features.candidates import SpatialCandidateGenerator
from src.features.alignment import align_trip_candidates
from src.features.feature_engineering import extract_trip_features, FEATURE_COLUMNS
from src.hmm.viterbi import ViterbiDecoder


def build_interactive_folium_map(
    trip: TripData,
    model_path: str | Path = "artifacts/model.joblib",
    output_html: str | Path = "reports/maps/route_map.html",
    k: int = 12,
) -> folium.Map:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run pipeline
    model = joblib.load(model_path)
    gen = SpatialCandidateGenerator(trip.nodes_xy, trip.arcs)
    cands = gen.generate_candidates(trip.track_xy, k=k)
    cands.trip_id = trip.trip_id

    align = None
    if trip.route is not None:
        align = align_trip_candidates(trip.route, cands, miss_penalty=150.0)

    df_feat = extract_trip_features(trip, cands, align)
    probs = model.predict_proba(df_feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]
    df_feat["pred_prob"] = probs

    cand_probs_list = [
        df_feat.loc[df_feat["observation_idx"] == i, "pred_prob"].to_numpy()
        for i in range(len(trip.track))
    ]

    decoder = ViterbiDecoder(trip.nodes_xy, trip.arcs, beta=15.0, disconnected_penalty=-45.0)
    v_res = decoder.decode(
        trip.trip_id,
        trip.track_xy,
        trip.track[:, 2],
        cands,
        cand_probs_list,
    )

    # Centroid [lat, lon]
    mean_lat = float(np.mean(trip.track[:, 1]))
    mean_lon = float(np.mean(trip.track[:, 0]))

    fmap = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Layer 1: Road network subset (surrounding bounding box)
    fg_roads = folium.FeatureGroup(name="Road Network (Nearby Segments)", show=False)
    # Collect unique candidate arcs to draw local road network
    all_candidate_arcs = set()
    for cand_arr in cands.candidate_arc_ids:
        all_candidate_arcs.update(cand_arr.tolist())

    for arc_id in all_candidate_arcs:
        u = trip.arcs[arc_id, 0]
        v = trip.arcs[arc_id, 1]
        line = [
            [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])],
            [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])],
        ]
        folium.PolyLine(line, color="#aaaaaa", weight=1.5, opacity=0.6).add_to(fg_roads)
    fg_roads.add_to(fmap)

    # Layer 2: Raw GPS Track Points
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    for i, (lon, lat, ts) in enumerate(trip.track):
        popup_text = f"Obs #{i}<br>Timestamp: {ts:.1f}s<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=3.5,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            popup=popup_text,
        ).add_to(fg_gps)
    fg_gps.add_to(fmap)

    # Layer 3: GPS Gaps (> 5s)
    fg_gaps = folium.FeatureGroup(name="Large GPS Gaps (>5s)", show=True)
    dts = np.diff(trip.track[:, 2])
    gap_indices = np.where(dts > 5.0)[0]
    for idx in gap_indices:
        p1 = [float(trip.track[idx, 1]), float(trip.track[idx, 0])]
        p2 = [float(trip.track[idx + 1, 1]), float(trip.track[idx + 1, 0])]
        folium.PolyLine(
            [p1, p2],
            color="#d62728",
            weight=4,
            dash_array="5, 10",
            popup=f"GPS Gap: {dts[idx]:.1f}s gap between #{idx} and #{idx+1}",
        ).add_to(fg_gaps)
    fg_gaps.add_to(fmap)

    # Layer 4: Classifier-Only Route
    fg_clf = folium.FeatureGroup(name="Classifier-Only Route", show=False)
    clf_lines = []
    for arc_id in v_res.classifier_arcs:
        if 0 <= arc_id < len(trip.arcs):
            u = trip.arcs[arc_id, 0]
            v = trip.arcs[arc_id, 1]
            clf_lines.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
            clf_lines.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
    if clf_lines:
        folium.PolyLine(
            clf_lines,
            color="#ff7f0e",
            weight=3,
            opacity=0.7,
            dash_array="6, 6",
            popup="Classifier-Only Top-1 (Without sequential decoding)",
        ).add_to(fg_clf)
    fg_clf.add_to(fmap)

    # Layer 5: Viterbi HMM Reconstructed Route
    fg_hmm = folium.FeatureGroup(name="Viterbi HMM Reconstructed Route", show=True)
    hmm_lines = []
    for arc_id in v_res.predicted_arcs:
        if 0 <= arc_id < len(trip.arcs):
            u = trip.arcs[arc_id, 0]
            v = trip.arcs[arc_id, 1]
            hmm_lines.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
            hmm_lines.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
    if hmm_lines:
        folium.PolyLine(
            hmm_lines,
            color="#2ca02c",
            weight=4.5,
            opacity=0.9,
            popup=f"Viterbi HMM Route (Continuity: {v_res.route_continuity_hmm:.2%})",
        ).add_to(fg_hmm)
    fg_hmm.add_to(fmap)

    # Layer 6: Ground Truth Development Route (if available)
    if trip.route is not None:
        fg_gt = folium.FeatureGroup(name="Reference Ground Truth Route", show=True)
        gt_lines = []
        for arc_id in trip.route:
            if 0 <= arc_id < len(trip.arcs):
                u = trip.arcs[arc_id, 0]
                v = trip.arcs[arc_id, 1]
                gt_lines.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
                gt_lines.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
        if gt_lines:
            folium.PolyLine(
                gt_lines,
                color="#9467bd",
                weight=2.5,
                opacity=0.7,
                popup=f"Reference Route ({len(trip.route)} arcs)",
            ).add_to(fg_gt)
        fg_gt.add_to(fmap)

    # Add interactive plugins
    folium.LayerControl(collapsed=False).add_to(fmap)
    plugins.Fullscreen(position="topright").add_to(fmap)
    plugins.MeasureControl(position="topleft").add_to(fmap)

    fmap.save(str(output_html))
    print(f"Interactive Folium map saved to {output_html}")
    return fmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive Folium maps for trips.")
    parser.add_argument("--trip-id", type=str, default="00000052")
    parser.add_argument("--partition", type=str, default="student_test")
    parser.add_argument("--output", type=Path, default=Path("reports/maps/route_map_00000052.html"))
    args = parser.parse_args()

    trip_dir = Path("data") / args.partition / args.trip_id
    trip = load_raw_trip(trip_dir, partition=args.partition)
    build_interactive_folium_map(trip, output_html=args.output)


if __name__ == "__main__":
    main()
