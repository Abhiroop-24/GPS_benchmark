from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np

from src.data.loader import load_raw_trip, TripData
from src.features.candidates import SpatialCandidateGenerator
from src.features.alignment import align_trip_candidates
from src.features.feature_engineering import extract_trip_features, FEATURE_COLUMNS
from src.hmm.viterbi import ViterbiDecoder
from src.api.schemas import PredictResponse, MetricsResponse


class InferenceEngine:
    """Server-side prediction and evaluation engine for Laptop 1."""

    def __init__(
        self,
        model_path: str | Path = "artifacts/model.joblib",
        data_base_dir: str | Path = "data",
    ):
        self.model_path = Path(model_path)
        self.data_base_dir = Path(data_base_dir)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict_trip(
        self,
        trip_id: str,
        partition: str = "student_test",
        k: int = 12,
    ) -> PredictResponse:
        t0 = time.time()
        trip_dir = self.data_base_dir / partition / trip_id
        if not trip_dir.exists():
            raise FileNotFoundError(f"Trip {trip_id} does not exist in partition '{partition}'")

        # 1. Load server-side track and road network
        trip = load_raw_trip(trip_dir, partition=partition)

        # 2. Candidate generation
        gen = SpatialCandidateGenerator(trip.nodes_xy, trip.arcs)
        cands = gen.generate_candidates(trip.track_xy, k=k)
        cands.trip_id = trip.trip_id

        # 3. Server-side alignment ONLY for permitted development validation (not leaked into features!)
        align = None
        if trip.route is not None:
            align = align_trip_candidates(trip.route, cands, miss_penalty=150.0)

        # 4. Feature extraction (NO route information passed into model inputs)
        df_feat = extract_trip_features(trip, cands, alignment=None)
        X = df_feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        # 5. Classifier inference
        probs = self.model.predict_proba(X)[:, 1]
        df_feat["pred_prob"] = probs

        cand_probs_list = []
        for i in range(len(trip.track)):
            obs_p = df_feat.loc[df_feat["observation_idx"] == i, "pred_prob"].to_numpy()
            cand_probs_list.append(obs_p)

        # 6. Viterbi decoding
        decoder = ViterbiDecoder(trip.nodes_xy, trip.arcs, beta=15.0, disconnected_penalty=-45.0)
        v_res = decoder.decode(
            trip.trip_id,
            trip.track_xy,
            trip.track[:, 2],
            cands,
            cand_probs_list,
        )

        elapsed = time.time() - t0

        # Construct geometry: GPS coords as [lat, lon]
        gps_coords = [[float(lat), float(lon)] for lon, lat, _ in trip.track]

        # Route nodes: [from_node, to_node] for each predicted arc
        route_nodes = []
        route_coords = []
        for arc_id in v_res.predicted_arcs:
            arc_id = int(arc_id)
            if 0 <= arc_id < len(trip.arcs):
                u = int(trip.arcs[arc_id, 0])
                v = int(trip.arcs[arc_id, 1])
                route_nodes.append([u, v])
                # Append coordinates of segment endpoints
                u_lat, u_lon = float(trip.nodes[u, 1]), float(trip.nodes[u, 0])
                v_lat, v_lon = float(trip.nodes[v, 1]), float(trip.nodes[v, 0])
                route_coords.append([u_lat, u_lon])
                route_coords.append([v_lat, v_lon])
            else:
                route_nodes.append([-1, -1])

        # Server-side development metrics calculation (if ground truth route was supplied)
        dev_metrics = None
        if align is not None:
            true_arcs = align.aligned_arc_ids
            valid = true_arcs >= 0
            if valid.sum() > 0:
                dev_metrics = {
                    "classifier_top1_accuracy": float((v_res.classifier_arcs[valid] == true_arcs[valid]).mean()),
                    "hmm_top1_accuracy": float((v_res.predicted_arcs[valid] == true_arcs[valid]).mean()),
                    "accuracy_improvement": float(
                        (v_res.predicted_arcs[valid] == true_arcs[valid]).mean()
                        - (v_res.classifier_arcs[valid] == true_arcs[valid]).mean()
                    ),
                    "candidate_miss_rate": float(align.candidate_miss_rate),
                }

        mean_confidence = float(np.mean([np.max(p) if len(p) > 0 else 0.0 for p in cand_probs_list]))

        return PredictResponse(
            trip_id=trip.trip_id,
            partition=partition,
            status="success",
            observation_count=len(trip.track),
            candidate_count=int(np.sum(cands.candidate_counts)),
            predicted_arc_ids=v_res.predicted_arcs.tolist(),
            classifier_only_route=v_res.classifier_arcs.tolist(),
            route_nodes=route_nodes,
            gps_coordinates=gps_coords,
            predicted_route_coordinates=route_coords,
            prediction_confidence=mean_confidence,
            route_continuity=v_res.route_continuity_hmm,
            execution_time_seconds=elapsed,
            metrics=dev_metrics,
        )
