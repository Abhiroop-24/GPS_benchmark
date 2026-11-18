from __future__ import annotations

import heapq
from typing import Dict, List, Tuple, Optional
import numpy as np


class RoadGraph:
    """Directed road network graph supporting bounded Dijkstra shortest paths."""

    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        self.nodes_xy = np.asarray(nodes_xy, dtype=np.float64)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.num_nodes = len(self.nodes_xy)
        self.num_arcs = len(self.arcs)

        # Compute arc lengths
        a = self.nodes_xy[self.arcs[:, 0]]
        b = self.nodes_xy[self.arcs[:, 1]]
        self.arc_lengths = np.linalg.norm(b - a, axis=1)

        # Build adjacency list: from_node -> list of (to_node, length, arc_id)
        self.adj: Dict[int, List[Tuple[int, float, int]]] = {i: [] for i in range(self.num_nodes)}
        for arc_id in range(self.num_arcs):
            u = int(self.arcs[arc_id, 0])
            v = int(self.arcs[arc_id, 1])
            w = float(self.arc_lengths[arc_id])
            self.adj[u].append((v, w, arc_id))

    def bounded_dijkstra(
        self,
        start_node: int,
        target_nodes: set[int],
        max_dist: float = 400.0,
    ) -> Dict[int, float]:
        """
        Runs bounded Dijkstra from start_node to find shortest distances
        to any node in target_nodes within max_dist.
        """
        distances: Dict[int, float] = {start_node: 0.0}
        pq = [(0.0, start_node)]
        found: Dict[int, float] = {}

        if start_node in target_nodes:
            found[start_node] = 0.0
            if len(found) == len(target_nodes):
                return found

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances.get(u, float("inf")):
                continue
            if d > max_dist:
                break

            for v, weight, _ in self.adj.get(u, []):
                new_d = d + weight
                if new_d <= max_dist and new_d < distances.get(v, float("inf")):
                    distances[v] = new_d
                    heapq.heappush(pq, (new_d, v))
                    if v in target_nodes:
                        found[v] = new_d
                        if len(found) == len(target_nodes):
                            return found

        return found


class TransitionModel:
    """Computes log transition probabilities between candidate arcs using road network topology."""

    def __init__(
        self,
        road_graph: RoadGraph,
        beta: float = 15.0,
        disconnected_penalty: float = -45.0,
    ):
        self.graph = road_graph
        self.beta = float(beta)
        self.disconnected_penalty = float(disconnected_penalty)

    def compute_transition_matrix(
        self,
        prev_arc_ids: np.ndarray,
        curr_arc_ids: np.ndarray,
        prev_pt: np.ndarray,
        curr_pt: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Computes the log transition matrix between prev_arc_ids (length K_prev)
        and curr_arc_ids (length K_curr).
        """
        K_prev = len(prev_arc_ids)
        K_curr = len(curr_arc_ids)
        log_trans = np.full((K_prev, K_curr), self.disconnected_penalty, dtype=np.float64)

        if K_prev == 0 or K_curr == 0:
            return log_trans

        gps_dist = float(np.linalg.norm(curr_pt - prev_pt))
        max_search = max(gps_dist * 2.5 + 50.0, 350.0)

        # Set of target entry nodes for current candidates
        target_nodes = set(int(self.graph.arcs[arc_id, 0]) for arc_id in curr_arc_ids)

        # For each prev candidate, run bounded Dijkstra from its exit node
        for j, prev_arc in enumerate(prev_arc_ids):
            prev_arc = int(prev_arc)
            exit_node = int(self.graph.arcs[prev_arc, 1])

            # Pre-query reachable targets from exit_node
            found_distances = self.graph.bounded_dijkstra(exit_node, target_nodes, max_dist=max_search)

            for k, curr_arc in enumerate(curr_arc_ids):
                curr_arc = int(curr_arc)
                curr_len = float(self.graph.arc_lengths[curr_arc])

                if prev_arc == curr_arc:
                    # Same arc transition
                    net_dist = gps_dist  # Consistent with movement along same segment
                    penalty = 0.0
                elif exit_node == self.graph.arcs[curr_arc, 0]:
                    # Direct consecutive connected arc
                    net_dist = curr_len
                    penalty = 0.0
                elif self.graph.arcs[curr_arc, 0] in found_distances:
                    # Connected through path
                    graph_dist = found_distances[self.graph.arcs[curr_arc, 0]]
                    net_dist = graph_dist + curr_len
                    penalty = 0.0
                else:
                    # Disconnected transition
                    log_trans[j, k] = self.disconnected_penalty
                    continue

                diff = abs(net_dist - gps_dist)
                # Timestamp gap scaling
                dt_scale = max(1.0, dt / 2.0)
                log_p = -diff / (self.beta * dt_scale) + penalty
                log_trans[j, k] = log_p

        return log_trans
