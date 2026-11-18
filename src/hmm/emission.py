from __future__ import annotations

from typing import List
import numpy as np


def compute_log_emissions(
    candidate_probabilities: List[np.ndarray],
    eps: float = 1e-6,
) -> List[np.ndarray]:
    """
    Convert model candidate probabilities into numerical log-emission probabilities.
    Normalizes candidate scores for each observation and applies numerical smoothing.
    """
    log_emissions: List[np.ndarray] = []

    for probs in candidate_probabilities:
        p = np.asarray(probs, dtype=np.float64)
        if len(p) == 0:
            log_emissions.append(np.array([0.0]))
            continue

        # Softmax or normalized probabilities across candidates
        total_p = np.sum(p)
        if total_p > 1e-12:
            norm_p = p / total_p
        else:
            norm_p = np.full(len(p), 1.0 / len(p))

        log_p = np.log(np.maximum(norm_p, eps))
        log_emissions.append(log_p)

    return log_emissions
