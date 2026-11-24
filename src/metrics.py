"""Metrics for ThreatLens, written out by hand so every number can be explained.

Definitions (positive class = "threat"):
    precision = TP / (TP + FP)   of the alerts we raised, how many were real
    recall    = TP / (TP + FN)   of the real threats, how many we alerted on
    F1        = harmonic mean of precision and recall
    false-alarm rate = FP / (FP + TN)   share of normal traffic we wrongly flagged
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFIRMED = "confirmed"
FALSE_POSITIVE = "false_positive"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {"precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall)}


def benchmark_metrics(scores: np.ndarray, is_threat: np.ndarray, threshold: float) -> dict:
    """Quality of the detector at ``threshold`` when ground truth is available in the file."""
    flagged = scores >= threshold
    tp = int((flagged & is_threat).sum())
    fp = int((flagged & ~is_threat).sum())
    fn = int((~flagged & is_threat).sum())
    tn = int((~flagged & ~is_threat).sum())
    out = prf(tp, fp, fn)
    out.update(tp=tp, fp=fp, fn=fn, tn=tn, false_alarm_rate=_safe_div(fp, fp + tn))
    return out


def threshold_sweep(scores: np.ndarray, is_threat: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Precision / recall / F1 at each threshold, for the benchmark trade-off chart."""
    thresholds = np.arange(50, 100.5, 1.0) if thresholds is None else thresholds
    return pd.DataFrame([{"threshold": float(t), **benchmark_metrics(scores, is_threat, t)}
                         for t in thresholds])


def labeled_precision(labels: pd.Series) -> dict:
    """Precision of the alerts the analyst has reviewed so far.

    Only confirmed / false-positive labels count. Recall cannot be computed from
    analyst labels alone, because we never learn about threats that were not flagged.
    """
    confirmed = int((labels == CONFIRMED).sum())
    false_pos = int((labels == FALSE_POSITIVE).sum())
    return {"confirmed": confirmed, "false_positive": false_pos,
            "reviewed": confirmed + false_pos,
            "precision": _safe_div(confirmed, confirmed + false_pos) if confirmed + false_pos else None}


def suggest_threshold(scores: np.ndarray, labels: np.ndarray, min_labels: int = 5) -> dict | None:
    """Suggest the flagging threshold that best fits the analyst's labels.

    ``labels`` holds "confirmed" / "false_positive" for each reviewed alert. We try every
    reviewed score as a candidate threshold and keep the one with the best F1, where
    recall is measured against the threats the analyst confirmed. Limits: it can only
    suggest thresholds at or above the lowest reviewed score, and it needs both kinds
    of label. Returns None when there is not enough evidence.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    reviewed = np.isin(labels, [CONFIRMED, FALSE_POSITIVE])
    scores, labels = scores[reviewed], labels[reviewed]
    n_confirmed = int((labels == CONFIRMED).sum())
    if len(scores) < min_labels or n_confirmed == 0 or n_confirmed == len(scores):
        return None
    best = None
    for t in np.unique(scores):
        above = scores >= t
        tp = int((above & (labels == CONFIRMED)).sum())
        fp = int((above & (labels == FALSE_POSITIVE)).sum())
        fn = n_confirmed - tp
        m = prf(tp, fp, fn)
        # strict ">" keeps the lowest threshold among ties, which favours recall.
        if best is None or m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}
    return best
