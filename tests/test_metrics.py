import numpy as np
import pandas as pd
import pytest

from src.metrics import (CONFIRMED, FALSE_POSITIVE, benchmark_metrics, labeled_precision, prf,
                         suggest_threshold, threshold_sweep)


def test_prf_hand_computed():
    m = prf(tp=8, fp=2, fn=8)          # precision 0.8, recall 0.5
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_prf_no_division_by_zero():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_benchmark_metrics_confusion_counts():
    scores = np.array([99, 90, 80, 70, 10])
    threat = np.array([True, False, True, False, False])
    m = benchmark_metrics(scores, threat, threshold=75)       # flags 99, 90, 80
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 2)
    assert m["false_alarm_rate"] == pytest.approx(1 / 3)


def test_threshold_sweep_recall_never_increases_with_threshold():
    rng = np.random.RandomState(0)
    scores, threat = rng.uniform(0, 100, 500), rng.rand(500) < 0.3
    recall = threshold_sweep(scores, threat)["recall"].to_numpy()
    assert (np.diff(recall) <= 1e-12).all()


def test_labeled_precision():
    out = labeled_precision(pd.Series([CONFIRMED, CONFIRMED, FALSE_POSITIVE, None], dtype=object))
    assert out == {"confirmed": 2, "false_positive": 1, "reviewed": 3, "precision": pytest.approx(2 / 3)}
    assert labeled_precision(pd.Series([], dtype=object))["precision"] is None


def test_suggest_threshold_separates_clusters():
    # false positives sit at scores 60-70, confirmed threats at 80-95: best cut is 80.
    scores = np.array([60, 65, 70, 80, 85, 90, 95])
    labels = np.array([FALSE_POSITIVE] * 3 + [CONFIRMED] * 4)
    best = suggest_threshold(scores, labels)
    assert best["threshold"] == 80 and best["precision"] == 1.0 and best["recall"] == 1.0


@pytest.mark.parametrize("labels", [
    [CONFIRMED] * 6,                                  # no false positives to learn from
    [FALSE_POSITIVE] * 6,                             # no confirmed threats
    [CONFIRMED, FALSE_POSITIVE, CONFIRMED],           # fewer than 5 labels
])
def test_suggest_threshold_needs_enough_evidence(labels):
    assert suggest_threshold(np.linspace(50, 99, len(labels)), np.array(labels)) is None
