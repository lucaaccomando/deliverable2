import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data_loader import is_attack
from src.model import (explain_records, load_bundle, percentile_scores, save_bundle, score_records,
                       summarize_reasons, train_bundle)


def test_percentile_scale_is_the_share_of_normal_traffic_below():
    calibration = np.arange(1, 101, dtype=float)             # 100 normal scores: 1..100
    assert percentile_scores(np.array([0.5]), calibration)[0] == 0.0     # more normal than all of them
    assert percentile_scores(np.array([50.0]), calibration)[0] == 50.0   # above 50% of normal traffic
    assert percentile_scores(np.array([1e9]), calibration)[0] == 100.0   # above all of it


def test_scores_are_bounded_and_one_per_record(bundle, sample):
    s = score_records(bundle, sample)
    assert len(s) == len(sample) and s["score"].between(0, 100).all()


def test_model_separates_attacks_from_normal_on_unseen_data(bundle, sample):
    s = score_records(bundle, sample)
    assert roc_auc_score(is_attack(sample["label"]), s["raw_score"]) > 0.9


def test_default_threshold_has_low_false_alarm_rate(bundle, sample):
    s = score_records(bundle, sample)["score"]
    normal = ~is_attack(sample["label"])
    assert (s[normal] >= 95).mean() < 0.06


def test_unseen_category_value_does_not_crash(bundle, sample):
    row = sample.head(3).copy()
    row["service"] = "brand_new_service"
    assert score_records(bundle, row)["score"].between(0, 100).all()


def test_explanation_finds_the_extreme_feature(bundle, sample):
    row = sample[sample["label"] == "normal"].head(1).copy()
    row["src_bytes"] = 5e8
    reasons = explain_records(bundle, row)[0]
    assert reasons[0]["feature"] == "src_bytes" and reasons[0]["z"] > 3


def test_explanation_flags_rare_category(bundle, sample):
    row = sample.head(1).copy()
    row["service"] = "brand_new_service"
    reasons = explain_records(bundle, row)[0]
    rare = [r for r in reasons if r["kind"] == "category"]
    assert rare and rare[0]["feature"] == "service" and rare[0]["typical"] == 0.0
    assert "service=brand_new_service" in summarize_reasons(reasons, limit=10)


def test_train_save_load_roundtrip(sample, tmp_path):
    normal = sample[sample["label"] == "normal"]
    small = train_bundle(normal, n_estimators=10)
    save_bundle(small, tmp_path / "m.joblib")
    loaded = load_bundle(tmp_path / "m.joblib")
    assert np.allclose(score_records(small, normal.head(20))["score"], score_records(loaded, normal.head(20))["score"])


def test_load_refuses_mismatched_sklearn_version(sample, tmp_path):
    small = train_bundle(sample[sample["label"] == "normal"], n_estimators=5)
    small.metadata["sklearn_version"] = "0.0.1"
    save_bundle(small, tmp_path / "old.joblib")
    with pytest.raises(RuntimeError, match="scikit-learn 0.0.1"):
        load_bundle(tmp_path / "old.joblib")
