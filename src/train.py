"""Train, evaluate and save the ThreatLens model.

Usage (from the repository root, with the NSL-KDD files in data/raw/):

    python -m src.train

The app itself never trains: it loads ``model/threatlens_model.joblib``. Re-run this
script only if you change the model or the scikit-learn version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_loader import FEATURE_COLUMNS, LABEL_COLUMN, is_attack, validate_frame
from src.metrics import benchmark_metrics, prf
from src.model import (DEFAULT_MODEL_PATH, build_pipeline, percentile_scores, raw_anomaly_scores,
                       save_bundle, train_bundle)

ROOT = Path(__file__).resolve().parent.parent
RAW_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN, "difficulty"]
CALIBRATED_THRESHOLD = 95.0   # the app's default slider value


def read_nslkdd(path: Path) -> pd.DataFrame:
    """Read an original (headerless) NSL-KDD file through the same validation the app uses."""
    return validate_frame(pd.read_csv(path, header=None, names=RAW_COLUMNS))


def evaluate(bundle, test: pd.DataFrame) -> dict:
    """Score the held-out KDDTest+ file at the calibrated threshold, and at an oracle one."""
    y = is_attack(test[LABEL_COLUMN])
    raw = raw_anomaly_scores(bundle.pipeline, test)
    scores = percentile_scores(raw, bundle.calibration_scores)
    at_default = benchmark_metrics(scores, y, CALIBRATED_THRESHOLD)

    # "Oracle" F1: the best threshold *chosen using the test labels*. Nobody can do this
    # in practice, so it is reported only to compare with older numbers that did.
    best = max((benchmark_metrics(scores, y, t) | {"threshold": float(t)} for t in np.arange(50, 100.01, 0.5)),
               key=lambda m: m["f1"])
    return {"auc": float(roc_auc_score(y, raw)), "at_threshold": CALIBRATED_THRESHOLD,
            "calibrated": at_default, "oracle_best_f1": best,
            "test_rows": int(len(test)), "test_attack_share": float(y.mean())}


def ablation(normal: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> list[dict]:
    """Does each preprocessing choice earn its place? Same data, same seed, one change at a time."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(normal))
    n_cal = int(0.2 * len(normal))
    cal, fit = normal.iloc[order[:n_cal]], normal.iloc[order[n_cal:]]
    y = is_attack(test[LABEL_COLUMN])
    rows = []
    for log, onehot in [(False, False), (True, False), (False, True), (True, True)]:
        pipe = build_pipeline(log_transform=log, one_hot=onehot, random_state=seed).fit(fit[FEATURE_COLUMNS])
        cal_scores = np.sort(raw_anomaly_scores(pipe, cal))
        raw = raw_anomaly_scores(pipe, test)
        m = benchmark_metrics(percentile_scores(raw, cal_scores), y, CALIBRATED_THRESHOLD)
        rows.append({"log_transform": log, "one_hot_categoricals": onehot,
                     "auc": float(roc_auc_score(y, raw)), "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "false_alarm_rate": m["false_alarm_rate"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/KDDTrain+.txt")
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/KDDTest+.txt")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    train, test = read_nslkdd(args.train), read_nslkdd(args.test)
    normal = train[~is_attack(train[LABEL_COLUMN])]
    print(f"KDDTrain+: {len(train):,} rows, {len(normal):,} normal (used for training). KDDTest+: {len(test):,} rows.")

    bundle = train_bundle(normal)
    results = evaluate(bundle, test)
    if not args.skip_ablation:
        results["ablation"] = ablation(normal, test)
    bundle.metadata["evaluation"] = {k: v for k, v in results.items() if k != "ablation"}

    save_bundle(bundle, args.out)
    metrics_path = args.out.with_name("metrics.json")
    metrics_path.write_text(json.dumps(results, indent=2))

    c, o = results["calibrated"], results["oracle_best_f1"]
    print(f"\nKDDTest+ AUC: {results['auc']:.3f}")
    print(f"At the app default (score >= {CALIBRATED_THRESHOLD:.0f}): precision {c['precision']:.3f}, "
          f"recall {c['recall']:.3f}, F1 {c['f1']:.3f}, false-alarm rate {c['false_alarm_rate']:.3f}")
    print(f"Oracle (threshold tuned on test labels, not achievable in practice): "
          f"F1 {o['f1']:.3f} at score >= {o['threshold']:.1f}")
    for r in results.get("ablation", []):
        print(f"  log={r['log_transform']!s:5} one-hot={r['one_hot_categoricals']!s:5} "
              f"AUC {r['auc']:.3f}  F1 {r['f1']:.3f}  FAR {r['false_alarm_rate']:.3f}")
    print(f"\nSaved model to {args.out} and metrics to {metrics_path}")


if __name__ == "__main__":
    main()
