"""Isolation Forest scoring and explanations for ThreatLens.

Three ideas live in this file:

1. **Pipeline.** Preprocessing (log transform + one-hot encoding) and the Isolation
   Forest are one scikit-learn ``Pipeline`` so training and scoring can never use
   different preprocessing.
2. **A 0-100 score that means something.** The forest's raw score has no natural unit.
   We convert it to a *percentile of normal traffic*: a score of 97 means "this
   connection is more unusual than 97% of held-out normal connections". The slider
   threshold is therefore directly related to the false-alarm rate on normal traffic.
3. **Explanations.** Isolation Forest does not explain individual predictions, so we
   add a simple, transparent approximation: how many standard deviations each feature
   is from the normal-traffic average, and which category values are rare in normal
   traffic. This is documented as an approximation, not what the forest "used".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.data_loader import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "threatlens_model.joblib"

# Z-scores are computed on log1p-transformed values. Many NSL-KDD features are almost
# constant in normal traffic (std near 0), which would make any non-zero value look
# infinitely strange. Flooring the std keeps z-scores finite and comparable.
MIN_STD = 0.25
# A category value seen in fewer than this fraction of normal connections is "rare".
RARE_CATEGORY_FREQ = 0.01


def build_pipeline(n_estimators: int = 200, random_state: int = 42,
                   log_transform: bool = True, one_hot: bool = True) -> Pipeline:
    """Preprocessing + Isolation Forest as a single Pipeline.

    ``log_transform`` / ``one_hot`` exist only so ``train.py`` can compare variants
    (see the ablation table in the README). The shipped model uses both.
    """
    numeric_step = FunctionTransformer(np.log1p, feature_names_out="one-to-one") if log_transform else "passthrough"
    transformers = [("num", numeric_step, NUMERIC_COLUMNS)]
    if one_hot:
        # handle_unknown="ignore": a service never seen in training becomes all zeros
        # instead of crashing the app.
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS))
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        # contamination is irrelevant here because we choose the threshold ourselves
        # from calibration data (see anomaly_scores), so it stays at the default.
        ("forest", IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
    ])


@dataclass
class ModelBundle:
    """Everything the app needs, saved together in one joblib file."""

    pipeline: Pipeline
    calibration_scores: np.ndarray          # sorted raw anomaly scores of held-out NORMAL traffic
    baseline: pd.DataFrame                  # per numeric feature: mean_log, std_log, median
    category_freq: dict                     # column -> {value: fraction of normal traffic}
    metadata: dict = field(default_factory=dict)


def raw_anomaly_scores(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous. (scikit-learn's score_samples is the opposite sign.)"""
    return -pipeline.score_samples(df[FEATURE_COLUMNS])


def percentile_scores(raw: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    """Convert raw scores to 0-100: the % of normal calibration traffic scoring lower."""
    return 100.0 * np.searchsorted(calibration_scores, raw, side="right") / len(calibration_scores)


def make_baseline(normal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Summarise normal traffic: used for explanations, saved with the model."""
    logged = np.log1p(normal[NUMERIC_COLUMNS])
    baseline = pd.DataFrame({
        "mean_log": logged.mean(),
        "std_log": logged.std(ddof=0).clip(lower=MIN_STD),
        "median": normal[NUMERIC_COLUMNS].median(),
    })
    freq = {c: normal[c].value_counts(normalize=True).to_dict() for c in CATEGORICAL_COLUMNS}
    return baseline, freq


def train_bundle(train_df: pd.DataFrame, calibration_fraction: float = 0.2,
                 n_estimators: int = 200, random_state: int = 42) -> ModelBundle:
    """Fit on normal traffic only, then calibrate the 0-100 score on unseen normal rows.

    ``train_df`` must contain only connections known to be normal (a clean baseline
    period). Calibrating on rows the forest never saw avoids an over-optimistic scale.
    """
    rng = np.random.RandomState(random_state)
    order = rng.permutation(len(train_df))
    n_cal = int(len(train_df) * calibration_fraction)
    calibration = train_df.iloc[order[:n_cal]]
    fit = train_df.iloc[order[n_cal:]]

    pipeline = build_pipeline(n_estimators=n_estimators, random_state=random_state)
    pipeline.fit(fit[FEATURE_COLUMNS])
    cal_scores = np.sort(raw_anomaly_scores(pipeline, calibration))
    baseline, freq = make_baseline(train_df)
    return ModelBundle(
        pipeline=pipeline, calibration_scores=cal_scores, baseline=baseline, category_freq=freq,
        metadata={"n_fit": len(fit), "n_calibration": len(calibration), "n_estimators": n_estimators,
                  "random_state": random_state, "sklearn_version": sklearn.__version__})


def save_bundle(bundle: ModelBundle, path: Path | str = DEFAULT_MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path | str = DEFAULT_MODEL_PATH) -> ModelBundle:
    bundle = joblib.load(path)
    saved = bundle.metadata.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        raise RuntimeError(
            f"The saved model was trained with scikit-learn {saved} but {sklearn.__version__} is installed. "
            "Install the version in requirements.txt or retrain with `python -m src.train`.")
    return bundle


def score_records(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Score every record. Returns a frame with ``raw_score`` and ``score`` (0-100).

    ``raw_score`` is kept because many strongly anomalous records all reach 100 on the
    percentile scale; ranking by ``raw_score`` keeps their order meaningful.
    """
    raw = raw_anomaly_scores(bundle.pipeline, df)
    return pd.DataFrame({"raw_score": raw,
                         "score": percentile_scores(raw, bundle.calibration_scores)},
                        index=df.index)


def explain_records(bundle: ModelBundle, df: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
    """For each record, the features that make it unusual compared with normal traffic.

    Each reason is a dict: ``feature``, ``value``, ``typical`` (normal median or the
    share of normal traffic that has this category), ``z`` (std devs from normal, on
    the log scale; None for categories) and ``kind`` ("numeric" or "category").
    Numeric reasons are ranked by |z|; rare categories are listed after them.
    """
    base = bundle.baseline
    values = df[NUMERIC_COLUMNS]
    z = (np.log1p(values) - base["mean_log"]) / base["std_log"]
    explanations: list[list[dict]] = []
    z_arr, val_arr = z.to_numpy(), values.to_numpy()
    for i in range(len(df)):
        top = np.argsort(-np.abs(z_arr[i]))[:top_k]
        reasons = [
            {"feature": NUMERIC_COLUMNS[j], "value": float(val_arr[i, j]),
             "typical": float(base["median"].iloc[j]), "z": float(z_arr[i, j]), "kind": "numeric"}
            for j in top if abs(z_arr[i, j]) >= 1.0
        ]
        for col in CATEGORICAL_COLUMNS:
            value = df[col].iloc[i]
            share = bundle.category_freq[col].get(value, 0.0)
            if share < RARE_CATEGORY_FREQ:
                reasons.append({"feature": col, "value": value, "typical": float(share),
                                "z": None, "kind": "category"})
        explanations.append(reasons)
    return explanations


def summarize_reasons(reasons: list[dict], limit: int = 4) -> str:
    """One-line text version of ``explain_records`` output (stored in SQLite, shown in CSV)."""
    parts = []
    for r in reasons[:limit]:
        if r["kind"] == "numeric":
            parts.append(f"{r['feature']} {r['z']:+.1f}σ")
        else:
            parts.append(f"{r['feature']}={r['value']} ({r['typical']:.2%} of normal)")
    return "; ".join(parts)
