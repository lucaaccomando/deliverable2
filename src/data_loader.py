"""Load and validate network connection records (NSL-KDD column format).

The app never trusts an uploaded file. ``load_connections`` either returns a clean
DataFrame that the model can score, or raises ``ValidationError`` with a message
that tells the analyst exactly what to fix.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import pandas as pd

# The 41 NSL-KDD features, in the order they appear in the original files.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# Optional ground-truth column. It is NEVER given to the model; it is only used to
# measure how well the detector did when the file happens to be labeled.
LABEL_COLUMN = "label"

# The original NSL-KDD .txt files have no header row: 41 features, then the attack
# label, then a "difficulty" score the benchmark authors added.
_RAW_COLUMN_COUNTS = {41: FEATURE_COLUMNS,
                      42: FEATURE_COLUMNS + [LABEL_COLUMN],
                      43: FEATURE_COLUMNS + [LABEL_COLUMN, "difficulty"]}

MAX_ROWS = 250_000
MAX_BYTES = 100 * 1024 * 1024

Source = Union[str, Path, bytes, BinaryIO]


class ValidationError(ValueError):
    """The uploaded file cannot be scored. ``str(error)`` is safe to show to the user."""


def _read_bytes(source: Source) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    if hasattr(source, "getvalue"):          # Streamlit UploadedFile / BytesIO
        return source.getvalue()
    return source.read()


def _parse(raw: bytes) -> pd.DataFrame:
    """Parse CSV bytes, handling both headered CSVs and the headerless NSL-KDD .txt."""
    if not raw.strip():
        raise ValidationError("The file is empty.")
    if len(raw) > MAX_BYTES:
        raise ValidationError(f"The file is larger than {MAX_BYTES // 1024 // 1024} MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValidationError(f"Could not read the file as CSV: {exc}") from exc

    # Headerless NSL-KDD file: pandas used the first data row as the header, so none of
    # the expected column names are present and the column count matches the format.
    if not set(FEATURE_COLUMNS) & set(df.columns) and df.shape[1] in _RAW_COLUMN_COUNTS:
        df = pd.read_csv(io.BytesIO(raw), header=None,
                         names=_RAW_COLUMN_COUNTS[df.shape[1]])
    return df


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, types and values. Returns a clean copy or raises ValidationError."""
    if len(df) == 0:
        raise ValidationError("The file has a header but no rows.")
    if len(df) > MAX_ROWS:
        raise ValidationError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10 else "")
        raise ValidationError(
            f"{len(missing)} required column(s) are missing: {shown}. "
            "Use the NSL-KDD column names (see the sample file) or the original 41-column .txt layout.")

    keep = FEATURE_COLUMNS + ([LABEL_COLUMN] if LABEL_COLUMN in df.columns else [])
    out = df[keep].copy()

    problems = []
    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(out[col], errors="coerce")
        bad = values.isna() | np.isinf(values)
        if bad.any():
            first = int(np.flatnonzero(bad.to_numpy())[0]) + 1
            problems.append(f"'{col}' has {int(bad.sum())} non-numeric or missing value(s) (first at data row {first})")
        elif (values < 0).any():
            first = int(np.flatnonzero((values < 0).to_numpy())[0]) + 1
            problems.append(f"'{col}' has {int((values < 0).sum())} negative value(s) (first at data row {first}); "
                            "connection counters and rates cannot be negative")
        out[col] = values
    for col in CATEGORICAL_COLUMNS:
        values = out[col].astype("string").str.strip()
        if values.isna().any() or (values == "").any():
            problems.append(f"'{col}' has {int((values.isna() | (values == '')).sum())} missing value(s)")
        out[col] = values.astype(object)
    if problems:
        raise ValidationError("The file has data problems:\n- " + "\n- ".join(problems[:8]))

    if LABEL_COLUMN in out.columns:
        out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(str).str.strip().str.lower()
    return out.reset_index(drop=True)


def load_connections(source: Source) -> pd.DataFrame:
    """Load a CSV of connection records and return a validated DataFrame."""
    return validate_frame(_parse(_read_bytes(source)))


def is_attack(labels: pd.Series) -> np.ndarray:
    """Ground-truth helper: anything not labelled 'normal' counts as an attack."""
    return (labels.astype(str).str.lower() != "normal").to_numpy()
