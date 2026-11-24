import io

import pandas as pd
import pytest

from src.data_loader import FEATURE_COLUMNS, ValidationError, load_connections, validate_frame


def csv_bytes(df: pd.DataFrame, header: bool = True) -> bytes:
    return df.to_csv(index=False, header=header).encode()


def test_sample_loads_with_label(sample):
    assert len(sample) == 3000
    assert list(sample.columns) == FEATURE_COLUMNS + ["label"]


def test_headerless_nslkdd_txt_layout(sample):
    """The original KDDTest+.txt has no header and has label + difficulty columns."""
    raw = sample.head(20).copy()
    raw["difficulty"] = 15
    df = load_connections(csv_bytes(raw, header=False))
    assert len(df) == 20 and "label" in df.columns and "difficulty" not in df.columns


def test_unlabeled_file_is_accepted(sample):
    df = load_connections(csv_bytes(sample.drop(columns="label").head(10)))
    assert "label" not in df.columns


def test_extra_columns_are_ignored(sample):
    extra = sample.head(5).assign(notes="x")
    assert "notes" not in load_connections(csv_bytes(extra)).columns


def test_missing_columns_listed_in_message(sample):
    with pytest.raises(ValidationError, match=r"2 required column\(s\) are missing: service, src_bytes"):
        load_connections(csv_bytes(sample.head(5).drop(columns=["service", "src_bytes"])))


def test_non_numeric_value_names_column_and_row(sample):
    bad = sample.head(5).copy()
    bad["src_bytes"] = bad["src_bytes"].astype(object)
    bad.loc[2, "src_bytes"] = "lots"
    with pytest.raises(ValidationError, match=r"'src_bytes' has 1 non-numeric .*data row 3"):
        load_connections(csv_bytes(bad))


def test_negative_values_rejected(sample):
    bad = sample.head(5).copy()
    bad.loc[0, "count"] = -1
    with pytest.raises(ValidationError, match="negative"):
        validate_frame(bad)


def test_missing_categorical_rejected(sample):
    bad = sample.head(5).copy()
    bad.loc[1, "service"] = None
    with pytest.raises(ValidationError, match="'service' has 1 missing"):
        load_connections(csv_bytes(bad))


@pytest.mark.parametrize("payload", [b"", b"   \n"])
def test_empty_file(payload):
    with pytest.raises(ValidationError, match="empty"):
        load_connections(payload)


def test_header_only_file(sample):
    with pytest.raises(ValidationError, match="no rows"):
        load_connections(csv_bytes(sample.head(0)))


def test_binary_garbage_gives_validation_error_not_crash():
    with pytest.raises(ValidationError):
        load_connections(bytes(range(256)) * 4)


def test_accepts_file_like_object(sample):
    assert len(load_connections(io.BytesIO(csv_bytes(sample.head(7))))) == 7
