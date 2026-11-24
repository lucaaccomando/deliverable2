"""Shared fixtures. The 3,000-row sample file doubles as realistic test data."""
import pandas as pd
import pytest

from src.data_loader import load_connections
from src.model import DEFAULT_MODEL_PATH, load_bundle


@pytest.fixture(scope="session")
def sample() -> pd.DataFrame:
    return load_connections("data/sample_nslkdd.csv")


@pytest.fixture(scope="session")
def bundle():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("model/threatlens_model.joblib missing; run `python -m src.train`")
    return load_bundle()
