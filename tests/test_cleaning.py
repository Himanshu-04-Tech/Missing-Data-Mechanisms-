"""Pytest suite for data cleaning, train/test splitting, and file immutability."""

import hashlib
from pathlib import Path
import sys
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import load_and_clean, load_config, make_split


@pytest.fixture(scope="session")
def config():
    return load_config(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def clean_data(config):
    raw_path = str(PROJECT_ROOT / config["raw_data_path"])
    return load_and_clean(raw_path, config=config)


def test_row_count(clean_data, config):
    """Verify cleaned dataset has exactly 21,436 rows."""
    assert len(clean_data) == config["expected_rows"]


def test_columns_list_and_order(clean_data, config):
    """Verify cleaned dataset has exact expected column names and ordering."""
    assert list(clean_data.columns) == config["expected_columns"]


def test_zero_missing_values(clean_data):
    """Verify zero NaN/null values exist across the entire dataframe."""
    assert clean_data.isna().sum().sum() == 0


def test_zero_duplicate_rows_or_indices(clean_data, config):
    """Verify row indices and property records have zero duplicate representations."""
    assert not clean_data.index.duplicated().any()
    raw_df = pd.read_csv(PROJECT_ROOT / config["raw_data_path"])
    assert raw_df[config["id_column"]].nunique() == len(clean_data)


def test_renovated_value_counts(clean_data, config):
    """Verify binary renovated feature counts match 20,526 (0) and 910 (1)."""
    counts = clean_data[config["renovated_column_target"]].value_counts().to_dict()
    assert counts.get(0) == 20526
    assert counts.get(1) == 910


def test_train_test_split(clean_data, config):
    """Verify 80/20 train/test split size and disjoint index sets."""
    train_df, test_df = make_split(clean_data, config=config)
    import math
    expected_rows = config["expected_rows"]
    expected_test = math.ceil(expected_rows * config["test_size"])
    expected_train = expected_rows - expected_test

    assert len(train_df) == expected_train
    assert len(test_df) == expected_test
    assert len(train_df) + len(test_df) == expected_rows
    assert set(train_df.index).isdisjoint(set(test_df.index))


def test_raw_csv_unmodified(config):
    """Verify raw CSV checksum, row count, and column count remain unchanged."""
    raw_path = PROJECT_ROOT / config["raw_data_path"]
    hasher = hashlib.md5()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == config["raw_csv_md5"]

    raw_df = pd.read_csv(raw_path)
    assert len(raw_df) == 21613
    assert len(raw_df.columns) == 21
