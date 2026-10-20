"""Data processing and cleaning pipeline for the missing data research project.

Steps follow the rigorous preprocessing specifications:
1. Load raw King County housing data.
2. Parse date, sort ascending, and deduplicate by ID keeping first sale.
3. Drop specified structural/geographic columns.
4. Construct renovated indicator and drop yr_renovated.
5. Compute log_price target variable.
6. Verify completeness (zero missing values).
7. Create reproducible train/test row split using base_seed.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_clean(
    path: str,
    config: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load and clean the raw dataset following ordered preprocessing steps.

    Steps performed in exact order:
      a. Read the CSV.
      b. Parse the date column (format like 20141013T000000), sort by date
         ascending, then keep only the FIRST sale per id (drop_duplicates on id,
         keep first). This removes repeat sales.
      c. Drop columns: id, date, zipcode, sqft_above, sqft_basement.
      d. Create renovated = 1 if yr_renovated > 0 else 0, then drop yr_renovated.
      e. Add log_price = natural log of price.
      f. Assert there are no missing values.

    Args:
        path: Path to the raw CSV file.
        config: Optional configuration dictionary. If None, loaded from config.yaml.
        save_path: Optional destination to save cleaned CSV. If None, uses config.

    Returns:
        Cleaned pd.DataFrame.
    """
    if config is None:
        config = load_config()

    date_format = config.get("date_format", "%Y%m%dT%H%M%S")
    date_col = config.get("date_column", "date")
    id_col = config.get("id_column", "id")
    target_col = config.get("target", "price")
    log_target_name = config.get("log_target_name", "log_price")
    columns_to_drop = config.get(
        "columns_to_drop", ["id", "date", "zipcode", "sqft_above", "sqft_basement"]
    )
    renovated_src = config.get("renovated_column_source", "yr_renovated")
    renovated_tgt = config.get("renovated_column_target", "renovated")
    expected_cols = config.get("expected_columns", None)

    # a. Read the CSV
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    df = pd.read_csv(raw_path)

    # b. Parse date, sort ascending, keep only FIRST sale per id
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.sort_values(by=date_col, ascending=True)
    df = df.drop_duplicates(subset=[id_col], keep="first")

    # c. Drop columns: id, date, zipcode, sqft_above, sqft_basement
    df = df.drop(columns=columns_to_drop)

    # d. Create renovated = 1 if yr_renovated > 0 else 0, then drop yr_renovated
    df[renovated_tgt] = (df[renovated_src] > 0).astype(int)
    df = df.drop(columns=[renovated_src])

    # e. Add log_price = natural log of price
    df[log_target_name] = np.log(df[target_col])

    # Reorder columns to expected structure if configured
    if expected_cols is not None:
        df = df[expected_cols]

    # Reset row index after sorting and deduplication
    df = df.reset_index(drop=True)

    # f. Assert there are no missing values
    missing_count = int(df.isna().sum().sum())
    assert missing_count == 0, f"Expected 0 missing values, found {missing_count}"

    # Save cleaned dataset if save_path or configured clean_data_path is specified
    output_path = save_path or config.get("clean_data_path")
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def make_split(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible train/test row split and persist split indices.

    Args:
        df: Cleaned pd.DataFrame.
        config: Optional configuration dictionary.
        output_path: Optional destination to save split_indices.csv.

    Returns:
        Tuple of (train_df, test_df).
    """
    if config is None:
        config = load_config()

    test_size = float(config.get("test_size", 0.2))
    base_seed = int(config.get("base_seed", 42))
    dest_path = output_path or config.get("split_indices_path", "data/processed/split_indices.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=base_seed,
        shuffle=True,
    )

    # Prepare split indices records
    train_indices = list(train_df.index)
    test_indices = list(test_df.index)

    split_df = pd.DataFrame(
        {
            "row_index": train_indices + test_indices,
            "split": ["train"] * len(train_indices) + ["test"] * len(test_indices),
        }
    ).sort_values("row_index").reset_index(drop=True)

    dest_file = Path(dest_path)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(dest_file, index=False)

    return train_df, test_df


if __name__ == "__main__":
    cfg = load_config()
    raw_file = cfg.get("raw_data_path", "data/raw/kc_house_data.csv")
    cleaned_df = load_and_clean(raw_file, config=cfg)
    train_data, test_data = make_split(cleaned_df, config=cfg)
    print(f"Data cleaning complete. Cleaned shape: {cleaned_df.shape}")
    print(f"Train split size: {len(train_data)}, Test split size: {len(test_data)}")
