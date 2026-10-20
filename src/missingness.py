"""Missing data generation module.

Simulates missing data mechanisms:
- MCAR (Missing Completely At Random): equal probability across all rows.
- MAR (Missing At Random): probability driven by an observed auxiliary feature.
- MNAR (Missing Not At Random): probability driven by the variable's own true value.

All simulation parameters and column specifications are strictly driven by config.yaml.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml
import zlib


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file from path or project root."""
    path = Path(config_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config_path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_seed(
    base_seed: int,
    mechanism: str,
    level: float,
    repeat: int,
    split_name: str,
    column: Optional[str] = None,
) -> int:
    """Generate a deterministic integer seed using numpy.random.SeedSequence.

    Maps mechanism, split, and column names to fixed integers without using Python's
    built-in hash(), ensuring strict reproducibility across runs and environments.

    Args:
        base_seed: Base integer seed (e.g., from config.yaml).
        mechanism: Mechanism name ('MCAR', 'MAR', 'MNAR').
        level: Missingness rate / proportion (e.g., 0.10, 0.30, 0.50).
        repeat: Replication iteration index (e.g., 0, 1, 2).
        split_name: Data split name ('train', 'test').
        column: Optional column name for column-specific seeding.

    Returns:
        Deterministic integer seed suitable for numpy Generator or RNG.
    """
    mech_map = {
        "MCAR": 1,
        "MAR": 2,
        "MNAR": 3,
    }
    split_map = {
        "train": 1,
        "test": 2,
        "val": 3,
        "validation": 3,
    }
    col_map = {
        "sqft_living": 1,
        "grade": 2,
        "bathrooms": 3,
        "yr_built": 4,
    }

    mech_str = str(mechanism).strip().upper()
    mech_id = mech_map.get(mech_str, zlib.crc32(mech_str.encode("utf-8")) % 10000 + 100)

    split_str = str(split_name).strip().lower()
    split_id = split_map.get(split_str, zlib.crc32(split_str.encode("utf-8")) % 10000 + 100)

    if column is None:
        col_id = 0
    else:
        col_str = str(column).strip()
        col_id = col_map.get(col_str, zlib.crc32(col_str.encode("utf-8")) % 10000 + 100)

    level_id = int(round(float(level) * 10000))

    entropy = [
        int(base_seed),
        int(mech_id),
        int(level_id),
        int(repeat),
        int(split_id),
        int(col_id),
    ]
    ss = np.random.SeedSequence(entropy)
    return int(ss.generate_state(1)[0])


def generate_missingness(
    df: pd.DataFrame,
    mechanism: str,
    rate: float,
    seed: Union[int, np.random.Generator],
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject missingness into specified columns according to configured mechanism and rate.

    Args:
        df: Input clean dataframe (train or test split).
        mechanism: Mechanism name ('MCAR', 'MAR', 'MNAR').
        rate: Missingness proportion (e.g., 0.10 to 0.50).
        seed: Deterministic integer seed or numpy Generator instance.
        config: Configuration dictionary (loaded from config.yaml if None).

    Returns:
        Tuple of (df_corrupted, mask_df):
        - df_corrupted: A copy of df with NaN where values were removed (original df untouched).
        - mask_df: Boolean dataframe, same index as df, columns = missingness_columns,
                   where True indicates a value was removed.
    """
    if config is None:
        config = load_config()

    missingness_cols = config["missingness_columns"]
    always_observed_cols = config["always_observed"]
    mar_driver = config["mar_driver"]
    strength = float(config.get("strength", 1.0))

    # Assert MAR driver is not in missingness_columns and has no NaN
    assert mar_driver not in missingness_cols, (
        f"MAR driver '{mar_driver}' must not be in missingness_columns {missingness_cols}"
    )
    assert mar_driver in df.columns, (
        f"MAR driver '{mar_driver}' not found in DataFrame columns"
    )
    assert not df[mar_driver].isna().any(), (
        f"MAR driver '{mar_driver}' contains NaN values"
    )

    # Validate mechanism
    mech = mechanism.strip().upper()
    valid_mechanisms = [m.upper() for m in config.get("mechanisms", ["MCAR", "MAR", "MNAR"])]
    if mech not in valid_mechanisms:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Expected one of {valid_mechanisms}")

    assert 0.0 <= rate <= 1.0, f"Rate must be between 0.0 and 1.0, got {rate}"

    # Setup generator
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # df_corrupted is a deep copy to ensure the original dataframe is completely untouched
    df_corrupted = df.copy()
    n_rows = len(df)
    m = int(round(rate * n_rows))

    mask_df = pd.DataFrame(False, index=df.index, columns=list(missingness_cols))

    for col in missingness_cols:
        assert col in df.columns, f"Missingness column '{col}' not found in DataFrame"

        if m == 0:
            continue

        if mech == "MCAR":
            weights = np.ones(n_rows, dtype=float)
        elif mech == "MAR":
            driver_vals = df[mar_driver].to_numpy(dtype=float)
            driver_std = np.std(driver_vals)
            if driver_std == 0.0 or np.isclose(driver_std, 0.0):
                weights = np.ones(n_rows, dtype=float)
            else:
                z = (driver_vals - np.mean(driver_vals)) / driver_std
                sz = strength * z
                weights = np.exp(sz - np.max(sz))
        elif mech == "MNAR":
            col_vals = df[col].to_numpy(dtype=float)
            col_std = np.std(col_vals)
            if col_std == 0.0 or np.isclose(col_std, 0.0):
                weights = np.ones(n_rows, dtype=float)
            else:
                z = (col_vals - np.mean(col_vals)) / col_std
                sz = strength * z
                weights = np.exp(sz - np.max(sz))
        else:
            raise ValueError(f"Unsupported mechanism: {mechanism}")

        weights_sum = np.sum(weights)
        if weights_sum == 0.0 or np.isnan(weights_sum):
            probs = np.ones(n_rows, dtype=float) / n_rows
        else:
            probs = weights / weights_sum
            probs = probs / np.sum(probs)

        chosen_indices = rng.choice(n_rows, size=m, replace=False, p=probs)
        mask_df.iloc[chosen_indices, mask_df.columns.get_loc(col)] = True
        df_corrupted.iloc[chosen_indices, df_corrupted.columns.get_loc(col)] = np.nan

    # Assert columns in always_observed never get missing values
    for obs_col in always_observed_cols:
        if obs_col in df_corrupted.columns:
            assert not df_corrupted[obs_col].isna().any(), (
                f"Always observed column '{obs_col}' contains missing values in corrupted dataframe!"
            )

    return df_corrupted, mask_df
