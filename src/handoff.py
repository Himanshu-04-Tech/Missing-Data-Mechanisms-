"""Handoff module for Member 2.

Provides convenient data loaders to access the 75 simulated missing data conditions,
ground truth evaluations, missingness masks, and train/test partition indices.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import yaml


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file from path or project root."""
    path = Path(config_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config_path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_conditions(config_path: str = "config.yaml") -> pd.DataFrame:
    """Return the sweep manifest as a pandas DataFrame.

    Lists all 75 simulated missingness conditions across mechanisms (MCAR, MAR, MNAR),
    missingness levels (0.10, 0.20, 0.30, 0.40, 0.50), and repeats (0 to 4),
    including seeds, file paths, NaN counts, and completeness metrics.

    Args:
        config_path: Path to configuration YAML file. Defaults to 'config.yaml'.

    Returns:
        pd.DataFrame: Manifest dataframe containing 75 condition rows.
    """
    config = _load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    gen_dir = project_root / config.get("generated_missing_dir", "data/generated_missing")
    manifest_path = gen_dir / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest file not found at '{manifest_path}'. "
            "Please run 'python scripts/run_sweep.py' first."
        )

    return pd.read_csv(manifest_path)


def load_condition(
    mechanism: str,
    level: Union[float, int],
    repeat: int,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Load a specific corrupted condition along with truth, mask, and split indices.

    Args:
        mechanism: Missingness mechanism name ('MCAR', 'MAR', or 'MNAR'), case-insensitive.
        level: Missingness rate proportion (e.g. 0.10, 0.20, 0.30, 0.40, 0.50).
        repeat: Repeat replication index (integer 0 to 4).
        config_path: Path to configuration YAML file. Defaults to 'config.yaml'.

    Returns:
        dict with the following keys:
            - 'corrupted': pd.DataFrame (21,436 rows x 17 columns) containing simulated NaNs
                           in the 4 missingness columns with row_index as DataFrame index.
            - 'truth': pd.DataFrame (21,436 rows x 17 columns) untouched clean data
                       with row_index as DataFrame index. FOR SCORING ONLY, never use as imputer input.
            - 'mask': pd.DataFrame (21,436 rows x 4 columns) boolean DataFrame corresponding to
                      the 4 missingness columns, where True indicates a missing value (equal to
                      corrupted[missingness_columns].isna()).
            - 'train_idx': np.ndarray of row indices belonging to the training split (17,148 rows).
            - 'test_idx': np.ndarray of row indices belonging to the test split (4,288 rows).

    Raises:
        ValueError: If mechanism, level, or repeat are not valid configurations.
        FileNotFoundError: If the requested condition file or ground truth is missing.
    """
    config = _load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    gen_dir = project_root / config.get("generated_missing_dir", "data/generated_missing")

    # Normalize inputs
    mech_norm = str(mechanism).strip().upper()
    valid_mechanisms = [m.upper() for m in config.get("mechanisms", ["MCAR", "MAR", "MNAR"])]
    if mech_norm not in valid_mechanisms:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Expected one of {valid_mechanisms}")

    lvl_float = float(level)
    valid_levels = [float(lvl) for lvl in config.get("levels", [0.10, 0.20, 0.30, 0.40, 0.50])]
    # Match level with floating tolerance
    matched_level = None
    for vl in valid_levels:
        if np.isclose(lvl_float, vl, atol=1e-4):
            matched_level = vl
            break
    if matched_level is None:
        raise ValueError(f"Invalid level {level}. Expected one of {valid_levels}")

    rep_int = int(repeat)
    n_repeats = int(config.get("n_repeats", 5))
    if rep_int < 0 or rep_int >= n_repeats:
        raise ValueError(f"Invalid repeat {repeat}. Expected 0 <= repeat < {n_repeats}")

    pct = int(round(matched_level * 100))
    corrupted_path = gen_dir / mech_norm / f"rate_{pct:02d}" / f"rep_{rep_int}.csv.gz"
    truth_path = gen_dir / "ground_truth.csv.gz"
    split_path = gen_dir / "split_indices.csv"

    if not corrupted_path.is_file():
        raise FileNotFoundError(f"Corrupted dataset not found at '{corrupted_path}'.")
    if not truth_path.is_file():
        raise FileNotFoundError(f"Ground truth dataset not found at '{truth_path}'.")
    if not split_path.is_file():
        # Fallback to config path if split_indices.csv is not yet in gen_dir
        split_path = project_root / config["split_indices_path"]

    # Load datasets with row_index as the DataFrame index
    corrupted_df = pd.read_csv(corrupted_path, index_col="row_index")
    truth_df = pd.read_csv(truth_path, index_col="row_index")
    split_df = pd.read_csv(split_path)

    missingness_cols = list(config["missingness_columns"])
    mask_df = corrupted_df[missingness_cols].isna()

    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].to_numpy()
    test_idx = split_df.loc[split_df["split"] == "test", "row_index"].to_numpy()

    return {
        "corrupted": corrupted_df,
        "truth": truth_df,
        "mask": mask_df,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
