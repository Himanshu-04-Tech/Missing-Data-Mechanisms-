"""Sweep generation module for missing data simulation.

Generates 75 corrupted housing datasets across 3 mechanisms (MCAR, MAR, MNAR),
5 missingness levels (10% to 50%), and 5 repeats (0 to 4), corrupting train and test
splits separately with deterministic seeds, and saving reassembled datasets, ground truth,
split indices, and an experiment manifest.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.missingness import generate_missingness, make_seed


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


def generate_single_condition(
    df_clean: pd.DataFrame,
    split_df: pd.DataFrame,
    mechanism: str,
    level: float,
    repeat: int,
    config: dict,
) -> Tuple[pd.DataFrame, Dict]:
    """Corrupt train and test splits separately and reassemble into original row order.

    Args:
        df_clean: Clean baseline dataframe (21,436 rows x 17 columns).
        split_df: Split indices dataframe with 'row_index' and 'split' columns.
        mechanism: Missingness mechanism ('MCAR', 'MAR', 'MNAR').
        level: Missingness rate proportion (0.10 to 0.50).
        repeat: Repeat iteration index (0 to 4).
        config: Study configuration dictionary.

    Returns:
        Tuple of (df_reassembled, stats_dict):
            - df_reassembled: Corrupted DataFrame in original row order (21,436 x 17).
            - stats_dict: Metadata including seeds, NaN counts, and completeness fractions.
    """
    base_seed = int(config["base_seed"])
    missingness_cols = list(config["missingness_columns"])

    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].values
    test_idx = split_df.loc[split_df["split"] == "test", "row_index"].values

    train_df = df_clean.iloc[train_idx].copy()
    test_df = df_clean.iloc[test_idx].copy()

    seed_train = make_seed(base_seed, mechanism, level, repeat, "train")
    seed_test = make_seed(base_seed, mechanism, level, repeat, "test")

    df_corr_train, mask_train = generate_missingness(train_df, mechanism, level, seed_train, config)
    df_corr_test, mask_test = generate_missingness(test_df, mechanism, level, seed_test, config)

    # Reassemble preserving the original row order
    df_reassembled = pd.concat([df_corr_train, df_corr_test]).sort_index()

    # Collect stats for the manifest
    stats: Dict = {
        "mechanism": mechanism,
        "level": float(level),
        "repeat": int(repeat),
        "seed_train": int(seed_train),
        "seed_test": int(seed_test),
    }

    for col in missingness_cols:
        stats[f"train_nan_{col}"] = int(df_corr_train[col].isna().sum())
    for col in missingness_cols:
        stats[f"test_nan_{col}"] = int(df_corr_test[col].isna().sum())

    train_no_missing = (~df_corr_train[missingness_cols].isna().any(axis=1)).mean()
    test_no_missing = (~df_corr_test[missingness_cols].isna().any(axis=1)).mean()

    stats["train_no_missing_fraction"] = float(train_no_missing)
    stats["test_no_missing_fraction"] = float(test_no_missing)

    return df_reassembled, stats


def run_sweep(
    config_path: str = "config.yaml",
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run full missingness simulation sweep and save all datasets, ground truth, and manifest.

    Args:
        config_path: Path to configuration YAML.
        project_root: Optional root directory path.
        verbose: If True, prints compact summary progress.

    Returns:
        pd.DataFrame: The manifest of all 75 simulated conditions.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    config = load_config(str(project_root / config_path) if not Path(config_path).is_file() else config_path)
    clean_csv_path = project_root / config["clean_data_path"]
    split_csv_path = project_root / config["split_indices_path"]
    gen_dir_rel = config.get("generated_missing_dir", "data/generated_missing")
    out_dir = project_root / gen_dir_rel

    out_dir.mkdir(parents=True, exist_ok=True)

    df_clean = pd.read_csv(clean_csv_path)
    split_df = pd.read_csv(split_csv_path)

    # 1. Save ground_truth.csv.gz (untouched clean data with row_index as first column)
    gt_path = out_dir / "ground_truth.csv.gz"
    df_gt = df_clean.copy()
    df_gt.insert(0, "row_index", df_gt.index)
    df_gt.to_csv(gt_path, index=False, compression="gzip")

    # 2. Save split copy
    split_copy_path = out_dir / "split_indices.csv"
    split_df.to_csv(split_copy_path, index=False)

    # 3. Iterate over all conditions
    manifest_rows = []
    mechanisms = config["mechanisms"]
    levels = config["levels"]
    n_repeats = int(config["n_repeats"])

    total_conditions = len(mechanisms) * len(levels) * n_repeats

    if verbose:
        print(f"Starting sweep generation: {total_conditions} conditions ({len(mechanisms)} mechs x {len(levels)} levels x {n_repeats} reps)...")

    for mech in mechanisms:
        for lvl in levels:
            pct = int(round(float(lvl) * 100))
            lvl_dir = out_dir / mech / f"rate_{pct:02d}"
            lvl_dir.mkdir(parents=True, exist_ok=True)

            for rep in range(n_repeats):
                df_reassembled, stats = generate_single_condition(
                    df_clean=df_clean,
                    split_df=split_df,
                    mechanism=mech,
                    level=lvl,
                    repeat=rep,
                    config=config,
                )

                rep_filename = f"rep_{rep}.csv.gz"
                file_rel_path = f"{gen_dir_rel}/{mech}/rate_{pct:02d}/{rep_filename}"
                target_path = lvl_dir / rep_filename

                df_to_save = df_reassembled.copy()
                df_to_save.insert(0, "row_index", df_to_save.index)
                df_to_save.to_csv(target_path, index=False, compression="gzip")

                row_record = {
                    "mechanism": mech,
                    "level": float(lvl),
                    "repeat": int(rep),
                    "path": file_rel_path,
                    "seed_train": stats["seed_train"],
                    "seed_test": stats["seed_test"],
                }
                # Add NaN counts and completeness stats
                for k, v in stats.items():
                    if k not in row_record:
                        row_record[k] = v

                manifest_rows.append(row_record)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    if verbose:
        print(f"Sweep complete. Generated {len(manifest_df)} conditions, ground truth, split copy, and manifest in '{gen_dir_rel}'.")

    return manifest_df
