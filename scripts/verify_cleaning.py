"""Verification script for King County housing data cleaning and preparation.

Verifies:
1. Number of rows (expected: 21,436).
2. Exact column list and order.
3. Zero missing values and zero duplicate rows/properties.
4. Value counts of the renovated feature.
5. Train and test partition sizes (80/20 split).
6. Integrity of the raw CSV file (confirms it is unmodified).
"""

import hashlib
from pathlib import Path
import sys
import pandas as pd

# Add project root to path so src modules import reliably
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import load_and_clean, load_config, make_split


def compute_file_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file for integrity verification."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 70)
    print("RUNNING DATA CLEANING & INTEGRITY VERIFICATION")
    print("=" * 70)

    # Load configuration (single source of truth)
    config = load_config(str(project_root / "config.yaml"))
    raw_data_path = project_root / config["raw_data_path"]
    clean_data_path = project_root / config["clean_data_path"]
    split_indices_path = project_root / config["split_indices_path"]

    # ---------------------------------------------------------
    # 0. Check raw CSV presence and pre-clean hash
    # ---------------------------------------------------------
    assert raw_data_path.is_file(), f"Raw data file missing at: {raw_data_path}"
    raw_md5_before = compute_file_md5(raw_data_path)
    expected_md5 = config.get("raw_csv_md5", raw_md5_before)

    # ---------------------------------------------------------
    # Run cleaning and splitting pipeline
    # ---------------------------------------------------------
    print("\n[Step 1] Executing load_and_clean()...")
    df_clean = load_and_clean(
        path=str(raw_data_path),
        config=config,
        save_path=str(clean_data_path),
    )

    print("\n[Step 2] Executing make_split()...")
    train_df, test_df = make_split(
        df=df_clean,
        config=config,
        output_path=str(split_indices_path),
    )

    # ---------------------------------------------------------
    # 1. Assert and print number of rows
    # ---------------------------------------------------------
    expected_rows = config["expected_rows"]
    actual_rows = len(df_clean)
    print(f"\n[Verification: Row Count]")
    print(f"  Cleaned dataset rows: {actual_rows} (Expected: {expected_rows})")
    assert actual_rows == expected_rows, (
        f"Row count mismatch! Expected {expected_rows}, got {actual_rows}"
    )
    print("  -> PASSED: Row count matches 21,436 exactly.")

    # ---------------------------------------------------------
    # 2. Assert and print final column list
    # ---------------------------------------------------------
    expected_columns = config["expected_columns"]
    actual_columns = list(df_clean.columns)
    print(f"\n[Verification: Final Column List]")
    print(f"  Column count: {len(actual_columns)}")
    print(f"  Columns: {actual_columns}")
    assert actual_columns == expected_columns, (
        f"Column list mismatch!\nExpected: {expected_columns}\nActual:   {actual_columns}"
    )
    print("  -> PASSED: Final column list and ordering match expected exactly.")

    # ---------------------------------------------------------
    # 3. Assert and print zero missing values & zero duplicate rows
    # ---------------------------------------------------------
    print(f"\n[Verification: Missing Values & Duplicates]")
    total_missing = int(df_clean.isna().sum().sum())
    print(f"  Total missing values: {total_missing}")
    assert total_missing == 0, f"Found {total_missing} missing values!"
    print("  -> PASSED: Zero missing values.")

    # Check index duplicates
    index_duplicates = int(df_clean.index.duplicated().sum())
    print(f"  Duplicate row indices: {index_duplicates}")
    assert index_duplicates == 0, f"Found {index_duplicates} duplicate indices!"
    print("  -> PASSED: Zero duplicate row indices.")

    # Check property-level duplicates (repeat sales removed)
    df_raw = pd.read_csv(raw_data_path)
    raw_repeat_sales = len(df_raw) - df_raw[config["id_column"]].nunique()
    print(f"  Raw repeat sales removed: {raw_repeat_sales}")
    print(f"  Cleaned unique property count: {len(df_clean)}")
    assert len(df_clean) == df_raw[config["id_column"]].nunique(), (
        "Cleaned dataset does not match number of unique property IDs!"
    )
    print("  -> PASSED: Zero duplicate property sales (repeat sales fully removed).")

    # ---------------------------------------------------------
    # 4. Assert and print value counts of renovated
    # ---------------------------------------------------------
    print(f"\n[Verification: Value Counts of 'renovated']")
    renovated_counts = df_clean[config["renovated_column_target"]].value_counts().to_dict()
    print("  Renovated distribution:")
    for val, count in sorted(renovated_counts.items()):
        print(f"    renovated = {val}: {count:,} ({count / len(df_clean):.2%})")
    assert set(renovated_counts.keys()).issubset({0, 1}), (
        f"Unexpected values in renovated column: {set(renovated_counts.keys())}"
    )
    assert renovated_counts[0] == 20526, f"Expected 20,526 non-renovated, got {renovated_counts[0]}"
    assert renovated_counts[1] == 910, f"Expected 910 renovated, got {renovated_counts[1]}"
    print("  -> PASSED: Renovated value counts verified (0: 20,526, 1: 910).")

    # ---------------------------------------------------------
    # 5. Assert and print train and test sizes
    # ---------------------------------------------------------
    print(f"\n[Verification: Train / Test Split Sizes]")
    train_size = len(train_df)
    test_size = len(test_df)
    total_split = train_size + test_size
    print(f"  Train split size: {train_size:,} ({train_size / total_split:.1%})")
    print(f"  Test split size:  {test_size:,} ({test_size / total_split:.1%})")
    print(f"  Total split rows: {total_split:,}")

    import math
    expected_test_rows = math.ceil(expected_rows * config["test_size"])
    expected_train_rows = expected_rows - expected_test_rows
    assert train_size == expected_train_rows, (
        f"Expected train size {expected_train_rows}, got {train_size}"
    )
    assert test_size == expected_test_rows, (
        f"Expected test size {expected_test_rows}, got {test_size}"
    )
    assert total_split == expected_rows, f"Total split mismatch with expected {expected_rows}"
    assert set(train_df.index).isdisjoint(set(test_df.index)), "Train and test splits overlap!"
    print("  -> PASSED: Train (17,148) and test (4,288) partitions are mutually disjoint.")

    # ---------------------------------------------------------
    # 6. Confirm raw CSV is completely unchanged
    # ---------------------------------------------------------
    print(f"\n[Verification: Raw CSV Immutability]")
    raw_md5_after = compute_file_md5(raw_data_path)
    print(f"  Raw CSV MD5 before: {raw_md5_before}")
    print(f"  Raw CSV MD5 after:  {raw_md5_after}")
    assert raw_md5_before == raw_md5_after, "Raw CSV was modified during execution!"
    assert raw_md5_after == expected_md5, f"Raw CSV MD5 does not match expected {expected_md5}!"

    # Verify raw file row and column counts
    assert len(df_raw) == 21613, f"Expected 21,613 raw rows, found {len(df_raw)}"
    assert len(df_raw.columns) == 21, f"Expected 21 raw columns, found {len(df_raw.columns)}"
    print("  -> PASSED: Raw CSV is byte-for-byte identical, with 21,613 rows and 21 columns.")

    # ---------------------------------------------------------
    # Summary Preview (First 5 Rows)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLEANED DATASET PREVIEW (head 5 rows):")
    print(f"Cleaned shape: {df_clean.shape}")
    print(df_clean.head(5))
    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
