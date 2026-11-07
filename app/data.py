"""Read-only loaders for the result tables. Nothing here is typed by hand:
every number shown in the app comes from these CSV files."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLES = ROOT / "results" / "tables"
EXP = ROOT / "results" / "experiment_results"


@st.cache_data
def config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def cases() -> pd.DataFrame:
    """60 rows: 4 methods x 3 mechanisms x 5 levels, with stat/pred values and Case A-D."""
    return pd.read_csv(TABLES / "cases_by_condition.csv")


@st.cache_data
def best_methods() -> pd.DataFrame:
    return pd.read_csv(TABLES / "table6b_best_methods.csv")


@st.cache_data
def case_sensitivity() -> pd.DataFrame:
    return pd.read_csv(TABLES / "case_sensitivity.csv")


@st.cache_data
def coefficients() -> pd.DataFrame:
    return pd.read_csv(EXP / "stat_recovery_coefficients.csv")


@st.cache_data
def predictive_reference() -> pd.Series:
    ref = pd.read_csv(EXP / "predictive_reference.csv")
    return ref[ref["is_primary"]].iloc[0]


@st.cache_data
def reference_ols() -> pd.DataFrame:
    return pd.read_csv(TABLES / "reference_ols.csv")


@st.cache_data
def missingness_summary() -> pd.DataFrame:
    return pd.read_csv(TABLES / "missingness_summary.csv")


@st.cache_data
def clean_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "clean_data.csv")


@st.cache_data
def split_indices() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "split_indices.csv")


def classify(stat_value, pred_value, stat_thr: float, pred_thr: float):
    """Case A-D from the two 'smaller is better' metrics (same rule as src/m3_cases.py)."""
    stat_good = stat_value <= stat_thr
    pred_good = pred_value <= pred_thr
    if hasattr(stat_good, "map"):
        out = pd.Series("D", index=stat_value.index)
        out[stat_good & pred_good] = "A"
        out[~stat_good & pred_good] = "B"
        out[stat_good & ~pred_good] = "C"
        return out
    if stat_good and pred_good:
        return "A"
    if pred_good:
        return "B"
    if stat_good:
        return "C"
    return "D"
