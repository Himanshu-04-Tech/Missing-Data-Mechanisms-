"""Live Lab engine: runs the project's own code on a small sample of houses.

Uses, unchanged:
  src.missingness.generate_missingness   (Member 1)
  src.imputation.impute / score_imputation (Member 2)
  src.statistics.fit_ols                   (statistical track)
and the same leakage rules as the pipeline: imputers and models are fit on
train rows only, the target is never an imputer input, truth is used only to score.

The numbers here are a demonstration on a sample. They are noisier than, and
not the same as, the paper's results, which come from the full 21,436 houses x 5 repeats.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import data as D  # noqa: F401  (puts the project root on sys.path)
from src.imputation import impute, score_imputation
from src.missingness import generate_missingness
from src.statistics import fit_ols, get_model_spec

LAB_RF_TREES = 150  # smaller than the paper's 300 trees so the demo answers in seconds


@st.cache_data(show_spinner=False)
def sample(n_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A random sample of houses that keeps the project's 80/20 train/test split."""
    df = D.clean_data()
    split = D.split_indices().set_index("row_index")["split"]
    rng = np.random.default_rng(seed)
    n_train = int(round(n_rows * 0.8))
    train = rng.choice(split[split == "train"].index.to_numpy(), n_train, replace=False)
    test = rng.choice(split[split == "test"].index.to_numpy(), n_rows - n_train, replace=False)
    idx = np.sort(np.concatenate([train, test]))
    return df.loc[idx].copy(), np.sort(train), np.sort(test)


@st.cache_data(show_spinner=False)
def corrupt(n_rows: int, seed: int, mechanism: str, level: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove values from train and test separately, exactly like the pipeline."""
    truth, train_idx, test_idx = sample(n_rows, seed)
    cfg = D.config()
    parts, masks = [], []
    for k, idx in enumerate([train_idx, test_idx]):
        c, m = generate_missingness(truth.loc[idx], mechanism, level, seed * 10 + k, cfg)
        parts.append(c)
        masks.append(m)
    corrupted = pd.concat(parts).loc[truth.index]
    mask = pd.concat(masks).loc[truth.index]
    return corrupted, mask


@st.cache_data(show_spinner=False)
def fill(n_rows: int, seed: int, mechanism: str, level: float, method: str) -> dict:
    truth, train_idx, test_idx = sample(n_rows, seed)
    corrupted, mask = corrupt(n_rows, seed, mechanism, level)
    cfg = D.config()

    filled = impute(corrupted, train_idx, method, cfg)
    scores = score_imputation(filled, truth, mask, train_idx, test_idx)

    # ── statistics: OLS on filled data vs OLS on the true data (same rows) ──
    ref = fit_ols(truth, cfg)
    imp = fit_ols(filled, cfg)
    _, _, miss_cols, _ = get_model_spec(cfg)
    stat = pd.DataFrame(
        {
            "term": ref.index,
            "beta_ref": ref["coef"].to_numpy(),
            "se_ref": ref["se"].to_numpy(),
            "beta_imp": imp["coef"].to_numpy(),
            "ci_lo_imp": imp["ci_low"].to_numpy(),
            "ci_hi_imp": imp["ci_high"].to_numpy(),
        }
    )
    stat = stat[stat["term"] != "const"]
    stat["z_diff"] = (stat["beta_imp"] - stat["beta_ref"]) / stat["se_ref"]
    stat["covers"] = (stat["ci_lo_imp"] <= stat["beta_ref"]) & (stat["beta_ref"] <= stat["ci_hi_imp"])
    stat["is_affected"] = stat["term"].isin(miss_cols)

    # ── prediction: RF trained on filled train vs RF trained on true train ──
    _, predictors, _, _ = get_model_spec(cfg)
    target = cfg["log_target_name"]

    def rf_rmse(frame: pd.DataFrame) -> tuple[float, float]:
        model = RandomForestRegressor(n_estimators=LAB_RF_TREES, max_features=1.0, n_jobs=-1, random_state=int(cfg["base_seed"]))
        model.fit(frame.loc[train_idx, predictors], frame.loc[train_idx, target])
        pred = model.predict(frame.loc[test_idx, predictors])
        y = frame.loc[test_idx, target]
        return float(np.sqrt(mean_squared_error(y, pred))), float(r2_score(y, pred))

    rmse_ref, r2_ref = rf_rmse(truth)
    rmse_imp, r2_imp = rf_rmse(filled)

    aff = stat[stat["is_affected"]]
    return {
        "filled": filled,
        "scores": scores,
        "stat": stat,
        "stat_value": float(aff["z_diff"].abs().mean()),
        "coverage": float(aff["covers"].mean()),
        "n_cover": int(aff["covers"].sum()),
        "n_aff": int(len(aff)),
        "rmse_ref": rmse_ref,
        "rmse_imp": rmse_imp,
        "r2_ref": r2_ref,
        "r2_imp": r2_imp,
        "pred_value": (rmse_imp - rmse_ref) / rmse_ref,
    }
