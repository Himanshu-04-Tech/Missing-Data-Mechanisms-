"""Evaluation metrics for missing data imputation and downstream regression models.

Calculates:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 score
- Parameter estimation coverage and confidence intervals
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_imputation(
    y_true: pd.Series,
    y_imputed: pd.Series,
) -> Dict[str, float]:
    """Evaluate imputation accuracy against ground truth values.

    Args:
        y_true: True uncorrupted series.
        y_imputed: Imputed series.

    Returns:
        Dictionary of RMSE, MAE, and R2.
    """
    valid_mask = (~y_true.isna()) & (~y_imputed.isna())
    yt = y_true[valid_mask]
    yp = y_imputed[valid_mask]

    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))

    return {"rmse": rmse, "mae": mae, "r2": r2}
