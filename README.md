# Missing Data Project

A data science research study investigating missing data mechanisms (MCAR, MAR, MNAR) and imputation methodologies on the King County housing dataset.

## 1. Project Overview

This study systematically introduces missingness across designated features at multiple levels (10% to 50%) under different statistical mechanisms, evaluates imputation techniques (mean, KNN, regression, random forest), and benchmarks downstream predictive performance.

## 2. Directory Structure

```text
.
├── config.yaml                     # Single source of truth for experiment parameters
├── requirements.txt                # Python project dependencies
├── README.md                       # Project documentation and usage guide
├── data/
│   ├── raw/                        # Untouched raw datasets (kc_house_data.csv)
│   ├── processed/                  # Cleaned data (clean_data.csv, split_indices.csv)
│   └── generated_missing/          # Missingness simulation datasets
├── src/                            # Modular pipeline code
│   ├── __init__.py
│   ├── data_processing.py          # Ordered loading, deduplication, cleaning & splitting
│   ├── missingness.py              # MCAR, MAR, MNAR generation functions
│   ├── statistics.py               # Statistical distributions, metrics, and bias calculations
│   └── evaluation.py               # Downstream evaluation metrics (RMSE, MAE, R2)
├── notebooks/                      # Exploratory and analytical Jupyter notebooks
├── results/
│   ├── tables/                     # Tabular experiment summaries and LaTeX/CSV exports
│   ├── figures/                    # Visualizations, diagnostic plots, and charts
│   └── experiment_results/         # Raw experiment run logs and metrics
├── scripts/
│   └── verify_cleaning.py          # Data cleaning verification & sanity check script
└── tests/
    └── test_cleaning.py            # Automated pytest suite for data pipeline
```

## 3. Configuration (`config.yaml`)

All global parameters and experiment controls are centralized in `config.yaml`:
- **`base_seed`**: 42 (ensures reproducibility)
- **`test_size`**: 0.2 (80% train / 20% test partition)
- **`target`**: `price` (and `log_price` as natural log target)
- **`missingness_columns`**: `[sqft_living, grade, bathrooms, yr_built]`
- **`always_observed`**: `[price, log_price, lat, long]`
- **`mechanisms`**: `[MCAR, MAR, MNAR]`
- **`levels`**: `[0.10, 0.20, 0.30, 0.40, 0.50]`
- **`imputation_methods`**: `[mean, knn, regression, random_forest]`
- **`mar_driver`**: `long`
- **`strength`**: 1.0

## 4. Setup & Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Running Data Processing & Verification

To run the data cleaning verification script:

```bash
python3 scripts/verify_cleaning.py
```

To run the automated test suite with `pytest`:

```bash
pytest tests/test_cleaning.py -v
```

### Verification Checks Performed:
- **Row count**: Verifies exactly 21,436 unique property sale records after removing repeat sales.
- **Column schema**: Confirms exactly 17 ordered features (`price`, `log_price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `yr_built`, `lat`, `long`, `sqft_living15`, `sqft_lot15`, `renovated`).
- **Data integrity**: Verifies 0 missing values and 0 duplicate property sales/indices.
- **Renovated feature**: Validates binary indicator (`0`: 20,526, `1`: 910).
- **Train/Test split**: Confirms 80% train (17,148 rows) and 20% test (4,288 rows) disjoint partitions.
- **Raw CSV immutability**: Verifies byte-level MD5 hash and raw dimensions (21,613 rows, 21 columns) remain untouched.

## 6. Handoff to Member 2

This section describes the generated missingness datasets and provides the loader interface for downstream imputation and evaluation experiments.

### Folder Layout

All simulated datasets are located under `data/generated_missing/`:

```text
data/generated_missing/
├── ground_truth.csv.gz          # Clean baseline data (with row_index column)
├── split_indices.csv            # Copy of train/test partition indices (17,148 train / 4,288 test)
├── manifest.csv                 # Registry of all 75 conditions with seeds & missingness stats
├── MCAR/
│   ├── rate_10/rep_{0..4}.csv.gz
│   ├── rate_20/rep_{0..4}.csv.gz
│   ├── rate_30/rep_{0..4}.csv.gz
│   ├── rate_40/rep_{0..4}.csv.gz
│   └── rate_50/rep_{0..4}.csv.gz
├── MAR/
│   └── ... (same subfolder structure)
└── MNAR/
    └── ... (same subfolder structure)
```

> **Note on Data Sharing & Exact Regeneration:**
> All 75 simulation conditions, ground truth, and manifest are 100% deterministic and can be regenerated from scratch at any time by running:
> ```bash
> python3 scripts/run_sweep.py
> ```
> Large `.csv.gz` files therefore do not need to be committed to version control or transferred manually.

### Column Meanings & Missingness Schema

Each dataset file includes `row_index` as its first column (matching the row position in `clean_data.csv`), followed by the 17 clean housing features:
- **`missingness_columns`** (values removed according to mechanism & rate):
  - `sqft_living`: Living area in square feet
  - `grade`: Overall grade assigned by King County grading system
  - `bathrooms`: Number of bathrooms
  - `yr_built`: Year property was built
- **`always_observed`** (strictly protected; 0 NaNs under all conditions):
  - `price`: Sale price (primary prediction target)
  - `log_price`: Natural logarithm of price
  - `lat`: Property latitude
  - `long`: Property longitude (MAR driver feature)
- **Other observed features**:
  - `bedrooms`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_living15`, `sqft_lot15`, `renovated`

> **CRITICAL RULE:**
> `ground_truth.csv.gz` (and the `truth` DataFrame) is strictly for **SCORING ONLY** (calculating imputation error metrics like RMSE/MAE). It **must never** be used as an input to any imputer or downstream model.

### How to Use the Loader (`src/handoff.py`)

Member 2 can list all available conditions and load any condition directly with:

```python
from src.handoff import list_conditions, load_condition

# 1. View all 75 condition configurations and metadata
manifest_df = list_conditions()

# 2. Load a specific condition (e.g., MAR at 30% missingness, repeat 2)
data = load_condition(mechanism="MAR", level=0.30, repeat=2)

corrupted = data["corrupted"]   # pd.DataFrame (21,436 x 17), NaNs in missingness columns, index = row_index
truth = data["truth"]           # pd.DataFrame (21,436 x 17), SCORING ONLY, index = row_index
mask = data["mask"]             # pd.DataFrame (21,436 x 4), boolean True where corrupted is NaN
train_idx = data["train_idx"]   # np.ndarray of 17,148 row indices for train partition
test_idx = data["test_idx"]     # np.ndarray of 4,288 row indices for test partition

# Access train and test partitions:
train_corrupted = corrupted.loc[train_idx]
test_corrupted = corrupted.loc[test_idx]

# Train your imputer strictly on train_corrupted:
# imputer.fit(train_corrupted)
```

## 7. Figures

- `fig_m1_missingness_heatmap.png`: 4-column missing mask heatmap for 300 train rows sorted by longitude across mechanisms.
- `fig_m2_missing_prob_by_driver.png`: Missingness probability of sqft_living across deciles of longitude and itself.
- `fig_m3_complete_rows_vs_level.png`: Train complete-row retention fraction vs missingness level with theoretical curve.
- `fig_m4_sqft_bias_vs_level.png`: Percent bias of observed sqft_living mean vs level across mechanisms.
- `fig_s1_zdiff_vs_level.png`: Mean absolute z-score difference vs level for all 15 and 4 affected predictors.
- `fig_s2_ci_coverage_vs_level.png`: Share of 95% confidence intervals covering reference parameters vs level.
- `fig_s3_bias_heatmap.png`: Heatmap of mean signed z-differences for all 15 predictors across 15 conditions.
- `fig_s4_coef_forest.png`: Forest plot of affected predictor coefficients on reference-SE scale at 30% missingness.

