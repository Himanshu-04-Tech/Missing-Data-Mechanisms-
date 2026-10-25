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
│   ├── generated_missing/          # Missingness simulation datasets (Member 1)
│   ├── imputed/                    # Imputed datasets, no target used (Member 2)
│   └── imputed_with_target/        # Imputed datasets, log_price used (Member 2, STATISTICS ONLY)
├── src/                            # Modular pipeline code
│   ├── __init__.py
│   ├── data_processing.py          # Ordered loading, deduplication, cleaning & splitting
│   ├── missingness.py              # MCAR, MAR, MNAR generation functions
│   ├── imputation.py               # Mean, KNN, regression, random-forest imputation + scoring
│   ├── statistics.py               # Statistical distributions, metrics, and bias calculations
│   └── evaluation.py               # Downstream evaluation metrics (RMSE, MAE, R2)
├── literature/                     # Literature notes (member2_imputation_literature.md)
├── report/                         # Report material (member2_explainer_and_viva.md)
├── notebooks/                      # Exploratory and analytical Jupyter notebooks
├── results/
│   ├── tables/                     # Tabular experiment summaries and LaTeX/CSV exports
│   ├── figures/                    # Visualizations, diagnostic plots, and charts
│   └── experiment_results/         # Raw experiment run logs and metrics
├── scripts/
│   ├── verify_cleaning.py          # Data cleaning verification & sanity check script
│   ├── run_imputation.py           # Impute, check, save and score all conditions
│   ├── imputation_progress.py      # Live progress view of imputation runs
│   ├── check_knn_k.py              # Sensitivity check for the KNN neighbour count
│   └── make_imputation_figures.py  # Imputation figures + method comparison table
└── tests/
    ├── test_cleaning.py            # Automated pytest suite for data pipeline
    └── test_imputation.py          # Automated pytest suite for imputation
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
- `fig_i1_nrmse_vs_level.png`: Imputation error (NRMSE on the filled cells) vs missingness level, per mechanism and imputation method.
- `fig_i2_sd_ratio_vs_level.png`: Spread kept by the fills (SD of filled values / SD of true values) vs missingness level.

## 8. Imputation (Member 2) and Handoff to Member 3

Member 2 filled the missing values of all 75 corrupted datasets with 4 methods and checked and scored every result.

### What was done

| Method | How a missing value is filled | Settings |
|---|---|---|
| `mean` | Train-set mean of the column (baseline) | — |
| `knn` | Average of the 5 most similar **train** houses, on standardized columns | k = 5 |
| `regression` | Iterative regression imputation (`IterativeImputer`, BayesianRidge); each missing column predicted from the others until fills stabilize | max_iter = 10, fills clipped to train range |
| `random_forest` | Same iterative scheme with a random forest (missForest-style) | 50 trees, min_samples_leaf = 3, max_iter = 5, fills clipped to train range |

All four are deterministic (fixed seed `base_seed` = 42): rerunning gives identical files.

### Leakage rules (enforced in `src/imputation.py` and tested in `tests/test_imputation.py`)

- Every imputer is **fit on train rows only**, then fills train and test rows.
- In `data/imputed/`, **`price` and `log_price` are never imputer inputs**.
- **Ground truth is never an imputer input**; it is used only to score the fills.
- **Only NaN cells are filled**; every observed cell is byte-identical to the corrupted file.

### Two variants

| Folder | Imputer inputs | May be used for |
|---|---|---|
| `data/imputed/` | all columns except `price`, `log_price` | **prediction** and statistics |
| `data/imputed_with_target/` | the same + `log_price` | **statistics only** |

> **CRITICAL RULE:** `data/imputed_with_target/` must **never** be used for prediction. Its fills were made using the target, so a model trained on it would be using information it would not have at prediction time.

The with-target variant exists because the missing-data literature recommends including the outcome in the imputation model when the goal is regression inference. Whether that holds for these deterministic imputers is something the statistical track can check by comparing the two folders.

### Folder layout (300 files per variant)

```text
data/imputed/{method}/{mechanism}/rate_{10..50}/rep_{0..4}.csv.gz
    method    = mean | knn | regression | random_forest
    mechanism = MCAR | MAR | MNAR
```

Each file has exactly the same format as Member 1's corrupted files: `row_index` first, then the 17 columns in the same order, 21,436 rows, **0 NaN**.

### How Member 3 loads a file

```python
import pandas as pd
from src.handoff import load_condition
from src.m3_common import imputed_path

imputed = pd.read_csv(imputed_path("knn", "MAR", 0.30, 2), index_col="row_index")
split = load_condition("MAR", 0.30, 2)            # for train_idx / test_idx

train = imputed.loc[split["train_idx"]]
test = imputed.loc[split["test_idx"]]
```

### Checks performed

- `scripts/run_imputation.py` checks every file before saving it: same rows and columns, no NaN, no infinite values, 0 observed cells changed.
- Member 3's validator (`src/m3_validate_imputed.run_validation`) was run on `data/imputed/`: **300 / 300 PASS**.
- `pytest tests/test_imputation.py`: 19 tests (fills every NaN, observed cells untouched, no train/test leakage, target ignored by default, determinism, fills within train range, scoring sanity).

### Imputation quality (Table 3)

From `results/tables/imputation_method_comparison.csv`: averages over the 4 columns, 5 levels and 5 repeats, on the filled cells only.

| Method | NRMSE MCAR | NRMSE MAR | NRMSE MNAR | Spread kept MCAR / MAR / MNAR | Seconds per run |
|---|---:|---:|---:|---:|---:|
| mean | 1.00 | 1.03 | 1.46 | 0 / 0 / 0 | 0.03 |
| knn | 0.56 | 0.58 | 0.83 | 0.84 / 0.82 / 0.70 | ~19 |
| regression | 0.59 | 0.58 | 0.84 | 0.84 / 0.83 / 0.69 | ~0.6 |
| random_forest | 0.49 | 0.51 | 0.72 | 0.89 / 0.87 / 0.74 | ~59 |

- **NRMSE** = RMSE / SD of the true column. 1 means "no better than guessing the average"; lower is better.
- **Spread kept** = SD of the filled values / SD of the true values at those cells. Below 1 means the fills are pulled toward the average; mean imputation fills a single constant, so its spread is 0.

Observations (imputation error only, not yet statistical or predictive recovery):

- On imputation error, random forest < KNN ≈ regression < mean under all three mechanisms.
- Every method is clearly worse under **MNAR**. All methods fill values that are **too low** there: at 30% MNAR, `sqft_living` fills are on average 340–960 sq ft below the true values. The hidden values are the large ones, and no observed column fully reveals that.
- All model-based methods **shrink the spread** of the filled values (15–30% less than the true spread). Shrunken fills can make relationships look stronger or weaker than they really are. That is a question for the statistical track, not something imputation error shows.
- Error on train rows and test rows is almost the same (for example random forest 0.575 vs 0.581), so the imputers do not overfit the train rows.

**With-target variant** (`data/imputed_with_target/`, also 300 / 300 PASS on Member 3's validator): adding `log_price` as an input lowers imputation error a little for every model-based method. Mean NRMSE MCAR / MAR / MNAR is knn 0.53 / 0.55 / 0.79, regression 0.56 / 0.55 / 0.78, random_forest 0.47 / 0.48 / 0.68; mean imputation is unchanged because it ignores other columns. Lower imputation error does **not** by itself mean better statistical recovery: fills built from `log_price` can strengthen the price relationship artificially. The statistical track should compare both folders.

### Hyperparameter check (KNN k)

`scripts/check_knn_k.py` → `results/tables/knn_k_check.csv` (30% missingness, repeat 0, scored on **train** cells only). Mean NRMSE for k = 3 / 5 / 10 / 20: MCAR 0.583 / 0.560 / 0.546 / 0.544, MAR 0.603 / 0.584 / 0.575 / 0.573, MNAR 0.832 / 0.824 / 0.827 / 0.839. Larger k helps slightly under MCAR and MAR (about 2–3%) but not under MNAR, where k = 5 is best. k = 5 (the scikit-learn default) was kept: no single k wins everywhere and the differences are small.

### Output files

| File | Content |
|---|---|
| `results/experiment_results/imputation_scores.csv` | One row per method × mechanism × level × repeat × column × split (train / test / all): rmse, mae, r2, nrmse, bias, sd_ratio, seconds |
| `results/tables/imputation_error_summary.csv` | The same, averaged over the 5 repeats (mean and sd) |
| `results/tables/imputation_method_comparison.csv` | Table 3 above |
| `results/tables/knn_k_check.csv` | KNN k sensitivity check |
| `*_with_target.csv` versions | The same tables for `data/imputed_with_target/` |

### Reproducing

```bash
python scripts/run_imputation.py                 # all 4 methods -> data/imputed/   (~1.5 h, mostly random forest)
python scripts/run_imputation.py --use-target    # all 4 methods -> data/imputed_with_target/
python scripts/run_imputation.py --methods knn   # one method only
python scripts/imputation_progress.py            # live progress view while it runs
python scripts/make_imputation_figures.py
python scripts/check_knn_k.py
pytest tests/test_imputation.py -v
```

The imputed folders are large (about 230 MB and 180 MB) but fully reproducible with the commands above, so they do not need to be shared by hand.

### Supporting documents

- `literature/member2_imputation_literature.md`: literature notes on the imputation methods, with checked citations and how they support each design choice.
- `report/member2_explainer_and_viva.md`: plain-language summary of the imputation work, design decisions, limitations and viva questions with answers.

### Known limitations

- All four methods are **single, deterministic** imputations: each missing cell gets one best guess, with no random noise. Standard errors computed on these datasets treat filled values as if they were observed, so they will tend to be too small. Multiple or stochastic imputation would address this but is not part of the current design.
- `grade`, `bathrooms` and `yr_built` fills are **not rounded** to valid values (for example grade 7.66). This is standard for these methods; rounding would add its own error.
- No method can recover what MNAR hides; the MNAR results show how much each method is misled, not a fix.
