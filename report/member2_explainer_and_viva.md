# Member 2: Imputation Explained, with Viva Preparation

This document summarizes the imputation work in plain language, so that the team can write the report and answer questions about it. The technical details are in README section 8; the literature is in `literature/member2_imputation_literature.md`.

---

## 1. The work in one paragraph

Member 1 produced 75 datasets with holes in 4 columns (`sqft_living`, `grade`, `bathrooms`, `yr_built`): 3 mechanisms × 5 levels × 5 repeats. Member 2 filled those holes with 4 methods (mean, KNN, regression, random forest), giving **300 filled datasets**. Every method learns **only from the training rows** and **never sees the price**. A second set of 300 files, which does use `log_price`, was made for the statistics track only. Every filled value was compared with the true value to measure how good each method's guesses are. Member 3's validator accepts all 600 files.

## 2. What was done, step by step

1. **Chose 4 methods**, from simplest to most flexible:
   - **Mean:** fill with the column's average (baseline).
   - **KNN:** fill with the average of the 5 most similar houses.
   - **Regression:** predict the missing value from the other columns with a linear model, repeating until the fills settle.
   - **Random forest:** the same idea, but with a random forest, which can learn curved relationships.
2. **Prevented leakage:**
   - Every method is fitted on the 17,148 training rows only.
   - Price and log_price are not inputs (in the main variant).
   - The true values are never inputs.
   - Only empty cells are changed.
3. **Ran all 300 conditions** and checked each output file before saving it.
4. **Scored the guesses** against the true values: NRMSE (error), bias (too high or too low), and spread kept.
5. **Checked a setting:** KNN with k = 3, 5, 10 and 20.
6. **Made a second variant** with `log_price` as an input, for statistics only.
7. **Wrote tests** (19 tests) and handed the files to Member 3.

## 3. Key results

| Method | Error (MCAR / MAR / MNAR) | Spread kept (MCAR / MAR / MNAR) | Time per dataset |
|---|---|---|---|
| Mean | 1.00 / 1.03 / 1.46 | 0 / 0 / 0 | 0.03 s |
| KNN | 0.56 / 0.58 / 0.83 | 0.84 / 0.82 / 0.70 | ~19 s |
| Regression | 0.59 / 0.58 / 0.84 | 0.84 / 0.83 / 0.69 | ~0.6 s |
| Random forest | 0.49 / 0.51 / 0.72 | 0.89 / 0.87 / 0.74 | ~59 s |

**How to read it:**
- **Error 1.0** = no better than guessing the average. Lower is better.
- **Spread 1.0** = the guesses vary as much as the real values. Below 1 = the guesses are bunched toward the middle.

**The 4 things to remember:**
1. Random forest guesses best, mean guesses worst.
2. Every method is much worse under MNAR, and all of them guess **too low** there.
3. Every method **squeezes the spread** of the values. This is the link to our research question: good guesses can still change the statistics.
4. Adding `log_price` as an input made the guesses a little better. Whether it helps the statistics is for the statistical track to measure.

## 4. Design decisions and why

| Decision | Why |
|---|---|
| Fit on train rows only | Otherwise the test rows would influence the fills, and prediction results would be too optimistic (leakage). |
| Price not used as an input | Price is the prediction target. Using it to fill the predictors leaks the answer. |
| Separate "with target" variant | The literature (Moons et al., 2006) recommends including the outcome when the goal is inference, so the statistics track can test it. It must never be used for prediction. |
| Scale columns before KNN | Otherwise `sqft_lot` (tens of thousands) would decide who counts as a "neighbour" by itself. |
| k = 5 for KNN | The default. The check showed no single k is best everywhere (k = 10 or 20 slightly better for MCAR/MAR, k = 5 best for MNAR). |
| Clip regression and random-forest fills to the train range | Prevents impossible values such as negative bathrooms. |
| No rounding of grade or bathrooms | Standard for these methods. Rounding adds its own error. Can be revisited as a team. |
| One deterministic fill per cell | Simpler and reproducible. The downside: standard errors on the filled data will look too small (see limitations). |
| Fixed seed (42) | Every run gives exactly the same files. |

## 5. Limitations (say these before the examiner does)

- **Single imputation:** each gap gets one best guess with no uncertainty attached, so later statistics treat guesses as if they were real data. Multiple imputation would fix this; it was not in the project plan.
- **MNAR cannot be solved by imputation:** the hidden values are the big ones, and nothing observed fully reveals them. Our MNAR results show how much each method is misled.
- **Random-forest settings were not tuned** (50 trees, 5 rounds), to keep running time reasonable (about 1 minute per dataset).
- **One dataset (King County housing):** results may differ on other data.

---

## 6. Viva questions and answers

### Basic

**Q1. What is imputation?**
Filling in missing values with estimated values so the dataset can be used, instead of deleting incomplete rows.

**Q2. Why not just delete rows with missing values?**
With 4 columns each 50% missing, only about 6% of rows are fully complete (Member 1's results). Deletion also biases the statistics under MAR and MNAR: Member 1's complete-case baseline shows confidence intervals covering the true value only about 40–73% of the time there.

**Q3. Why include mean imputation if it is so bad?**
It is the baseline. Without it, we couldn't say how much better the other methods actually are.

### Intermediate

**Q4. How does KNN imputation work?**
For a house with a missing value, find the 5 most similar houses in the training set (similarity measured on the other columns, after scaling), and use their average.

**Q5. Why did you scale the columns before KNN?**
Distance adds up differences across columns. Without scaling, a column measured in tens of thousands (`sqft_lot`) would swamp columns like `bathrooms`.

**Q6. What is iterative (chained-equation) imputation?**
Start with simple fills. Then, for each incomplete column, fit a model that predicts it from the others and update its fills. Repeat over all columns several times until the fills stop changing. Our regression method uses a linear model; our random-forest method uses a random forest.

**Q7. What does NRMSE mean and why use it?**
It is the root-mean-squared error divided by the column's standard deviation. This lets us compare columns with different units (square feet vs years). A value of 1 means "no better than the average".

### Methodology and leakage

**Q8. What is data leakage, and how did you prevent it?**
Leakage is when information that wouldn't be available in practice gets into the model. We fit every imputer on training rows only, never used price or log_price as inputs in the main files, and never used the true values except for scoring. Tests check that changing test rows doesn't change how training rows are filled, and that shuffling the price doesn't change the fills.

**Q9. If including the outcome is recommended, why isn't it in your main files?**
The main files feed the prediction model, whose job is to predict price. If price were used to fill the predictors, the predictors would already contain information about the answer. So the outcome is only used in the separate statistics-only variant.

**Q10. Did you fill the test rows using the test data?**
No. The imputer is fitted on training rows and then applied to the test rows. A test row's own observed values are used as input (just as they would be for a new house), but the test rows are never used for fitting.

### Statistical reasoning

**Q11. Your best method has the lowest error. Doesn't that mean it is the best for the project?**
Not necessarily. That is exactly our research question. All methods shrink the spread of the filled values, which can change regression coefficients and make standard errors look too small. Lowest imputation error does not guarantee correct statistical conclusions; that is measured by the statistical track.

**Q12. Why do all methods guess too low under MNAR?**
Under our MNAR design, larger values are more likely to be hidden. The imputer learns from the remaining (smaller) observed values, so its guesses are pulled down. No observed column fully reveals what was hidden.

**Q13. Why is mean imputation's error worst at 10% MNAR, not 50%?**
At 10%, only the most extreme (largest) values are removed, so they are furthest from the average. At 50%, many ordinary values are also removed, and those are closer to the average.

**Q14. What does single deterministic imputation do to standard errors?**
It treats guessed values as if they were measured, and the guesses have less spread than the real values. Both make the data look more certain than it is, so standard errors tend to be too small and confidence intervals too narrow.

### Critical challenges

**Q15. Why not use multiple imputation?**
It is the method recommended in the literature for inference (Schafer & Graham, 2002). Our project compares common single-imputation methods that are widely used in practice, especially in machine-learning pipelines. Seeing how they affect inference is part of our research question. Multiple imputation is a natural extension for future work.

**Q16. How do you know your results aren't just luck?**
Every condition has 5 repeats with different random missingness patterns, and we report the average and spread across repeats. The spread is small compared with the differences between methods: the standard deviation of NRMSE across repeats is at most 0.014 (median 0.004), while the gap between random forest and KNN is about 0.07 and between KNN and mean about 0.45.

**Q17. How do you know the code is correct?**
- Every output file is checked automatically before saving: no gaps left, no observed value changed.
- Member 3's independent validator passes all 600 files.
- 19 unit tests cover leakage, determinism, value ranges and scoring.
- Mean imputation gives error ≈ 1.0 under MCAR, which is the theoretically expected value.
