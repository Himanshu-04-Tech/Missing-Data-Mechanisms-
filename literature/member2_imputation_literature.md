# Literature Notes: Imputation Methods (Member 2)

These notes support the choice and evaluation of the four imputation methods used in this project (mean, KNN, regression, random forest).

Each source below was checked against a publisher, journal, PubMed or university repository record (authors, year, journal, volume, pages). DOIs are given only where a record confirmed them; "DOI to confirm" means look it up on the journal page before the final report. The notes only state what the abstracts or publisher records say. Where the full text was not read, this is noted. If a note is used in the report, re-read the original paper first.

---

## 1. Foundations: why the missingness mechanism matters

### 1.1 Rubin (1976): Inference and missing data
- **Citation:** Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592. (DOI to confirm: online records disagree.)
- **What it does:** It introduces the concepts now called MCAR, MAR and "ignorable" missingness. It states the conditions under which the process that causes missing data can be ignored when making inferences.
- **Key point:** Ignoring the missing-data process is only safe under specific conditions, roughly when data are missing at random.
- **Relevance to us:** This is the theoretical basis for Member 1's MCAR, MAR and MNAR design. It also explains why our MNAR results are worst for every imputation method: under MNAR, the observed data alone cannot tell the imputer what was hidden.

### 1.2 Schafer & Graham (2002): Missing data, our view of the state of the art
- **Citation:** Schafer, J. L., & Graham, J. W. (2002). Missing data: Our view of the state of the art. *Psychological Methods*, 7(2), 147–177. (DOI to confirm.)
- **What it does:** It reviews missing-data methods, clarifies the MAR concept, and recommends maximum likelihood and Bayesian multiple imputation. It summarizes the evidence against older ad-hoc methods and, with few exceptions, discourages their use.
- **Relevance to us:** This supports treating **mean imputation as a baseline only**, not as a recommended method. Our results show why: mean imputation fills every gap with one constant, so the spread of the filled values is zero (`sd_ratio` = 0), and its error is the worst in every condition.

### 1.3 Little & Rubin: Statistical analysis with missing data (book)
- **Citation:** Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
- **Use in the report:** A standard textbook reference for the definitions of MCAR, MAR and MNAR and for single vs multiple imputation. Cite specific chapters only after checking them.

---

## 2. The four methods

### 2.1 KNN imputation: Troyanskaya et al. (2001)
- **Citation:** Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., Botstein, D., & Altman, R. B. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6), 520–525. https://doi.org/10.1093/bioinformatics/17.6.520
- **What it does:** It compares three ways to fill missing values in gene-expression data: weighted K-nearest neighbours (KNNimpute), an SVD-based method, and row averages. It also tests filling with zeros. Three yeast datasets are used.
- **Key finding (abstract):** KNNimpute was more robust than the SVD method, and both clearly beat row-average and zero filling.
- **Relevance to us:** This is the classic reference for KNN imputation and for "neighbours beat averages". Our results agree: KNN roughly halves the error of mean imputation under MCAR and MAR (NRMSE ≈ 0.56–0.58 vs ≈ 1.0).
- **Difference from our setup:** Their KNN weights neighbours by distance; ours gives the 5 neighbours equal weight (the scikit-learn default). Their data is gene expression; ours is house sales.

### 2.2 Regression / chained-equation imputation: van Buuren & Groothuis-Oudshoorn (2011)
- **Citation:** van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–67. https://doi.org/10.18637/jss.v045.i03
- **What it does:** It describes MICE (multivariate imputation by chained equations). Each incomplete variable is imputed from the others with its own model, cycling through the variables repeatedly.
- **Relevance to us:** Our `regression` method uses the same chained-equation idea. It is implemented in Python with scikit-learn's `IterativeImputer`, which its documentation describes as inspired by MICE. (The paper is about an R package; we cite it for the method, not the software. R is not used in this project.)
- **Important difference:** MICE is normally run as **multiple imputation**: several filled datasets with random noise added, whose results are then pooled. We run **one deterministic imputation** (no noise). This is the main reason our fills are shrunk toward the average and why standard errors on our filled data will tend to be too small (see README section 8, Known limitations).

### 2.3 Random forest imputation: Stekhoven & Bühlmann (2012)
- **Citation:** Stekhoven, D. J., & Bühlmann, P. (2012). MissForest: Non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118. https://doi.org/10.1093/bioinformatics/btr597
- **What it does:** It proposes missForest, an iterative imputation method that predicts each incomplete variable with a random forest, repeating until the fills stabilize. It handles continuous and categorical variables together.
- **Key finding (abstract):** On biological datasets with 10–30% of values removed, missForest outperformed the other imputation methods compared, especially where complex interactions and non-linear relations were suspected.
- **Relevance to us:** Our `random_forest` method follows the same scheme (iterative imputation with a random forest). It also had the lowest imputation error in every mechanism (NRMSE 0.49 / 0.51 / 0.72 for MCAR / MAR / MNAR).
- **Difference from our setup:** We use scikit-learn's `IterativeImputer` with a `RandomForestRegressor`, not the original R package. Our variables are all numeric, so the mixed-type feature is not used. The settings (50 trees, 5 iterations) were chosen for running time and were not tuned.

### 2.4 Random forest inside MICE vs parametric models: Shah et al. (2014)
- **Citation:** Shah, A. D., Bartlett, J. W., Carpenter, J., Nicholas, O., & Hemingway, H. (2014). Comparison of random forest and parametric imputation models for imputing missing data using MICE: A CALIBER study. *American Journal of Epidemiology*, 179(6), 764–774. https://doi.org/10.1093/aje/kwt312
- **What it does:** It compares standard parametric MICE with random-forest MICE in simulation studies based on real patient records. Importantly, it judges the methods by the **statistical estimates** they produce (hazard ratios, bias, confidence-interval coverage), not just by how close the filled values are.
- **Key findings (abstract):** When the relationships were linear, both gave unbiased estimates and random forest gave narrower intervals. When a relationship was non-linear, random forest gave less biased estimates and better interval coverage.
- **Relevance to us:** This is the closest source to our research question. It shows that imputation methods should be compared on **statistical recovery** (coefficients, intervals), not only on imputation error. That is what our statistical track does with the 300 imputed files.
- **Difference from our setup:** They use multiple imputation with random draws; ours is single and deterministic.

---

## 3. Using the outcome in the imputation model

### 3.1 Moons et al. (2006)
- **Citation:** Moons, K. G. M., Donders, R. A. R. T., Stijnen, T., & Harrell, F. E. (2006). Using the outcome for imputation of missing predictor values was preferred. *Journal of Clinical Epidemiology*, 59(10), 1092–1101. (DOI to confirm.)
- **What it does:** It tests, by simulation, whether the outcome variable should be included when imputing missing predictors.
- **Key finding (abstract):** With **multiple imputation**, including the outcome gave regression coefficients close to the truth. Leaving it out gave biased, underestimated coefficients, under both MCAR and MAR.
- **Relevance to us:** This is why we produced the second variant, `data/imputed_with_target/`, where `log_price` is an imputer input. It is used for the statistical track only, because using the target to fill predictors would leak the answer into the prediction track.
- **Caution:** Their result is for **multiple** imputation with random draws. Our imputers are deterministic, so their conclusion cannot be assumed to carry over. The statistical track should compare `data/imputed/` with `data/imputed_with_target/` and report what actually happens.

---

## 4. Benchmarks that combine imputation quality and downstream performance

### 4.1 Jäger, Allhorn & Bießmann (2021): A benchmark for data imputation methods
- **Citation:** Jäger, S., Allhorn, A., & Bießmann, F. (2021). A benchmark for data imputation methods. *Frontiers in Big Data*, 4, 693674. https://doi.org/10.3389/fdata.2021.693674
- **What it does:** It benchmarks classical machine-learning imputation methods (including KNN and random forest) against deep-learning methods. Missingness is introduced synthetically on many datasets under MCAR, MAR and MNAR. The methods are scored on two things: **imputation quality** and the effect on a **downstream ML model**.
- **Relevance to us:** Our design is similar: synthetic MCAR/MAR/MNAR at several levels, imputation quality (Member 2), and downstream prediction (Member 3). Our project adds a third dimension, **statistical inference** (coefficients, standard errors, confidence intervals, p-values), and asks whether it agrees with the other two.
- **Note:** Details beyond the abstract (exact number of datasets and methods) should be checked in the full text before being quoted.

---

## 5. Software

- **scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. `SimpleImputer`, `KNNImputer` and `IterativeImputer` all come from this library.

---

## 6. How the literature supports our design choices

| Our choice | Supported by | Our own result |
|---|---|---|
| Mean imputation only as a baseline | Schafer & Graham (2002) | Worst error everywhere; spread of fills = 0 |
| KNN as a similarity-based method | Troyanskaya et al. (2001) | Roughly half the error of mean under MCAR and MAR |
| Iterative regression (chained equations) | van Buuren & Groothuis-Oudshoorn (2011) | Similar error to KNN, about 30× faster |
| Iterative random forest | Stekhoven & Bühlmann (2012); Shah et al. (2014) | Lowest error under all mechanisms |
| Judging imputation by statistical estimates, not just error | Shah et al. (2014); Moons et al. (2006) | Done by the statistical track on our files |
| Extra variant with the outcome as an input | Moons et al. (2006) | Lower imputation error; effect on inference still to be measured |
| MCAR / MAR / MNAR comparison | Rubin (1976); Jäger et al. (2021) | All methods are clearly worst under MNAR |

## 7. Gaps these sources leave for our project

These are stated as **opportunities based on the sources above**, not as claims that nobody has studied them:

1. Most of the sources judge imputation **either** by imputation error or downstream prediction (Troyanskaya; Stekhoven & Bühlmann; Jäger et al.) **or** by statistical estimates (Moons et al.; Shah et al.). Our project measures all three on the same imputed datasets.
2. The outcome-inclusion advice (Moons et al.) is shown for multiple imputation. Whether it helps or hurts with **single deterministic** imputers is something our two variants can test directly.
3. Our setting is housing data with an MNAR mechanism where large values are hidden. The sources above mostly use biomedical data.

## Sources checked

- Rubin (1976): [Harvard DASH](https://dash.harvard.edu/entities/publication/73120378-8764-6bd4-e053-0100007fdf3b)
- Schafer & Graham (2002): [Penn State record](https://pure.psu.edu/en/publications/missing-data-our-view-of-the-state-of-the-art/)
- Troyanskaya et al. (2001): [Princeton record](https://collaborate.princeton.edu/en/publications/missing-value-estimation-methods-for-dna-microarrays/)
- van Buuren & Groothuis-Oudshoorn (2011): [Journal of Statistical Software](https://www.jstatsoft.org/article/view/v045i03)
- Stekhoven & Bühlmann (2012): [PubMed](https://pubmed.ncbi.nlm.nih.gov/22039212/)
- Shah et al. (2014): [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/1427772/)
- Moons et al. (2006): [UMC Utrecht record](https://researchinformation.umcutrecht.nl/en/publications/using-the-outcome-for-imputation-of-missing-predictor-values-was-/)
- Jäger et al. (2021): [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8297389/)
