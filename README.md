# TON_IoT Network — Intrusion Detection & Attack-Type Prediction

A reproducible machine-learning study on the **TON_IoT network** dataset (Zeek flow logs from an IoT/IIoT testbed).
Notebooks and code comments are in **French** (team language). This README is the English entry point.

| Objective | Target | Models |
|---|---|---|
| **1. Anomaly detection** | `label` (0 = normal, 1 = attack) | Random Forest Classifier, SVM (linear vs RBF) |
| **2. Attack-type prediction** | `type` (10 classes), regressed on an ordinal code | Random Forest Regressor, XGBoost Regressor (+ reference classifiers) |
| **Extra: unsupervised experiment** | none during training | Isolation Forest |

Every score in this README comes from the result files in `reports/results/*.json`, which the notebooks generate.
None was typed by hand.

---

## 1. Headline results (test set, read once)

**Table A — anomaly detection** (full table: [reports/table_A_anomaly_detection.md](reports/table_A_anomaly_detection.md))

| Model | Train rows | F1 macro | Specificity | Recall | ROC-AUC | MCC | Fit (s) | Predict (ms / 1k rows) |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 65,007 | 0.9977 | 0.9971 | 0.9988 | 1.0000 | 0.9955 | 6.4 | 11.4 |
| SVM RBF | 25,000 | 0.9950 | 0.9936 | 0.9973 | 0.9989 | 0.9899 | 2.6 | 72.0 |
| Isolation Forest A (unsupervised) | 65,007 | 0.2040 | 0.5843 | 0.0959 | 0.1188 | −0.3580 | 2.5 | 26.3 |
| Isolation Forest B (semi-supervised, normal-only train) | 15,122 | 0.8792 | 0.7859 | 0.9557 | 0.9156 | 0.7592 | 1.2 | 28.9 |

**Table B — attack-type prediction** (full table: [reports/table_B_attack_type.md](reports/table_B_attack_type.md))

| Model | MAE | RMSE | R² | Accuracy | F1 macro | F1 weighted | Fit (s) | Predict (ms / 1k rows) |
|---|---|---|---|---|---|---|---|---|
| Random Forest Regressor | 0.0596 | 0.3812 | 0.9870 | 0.9721 | 0.9428 | 0.9728 | 34.7 | 11.2 |
| XGBoost Regressor | 0.0822 | 0.3764 | 0.9874 | 0.9694 | 0.9377 | 0.9699 | 17.4 | 8.4 |
| *RF Classifier (reference, not requested)* | – | – | – | 0.9810 | 0.9574 | 0.9814 | 37.1 | 17.6 |
| *XGB Classifier (reference, not requested)* | – | – | – | 0.9826 | 0.9605 | 0.9832 | 43.0 | 17.4 |

### How to read these numbers
- **The detection task is easy on this testbed, but not because of a single shortcut.**
  The robustness check ([reports/table_C_robustness.md](reports/table_C_robustness.md), validation set only) shows:
  - Removing every port, service, protocol, DNS, HTTP and SSL feature lowers F1 macro by only 0.0012.
  - With only volume and timing features, the score is still 0.992.
  - A single rule (`conn_state == S0`) already reaches F1 macro 0.81, and a depth-3 tree reaches 0.96.

  Scripted attacks leave very regular signatures. Read the ~0.997 scores as an **upper bound for this network**,
  not as expected performance on another network.
- **RF vs SVM:** the gap (+0.003 F1 macro) comes from one split and has no confidence interval, so it does not justify a
  strong preference. The RF is faster at inference; the RBF SVM has to be trained on a subsample (O(n²) cost).
- **Regressing a nominal target costs about 0.015–0.023 F1 macro** compared with a classifier using the same features and
  settings. The requested regressors are kept; the classifiers are reported only as a yardstick.
  `mitm` is the hardest class for every model (F1 0.67–0.74; 198 test rows).
- **Isolation Forest A ranks the traffic backwards** (ROC-AUC 0.12), because attacks make up 77 % of the traffic and the
  algorithm assumes anomalies are rare. This is a real finding of the study, not a bug. Variant B, trained only on
  normal traffic, recovers (ROC-AUC 0.92), but it uses the labels to filter the training set and gives 21 % false alarms.

---

## 2. What was wrong in the original project, and what changed

The full audit is in [reports/AUDIT.md](reports/AUDIT.md). The most important fixes:

1. **Train/test leakage.** De-duplication ran on more columns than the final features, so duplicates came back after
   feature selection: 4.1 % of test rows were exact copies of train rows, and **43 %** matched once the ephemeral
   `src_port` was ignored. Fixes:
   - the raw `src_port` is dropped (its bucket is kept);
   - de-duplication now runs on the final candidate feature space;
   - a cross-split pass drops valid/test rows that duplicate train;
   - an assertion checks that the overlap is 0.

   Dataset size: 211,043 raw rows → 108,346 unique flows → **65,007 / 21,482 / 21,438** (train / valid / test).
2. **Topology shortcuts removed:** `src_internal` and `dst_internal` were derived from the IP addresses that had been dropped.
3. **The test set was used to pick the probability calibrator.** The calibrator is now chosen on half of the validation set
   and refit on all of it.
4. **The ordinal encoding is now justified by data.** The documented order lost to a random permutation. Notebook 01
   compares, in CV, the documented order, the alphabetical order, 5 random permutations and an LDA-derived order. The
   LDA order wins under an explicit rule and is stored in `config.json`.
5. **Balancing is chosen with a parsimony rule** (the simplest strategy unless another is better by more than one CV std),
   and **SMOTENC** was added.
6. **Complete metrics**, with regression metrics kept separate from classification metrics; prediction time and
   training-set size are reported; hand-typed scores were removed; duplicate files were cleaned up;
   `requirements.txt`, tests and a pipeline runner were added.

---

## 3. Repository layout

```
├── data/
│   ├── train_test_network.csv        raw TON_IoT network file (input)
│   └── processed/                    X/y train|valid|test (parquet) + config.json (features, type order, removed columns)
├── notebooks/
│   ├── 01_prep.ipynb                 quality, EDA, features, dedup, split, selection, balancing preview, encoding study
│   ├── 02_anomaly_random_forest.ipynb
│   ├── 03_anomaly_svm.ipynb          linear vs RBF (+ Nyström approximation)
│   ├── 04_type_random_forest_regressor.ipynb   (+ reference RF classifier)
│   ├── 05_type_xgboost_regressor.ipynb         (+ reference XGB classifier)
│   ├── 06_isolation_forest.ipynb     unsupervised experiment (variants A and B)
│   └── 07_comparison.ipynb           Tables A/B, robustness check, final synthesis
├── src/
│   ├── ml_utils.py                   data loading, features, pipelines, balancing, metrics, leakage guards, results I/O
│   └── ml_plots.py                   shared plots (re-exported by ml_utils)
├── scripts/run_pipeline.py           runs every notebook in order
├── tests/                            pytest unit tests for src/
├── models/                           trained pipelines (*.joblib)
└── reports/
    ├── AUDIT.md                      technical audit of the original project
    ├── table_A_anomaly_detection.{csv,md}
    ├── table_B_attack_type.{csv,md}
    ├── table_C_robustness.{csv,md}
    ├── model_comparison.{csv,md}     A + B in one file
    ├── results/*.json                full metrics of every notebook (the source of every table)
    └── figures/*.png                 every figure, prefixed by notebook number
```

---

## 4. Methodology

### 4.1 Data preparation (`01_prep`)
- **Inspection:** dtypes, memory use, invalid values (negative counters, out-of-range ports, label/type consistency).
  `label == (type != "normal")` is asserted, so `type` is never used as a feature for `label`.
- **Missing values:** Zeek writes `-` for "absent", which is converted to `NaN` (a naive check reports 0 missing values).
  Columns with > 99 % missing values are dropped. `has_http` / `has_ssl` / `has_weird` keep the "protocol present" signal.
- **Feature engineering, row by row** (so no statistics can leak between rows): total bytes and packets, ratios, rates,
  IP overhead, port buckets, a same-port flag.
- **Columns dropped without looking at the target:** IP addresses, raw `src_port`, near-empty columns, and high-cardinality
  text (`dns_query`).
- **De-duplication** in the candidate feature space. Rows with identical features but different types (conflicts) are removed.
- **Outliers** are kept, because extreme volumes *are* the signal of DoS and exfiltration. Instead, `log1p` is applied
  inside the pipeline.
- **Split:** stratified by `type`, 60/20/20, then the cross-split de-duplication described in section 2.
- **Feature selection on train only:** constant columns, Spearman |ρ| > 0.97, and mutual information ≈ 0 with both targets.
  The result is 35 features (26 numeric, 9 categorical).
- **Encoding:** median imputation; categorical `NaN` becomes the category `missing` (absence is informative); one-hot with
  rare categories grouped. Standardisation is applied **only** for the SVM; trees do not need it.

### 4.2 Class imbalance
- **Binary target:** *normal* is the **minority** (≈ 23 %), so the false-alarm rate (specificity) is a key metric.
- **Type target:** `mitm` is rare (622 rows in train).
- **Strategies compared** in cross-validation: none, class/sample weights, random oversampling, SMOTE, SMOTENC.
  - Every sampler sits **inside** the `imblearn` pipeline, so it is refit on each training fold and never touches
    validation or test.
  - SMOTENC interpolates the numeric features and samples the categorical ones *before* one-hot encoding.
- **Selection rule:** keep the simplest strategy unless another one beats it by more than one CV standard deviation.
  - Detection keeps **no balancing**: all strategies are within 0.0005 F1 macro, which is below the noise level.
  - The type regressors keep **SMOTE**: it adds about +0.016 F1 macro over no balancing, and SMOTENC is within one std of it.

### 4.3 Models and optimisation
| Model | Tuning | Notes |
|---|---|---|
| RF classifier | RandomizedSearchCV (20 iterations, 3 folds, F1 macro): `n_estimators`, `max_depth`, `min_samples_*`, `max_features` | Gini and permutation importances |
| SVM | Linear: C on a log grid (smallest C within 1 std of the best). RBF: C × γ grid on 10k rows | RBF trained on 25k stratified rows (O(n²) cost); Nyström approximation also compared |
| RF regressor | Random search ranked by RMSE (the model's native loss) | Learning curve by number of trees |
| XGBoost regressor | Learning rate, depth, subsample, colsample, min_child_weight, λ, α, γ | Number of trees set by early stopping on a 10 % **train** hold-out (288 trees) |
| Isolation Forest | No label-based tuning | Stability across 5 seeds (Spearman ρ ≥ 0.99) |

Tuning and CV use stratified subsamples of the training set; every final model is refit on the full training set
(except the RBF SVM, as noted).

### 4.4 Decisions made on validation, never on test
- **Detection:** decision threshold (maximising F1 macro) and probability calibrator (Platt, isotonic or raw, chosen by
  Brier score on a held-out half of the validation set).
- **Type:** the 9 cut-points that turn the continuous regression output into a class, optimised for F1 macro by coordinate
  descent. Plain rounding is also reported.

### 4.5 Metrics
- **Detection:** accuracy, precision, recall (sensitivity), specificity, F1, F1 macro, F1 weighted, ROC-AUC, PR-AUC, MCC,
  FPR/FNR, confusion matrices, ROC and PR curves, calibration (Brier score and reliability plot), error rate per attack type.
- **Type, regression metrics** on the raw continuous output: MAE, RMSE, R².
- **Type, classification metrics** after decoding: accuracy, precision, recall and F1 (macro and weighted), balanced
  accuracy, MCC, per-class report, normalised confusion matrix.
  "Within ±1 class" is shown, but it carries little meaning because the class codes are arbitrary.

### 4.6 Why regress a categorical target, and how to read it
The task requires regressors. `type` is **nominal**, so any integer code imposes a distance between classes that does not
exist. To limit the damage, the code order is chosen from the data: the LDA order computed on train beat the documented
order and the random permutations in CV, by a margin about equal to one standard deviation of the permutations. The
continuous output is then decoded with cut-points chosen on validation. **R² ≈ 0.987 should not be read as "98.7 % of the
types explained"**: it measures closeness in an arbitrary code space. The F1 scores and the per-class reports are the
meaningful results, and the reference classifiers show what a model suited to the target achieves.

---

## 5. Reproducing the results

Requires Python 3.12 (tested on Windows).

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
```bash
.venv/Scripts/python scripts/run_pipeline.py
```
```bash
.venv/Scripts/python -m pytest -q
```

- On Linux/macOS, use `.venv/bin/python` instead of `.venv/Scripts/python`.
- `run_pipeline.py` runs 01 → 02 … 06 → 07 in place, logs each notebook's duration and stops at the first failure.
  The whole pipeline takes about 30 minutes on 12 cores.
- Run part of the pipeline with `--only 04 05` or `--skip-prep`.
- `random_state = 42` is used everywhere, and the versions in `requirements.txt` are pinned: the saved `joblib` pipelines
  only load with the same scikit-learn and xgboost versions.

Loading a saved model:
```python
import joblib, sys
sys.path.append("src"); import ml_utils as U
rf = joblib.load(U.MODEL_DIR / "rf_anomaly.joblib")          # complete pipeline: preprocessing + model
X_test, y_test = U.load_split("test")
proba = rf.predict_proba(X_test)[:, 1]
thr = U.load_results()["02_rf_anomaly"]["threshold"]           # threshold calibrated on validation
pred = (proba >= thr).astype(int)
```

---

## 6. Limitations
- **One testbed, one random split.** Train, validation and test share the same attack campaigns and hosts. A split by
  campaign, by time period, or a second dataset (for example the TON_IoT time-stamped logs) would be needed to claim
  generalisation.
- **Very separable data.** Shallow rules already explain most of the detection performance (see the robustness check).
- **No confidence intervals.** Gaps of a few thousandths between models should not be read as meaningful.
- **Nominal target treated as regression.** The cost is measured against the reference classifiers.
- **Isolation Forest is unsuited when attacks are the majority.** Its unsupervised use here is a negative result.
