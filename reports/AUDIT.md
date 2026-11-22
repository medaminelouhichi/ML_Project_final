# Technical audit of the original project (before refactoring)

Scope: `src/ml_utils.py`, notebooks `01`–`06`, `reports/results/*.json`, `data/processed/*`, repository layout.
Each finding is tagged **[kept]**, **[fixed]**, **[added]** or **[removed]**.

## What was already sound (kept)
- Zeek's `-` placeholder is converted to `NaN` before quality control. **[kept]**
- Leakage-free pipelines: preprocessing and resampling are refit inside every CV fold (`imblearn.Pipeline`). **[kept]**
- 60/20/20 stratified split. Feature selection is computed on train only. **[kept]**
- Thresholds, ordinal cut-points and hyper-parameters are chosen on validation or CV. **[kept]**
- A single shared module (`ml_utils.py`), so all notebooks use identical data and metrics. **[kept, extended]**

## Critical: data leakage and inflated scores
1. **Train/test duplicates survived the de-duplication.** De-duplication ran on a *wider* column set than the
   final model features. Train-only feature selection then dropped columns, and rows that differed only in a
   dropped column became identical again. Measured on the original `data/processed` files:
   - **4.1 %** of test rows were exact duplicates of a train row in the final feature space;
   - **43.5 %** of test rows matched a train row once the ephemeral `src_port` was ignored.

   This memorisation is the main reason for the F1 ≈ 0.999 and ROC-AUC = 1.0000 scores. **[fixed]**
   The raw `src_port` is dropped (its bucket is kept). De-duplication runs on the final, label-free candidate
   feature space. After train-only selection, a last cross-split pass removes any valid/test row that
   duplicates a train row, and an assertion checks that the overlap is zero.
2. **Topology shortcuts derived from the removed IP addresses** (`src_internal`, `dst_internal`). They bring back
   the testbed addressing plan, which does not transfer to another network. **[removed from the features]**
3. **The test set was used to choose the probability calibrator** (`best_cal = brier.idxmin()` on the
   *test* Brier score, in notebooks 02 and 03). **[fixed]** The calibrator is now chosen on a held-out half
   of the validation set, then refit on the full validation set.

## Methodology
4. **The ordinal encoding was not supported by the data.** In notebook 04 §1, a *random* class order beat the
   documented `TYPE_ORDER` (RMSE 0.369 vs 0.448, macro F1 0.907 vs 0.882), which contradicts the "similar
   codes = similar behaviour" argument. **[fixed]** The order is now studied in 01 with CV against the documented
   order, the alphabetical order, several random permutations, and a data-driven order (LDA on train only).
   The decision rule is explicit and the chosen order is stored in `config.json`.
5. **Balancing choices were made on noise.** `idxmax` picked between strategies that differed by about one fold
   standard deviation. **[fixed]** Selection now uses a parsimony rule: keep the simplest strategy unless another
   one is better by more than one standard deviation. **SMOTENC** (needed for categorical features) was missing
   and has been **[added]**.
6. **There was no classification baseline for the type target.** **[added]** Reference classifiers are clearly
   labelled as not requested and do not replace the required regressors.
7. **Metrics were incomplete.** Specificity, macro/weighted F1, per-class precision/recall/F1 and prediction
   time were missing. **[added]** Regression metrics (on the continuous output) are now reported separately
   from classification metrics (after decoding).
8. **The comparison was not like for like.** Training times covered different training-set sizes (SVM RBF on 20k
   rows vs RF on 110k). **[fixed]** Tables now report the number of training rows and the prediction time.
9. **Hard-coded "old version" scores in notebook 06** (typed by hand, not produced by code). The "SVM linéaire"
   row actually showed the RBF model. **[removed]**

## Repository quality
10. The 30 MB dataset was committed twice (root and `data/`), and two `*.executed.ipynb` copies duplicated
    their sources. **[removed]**
11. There was no `requirements.txt`, no pinned versions (pickled models depend on the exact versions), no tests
    and no single command to rerun everything. **[added]** `requirements.txt`, `tests/`, `scripts/run_pipeline.py`.
## Findings after the fixes
- Removing the leakage lowers detection scores only slightly (RF test F1 macro 0.9977). The robustness check in
  notebook 07 (`reports/table_C_robustness.md`) shows this is **not** a port or service shortcut: with only
  volume/timing features the score is 0.992, and a depth-3 tree reaches 0.96. The testbed is genuinely very separable,
  and its scores are an upper bound for this network only.
- Isolation Forest without labels ranks the traffic backwards (ROC-AUC 0.12), because attacks are the majority (77 %).
  This is a documented negative result.

## Repository additions
12. There was no unsupervised component. **[added]** Isolation Forest experiment (notebook 06).
    The comparison notebook becomes 07.
