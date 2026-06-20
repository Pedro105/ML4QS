# ML4QS Protocol and Results

This version is intended for direct report handover. It summarizes the **final pipeline**, the **design choices we made**, the **exact experiment protocols**, and the **main findings** from the saved notebook outputs.

## 1. Current Status

The pipeline is now organized as:

1. `01_preprocessing.ipynb`
2. `02_windowing.ipynb`
3. `03_features.ipynb`
4. `04_classical_ml.ipynb`
5. `05_deeplearning.ipynb`

The main changes versus the older version are:

- outlier detection switched from the slow ML4QS LOF implementation to a faster `scikit-learn` LOF implementation
- split logic was updated so **owner/person identification** and **context classification** use different session protocols
- PCA was added as an explicit alternative representation for the classical pipeline
- a reduced engineered feature set was added by grouping highly correlated features
- the deep learning notebook was adapted to the new split logic and now compares **TCN** and **GRU**

## 2. Final Dataset and Cohort

### 2.1 Recordings kept

The cleaned manifest contains **27 recordings** in total:

- normal walking:
  - `Asena`, `Darius`, `Jun`, `Oana`, `Pedro`
  - each has `session_1`, `session_2`, `session_3`
- crowded walking:
  - `Ana`, `Asena`, `Darius`, `Jun`, `Oana`, `Pedro`
  - each has `session_1`, `session_2`

Important note:

- `session_1`, `session_2`, `session_3` are assigned by **recording date / chronological order**, not by folder name suffix.

### 2.2 Window counts after `02_windowing.ipynb`

#### Owner/person experiment windows

Participants: `Asena`, `Darius`, `Jun`, `Oana`, `Pedro`

- train: `358` windows per participant
- test: `179` windows per participant

So owner authentication and person identification use:

- train: `1790` windows total
- test: `895` windows total

#### Context experiment windows

Using the 5 participants who have both normal and crowded data:

- train normal: `895`
- train crowded: `887`
- test normal: `895`
- test crowded: `891`

This is close to balanced, with small differences caused by recording length / valid window count.

## 3. Final Preprocessing Design

### 3.1 Chosen preprocessing decisions

The final preprocessing choices are:

- recordings trimmed to roughly `180 s`
- outlier detection run **per recording**, not pooled across people
- LOF run on all 6 raw sensor axes together:
  - `acc_x`, `acc_y`, `acc_z`
  - `gyro_x`, `gyro_y`, `gyro_z`
- LOF implementation:
  - `sklearn.neighbors.LocalOutlierFactor`
  - chunked processing for speed
  - `k = 5`
  - threshold = `1.5`
  - chunk size = `3000`
- flagged rows are set to `NaN`
- missing values are then filled by interpolation using the ML4QS imputation helper

### 3.2 Why LOF was used

We previously checked that the data was not well modeled as Gaussian, so Chauvenet was no longer the best fit. LOF is a more appropriate framework-consistent choice because:

- it does not assume normality
- it works on multivariate sensor points
- it can flag locally unusual 6-axis samples

### 3.3 How aggressive was the cleaning

LOF acted as **light cleaning**, not heavy distortion:

- average changed rows per recording: `1.894%`
- maximum changed rows on any recording: `3.492%`
  - this occurred for `Oana`, crowded, `session_1`

Per-participant changed-row percentages:

- `Jun`: `2.452%`
- `Darius`: `2.126%`
- `Oana`: `1.998%`
- `Ana`: `1.563%`
- `Pedro`: `1.528%`
- `Asena`: `1.492%`

This is a good result for the report because it shows the method removes only a small part of the signal.

## 4. Final Experiment Protocols

### 4.1 Experiment A: Owner authentication

Goal:

- binary classification: `Pedro` vs `non-Pedro`

Data:

- normal walking only
- participants: `Asena`, `Darius`, `Jun`, `Oana`, `Pedro`

Split:

- train = normal `session_1` + `session_2`
- test = normal `session_3`

Reason:

- owner authentication is the main identity-focused task, so we used the strongest within-context split by training on two sessions and testing on the third.

### 4.2 Experiment B: Person identification

Goal:

- multiclass classification of person identity

Data:

- same normal-only subset as owner authentication

Split:

- train = normal `session_1` + `session_2`
- test = normal `session_3`

Reason:

- same identity protocol as Experiment A, but extended from binary to multiclass.

### 4.3 Experiment C: Context classification

Goal:

- binary classification: `normal` vs `crowded`

Data:

- only participants with both contexts and enough sessions:
  - `Asena`, `Darius`, `Jun`, `Oana`, `Pedro`

Split:

- train = `session_1` for both normal and crowded
- test = `session_2` for both normal and crowded
- normal `session_3` is **excluded** from this experiment

Reason:

- this keeps the context experiment aligned across both contexts instead of giving normal walking extra training data.

### 4.4 Experiment D: Cross-context robustness

Goal:

- train an owner-authentication model on normal walking
- then test it on crowded walking **without retraining**

Protocol:

- use the owner-authentication training protocol from Experiment A
- evaluate:
  - on normal test windows
  - on crowded `session_2` windows

Reason:

- this tests whether the owner signal is robust to context shift.

## 5. Feature Engineering Design

### 5.1 Engineered features

The classical pipeline starts from **101 engineered features** per window, including:

- time-domain statistics
  - mean, std, median, min, max, RMS
- frequency-domain statistics
  - dominant frequency
  - spectral entropy
  - band powers
- derived signals
  - acceleration magnitude
  - gyroscope magnitude
- cross-axis correlations

### 5.2 PCA representation

PCA is used as an **alternative classical representation**:

- fit on the owner/person training subset only
- standardized before PCA
- kept enough components to explain `95%` variance
- final PCA representation size: **38 components**

Important:

- PCA is part of the **classical ML** pipeline
- PCA is **not** fed into TCN or GRU, because sequence models are supposed to learn from raw temporal windows

### 5.3 Reduced engineered feature set

We also created a reduced engineered feature set by grouping highly correlated features:

- correlation threshold: `|corr| > 0.95`
- redundant pairs found: `16`
- correlated groups found: `6`
- original engineered feature count: `101`
- reduced engineered feature count: `90`

Examples of absorbed feature groups:

- `gyro_magnitude_mean` absorbed:
  - `gyro_magnitude_median`
  - `gyro_magnitude_rms`
  - `gyro_y_rms`
  - `gyro_y_std`
- `gyro_z_band_1_3hz` absorbed:
  - `gyro_z_min`
  - `gyro_z_rms`
  - `gyro_z_std`
- `acc_y_std` absorbed:
  - `acc_magnitude_std`

### 5.4 Feature families that mattered most

Grouped by Random Forest importance on participant identification:

1. `Acceleration axes` + `Time-domain statistics`
2. `Gyroscope axes` + `Time-domain statistics`
3. `Gyroscope axes` + `Frequency-domain statistics`
4. `Acceleration axes` + `Frequency-domain statistics`
5. `Gyroscope magnitude` + `Time-domain statistics`

### 5.5 Top PCA drivers

The strongest original features contributing to the first principal components included:

- `Acceleration magnitude (RMS)`
- `Acceleration magnitude (standard deviation)`
- `Vertical acceleration (standard deviation)`
- `Gyroscope magnitude (mean)`
- `Gyroscope magnitude (median)`
- `Gyroscope magnitude (RMS)`
- `Roll rate (RMS)`
- `Roll rate (standard deviation)`
- `Lateral acceleration (mean / median / RMS)`
- `Yaw rate (RMS / standard deviation / band power 1-3 Hz)`

## 6. Classical ML: Final Findings

Three representations were compared:

- `Engineered` (`101` features)
- `PCA` (`38` features)
- `Reduced engineered` (`90` features)

Models compared:

- Logistic Regression
- Random Forest
- SVM (RBF)

### 6.1 Best classical result per task

| Task | Best feature set | Best model | Main metric(s) |
|---|---|---|---|
| Owner authentication | Engineered | Random Forest | Acc `1.0000`, F1 `1.0000`, AUC `1.0000` |
| Person identification | Engineered | SVM (RBF) | Acc `0.9989`, Macro-F1 `0.9989` |
| Context classification | Engineered | SVM (RBF) | Acc `0.8331`, F1 `0.8401`, AUC `0.8762` |
| Cross-context robustness (crowded test) | PCA | Logistic Regression | Acc `0.9304`, F1 `0.8075`, AUC `0.9742` |

### 6.2 Interpretation

- For **owner authentication** and **person identification**, the full engineered feature set worked best.
- For **context classification**, the engineered feature set also remained best.
- For **cross-context robustness**, **PCA helped**. It gave the strongest crowded-test generalization among the classical models.

This is a useful report point:

- PCA did **not** improve every task
- but it **did** improve the task where generalization across context mattered most

## 7. Deep Learning: Final Design and Findings

### 7.1 Why these deep models

The final deep learning notebook compares:

- **TCN**
- **GRU**

Why `GRU` was chosen as the second model:

- lighter than LSTM
- appropriate for short sequences and limited data
- gives a temporal recurrent baseline without overengineering

Why not Siamese / one-shot architecture:

- too complex for the available data and assignment scope
- would add design overhead without clear benefit here

### 7.2 Deep learning protocol

Input:

- raw `2 s` windows from `X_all.npy`
- 6 channels per timestep

Split:

- same updated owner-authentication split as the classical pipeline
- same crowded transfer test as the classical pipeline

Normalization:

- standardized from the training split only

Training:

- early stopping
- validation split inside training
- class-weighted cross-entropy

### 7.3 Final deep learning results

| Task | Model | Representation | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---:|---:|---:|---:|---:|
| Owner authentication | TCN | Raw 2 s windows | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| Owner authentication | GRU | Raw 2 s windows | `0.8726` | `0.7585` | `0.6109` | `1.0000` | `0.9983` |
| Cross-context robustness (normal test) | TCN | Raw 2 s windows | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| Cross-context robustness (crowded test) | TCN | Raw 2 s windows | `0.9978` | `0.9944` | `0.9944` | `0.9944` | `0.9995` |
| Cross-context robustness (normal test) | GRU | Raw 2 s windows | `0.8726` | `0.7585` | `0.6109` | `1.0000` | `0.9983` |
| Cross-context robustness (crowded test) | GRU | Raw 2 s windows | `0.6251` | `0.2894` | `0.2329` | `0.3820` | `0.5998` |

### 7.4 Deep vs classical takeaway

- `TCN` strongly outperformed `GRU`
- `GRU` is a useful lightweight baseline, but it did not compete well here
- the `TCN` result on crowded transfer is extremely strong and should be discussed carefully in the report
- among the classical models, the best crowded-transfer result was:
  - PCA + Logistic Regression
  - Acc `0.9304`, F1 `0.8075`
- among deep models, the best crowded-transfer result was:
  - TCN
  - Acc `0.9978`, F1 `0.9944`

This means the final report can say:

- the best classical approach for generalization was PCA + Logistic Regression
- the best deep approach was TCN on raw windows
- TCN produced the strongest cross-context transfer result overall

## 8. Recommended Figures for the Report

### Preprocessing

- `fig1_eda_gait_patterns.png`
- `fig3_outliers_before_after.png`
- `fig5_normal_vs_crowded.png`
- `fig6c_lof_threshold_selection.png`

### Feature engineering

- `fig16_pca_explained_variance.png`
- `fig17_pca_pc1_pc2_person_id.png`

The PCA loading plot was removed and replaced by printed / CSV output because it was not visually useful enough for the report.

### Classical ML

- `fig12_confusion_matrix_person_id.png`
- `fig13_roc_owner_auth.png`
- `fig14_cross_context_normal_vs_crowded.png`

### Deep learning

- `fig18_deep_training_curves.png`
- `fig19_deep_confusion_matrices.png`
- `fig20_deep_roc_curves.png`
- `fig21_deep_vs_classical_summary.png`

## 9. Main Files to Cite or Reuse

### Tables / CSVs

- `results/classical_ml_results.csv`
- `results/classical_ml_report_table.csv`
- `results/deeplearning_results.csv`
- `results/deeplearning_report_table.csv`
- `features/feature_family_importance_summary.csv`
- `features/redundant_feature_group_summary.csv`
- `features/pca_top_feature_drivers.csv`
- `figures/lof_cleaning_summary.csv`
- `figures/lof_outliers_per_participant.csv`

### Core notebooks

- `01_preprocessing.ipynb`
- `02_windowing.ipynb`
- `03_features.ipynb`
- `04_classical_ml.ipynb`
- `05_deeplearning.ipynb`

## 10. Suggested Report Narrative

If writing the report, the clean storyline is:

1. We standardized and trimmed the smartphone IMU recordings, then applied conservative multivariate LOF cleaning per recording.
2. We used two different split protocols because identity tasks and context tasks answer different questions.
3. We engineered interpretable time/frequency/correlation features and then tested:
   - the full engineered set
   - PCA
   - a reduced engineered set
4. Full engineered features worked best for within-context identity tasks.
5. PCA was most useful for **cross-context robustness**, where it improved classical generalization.
6. On the deep learning side, TCN and GRU were compared on raw windows.
7. TCN was clearly the stronger deep model and achieved the best overall cross-context performance.

## 11. Cautions / Things to Mention Honestly

- Session numbering is chronological, not folder-name-based.
- PCA is only used in the classical pipeline, not in the sequence models.
- Some results are very high, especially owner authentication and TCN crowded transfer, so they should be described carefully and not oversold without acknowledging dataset size and controlled collection conditions.
- The context-classification task is harder and clearly less perfect than the identity-focused tasks.

## 12. Short Handover Summary

If your groupmate wants the shortest possible summary:

- preprocessing: `scikit-learn` LOF + interpolation, light cleaning only
- owner/person split: normal `session_1` + `session_2` train, `session_3` test
- context split: `session_1` train, `session_2` test for both normal and crowded
- classical bests:
  - owner auth: engineered + RF
  - person ID: engineered + SVM
  - context: engineered + SVM
  - cross-context: PCA + Logistic Regression
- deep best:
  - TCN on raw windows
- comparison model:
  - GRU
- main result:
  - TCN gave the strongest cross-context robustness result overall
