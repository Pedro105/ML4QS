# ML4QS Project Report — Gait-Based User & Context Classification

## Overview

This project uses smartphone inertial sensor data (accelerometer + gyroscope) collected via Phyphox to build classifiers for two tasks:

1. **Person identification** — given a 2-second window of walking data, which of the 4 participants is it?
2. **Context classification** — was the person walking in a crowded environment or a normal one?

---

## 1. Data Collection

### Participants

| Task            | Participants                         | Sessions        |
| --------------- | ------------------------------------ | --------------- |
| Normal walking  | Pedro, Jun, Darius, Oana             | 3 sessions each |
| Crowded walking | Pedro, Jun, Darius, Oana, Ana, Asena | 2 sessions each |

Recordings were made at 100 Hz using Phyphox, targeting 180 seconds per session. Data is exported either as a single merged `Gait Data.csv` (Phyphox combined format) or as separate `Accelerometer.csv` + `Gyroscope.csv` files depending on the phone used.

### Folder naming convention

All raw recording folders follow a consistent scheme: `[Name]Uni[N]` for normal walking sessions and `[Name]Crowded[N]` for crowded sessions, nested under `[Name]/Crowded/` for crowded data. Full layout:

```
Pedro/
  PedroUni1, PedroUni2, PedroUni3
  Crowded/  PedroCrowded1, PedroCrowded2*, PedroCrowded3
Jun/
  JunUni1, JunUni2, JunUni3
  Crowded/  JunCrowded1, JunCrowded2
Darius/
  DariusUni1, DariusUni2, DariusUni3
  Crowded/  DariusCrowded1, DariusCrowded2
Oana/
  OanaUni1, OanaUni2, OanaUni3
  Crowded/  OanaCrowded1, OanaCrowded2
Ana/
  Crowded/  AnaCrowded1, AnaCrowded2*, AnaCrowded3
Asena/
  Crowded/  AsenaCrowded1, AsenaCrowded2
```

`*` = skipped recordings (too short): PedroCrowded2 (128 s, replaced by PedroCrowded3) and AnaCrowded2 (69 s, replaced by AnaCrowded3).

### Raw durations

Most recordings hit exactly 180 s. Known exceptions handled by the pipeline:

| Folder         | Duration | Action                              |
| -------------- | -------- | ----------------------------------- |
| AnaCrowded2    | 69 s     | Skipped (auto, below 160 s minimum) |
| AnaCrowded3    | 163.7 s  | Kept (above crowded minimum)        |
| DariusCrowded1 | 175.5 s  | Kept (above normal 175 s minimum)   |
| DariusCrowded2 | 355.6 s  | Trimmed to first 180 s              |
| PedroCrowded2  | 128.8 s  | Skipped (explicit SKIP_PATHS entry) |

---

## 2. Preprocessing (`01_preprocessing.ipynb`)

### Steps

1. **Load** raw CSVs and unify column names across both Phyphox export formats.
2. **Trim** every recording to exactly 180 s (first 180 s kept; recordings under 175 s for normal / 160 s for crowded are discarded).
3. **Outlier removal** using Chauvenet's criterion (ML4QS Ch. 3) independently per sensor axis. C = 1.5 was selected via sensitivity analysis — aggressive enough to remove hardware spikes (~52 m/s²) while preserving genuine gait peaks (~15–30 m/s²).
4. **Imputation** of nulled outlier samples using linear interpolation (ML4QS Ch. 3).
5. **Session assignment** by recording date: `session_1` is the earliest recording, `session_2` the next, and so on.

### Output — `cleaned_data/manifest.csv`

| Participant | Normal sessions | Crowded sessions |
| ----------- | --------------- | ---------------- |
| Pedro       | 3               | 2                |
| Jun         | 3               | 2                |
| Darius      | 3               | 2                |
| Oana        | 3               | 2                |
| Ana         | —               | 2                |
| Asena       | —               | 2                |

All 24 sessions are 175–180 s after processing.

---

## 3. Windowing & Train/Test Split (`02_windowing.ipynb`)

Each cleaned recording is sliced into 2-second windows with 50% overlap (1 s step) at 100 Hz → 200 samples per window, 6 channels (acc_x/y/z, gyro_x/y/z).

### Split strategy

| Context         | Train                 | Test      |
| --------------- | --------------------- | --------- |
| Normal walking  | session_1 + session_2 | session_3 |
| Crowded walking | session_1             | session_2 |

This is a **cross-session** evaluation: the model never sees any test-session windows during training. This tests generalisation to a completely new walking bout, not just held-out windows from the same recording.

### Window counts

| Participant | Normal train | Normal test | Crowded train | Crowded test |
| ----------- | ------------ | ----------- | ------------- | ------------ |
| Pedro       | 358          | 179         | 178           | 178          |
| Jun         | 358          | 179         | 179           | 179          |
| Darius      | 358          | 179         | 173           | 177          |
| Oana        | 358          | 179         | 179           | 179          |
| Ana         | —            | —           | 179           | 163          |
| Asena       | —            | —           | 178           | 178          |
| **Total**   | **1 432**    | **716**     | **1 066**     | **1 054**    |

### Exact data flow into each task

**Task 1 — Person Identification**

Only Pedro, Jun, Darius, Oana are used (the 4 with 3 normal sessions, flagged `cross_session_eligible`). Ana and Asena have no normal walking data and are excluded entirely from this task.

| Split | Sessions used                | Windows per person | Total |
| ----- | ---------------------------- | ------------------ | ----- |
| Train | normal session_1 + session_2 | 179 + 179 = 358    | 1,432 |
| Test  | normal session_3             | 179                | 716   |

Session_3 was recorded on a different day from sessions 1 and 2, so no window from the test recording ever appeared in training. The 4-class label is the participant name.

**Task 2 — Context Classification**

Only Pedro, Jun, Darius, Oana are used — the 4 who have _both_ normal and crowded recordings (`has_both_contexts`). Ana and Asena are excluded because they have no normal walking data, so the model cannot learn their normal baseline. This leaves approximately 700 crowded windows from Ana and Asena unused.

| Split | Sessions used                           | Crowded windows | Normal windows |
| ----- | --------------------------------------- | --------------- | -------------- |
| Train | crowded session_1 + normal sessions 1+2 | 709             | 1,432          |
| Test  | crowded session_2 + normal session_3    | 713             | 716            |

The test sessions (crowded session_2 and normal session_3) were all recorded on different days from their training counterparts — cross-session evaluation for both contexts. The binary label is `crowded` vs `normal`.

**Key limitation on both tasks:** The same 4 people appear in both train and test. The model has seen each person's gait during training (just from a different session), which likely inflates accuracy compared to a real scenario where an entirely new person is introduced. A leave-one-person-out evaluation (e.g. train on Pedro/Jun/Darius, test on Oana) would give a stricter and more honest measure of generalisation.

---

## 4. Feature Engineering (`03_features.ipynb`)

101 features are extracted per 2-second window across 8 signals: the 6 raw sensor axes plus scalar magnitude of acceleration and gyroscope.

**Per signal (8 × ~10 features):**

- Time domain: mean, std, median, min, max, RMS
- Frequency domain (ML4QS Ch. 4 `FourierTransformation`): dominant frequency, spectral entropy, band power in 0–1 Hz, 1–3 Hz, 3–5 Hz, >5 Hz

**Cross-axis (5 pairs):**

- Pearson correlation: (acc_x, acc_y), (acc_y, acc_z), (acc_x, acc_z), (gyro_x, gyro_y), (gyro_y, gyro_z)

17 feature pairs had |correlation| > 0.95, indicating some redundancy between magnitude signals and their component axes. No features were dropped; tree-based models handle this naturally and performance confirmed it was not a problem.

Inf/NaN values (rare edge cases in spectral entropy) were replaced with per-column training-set medians before fitting.

---

## 5. Classical ML (`04_classical_ml.ipynb`)

Four classifiers were evaluated with 5-fold cross-validated `GridSearchCV` on the training set only:

- Decision Tree
- Random Forest (100 estimators)
- k-NN
- SVM (RBF kernel)

k-NN and SVM inputs were `StandardScaler`-normalised; tree models received raw features.

---

## 6. Results

### Task 1: Person Identification

_Who is walking?_ — 4-class classification (Pedro, Jun, Darius, Oana), evaluated on normal walking cross-session (session_3 is test, never seen during training).

| Model         | Accuracy  | Macro F1  |
| ------------- | --------- | --------- |
| Decision Tree | 83.7%     | 83.1%     |
| Random Forest | 98.6%     | 98.6%     |
| SVM (RBF)     | 99.4%     | 99.4%     |
| **k-NN**      | **99.6%** | **99.6%** |

**Best: k-NN — 99.6% accuracy, 99.6% macro F1**

### Task 2: Context Classification

_Was the walking environment crowded?_ — binary classification across all 6 participants, evaluated on session_2 (crowded test) and session_3 (normal test).

| Model         | Accuracy  | F1        | ROC-AUC   |
| ------------- | --------- | --------- | --------- |
| Decision Tree | 72.4%     | 63.2%     | 0.728     |
| k-NN          | 81.7%     | 77.6%     | 0.979     |
| Random Forest | 87.5%     | 85.7%     | 0.989     |
| **SVM (RBF)** | **94.3%** | **94.0%** | **0.984** |

**Best: SVM (RBF) — 94.3% accuracy, 94.0% F1, AUC 0.984**

---

### Limitations

- **4 participants for person ID, 6 for context** — all results should be treated as feasibility demonstrations, not deployment benchmarks.
- **Same campus location** — all recordings were taken in a university building corridor, so the crowded vs normal distinction may partly reflect location-specific variation rather than purely walking dynamics.
- **No leave-one-person-out evaluation** — all participants appear in both train and test. A leave-one-out setup would give a more honest measure of generalisation to unseen individuals.
- **Ana's shorter crowded test session** (163.7 s vs 180 s target) means her test window count is ~9% lower than others, creating a slight class imbalance in context classification testing.
