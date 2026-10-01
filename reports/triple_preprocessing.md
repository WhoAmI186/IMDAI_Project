# Triple Dataset Preprocessing and Data Sanitization Report

**Document ID:** `REPORT-TRIPLE-PREPROCESSING-001`  
**Execution Date:** 2026-10-01  
**Project:** Cyber-Physical Smart Grid Anomaly Detection  
**Module:** `src/data/triple_loader.py` & `src/ml/triple_preprocessor.py`  
**Dataset Directory:** `dataset/triple/` (`data1.csv` to `data15.csv`)  
**Status:** Completed & Validated

---

## 1. Overview and Objectives

This report documents the end-to-end data ingestion, sanitization, feature selection, scaling, and sequence generation pipeline implemented for the candidate MSU/ORNL Power System synchrophasor dataset (`dataset/triple/`).

The primary objectives achieved:
1. Ingest all 15 scenario CSV files without modifying the raw files on disk.
2. Isolate and eliminate known data acquisition corruptions (`data4.csv` and `data8.csv`).
3. Enforce strict isolation of the ground-truth target label (`marker`) from input telemetry matrices to prevent data leakage.
4. Extract the approved **16 Layer 1 physical features**.
5. Fit the standard scaler strictly on uncompromised `NoEvents` telemetry from `data1..data10`.
6. Generate sliding sequences ($L = 60$, stride = 1) chronologically without crossing file boundaries.

---

## 2. Raw Ingestion and Data Sanitization Summary

| Metric | Count | Technical Description / Actions Taken |
|:---|:---:|:---|
| **Total Files Processed** | 15 | `data1.csv` through `data15.csv` |
| **Total Raw Ingested Rows** | 78,377 | All 15 files with 129 identical columns |
| **Total Clean Working Rows** | 78,361 | Exactly 16 rows removed across entire dataset |
| **Corrupted Rows Removed** | **8** | Indices 3974–3981 in `data4.csv` (column-shift acquisition artifact) |
| **Duplicate Rows Removed** | **8** | Indices 4287–4294 in `data8.csv` (identical contiguous logger buffer repeat) |
| **Missing Values (`NaN`)** | 0 | 0 missing values across all files |
| **Infinite Values (`inf`)** | 10,906 | Confined to 4 excluded impedance magnitude columns (`R1..4-PA:Z`) |

### Detailed Sanitization Actions:

1. **`data4.csv` Column-Shift Removal (Indices 3974–3981):**
   - **Root Cause:** A temporary hardware acquisition glitch caused PMU frame columns to shift by 2 positions (e.g. system frequency $60.0\text{ Hz}$ was logged into the current magnitude column `R1-PM12:I`, and floating-point phase angles were placed into the integer status word `R4:S`).
   - **Action:** Detected via index mask `[3974..3981]` and safely excluded from the working dataset in `TripleDataLoader.sanitize_scenario`. Raw CSV remains 100% untouched.
2. **`data8.csv` Contiguous Duplicate Removal (Indices 4287–4294):**
   - **Root Cause:** Serial logger buffer retransmission during a natural fault sequence produced 8 consecutive duplicate rows.
   - **Action:** Deduplicated using `df.duplicated(keep="first")`. Exactly 8 redundant rows removed.

---

## 3. Approved 16 Layer 1 Physical Feature Set

In accordance with the approved audit ([`reports/triple_dataset_audit.md`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/triple_dataset_audit.md)), the old 14-feature solar/wind configuration was retired in favor of the **16 core synchrophasor transmission features**:

| # | Feature Name | Physical Domain | Engineering Unit | Physical Role & Justification |
|:---:|:---|:---|:---:|:---|
| 1 | `R1-PM1:V` | Substation 1 (Bus 1) Voltage Phasor | $\text{V RMS}$ | Phase A voltage magnitude at sending bus (Relay 1) |
| 2 | `R1-PA1:VH` | Substation 1 (Bus 1) Voltage Phasor | $\text{degrees}$ | Phase A voltage phase angle at sending bus (Relay 1) |
| 3 | `R4-PM1:V` | Substation 1 (Bus 1) Voltage Phasor | $\text{V RMS}$ | Redundant Phase A voltage magnitude at Bus 1 (Relay 4) |
| 4 | `R2-PM1:V` | Substation 2 (Bus 2) Voltage Phasor | $\text{V RMS}$ | Phase A voltage magnitude at receiving bus (Relay 2) |
| 5 | `R2-PA1:VH` | Substation 2 (Bus 2) Voltage Phasor | $\text{degrees}$ | Phase A voltage phase angle at receiving bus (Relay 2) |
| 6 | `R3-PM1:V` | Substation 2 (Bus 2) Voltage Phasor | $\text{V RMS}$ | Redundant Phase A voltage magnitude at Bus 2 (Relay 3) |
| 7 | `R1-PM4:I` | Transmission Line 1 Current Phasor | $\text{A RMS}$ | Phase A current magnitude at sending terminal of Line 1 |
| 8 | `R1-PA4:IH` | Transmission Line 1 Current Phasor | $\text{degrees}$ | Phase A current phase angle at sending terminal of Line 1 |
| 9 | `R2-PM4:I` | Transmission Line 1 Current Phasor | $\text{A RMS}$ | Phase A current magnitude at receiving terminal of Line 1 |
| 10 | `R2-PA4:IH` | Transmission Line 1 Current Phasor | $\text{degrees}$ | Phase A current phase angle at receiving terminal of Line 1 |
| 11 | `R4-PM4:I` | Transmission Line 2 Current Phasor | $\text{A RMS}$ | Phase A current magnitude at sending terminal of Line 2 |
| 12 | `R3-PM4:I` | Transmission Line 2 Current Phasor | $\text{A RMS}$ | Phase A current magnitude at receiving terminal of Line 2 |
| 13 | `R1:F` | Grid Frequency Stability | $\text{Hz}$ | System electrical frequency measured at Substation 1 |
| 14 | `R1:DF` | Frequency Dynamics | $\text{Hz/s}$ | Rate of change of frequency ($df/dt$, ROCOF) at Substation 1 |
| 15 | `R2:F` | Grid Frequency Stability | $\text{Hz}$ | System electrical frequency measured at Substation 2 |
| 16 | `R1-PA:ZH` | Apparent Impedance Angle | $\text{radians}$ | Apparent impedance phase angle (replaces divergent $Z$ magnitude) |

---

## 4. Normal Scaling Protocol

- **Scaler Type:** `TripleStandardScaler` (z-score normalization: $\tilde{x} = (x - \mu) / \sigma$).
- **Calibration Split:** Fitted **strictly on uncompromised `NoEvents` telemetry** pooled across training scenarios `data1.csv` through `data10.csv`.
- **Sample Count Seen:** **3,080 clean normal samples**.
- **Leakage Prevention:** Zero `Attack` and zero `Natural` rows were exposed to the scaler. Validation (`data11..12`) and test (`data13..15`) data were strictly transformed using the frozen training parameters.
- **Artifact:** Serialized to [`models/triple_scaler.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/triple_scaler.json).

---

## 5. Sequence Generation and Accounting

Sliding sequence windows of length $L = 60$ and stride $s = 1$ were generated chronologically:

| Partition | Source Files | Total Clean Rows | Filter Applied | Output Sequences ($N, 60, 16$) | Warm-Up Rows Dropped | Sequences Lost Across Boundaries |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Training (Normal Only)** | `data1.csv` .. `data10.csv` | 52,464 total ($3,080$ NoEvents) | `marker == 'NoEvents'` | **2,490** | 59 per file ($590$ total) | 0 (boundary crossing strictly forbidden) |
| **Validation (Normal Only)** | `data11.csv`, `data12.csv` | 10,475 total ($529$ NoEvents) | `marker == 'NoEvents'` | **411** | 59 per file ($118$ total) | 0 |
| **Held-Out Test (Full Scenarios)** | `data13.csv`, `data14.csv`, `data15.csv` | 15,662 total | None (Continuous evaluation) | **15,485** | 59 per file ($177$ total) | 0 |

### Integrity Verification:
- **No File Boundary Leaks:** Sequence generation resets at the start of each file.
- **Temporal Alignment:** Every sequence ending at index $t$ corresponds exactly to row index $t$ in the sanitized file.
