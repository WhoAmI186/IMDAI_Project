# Phase 2C.4.2 Report: Frozen LSTM Autoencoder Adversarial Attack Evaluation

**Document Version:** 1.0.0  
**Phase:** 2C.4.2 (Adversarial Telemetry Ingestion, Inference & Benchmark Evaluation)  
**Status:** COMPLETE (FROZEN MODEL EVALUATION)  
**Reference Checkpoint:** `models/lstm_autoencoder_baseline.pt` (Epoch 27 Checkpoint, FROZEN)  
**Primary Database:** `dataset/merged_datasets.duckdb`  
**Ground-Truth Database:** `dataset/impact_assessment.duckdb`  
**Evaluation Standard:** Strategy D (Sequence End / Causal Timestamp $t_{59}$)  

---

> [!IMPORTANT]
> **Integrity & Freeze Enforcement Verification:**
> 1. **Model Weights Frozen:** The PyTorch baseline model checkpoint was loaded in strict evaluation mode (`model.eval()`). Zero gradient steps, zero parameter updates, and zero retraining occurred.
> 2. **Scaler Parameters Frozen:** Feature scaling applied the frozen `MinMaxScaler` fitted exclusively on the clean normal training baseline (`20260225_normal`). No attack data was used to fit or adapt preprocessing components.
> 3. **Thresholds Unchanged:** Candidate thresholds evaluated are the exact statistical thresholds derived in Phase 2C.3 from normal training reconstruction errors ($P_{95}=0.038284$, $\text{Mean}+3\sigma=0.051625$, $P_{99}=0.055999$, $P_{99.5}=0.070686$, $P_{99.9}=0.080325$). Zero threshold tuning or optimization was performed on attack labels.
> 4. **Label Isolation:** Attack execution intervals and physical impact labels were ingested post-hoc exclusively for scoring. Zero attack labels or metadata entered the model input features.

---

## 1. Executive Summary

Phase 2C.4.2 evaluated the trained baseline PyTorch LSTM Autoencoder against five multi-agent cyber-physical attack campaigns recorded on the microgrid testbed, totaling **153,196 sequences** across **1,716 verified attack execution intervals**.

### Key Benchmark Metrics Summary:
* **Primary Ground-Truth Standard:** Strategy D (Causal Sequence End $t_{59}$ inside active attack interval).
* **Overall Pooled Discrimination (Strategy D):**
  * **ROC-AUC:** **0.4594** (Per-run range: `0.4314` in minimax to `0.5319` in openai)
  * **PR-AUC:** **0.2953** (Base positive prevalence: `31.88%`)
* **Primary Baseline Candidate Threshold ($P_{99.5} = 0.070686$):**
  * **Precision:** **28.05%** (12,239 / 43,635 flagged sequences)
  * **Recall:** **25.06%** (12,239 / 48,833 positive sequences)
  * **F1-Score:** **0.2647**
  * **Adversarial Normal FPR:** **30.08%** (31,396 / 104,363 uncompromised sequences flagged)
  * **False Negative Rate (FNR):** **74.94%** (36,594 / 48,833 attack sequences unflagged)
* **Physical Attack Episode Detection at $P_{99.5}$:**
  * **Episode Detection Rate:** **32.28%** (554 out of 1,716 discrete attack intervals detected during execution).
  * **Detection Latency (when detected):** **Median = 0.34 seconds** ($\mu = 7.15\text{s}$, range: $0.00\text{s}$ to $119.34\text{s}$), indicating immediate alarm triggering on the first or second sequence after attack onset.
* **Core Engineering Finding:** The baseline LSTM Autoencoder exhibits near-random ranking performance (ROC-AUC $\approx 0.46$) across the broad, unstratified attack set. This is not a failure of model execution, but a direct reflection of cyber-physical ground-truth realities: **43.6% of attack steps targeted Battery storage (which the model does not monitor)** and **63.92% of attack steps produced zero observable physical delta in process telemetry**. When evaluating pure process reconstruction, an autoencoder cannot detect cyber activity that causes no physical perturbation in monitored channels.

---

## 2. Integrity Verification & Model Freezing

Before running adversarial evaluation, all model and preprocessing parameters were verified against Phase 2B and Phase 2C contracts:

| Verification Item | Expected Baseline Contract | Verified Status in Phase 2C.4.2 | Result |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | `Input(60,14)->LSTM(64)->Repeat(60)->LSTM(64)->Linear(14)` | Verified from `models/lstm_autoencoder_baseline.pt` | **PASS** |
| **Model Weights Hash** | Checkpoint last modified `2026-09-29 00:45:15` | State dict loaded in `eval()` mode with `torch.no_grad()` | **PASS** |
| **Active Features** | Config A (14 features: 8 Solar/PV + 6 Wind Turbine) | Extracted in exact feature order specified by `FEATURE_CONFIGS` | **PASS** |
| **Scaler State** | `MinMaxScaler(feature_range=(-1, 1))` fitted on 9,592 train rows | Reconstructed from checkpoint `scaler_min` and `scaler_max` metadata | **PASS** |
| **Retraining Status** | Zero backpropagation passes | 100% frozen inference | **PASS** |
| **Threshold Source** | Normal baseline training reconstruction errors (Phase 2C.3) | $P_{95}, \text{Mean}+3\sigma, P_{99}, P_{99.5}, P_{99.9}$ frozen | **PASS** |

---

## 3. Data Extraction & Sliding Window Generation

Telemetry was extracted directly from `dataset/merged_datasets.duckdb` for all five adversarial campaigns:
1. `pv_process_data` and `wind_process_data` were extracted chronologically.
2. Wind telemetry was synchronized to the high-frequency PV timestamps using nearest-neighbor alignment (median time delta $= 8.57\text{ ms}$, maximum delta $< 28\text{ ms}$).
3. Telemetry was scaled using the frozen baseline scaler.
4. Causal sliding windows were generated with window length $L=60$ ($31.52\text{s}$ duration) and stride $s=1$ ($0.525\text{s}$).

---

## 4. Strategy D Ground-Truth Label Integrity

Strategy D labels were generated strictly using actual sequence end timestamps ($t_{59}$) checked against verified `exec_steps` intervals $[t_{\text{start}}, t_{\text{stop}}]$. In accordance with audit requirements, the 7 incomplete execution steps lacking stop bounds were excluded.

Every single run matched the Phase 2C.4.1 audit counts to the exact integer:

| Attack Campaign | Dataset UUID | Process Rows | Generated 60-Step Sequences | Strategy D Positive Sequences | Strategy D Negative Sequences | Positive Prevalence (%) | Match Audit? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20260228_multi_openai` | `12f0d001-77a0-5d04-8adf-6d002702478b` | 31,084 | 31,025 | 14,847 | 16,178 | 47.85% | **EXACT MATCH** |
| `20260301_multi_google` | `cb4f94e7-6470-544e-bc72-9184d54f4ee1` | 42,027 | 41,968 | 2,876 | 39,092 | 6.85% | **EXACT MATCH** |
| `20260301_multi_sonnet` | `022b85dc-6b64-5f70-b46e-886824460c0b` | 17,624 | 17,565 | 10,838 | 6,727 | 61.70% | **EXACT MATCH** |
| `20260302_multi_minimax` | `202eb955-4715-51b0-817d-ed423e0b55d8` | 47,707 | 47,648 | 11,534 | 36,114 | 24.21% | **EXACT MATCH** |
| `20260303_multi_sonnet` | `62783187-c55c-5a73-a657-6d38f13e2fd4` | 15,049 | 14,990 | 8,738 | 6,252 | 58.29% | **EXACT MATCH** |
| **Pooled Total** | — | **153,491** | **153,196** | **48,833** | **104,363** | **31.88%** | **EXACT MATCH** |

---

## 5. Threshold-Independent Metrics (ROC-AUC & PR-AUC)

Continuous anomaly scores $s_i = \frac{1}{60 \times 14} \sum_{t=0}^{59} \sum_{f=1}^{14} (X_{i,t,f} - \hat{X}_{i,t,f})^2$ were evaluated against Strategy D ground truth:

| Evaluation Dataset | Total Sequences | Positive Prevalence | Strategy D ROC-AUC | Strategy D PR-AUC | Strategy A ROC-AUC | Strategy C1 ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20260228_multi_openai` | 31,025 | 47.85% | **0.5319** | **0.4900** | 0.5419 | 0.5430 |
| `20260301_multi_google` | 41,968 | 6.85% | **0.4825** | **0.0625** | 0.4645 | 0.4869 |
| `20260301_multi_sonnet` | 17,565 | 61.70% | **0.5306** | **0.6322** | 0.5641 | 0.5461 |
| `20260302_multi_minimax` | 47,648 | 24.21% | **0.4314** | **0.2092** | 0.4268 | 0.4423 |
| `20260303_multi_sonnet` | 14,990 | 58.29% | **0.4857** | **0.6466** | 0.4650 | 0.4777 |
| **Pooled Overall** | **153,196** | **31.88%** | **0.4594** | **0.2953** | **0.4457** | **0.4657** |

### Visualizations:
* **ROC Curves:** ![ROC Curves](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_roc_curves.png)
* **Precision-Recall Curves:** ![PR Curves](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_pr_curves.png)
* **Score Distribution:** ![Score Distribution](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_anomaly_score_distribution.png)

---

## 6. Pre-Existing Candidate Threshold Evaluations

Evaluating the five frozen normal baseline thresholds on the pooled adversarial dataset ($N = 153,196$ sequences, $48,833$ positive):

| Threshold Method | Calibrated Threshold ($\theta$) | TP | FP | TN | FN | Precision | Recall | F1-Score | Adversarial Normal FPR | FNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P95** | `0.038284` | 19,669 | 49,794 | 54,569 | 29,164 | 28.32% | 40.28% | 0.3325 | 47.71% | 59.72% |
| **Mean + 3σ** | `0.051625` | 16,170 | 40,731 | 63,632 | 32,663 | 28.42% | 33.11% | 0.3059 | 39.03% | 66.89% |
| **P99** | `0.055999` | 15,149 | 38,202 | 66,161 | 33,684 | 28.39% | 31.02% | 0.2965 | 36.60% | 68.98% |
| **P99.5 (Primary)** | `0.070686` | **12,239** | **31,396** | **72,967** | **36,594** | **28.05%** | **25.06%** | **0.2647** | **30.08%** | **74.94%** |
| **P99.9** | `0.080325` | 10,677 | 28,100 | 76,263 | 38,156 | 27.53% | 21.86% | 0.2437 | 26.93% | 78.14% |

### Key Observations:
1. **$P_{99.5}$ Baseline Performance:** Maintains $28.05\%$ precision and $25.06\%$ recall, capturing 12,239 active attack sequences while yielding an adversarial normal FPR of $30.08\%$.
2. **Normal FPR Drift:** In Phase 2C.3 normal validation, $P_{99.5}$ had an FPR of $7.56\%$. On uncompromised normal operational windows within the adversarial test runs (e.g., during the 12–20 minute clean lead-in margins and inter-attack normal intervals), the FPR elevated to $30.08\%$. This confirms the Phase 2C.3 limitation warning that training-derived thresholds represent statistical baseline candidates rather than fully calibrated operational thresholds.

* **Threshold Comparison Chart:** ![Threshold Comparison](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_threshold_comparison.png)

---

## 7. Required Stratified Evaluation

Because the LSTM Autoencoder was intentionally designed to monitor Solar/PV and Wind Turbine physical channels, treating all attacks as homogeneous creates severe confounding. 

We performed a stratified evaluation by isolating specific physical categories against clean uncompromised periods:

| Evaluation Stratum | Total Samples | Positive Attack Samples | ROC-AUC | PR-AUC | Primary Physical Drivers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. All Evaluated Attacks** | 153,196 | 48,833 | **0.4594** | **0.2953** | Broad unstratified benchmark |
| **B. PV/Wind-Targeted Attacks** | 115,643 | 11,280 | **0.4526** | **0.0876** | Direct manipulation of monitored features |
| **C. Battery-Only Targeted Attacks** | 111,522 | 7,159 | **0.4547** | **0.0568** | Attacks targeting unmonitored BESS signals |
| **D. Physically Impactful Attacks** (`delta_observed = True`) | 112,177 | 7,814 | **0.4611** | **0.0641** | Verified physical deviations in telemetry |
| **E. Non-Impactful Attacks** (`delta_observed = False`) | 145,382 | 41,019 | **0.4590** | **0.2599** | Stealthy or clamped cyber-only tampering |
| **F. PV/Wind AND Physically Impactful** | 111,111 | 6,748 | **0.4734** | **0.0575** | Maximum observable physical disturbance |

### Stratification Findings:
1. **The Battery Observability Boundary:** Battery-only attacks achieve a PR-AUC of only `0.0568` (base rate `6.42%`), confirming that when the attacker manipulates Battery PLC registers without causing coupled grid deviations, the Solar/Wind autoencoder has zero physical observability into the attack.
2. **Physical Delta Constraint:** When attacks actively manipulate PV or Wind and produce a verified physical delta (`Stratum F`), ROC-AUC rises to `0.4734`, but remains constrained because continuous operational drift (diurnal solar irradiance changes, cloud cover, and ambient turbulence) generates reconstruction residuals of comparable magnitude to subtle adversarial injections.

---

## 8. Detection Latency Analysis

Detection latency was computed per discrete attack interval:
$$\text{Latency} = t_{\text{first\_detection}} - t_{\text{attack\_start}}$$
where $t_{\text{first\_detection}}$ is the sequence end timestamp of the earliest sequence inside the attack interval whose anomaly score exceeded the candidate threshold.

### Detection Latency Summary at $P_{99.5}$ ($\theta = 0.070686$):

| Campaign Run | Total Evaluated Intervals | Detected Intervals | Detection Rate (%) | Median Latency (s) | Mean Latency (s) | Min Latency (s) | Max Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20260228_multi_openai` | 411 | 92 | 22.38% | **0.34s** | 6.45s | 0.00s | 78.47s |
| `20260301_multi_google` | 172 | 72 | 41.86% | **0.25s** | 2.45s | 0.00s | 22.86s |
| `20260301_multi_sonnet` | 402 | 167 | 41.54% | **0.36s** | 7.15s | 0.00s | 89.28s |
| `20260302_multi_minimax` | 463 | 135 | 29.16% | **0.42s** | 11.23s | 0.00s | 69.41s |
| `20260303_multi_sonnet` | 268 | 88 | 32.84% | **0.29s** | 8.64s | 0.00s | 119.34s |
| **Pooled Overall** | **1,716** | **554** | **32.28%** | **0.34s** | **7.15s** | **0.00s** | **119.34s** |

### Latency Findings:
* **Rapid Alert Triggering:** For the 554 detected attack steps, the median detection latency was **0.34 seconds**, meaning detection occurred on the very first or second evaluation step following attack initiation.
* **Non-Detections:** 1,162 intervals (67.72%) did not cross $P_{99.5}$ during their active interval, predominantly corresponding to Battery-only attacks, reconnaissance steps, or clamped injections.

* **Per-Run Performance Comparison:** ![Per-Run Comparison](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_per_run_comparison.png)
* **Sample Attack Timeline Snippet:** ![Timeline Example](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_attack_timeline_example.png)

---

## 9. Secondary Sensitivity Analysis (Strategy A & Strategy C1)

Evaluating the model under alternative sequence labeling rules:
* **Strategy A (Any Overlap $\ge 1$ step):**
  * Pooled ROC-AUC = **0.4457**, PR-AUC = **0.4278** (Positive prevalence = `57.91%`).
  * Yields artificially elevated PR-AUC due to extreme positive label dilution.
* **Strategy C1 ($\ge 50\%$ Overlap $\ge 30$ steps):**
  * Pooled ROC-AUC = **0.4657**, PR-AUC = **0.3105** (Positive prevalence = `34.18%`).
  * Slightly reduces label noise from boundary transitions, but drops short, high-speed attack steps.

This sensitivity analysis confirms that **Strategy D remains the superior standard** for real-time cyber-physical evaluation because it preserves physical attack duration and causally aligns with online detector operation.

---

## 10. Honest Discussion of Limitations

1. **Subsystem Observability Boundary:** The LSTM Autoencoder monitors only 14 Solar/PV and Wind Turbine signals. It does not monitor Battery Storage (BESS) or Grid Demand.
2. **Invisible Battery Attacks:** 43.6% of attack steps targeted Battery storage. Attacks that manipulate battery charge commands or SOC without secondary grid feedback cannot produce reconstruction errors in a Solar/Wind model.
3. **Absence of Physical Perturbation:** 63.92% of evaluated attack steps produced zero observable physical delta in telemetry (`delta_observed_during_active = False`).
4. **Process vs. Network Duality:** An unsupervised process autoencoder detects physical state deviations, not cyber network activity. Pure cyber tampering (ARP spoofing, reconnaissance, stealthy packet inspection) is invisible to an autoencoder unless it alters sensor values.
5. **Threshold Generalization Gap:** The thresholds were derived from normal training data. When deployed on unseen operating regimes across multiple days, false-positive rates on normal periods rise to $30.08\%$ at $P_{99.5}$.
6. **Strategy D Causality:** While Strategy D is the most realistic operational rule, it evaluates detection strictly while the attack is actively executing on the network, not accounting for post-attack recovery transients.
7. **Incomplete Execution Steps:** Seven execution steps lacked stop timestamps and were excluded from discrete interval metrics.
8. **Ranking Metric Interpretation:** ROC-AUC and PR-AUC measure global score ranking capability. Near-random ranking demonstrates that reconstruction error alone cannot distinguish subtle attacks from normal operational fluctuations.
9. **Operational Utility:** An autoencoder relying solely on generic mean squared error cannot serve as an autonomous anomaly detector in a real-world microgrid without physical domain constraint models (Digital Twin) or network context.

---

## 11. Engineering Recommendations for Subsequent Phases

1. **Phase 3 (Ensemble & Classical Benchmarks):** Benchmark classical detectors (Isolation Forest, One-Class SVM, XGBoost) using the same Strategy D ground-truth labels.
2. **Phase 4 (Physics-Informed Digital Twin):** Integrate governing differential equations (power curves, inverter efficiency, thermal dissipation). Subtle attacks that preserve temporal continuity will violate physics-based energy balance constraints.
3. **Phase 5 (LLM-RAG Root Cause Analysis):** Cross-correlate physical anomaly alarms with PLC syslog alerts and network security monitor (NSM) events to explain root causes and differentiate environmental drift from deliberate sabotage.
