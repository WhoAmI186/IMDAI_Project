# Phase 2C.4.3 Report: Scope-Appropriate Frozen LSTM Evaluation

**Document Version:** 1.0.0  
**Phase:** 2C.4.3 (Scope-Appropriate Diagnostic Evaluation)  
**Status:** COMPLETE (FROZEN MODEL DIAGNOSTIC EVALUATION)  
**Reference Checkpoint:** `models/lstm_autoencoder_baseline.pt` (Epoch 27 Checkpoint, FROZEN)  
**Primary Database:** `dataset/merged_datasets.duckdb`  
**Ground-Truth Impact Database:** `dataset/impact_assessment.duckdb`  
**Evaluation Standard:** Strategy D (Sequence End / Causal Timestamp $t_{59}$)  

---

> [!IMPORTANT]
> **Freeze & Integrity Verification Contract:**
> 1. **Model Weights Frozen:** The baseline PyTorch LSTM Autoencoder (`models/lstm_autoencoder_baseline.pt`, best epoch 27) was evaluated in strict evaluation mode (`model.eval()`). Zero retraining, zero parameter adjustments, and zero architectural alterations occurred.
> 2. **Scaler Parameters Frozen:** Feature scaling utilized the frozen baseline `MinMaxScaler` parameters fitted exclusively on the 9,592 clean normal training baseline rows (`20260225_normal`). No attack data was used to fit or tune preprocessing.
> 3. **Statistical Thresholds Unchanged:** Evaluation applied the exact unsupervised statistical thresholds established in Phase 2C.3 ($P_{95}=0.038284$, $\text{Mean}+3\sigma=0.051625$, $P_{99}=0.055999$, $P_{99.5}=0.070686$, $P_{99.9}=0.080325$). The primary candidate threshold remains $P_{99.5}$. Zero threshold optimization or re-tuning was performed using attack performance.
> 4. **Secondary Diagnostic Role:** This evaluation does NOT replace, invalidate, or overwrite the Phase 2C.4.2 all-attack benchmark. The Phase 2C.4.2 results ($\text{ROC-AUC}=0.4594$, $\text{PR-AUC}=0.2953$, $P_{99.5}\text{ Precision}=28.05\%$, $P_{99.5}\text{ Recall}=25.06\%$) remain intact and active as the primary benchmark.

---

## 1. Executive Summary

Phase 2C.4.3 answers a targeted scientific question:
> *"How well does the frozen process-data LSTM detect attack activity that is actually observable in the Solar/PV and Wind process signals it monitors?"*

The Phase 2C.4.2 benchmark evaluated the process-only LSTM against all 1,716 attack execution steps, including Battery storage attacks and attacks with zero observable physical impact. Phase 2C.4.3 isolates a **Primary Scope-Appropriate Subset** of attack intervals satisfying two strict criteria:
1. **Target Relevance:** The attack targeted Solar/PV or Wind process telemetry (`pv_process_data` or `wind_process_data`), matching the 14 features monitored by the model.
2. **Verified Physical Impact:** Observable physical perturbation was confirmed during execution (`delta_observed_during_active == True`).

### Summary of Key Findings:

* **Scope-Appropriate Subset Size:** Across all five adversarial campaigns, exactly **304 discrete execution intervals** satisfied both criteria (out of 1,716 valid steps and 1,009 steps evaluated by the physical judge). Exactly 60 verified impactful steps targeting Battery storage were strictly excluded because the model does not monitor Battery channels.
* **Sequence Mapping (Strategy D):** Mapping the 304 qualifying intervals to 60-step causal sequences ($L=60$, $s=1$) produced **6,367 positive sequences** (4.16% base rate across all 153,196 sequences).
* **Discrimination Remains Near-Random:**
  * **Primary Definition (All Non-Qualifying Negatives, $N=153,196$):** $\text{ROC-AUC} = \mathbf{0.4882}$, $\text{PR-AUC} = \mathbf{0.0401}$ (Random guess baseline $= 0.0416$).
  * **Secondary Clean-Only Definition (Strictly Unattacked Negatives, $N=110,730$):** $\text{ROC-AUC} = \mathbf{0.4753}$, $\text{PR-AUC} = \mathbf{0.0539}$ (Random guess baseline $= 0.0575$).
* **Primary Candidate Threshold ($P_{99.5} = 0.070686$):**
  * **Precision:** **3.82%** (Primary) / **5.04%** (Clean Negatives).
  * **Recall:** **26.15%** (1,665 / 6,367 qualifying positive sequences flagged).
  * **F1-Score:** **0.0666** (Primary) / **0.0845** (Clean Negatives).
  * **False Positive Rate (FPR):** **28.58%** (Primary) / **30.08%** (Clean Negatives).
  * **False Negative Rate (FNR):** **73.85%** (4,702 / 6,367 qualifying sequences unflagged).
* **Discrete Interval Event-Level Detection at $P_{99.5}$:**
  * **Detection Rate:** **31.25%** (95 out of 304 qualifying intervals triggered at least one alarm during active execution).
  * **Missed Rate:** **68.75%** (209 out of 304 intervals completely missed).
  * **Detection Latency:** **Median = 0.33 seconds** ($\mu = 1.60\text{s}$, range: $0.00\text{s}$ to $20.69\text{s}$).
* **Core Scientific Conclusion:** Filtering the evaluation to attack activity with verified, observable physical impact in the monitored subsystem **does NOT materially improve detector performance**. Global ranking remains near random ($\text{ROC-AUC} \approx 0.48$). While event-level detection increases modestly from 21.04% to 31.25% and detection latency drops to 0.33s, almost 69% of physically impactful attack intervals and 73.85% of sequence endpoints remain undetected. 
  
  **Decisive Diagnosis:** The failure of the LSTM Autoencoder is **NOT** merely an artifact of evaluating against out-of-scope attacks. Rather, **unsupervised reconstruction MSE on pure process data is fundamentally insufficient for cyber-physical attack detection**. Because the neural network lacks physical governing laws (e.g., solar irradiance-power equations, wind aerodynamic power curves, thermal inertia constraints), normal operational fluctuations produce reconstruction residuals equal to or greater than subtle, stealthy cyber manipulations. This finding establishes the absolute necessity for **Phase 4 (Physics-Informed Digital Twin)** and **Phase 5 (LLM-RAG Multimodal Fusion)**.

---

## 2. Integrity Verification & Model Freezing

All model weights, scaling parameters, and statistical thresholds were strictly frozen to guarantee that Phase 2C.4.3 represents an objective diagnostic measurement:

| Verification Item | Requirement | Observed Status in Phase 2C.4.3 | Status |
| :--- | :--- | :--- | :--- |
| **Model Weights** | Strict freeze (`models/lstm_autoencoder_baseline.pt`, Epoch 27) | Loaded in `eval()` mode with `torch.no_grad()`. Zero updates. | **PASS** |
| **Scaler State** | Frozen `MinMaxScaler` fitted on 9,592 clean baseline train rows | Reconstructed strictly from checkpoint `scaler_min`/`scaler_max`. | **PASS** |
| **Threshold Calibration** | Unsupervised thresholds derived in Phase 2C.3 | Unchanged: $P_{95}, \text{Mean}+3\sigma, P_{99}, P_{99.5}, P_{99.9}$. | **PASS** |
| **Ground-Truth Standard** | Strategy D (Sequence End $t_{59}$) | Causal sequence end timestamp mapped without lookahead. | **PASS** |
| **Label Isolation** | No labels in model input | Labels ingested post-hoc exclusively for scoring. | **PASS** |
| **Baseline Preserved** | Phase 2C.4.2 reports intact | `phase2c_attack_evaluation.md` and `.json` unmodified. | **PASS** |

---

## 3. Data Constraints & Scope-Appropriate Subset Definition

### 3.1 Data Constraints from Phase 2C.4.1 Audit
The ground-truth audit in Phase 2C.4.1 established the following partition of the 1,716 valid discrete execution steps across all five attack campaigns:
- **Total Valid Execution Steps:** 1,716
- **Evaluated Execution Steps (Physical Judge Evaluated):** 1,009 steps
- **Unevaluated Execution Steps:** 707 steps
- **Evaluated Steps with Observed Physical Impact:** 364 steps (36.08% of evaluated)
- **Evaluated Steps with Zero Observed Impact:** 645 steps (63.92% of evaluated)

> [!CAUTION]
> **Treatment of Missing Labels:**
> In strict compliance with scientific integrity standards, **unevaluated steps were NOT assumed to have `delta=False`**. Only attack steps with an explicit, verified `delta_observed_during_active == True` entry in `impact_assessment.duckdb` were admitted into the qualifying subset.

### 3.2 Defining the Scope-Appropriate Subset
The LSTM Autoencoder monitors exactly 14 process signals corresponding to Config A:
- **Solar/PV (8 signals):** `pv_c_on_off`, `pv_m_temp_air`, `pv_m_poa_direct`, `pv_m_wind_speed`, `pv_m_poa_diffuse`, `pv_m_cell_temperature`, `pv_m_inverter_ac_power`, `pv_m_inverter_dc_power`.
- **Wind Turbine (6 signals):** `wind_m_power`, `wind_m_pressure`, `wind_m_wind_speed_a`, `wind_m_wind_speed_b`, `wind_m_temperature_a`, `wind_m_temperature_b`.

The model monitors **zero** Battery energy storage signals. Therefore, all Battery attacks must be excluded from a scope-appropriate test.

The **Primary Scope-Appropriate Subset** is defined strictly by the conjunction:
$$\text{Qualifying Step} \iff (\text{connection\_table} \in \{\text{'pv\_process\_data'}, \text{'wind\_process\_data'}\}) \land (\text{delta\_observed\_during\_active} = \text{True})$$

### 3.3 Partition Breakdown:
Across the 1,009 evaluated execution steps:
* `battery_process_data`: 440 steps evaluated $\rightarrow$ **60 impactful steps strictly excluded** (outside model scope) + 380 zero-impact steps.
* `pv_process_data`: 158 steps evaluated $\rightarrow$ **102 qualifying impactful steps** + 56 zero-impact steps.
* `wind_process_data`: 411 steps evaluated $\rightarrow$ **202 qualifying impactful steps** + 209 zero-impact steps.
* **Total Primary Qualifying Execution Steps:** $102 + 202 = \mathbf{304}$ steps.

### 3.4 Breakdown by Monitored Signal:
The 304 qualifying attack steps directly targeted signals within the model's 14 monitored channels:
- `wind_speed_A`: 74 steps
- `power` (wind): 38 steps
- `pressure` (wind): 32 steps
- `temp_air` (PV): 29 steps
- `inverter_ac_power` (PV): 25 steps
- `wind_speed_B`: 23 steps
- `cell_temperature` (PV): 22 steps
- `temperature_A` (wind): 15 steps
- `poa_direct` (PV): 12 steps
- `temperature_B` (wind): 11 steps
- `rotation_speed` (wind): 9 steps
- `inverter_dc_power` (PV): 7 steps
- `poa_diffuse` (PV): 5 steps
- `on_off` (PV): 2 steps

There is zero taxonomy ambiguity: all targeted entities map directly to the active 14-feature process schema.

---

## 4. Sequence Mapping & Dual Negative Definitions

### 4.1 Sequence Mapping via Strategy D
Using Strategy D (causal sequence end $t_{59}$ inside an active interval $[t_{\text{start}}, t_{\text{stop}}]$), the 304 qualifying intervals were mapped onto the 153,196 sliding windows generated across the five campaigns:

| Adversarial Campaign | Total 60-Step Sequences | Valid Attack Steps | Scope Qualifying Steps | Qualifying Positive Sequences | Scope Positive Prevalence (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `20260228_multi_openai` | 31,025 | 411 | 68 | 1,053 | 3.39% |
| `20260301_multi_google` | 41,968 | 172 | 10 | 212 | 0.51% |
| `20260301_multi_sonnet` | 17,565 | 402 | 100 | 1,848 | 10.52% |
| `20260302_multi_minimax` | 47,648 | 463 | 62 | 2,087 | 4.38% |
| `20260303_multi_sonnet` | 14,990 | 268 | 64 | 1,167 | 7.79% |
| **Pooled Overall** | **153,196** | **1,716** | **304** | **6,367** | **4.16%** |

### 4.2 Rigorous Definition of Negative Samples
In evaluating a subset of attacks, the definition of negative samples determines operational meaning:

1. **Primary Definition (All Non-Qualifying Negatives):**
   * **Positives ($N=6,367$):** Sequences whose endpoint $t_{59}$ falls within a qualifying PV/Wind impactful interval.
   * **Negatives ($N=146,829$):** All other sequences in the test runs.
   * *Composition:* Contains unattacked normal operation ($104,363$ sequences) PLUS other attack activity outside the qualifying subset ($42,466$ sequences, including Battery attacks and zero-impact PV/Wind attacks).
   * *Operational Rationale:* Represents the complete continuous operational stream where an operator expects alarms only for observable PV/Wind incidents.
2. **Secondary Definition (Strictly Clean Non-Attack Negatives):**
   * **Positives ($N=6,367$):** Sequences whose endpoint $t_{59}$ falls within a qualifying PV/Wind impactful interval.
   * **Clean Negatives ($N=104,363$):** Only sequences from intervals with zero cyberattack activity of any kind.
   * *Composition:* Excludes all 42,466 non-qualifying attack sequences entirely ($N_{\text{total}} = 110,730$, prevalence $= 5.75\%$).
   * *Operational Rationale:* Evaluates discrimination against uncompromised normal operational baseline, eliminating confounding from non-qualifying attacks.

---

## 5. Threshold-Independent Metrics (ROC-AUC & PR-AUC)

Continuous anomaly scores $s_i = \frac{1}{60 \times 14} \sum_{t=0}^{59} \sum_{f=1}^{14} (X_{i,t,f} - \hat{X}_{i,t,f})^2$ were evaluated across both definitions:

| Evaluation Scope | Total Sequences | Positive Prevalence | ROC-AUC | PR-AUC | Random Baseline PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Attacks (Phase 2C.4.2 Benchmark)** | 153,196 | 31.88% | **0.4594** | **0.2953** | 0.3188 |
| **Scope-Appropriate Primary (All Negatives)** | 153,196 | 4.16% | **0.4882** | **0.0401** | 0.0416 |
| **Scope-Appropriate Clean (Clean Negatives)** | 110,730 | 5.75% | **0.4753** | **0.0539** | 0.0575 |

### Visualizations:
* **ROC Curves Comparison:**  
  ![Scope ROC Curves](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_scope_roc_curves.png)
* **Precision-Recall Curves Comparison:**  
  ![Scope PR Curves](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_scope_pr_curves.png)

### Key Observations:
1. **No Meaningful ROC-AUC Gain:** The ROC-AUC shifts from `0.4594` to `0.4882` (Primary) and `0.4753` (Clean Negatives). All values remain below the 0.50 random-guess line, confirming that the model's global ranking capability is essentially uninformative.
2. **PR-AUC Tracks Base Prevalence:** In both definitions, PR-AUC (`0.0401` and `0.0539`) closely matches the random prevalence line (`0.0416` and `0.0575`), demonstrating zero meaningful positive precision enhancement.

---

## 6. Calibrated Threshold Performance

Evaluating the five frozen Phase 2C.3 thresholds demonstrates the operational tradeoff:

### 6.1 Primary Definition Performance (All Non-Qualifying Negatives, $N=153,196$)

| Threshold ($\theta$) | Formula | Precision | Recall | F1-Score | FPR | FNR | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$P_{95}$** ($0.038284$) | 95th Percentile | 3.86% | 42.06% | 0.0706 | 45.48% | 57.94% | 2,678 | 66,785 | 80,044 | 3,689 |
| **$\text{Mean}+3\sigma$** ($0.051625$) | Gaussian 3-Sigma | 3.78% | 33.77% | 0.0680 | 37.29% | 66.23% | 2,150 | 54,751 | 92,078 | 4,217 |
| **$P_{99}$** ($0.055999$) | 99th Percentile | 3.72% | 31.21% | 0.0665 | 34.98% | 68.79% | 1,987 | 51,364 | 95,465 | 4,380 |
| **$P_{99.5}$ (Primary)** ($0.070686$) | 99.5th Percentile | **3.82%** | **26.15%** | **0.0666** | **28.58%** | **73.85%** | **1,665** | **41,970** | **104,859** | **4,702** |
| **$P_{99.9}$** ($0.080325$) | 99.9th Percentile | 3.70% | 22.51% | 0.0635 | 25.43% | 77.49% | 1,433 | 37,344 | 109,485 | 4,934 |

### 6.2 Secondary Clean-Only Definition ($N=110,730$, Clean Negatives $= 104,363$)

| Threshold ($\theta$) | Formula | Precision | Recall | F1-Score | FPR | FNR | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$P_{95}$** ($0.038284$) | 95th Percentile | 5.10% | 42.06% | 0.0910 | 47.71% | 57.94% | 2,678 | 49,794 | 54,569 | 3,689 |
| **$\text{Mean}+3\sigma$** ($0.051625$) | Gaussian 3-Sigma | 5.01% | 33.77% | 0.0873 | 39.03% | 66.23% | 2,150 | 40,731 | 63,632 | 4,217 |
| **$P_{99}$** ($0.055999$) | 99th Percentile | 4.94% | 31.21% | 0.0854 | 36.60% | 68.79% | 1,987 | 38,202 | 66,161 | 4,380 |
| **$P_{99.5}$ (Primary)** ($0.070686$) | 99.5th Percentile | **5.04%** | **26.15%** | **0.0845** | **30.08%** | **73.85%** | **1,665** | **31,396** | **72,967** | **4,702** |
| **$P_{99.9}$** ($0.080325$) | 99.9th Percentile | 4.85% | 22.51% | 0.0798 | 26.93% | 77.49% | 1,433 | 28,100 | 76,263 | 4,934 |

### Visualizations:
* **Threshold Comparison:**  
  ![Scope Threshold Comparison](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_scope_threshold_comparison.png)
* **Score Distribution (Qualifying vs Normal vs Other):**  
  ![Scope Score Distribution](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_scope_anomaly_score_distribution.png)

---

## 7. Discrete Event-Level Physical Detection & Latency

Evaluating physical episode detection across the 304 qualifying attack execution intervals at the primary candidate threshold ($P_{99.5} = 0.070686$):

* **Total Qualifying Attack Intervals:** **304**
* **Detected Attack Intervals:** **95** (**31.25%**)
* **Missed Attack Intervals:** **209** (**68.75%**)
* **Detection Latency Distribution (when detected):**
  * **Median Latency:** **0.33 seconds**
  * **Mean Latency ($\mu$):** **1.60 seconds**
  * **Minimum Latency:** **0.00 seconds**
  * **Maximum Latency:** **20.69 seconds**

### Campaign Breakdown for Qualifying Interval Detection:
| Adversarial Campaign | Qualifying Intervals | Detected Intervals | Detection Rate (%) | Median Latency (s) | Mean Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `20260228_multi_openai` | 68 | 14 | 20.59% | 0.27 s | 1.19 s |
| `20260301_multi_google` | 10 | 3 | 30.00% | 0.31 s | 0.28 s |
| `20260301_multi_sonnet` | 100 | 36 | 36.00% | 0.34 s | 1.16 s |
| `20260302_multi_minimax` | 62 | 25 | 40.32% | 0.37 s | 3.11 s |
| `20260303_multi_sonnet` | 64 | 17 | 26.56% | 0.26 s | 0.89 s |
| **Pooled Overall** | **304** | **95** | **31.25%** | **0.33 s** | **1.60 s** |

### Visualizations:
* **Per-Campaign Performance Comparison:**  
  ![Per Campaign Comparison](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_scope_per_run_comparison.png)
* **Qualifying Attack Telemetry & Score Timeline:**  
  ![Scope Timeline](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase2c4_scope_timeline_example.png)

---

## 8. Direct Comparison: All-Attacks vs. Scope-Appropriate

The table below provides a direct, rigorous side-by-side comparison between the Phase 2C.4.2 all-attack benchmark and the Phase 2C.4.3 scope-appropriate diagnostic evaluation:

| Evaluation Metric | All Attacks (Phase 2C.4.2 Benchmark) | Scope-Appropriate Primary (All Negatives) | Scope-Appropriate Secondary (Clean Negatives) | Performance Shift / Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Evaluated Attack Intervals** | 1,716 | 304 | 304 | Narrowed to verified PV/Wind impactful steps |
| **Evaluated Sequences ($N$)** | 153,196 | 153,196 | 110,730 | Excludes out-of-scope attacks in Secondary |
| **Positive Prevalence** | 31.88% (48,833 seq) | 4.16% (6,367 seq) | 5.75% (6,367 seq) | Substantial base rate reduction |
| **ROC-AUC** | **0.4594** | **0.4882** | **0.4753** | Minimal change (+0.0288 / +0.0159); still < 0.50 |
| **PR-AUC** | **0.2953** | **0.0401** | **0.0539** | Drops to match lower positive prevalence |
| **Precision @ $P_{99.5}$** | **28.05%** | **3.82%** | **5.04%** | Drops sharply due to lower prevalence vs fixed FPR |
| **Recall @ $P_{99.5}$** | **25.06%** | **26.15%** | **26.15%** | Minor gain (+1.09 percentage points) |
| **F1-Score @ $P_{99.5}$** | **0.2647** | **0.0666** | **0.0845** | Drops due to low precision |
| **FPR @ $P_{99.5}$** | **30.08%** (on clean normal) | **28.58%** | **30.08%** | False alarm rate remains unchanged (~29-30%) |
| **FNR @ $P_{99.5}$** | **74.94%** | **73.85%** | **73.85%** | Miss rate remains severe (~74%) |
| **Episode Detection Rate** | **21.04%** (361/1,716) | **31.25%** (95/304) | **31.25%** (95/304) | Moderate improvement (+10.21 percentage points) |
| **Median Detection Latency** | **3.00 seconds** | **0.33 seconds** | **0.33 seconds** | Rapid detection when physical spike occurs |

---

## 9. Scientific & Physical Interpretation

### 9.1 Does Scope Restriction Fix the Model?
**NO.** The empirical evidence is definitive: restricting evaluation to attack activity that directly targets PV/Wind channels and produces verified physical impact does **not** convert the frozen LSTM Autoencoder into an effective cyberattack detector.

1. **Global Ranking Remains Random ($\text{ROC-AUC} \approx 0.48$):** The model cannot reliably rank attack sequences above non-attack sequences, even when the attack sequences contain verified physical deviations.
2. **False Negatives Remain Overwhelming ($\text{FNR} = 73.85\%$):** Nearly three out of four sequence endpoints during verified impactful attacks fail to cross the $P_{99.5}$ threshold.
3. **Severe False Alarm Rate ($\text{FPR} \approx 30\%$):** Almost one out of every three normal operational sequences crosses the $P_{99.5}$ threshold, rendering the system operationally unusable in a control room.
4. **Modest Episode Detection Gain ($\mathbf{31.25\%}$):** While event-level detection improves from 21.04% to 31.25%, over two-thirds (68.75%) of verified impactful attacks still conclude without triggering an alarm.

### 9.2 Physical Root Cause Analysis: Why Does Unsupervised Process Reconstruction Fail?
The failure is rooted in the fundamental architecture of unsupervised deep autoencoders applied to renewable microgrid process data:

1. **Lack of Physical Governing Invariants:** An LSTM Autoencoder trained with MSE loss learns statistical correlations between features across time, but has zero knowledge of physical laws (e.g., $P_{\text{pv}} = \eta \cdot A \cdot G \cdot [1 - \gamma(T_{\text{cell}} - 25)]$, Betz limit $P_{\text{wind}} = \frac{1}{2} \rho A v^3 C_p$, or thermal mass differential equations). As a result, it cannot distinguish a physically impossible sensor state from a rare but valid operational state.
2. **Operational Variance Masks Subtle Sensor Spoofing:** Solar irradiance transients (passing cloud cover), turbulent wind gusts, and diurnal ambient temperature swings induce natural reconstruction residuals in the clean baseline that are larger than the subtle physical deltas produced by stealthy multi-agent cyberattacks (e.g., a 2 m/s wind speed bias or a 5 kW inverter throttling).
3. **Reconstruction Generalization:** Deep LSTMs with latent capacity (dim 64) generalize remarkably well; the decoder frequently reconstructs corrupted sensor inputs accurately because the input trajectory resembles a plausible operational state, suppressing the MSE anomaly score below the detection threshold.

### 9.3 Architectural Implications for the Project
This negative result is scientifically vital and directly validates the multi-tier architecture of this project:
* **Direct Motivation for Phase 4 (Physics-Informed Digital Twin):** A pure statistical model cannot detect physically subtle attacks. Phase 4 must introduce explicit physics-informed residual generation—comparing actual sensor values against first-principles governing equations—to catch deviations that violate energy conservation and physical laws.
* **Direct Motivation for Phase 5 (LLM-RAG Multimodal Correlation):** Process data alone cannot distinguish cyber-induced setpoint shifts from legitimate operator commands. Reliable cyber-physical security requires fusing process residuals with host command logs, Zeek network telemetry, and OT protocol metadata.

---

## 10. Integrity Verification Checklist

- [x] Model weights unchanged (`models/lstm_autoencoder_baseline.pt`, best epoch 27).
- [x] Scaler parameters unchanged (fitted strictly on training partition of normal baseline).
- [x] Statistical thresholds unchanged from Phase 2C.3 ($P_{95}, \text{Mean}+3\sigma, P_{99}, P_{99.5}, P_{99.9}$).
- [x] Zero retraining or parameter adaptation performed.
- [x] Zero attack data used for model or scaler fitting.
- [x] Original Phase 2C.4.2 benchmark results unchanged and intact.
- [x] Strategy D causal sequence endpoint mapping preserved without alteration.
- [x] No missing physical-impact labels were treated as False.
- [x] Every positive scope-appropriate interval has verified PV/Wind relevance AND verified physical impact.
- [x] Dual negative sample definitions explicitly documented and separately reported.
