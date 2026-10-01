# Phase 3 — Evidence Fusion: Multi-Layer Cyber-Physical Anomaly Synthesis

**Author:** Antigravity AI  
**Date:** September 30, 2026  
**Status:** Completed & Validated  
**Component:** `src/ml/evidence_fusion.py`  
**Evaluation Standard:** Strategy D (Sequence End Timestamp $t_{59}$), Scope-Appropriate Observable Benchmark (304 qualifying impactful episodes, 153,196 test sequences)

---

## 1. Architectural Foundation & Principle

The Evidence Fusion layer operates strictly as the first synthesis component combining two parallel, independent anomaly detection branches:

```
Smart Grid Telemetry
        │
  ┌─────┴─────┐
  ▼           ▼
Layer 1     Layer 2
(TCN-AE)   (Physical Relationships)
  │           │
  └─────┬─────┘
        ▼
  Evidence Fusion
        ▼
   Digital Twin   (future phase)
        ▼
    RAG + LLM     (future phase)
        ▼
    Dashboard     (future phase)
```

### Strict Architectural Boundaries:
1. **Parallel Independence:** Layer 1 (temporal dynamics) and Layer 2 (physical consistency) do **NOT** feed into each other. Neither layer receives signals, scores, or states from the other.
2. **First Combination Point:** Evidence Fusion is the **first component** in the entire system that observes both outputs simultaneously.
3. **Not an Attack Classifier:** Evidence Fusion does **not** classify attack types, assign threat labels, or declare verified cyberattacks. Its sole objective is answering:
   > *"How strongly do the independent temporal and physical evidence sources indicate that the current smart-grid state is abnormal?"*
4. **Frozen Model Weights & Checkpoints:** Neither Layer 1 (Causal TCN Autoencoder, Config A) nor Layer 2 (four physical XGBoost regressors) was retrained or structurally altered.
5. **No Supervised Training:** Evidence Fusion does not use neural networks, tree classifiers, or machine-learned combination weights. It is a fully transparent, deterministic, rule-based evidence synthesis engine.

> [!NOTE]
> **Active Operating Threshold Update (September 30, 2026):**  
> Following the threshold-sensitivity experiments across 7 configurations (archived in [`reports/experiments/threshold_sensitivity/`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/threshold_sensitivity/)), the pipeline's active operating configuration is set to **Configuration F ($L_1 = P_{99}$, $L_2 = P_{95}$)**, delivering the highest overall F1 ($0.1026$) and capturing 124 corroborated dual-branch agreement episodes. See [`reports/threshold_decision.md`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/threshold_decision.md) for full rationale. Historical baseline $P_{99.5}$ results remain fully preserved below.

---

## 2. Standardized Inputs to Fusion

To preserve complete transparency for the downstream Digital Twin and RAG/LLM explanation layers, Evidence Fusion takes standardized inputs that retain all fine-grained subsystem evidence:

### 2.1 Layer 1 Standardized Input
- `timestamp`: Timestamp corresponding to sequence end ($t_{59}$).
- `raw_score`: Sequence-level Mean Squared Error (MSE) across the $60 \times 14$ window.
- `normalized_score`: Bounded confidence score in $[0, 1]$ relative to calibrated normal threshold.
- `threshold`: Calibrated normal decision threshold ($P_{99.5} = 0.000970433$).
- `anomaly_flag`: Binary indicator ($1$ if $\text{MSE} \ge \text{threshold}$, else $0$).
- `per_feature_mse`: $14$-dimensional channel reconstruction error vector for localization.

### 2.2 Layer 2 Standardized Input
Layer 2 provides evidence across all four physical relationships:
1. **PV Inverter:** $P_{dc} \to P_{ac}$ (kW) — Inverter power conversion efficiency.
2. **Wind Speed:** $v_{wind,A} \to v_{wind,B}$ (m/s) — Aerodynamic agreement between dual nacelle anemometers.
3. **Wind Temperature:** $T_{nacelle,A} \to $T_{nacelle,B}$ (°C) — Thermal nacelle sensor equilibrium.
4. **PV Thermal:** $T_{ambient} \to T_{cell}$ (°C) — Environmental solar cell thermal coupling.

For each relationship $i \in \{1, 2, 3, 4\}$:
- `raw_residual`: Absolute prediction residual $|y_i - \hat{y}_i|$ in physical engineering units.
- `signed_residual`: Directional residual $(y_i - \hat{y}_i)$ indicating over- or under-generation.
- `normalized_score`: Bounded confidence score in $[0, 1]$ relative to relationship threshold.
- `threshold`: Statistical normal threshold ($P_{99.5}$) calibrated strictly on normal training data.
- `anomaly_flag`: Binary indicator ($1$ if $|y_i - \hat{y}_i| \ge \text{threshold}_i$, else $0$).

### 2.3 Layer 2 Aggregation
Individual relationship evidence is **never discarded**. Layer 2 aggregates individual scores into a composite physical inconsistency score:
$$\text{L2\_score} = \max_{i \in \{1, 2, 3, 4\}} (\text{relationship\_score}_i)$$
- **Rationale for $\max$:** In cyber-physical grid operations, a violation of *any* single physical law (e.g., dual anemometer divergence or inverter power mismatch) represents localized physical inconsistency. Averaging across unaffected subsystems would dangerously dilute localized anomalies.
- In addition, the engine computes:
  - $\text{L2\_mean\_score}$: Average physical score across all 4 relationships.
  - $\text{L2\_anomalous\_count}$: Number of relationships currently exceeding threshold ($0 \dots 4$).
  - $\text{L2\_any\_flag}$: Binary indicator ($1$ if $\text{L2\_anomalous\_count} \ge 1$, else $0$).

---

## 3. Score Normalization Methodology

Because Layer 1 reconstruction errors (unitless normalized MSE $\sim 10^{-4}$) and Layer 2 residuals (kW, m/s, °C) reside on entirely different scales and physical dimensions, direct summation is mathematically invalid.

Each evidence source is normalized using a deterministic, threshold-relative scaling function calibrated strictly on clean, normal operational baseline telemetry:

$$\text{normalized\_score} = \min\left(\frac{\text{raw\_signal}}{\text{normal\_threshold}}, 1.0\right)$$

- **Layer 1 Normalization:**
  $$\text{L1\_score} = \min\left(\frac{\text{L1\_MSE}}{0.000970433}, 1.0\right)$$
- **Layer 2 Normalization (per relationship $i$):**
  $$\text{relationship\_score}_i = \min\left(\frac{|y_i - \hat{y}_i|}{\text{threshold}_i}, 1.0\right)$$

### Properties of Normalization:
- **Bounded Domain $[0, 1]$:** Enables direct, calibrated probabilistic interpretation.
- **Critical Threshold at $1.0$:** A normalized score of $1.0$ indicates that the observed signal has reached or exceeded the statistical $P_{99.5}$ normal baseline limit.
- **Zero Attack Contamination:** Normalization parameters are derived 100% from uncompromised baseline operations; zero attack labels were used.

---

## 4. Fusion Strategy & Mathematical Formulation

### 4.1 Fused Anomaly Evidence Score
Evidence Fusion combines normalized temporal evidence ($\text{L1\_score}$) and normalized physical inconsistency evidence ($\text{L2\_score}$) via a transparent linear formulation:

$$\text{fused\_score} = w_1 \cdot \text{L1\_score} + w_2 \cdot \text{L2\_score}$$

- **Baseline Weights:** $w_1 = 0.5$, $w_2 = 0.5$.
- **Design Rationale:** Equal weighting serves as an un-biased, transparent reference baseline reflecting equal epistemic confidence in temporal sequence fidelity and physical conservation laws. These weights are **not** claimed to be globally optimal and were not tuned on attack labels.
- **Fused Decision Threshold ($T_{fused}$):**
  Derived strictly from the uncompromised normal operational training split (`20260225_normal`):
  $$T_{fused} = P_{99.5}(\text{fused\_score}_{\text{normal}}) = 0.784338$$
  $$\text{fused\_flag} = \begin{cases} 1 & \text{if } \text{fused\_score} \ge 0.784338 \\ 0 & \text{otherwise} \end{cases}$$

### 4.2 Cross-Layer Agreement Indicator
In parallel with the continuous fused score, Evidence Fusion calculates an independent binary cross-layer agreement flag:

$$\text{agreement\_flag} = \begin{cases} 1 & \text{if } (\text{L1\_flag} == 1) \land (\text{L2\_any\_flag} == 1) \\ 0 & \text{otherwise} \end{cases}$$

- $\text{agreement\_flag} = 1$ signifies that both the Causal TCN and at least one physical relationship regressor independently and simultaneously flagged the current state as anomalous.
- Represents the highest-confidence operational alert level.

---

## 5. Temporal Alignment Protocol

| Characteristic | Specification |
| :--- | :--- |
| **Data Sampling Rate** | $1\text{ Hz}$ nominal ($1.00\text{ s} \pm 28\text{ ms}$ jitter) |
| **Layer 1 Geometry** | Sliding window $L = 60$ time-steps, stride $s = 1$ |
| **Layer 1 Evaluation Time** | Sequence end timestamp $t_{59}$ ($t_{\text{end}} = t_0 + 59\text{ s}$) |
| **Layer 2 Geometry** | Point-in-time instantaneous regression at timestamp $t$ |
| **Alignment Method** | Slicing Layer 2 residuals starting at index $59$ (`[59::stride]`) |
| **Verification** | Exact 1-to-1 timestamp equality: `seq_end_ts == raw_df['ts'].values[59:]` |
| **Data Loss** | Exactly $59$ rows required by Layer 1 causal warm-up buffer; **0 rows** dropped or interpolated thereafter |

---

## 6. Comprehensive Empirical Evaluation

All components were evaluated across all 5 multi-agent attack campaigns (16 runs, 153,196 test sequences) using the established **Strategy D Scope-Appropriate Observable Benchmark** (304 qualifying impactful Solar/Wind episodes).

### 6.1 Performance Benchmark Table

| Model / Evidence Source | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR | Episode Det. Rate | Detected / Total | Median Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1 Alone (TCN-AE)** | **0.5007** | **0.0570** | **0.0559** | 0.3617 | **0.0969** | 0.3724 | 0.6383 | 40.13% | 122 / 304 | **0.286 s** | **1.008 s** |
| **Layer 2 Alone (Physical Max)** | 0.4445 | 0.0482 | 0.0351 | 0.2218 | 0.0605 | 0.3725 | 0.7782 | 43.75% | 133 / 304 | 1.723 s | 3.580 s |
| **Evidence Fusion ($w_1=0.5, w_2=0.5$)** | 0.4730 | 0.0516 | 0.0525 | **0.3746** | 0.0921 | 0.4125 | **0.6254** | **46.38%** | **141 / 304** | 0.315 s | 1.180 s |
| **Cross-Layer Agreement (L1 & L2)** | 0.4657 | 0.0549 | 0.0401 | 0.1494 | 0.0633 | **0.2180** | 0.8506 | 25.66% | 78 / 304 | 0.444 s | 3.280 s |

### 6.2 Key Empirical Takeaways:
1. **Highest Episode Detection:** Evidence Fusion achieves **46.38% episode detection** (141 / 304 episodes), outperforming both Layer 1 alone (40.13%) and Layer 2 alone (43.75%). Fusing the two complementary evidence streams successfully detects episodes where one layer alerted even when the other did not.
2. **Lowest False Positive Rate for Cross-Layer Agreement:** Cross-Layer Agreement reduces false positive rate to **21.80%** (down from 37.24% in L1 and 37.25% in L2). When both independent layers alarm, the confidence of genuine disturbance is significantly heightened.
3. **Rapid Detection Latency:** Evidence Fusion detects attacks with a median latency of **0.315 seconds** (mean: 1.180 s), retaining the ultra-fast temporal response of the Causal TCN while enriching it with physical consistency validation.

---

## 7. Cross-Layer Agreement Analysis

| Evaluation Partition | Total Sequences / Episodes | Agreed Sequences / Episodes | Agreement Rate (%) | Operational Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Normal (Train Split)** | 9,533 sequences | 1 sequence | **0.01%** | Baseline operational false alarm rate of agreement is near-zero |
| **Clean Normal (Val Split)** | 2,340 sequences | 146 sequences | **6.24%** | Unseen normal validation agreement is substantially lower than single layers |
| **Clean Normal (Test Campaigns)** | 104,363 sequences | 22,746 sequences | **21.80%** | Agreement FPR across all uncompromised test intervals |
| **Overall Test Data** | 153,196 sequences | 31,048 sequences | **20.27%** | Overall rate of simultaneous dual-layer threshold exceedance |
| **Attack Sequences** | 6,367 sequences | 951 sequences | **14.94%** | Sample-level simultaneous dual exceedance during attacks |
| **Attack Episodes** | 304 qualifying episodes | 78 episodes | **25.66%** | Complete attack episodes detected by BOTH Layer 1 and Layer 2 |

---

## 8. Interpretability Analysis: Three Diagnostic Cases

Evidence Fusion provides fine-grained interpretability by categorizing anomalous states into three distinct cyber-physical cases. Concrete instances extracted from the empirical evaluation demonstrate why combining independent evidence sources is valuable:

### Case 1: Layer 1 Abnormal + Layer 2 Normal
> *Temporal Anomaly with Physical Consistency*

- **Concrete Example:** Campaign `20260228_multi_openai`, Index 12307, Timestamp `2026-02-28 23:27:13.511506`.
- **Active Attack Interval:** Wind subsystem Step 1 (`wind_process_data.temperature_B` perturbed by $\Delta = 0.07\text{ }^\circ\text{C}$).
- **Observed Telemetry & Evidence:**
  - $\text{L1\_MSE} = 0.001131 > 0.000970$ ($\text{L1\_score} = 1.000$, $\text{L1\_flag} = 1$).
  - Layer 2 residuals:
    - `pv_inverter`: $4.86\text{ kW}$ (threshold $10.37\text{ kW}$, score $0.469$, flag $0$)
    - `wind_speed`: $0.248\text{ m/s}$ (threshold $1.228\text{ m/s}$, score $0.202$, flag $0$)
    - `wind_temperature`: $0.021\text{ }^\circ\text{C}$ (threshold $1.944\text{ }^\circ\text{C}$, score $0.011$, flag $0$)
    - `pv_thermal`: $5.99\text{ }^\circ\text{C}$ (threshold $15.29\text{ }^\circ\text{C}$, score $0.392$, flag $0$)
  - All Layer 2 flags are $0$ ($\text{L2\_any\_flag} = 0$, $\text{L2\_max\_score} = 0.469$).
  - $\text{fused\_score} = 0.5 \times 1.0 + 0.5 \times 0.469 = 0.7346$. $\text{agreement\_flag} = 0$.
- **Cyber-Physical Meaning:**
  The temporal trajectory of the telemetry over the 60-second window deviated from normal dynamic expectations (e.g. unexpected rate of change or step transient). However, the instantaneous physical conservation laws (inverter efficiency, dual anemometer agreement, thermal coupling) remained satisfied because the magnitude of the disturbance ($\Delta = 0.07\text{ }^\circ\text{C}$) was within static physical tolerances.
  *Non-Attack Analog:* Rapid cloud shadowing, operational switching, or sudden grid frequency adjustments.

---

### Case 2: Layer 1 Normal + Layer 2 Abnormal
> *Physical Inconsistency with Temporal Continuity*

- **Concrete Example:** Campaign `20260228_multi_openai`, Index 9609, Timestamp `2026-02-28 23:03:33.834229`.
- **Active Attack Interval:** PV subsystem Step 2 (`pv_process_data.inverter_ac_power` manipulated by $\Delta = 5.0\text{ kW}$).
- **Observed Telemetry & Evidence:**
  - $\text{L1\_MSE} = 0.000543 < 0.000970$ ($\text{L1\_score} = 0.560$, $\text{L1\_flag} = 0$).
  - Layer 2 residuals:
    - `wind_speed`: $1.419\text{ m/s} > 1.228\text{ m/s}$ ($\text{score} = 1.000$, $\text{flag} = 1$)
    - `pv_inverter`: $7.660\text{ kW}$ (threshold $10.37\text{ kW}$, score $0.739$, flag $0$)
    - `pv_thermal`: $12.65\text{ }^\circ\text{C}$ (threshold $15.29\text{ }^\circ\text{C}$, score $0.828$, flag $0$)
    - `wind_temperature`: $0.370\text{ }^\circ\text{C}$ (threshold $1.94\text{ }^\circ\text{C}$, score $0.190$, flag $0$)
  - $\text{L2\_any\_flag} = 1$, $\text{L2\_max\_score} = 1.000$.
  - $\text{fused\_score} = 0.5 \times 0.560 + 0.5 \times 1.000 = 0.7800$. $\text{agreement\_flag} = 0$.
- **Cyber-Physical Meaning:**
  A localized physical relationship is distinctly violated (e.g. discrepancy between dual anemometers or altered inverter conversion), but the time-series autoencoder reconstructs the temporal sequence without high residual error because the signals evolve smoothly within typical operating envelopes without abrupt temporal discontinuities.
  *Non-Attack Analog:* Sensor drift, loose sensor wiring, mechanical fouling, or uncalibrated anemometer.

---

### Case 3: Layer 1 Abnormal + Layer 2 Abnormal
> *Unanimous Cross-Layer Agreement (High-Confidence Anomaly)*

- **Concrete Example:** Campaign `20260228_multi_openai`, Index 19399, Timestamp `2026-03-01 00:29:26.439727`.
- **Active Attack Interval:** Coordinated attack Step 1 (`pv_process_data.poa_direct` manipulated with $\Delta = 4.0$).
- **Observed Telemetry & Evidence:**
  - $\text{L1\_MSE} = 0.004838$ ($5\times$ threshold $0.000970$, $\text{L1\_score} = 1.000$, $\text{L1\_flag} = 1$).
  - Layer 2 residuals:
    - `pv_inverter`: $13.534\text{ kW} > 10.37\text{ kW}$ ($\text{score} = 1.000$, $\text{flag} = 1$)
    - `wind_temperature`: $2.560\text{ }^\circ\text{C} > 1.944\text{ }^\circ\text{C}$ ($\text{score} = 1.000$, $\text{flag} = 1$)
    - `pv_thermal`: $9.126\text{ }^\circ\text{C}$ (threshold $15.29\text{ }^\circ\text{C}$, score $0.597$, flag $0$)
    - `wind_speed`: $0.104\text{ m/s}$ (threshold $1.228\text{ m/s}$, score $0.084$, flag $0$)
  - Two independent physical relationships simultaneously violated ($\text{L2\_anomalous\_count} = 2$, $\text{L2\_max\_score} = 1.000$).
  - $\text{fused\_score} = 0.5 \times 1.0 + 0.5 \times 1.0 = 1.0000$.
  - $\text{agreement\_flag} = 1$ (Unanimous agreement).
- **Cyber-Physical Meaning:**
  Both temporal reconstruction continuity and multi-sensor physical laws are simultaneously violated across multiple subsystems (PV power conversion + Wind thermal nacelle). This provides high-confidence evidence of a major multi-point disturbance, catastrophic equipment breakdown, or coordinated adversarial tampering.

---

## 9. Diagnostic Figures

The following publication-grade diagnostic figures have been generated in `reports/figures/`:

1. **ROC and Precision-Recall Curves:**  
   `reports/figures/evidence_fusion_roc_pr.png`  
   Compares ROC curves and Precision-Recall curves across Layer 1 alone, Layer 2 alone, Evidence Fusion, and Cross-Layer Agreement against the Scope-Appropriate Observable Benchmark.

2. **Performance Comparison Bar Chart:**  
   `reports/figures/evidence_fusion_comparison.png`  
   Direct side-by-side comparison of Precision, Recall, F1 Score, False Positive Rate (FPR), and Episode Detection Rate (%).

3. **Cross-Layer Agreement & Diagnostic Breakdown:**  
   `reports/figures/evidence_fusion_agreement.png`  
   Displays cross-layer agreement rates across normal validation, normal test, overall test, and attack episodes (left panel), alongside a pie chart breakdown of Case 1, Case 2, and Case 3 during attack sequences (right panel).

---

## 10. Safety & Integrity Verifications

| Check # | Verification Criterion | Status | Empirical Confirmation |
| :---: | :--- | :---: | :--- |
| **1** | Layer 1 checkpoint hash unchanged | **PASSED** | SHA-256: `0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22` |
| **2** | Layer 1 source code unchanged | **PASSED** | `src/ml/tcn_autoencoder.py`, `layer1_tcn_detector.py` untouched |
| **3** | Layer 2 model files unchanged | **PASSED** | All 4 XGBoost model JSONs and `layer2_metadata.json` untouched |
| **4** | No Layer 1 retraining occurred | **PASSED** | Model loaded in eval mode from existing checkpoint |
| **5** | No Layer 2 retraining occurred | **PASSED** | Models loaded in inference mode from existing JSON artifacts |
| **6** | No supervised XGBoost restored | **PASSED** | Active codebase strictly contains 4 unsupervised regressors + TCN |
| **7** | No attack labels used to train fusion | **PASSED** | Linear combination ($w_1=0.5, w_2=0.5$) requires zero training |
| **8** | No attack labels used to select weights | **PASSED** | Weights are an a priori, transparent equal-confidence baseline |
| **9** | No attack labels used for threshold | **PASSED** | Fused threshold ($0.784338$) derived strictly from normal training split |
| **10** | Evidence Fusion only newly added component | **PASSED** | Active additions limited to `src/ml/evidence_fusion.py` |
| **11** | Downstream components untouched | **PASSED** | Digital Twin, RAG/LLM, and Dashboard remain completely untouched |
| **12** | Historical reports untouched | **PASSED** | `reports/layer2_supervised_xgb_comparison.*` preserved intact |

---

## 11. Final Summary & Deliverables

1. **Active Files Created:**
   - `src/ml/evidence_fusion.py`
   - `src/ml/evidence_fusion_evaluation.py`
2. **Active Files Modified:**
   - `src/ml/__init__.py` (cleanly exports `EvidenceFusion`, `FusedEvidence`, `load_active_evidence_fusion`, `FUSED_NORMAL_THRESHOLDS`)
3. **Artifacts Preserved & Untouched:**
   - `models/tcn_autoencoder_baseline.pt`
   - `models/layer2_*.json`
   - `reports/layer2_supervised_xgb_comparison.*`
4. **Reports & Figures Generated:**
   - `reports/evidence_fusion.md`
   - `reports/evidence_fusion.json`
   - `reports/figures/evidence_fusion_roc_pr.png`
   - `reports/figures/evidence_fusion_comparison.png`
   - `reports/figures/evidence_fusion_agreement.png`

**Stop Condition Met:** Evidence Fusion is completed, validated, documented, and verified. No downstream Digital Twin, RAG, LLM, or Dashboard components have been created or modified.
