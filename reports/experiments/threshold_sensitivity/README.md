# Threshold Sensitivity Experiments: Archive & Historical Evidence

**Archive Directory:** `reports/experiments/threshold_sensitivity/`  
**Evaluation Standard:** Strategy D (Sequence End Timestamp $t_{59}$, Zero Future Leakage)  
**Scope Benchmark:** 5 Multi-Agent Adversarial Attack Campaigns (304 Qualifying Impactful Episodes, 153,196 Sequences, 6,367 Positive Sequences)  
**Calibration Baseline:** `20260225_normal` (Uncompromised Normal Operational Training Telemetry)  
**Status:** Completed Offline Sensitivity Analysis. **No models were retrained.**

---

## 1. Objective

The objective of this experimental series was to systematically evaluate whether moderately lower detection thresholds could improve attack-episode coverage and sequence recall across the parallel Layer 1 (Causal TCN Autoencoder) and Layer 2 (Physical XGBoost Regressors) detection branches, while quantifying the false-alarm burden imposed on downstream Evidence Fusion, Digital Twin state validation, and RAG/LLM contextual triage.

---

## 2. Experimental Setup

The evaluation preserved all production invariants:
- **Zero Retraining:** The active Causal TCN autoencoder checkpoint ([`models/tcn_autoencoder_baseline.pt`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/tcn_autoencoder_baseline.pt)) and the four Layer 2 XGBoost regression models ([`models/layer2_*_xgb.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/)) were strictly frozen.
- **Normal Distribution Exclusivity:** All candidate thresholds were calculated from empirical percentiles of reconstruction errors and residuals on uncompromised normal operational data (`20260225_normal`). **Zero attack data or labels were used to derive any threshold.**
- **Strategy D Protocol:** Window alignment uses the newest timestamp $t_{59}$. An attack episode is counted as detected if at least one sequence within its verified active interval triggers an alert.

---

## 3. Threshold Candidates

### Layer 1 (Causal TCN Autoencoder — Mean Feature MSE)
- **$P_{99.5}$ (Baseline):** $0.000970433$
- **$P_{99}$ (Moderate):** $0.000895460$
- **$P_{95}$ (Aggressive):** $0.000615804$

### Layer 2 (Four Physical Relationships)
1. **PV Inverter ($P_{\text{dc}} \to P_{\text{ac}}$, kW):**
   - $P_{99.5}$: $10.3677\text{ kW}$
   - $P_{99}$: $9.3677\text{ kW}$
   - $P_{95}$: $7.0729\text{ kW}$
2. **Wind Anemometer ($v_a \to v_b$, m/s):**
   - $P_{99.5}$: $1.2283\text{ m/s}$
   - $P_{99}$: $1.0186\text{ m/s}$
   - $P_{95}$: $0.5411\text{ m/s}$
3. **Wind Nacelle Temperature ($T_a \to T_b$, °C):**
   - $P_{99.5}$: $1.9437^\circ\text{C}$
   - $P_{99}$: $1.7402^\circ\text{C}$
   - $P_{95}$: $0.8566^\circ\text{C}$
4. **PV Thermal ($T_{\text{air}} \to T_{\text{cell}}$, °C):**
   - $P_{99.5}$: $15.2889^\circ\text{C}$
   - $P_{99}$: $14.9602^\circ\text{C}$
   - $P_{95}$: $10.3477^\circ\text{C}$

---

## 4. Seven Evidence Fusion Configurations Evaluated

Evidence Fusion was tested across seven distinct configuration pairings:
- **A (Baseline):** $L_1 P_{99.5} + L_2 P_{99.5}$
- **B (Symmetric Moderate):** $L_1 P_{99} + L_2 P_{99}$
- **C (Symmetric Aggressive):** $L_1 P_{95} + L_2 P_{95}$
- **D (Asymmetric Temporal Moderate):** $L_1 P_{99} + L_2 P_{99.5}$
- **E (Asymmetric Physical Moderate):** $L_1 P_{99.5} + L_2 P_{99}$
- **F (Asymmetric Physical Aggressive):** $L_1 P_{99} + L_2 P_{95}$
- **G (Asymmetric Temporal Aggressive):** $L_1 P_{95} + L_2 P_{99}$

---

## 5. Consolidated Results Table

| Config | L1 Tier | L2 Tier | Precision | Recall | F1 Score | Test FPR | FNR | Episodes Detected (of 304) | Episode Det. Rate | Agreement Episodes | Agreement Det. Rate | Normal Val Agreement Exceedance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | $P_{99.5}$ | $P_{99.5}$ | 0.0525 | 0.3746 | 0.0921 | 0.4125 | 0.6254 | 141 | 46.38% | 78 | 25.66% | 6.24% |
| **B** | $P_{99}$ | $P_{99}$ | 0.0539 | 0.4115 | 0.0953 | 0.4405 | 0.5885 | 153 | 50.33% | 93 | 30.59% | 6.62% |
| **C** | $P_{95}$ | $P_{95}$ | 0.0554 | **0.5513** | 0.1006 | 0.5738 | 0.4487 | **190** | **62.50%** | **147** | **48.36%** | 18.93% |
| **D** | $P_{99}$ | $P_{99.5}$ | 0.0524 | 0.3843 | 0.0922 | 0.4244 | 0.6157 | 146 | 48.03% | 82 | 26.97% | 6.54% |
| **E** | $P_{99.5}$ | $P_{99}$ | 0.0534 | 0.3959 | 0.0941 | 0.4281 | 0.6041 | 146 | 48.03% | 89 | 29.28% | 6.28% |
| **F** | **$P_{99}$** | **$P_{95}$** | **0.0572** | 0.5002 | **0.1026** | 0.5033 | 0.4998 | 169 | 55.59% | **124** | **40.79%** | 11.62% |
| **G** | $P_{95}$ | $P_{99}$ | 0.0546 | 0.4773 | 0.0981 | 0.5038 | 0.5227 | 175 | 57.57% | 104 | 34.21% | 10.17% |

---

## 6. Selected Configuration

**Selected Operating Point for Phase 4:** **Configuration F ($L_1 P_{99} + L_2 P_{95}$)**.

---

## 7. Reason for Selection

1. **Top Overall F1 Score:** Achieved an F1 score of **$0.1026$**, the highest among all 7 evaluated configurations.
2. **Substantial Corroboration Gain:** Detected **$124$ dual-branch agreement episodes ($40.79\%$)**, representing a $+33.3\%$ improvement over Configuration B ($93$ episodes) and $+59.0\%$ over baseline A ($78$ episodes).
3. **Physical Sensitivity Superiority:** In cyber-physical attacks, physical sensor manipulations violate energy balance and nacelle consistency. Lowering Layer 2 thresholds catches subtle physical discrepancies without being misled by weather-driven temporal transients.
4. **Fast Detection Latency:** Achieved median detection latency of $0.2739\text{ s}$ and mean latency of $0.6436\text{ s}$, providing the fastest alerting profile.

---

## 8. Limitations & Trade-Offs

- **False Positive Penalty:** Configuration F exhibits an $11.62\%$ normal validation agreement exceedance rate and a $50.33\%$ campaign FPR.
- **Dependency on Downstream Layers:** Configuration F is not intended as an autonomous alerting filter by itself. It relies on the downstream Digital Twin (physics simulation) and RAG/LLM (contextual incident triage) to filter physically explicable operational transients.
- **Reference Roles:** Configuration C ($P_{95}/P_{95}$) remains the high-sensitivity benchmark, while Configuration B ($P_{99}/P_{99}$) remains the conservative reference.

---

## 9. Integrity Checks Confirmed

- Zero model retraining occurred.
- Layer 1 checkpoint hash verified: `0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22`.
- All 14 Layer 1 features and Causal TCN architecture preserved.
- Layer 2 four XGBoost models and training data untouched.
- All candidate thresholds derived strictly from normal operational telemetry (`20260225_normal`). Zero attack data used for threshold derivation.
- No LSTM/GRU or supervised XGBoost code restored.
- Digital Twin, RAG/LLM, and Dashboard components remained untouched.

---

## 10. Next Project Phase: Digital Twin Integration

With Configuration F selected as the operating baseline:
1. **Digital Twin (Simscape / State Estimation):** Consume candidate alerts and evaluate whether observed states obey non-linear differential equations and power flow constraints under current operating setpoints.
2. **Contextual Triaging:** Separate true cyber-physical intrusions from cloud shadowing, turbine wake turbulence, or automated inverter clipping.
3. **RAG + LLM Integration:** Synthesize verified physical inconsistencies into structured incident reports for SOC operators.

---

## 11. Archived Documents & Artifacts

- **Detailed Symmetric Sensitivity Report:** [`threshold_sensitivity_results.md`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/threshold_sensitivity/threshold_sensitivity_results.md)
- **Detailed Asymmetric Sensitivity Report:** [`asymmetric_results.md`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/threshold_sensitivity/asymmetric_results.md)
- **Serialized JSON Metrics:** [`threshold_sensitivity_results.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/threshold_sensitivity/threshold_sensitivity_results.json), [`asymmetric_results.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/threshold_sensitivity/asymmetric_results.json)
- **Diagnostic Visualizations:** [`figures/`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/threshold_sensitivity/figures/)
- **Reusable Experiment Code:** [`experiments/threshold_sensitivity/`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/experiments/threshold_sensitivity/)
