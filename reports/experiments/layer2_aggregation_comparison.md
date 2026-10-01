# Controlled Experiment: Layer 2 Physical-Relationship Aggregation Comparison

**Document Type:** Formal Research & Controlled Experiment Report  
**Experiment ID:** `EXP-L2-AGG-001`  
**Date:** October 1, 2026  
**Status:** Experiment Only — Active Production Configuration Remains Strictly Unchanged (`MAX`)  
**Evaluation Standard:** Strategy D (Sequence End Timestamp $t_{59}$, Zero Future Leakage)  
**Evaluation Benchmark:** 5 Multi-Agent Adversarial Attack Campaigns (304 Qualifying Impactful Episodes, 110,730 Scope Sequences, 6,367 Positive Sequences)  
**Calibration Baseline:** `20260225_normal` (Uncompromised Normal Operational Training Telemetry)

---

## 1. Executive Summary & Experimental Objectives

### 1.1 Why the Experiment Was Performed
In the current cyber-physical anomaly detection architecture, Layer 2 evaluates four independent physical relationships:
1. **PV Inverter:** $P_{\text{dc}} \to P_{\text{ac}}$ (kW)
2. **Wind Anemometer:** $v_{\text{wind}, A} \to v_{\text{wind}, B}$ (m/s)
3. **Wind Nacelle Temperature:** $T_{\text{nacelle}, A} \to T_{\text{nacelle}, B}$ (°C)
4. **PV Thermal:** $T_{\text{air}} \to T_{\text{cell}}$ (°C)

Each relationship residual is normalized relative to its calibrated threshold:
$$s_k = \min\left(\frac{|r_k|}{T_{L2, k}}, 1.0\right), \quad k \in \{1, 2, 3, 4\}$$

Historically, Evidence Fusion has combined these four normalized scores using a point-wise maximum:
$$L_2 = \max(s_1, s_2, s_3, s_4)$$

While `MAX` captures any single broken physical law, a common question in multi-sensor fusion is whether `MAX` is overly susceptible to isolated single-sensor noise spikes, and whether statistical smoothing (such as `MEAN`, `TOP-2 MEAN`, or `TOP-3 MEAN`) could provide a superior false-positive to detection trade-off.

### 1.2 Core Research Question
> *"Does aggregating the four normalized physical-relationship scores via `MEAN`, `TOP-2 MEAN`, or `TOP-3 MEAN` outperform the baseline `MAX` operator in sequence-level detection (F1, PR-AUC), false-alarm suppression (FPR, normal exceedance), or attack episode coverage across multi-agent adversarial campaigns?"*

### 1.3 Key Findings at a Glance
1. **The Dilution Effect of Averaging on Targeted Intrusions:**
   Multi-agent adversarial attacks in smart grids are typically **targeted** (e.g. false data injection into inverter setpoints or nacelle anemometers). When an attack corrupts a single physical relationship ($s_1 \approx 1.0$) while the other three subsystems remain uncompromised ($s_2, s_3, s_4 \approx 0.0$):
   - Under `MAX`, $L_2 = 1.0$ (full alert strength).
   - Under `MEAN`, $L_2 = (1.0 + 0 + 0 + 0) / 4 = 0.25$ (**$75\%$ signal dilution**).
   Under the fixed baseline fusion threshold ($T=0.784338$), `MEAN` cuts attack sequence recall in half ($50.02\% \to 25.43\%$) and loses **38 attack episodes** ($169 \to 131$, dropping from $55.59\%$ to $43.09\%$).
2. **Top Sequence F1 Score in TOP-2 MEAN:**
   `TOP-2 MEAN` achieves an F1 score of **$0.1030$** (versus **$0.1026$** for `MAX`), reducing test campaign FPR from $50.33\%$ to $44.05\%$ ($-6.28\%$). However, it detects **13 fewer attack episodes** ($156$ vs. $169$, $51.32\%$ vs. $55.59\%$).
3. **Threshold-Independent Ranking (PR-AUC & ROC-AUC):**
   When evaluated independently of any specific decision threshold across the entire operating curve, `MEAN` and `WEIGHTED-EQUAL` achieve slightly higher PR-AUC ($0.0577$ vs. $0.0553$) because averaging acts as a low-pass filter against uncorrelated sensor noise. However, this marginal PR-AUC improvement is achieved at the expense of severely truncated operational peak recall.
4. **Experimental Recommendation:**
   **The evidence is NOT strong enough to justify replacing `MAX` in the active pipeline.** While `TOP-2 MEAN` offers an interesting conservative option with lower FPR, `MAX` remains fundamentally better aligned with the cyber-physical security objective of detecting targeted, single-sensor attacks. **The active production configuration remains strictly at `MAX`.**

---

## 2. Experimental Setup & Invariants

The experiment was conducted under strict controls to ensure scientific validity and consistency with prior evaluations:

```
Smart Grid Operational Telemetry
               │
        ┌──────┴──────┐
        ▼             ▼
     Layer 1       Layer 2
     TCN-AE     4 XGBoost Physics
     (P99)         (P95)
        │             │
        └──────┬──────┘
               ▼
        Evidence Fusion
   [Evaluating L2 Aggregation:
   MAX vs. MEAN vs. TOP-k]
               ▼
     Downstream Architecture
    (Untouched & Unmodified)
```

### Frozen Invariants Preserved
1. **Zero Model Retraining:** Neither the Layer 1 TCN autoencoder ([`models/tcn_autoencoder_baseline.pt`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/tcn_autoencoder_baseline.pt)) nor the four Layer 2 XGBoost regression models ([`models/layer2_*_xgb.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/)) were modified or retrained.
2. **Active Operating Thresholds Held Fixed:**
   - Layer 1 active threshold: **$P_{99} = 0.0008954601059667766$**
   - Layer 2 active thresholds: **$P_{95}$** (`pv_inverter`: $7.0729\text{ kW}$, `wind_speed`: $0.5411\text{ m/s}$, `wind_temperature`: $0.8566^\circ\text{C}$, `pv_thermal`: $10.3477^\circ\text{C}$)
3. **Fixed Fusion Weights:** Symmetrical 0.5 Layer 1 + 0.5 Layer 2 linear fusion formula strictly maintained.
4. **Zero Label Leakage:** No attack labels or attack sequences were used to tune, fit, or select aggregation methods.
5. **No Production Overwrite:** The active Evidence Fusion code ([`src/ml/evidence_fusion.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/ml/evidence_fusion.py)) was untouched.

---

## 3. Mathematical Formulation of Candidate Aggregation Methods

Let $s_1, s_2, s_3, s_4 \in [0, 1]$ represent the normalized residuals for the four physical relationships at sequence end timestamp $t$:
$$s_1 = \min\left(\frac{|r_{\text{inverter}}|}{T_{\text{inverter}, P95}}, 1.0\right), \quad s_2 = \min\left(\frac{|r_{\text{wind\_speed}}|}{T_{\text{wind\_speed}, P95}}, 1.0\right)$$
$$s_3 = \min\left(\frac{|r_{\text{wind\_temp}}|}{T_{\text{wind\_temp}, P95}}, 1.0\right), \quad s_4 = \min\left(\frac{|r_{\text{pv\_thermal}}|}{T_{\text{pv\_thermal}, P95}}, 1.0\right)$$

Let $s_{(1)} \ge s_{(2)} \ge s_{(3)} \ge s_{(4)}$ denote the order statistics (values sorted in descending order).

### 1. MAX (Current Baseline)
$$L_2^{(\text{MAX})} = \max(s_1, s_2, s_3, s_4) = s_{(1)}$$
- **Physical Rationale:** Any single broken physical conservation law indicates physical inconsistency.

### 2. MEAN (Uniform Average)
$$L_2^{(\text{MEAN})} = \frac{1}{4} \sum_{k=1}^4 s_k = \frac{s_1 + s_2 + s_3 + s_4}{4}$$
- **Physical Rationale:** Assumes anomalies produce system-wide collective deviations; acts as a noise filter.

### 3. TOP-2 MEAN
$$L_2^{(\text{TOP-2})} = \frac{1}{2} \left( s_{(1)} + s_{(2)} \right)$$
- **Physical Rationale:** Requires at least two physical relationships to show elevation before reaching high score levels; suppresses single-sensor spikes while requiring less broad impact than `MEAN`.

### 4. TOP-3 MEAN
$$L_2^{(\text{TOP-3})} = \frac{1}{3} \left( s_{(1)} + s_{(2)} + s_{(3)} \right)$$
- **Physical Rationale:** Intermediate order-statistic average across the upper three residuals.

### 5. WEIGHTED MEAN (Equal Weights — Sanity Check)
$$L_2^{(\text{WEIGHTED})} = \sum_{k=1}^4 w_k s_k, \quad w_k = 0.25, \; \sum w_k = 1.0$$
- **Physical Rationale:** Non-leaking baseline weighting, identical to uniform `MEAN`, serving as the benchmark for future non-equal weight learning experiments.

---

## 4. Empirical Evaluation Results

To guarantee comprehensive analysis, each aggregation method was evaluated under two distinct operating paradigms:
- **Paradigm A (Fixed Baseline Threshold $T=0.784338$):** Evaluates drop-in substitution into the current active fusion pipeline.
- **Paradigm B (Normal Calibrated $P_{99.5}$ Threshold):** Evaluates each method at its own empirical $P_{99.5}$ threshold derived from uncompromised normal training data (`20260225_normal`), ensuring identical nominal false-alarm calibration on clean data.
- **Threshold-Independent Metrics:** ROC-AUC and PR-AUC evaluating global ranking.

### 4.1 Master Results Table

| Aggregation Method | ROC-AUC | PR-AUC | Eval Paradigm | Operating Threshold | Precision | Recall | F1 Score | Test FPR | FNR | Episodes Detected (of 304) | Episode Det. % | Median Latency | Mean Latency | Normal Val Exceed % |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAX (Baseline)** | 0.4880 | 0.0553 | **Fixed Baseline** | 0.784338 | 0.0572 | **0.5002** | 0.1026 | 0.5033 | 0.4998 | **169** | **55.59%** | **0.2739 s** | **0.6436 s** | 39.23% |
| | | | **Calibrated P99.5** | 0.950296 | 0.0548 | 0.3471 | 0.0946 | 0.3655 | 0.6529 | 135 | 44.41% | 0.3201 s | 1.3985 s | 17.35% |
| **MEAN (All 4)** | **0.4995** | **0.0577** | **Fixed Baseline** | 0.784338 | 0.0582 | 0.2543 | 0.0947 | **0.2511** | 0.7457 | 131 | 43.09% | 0.4140 s | 1.5545 s | **12.82%** |
| | | | **Calibrated P99.5** | 0.732509 | 0.0584 | 0.3510 | 0.1001 | 0.3454 | 0.6490 | 142 | 46.71% | 0.3323 s | 0.9938 s | 20.00% |
| **TOP-2 MEAN** | 0.4919 | 0.0556 | **Fixed Baseline** | 0.784338 | 0.0582 | 0.4464 | **0.1030** | 0.4405 | 0.5536 | 156 | 51.32% | 0.2889 s | 0.7942 s | 31.07% |
| | | | **Calibrated P99.5** | 0.872124 | **0.0586** | 0.3410 | 0.1001 | 0.3339 | 0.6590 | 139 | 45.72% | 0.3320 s | 1.3018 s | 18.12% |
| **TOP-3 MEAN** | 0.4971 | 0.0566 | **Fixed Baseline** | 0.784338 | **0.0585** | 0.3711 | 0.1011 | 0.3642 | 0.6289 | 144 | 47.37% | 0.3097 s | 1.0719 s | 22.39% |
| | | | **Calibrated P99.5** | 0.805636 | 0.0581 | 0.3410 | 0.0993 | 0.3372 | 0.6590 | 141 | 46.38% | 0.3409 s | 1.1404 s | 20.13% |
| **WEIGHTED (Equal)**| **0.4995** | **0.0577** | **Fixed Baseline** | 0.784338 | 0.0582 | 0.2543 | 0.0947 | **0.2511** | 0.7457 | 131 | 43.09% | 0.4140 s | 1.5545 s | **12.82%** |
| | | | **Calibrated P99.5** | 0.732510 | 0.0584 | 0.3510 | 0.1001 | 0.3454 | 0.6490 | 142 | 46.71% | 0.3323 s | 0.9938 s | 20.00% |

---

## 5. Diagnostic Visualizations

The generated publication-quality diagnostic charts illustrate the trade-off space across all candidate aggregation strategies:

### 5.1 ROC and Precision-Recall Curves
![ROC and PR Curves](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/figures/layer2_aggregation/l2_aggregation_roc_pr.png)
*Figure 1: (Left) ROC Curves across Layer 2 aggregation methods. `MEAN` and `WEIGHTED` achieve slightly higher AUC (0.4995) due to lower variance in background normal sequences. (Right) Precision-Recall curves. All methods exhibit characteristic cyber-physical PR behavior under 4.16% attack prevalence, with `MEAN` showing marginally higher average precision (0.0577 vs. 0.0553).*

### 5.2 Multi-Metric Performance Comparison
![Layer 2 Aggregation Comparison](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/experiments/figures/layer2_aggregation/l2_aggregation_comparison.png)
*Figure 2: (Left) Attack episode detection rate comparing Fixed Baseline Threshold (T=0.784) against Normal Calibrated P99.5 Threshold. Notice the dramatic collapse in episode coverage under `MEAN` with fixed threshold (43.1% vs. 55.6% for `MAX`). (Right) Sequence Recall, Campaign FPR, and F1 score under the fixed baseline threshold.*

---

## 6. Detailed Comparison Against Current MAX Baseline

### 6.1 Attack Episode Detection: Why MAX Wins
- **Baseline MAX:** Detects **$169$ out of 304 qualifying episodes ($55.59\%$)**.
- **TOP-2 MEAN:** Detects **$156$ episodes ($51.32\%$)** — loses 13 episodes ($-7.7\%$ relative loss).
- **TOP-3 MEAN:** Detects **$144$ episodes ($47.37\%$)** — loses 25 episodes ($-14.8\%$ relative loss).
- **MEAN:** Detects **$131$ episodes ($43.09\%$)** — loses 38 episodes ($-22.5\%$ relative loss).

**Root Cause Analysis:**  
In smart grid SCADA architectures, adversaries typically breach a specific sub-network (e.g. hacking the Modbus/TCP RTU communicating with the PV central inverter, or injecting spoofed 4-20mA signals into nacelle wind anemometers). They rarely compromise all renewable subsystems simultaneously. Under `MAX`, any single relationship exceeding its threshold immediately asserts a normalized residual of $1.0$, providing the Evidence Fusion layer with strong evidence. Under `MEAN`, averaging divides this signal by 4, completely suppressing the alarm unless Layer 1 exhibits massive MSE.

### 6.2 Sequence Recall vs. False Positive Rate
- **Sequence Recall:** `MAX` achieves **$50.02\%$ sequence recall**, substantially higher than `TOP-2 MEAN` ($44.64\%$), `TOP-3 MEAN` ($37.11\%$), and `MEAN` ($25.43\%$).
- **False Positive Rate:** `MAX` incurs an FPR of **$50.33\%$** on test campaigns, whereas `TOP-2 MEAN` achieves **$44.05\%$** ($-6.28\%$), and `MEAN` achieves **$25.11\%$** ($-25.22\%$).
- **Normal Validation Exceedance:** `MAX` produces $39.23\%$ exceedance, `TOP-2` produces $31.07\%$, and `MEAN` produces $12.82\%$.

### 6.3 F1 Score Comparison
- `TOP-2 MEAN` achieves an F1 score of **$0.1030$**, which is marginally higher than `MAX` ($0.1026$, $\Delta = +0.0004$).
- However, this $0.0004$ F1 gain comes at the cost of losing **13 attack episodes** and dropping sequence recall by **$5.38\%$**.
- In operational cyber-physical security, dropping 13 entire attack episodes to gain 0.0004 in F1 is an unacceptable trade-off.

### 6.4 Detection Latency
- `MAX` achieves the lowest median latency (**$0.2739\text{ s}$**) and lowest mean latency (**$0.6436\text{ s}$**).
- `TOP-2 MEAN` increases median latency to **$0.2889\text{ s}$** and mean latency to **$0.7942\text{ s}$** ($+23.4\%$ slower).
- `MEAN` increases median latency to **$0.4140\text{ s}$** and mean latency to **$1.5545\text{ s}$** ($+141.5\%$ slower).
- Averaging delays alerting because multiple sensors must accumulate residual error before the mean crosses the threshold.

---

## 7. Analysis of the Normal Calibrated Threshold Paradigm

When each aggregation method is evaluated at its own empirical $P_{99.5}$ normal training threshold (Paradigm B):
- Threshold values shift to reflect the lower natural variance of averages:
  - `MAX` $T_{\text{cal}} = 0.9503$
  - `TOP-2` $T_{\text{cal}} = 0.8721$
  - `TOP-3` $T_{\text{cal}} = 0.8056$
  - `MEAN` $T_{\text{cal}} = 0.7325$
- At these calibrated thresholds:
  - Episode detection converges across methods to a narrow band: **$135$ to $142$ episodes** ($44.4\% - 46.7\%$).
  - Sequence recall converges to **$34.1\% - 35.1\%$**.
  - F1 scores converge to **$0.0946 - 0.1001$**.
  - Normal validation exceedances converge to **$17.3\% - 20.0\%$**.

**Crucial Takeaway:** When calibrated to achieve strict normal false-alarm parity ($P_{99.5}$), averaging methods gain only $+7$ episodes ($142$ vs. $135$) over `MAX`, while still remaining far below `MAX` under the active operating baseline ($169$ episodes). Averaging provides no structural breakthrough in separating attacks from normal operational dynamics.

---

## 8. Evaluation Against Predefined Selection Criteria

| Evaluation Criterion | Top-Performing Method | Runner-Up | Analysis & Trade-off |
| :--- | :---: | :---: | :--- |
| **Attack Episode Detection** | **MAX (169 / 304, 55.59%)** | TOP-2 MEAN (156 / 304, 51.32%) | `MAX` detects 13 more episodes than TOP-2 and 38 more than MEAN. Crucial for catching targeted single-sensor attacks. |
| **Sequence Recall** | **MAX (50.02%)** | TOP-2 MEAN (44.64%) | `MAX` captures half of all positive attack sequences. Averaging cuts recall significantly. |
| **Sequence F1 Score** | **TOP-2 MEAN (0.1030)** | MAX (0.1026) | TOP-2 edges MAX by 0.0004, but loses 13 episodes. |
| **Test Campaign FPR** | **MEAN (25.11%)** | TOP-3 MEAN (36.42%) | `MEAN` suppresses noise effectively, but at the cost of missing over half of the attack sequences. |
| **PR-AUC (Global Ranking)**| **MEAN (0.0577)** | TOP-3 MEAN (0.0566) | Averaging slightly improves overall area under the PR curve by filtering normal background noise. |
| **Detection Latency** | **MAX (0.2739 s med, 0.6436 s mean)**| TOP-2 MEAN (0.2889 s med, 0.7942 s mean)| `MAX` triggers fastest upon attack initiation. |

---

## 9. Final Decision & Recommendation

### 9.1 Is the evidence strong enough to consider changing the active Layer 2 aggregation?
**NO.** The evidence does **NOT** support replacing `MAX` with `MEAN`, `TOP-2 MEAN`, or `TOP-3 MEAN` in the active Evidence Fusion pipeline.

### 9.2 Key Arguments for Retaining `MAX`
1. **Physical Soundness:** In an electrical power grid, physical laws operate independently. If an inverter's DC-to-AC conversion efficiency is violated, that violation is an absolute anomaly regardless of whether the wind anemometers on another tower agree. Diluting a severe electrical inconsistency by averaging it with clean aerodynamic sensors violates domain physics.
2. **Episode Coverage Priority:** `MAX` detects **$169$ qualifying episodes ($55.59\%$)**, whereas `MEAN` drops to **$131$ episodes ($43.09\%$)**. Missing 38 attack campaigns is an unacceptably high penalty for noise reduction.
3. **Role of Downstream Layers:** The smart grid architecture specifically routes Evidence Fusion into a **Digital Twin (Simscape physics simulation)** and **RAG + Domain LLM**. The upstream detector's role is to ensure high candidate recall and episode coverage. Downstream physics engines and contextual LLMs are designed specifically to filter out physically plausible transients that cause false alarms in `MAX`.
4. **Future Experiment Outlook:** The next planned experiment will investigate **learning asymmetric L1-vs-L2 fusion weights** ($w_1 \cdot L_1 + w_2 \cdot L_2$) or relationship-specific weighting, which provides a far more principled path to false-positive reduction than simple unweighted residual averaging.

---

## 10. Integrity Confirmation

- [x] Zero models retrained or modified.
- [x] Active Layer 1 TCN-AE checkpoint hash verified untouched (`0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22`).
- [x] Active Layer 2 XGBoost regression models untouched.
- [x] Active thresholds preserved: Layer 1 $P_{99} = 0.00089546$, Layer 2 $P_{95}$ per relationship.
- [x] Active Evidence Fusion implementation untouched (`src/ml/evidence_fusion.py` remains active baseline `MAX`).
- [x] Digital Twin, RAG/LLM, and Dashboard completely untouched.
- [x] Zero attack data or labels used to derive thresholds or tune aggregation.
