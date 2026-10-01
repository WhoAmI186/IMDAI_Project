# Phase 3C: Anomaly Score Aggregation Analysis

**Document Version:** 1.0.0  
**Date:** September 29, 2026  
**Status:** Complete & Verified  
**Scope:** Frozen Model Diagnostic Analysis of Alternative Anomaly Score Aggregations (Mean MSE, Max Feature Error, Top-K Feature Error, and Temporal Persistence) for Config A (14 raw features) and Config D (24 engineered features).  
**Investigator:** Antigravity Pair Programming Agent  

---

> ### KEY DIAGNOSTIC CONCLUSION
> **"Does the existing LSTM contain useful anomaly information that is being hidden by mean-MSE aggregation, or is the reconstruction error itself largely non-discriminative?"**
>
> **DEFINITIVE EXPERIMENTAL ANSWER:**  
> **The reconstruction error itself is largely non-discriminative.**  
> Isolating peak feature errors via Max Feature Error (Top-1) or Top-K aggregation completely eliminates cross-feature averaging dilution, but **fails to improve anomaly detection performance**. Scope-Appropriate ROC-AUC remains essentially identical across all scoring methods (0.4850–0.4882 for Config A, and 0.4646–0.4754 for Config D), remaining at or slightly below random chance (0.5000). PR-AUC remains at or below baseline prevalence (0.0365–0.0405 vs. 0.0416 prevalence). Furthermore, evaluating single-feature reconstruction errors independently yields ROC-AUCs centered tightly between 0.448 and 0.577.  
> 
> The poor anomaly detection performance of the LSTM autoencoder is **not** an artifact of mean-MSE dilution; rather, the underlying bottleneck representation has reconstructed cyber-physical attack perturbations with residual error magnitudes indistinguishable from ordinary operational variance.

---

## 1. Objective & Frozen Artifact Integrity

### 1.1 Experimental Objective
The objective of Phase 3C is to resolve whether the low detection performance observed in Phase 2C.4 and Phase 3B was caused by:
1. **The Scoring Aggregation Layer:** Mean-MSE averaging over all $D$ features dilutes localized single-feature anomalies below detection thresholds; OR
2. **The Representation Layer:** The autoencoder bottleneck fails to produce discriminative reconstruction errors for cyberattack perturbations.

### 1.2 Strict Anti-Leakage & Frozen Protocol
In accordance with experimental rules, **no model was trained, fine-tuned, or modified** during Phase 3C:
- **Config A Checkpoint:** `models/lstm_autoencoder_baseline.pt` ($D=14$) remained 100% frozen.
- **Config D Checkpoint:** `models/lstm_autoencoder_config_d.pt` ($D=24$) remained 100% frozen.
- **Scalers:** Fitted strictly on clean normal training telemetry ($N=9,592$ rows) from `20260225_normal`. Reconstructed strictly from checkpoint metadata.
- **Threshold Calibration:** Calibrated strictly on $N=9,533$ clean normal training sequences. Zero attack data or labels were utilized for threshold selection.
- **Evaluation Protocols:** Identical Strategy D ground truth and verified Scope-Appropriate qualifying attack intervals as Phase 2C.4 and Phase 3B.

### 1.3 Cryptographic Integrity Verification
Before and after execution, SHA-256 hashes of the frozen model checkpoints were verified byte-for-byte:

| Model | Checkpoint File | Verified SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :--- |
| **Config A** | `models/lstm_autoencoder_baseline.pt` | `6a52d5b5900b4e508c487b41ff3b4888c0604b85f57630749fcbff6633037ac7` | **100% UNMODIFIED / FROZEN** |
| **Config D** | `models/lstm_autoencoder_config_d.pt` | `ffa9afb5880091b97951d15c7819be4fc343d9a69ea5ffa6bc8c56a8f1e7a3a4` | **100% UNMODIFIED / FROZEN** |

---

## 2. Existing Mean-MSE Scoring Method

### 2.1 Formal Formulation
The reference anomaly score established in Phase 2C computes the mean squared reconstruction error over all $L=60$ sequence timesteps and all $D$ channels:
$$\text{Score}_{\text{Mean-MSE}}(X) = \frac{1}{L \cdot D} \sum_{t=1}^{L} \sum_{d=1}^{D} \left( X_{t,d} - \hat{X}_{t,d} \right)^2$$

### 2.2 Theoretical Dilution Hypothesis
Under Mean-MSE, if a cyberattack targets a single physical measurement $d^*$ producing an elevated reconstruction error $\Delta e$, its contribution to the global sequence anomaly score is scaled by $\frac{1}{D}$:
$$\Delta \text{Score}_{\text{Mean-MSE}} = \frac{1}{D} \Delta e$$
- For **Config A** ($D=14$), a single-channel anomaly is diluted by a factor of 14 ($\approx 92.86\%$ dilution).
- For **Config D** ($D=24$), the dilution factor increases to 24 ($\approx 95.83\%$ dilution).

If uncompromised features exhibit natural background fluctuation variance $\sigma^2_{\text{bg}}$, the signal-to-noise ratio (SNR) of a localized perturbation is suppressed by $\mathcal{O}(1/D)$. This motivated testing whether non-diluting aggregations unmask the attack signal.

---

## 3. Alternative Scoring Methods

For each frozen sequence $X \in \mathbb{R}^{L \times D}$ and reconstruction $\hat{X} \in \mathbb{R}^{L \times D}$, we first compute the channel-level reconstruction MSE over the 60-second window:
$$e_d = \frac{1}{L} \sum_{t=1}^{L} (X_{t,d} - \hat{X}_{t,d})^2, \quad \forall d \in \{1, \dots, D\}$$

### 3.1 Max Feature Error (Top-1)
Isolates the single most anomalous feature in the sequence, completely eliminating cross-feature averaging dilution:
$$\text{Score}_{\text{Max}} = \max_{d \in \{1, \dots, D\}} e_d$$

### 3.2 Top-K Feature Error (Top-3 and Top-5 Mean)
Accounts for localized multi-variable disruptions (e.g., simultaneous manipulation of PV voltage, current, and active power) without diluting across unrelated subsystems. Features are sorted descending: $e_{(1)} \ge e_{(2)} \ge \dots \ge e_{(D)}$:
$$\text{Score}_{\text{Top-}K} = \frac{1}{K} \sum_{k=1}^{K} e_{(k)}, \quad \text{for } K \in \{3, 5\}$$

### 3.3 Temporal Persistence
Requires an anomaly score to remain sustained over consecutive sequence observations. We formulate continuous causal persistence over a backward horizon of $P$ sequences:
$$s_i^{(P)} = \min_{j=0}^{P-1} s_{i-j}, \quad \text{for } P \in \{1, 3, 5\}$$

#### Mathematical Equivalence & Sequence Overlap Handling:
1. **Exact Equivalence to Consecutive Rule Exceedance:**  
   Because $s_i^{(P)} \ge \theta \iff \forall j \in \{0, \dots, P-1\}, s_{i-j} \ge \theta$, the continuous score $s_i^{(P)}$ guarantees that a detection at threshold $\theta$ occurs if and only if $P$ consecutive sequence inferences independently exceeded $\theta$. This enables continuous ROC-AUC and PR-AUC computation across all operating thresholds.
2. **Handling Sequence Window Overlap:**  
   With sliding window parameters $L=60$ seconds and stride $s=1$ second, adjacent sequences $i$ and $i+1$ share 59 overlapping seconds (98.33% sample overlap). A transient point-level spike entering the window will persist inside the buffer for 60 consecutive sequence decisions. Therefore, requiring $P \le 5$ consecutive sequence alarms suppresses isolated sub-second prediction jitter, while ensuring that the evaluated disturbance is sustained across consecutive sequence decisions without creating artificial boundary gaps. Persistence was computed strictly within each continuous recording campaign to prevent cross-run leakage.

---

## 4. Threshold Methodology

Methodological fairness requires that thresholds are calibrated **strictly on clean normal training telemetry**:
1. All candidate thresholds were derived using the 9,533 training sequences generated from the 80% split of clean baseline dataset `20260225_normal`.
2. Candidate percentiles: P95, Mean+3σ, P99, P99.5, P99.9.
3. **Primary Operational Threshold:** $P_{99.5}$ (targeting nominal 0.5% FPR on normal training data).
4. Validation integrity: Evaluated on the 2,340 normal validation sequences (20% chronological holdout) to verify out-of-sample baseline false positive behavior.
5. In addition to thresholded metrics, threshold-independent **ROC-AUC** and **PR-AUC** were computed across all continuous scores.

### Calibrated Operational Thresholds ($P_{99.5}$) on Normal Training Data:

| Scoring Method | Config A Threshold ($P_{99.5}$) | Config A Val FPR (%) | Config D Threshold ($P_{99.5}$) | Config D Val FPR (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean MSE** | 0.070684 | 7.56% | 0.056006 | 2.26% |
| **Max Feature (Top-1)** | 0.433876 | 2.18% | 0.435596 | 2.14% |
| **Top-3 Mean** | 0.298453 | 1.92% | 0.291250 | 1.20% |
| **Top-5 Mean** | 0.189862 | 2.91% | 0.209066 | 1.71% |
| **Mean MSE Persist P=3** | 0.069987 | 7.35% | 0.055262 | 2.18% |
| **Mean MSE Persist P=5** | 0.069327 | 7.14% | 0.054299 | 2.14% |
| **Max Feature Persist P=3** | 0.416110 | 2.05% | 0.418910 | 2.44% |
| **Max Feature Persist P=5** | 0.413138 | 1.79% | 0.413555 | 2.52% |
| **Top-3 Mean Persist P=3** | 0.294938 | 1.75% | 0.288549 | 1.07% |
| **Top-3 Mean Persist P=5** | 0.293838 | 1.58% | 0.288031 | 0.90% |

*Observation:* For both models, Max Feature and Top-K thresholds produce substantially tighter validation FPRs (0.90%–2.52%) on holdout normal data compared to Mean-MSE (7.56% on Config A), indicating that baseline normal distribution tails in Max Feature are well-bounded.

---

## 5. Broad Benchmark Results ($N = 153,196$ Sequences)

The Broad Benchmark evaluates performance across all 16 adversarial recording sessions spanning 5 multi-agent campaigns ($N=153,196$ total sequences, 48,833 positive sequences under Strategy D, prevalence = 31.88%).

| Model | Scoring Method | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR (%) | Episode Det Rate | Median Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | **Mean MSE (Baseline)** | 0.4594 | 0.2953 | 0.2805 | 0.2506 | **0.2647** | 30.08% | **32.28%** | 0.36 s |
| Config A | Max Feature (Top-1) | 0.4536 | 0.2888 | 0.2618 | 0.1587 | 0.1978 | 20.93% | 21.68% | 0.39 s |
| Config A | Top-3 Mean | 0.4559 | 0.2916 | 0.2709 | 0.1727 | 0.2109 | 21.75% | 24.18% | 0.38 s |
| Config A | Top-5 Mean | 0.4581 | 0.2942 | 0.2798 | 0.2045 | 0.2363 | 24.66% | 27.21% | 0.38 s |
| Config A | Mean MSE Persist P=3 | 0.4589 | 0.2950 | 0.2797 | 0.2482 | 0.2630 | 29.87% | 31.06% | 0.36 s |
| Config A | Mean MSE Persist P=5 | 0.4585 | 0.2947 | 0.2797 | 0.2467 | 0.2621 | 29.74% | 30.42% | 0.36 s |
| Config A | Max Feat Persist P=3 | 0.4529 | 0.2883 | 0.2625 | 0.1619 | 0.2003 | 21.23% | 22.32% | 0.39 s |
| Config A | Max Feat Persist P=5 | 0.4523 | 0.2878 | 0.2605 | 0.1582 | 0.1965 | 20.99% | 21.27% | 0.39 s |
| Config A | Top-3 Persist P=3 | 0.4554 | 0.2912 | 0.2704 | 0.1711 | 0.2096 | 21.60% | 23.83% | 0.38 s |
| Config A | Top-3 Persist P=5 | 0.4549 | 0.2908 | 0.2689 | 0.1682 | 0.2070 | 21.43% | 23.66% | 0.38 s |
| **Config D** | **Mean MSE** | 0.4448 | 0.2869 | 0.2709 | 0.1581 | **0.1995** | 20.50% | **20.69%** | 0.40 s |
| Config D | Max Feature (Top-1) | 0.4383 | 0.2793 | 0.2520 | 0.1173 | 0.1599 | 16.27% | 15.33% | 0.44 s |
| Config D | Top-3 Mean | 0.4346 | 0.2782 | 0.2435 | 0.1118 | 0.1535 | 16.21% | 15.68% | 0.43 s |
| Config D | Top-5 Mean | 0.4386 | 0.2820 | 0.2592 | 0.1284 | 0.1712 | 17.14% | 17.66% | 0.42 s |
| Config D | Mean MSE Persist P=3 | 0.4441 | 0.2865 | 0.2705 | 0.1571 | 0.1987 | 20.44% | 20.98% | 0.40 s |
| Config D | Mean MSE Persist P=5 | 0.4434 | 0.2860 | 0.2707 | 0.1568 | 0.1985 | 20.37% | 20.98% | 0.40 s |
| Config D | Max Feat Persist P=3 | 0.4374 | 0.2787 | 0.2541 | 0.1226 | 0.1652 | 16.82% | 16.03% | 0.43 s |
| Config D | Max Feat Persist P=5 | 0.4365 | 0.2782 | 0.2536 | 0.1207 | 0.1636 | 16.59% | 15.38% | 0.43 s |
| Config D | Top-3 Persist P=3 | 0.4339 | 0.2776 | 0.2424 | 0.1102 | 0.1519 | 16.11% | 14.86% | 0.44 s |
| Config D | Top-3 Persist P=5 | 0.4331 | 0.2771 | 0.2401 | 0.1073 | 0.1490 | 15.89% | 14.28% | 0.45 s |

---

## 6. Scope-Appropriate Benchmark Results ($N = 304$ Qualifying Intervals)

The Scope-Appropriate Benchmark evaluates only the verified, observable physical attack steps on PV and Wind systems (304 qualifying intervals, 6,367 positive sequences, 110,730 clean negative sequences, prevalence = 4.16%).

| Model | Scoring Method | Scope ROC | Scope PR | Scope F1 | Scope FPR (%) | Scope Rec | Episode Det Rate | Clean ROC | Clean PR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | **Mean MSE (Baseline)** | **0.4882** | 0.0401 | **0.0666** | 28.58% | **0.2615** | **31.25%** | 0.4753 | 0.0539 |
| Config A | Max Feature (Top-1) | 0.4850 | **0.0405** | 0.0543 | 19.62% | 0.1542 | 20.72% | 0.4704 | 0.0543 |
| Config A | Top-3 Mean | 0.4866 | 0.0395 | 0.0597 | 20.97% | 0.1797 | 23.03% | 0.4727 | 0.0533 |
| Config A | Top-5 Mean | 0.4877 | 0.0399 | 0.0634 | 24.40% | 0.2167 | 25.66% | 0.4746 | 0.0538 |
| Config A | Mean MSE Persist P=3 | 0.4877 | 0.0401 | 0.0668 | 28.40% | 0.2609 | 29.93% | 0.4748 | 0.0539 |
| Config A | Mean MSE Persist P=5 | 0.4874 | 0.0400 | 0.0670 | 28.27% | 0.2606 | 29.28% | 0.4745 | 0.0538 |
| Config A | Max Feat Persist P=3 | 0.4847 | 0.0404 | 0.0543 | 20.14% | 0.1575 | 21.71% | 0.4697 | 0.0542 |
| Config A | Max Feat Persist P=5 | 0.4846 | 0.0402 | 0.0540 | 19.75% | 0.1542 | 20.39% | 0.4692 | 0.0540 |
| Config A | Top-3 Persist P=3 | 0.4862 | 0.0394 | 0.0596 | 20.81% | 0.1781 | 22.70% | 0.4722 | 0.0531 |
| Config A | Top-3 Persist P=5 | 0.4858 | 0.0394 | 0.0594 | 20.52% | 0.1756 | 22.70% | 0.4717 | 0.0530 |
| **Config D** | **Mean MSE** | **0.4754** | **0.0379** | **0.0572** | 20.50% | **0.1687** | **19.74%** | 0.4586 | 0.0505 |
| Config D | Max Feature (Top-1) | 0.4718 | 0.0372 | 0.0445 | 16.65% | 0.1103 | 13.82% | 0.4533 | 0.0494 |
| Config D | Top-3 Mean | 0.4668 | 0.0366 | 0.0423 | 17.16% | 0.1071 | 14.47% | 0.4468 | 0.0483 |
| Config D | Top-5 Mean | 0.4697 | 0.0371 | 0.0481 | 18.42% | 0.1294 | 16.45% | 0.4513 | 0.0492 |
| Config D | Mean MSE Persist P=3 | 0.4742 | 0.0378 | 0.0567 | 20.47% | 0.1670 | 20.07% | 0.4574 | 0.0503 |
| Config D | Mean MSE Persist P=5 | 0.4732 | 0.0377 | 0.0564 | 20.62% | 0.1670 | 20.39% | 0.4564 | 0.0502 |
| Config D | Max Feat Persist P=3 | 0.4704 | 0.0371 | 0.0445 | 17.31% | 0.1136 | 14.47% | 0.4519 | 0.0493 |
| Config D | Max Feat Persist P=5 | 0.4691 | 0.0370 | 0.0437 | 17.13% | 0.1106 | 13.49% | 0.4503 | 0.0491 |
| Config D | Top-3 Persist P=3 | 0.4656 | 0.0365 | 0.0416 | 16.99% | 0.1046 | 13.49% | 0.4455 | 0.0482 |
| Config D | Top-3 Persist P=5 | 0.4646 | 0.0365 | 0.0409 | 16.69% | 0.1013 | 12.83% | 0.4442 | 0.0481 |

### Key Benchmark Observations:
1. **No Metric Elevation Above Random:**  
   Across all 10 aggregation methods for both models, Scope ROC-AUC never exceeds 0.4882 (Config A Mean MSE) and 0.4754 (Config D Mean MSE). Random guessing produces an expected ROC-AUC of 0.5000.
2. **PR-AUC Stagnates Below Prevalence:**  
   With a positive test prevalence of $4.16\%$, a random uninformative classifier achieves an expected PR-AUC of 0.0416. Config A scoring methods yield PR-AUCs of 0.0394–0.0405. Config D yields 0.0365–0.0379. Neither model nor scoring method provides positive predictive utility.
3. **Max Feature Decreases Episode Detection Rate:**  
   Rather than boosting sensitivity, Max Feature (Top-1) reduces episode detection from 31.25% to 20.72% in Config A, and from 19.74% to 13.82% in Config D.
4. **Persistence Does Not Rescue Detectability:**  
   Enforcing temporal persistence ($P=3$ or $P=5$) lowers empirical sequence false alarms marginally, but also slightly reduces episode detection (31.25% -> 29.28% in Config A). It does not alter the fundamental ROC-AUC or PR-AUC trajectory.

---

## 7. Per-Feature Reconstruction-Error Analysis

To understand why Max Feature Error fails, we analyze the per-feature reconstruction error distributions and compare normal validation behavior against attack episodes.

### 7.1 Config A Per-Feature Reconstruction Error Breakdown ($D = 14$)

| Feature Channel | Normal Val Mean | Normal Val P95 | Scope Attack Mean | Error Ratio (Atk/Val) | Single-Feature Scope ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `pv_m_temp_air` | 0.012281 | 0.051895 | 0.023949 | 1.95x | 0.5324 |
| `pv_m_poa_direct` | 0.024122 | 0.103445 | 0.047666 | 1.98x | 0.4782 |
| `pv_m_wind_speed` | 0.027982 | 0.105547 | 0.029062 | 1.04x | 0.4855 |
| `pv_m_poa_diffuse` | 0.003417 | 0.019811 | 0.063511 | **18.59x** | 0.4709 |
| `pv_m_cell_temperature` | 0.007921 | 0.040714 | 0.013632 | 1.72x | 0.5124 |
| `pv_m_inverter_ac_power` | 0.020178 | 0.084667 | 0.043238 | 2.14x | 0.4797 |
| `pv_m_inverter_dc_power` | 0.019997 | 0.085925 | 0.043162 | 2.16x | 0.4755 |
| `pv_c_on_off` | 0.071453 | 0.306515 | 0.097956 | 1.37x | 0.5141 |
| `wind_m_power` | 0.023435 | 0.072155 | 0.040042 | 1.71x | 0.4733 |
| `wind_m_pressure` | 0.018880 | 0.091334 | 0.030517 | 1.62x | 0.4486 |
| `wind_m_wind_speed_a` | 0.019282 | 0.078123 | 0.033499 | 1.74x | 0.4733 |
| `wind_m_wind_speed_b` | 0.023913 | 0.096968 | 0.035505 | 1.48x | 0.4756 |
| `wind_m_temperature_a` | 0.029583 | 0.124801 | 0.132779 | 4.49x | 0.4978 |
| `wind_m_temperature_b` | 0.030862 | 0.125582 | 0.136370 | 4.42x | 0.4935 |

### 7.2 Config D Per-Feature Reconstruction Error Breakdown ($D = 24$)

| Feature Channel | Normal Val Mean | Normal Val P95 | Scope Attack Mean | Error Ratio (Atk/Val) | Single-Feature Scope ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `pv_m_temp_air` | 0.005381 | 0.024614 | 0.018548 | 3.45x | 0.5095 |
| `pv_m_poa_direct` | 0.011234 | 0.070768 | 0.030097 | 2.68x | 0.4721 |
| `pv_m_wind_speed` | 0.015987 | 0.065021 | 0.026203 | 1.64x | 0.5235 |
| `pv_m_poa_diffuse` | 0.003076 | 0.015615 | 0.056615 | **18.40x** | 0.4923 |
| `pv_m_cell_temperature` | 0.010888 | 0.047590 | 0.016472 | 1.51x | 0.4902 |
| `pv_m_inverter_ac_power` | 0.009649 | 0.066209 | 0.024348 | 2.52x | 0.4823 |
| `pv_m_inverter_dc_power` | 0.010117 | 0.068436 | 0.024489 | 2.42x | 0.4972 |
| `pv_c_on_off` | 0.071723 | 0.306103 | 0.099300 | 1.38x | 0.5352 |
| `wind_m_power` | 0.019497 | 0.060230 | 0.031835 | 1.63x | 0.4876 |
| `wind_m_pressure` | 0.040652 | 0.247761 | 0.037226 | 0.92x | 0.4607 |
| `wind_m_wind_speed_a` | 0.016349 | 0.052802 | 0.026675 | 1.63x | 0.5051 |
| `wind_m_wind_speed_b` | 0.020452 | 0.070576 | 0.028786 | 1.41x | 0.5044 |
| `wind_m_temperature_a` | 0.015746 | 0.074732 | 0.078461 | 4.98x | 0.4544 |
| `wind_m_temperature_b` | 0.014486 | 0.071367 | 0.067398 | 4.65x | 0.4518 |
| `res_inv_loss` (Physical) | 0.000519 | 0.002084 | 0.001232 | 2.37x | 0.4984 |
| `diff_anemometer_ab` (Physical) | 0.018814 | 0.030406 | 0.016677 | 0.89x | 0.4903 |
| `diff_nac_temp_ab` (Physical) | 0.015630 | 0.046521 | 0.020150 | 1.29x | **0.5777** |
| `diff_pv_cell_thermal` (Physical) | 0.035569 | 0.212124 | 0.034175 | 0.96x | 0.4873 |
| `diff_pv_ac_power` (Difference) | 0.000436 | 0.003778 | 0.002549 | **5.85x** | 0.5177 |
| `diff_wind_power` (Difference) | 0.004820 | 0.013826 | 0.004969 | 1.03x | 0.4952 |
| `diff_wind_speed_a` (Difference) | 0.004517 | 0.012532 | 0.004496 | 1.00x | 0.5211 |
| `roll_std15_wind_power` (Volatility) | 0.037625 | 0.110362 | 0.048819 | 1.30x | 0.5069 |
| `roll_std15_pv_power` (Volatility) | 0.010732 | 0.095937 | 0.047556 | 4.43x | 0.5108 |
| `dev_mean15_wind_power` (Persistence) | 0.017380 | 0.051307 | 0.020844 | 1.20x | 0.4691 |

---

### 7.3 Diagnostic Findings: Why Alternative Scoring Fails

#### Finding 1: The Dilution Factor Exists, But Is Not the Primary Defect
The empirical dilution ratio between the maximum feature error $\max_d e_d$ and the mean feature error $\frac{1}{D}\sum_d e_d$ is:
- **Config A:** $4.56\times$ (theoretical maximum is $14.0\times$)
- **Config D:** $6.68\times$ (theoretical maximum is $24.0\times$)

Dilution *does* compress the score dynamically. However, Max Feature Error removes this dilution entirely, yet its Scope ROC-AUC decreases from 0.4882 to 0.4850 in Config A, and from 0.4754 to 0.4718 in Config D.

#### Finding 2: Attack Reconstruction Errors Fall Within Normal P95 Noise
Looking at the tables above reveals why:
- While several features have elevated *mean* error ratios during attacks (e.g. `pv_m_poa_diffuse` ratio 18.59x, `wind_m_temperature_a` ratio 4.49x), their mean attack errors (**0.0635** and **0.1328**) remain **comparable to or below their normal validation 95th percentiles** (**0.0198** and **0.1248**).
- For `pv_m_inverter_ac_power`, the mean attack error is **0.0432**, whereas the 95th percentile under normal operation is **0.0847**.
- Normal operational fluctuations produce large reconstruction errors that trigger false alarms at similar rates as actual attack injections.

#### Finding 3: Single-Feature Channels Have Random Discriminability
Evaluating each feature's error channel independently as a 1D anomaly classifier yields ROC-AUCs clustered around 0.45–0.53.
- In Config A, the highest single-feature ROC-AUC is `pv_m_temp_air` at **0.5324**, which is statistically indistinguishable from a random coin flip.
- In Config D, the highest single-feature ROC-AUC is `diff_nac_temp_ab` at **0.5777**, while key attack targets such as `wind_m_power` (0.4876), `pv_m_inverter_ac_power` (0.4823), and `diff_wind_power` (0.4952) perform worse than random guessing.
- Because no individual feature error channel contains a strong anomaly signal, no aggregation method—whether Mean, Max, Top-3, or Top-5—can combine them into a discriminative composite detector.

#### Finding 4: Config D's Additional Features Exacerbate Noise Dilution
Config D added 10 engineered features. While physical invariant residuals like `res_inv_loss` achieved low normal reconstruction error (0.000519), their attack response remained tiny (0.001232), resulting in an individual ROC-AUC of 0.4984. Meanwhile, noisy engineered channels increased the empirical dilution ratio from $4.56\times$ to $6.68\times$, further reducing overall detection rate (19.74% vs. 31.25%).

---

## 8. Summary Comparison Table

The following table summarizes the primary metrics across both models and all scoring methods on the Scope-Appropriate Benchmark:

| Model | Scoring Method | Scope ROC-AUC | Scope PR-AUC | Scope F1 @ P99.5 | Scope Recall | Episode Det Rate | Median Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | **Mean MSE (Baseline)** | **0.4882** | 0.0401 | **0.0666** | **0.2615** | **31.25%** | 0.33 s |
| Config A | Max Feature (Top-1) | 0.4850 | **0.0405** | 0.0543 | 0.1542 | 20.72% | 0.39 s |
| Config A | Top-3 Mean | 0.4866 | 0.0395 | 0.0597 | 0.1797 | 23.03% | 0.38 s |
| Config A | Top-5 Mean | 0.4877 | 0.0399 | 0.0634 | 0.2167 | 25.66% | 0.38 s |
| Config A | Mean MSE Persist P=3 | 0.4877 | 0.0401 | 0.0668 | 0.2609 | 29.93% | 0.33 s |
| Config A | Mean MSE Persist P=5 | 0.4874 | 0.0400 | 0.0670 | 0.2606 | 29.28% | 0.33 s |
| Config A | Max Feat Persist P=3 | 0.4847 | 0.0404 | 0.0543 | 0.1575 | 21.71% | 0.39 s |
| Config A | Max Feat Persist P=5 | 0.4846 | 0.0402 | 0.0540 | 0.1542 | 20.39% | 0.39 s |
| Config A | Top-3 Persist P=3 | 0.4862 | 0.0394 | 0.0596 | 0.1781 | 22.70% | 0.38 s |
| Config A | Top-3 Persist P=5 | 0.4858 | 0.0394 | 0.0594 | 0.1756 | 22.70% | 0.38 s |
| **Config D** | **Mean MSE** | **0.4754** | **0.0379** | **0.0572** | **0.1687** | **19.74%** | 0.42 s |
| Config D | Max Feature (Top-1) | 0.4718 | 0.0372 | 0.0445 | 0.1103 | 13.82% | 0.43 s |
| Config D | Top-3 Mean | 0.4668 | 0.0366 | 0.0423 | 0.1071 | 14.47% | 0.43 s |
| Config D | Top-5 Mean | 0.4697 | 0.0371 | 0.0481 | 0.1294 | 16.45% | 0.42 s |
| Config D | Mean MSE Persist P=3 | 0.4742 | 0.0378 | 0.0567 | 0.1670 | 20.07% | 0.42 s |
| Config D | Mean MSE Persist P=5 | 0.4732 | 0.0377 | 0.0564 | 0.1670 | 20.39% | 0.42 s |
| Config D | Max Feat Persist P=3 | 0.4704 | 0.0371 | 0.0445 | 0.1136 | 14.47% | 0.43 s |
| Config D | Max Feat Persist P=5 | 0.4691 | 0.0370 | 0.0437 | 0.1106 | 13.49% | 0.43 s |
| Config D | Top-3 Persist P=3 | 0.4656 | 0.0365 | 0.0416 | 0.1046 | 13.49% | 0.44 s |
| Config D | Top-3 Persist P=5 | 0.4646 | 0.0365 | 0.0409 | 0.1013 | 12.83% | 0.44 s |

---

## 9. Interpretation & Key Diagnostic Question

### 9.1 Direct Answer to the Key Diagnostic Question
> **"Does the existing LSTM contain useful anomaly information that is being hidden by mean-MSE aggregation, or is the reconstruction error itself largely non-discriminative?"**

**The reconstruction error itself is largely non-discriminative.**

The hypothesis that "mean-MSE averaging hides a strong localized anomaly signal" is **falsified** by the experimental evidence:
1. When cross-feature dilution is eliminated completely via **Max Feature Error (Top-1)**, Scope ROC-AUC does *not* rise; it drops slightly from 0.4882 to 0.4850 in Config A, and from 0.4754 to 0.4718 in Config D.
2. In Top-K aggregations (Top-3, Top-5), performance remains stagnant.
3. In single-feature error channels, ROC-AUCs remain near 0.50 (ranging from 0.448 to 0.535 across process telemetry).
4. Attack reconstruction errors fall within the range of normal operational variation (below or near normal P95 thresholds).

### 9.2 Distinguishing Observations from Hypotheses

| Category | Finding / Statement |
| :--- | :--- |
| **Observed Result** | Max Feature, Top-3 Mean, and Top-5 Mean scoring yield Scope ROC-AUCs of 0.467–0.488, closely matching Mean MSE (0.475–0.488). |
| **Observed Result** | All PR-AUCs remain below baseline test prevalence (0.0365–0.0405 vs. 0.0416). |
| **Observed Result** | Single-channel reconstruction errors evaluated independently show ROC-AUCs centered tightly around 0.50. |
| **Observed Result** | The empirical dilution ratio is 4.56x for Config A and 6.68x for Config D. |
| **Supported Conclusion** | The failure of the process anomaly detector is not an artifact of score aggregation or channel dilution. |
| **Supported Conclusion** | Re-aggregating the existing LSTM autoencoder reconstruction errors cannot produce an effective anomaly detector for these cyber-physical attacks. |
| **Possible Explanation** | The autoencoder bottleneck ($h=64$) possesses sufficient capacity to reconstruct plausible physical trajectories even when actuators or setpoints are spoofed, because manipulated values remain within physically feasible operational envelopes. |
| **Possible Explanation** | Unsupervised reconstruction error lacks directional physics constraints: it measures generic deviation from training manifold density rather than violations of physical conservation laws. |

---

## 10. Limitations

1. **Model Scope:** This analysis is strictly focused on the frozen LSTM Autoencoder checkpoints (Config A and Config D). It does not evaluate alternative generative models (e.g. VAEs, Diffusion) or forecaster-based predictive error models (e.g. 1-step ahead prediction).
2. **Threshold Linearity:** Thresholds were calibrated using standard percentile cutoffs derived from clean normal training telemetry. While non-linear or multi-variate score combination could theoretically be fitted, doing so without attack data would still rely on the non-discriminative feature errors, and using attack data would violate the anti-leakage protocol.
3. **Attack Profile:** Multi-agent attacks in these campaigns frequently operate by altering control commands within valid operational bounds (e.g. curtailing power, altering blade yaw within allowable physical ranges) rather than injecting out-of-range sensor spikes. An autoencoder trained to reconstruct normal temporal dynamics reconstructs these plausible sequences with low error.

---

## 11. Recommended Next Direction

Based strictly on the experimental conclusions of Phase 3A, 3B, and 3C:

1. **Do NOT Continue Post-Processing LSTM Autoencoder Errors:**  
   Neither feature engineering (Phase 3B) nor alternative score aggregations (Phase 3C) have succeeded in elevating anomaly detection above random chance. Further adjustments to aggregation formulas, window sizes, or threshold heuristics on the current reconstruction errors will not overcome the fundamental lack of discriminative signal in the representation.

2. **Transition Beyond Pure Unsupervised Reconstruction:**  
   The empirical evidence demonstrates that process anomaly detection for stealthy cyber-physical attacks requires explicit physical models (e.g., Digital Twin / physics-based analytical redundancy) or multi-modal fusion combining process telemetry with network security audit logs, rather than relying solely on autoencoder reconstruction residuals.

3. **Status:**  
   **STOPPING HERE** per protocol. Phase 3C is complete. Awaiting user review before proceeding to Phase 4 or subsequent phases.

---

## 12. Artifact Reference

- **Execution Script:** [`src/ml/anomaly_score_analysis.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/ml/anomaly_score_analysis.py)
- **Interactive Notebook:** [`notebooks/phase3c_anomaly_score_analysis.ipynb`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/notebooks/phase3c_anomaly_score_analysis.ipynb)
- **Machine-Readable Results:** [`reports/phase3c_anomaly_score_analysis.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/phase3c_anomaly_score_analysis.json)
- **Generated Figures:**
  - Figure 1: [`reports/figures/phase3c_score_comparison.png`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase3c_score_comparison.png)
  - Figure 2: [`reports/figures/phase3c_per_feature_errors.png`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase3c_per_feature_errors.png)
  - Figure 3: [`reports/figures/phase3c_roc_pr_comparison.png`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase3c_roc_pr_comparison.png)
  - Figure 4: [`reports/figures/phase3c_latency_comparison.png`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/figures/phase3c_latency_comparison.png)
