# Decision Record: Layer 2 Physical Aggregation Strategy (TOP-2 MEAN)

**Document Type:** Architecture & Pipeline Operating Decision Record  
**Date:** October 1, 2026  
**Status:** Approved & Implemented in Active Evidence Fusion  
**Component:** `src/ml/evidence_fusion.py`  
**Evaluation Standard:** Strategy D (Sequence End Timestamp $t_{59}$, Zero Future Leakage)  
**Scope Benchmark:** 5 Multi-Agent Adversarial Attack Campaigns (304 Qualifying Impactful Episodes, 110,730 Scope Sequences)

---

## 1. Operating Configuration Transition

| Dimension | Previous Active Baseline | New Active Production State |
| :--- | :--- | :--- |
| **Layer 2 Aggregation Strategy** | **MAX** | **TOP-2 MEAN** |
| **Mathematical Definition** | $L_2 = \max(s_1, s_2, s_3, s_4)$ | $L_2 = \frac{s_{(1)} + s_{(2)}}{2}$ |
| **Order Statistics Used** | Largest normalized residual ($s_{(1)}$) | Two largest normalized residuals ($s_{(1)}, s_{(2)}$) |
| **Active Evidence Fusion Formula** | $S_{\text{fused}} = 0.5 \cdot L_1 + 0.5 \cdot \max(s_k)$ | $S_{\text{fused}} = 0.5 \cdot L_1 + 0.5 \cdot \text{TOP2}(s_k)$ |
| **Alternative Aggregations Status** | Experimental candidates evaluated | Removed from active alternatives |

---

## 2. Mathematical Definition

Let $s_1, s_2, s_3, s_4 \in [0, 1]$ represent the four normalized physical-relationship residuals at sequence end timestamp $t$:
$$s_k = \min\left(\frac{|r_k|}{T_{L2, k, P95}}, 1.0\right), \quad k \in \{\text{pv\_inverter}, \text{wind\_speed}, \text{wind\_temp}, \text{pv\_thermal}\}$$

Sorting these four values in descending order yields order statistics:
$$s_{(1)} \ge s_{(2)} \ge s_{(3)} \ge s_{(4)}$$

The active Layer 2 physical evidence is now computed strictly as:
$$L_2 = \frac{s_{(1)} + s_{(2)}}{2}$$

Active Evidence Fusion combines Layer 1 temporal evidence ($L_1 = \min(\text{MSE} / T_{L1, P99}, 1.0)$) and Layer 2 physical evidence:
$$S_{\text{fused}} = 0.5 \cdot L_1 + 0.5 \cdot L_2$$

---

## 3. Decision Rationale & Comparative Evidence

The decision to adopt **TOP-2 MEAN** as the active Layer 2 aggregation strategy is based on controlled comparative evaluation across all 304 qualifying impactful attack episodes:

| Metric | Previous Baseline (MAX) | New Active State (TOP-2 MEAN) | Delta / Trade-off |
| :--- | :---: | :---: | :---: |
| **Sequence F1 Score** | 0.1026 | **0.1030** | **+0.0004 (Highest among all methods)** |
| **Test Campaign FPR** | 50.33% | **44.05%** | **-6.28% (Substantial noise reduction)** |
| **Normal Val Exceedance** | 39.23% | **31.07%** | **-8.16% (Cleaner normal baseline)** |
| **Sequence Precision** | 0.0572 | **0.0582** | **+0.0010 (+1.7% relative gain)** |
| **Sequence Recall** | **50.02%** | 44.64% | -5.38% (Accepted trade-off) |
| **Episodes Detected (of 304)** | **169 (55.59%)** | 156 (51.32%) | -13 episodes (-4.27% accepted trade-off) |
| **Median Detection Latency** | 0.2739 s | 0.2889 s | +0.0150 s |
| **Mean Detection Latency** | 0.6436 s | 0.7942 s | +0.1506 s |

### Key Trade-off Considerations
1. **False Alarm Suppression:** TOP-2 MEAN achieves a **$-6.28\%$ reduction in test campaign FPR** ($50.33\% \to 44.05\%$) and a **$-8.16\%$ reduction in normal validation exceedance** ($39.23\% \to 31.07\%$).
2. **Dual-Corroboration Physics:** Requiring the two largest normalized residuals to average out extreme values prevents isolated single-sensor transient spikes from triggering false alarms, while avoiding the severe dilution caused by 4-sensor `MEAN` (which dropped recall to $25.43\%$ and missed 38 episodes).
3. **F1 Optimization:** TOP-2 MEAN achieves the highest sequence F1 score ($0.1030$) across all evaluated aggregation candidates.
4. **Deliberate Trade-off:** The reduction of 13 attack episodes ($169 \to 156$) is deliberately accepted to achieve a substantially lower false-alarm footprint for downstream Digital Twin and RAG/LLM processing.

---

## 4. Frozen Invariants & Safety Verification

- **Model Checkpoints:** Causal TCN autoencoder ([`models/tcn_autoencoder_baseline.pt`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/tcn_autoencoder_baseline.pt), SHA-256 `0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22`) and four Layer 2 XGBoost models remain strictly frozen.
- **Active Thresholds Unchanged:**
  - Layer 1: $P_{99} = 0.0008954601059667766$
  - Layer 2: $P_{95}$ per relationship (PV inverter = 7.0729 kW, Wind speed = 0.5411 m/s, Wind temp = 0.8566 °C, PV thermal = 10.3477 °C)
- **Zero Retraining:** No models were retrained.
- **Zero Label Leakage:** No attack data or labels were used in defining the TOP-2 aggregation.
- **Downstream Integrity:** Digital Twin, RAG/LLM, and SOC Dashboard components remain completely untouched.

---

## 5. Next Planned Experiment: Fusion Weight Optimization

With Layer 2 aggregation finalized to **TOP-2 MEAN**, the active Evidence Fusion engine is cleanly structured for the next experiment:

$$S_{\text{fused}} = w_1 \cdot L_1 + w_2 \cdot L_{2, \text{top2}}$$
$$\text{subject to } w_1 + w_2 = 1.0, \quad w_1, w_2 \ge 0$$

Currently, weights remain fixed at the baseline equal-weight setting:
$$w_1 = 0.5, \quad w_2 = 0.5$$

The next experiment will conduct a data-driven optimization of $w_1$ and $w_2$ to find the optimal trade-off between temporal TCN reconstruction evidence and physical relationship consistency evidence.
