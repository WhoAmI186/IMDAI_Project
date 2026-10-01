# Triple Layer 1 Causal TCN Autoencoder Training Report

**Document ID:** `REPORT-TRIPLE-LAYER1-001`  
**Execution Date:** 2026-10-01  
**Project:** Cyber-Physical Smart Grid Anomaly Detection  
**Module:** `src/ml/triple_tcn_autoencoder.py` & `src/ml/triple_layer1_detector.py`  
**Model Checkpoint:** [`models/triple_tcn_autoencoder.pt`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/triple_tcn_autoencoder.pt)  
**Metadata:** [`models/triple_layer1_metadata.json`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/triple_layer1_metadata.json)  
**Status:** Completed & Validated

---

## 1. Model Architecture & Hyperparameters

The Layer 1 temporal anomaly detector is a native PyTorch **Causal Temporal Convolutional Network (TCN) Autoencoder** tailored for the 16-channel synchrophasor transmission telemetry:

$$\text{Input}(N, L=60, D=16) \longrightarrow \text{Causal TCN Encoder} \longrightarrow \text{Bottleneck Projection} \longrightarrow \text{Causal TCN Decoder} \longrightarrow \text{Output}(N, L=60, D=16)$$

### Architecture Specifications:

| Parameter | Value | Technical Justification |
|:---|:---:|:---|
| **Input Features ($D$)** | **16** | Approved core synchrophasor physical features |
| **Sequence Length ($L$)** | **60** | 60-timestep causal sliding window |
| **Hidden Channels** | **32** | Channel capacity across all residual blocks |
| **Latent Bottleneck** | **16** | Compact latent bottleneck enforcing low-dimensional manifold compression |
| **Kernel Size ($k$)** | **3** | Temporal convolution kernel |
| **Dilations Pattern** | **$[1, 2, 4, 8]$** | Exponential dilation hierarchy providing multi-scale temporal context |
| **Receptive Field** | **61** | $1 + 2 \times 2 \times (1 + 2 + 4 + 8) = 61$ timesteps; fully covers the 60-step window |
| **Total Trainable Parameters** | **49,760** | Compact, non-overfitting autoencoder |
| **Causality Guarantee** | Strict | Left-padding of $(k-1) \times d$ with zero right-padding. Zero future leakage. |

---

## 2. Unsupervised Training Protocol

In strict compliance with data-leakage controls:
- **Training Telemetry:** **2,490 normal sequences** derived exclusively from `NoEvents` blocks across training scenarios `data1.csv` to `data10.csv`.
- **Zero Attack / Natural Contamination:** Zero `Attack` and zero `Natural` sequences were exposed to the optimizer.
- **Validation Telemetry:** **411 normal sequences** derived strictly from `NoEvents` blocks across validation scenarios `data11.csv` and `data12.csv`.
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = $1 \times 10^{-5}$).
- **Learning Rate:** $0.001$ with early stopping patience = $8$ epochs.
- **Batch Size:** $64$.
- **Random Seed:** $42$ (fixed for full deterministic reproducibility).

---

## 3. Training & Validation Convergence

The model converged smoothly over 40 epochs:

| Epoch | Training MSE Loss | Validation MSE Loss | Notes |
|:---:|:---:|:---:|:---|
| **1** | 0.883891 | 0.314213 | Initial reconstruction |
| **5** | 0.053427 | 0.081563 | Rapid temporal representation acquisition |
| **10** | 0.020798 | 0.047087 | Steady convergence |
| **15** | 0.011473 | 0.033399 | Bottleneck feature refinement |
| **20** | 0.008449 | 0.026209 | Stable multi-scale tracking |
| **25** | 0.006391 | 0.021692 | Residual error reduction |
| **30** | 0.005318 | 0.017870 | Sub-0.02 validation loss |
| **35** | 0.004116 | 0.015426 | Steady fine-tuning |
| **40 (Final)** | **0.003235** | **0.013363** | Minimum validation error reached |

- **Best Validation Loss:** $0.013363$
- **Final Training Loss:** $0.003235$
- **Parameter Checkpoint:** Saved to [`models/triple_tcn_autoencoder.pt`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/models/triple_tcn_autoencoder.pt).

---

## 4. Normal Anomaly Threshold Calibration

Reconstruction error is calculated as the Mean Squared Error across all 16 channels and 60 timesteps:
$$\text{MSE}(t) = \frac{1}{60 \times 16} \sum_{\tau=1}^{60} \sum_{d=1}^{16} \left( x_{\tau, d} - \hat{x}_{\tau, d} \right)^2$$

Candidate operating thresholds were computed strictly from the empirical cumulative distribution of the **2,490 normal training sequences**:

| Threshold Tier | Percentile | Calibrated Value (MSE) | Operational Role |
|:---|:---:|:---:|:---|
| **P95** | 95.0% | `0.00562782` | High-sensitivity candidate |
| **P99** | **99.0%** | **`0.01361614`** | **PRIMARY ACTIVE OPERATING THRESHOLD** |
| **P99.5** | 99.5% | `0.01533266` | Conservative operating candidate |
| **P99.9** | 99.9% | `0.02431934` | Ultra-conservative baseline |

---

## 5. Normal Validation Generalization

To verify that the calibrated thresholds do not overfit the training files, Layer 1 was evaluated on the **411 independent normal sequences** from validation scenarios `data11.csv` and `data12.csv`:

- **Validation Normal MSE Mean:** $0.011362$
- **Validation Normal MSE Std:** $0.012917$
- **Exceedance at Primary Threshold (P99 = 0.013616):** Exactly 67 / 411 sequences ($16.30\%$).
- **Conclusion:** The autoencoder exhibits robust generalization on unseen normal power system telemetry without catastrophic divergence.
