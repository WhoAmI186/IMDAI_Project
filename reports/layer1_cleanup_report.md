# Layer 1 Cleanup & Final Architecture Report

**Project:** Smart Grid Cyber-Physical Anomaly Detection  
**Workspace:** c:\Users\LENOVO\Downloads\IMDAI project  
**Date:** September 29, 2026  
**Status:** Verification & Cleanup Completed  

---

## 1. What Active Code Was Retained

The active production codebase in src/ml/ contains strictly the modular components required to run, evaluate, and train the active Layer 1 Causal TCN pipeline:

- src/ml/tcn_autoencoder.py: Native PyTorch Causal Temporal Convolutional Network Autoencoder architecture with weight-normalized residual blocks and causal left-padding (receptive field = 61 steps, 49,438 parameters).
- src/ml/layer1_tcn_detector.py: Primary production inference wrapper (Layer1TCNDetector, load_active_layer1_detector) executing end-to-end data scaling, sliding-window generation, Mean Feature MSE scoring, and P99.5 threshold evaluation.
- src/ml/train_and_evaluate_tcn.py: Active standalone training and multi-campaign evaluation script for TCN Layer 1.
- src/ml/sequence_generator.py: Boundary-safe sliding-window generation utility for temporal autoencoders (=60$, =1$).
- src/ml/anomaly_scoring.py: Candidate threshold calibration ($, $, .5$, .9$, Mean+3$\\sigma$) and preserved experimental temporal persistence utility (compute_temporal_persistence).
- src/ml/attack_evaluation.py: Model-agnostic dataset extraction, verified attack interval parsing, and classification metric calculations.
- src/ml/scope_appropriate_evaluation.py: Model-agnostic scope filtering and Strategy D sequence labeling.
- src/ml/preprocessing.py: Normal baseline preprocessor establishing the authoritative 14 physical channels.
- src/ml/baseline_analysis.py: Statistical baseline characterization utilities.
- src/ml/__init__.py: Clean package exports exposing only active components and utilities.
- models/tcn_autoencoder_baseline.pt: Active frozen trained TCN checkpoint ({val} = 0.000561$, calibrated .5 = 0.000970$).

---

## 2. What Experimental Code Was Removed

All legacy, recurrent, and non-active experimental implementations and checkpoints have been removed:
- models/tcn_autoencoder_config_d.pt: Removed from active model directory.
- src/ml/feature_engineering.py: Removed from active ML source (Config D code).
- src/ml/lstm_autoencoder.py: Removed.
- src/ml/gru_autoencoder.py: Removed.
- src/ml/train_and_evaluate_phase3b.py: Removed.
- src/ml/anomaly_score_analysis.py: Removed.
- src/ml/train_and_evaluate_gru_tcn.py: Removed.
- src/archive/: Completely removed.
- models/archive/: Completely removed.
- All legacy LSTM/GRU model checkpoints (models/lstm_autoencoder_*.pt, models/gru_autoencoder_baseline.pt): Removed.
- All legacy experiment notebooks (
otebooks/phase2c_*.ipynb, 
otebooks/phase3b_*.ipynb, 
otebooks/phase3c_*.ipynb, 
otebooks/phase3d_*.ipynb): Removed.

---

## 3. What Historical Results Were Preserved

All historical results across all phases are preserved in authoritative documentation and machine-readable data files:
- 
eports/layer1_experiment_history.md: Complete consolidated history from Phase 2C through Phase 3E.
- 
eports/tcn_config_a_vs_d_comparison.md: Dedicated formal report for Config A vs Config D and score aggregations.
- 
eports/tcn_config_a_vs_d_experiment.json: Complete machine-readable results for Config A vs D and all 6 scoring aggregations.
- 
eports/phase3d_gru_tcn_comparison.md & .json: Preserved 3-way architecture comparison metrics.
- 
eports/phase3b_feature_engineering_training.md & .json: Preserved feature ablation results.
- 
eports/phase3c_anomaly_score_analysis.md & .json: Preserved error decomposition analysis.
- 
eports/phase2c_*.md: Preserved baseline evaluation benchmarks.
- 
eports/figures/: Preserved all performance bar charts, loss curves, ROC/PR curves, and latency distributions.

---

## 4. What Happened to Config D

TCN Config D (24 features) was trained and evaluated under strictly identical conditions to Config A. It achieved {val} = 0.000770$ and .5 = 0.003619$. 
- **Decision:** Config D was **rejected for active Layer 1 deployment** because its engineered rolling and difference features introduced substantial background variance, inflating the decision threshold by 3.73x. This caused an unacceptable 91.35% false negative rate and collapsed episode detection from 40.13% down to 10.20%.
- **Cleanup Action:** The experimental checkpoint models/tcn_autoencoder_config_d.pt and runtime feature engineering script src/ml/feature_engineering.py were removed from the active codebase. All experimental metrics and findings are permanently preserved in 
eports/tcn_config_a_vs_d_comparison.md and 
eports/layer1_experiment_history.md.

---

## 5. What Happened to the Score Aggregation Experiments

Six anomaly-score aggregation methods were evaluated on TCN Config A:
1. Mean Feature MSE ( = 0.0735$, DetRate $= 40.13\%$)
2. Maximum Feature MSE ( = 0.0629$, DetRate $= 24.01\%$)
3. Top-3 Feature MSE ( = 0.0674$, DetRate $= 32.57\%$)
4. Top-5 Feature MSE ( = 0.0688$, DetRate $= 36.18\%$)
5. Persistence P3 ( = 0.0730$, DetRate $= 40.46\%$)
6. Persistence P5 ( = 0.0727$, DetRate $= 40.13\%$)

- **Decision:** **Mean Feature MSE** was selected as the **active operational scoring method** because it achieved the highest F1 score and best consolidated cross-channel physical evidence.
- **Categorization:** Max, Top-3, Top-5, and P5 are classified as **historical experiments**.

---

## 6. Confirmation: P3 Was Preserved

**CONFIRMED.** The mathematical implementation of causal temporal persistence (=3$) is preserved in src/ml/anomaly_scoring.py via the utility function compute_temporal_persistence(scores, horizon=3). It is exported in src/ml/__init__.py for potential downstream multi-step persistence requirements.

---

## 7. Confirmation: P3 Is NOT Active

**CONFIRMED.** P3 is **NOT active** in the production pipeline. In Layer1TCNDetector (src/ml/layer1_tcn_detector.py), sequence scoring operates strictly via standard, instantaneous Mean Feature MSE without temporal persistence.

---

## 8. Confirmation: Mean Feature MSE Is Active

**CONFIRMED.** The active Layer 1 inference class Layer1TCNDetector calculates scalar anomaly scores strictly as:
\\text{score}_i = \\frac{1}{60 \\times 14} \\sum_{t=1}^{60} \\sum_{d=1}^{14} (X_{i,t,d} - \\hat{X}_{i,t,d})^2

---

## 9. Confirmation: Config A Is Active

**CONFIRMED.** The active Layer 1 detector ingests strictly the **14 authoritative raw physical telemetry channels** (AUTHORITATIVE_14_FEATURES). No engineered, difference, rolling, or residual features are computed in the active pipeline.

---

## 10. Confirmation: No Model Was Retrained

**CONFIRMED.** The active baseline model checkpoint models/tcn_autoencoder_baseline.pt was **not retrained** during cleanup. Its weights, architecture, scaler bounds, and calibrated .5 = 0.000970$ threshold remain byte-for-byte identical to its frozen Phase 3D state.

---

## 11. Confirmation: Layer 2 & Downstream Components Untouched

**CONFIRMED.** No modifications, implementations, or restructuring were made to:
- Layer 2 (Physical Relationships)
- Evidence Fusion
- Digital Twin
- RAG & LLM Agents
- Operator Dashboard

---

## 12. Final Active Layer 1 Pipeline

\\boxed{\\begin{matrix}
\\text{\\textbf{Smart Grid Process Telemetry (Solar/PV \\& Wind)}} \\\\
\\downarrow \\\\
\\text{\\textbf{14 Authoritative Raw Physical Features (Config A)}} \\\\
\\downarrow \\\\
\\text{\\textbf{MinMaxScaler ([-1.0, 1.0], fitted strictly on normal train)}} \\\\
\\downarrow \\\\
\\text{\\textbf{60-Timestep Sequences (L=60, stride s=1)}} \\\\
\\downarrow \\\\
\\text{\\textbf{Causal TCN Autoencoder (Receptive field = 61 steps, 49,438 params)}} \\\\
\\downarrow \\\\
\\text{\\textbf{Mean Feature MSE Anomaly Score}} \\\\
\\downarrow \\\\
\\text{\\textbf{P99.5 Decision Threshold = 0.000970}} \\\\
\\downarrow \\\\
\\text{\\textbf{Temporal Anomaly Evidence (Continuous scores, binary detections, residuals)}}
\\end{matrix}}

### Active Architectural Summary
| Component | Operational Specification |
| :--- | :--- |
| **Model Architecture** | Causal Temporal Convolutional Network Autoencoder (TCN-AE) |
| **Active Checkpoint** | models/tcn_autoencoder_baseline.pt |
| **Active Inference Class** | src.ml.layer1_tcn_detector.Layer1TCNDetector |
| **Input Features** | 14 raw physical channels (Config A) |
| **Temporal Window** | 60 time steps (=60$, stride =1$) |
| **Receptive Field** | 61 time steps (strictly causal, left-padded) |
| **Parameter Count** | 49,438 trainable parameters |
| **Anomaly Score** | Mean Feature MSE over complete  \\times 14$ window |
| **Decision Threshold** | .5 = 0.000970$ (calibrated strictly on normal baseline) |
| **Observable Scope F1** | .0735$ |
| **Observable Episode Det** | .13\\%$ ( / 304$ attack episodes) |
| **Median Latency** | .29\\text{ s}$ |
