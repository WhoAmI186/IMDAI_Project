# Layer 2 — Supervised XGBoost Experiment Cleanup Report

**Date:** September 30, 2026  
**Status:** Completed  
**Scope:** Removal of experimental supervised XGBoost code and model artifacts from the active implementation path.

---

## 1. Executive Summary

The controlled experiment evaluating supervised XGBoost classification against the active unsupervised physical-relationship XGBoost regressors has concluded. Following strict architectural guidelines, the supervised experimental branch has been **completely removed from the active implementation and model path**.

All empirical comparison findings, metric analyses, and diagnostic plots have been **preserved as historical research evidence** in the project's reporting directory.

### Key Confirmation Items:
- **Supervised XGBoost Experiment Removed:** All source code, training scripts, evaluation scripts, and model serialization artifacts associated with the supervised experiment have been deleted from active directories.
- **Results Preserved as Historical/Experimental Evidence:** The comprehensive benchmark report ([reports/layer2_supervised_xgb_comparison.md](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/layer2_supervised_xgb_comparison.md)), raw evaluation metrics ([reports/layer2_supervised_xgb_comparison.json](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/layer2_supervised_xgb_comparison.json)), and all 6 associated diagnostic figures in `reports/figures/` are fully preserved.
- **Active Layer 2 Architecture Unchanged:** The active Layer 2 pipeline consists exclusively of the four unsupervised physical-relationship XGBoost regressors:
  1. PV Inverter (`pv_inverter`: $P_{dc} \to P_{ac}$)
  2. Wind Anemometer (`wind_speed`: $v_{wind,A} \to v_{wind,B}$)
  3. Wind Temperature (`wind_temperature`: $T_{nacelle,A} \to T_{nacelle,B}$)
  4. PV Thermal (`pv_thermal`: $T_{ambient} \to T_{cell}$)
- **Layer 1 Completely Untouched:** Layer 1 (Causal TCN Autoencoder, Config A, Mean Feature MSE, $P_{99.5} = 0.000970$) was not modified in any way. Checkpoint SHA-256 (`0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22`) remains byte-identical.
- **No Retraining Performed:** No models were retrained or modified during this cleanup.
- **Downstream Components Untouched:** No downstream Evidence Fusion, Digital Twin, RAG/LLM, or Dashboard components were created or modified.

---

## 2. Inventory of Deleted Files

The following experimental source modules and model artifacts were permanently removed from the active codebase:

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Source Code** | `src/ml/layer2_supervised_xgb.py` | Experimental supervised XGBoost classifier wrapper and residual feature extractor |
| **Source Code** | `src/ml/layer2_supervised_training.py` | Experimental training script for direct and residual-based supervised XGBoost |
| **Source Code** | `src/ml/layer2_supervised_evaluation.py` | Experimental evaluation script comparing supervised vs. unsupervised Layer 2 |
| **Model Artifact** | `models/layer2_supervised_xgb.json` | Direct supervised XGBoost booster JSON |
| **Model Artifact** | `models/layer2_supervised_xgb.meta.json` | Direct supervised XGBoost metadata JSON |
| **Model Artifact** | `models/layer2_supervised_residuals_xgb.json` | Residual-based supervised XGBoost booster JSON |
| **Model Artifact** | `models/layer2_supervised_residuals_xgb.meta.json` | Residual-based supervised XGBoost metadata JSON |

---

## 3. Inventory of Preserved Files

### 3.1 Active Production Code & Models (Completely Untouched)

| Category | File Path | Status | Verification |
| :--- | :--- | :--- | :--- |
| **Layer 1 Checkpoint** | `models/tcn_autoencoder_baseline.pt` | Preserved / Untouched | SHA-256: `0DAAB40B...4A0F22` |
| **Layer 1 Detector** | `src/ml/layer1_tcn_detector.py` | Preserved / Untouched | Active TCN inference pipeline |
| **Layer 1 Scoring** | `src/ml/anomaly_scoring.py` | Preserved / Untouched | Active scoring & persistence |
| **Layer 1 Preprocessing**| `src/ml/preprocessing.py` | Preserved / Untouched | Robust scaler & temporal splits |
| **Layer 2 Detector** | `src/ml/layer2_physical_relationships.py` | Preserved / Untouched | Active 4-regressor physical checker |
| **Layer 2 Training** | `src/ml/layer2_training.py` | Preserved / Untouched | Active regressor training pipeline |
| **Layer 2 Evaluation** | `src/ml/layer2_evaluation.py` | Preserved / Untouched | Active physics evaluation pipeline |
| **Layer 2 Model** | `models/layer2_pv_inverter_xgb.json` | Preserved / Untouched | $P_{99.5} = 4.298\text{ kW}$ |
| **Layer 2 Model** | `models/layer2_wind_speed_xgb.json` | Preserved / Untouched | $P_{99.5} = 0.536\text{ m/s}$ |
| **Layer 2 Model** | `models/layer2_wind_temperature_xgb.json` | Preserved / Untouched | $P_{99.5} = 0.655\text{ }^\circ\text{C}$ |
| **Layer 2 Model** | `models/layer2_pv_thermal_xgb.json` | Preserved / Untouched | $P_{99.5} = 3.997\text{ }^\circ\text{C}$ |
| **Layer 2 Metadata** | `models/layer2_metadata.json` | Preserved / Untouched | Thresholds, features, test metrics |
| **ML Init Entrypoint** | `src/ml/__init__.py` | Preserved / Verified | Cleanly exports L1 & active L2 |

### 3.2 Historical Research Evidence Preserved

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Comparison Report** | `reports/layer2_supervised_xgb_comparison.md` | Comprehensive experimental report (methodology, leave-one-campaign-out CV, decision analysis) |
| **Metrics Data** | `reports/layer2_supervised_xgb_comparison.json` | Machine-readable metrics across all models, campaigns, and cross-validation splits |
| **Diagnostic Plot** | `reports/figures/layer2_supervised_roc_curves.png` | ROC curves comparing unsupervised vs. direct & residual supervised XGBoost |
| **Diagnostic Plot** | `reports/figures/layer2_supervised_pr_curves.png` | Precision-Recall curves across evaluation scenarios |
| **Diagnostic Plot** | `reports/figures/layer2_supervised_confusion_matrix.png` | Confusion matrices at default and optimal operating thresholds |
| **Diagnostic Plot** | `reports/figures/layer2_supervised_per_campaign.png` | Per-campaign detection rate breakdown highlighting zero-day vs. seen attack performance |
| **Diagnostic Plot** | `reports/figures/layer2_supervised_vs_unsupervised.png` | Side-by-side comparison of F1, Recall, Precision, and generalization capabilities |
| **Diagnostic Plot** | `reports/figures/layer2_supervised_threshold_curve.png` | Threshold sensitivity sweep across decision thresholds |

---

## 4. Verification and Validation

1. **Active Pipeline Verification:**
   Executed programmatic import and loader test:
   ```python
   import src.ml as ml
   l1 = ml.load_active_layer1_detector()
   l2 = ml.load_active_layer2_detector()
   ```
   - Layer 1 loaded successfully: `Layer1TCNDetector` (Threshold = `0.000970`, 49,438 parameters, CUDA-accelerated).
   - Layer 2 loaded successfully: `Layer2PhysicalRelationships` with active model set:
     `['pv_inverter', 'wind_speed', 'wind_temperature', 'pv_thermal']`.
2. **Codebase Cleanliness:**
   - Ripgrep search across `src/` confirmed zero remaining references or imports to `layer2_supervised_xgb`, `layer2_supervised_training`, or `layer2_supervised_evaluation`.
   - Models directory contains strictly the 5 active Layer 2 files (`layer2_metadata.json` + 4 booster JSONs) and the Layer 1 checkpoint (`tcn_autoencoder_baseline.pt`). Zero supervised artifacts remain.
3. **Architectural Separation:**
   - Layer 1 and Layer 2 remain parallel independent anomaly detectors.
   - Downstream components (Evidence Fusion, Digital Twin, RAG/LLM) remain untouched and awaiting their dedicated project phases.
