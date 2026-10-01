# IMDAI Project Comprehensive Inventory & Reorganization Audit

**Document ID:** `REPORT-PROJECT-CLEANUP-AUDIT-001`  
**Execution Date:** 2026-10-01  
**Project Root:** `c:\Users\LENOVO\Downloads\IMDAI project\`  
**Audit Mode:** Read-Only (Zero Files Moved, Renamed, Deleted, or Altered)  
**Total Scanned Files:** 497 files (~8.96 GB total disk footprint)  

---

## 1. Project Overview

The **Cyber-Physical Smart Grid Anomaly Detection and Investigation (IMDAI)** system is designed to monitor multi-substation transmission grid synchrophasor telemetry (PMU), detect anomalies using an unsupervised dual-layer machine learning architecture, verify grid physics and switchgear topology through a deterministic Digital Twin, and formulate structured, operator-oriented forensic explanations via a downstream LLM Investigation layer.

### Two Evolutionary Generations in the Repository:
1. **Generation 1 (Historical Baseline / Exploratory Phase):**
   Developed on a DuckDB-backed synthetic solar and wind SCADA/PLC simulation dataset (`dataset/attacks/`, `dataset/normal/`, `dataset/merged_datasets.duckdb`). Featured initial LSTM/TCN autoencoders and XGBoost physical regressors for inverter and wind turbine relationships (`src/ml/preprocessing.py`, `src/ml/layer1_tcn_detector.py`, `src/ml/layer2_physical_relationships.py`). This work established proof-of-concept evidence fusion and sensitivity analysis, but has been completely superseded.
2. **Generation 2 (Current Production Active Pipeline):**
   Built upon the canonical 15-dataset power transmission grid synchrophasor benchmark (`dataset/triple/`), featuring a 2-bus, 2-line transmission network with 4 circuit breakers (`topology/grid_topology.json`). Utilizes a Causal TCN Autoencoder (Layer 1), 4 physical transmission XGBoost regressors (Layer 2), P99 Evidence Fusion, a full deterministic Power System Digital Twin, and an LLM Investigation Layer (`src/llm/`). All 36 automated unit and scenario tests currently validate this pipeline with a 100% pass rate.

---

## 2. Complete Directory Tree

```
IMDAI project/
│
├── .pytest_cache/                                # [Cache / Temporary] (5 files, 0.00 MB)
│   ├── .gitignore
│   ├── CACHEDIR.TAG
│   └── v/cache/
│       ├── lastfailed
│       ├── nodeids
│       └── stepwise
│
├── dataset/                                      # [Dataset] (187 files, 8,928.93 MB)
│   ├── attacks/                                  # Historical Gen 1 attacks (140 files, 4,116.32 MB)
│   │   ├── 20260228_openai/                     # Campaign 1: process.parquet, Zeek, NSM, PCAP, etc.
│   │   ├── 20260301_google/                     # Campaign 2: process.parquet, Zeek, NSM, PCAP, etc.
│   │   ├── 20260301_sonnet/                     # Campaign 3: process.parquet, Zeek, NSM, PCAP, etc.
│   │   ├── 20260302_minimax/                    # Campaign 4: process.parquet, Zeek, NSM, PCAP, etc.
│   │   └── 20260303_sonnet/                     # Campaign 5: process.parquet, Zeek, NSM, PCAP, etc.
│   ├── normal/                                   # Historical Gen 1 normal baseline (27 files, 402.94 MB)
│   │   └── 20260225_normal/                     # Baseline telemetry, PCAPs, Zeek logs
│   ├── topology/                                 # Historical Gen 1 SCADA topology (3 files, 0.01 MB)
│   │   ├── modbus.json
│   │   ├── network.json
│   │   └── s7_connections.json
│   ├── impact_assessment.duckdb                  # Historical Gen 1 database (1 file, 2.26 MB)
│   ├── merged_datasets.duckdb                    # Historical Gen 1 database (1 file, 4,329.51 MB)
│   └── triple/                                   # CURRENT ACTIVE BENCHMARK DATASET (15 files, 77.88 MB)
│       ├── data1.csv .. data4.csv                # Training Split (NoEvents normal telemetry)
│       ├── data5.csv .. data6.csv                # Validation Split (NoEvents normal telemetry)
│       ├── data7.csv .. data9.csv                # Natural Events (Transmission faults & breaker ops)
│       ├── data10.csv .. data12.csv              # Cyberattack Scenarios (FDI, uncoordinated trips)
│       └── data13.csv .. data15.csv              # Held-Out Blind Test Benchmark
│
├── experiments/                                  # [Historical Experiments] (4 files, 0.11 MB)
│   ├── fusion_weight_optimization_pso.py         # Historical PSO fusion optimization on Gen 1 data
│   ├── layer2_aggregation/
│   │   └── layer2_aggregation_experiment.py      # Historical Layer 2 aggregation study
│   └── threshold_sensitivity/
│       ├── asymmetric_threshold_sensitivity.py   # Historical asymmetric threshold study
│       └── threshold_sensitivity.py              # Historical P95-P99.9 threshold sensitivity
│
├── models/                                       # [Models / Checkpoints] (14 files, 1.93 MB)
│   ├── layer2_metadata.json                      # Historical Gen 1 Layer 2 metadata (6.1 KB)
│   ├── layer2_pv_inverter_xgb.json               # Historical Gen 1 Solar Inverter model (180.9 KB)
│   ├── layer2_pv_thermal_xgb.json                # Historical Gen 1 Solar Thermal model (172.0 KB)
│   ├── layer2_wind_speed_xgb.json                # Historical Gen 1 Wind Speed model (190.6 KB)
│   ├── layer2_wind_temperature_xgb.json          # Historical Gen 1 Wind Temp model (192.3 KB)
│   ├── tcn_autoencoder_baseline.pt               # Historical Gen 1 TCN-AE baseline (243.5 KB)
│   ├── triple_layer1_metadata.json               # ACTIVE Layer 1 Metadata & Threshold (0.7 KB)
│   ├── triple_layer2_bus1_voltage_xgb.json       # ACTIVE Layer 2 Bus 1 Voltage Model (167.4 KB)
│   ├── triple_layer2_bus2_voltage_xgb.json       # ACTIVE Layer 2 Bus 2 Voltage Model (181.9 KB)
│   ├── triple_layer2_line1_current_xgb.json      # ACTIVE Layer 2 Line 1 Current Model (196.4 KB)
│   ├── triple_layer2_line2_current_xgb.json      # ACTIVE Layer 2 Line 2 Current Model (196.9 KB)
│   ├── triple_layer2_metadata.json               # ACTIVE Layer 2 Metadata & Thresholds (1.9 KB)
│   ├── triple_scaler.json                        # ACTIVE Layer 1 RobustScaler Parameters (1.5 KB)
│   └── triple_tcn_autoencoder.pt                 # ACTIVE Layer 1 Causal TCN-AE Checkpoint (242.7 KB)
│
├── notebooks/                                    # [Exploratory Notebooks] (5 files, 0.05 MB)
│   ├── inspect_parquet.ipynb                     # Historical Gen 1 inspection notebook (2.1 KB)
│   ├── phase2a_baseline_analysis.ipynb           # Historical Gen 1 baseline exploration (8.4 KB)
│   ├── phase2b_preprocessing.ipynb               # Historical Gen 1 LSTM sequence design (12.1 KB)
│   ├── phase2c_attack_ground_truth_mapping.ipynb # Historical Gen 1 attack mapping (11.2 KB)
│   └── phase3a_feature_engineering_analysis.ipynb# Historical Gen 1 feature engineering (12.7 KB)
│
├── reports/                                      # [Reports & Figures] (162 files, 28.98 MB)
│   ├── *.md                                      # 41 Markdown Documentation & Audit Reports
│   ├── *.json                                    # 28 JSON Evaluation & Benchmark Metrics
│   ├── *.csv                                     # 3 CSV Event Analysis Records
│   ├── experiments/                              # Experiment Reports & Figures
│   │   ├── figures/                              # 17 PNG Experiment Figures
│   │   └── threshold_sensitivity/                # Threshold Sensitivity Sub-Reports & Figures
│   └── figures/                                  # 73 PNG Architectural, ROC, & Robustness Figures
│
├── scratch/                                      # [Scratch / Scripts] (24 files, 0.18 MB)
│   ├── analyze_attacks.py                        # Ad-hoc analysis script
│   ├── analyze_physical_relationships.py         # Ad-hoc physical relationship test
│   ├── audit_inventory_helper.py                 # Inventory audit utility
│   ├── audit_triple_dataset.py                   # Triple dataset schema audit
│   ├── check_constant_channels.py                # Telemetry constant channel check
│   ├── check_high_zero_cols.py                   # Zero-value distribution check
│   ├── check_model_memory.py                     # Qwen3-8B VRAM footprint calculator
│   ├── deep_diagnostic_audit.py                  # Core Digital Twin forensic audit
│   ├── detailed_audit.py                         # Triple feature audit
│   ├── evaluate_digital_twin_post_implementation.py # Core DT regression benchmark
│   ├── evaluate_llm_integration.py               # Core LLM 5-scenario evaluation
│   ├── evaluate_triple_pipeline.py               # Core pipeline held-out benchmark
│   ├── inspect_cases.py                          # Case inspection snippet
│   ├── print_block_summary.py                    # Dataset block summary snippet
│   ├── run_full_triple_audit.py                  # Full dataset feature audit
│   ├── run_preimplementation_validation.py       # DT pre-implementation study
│   ├── run_triple_robustness_audit.py            # Comprehensive robustness audit
│   ├── test_layer2_candidates.py                 # Ad-hoc regressor candidate test
│   ├── test_qwen_loading.py                      # Qwen3-8B loading smoke test
│   ├── train_triple_layer1.py                    # Core Layer 1 Causal TCN-AE training script
│   ├── train_triple_layer2.py                    # Core Layer 2 XGBoost training script
│   └── validate_digital_twin.py                  # DT validation runner
│
├── src/                                          # [Source Code] (87 files, 0.88 MB)
│   ├── __init__.py                               # Root package definition
│   ├── data/                                     # Data ingestion & loaders (6 files)
│   │   ├── duckdb_loader.py                      # Historical Gen 1 DuckDB loader
│   │   ├── evaluation_data.py                    # Historical Gen 1 evaluation loader
│   │   ├── network_data.py                       # Historical Gen 1 Zeek/NSM network loader
│   │   ├── process_data.py                       # Historical Gen 1 process telemetry loader
│   │   ├── triple_loader.py                      # ACTIVE Triple synchrophasor CSV loader
│   │   └── __init__.py
│   ├── digital_twin/                             # ACTIVE DIGITAL TWIN REASONING LAYER (9 files)
│   │   ├── digital_twin.py                       # Main PowerSystemDigitalTwin orchestrator
│   │   ├── event_investigator.py                 # Hypothesis generation & cause attribution
│   │   ├── physical_checks.py                    # Deterministic physical conservation laws
│   │   ├── schemas.py                            # Typed dataclass schemas & enumerations
│   │   ├── state.py                              # Dynamic grid state reconstruction
│   │   ├── telemetry_mapper.py                   # PMU feature mapping to physical components
│   │   ├── topology.py                           # Topology graph query & switchgear status
│   │   ├── topology_checks.py                    # Topology / breaker consistency verifier
│   │   └── __init__.py
│   ├── llm/                                      # ACTIVE LLM INVESTIGATION LAYER (6 files)
│   │   ├── config.py                             # Replaceable backend configuration
│   │   ├── investigator.py                       # SmartGridInvestigator end-to-end service
│   │   ├── llm_client.py                         # Multi-backend LLM client & JSON parser
│   │   ├── prompt_builder.py                     # Evidence grounding & prompt guardrails
│   │   ├── schemas.py                            # LLM input/output schemas & semantic checks
│   │   └── __init__.py
│   └── ml/                                       # Machine Learning Modules (20 files)
│       ├── triple_preprocessor.py                # ACTIVE Triple normalization & windowing
│       ├── triple_tcn_autoencoder.py             # ACTIVE Layer 1 Causal TCN-AE PyTorch architecture
│       ├── triple_layer1_detector.py             # ACTIVE Layer 1 inference & error scoring
│       ├── triple_layer2_regressors.py           # ACTIVE Layer 2 4x XGBoost physical models
│       ├── triple_evidence_fusion.py             # ACTIVE Layer 1 + Layer 2 P99 Evidence Fusion
│       ├── anomaly_scoring.py                    # Historical Gen 1 thresholding
│       ├── attack_evaluation.py                  # Historical Gen 1 attack metrics
│       ├── baseline_analysis.py                  # Historical Gen 1 normal statistics
│       ├── evidence_fusion.py                    # Historical Gen 1 evidence fusion
│       ├── evidence_fusion_evaluation.py         # Historical Gen 1 multi-campaign eval
│       ├── layer1_tcn_detector.py                # Historical Gen 1 Layer 1 detector
│       ├── layer2_evaluation.py                  # Historical Gen 1 Layer 2 eval
│       ├── layer2_physical_relationships.py      # Historical Gen 1 Layer 2 detector
│       ├── layer2_training.py                    # Historical Gen 1 Layer 2 trainer
│       ├── preprocessing.py                      # Historical Gen 1 preprocessor
│       ├── scope_appropriate_evaluation.py       # Historical Gen 1 evaluation utils
│       ├── sequence_generator.py                 # Historical Gen 1 sliding-window generator
│       ├── tcn_autoencoder.py                    # Historical Gen 1 TCN architecture
│       ├── train_and_evaluate_tcn.py             # Historical Gen 1 training runner
│       └── __init__.py
│
├── tests/                                        # [Automated Test Suite] (8 files, 0.17 MB)
│   ├── test_digital_twin.py                      # ACTIVE: 18 tests for Digital Twin pipeline
│   ├── test_llm_integration.py                   # ACTIVE: 7 tests for LLM Investigation layer
│   ├── test_physical_checks.py                   # ACTIVE: 6 tests for conservation laws
│   └── test_topology.py                          # ACTIVE: 5 tests for topology graph queries
│
└── topology/                                     # [Physical Configuration] (1 file, 0.01 MB)
    └── grid_topology.json                        # ACTIVE: Transmission grid single-line diagram
```

---

## 3. Current Active Pipeline

The production pipeline is organized strictly in a feed-forward, modular hierarchy:

```
                      Smart Grid PMU Telemetry (30 Hz)
                         [dataset/triple/data*.csv]
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
            [Layer 1: Causal TCN-AE]  [Layer 2: Physical XGBoost]
             src/ml/triple_tcn_*       src/ml/triple_layer2_*
             models/triple_tcn_*.pt    models/triple_layer2_*.json
                         │                       │
                         └───────────┬───────────┘
                                     ▼
                      [Layer 1 + Layer 2 Evidence Fusion]
                          src/ml/triple_evidence_fusion.py
                          (P99 Threshold Decision: 0.7410)
                                     │
                                     ▼
                      [Power System Digital Twin Layer]
                           src/digital_twin/*.py
                        topology/grid_topology.json
                         (State Reconstruction, Topology
                           & Physical Conservation Laws)
                                     │
                                     ▼
                        [Sanitization & Extraction]
                         LLMInvestigationInput
                                     │
                                     ▼
                        [LLM Investigation Layer]
                              src/llm/*.py
                       (PromptBuilder -> LLMClient ->
                         Semantic Integrity Validator)
                                     │
                                     ▼
                           LLMInvestigationOutput
                       (Typed, Grounded JSON Report:
                        Summary, Evidence, Hypotheses,
                        Recommendations, Limitations)
```

---

## 4. Active Files Audit

These files are actively imported, executed, or tested in the current project:

| Relative Path | Size | Role in Active Pipeline |
|:---|:---:|:---|
| `dataset/triple/data1.csv` .. `data15.csv` | 77.88 MB | Benchmark synchrophasor PMU datasets (Train, Val, Natural, Attack, Test). |
| `topology/grid_topology.json` | 10.02 KB | Single-line diagram specification for the 2-bus transmission system. |
| `models/triple_tcn_autoencoder.pt` | 242.7 KB | Trained PyTorch weights for Layer 1 Causal TCN Autoencoder. |
| `models/triple_scaler.json` | 1.5 KB | RobustScaler median and IQR parameters for normal PMU normalization. |
| `models/triple_layer1_metadata.json` | 0.7 KB | Layer 1 architecture parameters and P99 anomaly threshold (0.0766). |
| `models/triple_layer2_bus1_voltage_xgb.json` | 167.4 KB | Layer 2 XGBoost regressor predicting Bus 1 voltage. |
| `models/triple_layer2_bus2_voltage_xgb.json` | 181.9 KB | Layer 2 XGBoost regressor predicting Bus 2 voltage. |
| `models/triple_layer2_line1_current_xgb.json` | 196.4 KB | Layer 2 XGBoost regressor predicting Line 1 current. |
| `models/triple_layer2_line2_current_xgb.json` | 196.9 KB | Layer 2 XGBoost regressor predicting Line 2 current. |
| `models/triple_layer2_metadata.json` | 1.9 KB | Layer 2 feature columns, residual thresholds, and RMSE metrics. |
| `src/data/triple_loader.py` | 7.88 KB | Ingestion and split management for `dataset/triple/`. |
| `src/ml/triple_preprocessor.py` | 11.67 KB | PMU normalization and causal sliding sequence generation. |
| `src/ml/triple_tcn_autoencoder.py` | 12.31 KB | PyTorch Causal TCN-AE with causal dilation and residual blocks. |
| `src/ml/triple_layer1_detector.py` | 5.54 KB | Layer 1 reconstruction error evaluation and anomaly flagging. |
| `src/ml/triple_layer2_regressors.py` | 9.58 KB | Layer 2 physical residual computation and relationship evaluation. |
| `src/ml/triple_evidence_fusion.py` | 6.36 KB | Independent evidence combination: $0.7 \cdot L1 + 0.3 \cdot L2$. |
| `src/digital_twin/digital_twin.py` | 8.81 KB | Central orchestrator integrating state, physics, and investigation. |
| `src/digital_twin/event_investigator.py` | 13.98 KB | Deterministic hypothesis generation and cause attribution. |
| `src/digital_twin/physical_checks.py` | 13.97 KB | Kirchhoff's laws and transmission power-flow direction checks. |
| `src/digital_twin/schemas.py` | 4.44 KB | Typed data containers for Digital Twin inputs and outputs. |
| `src/digital_twin/state.py` | 5.62 KB | Dynamic grid state reconstruction (energized/de-energized lines). |
| `src/digital_twin/telemetry_mapper.py` | 5.46 KB | Mapping 129-column PMU channel names to topology components. |
| `src/digital_twin/topology.py` | 3.23 KB | Graph-based query module for bus, line, and breaker statuses. |
| `src/digital_twin/topology_checks.py` | 19.00 KB | Consistency verification between switchgear contacts and power flow. |
| `src/digital_twin/__init__.py` | 1.38 KB | Digital Twin package exports. |
| `src/llm/config.py` | 1.58 KB | Configuration dataclass for LLM backends and parameters. |
| `src/llm/investigator.py` | 4.88 KB | End-to-end investigation service bridging Digital Twin to LLM. |
| `src/llm/llm_client.py` | 20.26 KB | Multi-backend LLM client (Transformers, OpenAI-compatible, Expert). |
| `src/llm/prompt_builder.py` | 5.22 KB | Structured prompt builder enforcing OBSERVATIONS != HYPOTHESES. |
| `src/llm/schemas.py` | 12.16 KB | Typed schemas and semantic integrity validation for LLM outputs. |
| `src/llm/__init__.py` | 1.04 KB | LLM module package exports. |
| `tests/test_digital_twin.py` | 22.67 KB | 18 automated tests for the Digital Twin pipeline. |
| `tests/test_llm_integration.py` | 17.38 KB | 7 automated tests for LLM integration and semantic guardrails. |
| `tests/test_physical_checks.py` | 3.33 KB | 6 automated tests for deterministic physical conservation laws. |
| `tests/test_topology.py` | 1.68 KB | 5 automated tests for topology parsing and status queries. |

---

## 5. Historical & Experimental Files Audit

These files were created during earlier phases (Generation 1, exploratory research, or sensitivity experiments) and have zero imports from the active pipeline:

### 5.1 Historical Source Code (`src/`)
- `src/data/duckdb_loader.py` (6.86 KB): DuckDB database loader for the older solar/wind dataset.
- `src/data/evaluation_data.py` (8.79 KB): Evaluation batching on older DuckDB parquet data.
- `src/data/network_data.py` (6.06 KB): Zeek and commercial NSM log loader.
- `src/data/process_data.py` (15.89 KB): Process telemetry extraction from older parquet files.
- `src/ml/preprocessing.py` (13.98 KB): Preprocessor for older solar/wind telemetry.
- `src/ml/sequence_generator.py` (14.64 KB): Sliding-window generator for older telemetry.
- `src/ml/tcn_autoencoder.py` (16.84 KB): Original TCN architecture for older 28-feature data.
- `src/ml/train_and_evaluate_tcn.py` (14.36 KB): Training script for older TCN baseline.
- `src/ml/layer1_tcn_detector.py` (11.30 KB): Layer 1 detector for older solar/wind data.
- `src/ml/layer2_physical_relationships.py` (13.85 KB): Layer 2 detector for solar/wind relationships.
- `src/ml/layer2_training.py` (10.59 KB): Training pipeline for solar/wind XGBoost models.
- `src/ml/layer2_evaluation.py` (21.83 KB): Evaluation pipeline for solar/wind Layer 2 models.
- `src/ml/evidence_fusion.py` (24.91 KB): Evidence fusion engine for older solar/wind data.
- `src/ml/evidence_fusion_evaluation.py` (20.64 KB): Evaluation pipeline for older evidence fusion.
- `src/ml/anomaly_scoring.py` (10.74 KB): Anomaly scoring for older process telemetry.
- `src/ml/attack_evaluation.py` (11.25 KB): Attack evaluation utilities for older data.
- `src/ml/baseline_analysis.py` (14.77 KB): Baseline EDA for older normal telemetry.
- `src/ml/scope_appropriate_evaluation.py` (6.18 KB): Scope-appropriate evaluation for older detectors.

### 5.2 Historical Experiments (`experiments/`)
- `experiments/fusion_weight_optimization_pso.py` (39.12 KB): PSO fusion weight optimizer on Gen 1 data.
- `experiments/layer2_aggregation/layer2_aggregation_experiment.py` (20.05 KB): Layer 2 aggregation study.
- `experiments/threshold_sensitivity/threshold_sensitivity.py` (31.75 KB): P95-P99.9 threshold sensitivity.
- `experiments/threshold_sensitivity/asymmetric_threshold_sensitivity.py` (19.65 KB): Asymmetric threshold sensitivity.

### 5.3 Historical Models (`models/`)
- `models/tcn_autoencoder_baseline.pt` (243.5 KB): Trained TCN-AE on older solar/wind data.
- `models/layer2_pv_inverter_xgb.json` (180.9 KB): Trained PV inverter model on older data.
- `models/layer2_pv_thermal_xgb.json` (172.0 KB): Trained PV thermal model on older data.
- `models/layer2_wind_speed_xgb.json` (190.6 KB): Trained wind speed model on older data.
- `models/layer2_wind_temperature_xgb.json` (192.3 KB): Trained wind temp model on older data.
- `models/layer2_metadata.json` (6.1 KB): Layer 2 metadata for older solar/wind models.

---

## 6. Duplicate & Superseded Files

- **SCADA Topology vs. Grid Topology:**
  - `dataset/topology/modbus.json`, `network.json`, `s7_connections.json` (Historical Gen 1 SCADA network configs).
  - Superseded by `topology/grid_topology.json` (Active 2-bus transmission system topology).
- **Evidence Fusion Implementation:**
  - `src/ml/evidence_fusion.py` (Superseded by `src/ml/triple_evidence_fusion.py`).
- **Layer 1 Detector:**
  - `src/ml/layer1_tcn_detector.py` (Superseded by `src/ml/triple_layer1_detector.py`).
- **Layer 2 Regressors:**
  - `src/ml/layer2_physical_relationships.py` (Superseded by `src/ml/triple_layer2_regressors.py`).
- **Preprocessor:**
  - `src/ml/preprocessing.py` (Superseded by `src/ml/triple_preprocessor.py`).

---

## 7. Dataset Inventory

| Dataset Directory / File | File Count | Size | Used by Active Pipeline? | Recommended Disposition | Reason |
|:---|:---:|:---:|:---:|:---:|:---|
| `dataset/triple/` | 15 CSVs | 77.88 MB | **YES (Primary)** | **KEEP IN MAIN** | Core 15-dataset benchmark for active ML models, Digital Twin, and tests. |
| `dataset/attacks/` | 140 files | 4,116.32 MB | **NO** | **ARCHIVE** | Historical Gen 1 raw PCAP, Zeek, and parquet files. Occupies 4.12 GB. |
| `dataset/normal/` | 27 files | 402.94 MB | **NO** | **ARCHIVE** | Historical Gen 1 normal telemetry and network captures. Occupies 403 MB. |
| `dataset/merged_datasets.duckdb` | 1 file | 4,329.51 MB | **NO** | **ARCHIVE** | Historical Gen 1 compiled DuckDB database. Occupies 4.33 GB. |
| `dataset/impact_assessment.duckdb` | 1 file | 2.26 MB | **NO** | **ARCHIVE** | Historical Gen 1 impact database. |
| `dataset/topology/` | 3 JSONs | 0.01 MB | **NO** | **ARCHIVE** | Historical Gen 1 Modbus and S7 network topologies. |

*Storage Impact:* Archiving the older Generation 1 dataset frees **8.85 GB** (99.1%) of disk space from the active project root while preserving 100% of the active 77.88 MB dataset.

---

## 8. Model / Checkpoint Inventory

| Model / Metadata File | Size | Generation | Active Status | Reproducibility | Breaking Risk if Deleted |
|:---|:---:|:---:|:---:|:---|:---:|
| `models/triple_tcn_autoencoder.pt` | 242.7 KB | Gen 2 | **ACTIVE** | Reproducible via `scratch/train_triple_layer1.py` | **CRITICAL: Breaks Layer 1** |
| `models/triple_scaler.json` | 1.5 KB | Gen 2 | **ACTIVE** | Generated by `train_triple_layer1.py` | **CRITICAL: Breaks Preprocessor** |
| `models/triple_layer1_metadata.json` | 0.7 KB | Gen 2 | **ACTIVE** | Generated by `train_triple_layer1.py` | **CRITICAL: Breaks Layer 1** |
| `models/triple_layer2_bus1_voltage_xgb.json` | 167.4 KB | Gen 2 | **ACTIVE** | Reproducible via `scratch/train_triple_layer2.py` | **CRITICAL: Breaks Layer 2** |
| `models/triple_layer2_bus2_voltage_xgb.json` | 181.9 KB | Gen 2 | **ACTIVE** | Reproducible via `train_triple_layer2.py` | **CRITICAL: Breaks Layer 2** |
| `models/triple_layer2_line1_current_xgb.json` | 196.4 KB | Gen 2 | **ACTIVE** | Reproducible via `train_triple_layer2.py` | **CRITICAL: Breaks Layer 2** |
| `models/triple_layer2_line2_current_xgb.json` | 196.9 KB | Gen 2 | **ACTIVE** | Reproducible via `train_triple_layer2.py` | **CRITICAL: Breaks Layer 2** |
| `models/triple_layer2_metadata.json` | 1.9 KB | Gen 2 | **ACTIVE** | Generated by `train_triple_layer2.py` | **CRITICAL: Breaks Layer 2** |
| `models/tcn_autoencoder_baseline.pt` | 243.5 KB | Gen 1 | Historical | Reproducible via `train_and_evaluate_tcn.py` | None to active pipeline |
| `models/layer2_pv_inverter_xgb.json` | 180.9 KB | Gen 1 | Historical | Reproducible via `layer2_training.py` | None to active pipeline |
| `models/layer2_pv_thermal_xgb.json` | 172.0 KB | Gen 1 | Historical | Reproducible via `layer2_training.py` | None to active pipeline |
| `models/layer2_wind_speed_xgb.json` | 190.6 KB | Gen 1 | Historical | Reproducible via `layer2_training.py` | None to active pipeline |
| `models/layer2_wind_temperature_xgb.json` | 192.3 KB | Gen 1 | Historical | Reproducible via `layer2_training.py` | None to active pipeline |
| `models/layer2_metadata.json` | 6.1 KB | Gen 1 | Historical | Generated by `layer2_training.py` | None to active pipeline |

---

## 9. Notebook Inventory

All 5 notebooks in `notebooks/` belong to the exploratory Phase 2 & 3 research on the older DuckDB dataset:

| Notebook Path | Size | Purpose | Conversion Status | Recommended Disposition |
|:---|:---:|:---|:---|:---:|
| `notebooks/inspect_parquet.ipynb` | 2.18 KB | Inspect schema of raw parquet files from Gen 1. | Converted to `duckdb_inspection.md`. | **ARCHIVE** |
| `notebooks/phase2a_baseline_analysis.ipynb` | 8.38 KB | Statistical EDA on normal solar/wind data. | Converted to `reports/phase2a_baseline_analysis.md`. | **ARCHIVE** |
| `notebooks/phase2b_preprocessing.ipynb` | 12.15 KB | Sliding window sequence design for LSTM. | Converted to `src/ml/preprocessing.py`. | **ARCHIVE** |
| `notebooks/phase2c_attack_ground_truth_mapping.ipynb` | 11.21 KB | Aligning attack start/end timestamps. | Converted to `reports/phase2c_attack_ground_truth_mapping.md`. | **ARCHIVE** |
| `notebooks/phase3a_feature_engineering_analysis.ipynb` | 12.70 KB | Feature correlation analysis for Layer 2. | Converted to `src/ml/layer2_physical_relationships.py`. | **ARCHIVE** |

---

## 10. Report Inventory

The `reports/` folder contains 162 total files:

### 10.1 Active Production Documentation & Certified Evidence (KEEP IN MAIN)
- `reports/triple_dataset_audit.md` & `.json`, `.csv`: Baseline audit of the active 15 PMU datasets.
- `reports/triple_preprocessing.md`: Normalization and feature specification for active pipeline.
- `reports/triple_layer1_training.md` & `.json`: Training documentation for active Causal TCN-AE.
- `reports/triple_layer2_training.md` & `.json`: Training documentation for active XGBoost models.
- `reports/triple_evaluation.md` & `.json`: Independent benchmark evaluation of active ML models.
- `reports/triple_robustness_audit.md` & `.json`: Forensic robustness and generalization audit.
- `reports/digital_twin_design.md`: Core architectural specification of the Power System Digital Twin.
- `reports/digital_twin_preimplementation_validation.md` & `.json`: Pre-implementation validation.
- `reports/digital_twin_validation.md` & `.json`: Validation study on held-out scenarios.
- `reports/digital_twin_diagnostic_audit.md` & `.json`: Diagnostic audit of breaker and state checks.
- `reports/digital_twin_post_implementation_evaluation.md` & `.json`: Post-implementation benchmark.
- `reports/llm_integration.md`: Certified LLM integration architecture report (36/36 tests passing).
- `reports/llm_evaluation_summary.json`: Preserved benchmark records (Deterministic & Real-model).
- `reports/qwen_crash_diagnostic.md`: Post-crash hardware and log diagnostic report.
- `reports/qwen3_real_inference_evaluation.md`: Real Qwen3-8B hardware boundary audit report.
- `reports/figures/triple_robustness/` (8 PNGs): Visual evidence supporting active pipeline audit.

### 10.2 Historical Research Reports (MOVE TO ARCHIVE)
- Reports documenting Phase 2A through Phase 3D (`reports/phase2*`, `reports/phase3*`).
- Reports documenting older Layer 1 and Layer 2 experiments (`reports/layer1_*`, `reports/layer2_*`).
- Reports documenting DuckDB data preparation (`reports/duckdb_inspection.md`, `data_preparation_plan.md`).
- Historical experiment reports in `reports/experiments/` (PSO optimization, Layer 2 aggregation, threshold sensitivity).
- Large intermediate event analysis CSVs (`reports/digital_twin_event_analysis.csv` ~2.95 MB, `reports/digital_twin_post_implementation_event_analysis.csv` ~2.59 MB).

---

## 11. Test Inventory

All test files reside in `tests/` and validate the current production pipeline:

| Test File | Size | Test Count | Target Component | Status | Disposition |
|:---|:---:|:---:|:---|:---:|:---:|
| `tests/test_digital_twin.py` | 22.67 KB | 18 | Digital Twin end-to-end event reasoning | **18/18 PASS** | **KEEP IN MAIN** |
| `tests/test_llm_integration.py` | 17.38 KB | 7 | LLM Investigation Layer & guardrails | **7/7 PASS** | **KEEP IN MAIN** |
| `tests/test_physical_checks.py` | 3.33 KB | 6 | Deterministic conservation laws | **6/6 PASS** | **KEEP IN MAIN** |
| `tests/test_topology.py` | 1.68 KB | 5 | GridTopology graph queries | **5/5 PASS** | **KEEP IN MAIN** |

*All 36/36 tests pass in 8.65 seconds.*

---

## 12. Scratch / Temporary Files Audit

The `scratch/` directory contains 24 files with distinct functional roles:

### 12.1 Core Training & Evaluation Scripts (PROMOTE to `scripts/`)
These scripts are essential for reproducing models, running benchmarks, and executing diagnostics:
1. `scratch/train_triple_layer1.py` &mdash; Trains the active Layer 1 Causal TCN-AE.
2. `scratch/train_triple_layer2.py` &mdash; Trains the active Layer 2 XGBoost models.
3. `scratch/evaluate_triple_pipeline.py` &mdash; Evaluates the ML pipeline on held-out test data.
4. `scratch/evaluate_digital_twin_post_implementation.py` &mdash; Evaluates the Digital Twin.
5. `scratch/evaluate_llm_integration.py` &mdash; Evaluates the LLM Investigation layer.
6. `scratch/run_triple_robustness_audit.py` &mdash; Generates the comprehensive robustness audit.
7. `scratch/deep_diagnostic_audit.py` &mdash; Executes forensic deep diagnostics on the Digital Twin.
8. `scratch/validate_digital_twin.py` &mdash; Validates Digital Twin hypotheses on test cases.
9. `scratch/check_model_memory.py` &mdash; Computes mathematical VRAM requirements for LLMs.

### 12.2 One-Off Exploratory Scripts (SAFE TO ARCHIVE)
- `scratch/audit_triple_dataset.py`
- `scratch/run_full_triple_audit.py`
- `scratch/detailed_audit.py`
- `scratch/run_preimplementation_validation.py`
- `scratch/analyze_attacks.py`
- `scratch/analyze_physical_relationships.py`
- `scratch/test_layer2_candidates.py`
- `scratch/test_qwen_loading.py`

### 12.3 Throwaway Debug Snippets (CANDIDATE FOR DELETION)
- `scratch/check_constant_channels.py` (1.1 KB)
- `scratch/check_high_zero_cols.py` (0.6 KB)
- `scratch/inspect_cases.py` (1.2 KB)
- `scratch/print_block_summary.py` (1.0 KB)
- `scratch/audit_inventory_helper.py` (4.9 KB)

---

## 13. Configuration & Environment Audit

| File / Component | Exists? | Status | Assessment |
|:---|:---:|:---:|:---|
| `.gitignore` | **NO** (Only inside `.pytest_cache/`) | **MISSING** | A root `.gitignore` is urgently needed to prevent caching directories (`.pytest_cache`, `__pycache__`) and local Hugging Face artifacts from being committed. |
| `requirements.txt` | **NO** | **MISSING** | A pinned `requirements.txt` is required to ensure reproducibility across Python 3.12, PyTorch 2.14+cu126, XGBoost, Transformers, and Accelerate. |
| `pyproject.toml` | **NO** | Optional | Recommended for standardizing package installation (`pip install -e .`). |
| `.env` / `.env.example` | **NO** | Optional | Recommended for configuring `LLM_MODEL`, `LLM_BACKEND`, and `LLM_API_BASE`. |
| `topology/grid_topology.json` | **YES** | **ACTIVE** | Must remain in main project at `topology/grid_topology.json`. |

---

## 14. Dependency & Reference Analysis

A cross-module reference scan was performed to ensure that no proposed archive or deletion candidate breaks any active component:
- **No Active File Imports Gen 1 Code:** None of `src/digital_twin/`, `src/llm/`, `src/data/triple_loader.py`, or `src/ml/triple_*` import from `src/ml/layer1_tcn_detector.py`, `src/ml/layer2_physical_relationships.py`, `src/ml/evidence_fusion.py`, or `src/data/duckdb_loader.py`.
- **Active Tests Depend Exclusively on Active Files:** All 36 tests in `tests/` import strictly from `src.digital_twin`, `src.llm`, and `topology.grid_topology`.
- **Model Checkpoints:** Active code loads only the 8 `triple_*` files in `models/`. The 6 older model files (`tcn_autoencoder_baseline.pt`, etc.) are completely unreferenced by active code.

---

## 15. Git Status

- **Git Repository Initialized:** **NO** (`fatal: not a git repository`).
- The directory is currently an unversioned folder on Windows (`c:\Users\LENOVO\Downloads\IMDAI project`).
- **Recommendation:** After cleaning and organizing the folder, initialize a clean Git repository (`git init`), add a comprehensive `.gitignore`, and create the initial commit of the clean active pipeline.

---

## 16. Recommended Main Project Structure

```
IMDAI project/
│
├── .gitignore                                    # NEW: Standard Python/PyTorch/IDE ignore rules
├── README.md                                     # NEW: Project documentation & quickstart
├── requirements.txt                              # NEW: Pinned dependencies
├── pyproject.toml                                # NEW: Package build & test configuration
│
├── dataset/
│   └── triple/                                   # ACTIVE BENCHMARK DATASET (15 CSVs)
│
├── models/                                       # ACTIVE CHECKPOINTS ONLY (8 files)
│   ├── triple_tcn_autoencoder.pt
│   ├── triple_scaler.json
│   ├── triple_layer1_metadata.json
│   ├── triple_layer2_bus1_voltage_xgb.json
│   ├── triple_layer2_bus2_voltage_xgb.json
│   ├── triple_layer2_line1_current_xgb.json
│   ├── triple_layer2_line2_current_xgb.json
│   └── triple_layer2_metadata.json
│
├── reports/                                      # ACTIVE DOCUMENTATION & CERTIFIED BENCHMARKS
│   ├── digital_twin_design.md
│   ├── digital_twin_post_implementation_evaluation.md
│   ├── digital_twin_validation.md
│   ├── llm_integration.md
│   ├── llm_evaluation_summary.json
│   ├── qwen3_real_inference_evaluation.md
│   ├── qwen_crash_diagnostic.md
│   ├── triple_dataset_audit.md
│   ├── triple_evaluation.md
│   ├── triple_robustness_audit.md
│   └── figures/triple_robustness/
│
├── scripts/                                      # PROMOTED REPRODUCIBILITY TOOLS (from scratch/)
│   ├── train_triple_layer1.py                    # Train Layer 1 Causal TCN-AE
│   ├── train_triple_layer2.py                    # Train Layer 2 XGBoost Regressors
│   ├── evaluate_triple_pipeline.py               # Benchmark test data
│   ├── evaluate_digital_twin.py                  # Evaluate Digital Twin
│   ├── evaluate_llm_integration.py               # Benchmark LLM Layer
│   ├── run_robustness_audit.py                   # Run robustness audit
│   └── check_model_memory.py                     # LLM VRAM budget calculator
│
├── src/                                          # PRODUCTION SOURCE CODE
│   ├── data/
│   │   ├── triple_loader.py
│   │   └── __init__.py
│   ├── digital_twin/                             # 9 active Digital Twin modules
│   ├── llm/                                      # 6 active LLM modules
│   └── ml/                                       # 5 active Triple ML modules
│       ├── triple_evidence_fusion.py
│       ├── triple_layer1_detector.py
│       ├── triple_layer2_regressors.py
│       ├── triple_preprocessor.py
│       ├── triple_tcn_autoencoder.py
│       └── __init__.py
│
├── tests/                                        # AUTOMATED TESTS (36/36 Passing)
│   ├── test_digital_twin.py
│   ├── test_llm_integration.py
│   ├── test_physical_checks.py
│   └── test_topology.py
│
└── topology/
    └── grid_topology.json                        # ACTIVE TRANSMISSION TOPOLOGY
```

---

## 17. Recommended Archive Structure

Historical files can be moved into a top-level `archive/` folder (or moved to external storage to save 8.85 GB):

```
archive/
│
├── dataset_gen1/                                 # Older Solar/Wind Datasets (~8.85 GB)
│   ├── attacks/
│   ├── normal/
│   ├── topology/
│   ├── impact_assessment.duckdb
│   └── merged_datasets.duckdb
│
├── experiments_gen1/                             # Historical experiment scripts
│   ├── fusion_weight_optimization_pso.py
│   ├── layer2_aggregation/
│   └── threshold_sensitivity/
│
├── models_gen1/                                  # Older model checkpoints (6 files, 0.95 MB)
│   ├── tcn_autoencoder_baseline.pt
│   └── layer2_*.json
│
├── notebooks_gen1/                               # Exploratory Jupyter Notebooks (5 files)
│   ├── inspect_parquet.ipynb
│   └── phase*.ipynb
│
├── reports_gen1/                                 # Historical research reports & figures
│   ├── phase2_*.md / .json
│   ├── phase3_*.md / .json
│   ├── layer1_*.md / layer2_*.md
│   ├── digital_twin_event_analysis.csv (large)
│   └── experiments/
│
└── src_gen1/                                     # Superseded Generation 1 source code
    ├── data/ (duckdb_loader, process_data, etc.)
    └── ml/ (preprocessing, layer1_tcn_detector, etc.)
```

---

## 18. Candidate Files for Deletion

These files are ephemeral caches or throwaway debug snippets that contain no reproducible value and can be safely deleted:

| File / Folder Path | Type | Size | Reason for Deletion |
|:---|:---:|:---:|:---|
| `.pytest_cache/` | Cache | ~12 KB | Automatically regenerated on next `pytest` run. |
| `**/__pycache__/` (7 directories) | Cache | 642 KB | Compiled Python bytecode; auto-regenerated by Python. |
| `scratch/check_constant_channels.py` | Scratch | 1.1 KB | Temporary 10-line ad-hoc check from dataset audit. |
| `scratch/check_high_zero_cols.py` | Scratch | 0.6 KB | Temporary 8-line ad-hoc check from dataset audit. |
| `scratch/inspect_cases.py` | Scratch | 1.2 KB | Temporary snippet used for quick console inspection. |
| `scratch/print_block_summary.py` | Scratch | 1.0 KB | Temporary snippet used for printing table blocks. |
| `scratch/audit_inventory_helper.py` | Scratch | 4.9 KB | Temporary helper script written during this audit. |

---

## 19. Files That MUST NOT Be Deleted

> [!CAUTION]
> Deleting ANY file listed below will immediately break model loading, pre-processing, Digital Twin reasoning, LLM investigation, or automated test execution.

1. **Active Checkpoints & Scalers (`models/`):**
   - `models/triple_tcn_autoencoder.pt`
   - `models/triple_scaler.json`
   - `models/triple_layer1_metadata.json`
   - `models/triple_layer2_bus1_voltage_xgb.json`
   - `models/triple_layer2_bus2_voltage_xgb.json`
   - `models/triple_layer2_line1_current_xgb.json`
   - `models/triple_layer2_line2_current_xgb.json`
   - `models/triple_layer2_metadata.json`
2. **Active Dataset (`dataset/triple/`):**
   - `dataset/triple/data1.csv` through `data15.csv` (all 15 CSV files).
3. **Active Grid Topology (`topology/`):**
   - `topology/grid_topology.json`.
4. **Active Source Modules (`src/`):**
   - `src/data/triple_loader.py`
   - `src/ml/triple_preprocessor.py`
   - `src/ml/triple_tcn_autoencoder.py`
   - `src/ml/triple_layer1_detector.py`
   - `src/ml/triple_layer2_regressors.py`
   - `src/ml/triple_evidence_fusion.py`
   - All 9 modules in `src/digital_twin/`
   - All 6 modules in `src/llm/`
5. **Active Test Suite (`tests/`):**
   - `tests/test_digital_twin.py`
   - `tests/test_llm_integration.py`
   - `tests/test_physical_checks.py`
   - `tests/test_topology.py`

---

## 20. Items Requiring User Confirmation

Before performing any cleanup or reorganization, your explicit decision is requested on these four strategic items:

1. **Generation 1 Dataset Disposition (8.85 GB):**
   Should the older DuckDB and raw attack/normal PCAP dataset (`dataset/attacks/`, `dataset/normal/`, `dataset/merged_datasets.duckdb`) be moved into a local `archive/dataset_gen1/` subfolder, or moved to an external drive / compressed ZIP to recover 8.85 GB of laptop SSD storage?
2. **Promotion of Core Scripts (`scratch/` &rarr; `scripts/`):**
   Do you approve promoting the 9 core training and evaluation scripts (`train_triple_layer1.py`, `train_triple_layer2.py`, `evaluate_triple_pipeline.py`, etc.) into a clean, permanent `scripts/` directory?
3. **Creation of Standard Configuration Files:**
   Do you approve generating a standardized root `.gitignore`, `requirements.txt`, and `pyproject.toml`?
4. **Git Initialization:**
   Do you want to initialize a local Git repository (`git init`) once the active structure is clean?

---

## Final Classification Lists (Safety Rule)

### A. KEEP IN MAIN PROJECT
- **Dataset:** `dataset/triple/` (all 15 files: `data1.csv` .. `data15.csv`)
- **Topology:** `topology/grid_topology.json`
- **Models:** All 8 active checkpoints (`triple_tcn_autoencoder.pt`, `triple_scaler.json`, `triple_layer1_metadata.json`, `triple_layer2_*.json`)
- **Source Code:**
  - `src/data/triple_loader.py`
  - `src/ml/triple_preprocessor.py`, `triple_tcn_autoencoder.py`, `triple_layer1_detector.py`, `triple_layer2_regressors.py`, `triple_evidence_fusion.py`
  - `src/digital_twin/` (all 9 files)
  - `src/llm/` (all 6 files)
- **Tests:** `tests/` (all 4 test files)
- **Core Reports:**
  - `reports/digital_twin_design.md`
  - `reports/digital_twin_post_implementation_evaluation.md`
  - `reports/digital_twin_validation.md`
  - `reports/digital_twin_diagnostic_audit.md`
  - `reports/digital_twin_preimplementation_validation.md`
  - `reports/llm_integration.md`
  - `reports/llm_evaluation_summary.json`
  - `reports/qwen3_real_inference_evaluation.md`
  - `reports/qwen_crash_diagnostic.md`
  - `reports/triple_dataset_audit.md`, `.json`, `.csv`
  - `reports/triple_evaluation.md`, `.json`
  - `reports/triple_layer1_training.md`, `.json`
  - `reports/triple_layer2_training.md`, `.json`
  - `reports/triple_preprocessing.md`
  - `reports/triple_robustness_audit.md`, `.json`
  - `reports/figures/triple_robustness/` (all 8 figures)
- **Core Scripts (to promote from `scratch/` to `scripts/`):**
  - `train_triple_layer1.py`
  - `train_triple_layer2.py`
  - `evaluate_triple_pipeline.py`
  - `evaluate_digital_twin_post_implementation.py`
  - `evaluate_llm_integration.py`
  - `run_triple_robustness_audit.py`
  - `deep_diagnostic_audit.py`
  - `validate_digital_twin.py`
  - `check_model_memory.py`

### B. MOVE TO ARCHIVE
*(Items moved here preserve research history while decluttering the main active workspace)*
- **`dataset/attacks/` (140 files, 4.12 GB):** Historical Gen 1 raw PCAP, Zeek, and parquet files unreferenced by active pipeline.
- **`dataset/normal/` (27 files, 402.9 MB):** Historical Gen 1 normal captures unreferenced by active pipeline.
- **`dataset/merged_datasets.duckdb` & `impact_assessment.duckdb` (4.33 GB):** Historical Gen 1 databases unreferenced by active pipeline.
- **`dataset/topology/` (3 files):** Historical Gen 1 Modbus/S7 network topologies superseded by `topology/grid_topology.json`.
- **`models/tcn_autoencoder_baseline.pt` & `models/layer2_*.json` (6 files, 0.95 MB):** Historical Gen 1 model weights superseded by active `triple_*` models.
- **`notebooks/` (all 5 notebooks):** Historical exploratory notebooks from Phase 2 and 3 whose findings are fully documented in reports.
- **`experiments/` (all 4 files):** Historical PSO and sensitivity experiments performed on Gen 1 data.
- **`src/data/` (duckdb_loader, evaluation_data, network_data, process_data):** Historical Gen 1 loaders with 0 active pipeline imports.
- **`src/ml/` (preprocessing, layer1_tcn_detector, layer2_physical_relationships, evidence_fusion, etc.):** Historical Gen 1 ML code superseded by `triple_*` modules.
- **`reports/phase2*`, `reports/phase3*`, `reports/layer1_*`, `reports/layer2_*`, `reports/duckdb_*`:** Historical documentation of earlier phases.
- **`scratch/` ad-hoc audit runners (`audit_triple_dataset.py`, `run_full_triple_audit.py`, `detailed_audit.py`, `run_preimplementation_validation.py`, `analyze_attacks.py`, `analyze_physical_relationships.py`, `test_layer2_candidates.py`, `test_qwen_loading.py`):** Completed exploration scripts.

### C. POSSIBLY DELETE
*(Zero loss of intellectual property or reproducibility)*
- **`scratch/check_constant_channels.py` (1.1 KB):** Ephemeral 10-line scratch snippet.
- **`scratch/check_high_zero_cols.py` (0.6 KB):** Ephemeral 8-line scratch snippet.
- **`scratch/inspect_cases.py` (1.2 KB):** Ephemeral 15-line scratch snippet.
- **`scratch/print_block_summary.py` (1.0 KB):** Ephemeral 12-line scratch snippet.
- **`scratch/audit_inventory_helper.py` (4.9 KB):** Ephemeral helper script used during this audit.
- **`.pytest_cache/` (5 files, ~12 KB):** Automatically regenerated on next `pytest` run.
- **`**/__pycache__/` (52 `.pyc` files across 7 directories, ~642 KB):** Automatically regenerated bytecode.
