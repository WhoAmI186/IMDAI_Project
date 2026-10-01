# IMDAI: Cyber-Physical Smart Grid Anomaly Detection & Digital Twin Framework

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/pytest-36%2F36%20passing-brightgreen.svg)]()
[![Model Precision](https://img.shields.io/badge/Attack%20F1-0.9705-success.svg)]()
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9979-success.svg)]()
[![Episode Coverage](https://img.shields.io/badge/Attack%20Coverage-100%25-brightgreen.svg)]()

An end-to-end, multi-tiered cyber-physical anomaly detection, state validation, and automated forensic investigation pipeline for modern power transmission grids. Built and benchmarked on high-frequency synchrophasor PMU telemetry from the **Mississippi State University / Oak Ridge National Laboratory (MSU/ORNL)** benchmark dataset.

---

## 🏗️ System Architecture

The pipeline enforces a strict hierarchical flow from high-frequency sensor telemetry to deterministic physical validation and sandboxed LLM investigation:

```
Smart Grid PMU Data (30 Hz Stream / 129 Channels)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
      Layer 1               Layer 2
  Causal TCN-AE        Physical Regressors
(Temporal Dynamics)     (XGBoost Residuals)
         │                     │
         └──────────┬──────────┘
                    ▼
             Evidence Fusion
         (Calibrated P99 Threshold)
                    ▼
           Power System Digital Twin
    (State Reconstruction & Physical Laws)
                    ▼
       Sanitized Evidence Extraction
                    ▼
          LLM Investigation Layer
     (Qwen3-8B / Structured Reasoning)
                    ▼
    Typed JSON Forensic & Triage Report
```

---

## ⚡ Key Features & Pillars

### 1. Multi-Tiered Anomaly Detection
- **Layer 1: Causal TCN-Autoencoder (`src/ml/triple_tcn_autoencoder.py`)**  
  Captures complex temporal correlations across 2.0-second sliding windows ($L = 60$ steps at 30 Hz). Built with causal, dilated 1D convolutions ensuring **zero future lookahead bias**.
- **Layer 2: Physical Relationship Regressors (`src/ml/triple_layer2_regressors.py`)**  
  Supervised gradient-boosted trees (XGBoost) modeling physical power transmission invariants across bus voltages and line currents. Flags discrepancies where telemetry violates physical laws.
- **Evidence Fusion Engine (`src/ml/triple_evidence_fusion.py`)**  
  Unifies reconstruction loss and regression residuals using statistically calibrated thresholds (P95/P99) derived purely from normal operational baselines.

### 2. Deterministic Power System Digital Twin (`src/digital_twin/`)
- Reconstructs topological and electrical states across an **IEEE 4-bus, 2-generator, 2-line transmission network** defined in `topology/grid_topology.json`.
- Enforces physical laws:
  - **Equipotential checks** across closed breakers and transmission lines.
  - **Kirchhoff's Current Law (KCL)** balance at substation nodes.
  - **Breaker-relay state consistency** (detecting tripped lines vs phantom breaker signals).
- Crucially differentiates **legitimate natural physical faults** (sags, lightning surges) from **adversarial cyber tampering** (false data injection, command injection).

### 3. Grounded LLM Forensic Investigation (`src/llm/`)
- Translates compact, structured Digital Twin state vectors and violation logs into actionable operator briefings.
- **Strict Guardrails**:
  - **Downstream Isolation**: Only receives sanitized, pre-filtered engineering evidence. Never ingests raw sensor noise or ground-truth dataset labels.
  - **Factual Grounding**: Strictly separates factual observations (`observations`) from speculative reasoning (`hypotheses`).
  - **No Autonomous Authority**: Proposes investigative hypotheses with confidence levels, leaving final operational control to human grid operators.
- **Flexible Backend**: Supports `OpenAICompatibleBackend` (optimized for local GGUF models via `llama.cpp` / vLLM / Ollama), `TransformersBackend`, and a fallback `DeterministicExpertBackend`.

---

## 📊 Benchmark Results

Evaluated on strictly held-out test scenarios (`data13.csv`, `data14.csv`, `data15.csv`) with zero contamination in training or threshold calibration:

| Metric | Layer 1 (TCN-AE) | Layer 2 (XGBoost) | Evidence Fusion (P99) |
|---|:---:|:---:|:---:|
| **Attack $F_1$ Score** | **0.9882** | 0.8996 | **0.9705** |
| **Attack PR-AUC** | **0.9937** | 0.9906 | **0.9979** |
| **Attack Recall** | 98.26% | 82.17% | **95.02%** |
| **False Positive Rate (FPR)** | 10.99% | 9.21% | 14.70% |
| **Attack Episode Coverage** | 100% (6/6) | 100% (6/6) | **100% (6/6)** |
| **Mean Detection Latency** | 0.033 s (1 step) | 0.033 s (1 step) | **0.033 s (1 step)** |

---

## 📂 Repository Structure

```
IMDAI_Project/
├── dataset/
│   └── triple/                  # 15 scenario CSVs (data1.csv - data15.csv)
├── models/                      # Trained model checkpoints & metadata
│   ├── triple_tcn_autoencoder.pt
│   ├── triple_scaler.json
│   ├── triple_layer1_metadata.json
│   ├── triple_layer2_*_xgb.json
│   └── triple_layer2_metadata.json
├── reports/                     # Comprehensive technical & experimental reports
├── scripts/                     # Reproducibility & evaluation scripts
│   ├── train_triple_layer1.py
│   ├── train_triple_layer2.py
│   ├── evaluate_triple_pipeline.py
│   ├── evaluate_digital_twin_post_implementation.py
│   ├── evaluate_llm_integration.py
│   └── run_triple_robustness_audit.py
├── src/                         # Core Python package
│   ├── data/                    # Synchrophasor data loading & batching
│   ├── digital_twin/            # State estimation, topology & physics checkers
│   ├── llm/                     # Structured prompt builder, client & investigator
│   └── ml/                      # TCN-AE, XGBoost regressors, evidence fusion
├── tests/                       # Unit and integration test suite
│   ├── test_digital_twin.py
│   ├── test_llm_integration.py
│   ├── test_physical_checks.py
│   └── test_topology.py
├── topology/                    # Power grid network topology definitions
│   └── grid_topology.json
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/WhoAmI186/IMDAI_Project.git
cd IMDAI_Project

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

*(Ensure PyTorch, XGBoost, and Scikit-Learn are installed with your appropriate CUDA or CPU backend).*

### 2. Run Test Suite

Verify system integrity and all component interfaces:

```bash
pytest
```
*Expected: 36 passed tests across topology, physical checks, digital twin, and LLM integration.*

### 3. Evaluate the Complete Pipeline

Run the held-out benchmark evaluation on `data13.csv` - `data15.csv`:

```bash
python scripts/evaluate_triple_pipeline.py
```

### 4. Evaluate Digital Twin Validation & Triage

Validate the physical state reconstructor on test episodes:

```bash
python scripts/evaluate_digital_twin_post_implementation.py
```

### 5. Running LLM Investigation

By default, the LLM layer runs in offline mode using the deterministic rule-based expert backend.

To connect a local LLM (e.g., **Qwen3-8B** quantized to GGUF format via `llama.cpp`):

1. Launch `llama-server`:
   ```bash
   llama-server -m models/qwen3-8b-instruct-q4_k_m.gguf -c 4096 --port 8080 -ngl 28
   ```
2. Run the investigation script with the OpenAI-compatible backend:
   ```bash
   python scripts/evaluate_llm_integration.py --backend openai --api-base http://localhost:8080/v1
   ```

---

## 🛡️ License & Acknowledgements

- **Benchmark Data:** Created by Mississippi State University (MSU) and Oak Ridge National Laboratory (ORNL).
- **Project Reference:** Developed as part of the Cyber-Physical Smart Grid AI Research Initiative.