# Phase 1B: Data Selection and Preparation Planning Report

**Document Version:** 1.1.0 (Scope Refinement Update)  
**Date:** September 28, 2026  
**Status:** Approved for Implementation  
**Target Next Phase:** Phase 2 (Machine Learning & Anomaly Detection)  
**Primary Code Modules:**
- [`src/data/duckdb_loader.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/duckdb_loader.py)
- [`src/data/process_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/process_data.py)
- [`src/data/evaluation_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/evaluation_data.py)
- [`src/data/network_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/network_data.py)

---

## Executive Summary

Phase 1B establishes the data selection, transformation, and preparation architecture for the smart grid cyber-physical anomaly detection system. Following the comprehensive inspection in [`reports/duckdb_inspection.md`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/reports/duckdb_inspection.md), Phase 1B establishes modular, memory-efficient data extraction pipelines from the 4.22 GB `merged_datasets.duckdb` and 2.26 MB `impact_assessment.duckdb` repositories.

### Project Scope Refinement (Primary Physical ML Domains)

> [!IMPORTANT]
> **SCOPE REFINEMENT DECISION:**  
> To ensure high precision, well-conditioned physical dynamics, and targeted scope control, the **primary physical domains for the ML anomaly-detection pipeline** are:
> 1. **Solar / Photovoltaic (PV)**
> 2. **Wind Generation**
>
> **Excluded from Primary ML Modeling:**
> - **Battery Energy Storage System (BESS)**
> - **Microgrid Power Demand**
>
> **Data Layer Preservation:**  
> Battery and Grid Demand telemetry remain **100% available** in the raw DuckDB database (`merged_datasets.duckdb`) and in the extraction modules ([`src/data/process_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/process_data.py)). They are **NOT** deleted or dropped from the data layer; they are simply excluded from the primary ML feature space for downstream model training and evaluation.

### Downstream ML Workflow Architecture

The downstream pipeline architecture across the project phases is structured as follows:

```
Solar/PV + Wind Process Measurements
               ↓
         Preprocessing
               ↓
       LSTM Autoencoder
               ↓
         Anomaly Score
               ↓
         Digital Twin
               ↓
           LLM + RAG
               ↓
Anomaly Explanation / Investigation
```

### Architectural Decisions Summary
1. **Primary Process Data:** Cyber-physical telemetry extracted from `battery_process_data`, `pv_process_data`, `wind_process_data`, and `demand_process_data`. The primary ML feature subspace is focused on **Solar/PV and Wind dynamic measurements**.
2. **Secondary Communication Data:** Statistical flow and traffic features aggregated from `packets` using **metadata only** (sizes, protocols, EtherTypes, ports, IPs, flow hashes). Raw packet payloads (`raw_packet` BLOBs) are strictly bypassed.
3. **Context / Evaluation Data:** `attack_session`, `exec_steps`, `attack_datamod_history`, `logs`, and `judge_results` (from `impact_assessment.duckdb`) are extracted into an **independent evaluation schema**. They are completely quarantined from the feature datasets.
4. **Digital Twin Context:** `network.json`, `modbus.json`, and `s7_connections.json` provide register mappings, physical units, and S7 industrial communication topologies.
5. **Zero Data Leakage:** Anomaly models will be trained **strictly on uncompromised baseline operational data** (`20260225_normal`, >3 days continuous data). No attack indicators, impact flags, or future temporal information will exist in feature matrices.

---

## Architecture & Data Flow

```mermaid
graph TD
    subgraph Raw Repositories (Read-Only)
        MDB[("merged_datasets.duckdb (4.22 GB)")]
        IDB[("impact_assessment.duckdb (2.26 MB)")]
        TOP[("dataset/topology/*.json")]
    end

    subgraph Phase 1B Extraction Modules
        LDR["src/data/duckdb_loader.py<br/>(DuckDBManager: read_only, zero-copy COPY TO)"]
        PRC["src/data/process_data.py<br/>(ProcessDataExtractor: ASOF Join & Time Bucket)"]
        NET["src/data/network_data.py<br/>(NetworkDataExtractor: Metadata-Only Flow Aggregator)"]
        EVL["src/data/evaluation_data.py<br/>(EvaluationDataExtractor: Attack Windows & Judge Impact)"]
    end

    subgraph Data Quarantine Wall (Strict Anti-Leakage Boundary)
        WALL["=================== STRICT ISOLATION WALL ==================="]
    end

    subgraph Phase 2 Datasets
        D_NORM["Clean Normal Baseline Features<br/>(11,992 steps @ 0.53s | 25 signals)<br/>Target: Unsupervised Training & Twin Invariants"]
        D_TEST["Evaluation Run Features<br/>(153,491 steps | 25 signals)<br/>Target: Blind Anomaly Detection Inference"]
        D_COMM["Network Traffic Features (Optional)<br/>(Aggregated 1s/5s windows | Metadata Only)"]
        D_EVAL["Independent Ground-Truth Evaluation Table<br/>(1,723 Attack Windows | 1,009 Physical Judge Labels)<br/>Target: Post-Hoc Metrics (P, R, F1, Latency)"]
    end

    MDB --> LDR
    IDB --> LDR
    TOP --> NET
    LDR --> PRC
    LDR --> NET
    LDR --> EVL

    PRC -->|Normal Run Only| D_NORM
    PRC -->|Adversarial Runs| D_TEST
    NET -->|Windowed Headers| D_COMM
    
    EVL -.->|QUARANTINED LABELS| WALL
    WALL -.-> D_EVAL

    style WALL fill:#f96,stroke:#333,stroke-width:3px,stroke-dasharray: 5 5;
    style D_NORM fill:#bbf,stroke:#333,stroke-width:2px;
    style D_TEST fill:#bbf,stroke:#333,stroke-width:2px;
    style D_EVAL fill:#fbb,stroke:#333,stroke-width:2px;
```

---

## 1. Process Telemetry Extraction

### Purpose & Requirements
Process telemetry reflects the physical state of the microgrid testbed across four distributed generation and storage subsystems:
- **Battery Energy Storage System (BESS):** `battery_process_data` (165,483 rows)
- **Photovoltaic Solar Generation (PV):** `pv_process_data` (165,483 rows)
- **Wind Turbine Generation:** `wind_process_data` (165,483 rows)
- **Electrical Demand / Load:** `demand_process_data` (87,101 rows)

### Extraction Strategy
To avoid loading the 4.22 GB DuckDB database into RAM, telemetry extraction is implemented in [`src/data/process_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/process_data.py) using selective SQL queries. Subsystems are queried only for necessary numerical and boolean telemetry columns.

Two extraction modes are provided:
1. **Subsystem-Level Extraction (`extract_subsystem_telemetry`):** Extracts isolated tables for subsystem-specific models (e.g. battery degradation model or wind aerodynamic model).
2. **System-Wide Aligned Extraction (`extract_aligned_telemetry_asof` / `extract_resampled_telemetry`):** Combines all 4 subsystems into a comprehensive 25-signal feature vector representing total microgrid state.

---

## 2. Timestamp Handling & Temporal Alignment

### Data Properties
All timestamps in the DuckDB databases use DuckDB's `TIMESTAMP WITH TIME ZONE` (microsecond resolution, offset `+05:30`).

### Empirical Jitter & Scan Cycle Offset
Deep analysis revealed a crucial property of the testbed SCADA system:
- `battery_process_data`, `pv_process_data`, and `wind_process_data` share the **exact same row count** (165,483 rows across all 6 runs; exactly 11,992 rows in the baseline run).
- However, their timestamps are not identical: they differ by an average of **0.017 seconds (17 milliseconds)** (maximum difference: 0.044 seconds).
- **Physical Reason:** Industrial PLCs poll sensors and transmit SCADA packets sequentially within a periodic 500ms scan cycle. A strict inner join on `battery.ts = pv.ts` results in 0 matches.
- `demand_process_data` is sampled on a 1.00s period (87,101 rows total; 6,310 rows in baseline), approximately half the frequency of the generation/storage units.

### Alignment Solutions
We implemented two non-destructive alignment strategies:

#### Strategy A: Causal Point-in-Time Alignment (`ASOF JOIN`)
```sql
SELECT b.dataset_id, b.ts, ...
FROM battery_process_data b
ASOF JOIN pv_process_data p 
    ON b.dataset_id = p.dataset_id AND b.ts >= p.ts
ASOF JOIN wind_process_data w 
    ON b.dataset_id = w.dataset_id AND b.ts >= w.ts
ASOF JOIN demand_process_data d 
    ON b.dataset_id = d.dataset_id AND b.ts >= d.ts
WHERE b.dataset_id = :dataset_id
ORDER BY b.ts;
```
- **Preserved Resolution:** Preserves full ~0.53s sampling rate (11,992 rows in normal run).
- **Causality Guarantee:** Enforces $t_{battery} \ge t_{pv, wind, demand}$. Every row pairs the current battery measurement with the *most recently observed* PV, Wind, and Demand telemetry.
- **Zero Lookahead:** Strictly prevents future data from leaking into the present.

#### Strategy B: Uniform Time-Bucket Resampling (`time_bucket`)
```sql
SELECT dataset_id, time_bucket(INTERVAL '1 second', ts) as ts, ...
FROM ...
GROUP BY 1, 2
ORDER BY ts;
```
- **Uniform Grid:** Resamples telemetry onto a fixed 1.0-second or 0.5-second time step for models (such as autoencoders or recurrent networks) that require equidistant time indices.
- **Aggregation:** Averages continuous measurements and computes boolean OR/modal state for control flags.

---

## 3. Dataset & Run Identification

The catalog is formalized in [`src/data/duckdb_loader.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/duckdb_loader.py):

| Dataset ID (UUID) | Run Identifier | Relative Testbed Path | Run Type | Orchestration Agent | Rows (Proc) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `21c851bc-384f-5b81-8747-6dcacdceff35` | `normal` | `2026_AttackPlans/20260225_normal` | **Baseline** | None (Uncompromised) | 11,992 |
| `12f0d001-77a0-5d04-8adf-6d002702478b` | `multi_openai` | `2026_AttackPlans/20260228_multi_openai` | Adversarial | OpenAI GPT-5.3 Codex | 31,084 |
| `022b85dc-6b64-5f70-b46e-886824460c0b` | `multi_sonnet_1` | `2026_AttackPlans/20260301_multi_sonnet` | Adversarial | Anthropic Claude Sonnet 4.6 | 17,624 |
| `cb4f94e7-6470-544e-bc72-9184d54f4ee1` | `multi_google` | `2026_AttackPlans/20260301_multi_google` | Adversarial | Google Gemini 3.1 Pro | 42,027 |
| `202eb955-4715-51b0-817d-ed423e0b55d8` | `multi_minimax` | `2026_AttackPlans/20260302_multi_minimax` | Adversarial | MiniMax M2.5 | 47,707 |
| `62783187-c55c-5a73-a657-6d38f13e2fd4` | `multi_sonnet_2` | `2026_AttackPlans/20260303_multi_sonnet` | Adversarial | Anthropic Claude Sonnet 4.6 | 15,049 |
| **Total** | | | | | **165,483** |

---

## 4. Missing-Value Analysis

An exhaustive query across all 165,483 rows in `battery_process_data`, `pv_process_data`, `wind_process_data` and 87,101 rows in `demand_process_data` produced the following verification:

```
Subsystem Data Quality Audit:
- battery_process_data:  0 nulls across all 9 columns (100.0% complete)
- pv_process_data:       0 nulls across all 10 columns (100.0% complete)
- wind_process_data:     0 nulls across all 11 columns (100.0% complete)
- demand_process_data:   0 nulls across all 3 columns (100.0% complete)
```

**Quality Conclusion:** The primary process telemetry is **100% complete with zero missing values**. No imputation or artificial synthetic value generation is necessary.

---

## 5. Duplicate Detection

Audit of `(dataset_id, ts)` uniqueness:
- `battery_process_data`: **0 duplicates**
- `pv_process_data`: **0 duplicates**
- `wind_process_data`: **0 duplicates**
- `demand_process_data`: **0 duplicates**

**Quality Conclusion:** Every row in every process table possesses a strictly unique `(dataset_id, ts)` key. Primary key integrity is verified.

---

## 6. Sampling-Rate Analysis

| Table | Min Interval | Max Interval | Mean Interval | Std Dev | Effective Frequency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `battery_process_data` | 0.520s | 0.555s | **0.531s** | 0.008s | ~1.88 Hz |
| `pv_process_data` | 0.522s | 0.553s | **0.531s** | 0.007s | ~1.88 Hz |
| `wind_process_data` | 0.520s | 0.553s | **0.531s** | 0.007s | ~1.88 Hz |
| `demand_process_data` | 0.974s | 1.024s | **1.000s** | 0.008s | ~1.00 Hz |

**Analysis:** Sampling jitter is minimal (<10ms standard deviation). The process data reflects high-precision hardware timers typical of real-time industrial PLC loops.

---

## 7. Signal Selection & Physical Units

Cross-referenced with testbed topology configurations (`modbus.json` and `s7_connections.json`), the complete selected feature set contains **25 process signals**:

| Subsystem | Signal Name | Type | Physical Unit | Description / Function |
| :--- | :--- | :--- | :--- | :--- |
| **Battery** | `batt_c_on_off` | Control (Boolean) | Binary (0/1) | BESS main contactor enable / disable command |
| | `batt_c_target_power` | Control (Integer) | Watts (W) | Charging (>0) or discharging (<0) setpoint |
| | `batt_m_current` | Measurement (Float) | Amperes (A) | Battery DC terminal current |
| | `batt_m_voltage` | Measurement (Float) | Volts (V) | Battery DC terminal voltage |
| | `batt_m_temperature` | Measurement (Float) | °C | Battery cell pack temperature |
| | `batt_m_state_of_charge` | Measurement (Float) | % (0 - 100) | State of Charge (SoC) estimated by BMS |
| | `batt_m_actual_charge_power` | Measurement (Integer)| Watts (W) | Real-time actual charging power delivered |
| **Photovoltaic**| `pv_c_on_off` | Control (Boolean) | Binary (0/1) | Solar inverter AC connection switch |
| | `pv_m_temp_air` | Measurement (Float) | °C | Ambient outdoor air temperature |
| | `pv_m_poa_direct` | Measurement (Integer)| W/m² | Plane of Array beam / direct solar irradiance |
| | `pv_m_wind_speed` | Measurement (Float) | m/s | Anemometer wind speed at solar panel plane |
| | `pv_m_poa_diffuse` | Measurement (Integer)| W/m² | Plane of Array diffuse / scattered irradiance |
| | `pv_m_cell_temperature`| Measurement (Float) | °C | Surface temperature of silicon PV cells |
| | `pv_m_inverter_ac_power`| Measurement (Integer)| kW | Alternating Current grid feed-in power |
| | `pv_m_inverter_dc_power`| Measurement (Integer)| kW | Direct Current raw photovoltaic power |
| **Wind Turbine**| `wind_c_blade_rotation` | Control (Integer) | Degrees (°) | Turbine blade pitch angle control command |
| | `wind_c_rotation_speed` | Control (Integer) | RPM | Rotor rotational speed setpoint |
| | `wind_m_power` | Measurement (Integer)| Watts (W) | Generator electrical active power output |
| | `wind_m_height` | Measurement (Integer)| Meters (m) | Nacelle / hub height sensor |
| | `wind_m_pressure` | Measurement (Float) | hPa (mbar) | Barometric atmospheric air pressure |
| | `wind_m_wind_speed_a` | Measurement (Float) | m/s | Primary ultrasonic wind speed sensor |
| | `wind_m_wind_speed_b` | Measurement (Float) | m/s | Secondary mechanical cup anemometer |
| | `wind_m_temperature_a` | Measurement (Float) | °C | Generator nacelle interior temperature |
| | `wind_m_temperature_b` | Measurement (Float) | °C | Outdoor ambient nacelle temperature |
| **Grid Demand** | `grid_m_demand` | Measurement (Float) | Watts (W) | Electrical power consumed by microgrid loads |

### Primary Physical Domains for the ML Pipeline

Following the project scope refinement decision, the downstream ML anomaly-detection pipeline is focused specifically on the following two primary physical domains:

1. **Solar / Photovoltaic (PV) Domain:**
   - Retains the relevant dynamic physical measurements identified in Phase 2A:
     - `pv_m_temp_air` (Ambient air temperature)
     - `pv_m_poa_direct` (Direct plane-of-array irradiance)
     - `pv_m_wind_speed` (Local wind speed at PV plane)
     - `pv_m_poa_diffuse` (Diffuse plane-of-array irradiance)
     - `pv_m_cell_temperature` (Surface temperature of silicon cells)
     - `pv_m_inverter_ac_power` (AC active power generation)
     - `pv_m_inverter_dc_power` (DC generated power)
     - `pv_c_on_off` (Inverter contactor control state)

2. **Wind Generation Domain:**
   - Retains the relevant dynamic physical measurements identified in Phase 2A:
     - `wind_m_power` (Turbine active power output)
     - `wind_m_pressure` (Barometric atmospheric pressure)
     - `wind_m_wind_speed_a` (Primary ultrasonic wind speed)
     - `wind_m_wind_speed_b` (Secondary anemometer wind speed)
     - `wind_m_temperature_a` (Nacelle internal temperature)
     - `wind_m_temperature_b` (Ambient housing temperature)
   - *Constant wind signals* (`wind_m_height` fixed at 116m, `wind_c_blade_rotation` fixed at 0°, `wind_c_rotation_speed` fixed at 0 RPM) remain subject to Phase 2B evaluation/rules.

3. **Excluded from Primary ML Modeling:**
   - **Battery Energy Storage System (BESS):** `batt_c_*` and `batt_m_*` signals.
   - **Microgrid Load Demand:** `grid_m_demand`.
   - *Data Layer Preservation:* Both Battery and Grid Demand data remain fully available and intact in the raw DuckDB database and in the data extraction module ([`src/data/process_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/process_data.py)). They are **NOT** deleted from the codebase or data layer; they are excluded solely from the primary ML feature space for scope control.

> [!NOTE]
> **Phase 2B Boundary:** This scope update does **NOT** make the final Phase 2B feature-selection decision yet. Phase 2B will separately evaluate exact feature combinations, scaling, sequence length, and chronological train/validation split.

---

## 8. Separation of Control (`C_*`) and Measurement (`M_*`) Signals

A core engineering principle of this project is the **strict mathematical distinction between Control Signals and Measurement Signals**:

```
      +-------------------------------------------------------------+
      |                   EXOGENOUS INPUTS                          |
      |   Environmental: temp_air, poa_direct, wind_speed, demand   |
      |   Control Commands (C_*): target_power, blade_rotation      |
      +------------------------------+------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |         PHYSICAL PROCESS             |
                  |  Battery BMS, Inverters, Turbines    |
                  +------------------+-------------------+
                                     |
                                     v
      +-------------------------------------------------------------+
      |                   SYSTEM MEASUREMENTS (M_*)                 |
      |   State Responses: voltage, current, soc, inverter_power... |
      +-------------------------------------------------------------+
```

### Why this separation is vital:
1. **Physical Causality:** In control theory, commands $C(t)$ and environmental forcings $E(t)$ act as *exogenous inputs*. Measurements $M(t)$ are *system state responses* governed by physical conservation laws:
   $$\dot{x}(t) = f(x(t), C(t), E(t)), \quad M(t) = g(x(t), C(t))$$
2. **False Data Injection (FDI) Signatures:** Attackers may tamper with control commands (actuator spoofing) OR tamper with sensor measurements (telemetry falsification).
   - In actuator attacks: $C(t)$ changes unexpectedly, driving $M(t)$ away from normal operating bounds.
   - In sensor falsification: $M(t)$ reports an impossible value that contradicts physical invariants (e.g., $P_{actual} 
e V \cdot I$, or solar power generated at zero irradiance).
3. **Digital Twin Architecture:** The Physics-Informed Digital Twin takes $C(t)$ and $E(t)$ as inputs to forecast expected $\hat{M}(t)$. The anomaly residual is defined as:
   $$r(t) = M(t) - \hat{M}(t)$$
   Mixing control signals with measurements into an undifferentiated feature bag would destroy this causal architecture.

---

## 9. Normal-Operation Data Identification

- **Dataset Identifier:** `21c851bc-384f-5b81-8747-6dcacdceff35` (`2026_AttackPlans/20260225_normal`).
- **Time Range:** February 25, 2026, 22:02:01 IST to March 1, 2026, 03:20:08 IST.
- **Duration:** Over 3 full continuous days (77.3 hours).
- **Row Count:** Exactly **11,992 ASOF-aligned steps** at ~0.53s sampling rate.
- **Integrity Certification:**
  - `attack_session`: 0 sessions recorded during this period.
  - `exec_steps`: 0 execution steps recorded.
  - `attack_datamod_history`: 0 packet modifications.
  - `logs`: Only routine operational value-change alerts (*"Wertänderung"*).

**Usage Contract:** This dataset is designated **exclusively for Phase 2 training**. It will be used to fit baseline distributions, calibrate Digital Twin differential equations, and train autoencoders/estimators without risk of adversarial contamination.

---

## 10. Attack-Window Metadata Preparation

The attack orchestration metadata from `merged_datasets.duckdb` is parsed into discrete, non-overlapping or concurrent attack execution intervals:
- **Campaign Sessions:** 436 attack sessions initiated by autonomous LLM agents (Claude Sonnet 4.6, GPT-5.3 Codex, MiniMax M2.5, Gemini 3.1 Pro).
- **Execution Intervals:** 1,723 paired `(start_ts, stop_ts)` step windows extracted from `exec_steps`.
- **Packet Injections:** 1,555,263 modification records from `attack_datamod_history`.

Implemented in [`src/data/evaluation_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/evaluation_data.py):
```python
gt_intervals = eval_extractor.get_ground_truth_attack_intervals(dataset_id)
```
Each interval records:
- `attack_session_id`, `step_idx`, `worker_id`
- `attacker_model` (e.g. `openai/gpt-5.3-codex`)
- `target_state` (e.g. `Battery Storage Mismanagement`)
- `start_ts` and `stop_ts`
- `actual_duration_sec`
- `target_subsystem_table` and `target_signal`
- `injected_delta`

---

## 11. Communication Metadata Preparation

Adhering strictly to project constraints, **raw packet payloads (`raw_packet` BLOBs) are never loaded or analyzed**. 

Instead, [`src/data/network_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/network_data.py) provides windowed aggregation of packet headers (Layer 2, 3, 4):
- **Aggregation Windows:** 1.0 second, 5.0 seconds, or 10.0 seconds.
- **Features Extracted:**
  - `total_packet_count`, `total_byte_count`, `avg_packet_size`, `max_packet_size`
  - `tcp_packet_count`, `udp_packet_count`, `icmp_packet_count`
  - `profinet_packet_count` (EtherType `0x8892` / 34962)
  - `s7_iso_tcp_packet_count` (TCP port 102)
  - `distinct_flow_count` (Community ID count)
  - `distinct_src_ip_count`, `distinct_dst_ip_count`
  - `attacker_ip_packet_count` (Traffic originating from or destined to attacker worker IPs `172.22.66.81`–`88`)

**Integration with Topology:**
Traffic patterns align directly with `dataset/topology/network.json` and `s7_connections.json`, allowing the system to monitor expected communication bandwidth between `MasterPCS` (172.22.66.21) and the subordinate PLCs.

---

## 12. Evaluation & Ground-Truth Preparation

### Purpose & Architecture
Ground truth serves one purpose: **evaluating the performance of anomaly detection models post-hoc**. 

### LLM Judge Consequence Validation
From `impact_assessment.duckdb`, the table `judge_results` (1,009 evaluated steps) provides empirical ground truth on whether an attack actually disturbed physical grid dynamics:
- `delta_observed_during_active = True`: The attack succeeded in shifting the physical sensor/actuator reading (36.0% of steps).
- `delta_observed_during_active = False`: The attack packet injection was ineffective, filtered, or absorbed by grid inertia (58.7% of steps).
- `after_similar_to_before`: Demonstrates whether the signal returned to baseline after attack termination.

### Point-in-Time Evaluation Generator
[`src/data/evaluation_data.py`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/evaluation_data.py) provides `generate_evaluation_labels(telemetry_timestamps, dataset_id)`:
- Projects interval-based ground truth onto any query timestamp array.
- Generates boolean flags: `is_attack_interval`, `is_physically_impactful`, `active_attack_count`, and `targeted_signals`.
- **Zero Contamination:** This function is strictly decoupled from the feature loader.

---

## 13. Data Leakage Prevention Architecture

Data leakage in cyber-physical anomaly detection produces artificially inflated accuracy metrics that collapse when deployed in real-world infrastructure. Phase 1B implements four structural defenses:

| Leakage Vector | Risk | Architectural Defense in Phase 1B | Verification |
| :--- | :--- | :--- | :--- |
| **Label Leakage** | Attack indicators present in feature vectors | Strict physical isolation. Features contain ONLY physical signals (`C_*`, `M_*`). Attack IDs, model names, and labels are stored in separate evaluation tables. | `assert 'label' not in feature_df.columns` |
| **Temporal Lookahead** | Downstream data leaking into upstream timestamps | Causal `ASOF JOIN` enforcing $t_{battery} \ge t_{other}$. Rolling aggregations use strictly backward-looking windows. | No future offsets used in DuckDB queries. |
| **Train/Test Contamination** | Mixing attack run data into baseline training | Model training uses ONLY dataset `21c851bc-384f-5b81-8747-6dcacdceff35`. Adversarial runs are quarantined for inference only. | Filtered by `dataset_id = NORMAL_DATASET_ID`. |
| **Preprocessor Snooping** | Normalizing features using test-set min/max/mean | All scalers (StandardScaler, MinMaxScaler) in Phase 2 must be `fit()` strictly on the Normal baseline, then `transform()` on test runs. | Documented in Phase 2 contract. |

---

## Detailed Transformation Audit

For every data transformation implemented in Phase 1B, the four required dimensions are documented below:

### Transformation 1: Causal Point-in-Time Subsystem Join (`ASOF JOIN`)
- **Why it is needed:** Industrial PLCs poll sensors asynchronously within each 500ms cycle (~17ms offset). Telemetry must be combined into a unified system state vector.
- **What data it affects:** `battery_process_data`, `pv_process_data`, `wind_process_data`, `demand_process_data`.
- **What information is preserved:** All 25 original physical signals, microsecond timestamps, numerical values, and exact physical units.
- **Could it introduce data leakage?** **NO.** The condition `b.ts >= other.ts` strictly prevents lookahead leakage. It matches only contemporary or prior observations.

### Transformation 2: Uniform Grid Resampling (`time_bucket`)
- **Why it is needed:** Neural architectures (autoencoders, LSTM, Transformer) require discrete, equidistant time steps ($\Delta t = 1.0\text{ s}$ or $0.5\text{ s}$).
- **What data it affects:** Timestamps and numerical measurements across all 4 process tables.
- **What information is preserved:** Average physical power, voltage, temperature, and modal contactor states across each bucket.
- **Could it introduce data leakage?** **NO.** Aggregation occurs strictly within non-overlapping past intervals.

### Transformation 3: Packet Header Metadata Feature Aggregation
- **Why it is needed:** The raw `packets` table contains 24M records and raw byte BLOBs (4+ GB). Ingesting raw packets into memory is inefficient and unnecessary.
- **What data it affects:** `packets` header columns (`size`, `transport`, `ethertype`, `ports`, `IPs`, `community_id`).
- **What information is preserved:** Packet counts, byte throughput, transport ratios, S7/Profinet frequencies, flow diversity, and attacker IP activity.
- **Could it introduce data leakage?** **NO.** Uses only passive header metadata binned strictly within timestamp windows.

### Transformation 4: Attack Interval Projection to Evaluation Labels
- **Why it is needed:** Models output continuous anomaly scores $s(t)$. Evaluation requires ground-truth labels at each score timestamp to compute AUROC, Precision, Recall, and F1.
- **What data it affects:** Evaluates `exec_steps`, `attack_session`, and `judge_results`.
- **What information is preserved:** Interval timing, target state, attack duration, and physical impact flags.
- **Could it introduce data leakage?** **NO.** Labels are generated as a standalone post-hoc evaluation array and are NEVER passed to feature extractors.

---

## Datasets Passed to Phase 2 ML

Phase 1B delivers four precisely defined dataset contracts for Phase 2 Machine Learning:

### Dataset 1: Baseline Normal Training Telemetry (`df_train_normal`)
- **Source:** [`ProcessDataExtractor.extract_normal_training_set()`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/process_data.py)
- **Dataset ID:** `21c851bc-384f-5b81-8747-6dcacdceff35` (`20260225_normal`)
- **Extracted Data Dimensions:** 11,992 rows × 27 columns (2 metadata + 25 physical signals across all 4 subsystems, preserved in data layer)
- **Primary ML Modeling Subspace:** Focused on **Solar/PV + Wind dynamic physical measurements** (Battery and Grid Demand are excluded from the primary ML training feature space for scope control).
- **Time Coverage:** Feb 25, 2026 22:02 to March 1, 2026 03:20 (77.3 hours continuous clean baseline)
- **Usage:** Fit Digital Twin physical equations, train unsupervised Autoencoders / One-Class SVM / Isolation Forest, fit normalization scalers.
- **Full Extracted Schema:**
  - `dataset_id` (UUID), `ts` (TIMESTAMPTZ)
  - 5 Controls: `batt_c_on_off`, `batt_c_target_power`, `pv_c_on_off`, `wind_c_blade_rotation`, `wind_c_rotation_speed`
  - 20 Measurements: `batt_m_*` (5), `pv_m_*` (7), `wind_m_*` (7), `grid_m_demand` (1)

### Dataset 2: Adversarial Evaluation Telemetry (`df_test_adversarial`)
- **Source:** [`ProcessDataExtractor.extract_aligned_telemetry_asof(dataset_id)`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/process_data.py)
- **Dataset IDs:** 5 adversarial runs (`multi_openai`, `multi_sonnet_1`, `multi_google`, `multi_minimax`, `multi_sonnet_2`)
- **Extracted Data Dimensions:** 153,491 total rows × 27 columns (all 4 subsystems preserved in data layer)
- **Primary ML Modeling Subspace:** Evaluates model inference across the corresponding **Solar/PV + Wind physical measurements**.
- **Usage:** Ingested blindly by Phase 2 anomaly detectors to generate continuous anomaly scores $\hat{y}(t)$ without access to ground truth.

### Dataset 3: Network Traffic Flow Features (`df_network_features`) [Optional Multi-Modal]
- **Source:** [`NetworkDataExtractor.extract_traffic_features_windowed(dataset_id)`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/network_data.py)
- **Dimensions:** Binned at 1.0s or 5.0s windows × 15 traffic features
- **Usage:** Secondary multi-modal intrusion correlation (detecting ARP spoofing spikes, S7 scan floods, and unauthorized attacker IP communication).

### Dataset 4: Independent Ground-Truth Evaluation Table (`df_ground_truth`)
- **Source:** [`EvaluationDataExtractor.get_ground_truth_attack_intervals()`](file:///c:/Users/LENOVO/Downloads/IMDAI%20project/src/data/evaluation_data.py)
- **Dimensions:** 1,723 attack step intervals × 15 metadata & impact columns
- **Usage:** Evaluated post-hoc against model anomaly detections to compute:
  1. **Overall Detection Metric:** Precision, Recall, F1 against all attack attempts.
  2. **Physical Impact Metric:** Recall and Time-to-Detect against *physically impactful* attacks (`delta_observed_during_active = True`).
  3. **Stealth Metric:** Detection performance against *ineffective or subtle* attacks (`delta_observed_during_active = False`).
