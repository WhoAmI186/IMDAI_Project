# Qwen3-8B Crash Diagnostic

**Document ID:** `REPORT-QWEN-CRASH-DIAGNOSTIC-001`  
**Date:** 2026-10-01  
**Project:** Cyber-Physical Smart Grid Anomaly Detection and Investigation  
**Status:** Post-Crash Diagnostic Completed (Read-Only)  

---

## 1. Environment

- **Python Version:** 3.12.10 (`tags/v3.12.10:0cc8128`, 64-bit AMD64)
- **PyTorch Version:** 2.14.0+cu126
- **CUDA Available to PyTorch:** True (CUDA 12.6, 1 device detected)
- **GPU Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (Driver Version: 595.97, Driver Model: WDDM)
- **Total VRAM:** 6,141 MiB (~6.00 GB)
- **Current VRAM In Use:** 866 MiB (OS Desktop, Antigravity IDE, Edge/Chrome, Background apps)
- **Current PyTorch VRAM Allocated:** 0 bytes (No model currently loaded in memory)
- **Current PyTorch VRAM Reserved:** 0 bytes
- **Core ML/LLM Packages:**
  - `transformers`: 5.18.0
  - `accelerate`: 1.15.0
  - `bitsandbytes`: 0.50.2
  - `requests`: 2.34.2
- **External Inference Servers:**
  - `ollama`: Not found in PATH; port 11434 connection refused (not running)
  - `vllm`: Not found in PATH / not installed
  - `lmstudio`: Not found in PATH / not configured

---

## 2. LLM Configuration

Inspected files:
- `src/llm/config.py`
- `src/llm/llm_client.py`
- `src/llm/investigator.py`
- `src/llm/prompt_builder.py`
- `src/llm/schemas.py`

### Parameter Breakdown:
- **Configured Model:** `Qwen/Qwen3-8B` (default in `LLMConfig.model_name`, overridable via `LLM_MODEL`)
- **Configured Backend:** `auto` (default in `LLMConfig.backend`, resolves to `TransformersBackend` when `transformers` is installed and no local server is listening on port 11434)
- **Device:** `cuda`
- **Quantization Setting:** `4bit`
- **Dtype:** `torch.float16` when CUDA is available, fallback to `torch.float32`
- **Max Input Tokens:** Uncapped at tokenizer level; bounded by compact upstream Digital Twin context (~200–350 tokens)
- **Max Output Tokens:** 1024 tokens (`max_tokens = 1024`)
- **Temperature:** 0.1 (`do_sample = True` if temperature > 0)
- **Generation Settings:** `max_new_tokens=1024`, `do_sample=True`, `temperature=0.1`, chat template applied via `tokenizer.apply_chat_template`
- **Loading Mechanism:**
  - At `__init__`: No loading (`model = None`, `tokenizer = None`)
  - Lazy loading: **YES**, on first call to `generate()` if `model is None`
  - Explicit loading: Supported via `LLMClient.load_model()` / `TransformersBackend.load()`
- **CUDA Device Map:** `"device_map": "auto"`
- **CPU Offloading:** Implicitly allowed by `device_map="auto"`, but no explicit disk offload folder configured
- **bitsandbytes Usage:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` is instantiated if `bitsandbytes` is imported; falls back to raw `float16` if import fails.

---

## 3. Backend Trace of Previous Attempt

The previous execution attempted to run:
`scratch/test_qwen_loading.py`

Trace:
- **Backend:** `TransformersBackend` (direct PyTorch / HuggingFace `AutoModelForCausalLM` pipeline)
- **Model Target:** `Qwen/Qwen3-8B`
- **Device:** `cuda` (device_map="auto")
- **Quantization:** `4bit` (NF4 via `BitsAndBytesConfig`, double quant enabled)
- **Compute Dtype:** `torch.float16`

---

## 4. Model Cache State

Inspected Hugging Face Hub Cache at:  
`C:\Users\LENOVO\.cache\huggingface\hub\models--Qwen--Qwen3-8B`

- **Model Folder Exists:** YES
- **Approximate Total Size:** **15.18 MB**
- **Files Present:**
  - `tokenizer.json` (11.42 MB)
  - `vocab.json` (2.78 MB)
  - `merges.txt` (1.67 MB)
  - `model.safetensors.index.json` (32.88 KB)
  - `tokenizer_config.json` (9.73 KB)
  - `config.json` (728 B)
  - `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff.a39a4982.incomplete` (0 bytes)
- **Weight Download Complete:** **NO (INCOMPLETE)**
- **Partial/Incomplete Files:** 1 incomplete file of 0 bytes in the `blobs` directory. None of the ~16 GB unquantized safetensor weight shards were downloaded.

---

## 5. Crash Evidence & Forensics

### 5.1 Terminal & Subagent Execution Logs
From `C:\Users\LENOVO\.gemini\antigravity-ide\brain\88157e01-08c0-47e6-8898-d8a820a1bbcf\.system_generated\tasks\task-1634.log`:
```
W1001 10:11:47.657000 31932 site-packages\torch\utils\flop_counter.py:113] triton not found...
================================================================================
QWEN3-8B 4-BIT LOADING SMOKE TEST
================================================================================
CUDA Available: True
Device Name: NVIDIA GeForce RTX 4050 Laptop GPU
Total VRAM: 6.00 GB

Attempting to load Qwen/Qwen3-8B with 4-bit BitsAndBytes quantization...
1. Loading Tokenizer...
   Tokenizer loaded successfully. Vocab size: 151669

2. Configuring 4-bit NF4 Quantization for RTX 4050 (6GB VRAM)...

3. Loading Model with device_map='auto'...
Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]
```
The execution log ceased abruptly at `Fetching 5 files: 0%` at 10:11:50.

### 5.2 Operating System & Kernel Event Logs
- **System Boot Timestamp:** `24-09-2026 13:55:06`
- **Kernel Reboot / Blue Screen (BugCheck):** **NONE**. The OS did not experience a kernel panic, blue screen of death (BSOD), or dirty power cycle (Event ID 41 / 1001 are absent).
- **GPU Driver TDR / nvlddmkm Resets:** **NONE**. Queries for `nvlddmkm` and `Display` driver resets returned 0 events.
- **Power Management Events:**
  - `10:25:38`: `Microsoft-Windows-Kernel-Power` Event 506: *The system is entering Modern Standby. Reason: Lid.*
  - `10:27:21`: `Microsoft-Windows-Kernel-Power` Event 507: *The system is exiting Modern Standby. Reason: Lid.*
- **Process Status:** PID 31932 (`python.exe` running `test_qwen_loading.py`) is no longer running. No post-mortem crash dump was generated in Windows Error Reporting (`WER`) or `%LOCALAPPDATA%\CrashDumps`.

---

## 6. Likely Cause Classification

| Potential Cause | Classification | Technical Evidence & Analysis |
|:---|:---:|:---|
| **A. Unquantized Weight Download Saturation / Network Stall** | **STRONG INDICATION** | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", quantization_config=bnb_config)` requires downloading the full ~16 GB unquantized FP16 weights before quantizing them on-the-fly. The download halted at `0% (0/5 files)` with only 15 MB written, causing a complete network/process stall. |
| **B. Python Process Abort / Window Hang** | **STRONG INDICATION** | The process stopped logging without raising a Python exception or traceback, consistent with a process termination, memory ceiling freeze, or user intervention after an unresponsive wait. |
| **C. CUDA Out-of-Memory (OOM)** | **REFUTED** | The weights were never loaded into host RAM or VRAM; only the tokenizer (15 MB) was loaded before the stall. No CUDA OOM exception was thrown. |
| **D. Windows TDR / Display Driver Crash** | **REFUTED** | Zero TDR events or `nvlddmkm` timeout events recorded in Windows System Event Logs. |
| **E. Kernel Crash / Thermal Shutdown** | **REFUTED** | Last boot time is 2026-09-24 13:55:06. The machine remained running continuously. |

---

## 7. Project Integrity Verification

All ML, Digital Twin, and evaluation artifacts were audited against recorded baselines:

### 7.1 Frozen ML Checkpoint Cryptographic Hashes (SHA-256)
| Artifact | Recorded Hash (`reports/llm_integration.md`) | Current Computed Hash | Status |
|:---|:---|:---|:---:|
| `models/triple_tcn_autoencoder.pt` | `b1ad0ee1d4706c9dfa5d82b3eb2be003d27406f567820780f797eba8a5f60ef6` | `b1ad0ee1d4706c9dfa5d82b3eb2be003d27406f567820780f797eba8a5f60ef6` | **VERIFIED (FROZEN)** |
| `models/triple_scaler.json` | `5272798f6a1419776e4710961c20d0aca09db41bacf8892c0dab038df0afa16e` | `5272798f6a1419776e4710961c20d0aca09db41bacf8892c0dab038df0afa16e` | **VERIFIED (FROZEN)** |
| `models/triple_layer1_metadata.json` | `b76e011aa8242cbdd15dfdf0c2b459482f635d065590a650e4dd0aedd93c05bf` | `b76e011aa8242cbdd15dfdf0c2b459482f635d065590a650e4dd0aedd93c05bf` | **VERIFIED (FROZEN)** |
| `models/triple_layer2_metadata.json` | `77c9bec9ea9980e681ca60168c549d13239b15758fcab097e4fc932543ab1107` | `77c9bec9ea9980e681ca60168c549d13239b15758fcab097e4fc932543ab1107` | **VERIFIED (FROZEN)** |
| `models/triple_layer2_bus1_voltage_xgb.json` | `6317358b6e1609b822a198b38397d4713623ed6ba3100e7ef1b8b6e15f199652` | `6317358b6e1609b822a198b38397d4713623ed6ba3100e7ef1b8b6e15f199652` | **VERIFIED (FROZEN)** |
| `models/triple_layer2_bus2_voltage_xgb.json` | `ccb5f33f9e8f5af3b8f39553d6f6689c195369a071ca40d05d2fe1b49240d9e1` | `ccb5f33f9e8f5af3b8f39553d6f6689c195369a071ca40d05d2fe1b49240d9e1` | **VERIFIED (FROZEN)** |
| `models/triple_layer2_line1_current_xgb.json` | `1ba8caccaaad22369374559d9f21fe8929be8d3035198c8599bb8f05b389fcb6` | `1ba8caccaaad22369374559d9f21fe8929be8d3035198c8599bb8f05b389fcb6` | **VERIFIED (FROZEN)** |
| `models/triple_layer2_line2_current_xgb.json` | `623c55929466ea26ee9a0b9e01ebdc349a7c4b2f04cddca0ebb756515bfd0ef0` | `623c55929466ea26ee9a0b9e01ebdc349a7c4b2f04cddca0ebb756515bfd0ef0` | **VERIFIED (FROZEN)** |

### 7.2 Component Modification Status
- **ML Pipeline Source Files:** **NO CHANGE** (Unmodified since 2026-10-01 04:30)
- **Digital Twin Source Files:** **NO CHANGE** (Unmodified since 2026-10-01 06:17)
- **LLM Module Files:** **NO CHANGE** (Unmodified since 2026-10-01 09:59)
- **Dataset Files:** **NO CHANGE**
- **Existing Reports:** Neither `reports/llm_integration.md` nor `reports/llm_evaluation_summary.json` was altered or overwritten.

---

## 8. Safety Recommendations Before Next Inference Attempt

1. **Do NOT invoke on-the-fly quantization of raw 16-bit 8B models via Transformers:**
   Using `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", quantization_config=bnb_config)` forces `huggingface_hub` to download 16 GB of raw FP16/BF16 shards. This requires substantial network bandwidth, ~20+ GB host RAM, and can cause system freezing or storage exhaustion on laptops.
2. **Use Pre-Quantized Models or Specialized Quantized Formats:**
   - **Option A (Recommended for Local Zero-Crash Execution):** Use a pre-quantized GGUF or AWQ model via Ollama (e.g. `ollama run qwen2.5:7b` or `qwen2.5:3b`), where the total download size is only 3.5–4.5 GB, loads in seconds, and natively runs within 4 GB of VRAM without Python memory bloat.
   - **Option B (HuggingFace AWQ/GPTQ):** Use a pre-quantized 4-bit repository (e.g. `Qwen/Qwen2.5-7B-Instruct-AWQ` or `Qwen/Qwen2.5-3B-Instruct`), eliminating the need to download 16 GB of unquantized float16 weights.
   - **Option C (Default Offline Verification):** Keep `LLM_BACKEND=deterministic_expert` for all automated pipelines, testing, and continuous integration.
