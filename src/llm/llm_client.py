"""LLM Client and Replaceable Inference Backend Interface.

Supports:
1. TransformersBackend: Local HuggingFace Qwen models (RTX 4050 6GB quantized / half-precision).
2. OpenAICompatibleBackend: Local inference servers (Ollama, vLLM, LM Studio) at http://localhost:...
3. DeterministicExpertBackend: High-fidelity offline deterministic engine strictly conforming
   to the Digital Twin evidence and prompt rules for automated unit tests and CI/CD.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from src.llm.config import LLMConfig
from src.llm.schemas import (
    AnomalyAssessment,
    GridStateAssessment,
    HypothesisExplanation,
    LLMInvestigationInput,
    LLMInvestigationOutput,
    PhysicalAssessment,
)

logger = logging.getLogger(__name__)


class BaseLLMBackend(ABC):
    """Abstract interface for replaceable LLM backends."""

    @abstractmethod
    def load(self) -> None:
        """Loads or connects to the model backend."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generates raw text response for the given prompt and system instructions."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the backend is ready and functional."""
        pass


class OpenAICompatibleBackend(BaseLLMBackend):
    """Client for local OpenAI-compatible inference servers (Ollama, vLLM, LM Studio)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_url = f"{config.api_base.rstrip('/')}/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    def load(self) -> None:
        pass

    def is_available(self) -> bool:
        try:
            test_url = f"{self.config.api_base.rstrip('/')}/models"
            resp = requests.get(test_url, headers=self.headers, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        resp = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class TransformersBackend(BaseLLMBackend):
    """Local Hugging Face Transformers backend with GPU/quantization support."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None

    def is_available(self) -> bool:
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local HuggingFace model: {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if self.config.quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to float16.")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )
        logger.info("Model loaded successfully.")

    def generate(self, prompt: str, system_prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=(self.config.temperature > 0),
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class DeterministicExpertBackend(BaseLLMBackend):
    """Deterministic, domain-grounded expert reasoning backend.

    Synthesizes compliant structured investigation outputs directly from the
    prompt context adhering strictly to the prompt builder rules, schema constraints,
    and separation of observations vs hypotheses. Ideal for offline evaluation,
    deterministic test assertions, and zero-download verification.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Parses the structured Digital Twin evidence embedded in prompt and synthesizes JSON."""
        # Extract fields from prompt context
        anomaly_detected = "ANOMALY DETECTED" in prompt
        
        # Extract topology state
        topo_match = re.search(r"Overall Topology State:\s*([A-Za-z0-9_]+)", prompt)
        topology_state = topo_match.group(1) if topo_match else "UNKNOWN"

        # Extract fusion score
        fusion_match = re.search(r"Evidence Fusion Score:\s*([0-9\.]+)", prompt)
        fusion_score = float(fusion_match.group(1)) if fusion_match else 0.0

        # Extract consistency results
        phys_match = re.search(r"Physical Conservation Laws:\s*([A-Za-z]+)", prompt)
        phys_cons = phys_match.group(1).lower() if phys_match else "unknown"

        topo_c_match = re.search(r"Topology / Breaker Consistency:\s*([A-Za-z]+)", prompt)
        topo_cons = topo_c_match.group(1).lower() if topo_c_match else "unknown"

        meas_match = re.search(r"Measurement Redundancy Consistency:\s*([A-Za-z]+)", prompt)
        meas_cons = meas_match.group(1).lower() if meas_match else "unknown"

        # Extract breaker states
        breakers_match = re.search(r"Circuit Breaker States:\s*(\{[^\}]+\})", prompt)
        breaker_states = {}
        if breakers_match:
            try:
                breaker_states = json.loads(breakers_match.group(1))
            except Exception:
                pass

        # Extract affected components
        aff_match = re.search(r"Affected Components:\s*([^\n]+)", prompt)
        affected_comps = []
        if aff_match:
            raw_aff = aff_match.group(1).strip()
            if raw_aff and raw_aff != "None":
                affected_comps = [c.strip() for c in raw_aff.split(",") if c.strip()]

        # Extract evidence statements
        evidence_statements = []
        ev_matches = re.findall(r"\*\s+([^\n]+)", prompt)
        for ev in ev_matches:
            # exclude schema examples
            if not ev.startswith("<") and not ev.startswith("Direct"):
                evidence_statements.append(ev)

        # Synthesize domain-grounded structured assessment
        if not anomaly_detected and topo_cons == "consistent" and phys_cons == "consistent":
            summary = "Grid operating within normal steady-state tolerances with fully consistent physical laws and breaker positions."
            severity = "NONE"
            explanations = [
                {
                    "hypothesis": "Normal steady-state grid operation",
                    "reasoning": "Evidence Fusion score is below detection threshold, power flows align with closed breaker contacts, and all physical conservation laws are verified.",
                    "confidence": 0.98,
                }
            ]
            recommendations = [
                "Continue routine synchrophasor streaming and automated baseline monitoring.",
                "Maintain standard periodic relay state verification.",
            ]
            limitations = [
                "Telemetry observation window represents a single temporal snapshot.",
            ]

        elif topo_cons == "inconsistent":
            summary = f"Detected physical topology inconsistency where breaker contact status contradicts power flow under state {topology_state}."
            severity = "HIGH"
            explanations = [
                {
                    "hypothesis": "Unauthorized remote trip or breaker status manipulation",
                    "reasoning": "Power flow measurements contradict reported auxiliary breaker contact status or uncoordinated trip logic, which is characteristic of unauthorized control commands or false status injection.",
                    "confidence": 0.85,
                },
                {
                    "hypothesis": "Auxiliary contact mechanical discrepancy or relay sensor failure",
                    "reasoning": "A mechanical auxiliary switch failure or transducer fault on the breaker status circuit could cause reported status to disagree with actual conductor energization.",
                    "confidence": 0.55,
                }
            ]
            recommendations = [
                "Inspect substation GOOSE messaging logs and remote terminal unit (RTU) command historians for unauthorized trip commands.",
                "Dispatch substation technicians to perform physical inspection of breaker auxiliary contact microswitches.",
                "Cross-examine secondary differential protection relays to verify physical line current continuity.",
            ]
            limitations = [
                "Digital Twin detects topological contradiction but cannot independently determine cyber vs mechanical intent without cyber network telemetry.",
            ]

        elif topology_state in ["LINE_1_OUTAGE", "LINE_2_OUTAGE", "ACTIVE_TRANSMISSION_FAULT"] and topo_cons == "consistent":
            summary = f"Detected physical transmission event consistent with network topology resulting in {topology_state}."
            severity = "MODERATE" if "OUTAGE" in topology_state else "HIGH"
            explanations = [
                {
                    "hypothesis": "Physical transmission line fault and coordinated protection clearing",
                    "reasoning": "Line de-energization or overcurrent is fully corroborated by bilateral open circuit breakers, conforming to coordinated protective relay isolation.",
                    "confidence": 0.90,
                },
                {
                    "hypothesis": "Scheduled transmission line switching or maintenance operation",
                    "reasoning": "Coordinated breaker opening and subsequent line de-energization matches standard manual or automated line isolation procedures.",
                    "confidence": 0.40,
                }
            ]
            recommendations = [
                "Correlate event timestamp with lightning strike detection networks, weather radar, and SCADA protection trip logs.",
                "Verify distance protection Zone 1 / Zone 2 relay trip flags in digital fault recorder (DFR) records.",
                "Confirm whether scheduled transmission switching orders were dispatched for the affected corridor.",
            ]
            limitations = [
                "Physical fault classification relies on primary PMU telemetry; fault impedance parameters require detailed DFR oscillography.",
            ]

        elif anomaly_detected and topo_cons == "consistent" and phys_cons == "consistent":
            summary = "Statistical anomaly detected by ML models while deterministic physical conservation laws and switchgear topology remain verified normal."
            severity = "MODERATE"
            explanations = [
                {
                    "hypothesis": "Statistical feature drift or inter-area dynamic power swing",
                    "reasoning": "Evidence Fusion triggered due to multi-variate statistical deviation, yet all physical conservation checks and breaker positions are verified normal.",
                    "confidence": 0.70,
                },
                {
                    "hypothesis": "Stealthy false data injection within normal operational bounds",
                    "reasoning": "An adversary may be manipulating continuous measurements subtly without exceeding physical breaker trip thresholds or violating local conservation laws.",
                    "confidence": 0.45,
                }
            ]
            recommendations = [
                "Monitor multi-channel feature reconstruction residuals to identify drifting sensor channels.",
                "Check SCADA historian for recent generation redispatch or large load switching events in adjacent balancing authorities.",
                "Verify PMU GPS clock synchronization across all substations to rule out reference phase angle drift.",
            ]
            limitations = [
                "In-band anomalies with normal physical topology cannot be definitively attributed to cyber or operational dynamics without network audit logs.",
            ]

        else:
            summary = "Ambiguous or unclassified operational event with incomplete or diverging telemetry."
            severity = "LOW"
            explanations = [
                {
                    "hypothesis": "Inconclusive telemetry or transient measurement divergence",
                    "reasoning": "Digital Twin observations do not match standard physical failure patterns or clear topological inconsistencies.",
                    "confidence": 0.50,
                }
            ]
            recommendations = [
                "Retrieve high-speed point-on-wave waveform captures from adjacent digital fault recorders.",
                "Verify synchrophasor communication network packet loss rates.",
            ]
            limitations = [
                "Insufficient evidence available in the current evaluation window to form a definitive engineering hypothesis.",
            ]

        result_dict = {
            "event_summary": summary,
            "anomaly_assessment": {
                "detected": anomaly_detected,
                "severity": severity,
                "fusion_score": fusion_score,
            },
            "observed_evidence": evidence_statements if evidence_statements else [
                "Digital Twin executed deterministic physical and topology consistency checks."
            ],
            "grid_state": {
                "topology_state": topology_state,
                "affected_components": affected_comps,
                "breaker_states": breaker_states,
            },
            "physical_assessment": {
                "physical_consistency": phys_cons,
                "topology_consistency": topo_cons,
                "measurement_consistency": meas_cons,
            },
            "possible_explanations": explanations,
            "recommended_investigation": recommendations,
            "limitations": limitations,
        }
        return json.dumps(result_dict, indent=2)


class LLMClient:
    """High-level client for model loading, prompt generation, and structured output parsing."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.backend: BaseLLMBackend = self._initialize_backend()

    def _initialize_backend(self) -> BaseLLMBackend:
        backend_choice = self.config.backend

        if backend_choice == "openai_compatible":
            return OpenAICompatibleBackend(self.config)
        elif backend_choice == "transformers":
            return TransformersBackend(self.config)
        elif backend_choice == "deterministic_expert":
            return DeterministicExpertBackend(self.config)
        elif backend_choice == "auto":
            # Auto-detection priority:
            # 1. Check if OpenAI-compatible server is running (e.g. Ollama/vLLM)
            openai_backend = OpenAICompatibleBackend(self.config)
            if openai_backend.is_available():
                logger.info(f"Connected to local OpenAI-compatible endpoint at {self.config.api_base}")
                return openai_backend

            # 2. Check if transformers is installed
            trans_backend = TransformersBackend(self.config)
            if trans_backend.is_available():
                logger.info(f"Using Hugging Face Transformers backend for {self.config.model_name}")
                return trans_backend

            # 3. Deterministic expert fallback
            logger.info("Using DeterministicExpertBackend for offline verifiable reasoning.")
            return DeterministicExpertBackend(self.config)
        else:
            raise ValueError(f"Unknown backend '{backend_choice}'. Supported: 'auto', 'transformers', 'openai_compatible', 'deterministic_expert'.")

    def load_model(self) -> None:
        """Explicitly triggers model loading on the active backend."""
        self.backend.load()

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generates raw text response."""
        return self.backend.generate(prompt, system_prompt)

    def generate_structured(self, prompt: str, system_prompt: str) -> LLMInvestigationOutput:
        """Generates text, cleans reasoning/markdown formatting, and validates JSON schema."""
        raw_output = self.generate(prompt, system_prompt)

        # 1. Strip reasoning tags if present (e.g. <think>...</think>)
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)

        # 2. Extract JSON block from markdown fences if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try finding outermost braces
            brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(1).strip()
            else:
                json_str = cleaned.strip()

        # 3. Parse JSON
        try:
            parsed_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw Output:\n{raw_output}")

        # 4. Construct typed schema object
        output_obj = LLMInvestigationOutput.from_dict(parsed_dict)

        # 5. Check semantic integrity (observations != hypotheses, valid confidence)
        violations = output_obj.validate_semantic_integrity()
        if violations:
            logger.warning(f"Semantic integrity warnings in LLM output: {violations}")

        return output_obj
