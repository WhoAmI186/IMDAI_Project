"""Configuration management for the Smart Grid LLM Investigation module.

Supports local GPU inference (e.g. RTX 4050 6GB), local OpenAI-compatible
inference servers (Ollama, vLLM, LM Studio), and deterministic offline execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration options for the LLM client and inference backend."""

    model_name: str = "Qwen/Qwen3-8B"
    backend: str = "auto"  # "auto", "transformers", "openai_compatible", "deterministic_expert"
    api_base: str = "http://localhost:11434/v1"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    quantization: str = "4bit"  # "4bit", "8bit", "fp16", "none"
    device: str = "cuda"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Loads configuration from environment variables with safe defaults."""
        return cls(
            model_name=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            backend=os.getenv("LLM_BACKEND", "auto").lower(),
            api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            quantization=os.getenv("LLM_QUANTIZATION", "4bit").lower(),
            device=os.getenv("LLM_DEVICE", "cuda"),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT", "30.0")),
        )
