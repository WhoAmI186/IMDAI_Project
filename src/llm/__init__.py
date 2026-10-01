"""Smart Grid LLM Investigation Module.

Provides structured, grounded event investigation, explanation, and operator guidance
downstream of the Power System Digital Twin.
"""

from src.llm.config import LLMConfig
from src.llm.investigator import SmartGridInvestigator
from src.llm.llm_client import (
    BaseLLMBackend,
    DeterministicExpertBackend,
    LLMClient,
    OpenAICompatibleBackend,
    TransformersBackend,
)
from src.llm.prompt_builder import PromptBuilder
from src.llm.schemas import (
    AnomalyAssessment,
    GridStateAssessment,
    HypothesisExplanation,
    LLMInvestigationInput,
    LLMInvestigationOutput,
    PhysicalAssessment,
)

__all__ = [
    "LLMConfig",
    "SmartGridInvestigator",
    "LLMClient",
    "BaseLLMBackend",
    "TransformersBackend",
    "OpenAICompatibleBackend",
    "DeterministicExpertBackend",
    "PromptBuilder",
    "LLMInvestigationInput",
    "LLMInvestigationOutput",
    "AnomalyAssessment",
    "GridStateAssessment",
    "PhysicalAssessment",
    "HypothesisExplanation",
]
