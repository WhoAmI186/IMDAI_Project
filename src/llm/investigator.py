"""Smart Grid Event Investigator.

Coordinates the Digital Twin -> LLM investigation pipeline:
1. Ingests DigitalTwinOutput.
2. Extracts compact, sanitized LLMInvestigationInput (no raw PMU, no ground truth).
3. Builds grounded prompts via PromptBuilder.
4. Generates structured JSON via LLMClient.
5. Returns validated LLMInvestigationOutput adhering strictly to OBSERVATIONS != HYPOTHESES.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from src.digital_twin.schemas import DigitalTwinOutput
from src.llm.config import LLMConfig
from src.llm.llm_client import LLMClient
from src.llm.prompt_builder import PromptBuilder
from src.llm.schemas import LLMInvestigationInput, LLMInvestigationOutput

logger = logging.getLogger(__name__)


class SmartGridInvestigator:
    """End-to-end investigation service bridging the Digital Twin and LLM reasoning."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.llm_client = llm_client or LLMClient(self.config)
        self.prompt_builder = prompt_builder or PromptBuilder()

    def investigate(
        self,
        dt_output: Union[DigitalTwinOutput, Dict[str, Any]],
    ) -> LLMInvestigationOutput:
        """Executes an LLM-assisted investigation of a single Digital Twin output.

        Parameters
        ----------
        dt_output : DigitalTwinOutput or dict
            The structured output from the PowerSystemDigitalTwin.

        Returns
        -------
        LLMInvestigationOutput
            Validated structured investigation containing summary, observed evidence,
            grid state, physical assessment, plausible hypotheses, recommended actions,
            and known limitations.
        """
        # 1. Transform DigitalTwinOutput to sanitized LLMInvestigationInput
        if isinstance(dt_output, DigitalTwinOutput):
            dt_input = LLMInvestigationInput.from_digital_twin_output(dt_output)
        elif isinstance(dt_output, dict):
            # Convert dictionary representation to DigitalTwinOutput first
            dt_obj = DigitalTwinOutput(
                timestamp=str(dt_output.get("timestamp", "0")),
                grid_state=dt_output.get("grid_state", {}),
                anomaly_evidence=dt_output.get("anomaly_evidence", {}),
                physical_analysis=dt_output.get("physical_analysis", {}),
                component_states=dt_output.get("component_states", {}),
                investigation=dt_output.get("investigation", {}),
                evidence=dt_output.get("evidence", []),
            )
            dt_input = LLMInvestigationInput.from_digital_twin_output(dt_obj)
        else:
            raise TypeError(f"Expected DigitalTwinOutput or dict, got {type(dt_output)}")

        # 2. Build system and user prompts
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(dt_input)

        # 3. Generate structured response
        investigation_result = self.llm_client.generate_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return investigation_result

    def investigate_batch(
        self,
        dt_outputs: List[Union[DigitalTwinOutput, Dict[str, Any]]],
    ) -> List[LLMInvestigationOutput]:
        """Investigates a sequence of Digital Twin outputs."""
        results = []
        for i, out in enumerate(dt_outputs):
            try:
                res = self.investigate(out)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed to investigate event at index {i}: {e}")
                # Fallback to an empty or partial investigation record
                fallback = LLMInvestigationOutput(
                    event_summary=f"Investigation failed at step {i}: {e}",
                    anomaly_assessment={"detected": False, "severity": "NONE", "fusion_score": 0.0},
                    observed_evidence=["Investigation exception occurred."],
                    grid_state={"topology_state": "UNKNOWN", "affected_components": [], "breaker_states": {}},
                    physical_assessment={"physical_consistency": "unknown", "topology_consistency": "unknown", "measurement_consistency": "unknown"},
                    possible_explanations=[{"hypothesis": "Pipeline execution error", "reasoning": str(e), "confidence": 0.0}],
                    recommended_investigation=["Check LLM backend connectivity and logs."],
                    limitations=["Analysis aborted due to execution error."],
                )
                results.append(fallback)
        return results
