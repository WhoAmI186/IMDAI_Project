"""Schemas and Data Interfaces for the LLM Investigation Layer.

Defines:
1. LLMInvestigationInput: Compact, sanitized representation extracted from DigitalTwinOutput.
   - Zero raw PMU data.
   - Zero ground-truth labels.
2. LLMInvestigationOutput: Structured JSON schema returned by the LLM.
   - Strict separation between observed evidence and speculative hypotheses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.digital_twin.schemas import DigitalTwinOutput


@dataclass
class LLMInvestigationInput:
    """Sanitized, compact representation passed from Digital Twin to the LLM.

    Contains exclusively physical state reconstructions, deterministic check outcomes,
    anomaly evidence scores, and engineering evidence statements.
    Explicitly excludes raw 129-column PMU time-series and dataset ground-truth labels.
    """

    timestamp: str
    anomaly_detected: bool
    fusion_score: float
    layer1_score: float
    layer2_score: float
    topology_state: str
    line_states: Dict[str, str]
    bus_states: Dict[str, str]
    breaker_states: Dict[str, str]
    affected_components: List[str]
    physical_consistency: str
    topology_consistency: str
    measurement_consistency: str
    digital_twin_hypothesis: str
    digital_twin_confidence: float
    supporting_evidence: List[str]
    uncertainties: List[str] = field(default_factory=list)

    @classmethod
    def from_digital_twin_output(cls, dt_out: DigitalTwinOutput) -> LLMInvestigationInput:
        """Constructs an LLMInvestigationInput from a DigitalTwinOutput instance."""
        anomaly_ev = dt_out.anomaly_evidence
        grid_state = dt_out.grid_state
        phys_analysis = dt_out.physical_analysis
        investigation = dt_out.investigation
        comp_states = dt_out.component_states

        # Extract localized component groups
        line_states = {k: v for k, v in comp_states.items() if k.startswith("Line_")}
        bus_states = {k: v for k, v in comp_states.items() if k.startswith("Bus_")}
        breaker_states = {k: v for k, v in comp_states.items() if k.startswith("BR")}

        affected = []
        affected.extend(grid_state.get("affected_lines", []))
        affected.extend(grid_state.get("affected_buses", []))
        affected.extend(grid_state.get("affected_relays", []))

        # Check if an anomaly was raised upstream
        fusion_flag = int(anomaly_ev.get("fusion_flag", 0))
        anomaly_detected = (fusion_flag == 1)

        return cls(
            timestamp=dt_out.timestamp,
            anomaly_detected=anomaly_detected,
            fusion_score=float(anomaly_ev.get("fusion_score", 0.0)),
            layer1_score=float(anomaly_ev.get("layer1_score", 0.0)),
            layer2_score=float(anomaly_ev.get("layer2_scores", {}).get("aggregated_score", 0.0)
                               if isinstance(anomaly_ev.get("layer2_scores"), dict) else 0.0),
            topology_state=grid_state.get("topology_state", "UNKNOWN"),
            line_states=line_states,
            bus_states=bus_states,
            breaker_states=breaker_states,
            affected_components=sorted(list(set(affected))),
            physical_consistency=phys_analysis.get("physical_consistency", "unknown"),
            topology_consistency=phys_analysis.get("topology_consistency", "unknown"),
            measurement_consistency=phys_analysis.get("measurement_consistency", "unknown"),
            digital_twin_hypothesis=investigation.get("primary_hypothesis", "Unknown / insufficient evidence"),
            digital_twin_confidence=float(investigation.get("confidence", 0.5)),
            supporting_evidence=list(dt_out.evidence),
            uncertainties=list(investigation.get("uncertainty", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Formats the structured input into a clean, human-readable text block for prompting."""
        lines = [
            f"Timestamp: {self.timestamp}",
            f"Anomaly Detection Status: {'ANOMALY DETECTED' if self.anomaly_detected else 'NORMAL (No Anomaly Flagged)'}",
            f"Evidence Fusion Score: {self.fusion_score:.4f}",
            f"Layer 1 Score: {self.layer1_score:.4f} | Layer 2 Score: {self.layer2_score:.4f}",
            "",
            "Grid State Reconstruction:",
            f"  - Overall Topology State: {self.topology_state}",
            f"  - Line States: {json.dumps(self.line_states)}",
            f"  - Bus States: {json.dumps(self.bus_states)}",
            f"  - Circuit Breaker States: {json.dumps(self.breaker_states)}",
            f"  - Affected Components: {', '.join(self.affected_components) if self.affected_components else 'None'}",
            "",
            "Physical & Consistency Assessments:",
            f"  - Physical Conservation Laws: {self.physical_consistency.upper()}",
            f"  - Topology / Breaker Consistency: {self.topology_consistency.upper()}",
            f"  - Measurement Redundancy Consistency: {self.measurement_consistency.upper()}",
            "",
            "Deterministic Digital Twin Findings:",
            f"  - Digital Twin Hypothesis: {self.digital_twin_hypothesis}",
            f"  - Deterministic Confidence: {self.digital_twin_confidence:.2f}",
        ]
        if self.supporting_evidence:
            lines.append("  - Observed Evidence Statements:")
            for ev in self.supporting_evidence:
                lines.append(f"    * {ev}")
        if self.uncertainties:
            lines.append("  - Known Uncertainties / Data Limitations:")
            for un in self.uncertainties:
                lines.append(f"    * {un}")

        return "\n".join(lines)


# ==============================================================================
# STRUCTURED LLM OUTPUT SCHEMAS
# ==============================================================================

@dataclass
class AnomalyAssessment:
    """LLM assessment of the detected anomaly."""
    detected: bool
    severity: str  # "NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"
    fusion_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GridStateAssessment:
    """LLM summary of the reconstructed power grid state."""
    topology_state: str
    affected_components: List[str]
    breaker_states: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalAssessment:
    """LLM synthesis of physical and topological consistency."""
    physical_consistency: str  # "consistent", "inconsistent", "unknown"
    topology_consistency: str  # "consistent", "inconsistent", "unknown"
    measurement_consistency: str  # "consistent", "inconsistent", "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisExplanation:
    """A plausible explanation or hypothesis formulated by the LLM."""
    hypothesis: str
    reasoning: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMInvestigationOutput:
    """Complete structured JSON output returned by the LLM Investigation Layer.

    Adheres strictly to the requirement: OBSERVATIONS != HYPOTHESES.
    """
    event_summary: str
    anomaly_assessment: AnomalyAssessment
    observed_evidence: List[str]
    grid_state: GridStateAssessment
    physical_assessment: PhysicalAssessment
    possible_explanations: List[HypothesisExplanation]
    recommended_investigation: List[str]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_summary": self.event_summary,
            "anomaly_assessment": self.anomaly_assessment.to_dict(),
            "observed_evidence": list(self.observed_evidence),
            "grid_state": self.grid_state.to_dict(),
            "physical_assessment": self.physical_assessment.to_dict(),
            "possible_explanations": [exp.to_dict() for exp in self.possible_explanations],
            "recommended_investigation": list(self.recommended_investigation),
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMInvestigationOutput:
        """Constructs and validates LLMInvestigationOutput from a raw dictionary."""
        anomaly_data = data.get("anomaly_assessment", {})
        anomaly_assessment = AnomalyAssessment(
            detected=bool(anomaly_data.get("detected", False)),
            severity=str(anomaly_data.get("severity", "NONE")),
            fusion_score=float(anomaly_data.get("fusion_score", 0.0)),
        )

        grid_data = data.get("grid_state", {})
        grid_state = GridStateAssessment(
            topology_state=str(grid_data.get("topology_state", "UNKNOWN")),
            affected_components=list(grid_data.get("affected_components", [])),
            breaker_states=dict(grid_data.get("breaker_states", {})),
        )

        phys_data = data.get("physical_assessment", {})
        physical_assessment = PhysicalAssessment(
            physical_consistency=str(phys_data.get("physical_consistency", "unknown")),
            topology_consistency=str(phys_data.get("topology_consistency", "unknown")),
            measurement_consistency=str(phys_data.get("measurement_consistency", "unknown")),
        )

        explanations_data = data.get("possible_explanations", [])
        possible_explanations = [
            HypothesisExplanation(
                hypothesis=str(exp.get("hypothesis", "")),
                reasoning=str(exp.get("reasoning", "")),
                confidence=float(exp.get("confidence", 0.0)),
            )
            for exp in explanations_data
        ]

        return cls(
            event_summary=str(data.get("event_summary", "")),
            anomaly_assessment=anomaly_assessment,
            observed_evidence=list(data.get("observed_evidence", [])),
            grid_state=grid_state,
            physical_assessment=physical_assessment,
            possible_explanations=possible_explanations,
            recommended_investigation=list(data.get("recommended_investigation", [])),
            limitations=list(data.get("limitations", [])),
        )

    @classmethod
    def from_json(cls, json_str: str) -> LLMInvestigationOutput:
        """Parses a JSON string into an LLMInvestigationOutput instance."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate_semantic_integrity(self) -> List[str]:
        """Validates that the output adheres to key semantic rules and separation of concerns.

        Returns a list of violation messages if any rule is broken.
        """
        violations = []

        # Rule 1: OBSERVATIONS != HYPOTHESES
        # Observed evidence must describe observations, not speculative attack conclusions
        forbidden_in_observations = ["definitely an attack", "confirmed cyberattack", "attacker did", "malicious payload"]
        for ev in self.observed_evidence:
            ev_lower = ev.lower()
            for phrase in forbidden_in_observations:
                if phrase in ev_lower:
                    violations.append(f"Observed evidence contains speculative conclusion: '{ev}'")

        # Rule 2: Must have at least one explanation and one investigation recommendation
        if not self.possible_explanations:
            violations.append("Output must contain at least one plausible explanation.")
        if not self.recommended_investigation:
            violations.append("Output must contain recommended investigation steps.")

        # Rule 3: Confidences in explanations must be between 0.0 and 1.0
        for exp in self.possible_explanations:
            if not (0.0 <= exp.confidence <= 1.0):
                violations.append(f"Hypothesis confidence {exp.confidence} out of range [0.0, 1.0].")

        return violations
