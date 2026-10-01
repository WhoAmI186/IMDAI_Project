"""Schemas and Data Containers for the Power System Digital Twin.

Defines standardized data interfaces for inputs, intermediate states, and outputs
conforming to the Cyber-Physical Smart Grid architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ComponentStatus(str, Enum):
    """Operational status of a power system component (Line, Bus, Generator)."""

    ENERGIZED = "ENERGIZED"
    DE_ENERGIZED = "DE_ENERGIZED"
    FAULT = "FAULT"
    ABNORMAL = "ABNORMAL"
    UNKNOWN = "UNKNOWN"


class BreakerStatus(str, Enum):
    """Operational contact state of a circuit breaker."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class ConsistencyStatus(str, Enum):
    """Consistency classification for physical laws, topology, and measurements."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


class PrimaryHypothesis(str, Enum):
    """Primary operational hypothesis explaining the observed grid state."""

    NORMAL_OPERATION = "Normal steady-state grid operation"
    PHYSICAL_OUTAGE_CONSISTENT = "Physical event consistent with observed topology"
    UNEXPECTED_TOPOLOGY_INCONSISTENCY = "Unexpected topology/state inconsistency; possible cyber-related event"
    SENSOR_MEASUREMENT_ISSUE = "Possible measurement/sensor issue"
    UNEXPECTED_PHYSICAL_DISCREPANCY = "Unexpected physical law discrepancy without clear topology explanation"
    STATISTICAL_DISTURBANCE_NORMAL_TOPOLOGY = "Statistical ML anomaly with verified normal physical topology"
    INSUFFICIENT_EVIDENCE = "Unknown / insufficient evidence"


@dataclass
class DigitalTwinInput:
    """Structured input consumed by the Digital Twin from telemetry and ML layers."""

    timestamp: Union[int, float, str]
    telemetry: Dict[str, float]
    layer1: Dict[str, Any]  # {"anomaly_score": float, "anomaly_flag": int}
    layer2: Dict[str, Any]  # {"bus1_voltage_score": ..., "aggregated_score": ...}
    fusion: Dict[str, Any]  # {"fused_score": float, "anomaly_flag": int}


@dataclass
class GridStateReconstruction:
    """Reconstructed topological state and localized affected components."""

    topology_state: str
    affected_buses: List[str] = field(default_factory=list)
    affected_lines: List[str] = field(default_factory=list)
    affected_relays: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyEvidenceContainer:
    """Packaged ML anomaly evidence passed from Layer 1, Layer 2, and Evidence Fusion."""

    layer1_score: float
    layer2_scores: Dict[str, float]
    fusion_score: float
    fusion_flag: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalAnalysisResult:
    """Outcomes of deterministic physical law and topology consistency checks."""

    physical_consistency: str  # "consistent", "inconsistent", "unknown"
    measurement_consistency: str  # "consistent", "inconsistent", "unknown"
    topology_consistency: str  # "consistent", "inconsistent", "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationReport:
    """Synthesized engineering hypothesis, possible causes, and confidence metrics."""

    possible_causes: List[str] = field(default_factory=list)
    primary_hypothesis: str = PrimaryHypothesis.NORMAL_OPERATION.value
    confidence: float = 1.0
    uncertainty: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DigitalTwinOutput:
    """Complete structured output produced by the Digital Twin."""

    timestamp: str
    grid_state: Dict[str, Any]
    anomaly_evidence: Dict[str, Any]
    physical_analysis: Dict[str, Any]
    component_states: Dict[str, str]
    investigation: Dict[str, Any]
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "grid_state": self.grid_state,
            "anomaly_evidence": self.anomaly_evidence,
            "physical_analysis": self.physical_analysis,
            "component_states": self.component_states,
            "investigation": self.investigation,
            "evidence": self.evidence,
        }
