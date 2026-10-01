"""Test Suite for the Smart Grid LLM Investigation Module.

Validates the complete Digital Twin -> Prompt Builder -> LLM Client -> Structured JSON pipeline.
Tests 5 canonical operational scenarios:
1. Normal steady-state operation
2. Natural line outage with consistent topology
3. Anomalous event with topology inconsistency (cyber manipulation hypothesis)
4. Statistical ML anomaly with normal topology
5. Insufficient evidence / ambiguous event
"""

import json
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.digital_twin.schemas import (
    AnomalyEvidenceContainer,
    ComponentStatus,
    ConsistencyStatus,
    DigitalTwinOutput,
    GridStateReconstruction,
    InvestigationReport,
    PhysicalAnalysisResult,
    PrimaryHypothesis,
)
from src.llm.config import LLMConfig
from src.llm.investigator import SmartGridInvestigator
from src.llm.llm_client import LLMClient
from src.llm.prompt_builder import PromptBuilder
from src.llm.schemas import LLMInvestigationInput, LLMInvestigationOutput


@pytest.fixture
def investigator():
    """Provides a deterministic SmartGridInvestigator instance for testing."""
    config = LLMConfig(backend="deterministic_expert", temperature=0.0)
    return SmartGridInvestigator(config=config)


# ==============================================================================
# TEST 1: Normal Steady-State Operation
# ==============================================================================

def test_1_normal_steady_state(investigator):
    """TEST 1: Normal/steady state grid operation."""
    dt_out = DigitalTwinOutput(
        timestamp="test_normal_t0",
        grid_state={
            "topology_state": "ALL_LINES_IN_SERVICE",
            "affected_buses": [],
            "affected_lines": [],
            "affected_relays": [],
        },
        anomaly_evidence={
            "layer1_score": 0.05,
            "layer2_scores": {"aggregated_score": 0.02},
            "fusion_score": 0.035,
            "fusion_flag": 0,
        },
        physical_analysis={
            "physical_consistency": "consistent",
            "topology_consistency": "consistent",
            "measurement_consistency": "consistent",
        },
        component_states={
            "Line_1": "ENERGIZED",
            "Line_2": "ENERGIZED",
            "Bus_1": "ENERGIZED",
            "Bus_2": "ENERGIZED",
            "BR1": "CLOSED",
            "BR2": "CLOSED",
            "BR3": "CLOSED",
            "BR4": "CLOSED",
        },
        investigation={
            "primary_hypothesis": PrimaryHypothesis.NORMAL_OPERATION.value,
            "confidence": 0.98,
            "possible_causes": ["Normal steady-state grid conditions"],
            "uncertainty": [],
        },
        evidence=[
            "All monitored physical electrical laws and topology states are consistent."
        ],
    )

    result = investigator.investigate(dt_out)

    assert isinstance(result, LLMInvestigationOutput)
    assert result.anomaly_assessment.detected is False
    assert result.anomaly_assessment.severity == "NONE"
    assert result.grid_state.topology_state == "ALL_LINES_IN_SERVICE"
    assert result.physical_assessment.physical_consistency == "consistent"
    assert result.physical_assessment.topology_consistency == "consistent"

    # Verify hypotheses include normal operation
    hyp_names = [exp.hypothesis.lower() for exp in result.possible_explanations]
    assert any("normal" in h for h in hyp_names)

    # Verify observations do NOT contain speculative attack claims
    violations = result.validate_semantic_integrity()
    assert len(violations) == 0


# ==============================================================================
# TEST 2: Natural Line Outage with Consistent Topology
# ==============================================================================

def test_2_natural_line_outage_consistent_topology(investigator):
    """TEST 2: Natural line outage with topology consistent."""
    dt_out = DigitalTwinOutput(
        timestamp="test_natural_t1",
        grid_state={
            "topology_state": "LINE_1_OUTAGE",
            "affected_buses": [],
            "affected_lines": ["Line_1"],
            "affected_relays": ["R1", "R2"],
        },
        anomaly_evidence={
            "layer1_score": 0.85,
            "layer2_scores": {"aggregated_score": 0.78},
            "fusion_score": 0.815,
            "fusion_flag": 1,
        },
        physical_analysis={
            "physical_consistency": "consistent",
            "topology_consistency": "consistent",
            "measurement_consistency": "consistent",
        },
        component_states={
            "Line_1": "DE_ENERGIZED",
            "Line_2": "ENERGIZED",
            "Bus_1": "ENERGIZED",
            "Bus_2": "ENERGIZED",
            "BR1": "OPEN",
            "BR2": "OPEN",
            "BR3": "CLOSED",
            "BR4": "CLOSED",
        },
        investigation={
            "primary_hypothesis": PrimaryHypothesis.PHYSICAL_OUTAGE_CONSISTENT.value,
            "confidence": 0.90,
            "possible_causes": [
                "Transmission line physical fault (phase-to-ground or tree contact)",
                "Coordinated protective relay clearing action",
            ],
            "uncertainty": [],
        },
        evidence=[
            "Evidence Fusion triggered: FusedScore=0.8150 exceeds threshold; L1_flag=1, L2_flag=1.",
            "Physical anomaly matches recognized grid reconfiguration: LINE_1_OUTAGE.",
        ],
    )

    result = investigator.investigate(dt_out)

    assert isinstance(result, LLMInvestigationOutput)
    assert result.anomaly_assessment.detected is True
    assert result.grid_state.topology_state == "LINE_1_OUTAGE"
    assert "Line_1" in result.grid_state.affected_components
    assert result.grid_state.breaker_states["BR1"] == "OPEN"
    assert result.grid_state.breaker_states["BR2"] == "OPEN"
    assert result.physical_assessment.topology_consistency == "consistent"

    # Plausible explanation must discuss physical fault or protective clearing
    hyp_text = " ".join([exp.hypothesis.lower() + " " + exp.reasoning.lower() for exp in result.possible_explanations])
    assert "fault" in hyp_text or "clearing" in hyp_text or "outage" in hyp_text

    # Semantic integrity check
    violations = result.validate_semantic_integrity()
    assert len(violations) == 0


# ==============================================================================
# TEST 3: Anomalous Event with Topology Inconsistency
# ==============================================================================

def test_3_anomalous_event_topology_inconsistency(investigator):
    """TEST 3: Anomalous event with topology inconsistency (possible cyber manipulation)."""
    dt_out = DigitalTwinOutput(
        timestamp="test_attack_t2",
        grid_state={
            "topology_state": "LINE_1_OUTAGE",
            "affected_buses": [],
            "affected_lines": ["Line_1"],
            "affected_relays": ["R1"],
        },
        anomaly_evidence={
            "layer1_score": 0.92,
            "layer2_scores": {"aggregated_score": 0.88},
            "fusion_score": 0.900,
            "fusion_flag": 1,
        },
        physical_analysis={
            "physical_consistency": "consistent",
            "topology_consistency": "inconsistent",
            "measurement_consistency": "consistent",
        },
        component_states={
            "Line_1": "DE_ENERGIZED",
            "Line_2": "ENERGIZED",
            "Bus_1": "ENERGIZED",
            "Bus_2": "ENERGIZED",
            "BR1": "CLOSED",
            "BR2": "CLOSED",
            "BR3": "CLOSED",
            "BR4": "CLOSED",
        },
        investigation={
            "primary_hypothesis": PrimaryHypothesis.UNEXPECTED_TOPOLOGY_INCONSISTENCY.value,
            "confidence": 0.85,
            "possible_causes": [
                "Unauthorized remote trip command to circuit breaker",
                "False breaker status injection (FDI)",
            ],
            "uncertainty": [],
        },
        evidence=[
            "Evidence Fusion triggered: FusedScore=0.9000 exceeds threshold; L1_flag=1, L2_flag=1.",
            "Observed power flows contradict reported circuit breaker status or protection logic.",
            "Line_1 current is 0.0A (de-energized) while breakers BR1 and BR2 report CLOSED.",
        ],
    )

    result = investigator.investigate(dt_out)

    assert isinstance(result, LLMInvestigationOutput)
    assert result.anomaly_assessment.detected is True
    assert result.anomaly_assessment.severity == "HIGH"
    assert result.physical_assessment.topology_consistency == "inconsistent"

    # Must identify cyber manipulation OR sensor failure as hypotheses (NOT definitive proof)
    hypotheses = [exp.hypothesis.lower() for exp in result.possible_explanations]
    assert any("unauthorized" in h or "breaker" in h or "status" in h for h in hypotheses)

    # Verification: Observations must NOT state "definitely a cyberattack"
    for ev in result.observed_evidence:
        assert "definitely an attack" not in ev.lower()
        assert "confirmed cyberattack" not in ev.lower()

    # Recommendations must include investigating GOOSE logs or RTU
    recs = " ".join(result.recommended_investigation).lower()
    assert "goose" in recs or "substation" in recs or "breaker" in recs


# ==============================================================================
# TEST 4: Statistical ML Anomaly with Normal Topology
# ==============================================================================

def test_4_statistical_ml_anomaly_normal_topology(investigator):
    """TEST 4: Statistical ML anomaly with normal physical topology."""
    dt_out = DigitalTwinOutput(
        timestamp="test_stat_t3",
        grid_state={
            "topology_state": "ALL_LINES_IN_SERVICE",
            "affected_buses": [],
            "affected_lines": [],
            "affected_relays": [],
        },
        anomaly_evidence={
            "layer1_score": 0.76,
            "layer2_scores": {"aggregated_score": 0.40},
            "fusion_score": 0.742,
            "fusion_flag": 1,
        },
        physical_analysis={
            "physical_consistency": "consistent",
            "topology_consistency": "consistent",
            "measurement_consistency": "consistent",
        },
        component_states={
            "Line_1": "ENERGIZED",
            "Line_2": "ENERGIZED",
            "Bus_1": "ENERGIZED",
            "Bus_2": "ENERGIZED",
            "BR1": "CLOSED",
            "BR2": "CLOSED",
            "BR3": "CLOSED",
            "BR4": "CLOSED",
        },
        investigation={
            "primary_hypothesis": PrimaryHypothesis.STATISTICAL_DISTURBANCE_NORMAL_TOPOLOGY.value,
            "confidence": 0.70,
            "possible_causes": [
                "Statistical anomaly detected by ML pipeline without physical conservation law violation",
                "Inter-area power oscillation or dynamic transient swing within operating limits",
            ],
            "uncertainty": [],
        },
        evidence=[
            "Evidence Fusion triggered statistical anomaly (FusedScore=0.7420), but all deterministic physical conservation laws and breaker contacts are verified normal."
        ],
    )

    result = investigator.investigate(dt_out)

    assert isinstance(result, LLMInvestigationOutput)
    assert result.anomaly_assessment.detected is True
    assert result.grid_state.topology_state == "ALL_LINES_IN_SERVICE"
    assert result.physical_assessment.physical_consistency == "consistent"
    assert result.physical_assessment.topology_consistency == "consistent"

    # Hypotheses should address statistical drift, oscillation, or stealthy FDI
    hyp_text = " ".join([exp.hypothesis.lower() + " " + exp.reasoning.lower() for exp in result.possible_explanations])
    assert "statistical" in hyp_text or "drift" in hyp_text or "oscillation" in hyp_text or "injection" in hyp_text

    # Crucial check: Does NOT falsely claim physical laws were violated
    assert result.physical_assessment.physical_consistency == "consistent"


# ==============================================================================
# TEST 5: Insufficient Evidence / Ambiguous Event
# ==============================================================================

def test_5_insufficient_evidence_ambiguous(investigator):
    """TEST 5: Insufficient evidence / ambiguous event."""
    dt_out = DigitalTwinOutput(
        timestamp="test_unknown_t4",
        grid_state={
            "topology_state": "UNKNOWN",
            "affected_buses": [],
            "affected_lines": [],
            "affected_relays": [],
        },
        anomaly_evidence={
            "layer1_score": 0.0,
            "layer2_scores": {},
            "fusion_score": 0.0,
            "fusion_flag": 0,
        },
        physical_analysis={
            "physical_consistency": "unknown",
            "topology_consistency": "unknown",
            "measurement_consistency": "unknown",
        },
        component_states={
            "Line_1": "UNKNOWN",
            "Line_2": "UNKNOWN",
            "Bus_1": "UNKNOWN",
            "Bus_2": "UNKNOWN",
            "BR1": "UNKNOWN",
            "BR2": "UNKNOWN",
            "BR3": "UNKNOWN",
            "BR4": "UNKNOWN",
        },
        investigation={
            "primary_hypothesis": PrimaryHypothesis.INSUFFICIENT_EVIDENCE.value,
            "confidence": 0.50,
            "possible_causes": ["Telemetry ambiguous, missing, or incomplete"],
            "uncertainty": ["Critical synchrophasor channels or breaker status words are absent."],
        },
        evidence=[
            "Critical synchrophasor channels or breaker status words are absent."
        ],
    )

    result = investigator.investigate(dt_out)

    assert isinstance(result, LLMInvestigationOutput)
    assert result.grid_state.topology_state == "UNKNOWN"
    assert result.physical_assessment.physical_consistency == "unknown"

    # Limitations must explicitly state data is incomplete or insufficient
    lim_text = " ".join(result.limitations).lower()
    assert "insufficient" in lim_text or "limitation" in lim_text or "missing" in lim_text


# ==============================================================================
# TEST 6: Input Sanitization & Ground-Truth Independence
# ==============================================================================

def test_6_input_sanitization_no_ground_truth():
    """Verifies that LLMInvestigationInput completely excludes raw PMU rows and ground-truth labels."""
    dt_out = DigitalTwinOutput(
        timestamp="row_100",
        grid_state={"topology_state": "ALL_LINES_IN_SERVICE"},
        anomaly_evidence={"fusion_score": 0.1, "layer1_score": 0.05, "fusion_flag": 0},
        physical_analysis={"physical_consistency": "consistent", "topology_consistency": "consistent", "measurement_consistency": "consistent"},
        component_states={"Line_1": "ENERGIZED", "BR1": "CLOSED"},
        investigation={"primary_hypothesis": "Normal steady-state grid operation", "confidence": 0.98},
        evidence=["Normal telemetry."],
    )

    llm_inp = LLMInvestigationInput.from_digital_twin_output(dt_out)
    inp_dict = llm_inp.to_dict()

    # Verify no ground-truth marker fields exist
    assert "marker" not in inp_dict
    assert "attack" not in inp_dict
    assert "natural" not in inp_dict
    assert "raw_features" not in inp_dict
    assert "features_df" not in inp_dict


# ==============================================================================
# TEST 7: JSON Parsing and Markdown Fence Stripping
# ==============================================================================

def test_7_markdown_code_fence_stripping():
    """Verifies that LLMClient properly strips markdown fences and reasoning tags."""
    config = LLMConfig(backend="deterministic_expert")
    client = LLMClient(config)

    raw_response_with_markdown = """
Here is the investigation:
```json
{
  "event_summary": "Test event summary",
  "anomaly_assessment": {
    "detected": true,
    "severity": "HIGH",
    "fusion_score": 0.85
  },
  "observed_evidence": ["Evidence 1"],
  "grid_state": {
    "topology_state": "LINE_1_OUTAGE",
    "affected_components": ["Line_1"],
    "breaker_states": {"BR1": "OPEN"}
  },
  "physical_assessment": {
    "physical_consistency": "consistent",
    "topology_consistency": "consistent",
    "measurement_consistency": "consistent"
  },
  "possible_explanations": [
    {
      "hypothesis": "Transmission line fault",
      "reasoning": "Line 1 is de-energized with open breakers.",
      "confidence": 0.9
    }
  ],
  "recommended_investigation": ["Check DFR"],
  "limitations": ["Single snapshot"]
}
```
"""

    # Mock the backend generate return
    class MockBackend:
        def generate(self, p, s):
            return raw_response_with_markdown

    client.backend = MockBackend()
    res = client.generate_structured("prompt", "system")

    assert res.event_summary == "Test event summary"
    assert res.anomaly_assessment.severity == "HIGH"
    assert res.grid_state.topology_state == "LINE_1_OUTAGE"
