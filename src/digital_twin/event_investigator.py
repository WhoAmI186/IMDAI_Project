"""Event Investigation and Evidence Synthesis Engine for the Power System Digital Twin.

Correlates ML anomaly evidence, deterministic physical laws, and topology-state consistency
to formulate structured engineering hypotheses and actionable forensic evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.digital_twin.physical_checks import PhysicalCheckResult
from src.digital_twin.schemas import (
    ConsistencyStatus,
    GridStateReconstruction,
    InvestigationReport,
    PhysicalAnalysisResult,
    PrimaryHypothesis,
)
from src.digital_twin.topology_checks import TopologyCheckResult


class EventInvestigator:
    """Synthesizes factual telemetry evidence, physical consistency, and topology states."""

    def investigate(
        self,
        grid_state: GridStateReconstruction,
        component_states: Dict[str, str],
        physical_results: List[PhysicalCheckResult],
        topology_results: List[TopologyCheckResult],
        layer1_evidence: Dict[str, Any],
        layer2_evidence: Dict[str, Any],
        fusion_evidence: Dict[str, Any],
    ) -> Tuple[PhysicalAnalysisResult, InvestigationReport, List[str]]:
        """Conducts structured forensic investigation without black-box classifiers or LLMs."""
        evidence_statements: List[str] = []
        possible_causes: List[str] = []
        uncertainties: List[str] = []

        # 1. Evaluate Overall Physical Law Consistency
        inconsistent_phys = [r for r in physical_results if r.status == ConsistencyStatus.INCONSISTENT]
        unknown_phys = [r for r in physical_results if r.status == ConsistencyStatus.UNKNOWN]

        if inconsistent_phys:
            physical_consistency = ConsistencyStatus.INCONSISTENT.value
            for r in inconsistent_phys:
                evidence_statements.append(f"Physical violation: {r.details}")
        elif unknown_phys and len(unknown_phys) >= 2:
            physical_consistency = ConsistencyStatus.UNKNOWN.value
            uncertainties.append("Multiple physical checks returned UNKNOWN due to missing telemetry.")
        else:
            physical_consistency = ConsistencyStatus.CONSISTENT.value

        # 2. Evaluate Measurement Consistency
        # Checks whether redundant sensors on the same bus or line agree
        redundancy_violations = [
            r for r in inconsistent_phys
            if "equipotential" in r.check_name or "frequency_synchronization" in r.check_name
        ]
        if redundancy_violations:
            measurement_consistency = ConsistencyStatus.INCONSISTENT.value
            for r in redundancy_violations:
                evidence_statements.append(f"Measurement discrepancy: {r.details}")
        else:
            measurement_consistency = ConsistencyStatus.CONSISTENT.value

        # 3. Evaluate Topology Consistency
        inconsistent_topo = [r for r in topology_results if r.status == ConsistencyStatus.INCONSISTENT]
        unknown_topo = [r for r in topology_results if r.status == ConsistencyStatus.UNKNOWN]

        if inconsistent_topo:
            topology_consistency = ConsistencyStatus.INCONSISTENT.value
            for r in inconsistent_topo:
                evidence_statements.append(f"Topology inconsistency: {r.evidence_statement}")
        elif unknown_topo and len(unknown_topo) == len(topology_results):
            topology_consistency = ConsistencyStatus.UNKNOWN.value
            uncertainties.append("Breaker status information is completely missing or UNKNOWN.")
        else:
            topology_consistency = ConsistencyStatus.CONSISTENT.value

        physical_analysis = PhysicalAnalysisResult(
            physical_consistency=physical_consistency,
            measurement_consistency=measurement_consistency,
            topology_consistency=topology_consistency,
        )

        # 4. Formulate Primary Hypothesis and Possible Causes
        fused_flag = int(fusion_evidence.get("anomaly_flag", 0))
        fused_score = float(fusion_evidence.get("fused_score", 0.0))
        l1_flag = int(layer1_evidence.get("anomaly_flag", 0))
        l2_flag = int(layer2_evidence.get("aggregated_score", 0.0) >= 1.0)

        # Log ML anomaly flags as evidence
        if fused_flag == 1:
            evidence_statements.append(
                f"Evidence Fusion triggered: FusedScore={fused_score:.4f} exceeds threshold; "
                f"L1_flag={l1_flag}, L2_flag={l2_flag}."
            )

        # Case 0: Insufficient Evidence (missing telemetry or unknown topology)
        if grid_state.topology_state == "UNKNOWN" or physical_consistency == ConsistencyStatus.UNKNOWN.value or topology_consistency == ConsistencyStatus.UNKNOWN.value:
            primary_hyp = PrimaryHypothesis.INSUFFICIENT_EVIDENCE.value
            possible_causes.append("Telemetry ambiguous, missing, or incomplete")
            confidence = 0.50
            uncertainties.append("Insufficient data to establish definitive grid operational state.")
            evidence_statements.append("Critical synchrophasor channels or breaker status words are absent.")

        # Case A: Normal Steady-State Operation
        elif fused_flag == 0 and l1_flag == 0 and not inconsistent_phys and not inconsistent_topo:
            primary_hyp = PrimaryHypothesis.NORMAL_OPERATION.value
            possible_causes.append("Normal steady-state grid conditions")
            confidence = 0.98
            evidence_statements.append("All monitored physical electrical laws and topology states are consistent.")

        # Case B: Unexpected Topology Inconsistency -> Possible Cyber Manipulation
        elif inconsistent_topo:
            primary_hyp = PrimaryHypothesis.UNEXPECTED_TOPOLOGY_INCONSISTENCY.value
            possible_causes.extend([
                "Unauthorized remote trip command to circuit breaker",
                "False breaker status injection (FDI)",
                "Relay setting manipulation causing premature opening",
                "Uncoordinated switching action without grid fault",
            ])
            confidence = 0.85
            evidence_statements.append(
                "Observed power flows contradict reported circuit breaker status or protection logic."
            )

        # Case C: Sensor / Measurement Discrepancy
        elif measurement_consistency == ConsistencyStatus.INCONSISTENT.value and not inconsistent_topo and grid_state.topology_state == "ALL_LINES_IN_SERVICE":
            primary_hyp = PrimaryHypothesis.SENSOR_MEASUREMENT_ISSUE.value
            possible_causes.extend([
                "PMU instrumentation transformer failure",
                "Sensor measurement drift or loss of synchronization",
                "False data injection targeting individual telemetry stream",
            ])
            confidence = 0.80
            evidence_statements.append(
                "Redundant measurements on equipotential bus or synchronized frequency diverge while power flow remains intact."
            )

        # Case D: Physical Event Consistent with Observed Topology (Natural Fault / Coordinated Trip)
        elif fused_flag == 1 and topology_consistency == ConsistencyStatus.CONSISTENT.value and (
            grid_state.topology_state in ["LINE_1_OUTAGE", "LINE_2_OUTAGE", "BUS_FAULT_COLLAPSE", "ACTIVE_TRANSMISSION_FAULT"]
        ):
            primary_hyp = PrimaryHypothesis.PHYSICAL_OUTAGE_CONSISTENT.value
            possible_causes.extend([
                "Transmission line physical fault (phase-to-ground or tree contact)",
                "Coordinated protective relay clearing action",
                "Scheduled or automatic transmission line isolation",
            ])
            confidence = 0.90
            evidence_statements.append(
                f"Physical anomaly matches recognized grid reconfiguration: {grid_state.topology_state}."
            )

        # Case E: Unexpected Physical Discrepancy (True Deterministic Physical Conservation Violation)
        elif len(inconsistent_phys) > 0:
            primary_hyp = PrimaryHypothesis.UNEXPECTED_PHYSICAL_DISCREPANCY.value
            possible_causes.extend([
                "Deterministic physical conservation law violation (KCL, KVL, frequency, or phase unbalance)",
                "High-impedance fault with physical leakage below trip threshold",
                "Instrument transformer measurement divergence or physical sensor failure",
            ])
            confidence = 0.85
            evidence_statements.append(
                f"Deterministic physical conservation laws violated: {', '.join([r.details for r in inconsistent_phys])}."
            )

        # Case F: Statistical ML Anomaly under Normal Topology
        elif fused_flag == 1 and grid_state.topology_state == "ALL_LINES_IN_SERVICE":
            primary_hyp = PrimaryHypothesis.STATISTICAL_DISTURBANCE_NORMAL_TOPOLOGY.value
            possible_causes.extend([
                "Statistical anomaly detected by ML pipeline without physical conservation law violation",
                "Inter-area power oscillation or dynamic transient swing within operating limits",
                "Synchrophasor phase-angle reference baseline shift across recording runs",
            ])
            confidence = 0.70
            evidence_statements.append(
                f"Evidence Fusion triggered statistical anomaly (FusedScore={fused_score:.4f}), "
                "but all deterministic physical conservation laws and breaker contacts are verified normal."
            )

        # Case G: Insufficient Evidence
        else:
            primary_hyp = PrimaryHypothesis.INSUFFICIENT_EVIDENCE.value
            possible_causes.append("Telemetry ambiguous or incomplete")
            confidence = 0.50
            uncertainties.append("Insufficient data to establish definitive grid operational state.")

        investigation = InvestigationReport(
            possible_causes=possible_causes,
            primary_hypothesis=primary_hyp,
            confidence=confidence,
            uncertainty=uncertainties,
        )

        return physical_analysis, investigation, evidence_statements
