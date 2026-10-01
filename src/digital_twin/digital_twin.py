"""Power System Digital Twin Orchestrator.

The core contextual and topological reasoning engine that consumes ML evidence
and synchrophasor telemetry to reconstruct physical grid states and generate
explainable forensic evidence for downstream RAG + LLM analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.data.triple_loader import TripleScenarioData
from src.digital_twin.event_investigator import EventInvestigator
from src.digital_twin.physical_checks import PhysicalChecker
from src.digital_twin.schemas import (
    AnomalyEvidenceContainer,
    DigitalTwinInput,
    DigitalTwinOutput,
    GridStateReconstruction,
    InvestigationReport,
    PhysicalAnalysisResult,
)
from src.digital_twin.state import StateReconstructor
from src.digital_twin.telemetry_mapper import TelemetryMapper
from src.digital_twin.topology import GridTopology
from src.digital_twin.topology_checks import TopologyChecker
from src.ml.triple_evidence_fusion import TripleFusedEvidence


class PowerSystemDigitalTwin:
    """Topology-aware, deterministic Digital Twin for the two-generator four-bus smart grid."""

    def __init__(
        self,
        topology_path: Optional[Union[str, Path]] = None,
        bus_voltage_tol_v: float = 600.0,
        line_current_tol_a: float = 35.0,
        freq_spread_tol_hz: float = 0.05,
    ) -> None:
        self.topology = GridTopology(topology_path)
        self.telemetry_mapper = TelemetryMapper()
        self.physical_checker = PhysicalChecker(
            bus_voltage_tol_v=bus_voltage_tol_v,
            line_current_tol_a=line_current_tol_a,
            freq_spread_tol_hz=freq_spread_tol_hz,
        )
        self.topology_checker = TopologyChecker(self.topology)
        self.state_reconstructor = StateReconstructor(self.topology, self.topology_checker)
        self.event_investigator = EventInvestigator()

    def reset(self) -> None:
        """Resets temporal contradiction counters and rolling baseline buffers."""
        self.topology_checker.reset()

    def process_event(self, event_input: Union[DigitalTwinInput, Dict[str, Any]]) -> DigitalTwinOutput:
        """Processes a single temporal event window through the Digital Twin pipeline."""
        if isinstance(event_input, dict):
            inp = DigitalTwinInput(
                timestamp=event_input.get("timestamp", "0"),
                telemetry=event_input.get("telemetry", {}),
                layer1=event_input.get("layer1", {}),
                layer2=event_input.get("layer2", {}),
                fusion=event_input.get("fusion", {}),
            )
        else:
            inp = event_input

        # 1. Map Telemetry into Structured Relay/Breaker Objects
        relays = self.telemetry_mapper.map_all_relays(inp.telemetry)

        # 2. Run Deterministic Physical Conservation Checks
        phys_results = self.physical_checker.run_all_checks(relays)

        # 3. Run Topology and Breaker-State Checks
        buses_energized = (
            self.topology_checker.infer_bus_status("Bus_1", relays).value == "ENERGIZED"
            and self.topology_checker.infer_bus_status("Bus_2", relays).value == "ENERGIZED"
        )
        topo_results = [
            self.topology_checker.check_line_breaker_consistency("Line_1", relays, buses_energized),
            self.topology_checker.check_line_breaker_consistency("Line_2", relays, buses_energized),
        ]
        l1_flag = int(inp.layer1.get("anomaly_flag", 0))
        l2_flag = int(inp.layer2.get("aggregated_score", 0.0) >= 1.0)
        uncoord_results = self.topology_checker.check_uncoordinated_trip(relays, l1_flag, l2_flag)
        topo_results.extend(uncoord_results)

        # 4. Reconstruct Grid State and Component Health (freeze baseline if anomaly active)
        fused_flag = int(inp.fusion.get("anomaly_flag", 0))
        freeze_baseline = (fused_flag == 1 or l1_flag == 1 or l2_flag == 1)
        grid_state, component_states = self.state_reconstructor.reconstruct_state(
            relays, phys_results, freeze_baseline=freeze_baseline
        )

        # 5. Synthesize Physical Analysis and Engineering Investigation
        phys_analysis, investigation, evidence_statements = self.event_investigator.investigate(
            grid_state=grid_state,
            component_states=component_states,
            physical_results=phys_results,
            topology_results=topo_results,
            layer1_evidence=inp.layer1,
            layer2_evidence=inp.layer2,
            fusion_evidence=inp.fusion,
        )

        # 6. Package Standardized Digital Twin Output
        anomaly_evidence = {
            "layer1_score": float(inp.layer1.get("anomaly_score", 0.0)),
            "layer2_scores": inp.layer2.get("relationship_scores", {}),
            "fusion_score": float(inp.fusion.get("fused_score", 0.0)),
            "fusion_flag": int(inp.fusion.get("anomaly_flag", 0)),
        }

        return DigitalTwinOutput(
            timestamp=str(inp.timestamp),
            grid_state=grid_state.to_dict(),
            anomaly_evidence=anomaly_evidence,
            physical_analysis=phys_analysis.to_dict(),
            component_states=component_states,
            investigation=investigation.to_dict(),
            evidence=evidence_statements,
        )

    def process_scenario_stream(
        self,
        scenario: TripleScenarioData,
        fused_evidence: TripleFusedEvidence,
        step_indices: Optional[List[int]] = None,
    ) -> List[DigitalTwinOutput]:
        """Processes a sequence of time steps from a pre-evaluated scenario."""
        self.reset()
        outputs: List[DigitalTwinOutput] = []

        total_steps = len(fused_evidence.fused_score)
        indices = step_indices if step_indices is not None else list(range(total_steps))

        # Build feature DataFrame aligned with sequence endpoints
        features_df = scenario.features_df
        end_indices = fused_evidence.end_indices

        for idx in indices:
            if idx >= total_steps:
                continue

            csv_row_idx = end_indices[idx]
            row_telemetry = features_df.iloc[csv_row_idx].to_dict()

            l1_ev = {
                "anomaly_score": float(fused_evidence.layer1_normalized[idx]),
                "anomaly_flag": int(fused_evidence.layer1_flag[idx]),
                "mse": float(fused_evidence.layer1_mse[idx]),
            }
            l2_ev = {
                "aggregated_score": float(fused_evidence.layer2_top2_mean[idx]),
                "mean_score": float(fused_evidence.layer2_mean[idx]),
                "max_score": float(fused_evidence.layer2_max[idx]),
                "relationship_scores": {},
            }
            fusion_ev = {
                "fused_score": float(fused_evidence.fused_score[idx]),
                "anomaly_flag": int(fused_evidence.fused_flag[idx]),
            }

            event_input = DigitalTwinInput(
                timestamp=f"{scenario.filename}_t{idx}_row{csv_row_idx}",
                telemetry=row_telemetry,
                layer1=l1_ev,
                layer2=l2_ev,
                fusion=fusion_ev,
            )

            out = self.process_event(event_input)
            outputs.append(out)

        return outputs
