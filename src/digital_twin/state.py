"""Grid State Reconstruction Engine for the Power System Digital Twin.

Reconstructs macro grid topology state, localized component health, and affected assets
from multi-relay synchrophasor telemetry and physical checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.digital_twin.physical_checks import PhysicalCheckResult
from src.digital_twin.schemas import BreakerStatus, ComponentStatus, GridStateReconstruction
from src.digital_twin.telemetry_mapper import RelayTelemetry
from src.digital_twin.topology import GridTopology
from src.digital_twin.topology_checks import TopologyChecker


class StateReconstructor:
    """Synthesizes physical telemetry and topology checks to reconstruct power grid state."""

    def __init__(
        self,
        topology: Optional[GridTopology] = None,
        topology_checker: Optional[TopologyChecker] = None,
    ) -> None:
        self.topology = topology or GridTopology()
        self.topology_checker = topology_checker or TopologyChecker(self.topology)

    def reconstruct_state(
        self,
        relays: Dict[str, RelayTelemetry],
        physical_results: List[PhysicalCheckResult],
        rolling_baselines: Optional[Dict[str, float]] = None,
        freeze_baseline: bool = False,
    ) -> Tuple[GridStateReconstruction, Dict[str, str]]:
        """Reconstructs overall topology state, identifies affected assets, and produces component status dictionary."""
        component_states: Dict[str, str] = {}
        affected_buses: List[str] = []
        affected_lines: List[str] = []
        affected_relays: List[str] = []

        # 1. Infer Line States with composite relative current and baseline awareness
        l1_base = rolling_baselines.get("Line_1") if rolling_baselines else None
        l2_base = rolling_baselines.get("Line_2") if rolling_baselines else None
        l1_status = self.topology_checker.infer_line_status("Line_1", relays, rolling_baseline=l1_base, freeze_baseline=freeze_baseline)
        l2_status = self.topology_checker.infer_line_status("Line_2", relays, rolling_baseline=l2_base, freeze_baseline=freeze_baseline)
        component_states["Line_1"] = l1_status.value
        component_states["Line_2"] = l2_status.value

        if l1_status in [ComponentStatus.DE_ENERGIZED, ComponentStatus.FAULT, ComponentStatus.ABNORMAL]:
            affected_lines.append("Line_1")
        if l2_status in [ComponentStatus.DE_ENERGIZED, ComponentStatus.FAULT, ComponentStatus.ABNORMAL]:
            affected_lines.append("Line_2")

        # 2. Infer Bus States
        b1_status = self.topology_checker.infer_bus_status("Bus_1", relays)
        b2_status = self.topology_checker.infer_bus_status("Bus_2", relays)
        component_states["Bus_1"] = b1_status.value
        component_states["Bus_2"] = b2_status.value

        if b1_status in [ComponentStatus.FAULT, ComponentStatus.ABNORMAL]:
            affected_buses.append("Bus_1")
        if b2_status in [ComponentStatus.FAULT, ComponentStatus.ABNORMAL]:
            affected_buses.append("Bus_2")

        # 3. Infer Breaker States
        for br_id, br_info in self.topology.breakers.items():
            ctrl_relay = br_info.get("controlling_relay")
            r_data = relays.get(ctrl_relay) if ctrl_relay else None
            br_state = r_data.breaker_status if r_data else BreakerStatus.UNKNOWN
            component_states[br_id] = br_state.value
            if br_state == BreakerStatus.OPEN and ctrl_relay:
                if ctrl_relay not in affected_relays:
                    affected_relays.append(ctrl_relay)

        # 4. Integrate Inconsistencies from Physical Checks
        for res in physical_results:
            if res.status == "inconsistent":
                for comp in res.affected_components:
                    if comp.startswith("Bus") and comp not in affected_buses:
                        affected_buses.append(comp)
                    elif comp.startswith("Line") and comp not in affected_lines:
                        affected_lines.append(comp)
                    elif comp.startswith("R") and comp not in affected_relays:
                        affected_relays.append(comp)

        # 5. Synthesize Macro Topology State
        if l1_status == ComponentStatus.UNKNOWN or l2_status == ComponentStatus.UNKNOWN:
            topology_state = "UNKNOWN"
        elif b1_status == ComponentStatus.FAULT or b2_status == ComponentStatus.FAULT:
            topology_state = "BUS_FAULT_COLLAPSE"
        elif l1_status == ComponentStatus.ENERGIZED and l2_status == ComponentStatus.ENERGIZED:
            topology_state = "ALL_LINES_IN_SERVICE"
        elif l1_status == ComponentStatus.DE_ENERGIZED and l2_status == ComponentStatus.ENERGIZED:
            topology_state = "LINE_1_OUTAGE"
        elif l1_status == ComponentStatus.ENERGIZED and l2_status == ComponentStatus.DE_ENERGIZED:
            topology_state = "LINE_2_OUTAGE"
        elif l1_status == ComponentStatus.DE_ENERGIZED and l2_status == ComponentStatus.DE_ENERGIZED:
            topology_state = "BOTH_LINES_OUTAGE_ISLANDED"
        elif l1_status == ComponentStatus.FAULT or l2_status == ComponentStatus.FAULT:
            topology_state = "ACTIVE_TRANSMISSION_FAULT"
        else:
            topology_state = "ABNORMAL_TOPOLOGY_STATE"

        reconstruction = GridStateReconstruction(
            topology_state=topology_state,
            affected_buses=affected_buses,
            affected_lines=affected_lines,
            affected_relays=affected_relays,
        )

        return reconstruction, component_states
