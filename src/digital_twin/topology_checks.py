"""Topology-Aware State and Breaker Verification for the Power System Digital Twin.

Correlates breaker contact states, line current flows, and bus energization to
detect topology changes, unexpected open circuits, and unauthorized trip commands.
Incorporates composite relative overcurrent logic and temporal persistence debouncing.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.digital_twin.schemas import BreakerStatus, ComponentStatus, ConsistencyStatus
from src.digital_twin.telemetry_mapper import RelayTelemetry
from src.digital_twin.topology import GridTopology


@dataclass
class TopologyCheckResult:
    """Outcome of a topology-state consistency evaluation."""

    check_name: str
    status: ConsistencyStatus
    component_id: str
    details: str
    evidence_statement: str


class TopologyChecker:
    """Evaluates whether observed electrical flows match reported circuit breaker and topology states."""

    def __init__(
        self,
        topology: Optional[GridTopology] = None,
        deenergized_current_a: float = 30.0,
        fault_current_a: float = 800.0,
        bus_undervoltage_v: float = 90000.0,
        current_noise_floor: float = 100.0,
        relative_current_ratio: float = 2.0,
        unbalance_gate: float = 1200.0,
        undervoltage_gate: float = 115000.0,
        baseline_window_samples: int = 30,
        debounce_samples: int = 2,
    ) -> None:
        self.topology = topology or GridTopology()
        self.deenergized_current_a = deenergized_current_a
        self.fault_current_a = fault_current_a
        self.bus_undervoltage_v = bus_undervoltage_v
        self.current_noise_floor = current_noise_floor
        self.relative_current_ratio = relative_current_ratio
        self.unbalance_gate = unbalance_gate
        self.undervoltage_gate = undervoltage_gate
        self.baseline_window_samples = baseline_window_samples
        self.debounce_samples = debounce_samples

        self.contradiction_counters: Dict[str, int] = defaultdict(int)
        self.baseline_buffers: Dict[str, deque] = {
            "Line_1": deque(maxlen=self.baseline_window_samples),
            "Line_2": deque(maxlen=self.baseline_window_samples),
        }
        self.is_frozen: Dict[str, bool] = {"Line_1": False, "Line_2": False}

    def reset(self) -> None:
        """Resets all temporal contradiction counters and rolling baseline buffers."""
        self.contradiction_counters.clear()
        self.baseline_buffers = {
            "Line_1": deque(maxlen=self.baseline_window_samples),
            "Line_2": deque(maxlen=self.baseline_window_samples),
        }
        self.is_frozen = {"Line_1": False, "Line_2": False}

    def infer_line_status(
        self,
        line_id: str,
        relays: Dict[str, RelayTelemetry],
        rolling_baseline: Optional[float] = None,
        freeze_baseline: bool = False,
    ) -> ComponentStatus:
        """Infers electrical status (ENERGIZED, DE_ENERGIZED, FAULT, UNKNOWN) of a transmission line.

        Incorporates absolute thresholds as well as composite relative overcurrent logic:
        relative_ratio >= 2.0 AND (three_phase_unbalance >= 1200 V OR bus_voltage < 115000 V).
        """
        line_info = self.topology.get_line(line_id)
        if not line_info:
            return ComponentStatus.UNKNOWN

        r_send = relays.get(line_info["sending_relay"])
        r_recv = relays.get(line_info["receiving_relay"])

        currents = []
        if r_send and r_send.current_a is not None:
            currents.append(r_send.current_a)
        if r_recv and r_recv.current_a is not None:
            currents.append(r_recv.current_a)

        if not currents:
            return ComponentStatus.UNKNOWN

        mean_current = float(sum(currents) / len(currents))

        # Determine effective baseline safely from prior valid samples (do not include current sample)
        if rolling_baseline is not None:
            effective_baseline = rolling_baseline
        elif line_id in self.baseline_buffers and len(self.baseline_buffers[line_id]) >= 10:
            effective_baseline = float(np.median(self.baseline_buffers[line_id]))
        else:
            effective_baseline = None

        # 1. Absolute gross fault current threshold
        if mean_current >= self.fault_current_a:
            status = ComponentStatus.FAULT
        # 2. Composite relative overcurrent threshold
        elif (
            effective_baseline is not None
            and effective_baseline > 10.0
            and mean_current >= self.current_noise_floor
            and (mean_current / effective_baseline) >= self.relative_current_ratio
        ):
            # Check composite conditions: three-phase unbalance OR bus undervoltage
            unbalance_detected = False
            undervoltage_detected = False

            # Check voltage unbalance on sending/receiving relays
            for r in [r_send, r_recv]:
                if r and r.voltage_a is not None and r.voltage_b is not None and r.voltage_c is not None:
                    diff = max(r.voltage_a, r.voltage_b, r.voltage_c) - min(r.voltage_a, r.voltage_b, r.voltage_c)
                    if diff >= self.unbalance_gate:
                        unbalance_detected = True
                        break

            # Check bus undervoltage
            for r in [r_send, r_recv]:
                if r and r.voltage_a is not None and r.voltage_a < self.undervoltage_gate:
                    undervoltage_detected = True
                    break

            if unbalance_detected or undervoltage_detected:
                status = ComponentStatus.FAULT
            else:
                # High current due to load transfer without unbalance or undervoltage
                status = ComponentStatus.ENERGIZED
        elif mean_current <= self.deenergized_current_a:
            status = ComponentStatus.DE_ENERGIZED
        else:
            status = ComponentStatus.ENERGIZED

        # Baseline safety update: freeze baseline on anomaly/fault, do not contaminate with anomalous current
        if freeze_baseline or status == ComponentStatus.FAULT or self.is_frozen.get(line_id, False):
            if status == ComponentStatus.FAULT:
                self.is_frozen[line_id] = True
        else:
            if line_id in self.baseline_buffers and mean_current > 10.0:
                self.baseline_buffers[line_id].append(mean_current)

        return status

    def infer_bus_status(self, bus_id: str, relays: Dict[str, RelayTelemetry]) -> ComponentStatus:
        """Infers electrical status of a substation busbar."""
        bus_info = self.topology.get_bus(bus_id)
        if not bus_info or bus_info.get("state_status") == "UNKNOWN":
            return ComponentStatus.UNKNOWN

        monitored_relays = bus_info.get("monitoring_relays", [])
        voltages = []
        for r_id in monitored_relays:
            r = relays.get(r_id)
            if r and r.voltage_a is not None:
                voltages.append(r.voltage_a)

        if not voltages:
            return ComponentStatus.UNKNOWN

        mean_voltage = float(sum(voltages) / len(voltages))

        if mean_voltage < self.bus_undervoltage_v:
            return ComponentStatus.FAULT
        elif mean_voltage >= 115000.0:
            return ComponentStatus.ENERGIZED
        else:
            return ComponentStatus.ABNORMAL

    def check_line_breaker_consistency(
        self,
        line_id: str,
        relays: Dict[str, RelayTelemetry],
        buses_energized: bool,
    ) -> TopologyCheckResult:
        """Verifies consistency between circuit breaker positions and actual line current flow with temporal debounce."""
        line_info = self.topology.get_line(line_id)
        if not line_info:
            return TopologyCheckResult(
                check_name=f"{line_id}_breaker_consistency",
                status=ConsistencyStatus.UNKNOWN,
                component_id=line_id,
                details="Line topology definition not found.",
                evidence_statement=f"Topology record for {line_id} is missing.",
            )

        r_send = relays.get(line_info["sending_relay"])
        r_recv = relays.get(line_info["receiving_relay"])
        line_status = self.infer_line_status(line_id, relays)

        br_send_state = r_send.breaker_status if r_send else BreakerStatus.UNKNOWN
        br_recv_state = r_recv.breaker_status if r_recv else BreakerStatus.UNKNOWN
        check_id = f"{line_id}_breaker_consistency"

        # Case 1: Missing breaker state information
        if br_send_state == BreakerStatus.UNKNOWN or br_recv_state == BreakerStatus.UNKNOWN:
            self.contradiction_counters[check_id] = 0
            return TopologyCheckResult(
                check_name=check_id,
                status=ConsistencyStatus.UNKNOWN,
                component_id=line_id,
                details=f"Breaker states for {line_id} cannot be determined reliably from telemetry.",
                evidence_statement=f"{line_id}: Breaker position telemetry is UNKNOWN.",
            )

        # Case 2: Line is de-energized (I < 30A)
        if line_status == ComponentStatus.DE_ENERGIZED:
            if br_send_state == BreakerStatus.OPEN or br_recv_state == BreakerStatus.OPEN:
                self.contradiction_counters[check_id] = 0
                return TopologyCheckResult(
                    check_name=check_id,
                    status=ConsistencyStatus.CONSISTENT,
                    component_id=line_id,
                    details=f"{line_id} is de-energized and breaker reports OPEN (BR_send={br_send_state.value}, BR_recv={br_recv_state.value}).",
                    evidence_statement=f"{line_id} outage is consistent with reported open breaker position.",
                )
            elif buses_energized and br_send_state == BreakerStatus.CLOSED and br_recv_state == BreakerStatus.CLOSED:
                # Breakers report closed, buses are energized, but current is 0!
                self.contradiction_counters[check_id] += 1
                if self.contradiction_counters[check_id] >= self.debounce_samples:
                    return TopologyCheckResult(
                        check_name=check_id,
                        status=ConsistencyStatus.INCONSISTENT,
                        component_id=line_id,
                        details=f"{line_id} current is 0A despite reported CLOSED breakers and energized buses.",
                        evidence_statement=f"{line_id} unexpected de-energization: breakers indicate CLOSED while power flow is 0A (possible false breaker telemetry or physical open phase).",
                    )
                else:
                    return TopologyCheckResult(
                        check_name=check_id,
                        status=ConsistencyStatus.CONSISTENT,
                        component_id=line_id,
                        details=f"{line_id} transient de-energization under debounce holdoff (sample {self.contradiction_counters[check_id]}/{self.debounce_samples}).",
                        evidence_statement=f"{line_id} breaker consistency in transient holdoff.",
                    )

        # Case 3: Line is energized (I > 30A)
        if line_status == ComponentStatus.ENERGIZED or line_status == ComponentStatus.FAULT:
            if br_send_state == BreakerStatus.OPEN or br_recv_state == BreakerStatus.OPEN:
                # Breakers report open, but line current is actively flowing!
                self.contradiction_counters[check_id] += 1
                if self.contradiction_counters[check_id] >= self.debounce_samples:
                    return TopologyCheckResult(
                        check_name=check_id,
                        status=ConsistencyStatus.INCONSISTENT,
                        component_id=line_id,
                        details=f"{line_id} carries active current while breaker reports OPEN.",
                        evidence_statement=f"{line_id} inconsistency: active power flow observed despite reported OPEN breaker position (possible spoofed breaker status).",
                    )
                else:
                    return TopologyCheckResult(
                        check_name=check_id,
                        status=ConsistencyStatus.CONSISTENT,
                        component_id=line_id,
                        details=f"{line_id} transient active flow with open breaker under debounce holdoff (sample {self.contradiction_counters[check_id]}/{self.debounce_samples}).",
                        evidence_statement=f"{line_id} breaker consistency in transient holdoff.",
                    )
            else:
                self.contradiction_counters[check_id] = 0
                return TopologyCheckResult(
                    check_name=check_id,
                    status=ConsistencyStatus.CONSISTENT,
                    component_id=line_id,
                    details=f"{line_id} is energized with reported CLOSED breakers.",
                    evidence_statement=f"{line_id} energized state is consistent with closed breakers.",
                )

        self.contradiction_counters[check_id] = 0
        return TopologyCheckResult(
            check_name=check_id,
            status=ConsistencyStatus.UNKNOWN,
            component_id=line_id,
            details=f"Indeterminate state combination for {line_id}.",
            evidence_statement=f"{line_id}: Indeterminate topology status.",
        )

    def check_uncoordinated_trip(
        self,
        relays: Dict[str, RelayTelemetry],
        layer1_flag: int,
        layer2_flag: int,
    ) -> List[TopologyCheckResult]:
        """Checks whether any breaker tripped without a preceding physical fault or overcurrent with temporal debounce."""
        results = []

        for line_id in ["Line_1", "Line_2"]:
            line_info = self.topology.get_line(line_id)
            if not line_info:
                continue

            r_send = relays.get(line_info["sending_relay"])
            r_recv = relays.get(line_info["receiving_relay"])

            br_send_name = line_info["sending_breaker"]
            br_recv_name = line_info["receiving_breaker"]

            br_send_open = (r_send.breaker_status == BreakerStatus.OPEN) if r_send else False
            br_recv_open = (r_recv.breaker_status == BreakerStatus.OPEN) if r_recv else False
            line_status = self.infer_line_status(line_id, relays)

            # Case A: Both breakers opened together on a de-energized line (Coordinated line outage)
            if br_send_open and br_recv_open and line_status == ComponentStatus.DE_ENERGIZED:
                self.contradiction_counters[f"{br_send_name}_trip_coordination"] = 0
                self.contradiction_counters[f"{br_recv_name}_trip_coordination"] = 0
                results.append(
                    TopologyCheckResult(
                        check_name=f"{line_id}_trip_coordination",
                        status=ConsistencyStatus.CONSISTENT,
                        component_id=line_id,
                        details=f"{line_id} both terminal breakers ({br_send_name}, {br_recv_name}) opened with line de-energized.",
                        evidence_statement=f"{line_id} coordinated line isolation: both terminal breakers open.",
                    )
                )
            # Case B: Single-ended trip (one open, one closed) without fault
            elif (br_send_open != br_recv_open):
                open_br = br_send_name if br_send_open else br_recv_name
                closed_br = br_recv_name if br_send_open else br_send_name
                check_id = f"{open_br}_trip_coordination"
                self.contradiction_counters[check_id] += 1
                self.contradiction_counters[f"{closed_br}_trip_coordination"] = 0

                if self.contradiction_counters[check_id] >= self.debounce_samples:
                    results.append(
                        TopologyCheckResult(
                            check_name=check_id,
                            status=ConsistencyStatus.INCONSISTENT,
                            component_id=open_br,
                            details=f"{open_br} opened asymmetrically while the opposite line terminal breaker remained closed.",
                            evidence_statement=f"{open_br} uncoordinated single-ended trip: breaker opened without bilateral isolation (possible cyber command or single relay trip).",
                        )
                    )
                else:
                    results.append(
                        TopologyCheckResult(
                            check_name=check_id,
                            status=ConsistencyStatus.CONSISTENT,
                            component_id=open_br,
                            details=f"{open_br} single-ended trip under debounce holdoff (sample {self.contradiction_counters[check_id]}/{self.debounce_samples}).",
                            evidence_statement=f"{open_br} trip coordination in transient holdoff.",
                        )
                    )
            # Case C: Breaker open while line carries current or no fault evident
            elif (br_send_open or br_recv_open) and line_status != ComponentStatus.DE_ENERGIZED:
                open_br = br_send_name if br_send_open else br_recv_name
                check_id = f"{open_br}_trip_coordination"
                self.contradiction_counters[check_id] += 1

                if self.contradiction_counters[check_id] >= self.debounce_samples:
                    results.append(
                        TopologyCheckResult(
                            check_name=check_id,
                            status=ConsistencyStatus.INCONSISTENT,
                            component_id=open_br,
                            details=f"{open_br} reported OPEN while line is carrying active current.",
                            evidence_statement=f"{open_br} inconsistent trip: breaker indicates OPEN while current flows.",
                        )
                    )
                else:
                    results.append(
                        TopologyCheckResult(
                            check_name=check_id,
                            status=ConsistencyStatus.CONSISTENT,
                            component_id=open_br,
                            details=f"{open_br} trip under debounce holdoff (sample {self.contradiction_counters[check_id]}/{self.debounce_samples}).",
                            evidence_statement=f"{open_br} trip coordination in transient holdoff.",
                        )
                    )
            else:
                self.contradiction_counters[f"{br_send_name}_trip_coordination"] = 0
                self.contradiction_counters[f"{br_recv_name}_trip_coordination"] = 0

        return results
