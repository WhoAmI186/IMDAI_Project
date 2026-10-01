"""Deterministic Physical Conservation Checks for the Power System Digital Twin.

Enforces fundamental electrical laws (Kirchhoff's Current Law, Equipotential Bus Voltage,
Frequency Synchronism, and Three-Phase Balance) to identify physical anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.digital_twin.schemas import ConsistencyStatus
from src.digital_twin.telemetry_mapper import RelayTelemetry


@dataclass
class PhysicalCheckResult:
    """Outcome of a single deterministic physical check."""

    check_name: str
    status: ConsistencyStatus
    discrepancy: float
    threshold: float
    details: str
    affected_components: List[str] = field(default_factory=list)


class PhysicalChecker:
    """Executes deterministic physical law verification across synchrophasor measurements."""

    def __init__(
        self,
        bus_voltage_tol_v: float = 600.0,
        line_current_tol_a: float = 35.0,
        freq_spread_tol_hz: float = 0.05,
        three_phase_unbalance_tol_v: float = 1200.0,
    ) -> None:
        self.bus_voltage_tol_v = bus_voltage_tol_v
        self.line_current_tol_a = line_current_tol_a
        self.freq_spread_tol_hz = freq_spread_tol_hz
        self.three_phase_unbalance_tol_v = three_phase_unbalance_tol_v

    def check_bus1_voltage_equipotential(self, relays: Dict[str, RelayTelemetry]) -> PhysicalCheckResult:
        """Verifies Substation 1 Bus 1 voltage agreement between R1 and R4."""
        r1_v = relays.get("R1", RelayTelemetry(relay_id="R1")).voltage_a
        r4_v = relays.get("R4", RelayTelemetry(relay_id="R4")).voltage_a

        if r1_v is None or r4_v is None:
            return PhysicalCheckResult(
                check_name="bus1_voltage_equipotential",
                status=ConsistencyStatus.UNKNOWN,
                discrepancy=0.0,
                threshold=self.bus_voltage_tol_v,
                details="Missing voltage telemetry from R1 or R4 on Bus 1.",
                affected_components=["Bus_1"],
            )

        diff = abs(r1_v - r4_v)
        is_consistent = diff <= self.bus_voltage_tol_v
        status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT
        details = f"Bus 1 Equipotential: R1={r1_v:.1f}V, R4={r4_v:.1f}V, Diff={diff:.1f}V (tol={self.bus_voltage_tol_v:.1f}V)"

        return PhysicalCheckResult(
            check_name="bus1_voltage_equipotential",
            status=status,
            discrepancy=diff,
            threshold=self.bus_voltage_tol_v,
            details=details,
            affected_components=["Bus_1"] if not is_consistent else [],
        )

    def check_bus2_voltage_equipotential(self, relays: Dict[str, RelayTelemetry]) -> PhysicalCheckResult:
        """Verifies Substation 2 Bus 2 voltage agreement between R2 and R3."""
        r2_v = relays.get("R2", RelayTelemetry(relay_id="R2")).voltage_a
        r3_v = relays.get("R3", RelayTelemetry(relay_id="R3")).voltage_a

        if r2_v is None or r3_v is None:
            return PhysicalCheckResult(
                check_name="bus2_voltage_equipotential",
                status=ConsistencyStatus.UNKNOWN,
                discrepancy=0.0,
                threshold=self.bus_voltage_tol_v,
                details="Missing voltage telemetry from R2 or R3 on Bus 2.",
                affected_components=["Bus_2"],
            )

        diff = abs(r2_v - r3_v)
        is_consistent = diff <= self.bus_voltage_tol_v
        status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT
        details = f"Bus 2 Equipotential: R2={r2_v:.1f}V, R3={r3_v:.1f}V, Diff={diff:.1f}V (tol={self.bus_voltage_tol_v:.1f}V)"

        return PhysicalCheckResult(
            check_name="bus2_voltage_equipotential",
            status=status,
            discrepancy=diff,
            threshold=self.bus_voltage_tol_v,
            details=details,
            affected_components=["Bus_2"] if not is_consistent else [],
        )

    def check_line1_current_continuity(self, relays: Dict[str, RelayTelemetry]) -> PhysicalCheckResult:
        """Verifies Line 1 series current continuity (Kirchhoff's Current Law) between R1 and R2."""
        r1_i = relays.get("R1", RelayTelemetry(relay_id="R1")).current_a
        r2_i = relays.get("R2", RelayTelemetry(relay_id="R2")).current_a

        if r1_i is None or r2_i is None:
            return PhysicalCheckResult(
                check_name="line1_current_continuity",
                status=ConsistencyStatus.UNKNOWN,
                discrepancy=0.0,
                threshold=self.line_current_tol_a,
                details="Missing current telemetry from R1 or R2 on Line 1.",
                affected_components=["Line_1"],
            )

        diff = abs(r1_i - r2_i)
        is_consistent = diff <= self.line_current_tol_a
        status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT
        details = f"Line 1 Current Continuity: R1={r1_i:.1f}A, R2={r2_i:.1f}A, Diff={diff:.1f}A (tol={self.line_current_tol_a:.1f}A)"

        return PhysicalCheckResult(
            check_name="line1_current_continuity",
            status=status,
            discrepancy=diff,
            threshold=self.line_current_tol_a,
            details=details,
            affected_components=["Line_1"] if not is_consistent else [],
        )

    def check_line2_current_continuity(self, relays: Dict[str, RelayTelemetry]) -> PhysicalCheckResult:
        """Verifies Line 2 series current continuity (Kirchhoff's Current Law) between R4 and R3."""
        r4_i = relays.get("R4", RelayTelemetry(relay_id="R4")).current_a
        r3_i = relays.get("R3", RelayTelemetry(relay_id="R3")).current_a

        if r4_i is None or r3_i is None:
            return PhysicalCheckResult(
                check_name="line2_current_continuity",
                status=ConsistencyStatus.UNKNOWN,
                discrepancy=0.0,
                threshold=self.line_current_tol_a,
                details="Missing current telemetry from R4 or R3 on Line 2.",
                affected_components=["Line_2"],
            )

        diff = abs(r4_i - r3_i)
        is_consistent = diff <= self.line_current_tol_a
        status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT
        details = f"Line 2 Current Continuity: R4={r4_i:.1f}A, R3={r3_i:.1f}A, Diff={diff:.1f}A (tol={self.line_current_tol_a:.1f}A)"

        return PhysicalCheckResult(
            check_name="line2_current_continuity",
            status=status,
            discrepancy=diff,
            threshold=self.line_current_tol_a,
            details=details,
            affected_components=["Line_2"] if not is_consistent else [],
        )

    def check_frequency_synchronization(self, relays: Dict[str, RelayTelemetry]) -> PhysicalCheckResult:
        """Verifies electrical frequency agreement across all synchronized PMU reporting relays."""
        freqs = [r.frequency for r in relays.values() if r.frequency is not None and not np.isnan(r.frequency)]

        if len(freqs) < 2:
            return PhysicalCheckResult(
                check_name="frequency_synchronization",
                status=ConsistencyStatus.UNKNOWN,
                discrepancy=0.0,
                threshold=self.freq_spread_tol_hz,
                details="Fewer than 2 valid frequency measurements available.",
                affected_components=[],
            )

        spread = float(max(freqs) - min(freqs))
        is_consistent = spread <= self.freq_spread_tol_hz
        status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT
        details = f"Grid Frequency Spread: Min={min(freqs):.4f}Hz, Max={max(freqs):.4f}Hz, Spread={spread:.4f}Hz (tol={self.freq_spread_tol_hz:.4f}Hz)"

        affected = []
        if not is_consistent:
            # Locate outlier relay
            mean_f = float(np.mean(freqs))
            for r_id, r in relays.items():
                if r.frequency is not None and abs(r.frequency - mean_f) > (self.freq_spread_tol_hz / 2.0):
                    affected.append(r_id)

        return PhysicalCheckResult(
            check_name="frequency_synchronization",
            status=status,
            discrepancy=spread,
            threshold=self.freq_spread_tol_hz,
            details=details,
            affected_components=affected,
        )

    def check_three_phase_balance(self, relays: Dict[str, RelayTelemetry]) -> List[PhysicalCheckResult]:
        """Checks three-phase voltage balance across each relay to detect phase-to-ground faults."""
        results = []
        for r_id, r in relays.items():
            if r.voltage_a is None or r.voltage_b is None or r.voltage_c is None:
                continue

            v_diffs = [
                abs(r.voltage_a - r.voltage_b),
                abs(r.voltage_b - r.voltage_c),
                abs(r.voltage_c - r.voltage_a),
            ]
            max_unbalance = max(v_diffs)
            is_consistent = max_unbalance <= self.three_phase_unbalance_tol_v
            status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT
            details = f"{r_id} 3-Phase Voltage Unbalance: MaxDiff={max_unbalance:.1f}V (tol={self.three_phase_unbalance_tol_v:.1f}V)"

            results.append(
                PhysicalCheckResult(
                    check_name=f"{r_id}_three_phase_balance",
                    status=status,
                    discrepancy=max_unbalance,
                    threshold=self.three_phase_unbalance_tol_v,
                    details=details,
                    affected_components=[r_id] if not is_consistent else [],
                )
            )

        return results

    def run_all_checks(self, relays: Dict[str, RelayTelemetry]) -> List[PhysicalCheckResult]:
        """Runs all deterministic physical checks and returns comprehensive findings."""
        checks = [
            self.check_bus1_voltage_equipotential(relays),
            self.check_bus2_voltage_equipotential(relays),
            self.check_line1_current_continuity(relays),
            self.check_line2_current_continuity(relays),
            self.check_frequency_synchronization(relays),
        ]
        checks.extend(self.check_three_phase_balance(relays))
        return checks
