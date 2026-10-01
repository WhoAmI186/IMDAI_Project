"""Unit tests for deterministic physical conservation checks."""

import pytest

from src.digital_twin.physical_checks import PhysicalChecker
from src.digital_twin.schemas import ConsistencyStatus
from src.digital_twin.telemetry_mapper import RelayTelemetry


def test_bus1_equipotential_consistent():
    checker = PhysicalChecker(bus_voltage_tol_v=600.0)
    relays = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=131500.0),
        "R4": RelayTelemetry(relay_id="R4", voltage_a=131550.0),
    }
    res = checker.check_bus1_voltage_equipotential(relays)
    assert res.status == ConsistencyStatus.CONSISTENT
    assert res.discrepancy == 50.0
    assert len(res.affected_components) == 0


def test_bus1_equipotential_inconsistent():
    checker = PhysicalChecker(bus_voltage_tol_v=600.0)
    relays = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=131500.0),
        "R4": RelayTelemetry(relay_id="R4", voltage_a=129000.0),  # 2500 V difference
    }
    res = checker.check_bus1_voltage_equipotential(relays)
    assert res.status == ConsistencyStatus.INCONSISTENT
    assert res.discrepancy == 2500.0
    assert "Bus_1" in res.affected_components


def test_line1_continuity_consistent():
    checker = PhysicalChecker(line_current_tol_a=35.0)
    relays = {
        "R1": RelayTelemetry(relay_id="R1", current_a=400.0),
        "R2": RelayTelemetry(relay_id="R2", current_a=405.0),
    }
    res = checker.check_line1_current_continuity(relays)
    assert res.status == ConsistencyStatus.CONSISTENT
    assert res.discrepancy == 5.0


def test_line1_continuity_inconsistent():
    checker = PhysicalChecker(line_current_tol_a=35.0)
    relays = {
        "R1": RelayTelemetry(relay_id="R1", current_a=400.0),
        "R2": RelayTelemetry(relay_id="R2", current_a=100.0),  # 300 A difference
    }
    res = checker.check_line1_current_continuity(relays)
    assert res.status == ConsistencyStatus.INCONSISTENT
    assert res.discrepancy == 300.0
    assert "Line_1" in res.affected_components


def test_frequency_synchronization():
    checker = PhysicalChecker(freq_spread_tol_hz=0.05)
    # Consistent
    relays_ok = {
        "R1": RelayTelemetry(relay_id="R1", frequency=60.001),
        "R2": RelayTelemetry(relay_id="R2", frequency=59.998),
        "R3": RelayTelemetry(relay_id="R3", frequency=60.003),
        "R4": RelayTelemetry(relay_id="R4", frequency=60.000),
    }
    res_ok = checker.check_frequency_synchronization(relays_ok)
    assert res_ok.status == ConsistencyStatus.CONSISTENT

    # Inconsistent (R2 frequency spoofed to 60.2 Hz)
    relays_bad = {
        "R1": RelayTelemetry(relay_id="R1", frequency=60.001),
        "R2": RelayTelemetry(relay_id="R2", frequency=60.200),
        "R3": RelayTelemetry(relay_id="R3", frequency=60.003),
        "R4": RelayTelemetry(relay_id="R4", frequency=60.000),
    }
    res_bad = checker.check_frequency_synchronization(relays_bad)
    assert res_bad.status == ConsistencyStatus.INCONSISTENT
    assert "R2" in res_bad.affected_components


def test_missing_telemetry_returns_unknown():
    checker = PhysicalChecker()
    relays = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=131000.0),
        # R4 is missing
    }
    res = checker.check_bus1_voltage_equipotential(relays)
    assert res.status == ConsistencyStatus.UNKNOWN
