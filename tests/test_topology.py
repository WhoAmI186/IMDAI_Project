"""Unit tests for GridTopology loading and querying."""

from pathlib import Path
import pytest

from src.digital_twin.topology import GridTopology


def test_topology_loading():
    topology = GridTopology()
    assert topology.system_info["nominal_voltage_kv"] == 138.0
    assert len(topology.buses) == 4
    assert len(topology.lines) == 2
    assert len(topology.breakers) == 4
    assert len(topology.relays) == 4


def test_bus_queries():
    topology = GridTopology()
    b1 = topology.get_bus("Bus_1")
    assert b1 is not None
    assert b1["substation"] == "Substation_1"
    assert "R1" in b1["monitoring_relays"]
    assert "R4" in b1["monitoring_relays"]

    b3 = topology.get_bus("Bus_3")
    assert b3 is not None
    assert b3["state_status"] == "UNKNOWN"


def test_line_queries():
    topology = GridTopology()
    l1 = topology.get_line("Line_1")
    assert l1 is not None
    assert l1["from_bus"] == "Bus_1"
    assert l1["to_bus"] == "Bus_2"
    assert l1["sending_relay"] == "R1"
    assert l1["receiving_relay"] == "R2"

    relays_l1 = topology.get_line_relays("Line_1")
    assert relays_l1 == ["R1", "R2"]

    breakers_l1 = topology.get_line_breakers("Line_1")
    assert breakers_l1 == ["BR1", "BR2"]


def test_breaker_queries():
    topology = GridTopology()
    br1 = topology.get_breaker("BR1")
    assert br1 is not None
    assert br1["controlling_relay"] == "R1"
    assert br1["line"] == "Line_1"
    assert br1["trip_bitmask"] == 2048


def test_relay_queries():
    topology = GridTopology()
    r1 = topology.get_relay("R1")
    assert r1 is not None
    assert r1["protected_line"] == "Line_1"
    assert r1["monitored_bus"] == "Bus_1"
