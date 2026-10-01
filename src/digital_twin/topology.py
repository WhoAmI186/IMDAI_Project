"""Grid Topology Module for the Power System Digital Twin.

Loads, validates, and provides structured access to the power grid topology configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPOLOGY_PATH = WORKSPACE_ROOT / "topology" / "grid_topology.json"


class GridTopology:
    """Encapsulates the two-generator, four-bus, two-parallel-line power transmission network."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_TOPOLOGY_PATH
        if not self.config_path.exists():
            raise FileNotFoundError(f"Grid topology configuration not found at: {self.config_path}")

        with open(self.config_path, "r") as fp:
            self.data: Dict[str, Any] = json.load(fp)

        self.system_info: Dict[str, Any] = self.data.get("system_info", {})
        self.substations: Dict[str, Any] = self.data.get("substations", {})
        self.buses: Dict[str, Any] = self.data.get("buses", {})
        self.lines: Dict[str, Any] = self.data.get("transmission_lines", {})
        self.generators: Dict[str, Any] = self.data.get("generators", {})
        self.breakers: Dict[str, Any] = self.data.get("circuit_breakers", {})
        self.relays: Dict[str, Any] = self.data.get("protective_relays", {})
        self.physical_laws: Dict[str, Any] = self.data.get("physical_laws", {})

    def get_bus(self, bus_id: str) -> Optional[Dict[str, Any]]:
        return self.buses.get(bus_id)

    def get_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        return self.lines.get(line_id)

    def get_breaker(self, breaker_id: str) -> Optional[Dict[str, Any]]:
        return self.breakers.get(breaker_id)

    def get_relay(self, relay_id: str) -> Optional[Dict[str, Any]]:
        return self.relays.get(relay_id)

    def get_line_relays(self, line_id: str) -> List[str]:
        line = self.get_line(line_id)
        if not line:
            return []
        return [line.get("sending_relay", ""), line.get("receiving_relay", "")]

    def get_line_breakers(self, line_id: str) -> List[str]:
        line = self.get_line(line_id)
        if not line:
            return []
        return [line.get("sending_breaker", ""), line.get("receiving_breaker", "")]

    def get_bus_relays(self, bus_id: str) -> List[str]:
        bus = self.get_bus(bus_id)
        if not bus:
            return []
        return bus.get("monitoring_relays", [])

    def get_bus_lines(self, bus_id: str) -> List[str]:
        connected = []
        for line_id, line in self.lines.items():
            if line.get("from_bus") == bus_id or line.get("to_bus") == bus_id:
                connected.append(line_id)
        return connected

    def get_all_lines(self) -> List[str]:
        return list(self.lines.keys())

    def get_all_buses(self) -> List[str]:
        return list(self.buses.keys())

    def get_all_breakers(self) -> List[str]:
        return list(self.breakers.keys())

    def get_all_relays(self) -> List[str]:
        return list(self.relays.keys())
