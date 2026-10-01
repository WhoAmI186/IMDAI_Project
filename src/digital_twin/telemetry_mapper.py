"""Telemetry Mapper Module for the Power System Digital Twin.

Extracts, normalizes, and groups raw synchrophasor measurements into structured
physical quantities and relay/breaker status indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.digital_twin.schemas import BreakerStatus


@dataclass
class RelayTelemetry:
    """Grouped physical measurements and status bits for a single protection relay."""

    relay_id: str
    voltage_a: Optional[float] = None
    voltage_b: Optional[float] = None
    voltage_c: Optional[float] = None
    voltage_angle_a: Optional[float] = None
    voltage_angle_b: Optional[float] = None
    voltage_angle_c: Optional[float] = None
    current_a: Optional[float] = None
    current_b: Optional[float] = None
    current_c: Optional[float] = None
    current_angle_a: Optional[float] = None
    current_angle_b: Optional[float] = None
    current_angle_c: Optional[float] = None
    frequency: Optional[float] = None
    rocof: Optional[float] = None
    impedance_angle: Optional[float] = None
    impedance_magnitude: Optional[float] = None
    status_word: Optional[int] = None
    trip_log: Optional[int] = None
    breaker_status: BreakerStatus = BreakerStatus.UNKNOWN


class TelemetryMapper:
    """Parses raw telemetry dictionaries into structured power system observations."""

    @staticmethod
    def extract_relay_telemetry(relay_id: str, raw_data: Dict[str, Any]) -> RelayTelemetry:
        """Extracts all physical channels and status indicators for a given relay."""
        prefix = f"{relay_id}-"

        # Voltages
        v_a = raw_data.get(f"{prefix}PM1:V")
        v_b = raw_data.get(f"{prefix}PM2:V")
        v_c = raw_data.get(f"{prefix}PM3:V")
        va_ang = raw_data.get(f"{prefix}PA1:VH")
        vb_ang = raw_data.get(f"{prefix}PA2:VH")
        vc_ang = raw_data.get(f"{prefix}PA3:VH")

        # Currents
        i_a = raw_data.get(f"{prefix}PM4:I")
        i_b = raw_data.get(f"{prefix}PM5:I")
        i_c = raw_data.get(f"{prefix}PM6:I")
        ia_ang = raw_data.get(f"{prefix}PA4:IH")
        ib_ang = raw_data.get(f"{prefix}PA5:IH")
        ic_ang = raw_data.get(f"{prefix}PA6:IH")

        # System Frequency & ROCOF
        freq = raw_data.get(f"{relay_id}:F")
        rocof = raw_data.get(f"{relay_id}:DF")

        # Impedance
        z_ang = raw_data.get(f"{prefix}PA:ZH")
        z_mag = raw_data.get(f"{prefix}PA:Z")
        # Handle inf in impedance magnitude
        if z_mag is not None and (np.isinf(z_mag) or np.isnan(z_mag)):
            z_mag = None

        # Relay status word and log
        raw_s = raw_data.get(f"{relay_id}:S")
        s_word: Optional[int] = None
        if raw_s is not None:
            try:
                s_word = int(raw_s)
            except (ValueError, TypeError):
                s_word = None

        relay_num = relay_id.replace("R", "")
        raw_trip = raw_data.get(f"relay{relay_num}_log")
        trip_log: Optional[int] = None
        if raw_trip is not None:
            try:
                trip_log = int(raw_trip)
            except (ValueError, TypeError):
                trip_log = None

        # Determine breaker status from indicators
        breaker_state = BreakerStatus.UNKNOWN
        if trip_log is not None or s_word is not None:
            # Check trip log: 1 indicates open/trip commanded
            trip_active = (trip_log == 1) if trip_log is not None else False
            # Check bit 11 (2048) in status word: breaker open / trip bit in SEL-421
            bit_open = ((s_word & 2048) != 0) if s_word is not None else False

            if trip_active or bit_open:
                breaker_state = BreakerStatus.OPEN
            elif (trip_log == 0 or trip_log is None) and (s_word == 0 or s_word is None):
                breaker_state = BreakerStatus.CLOSED

        return RelayTelemetry(
            relay_id=relay_id,
            voltage_a=float(v_a) if v_a is not None else None,
            voltage_b=float(v_b) if v_b is not None else None,
            voltage_c=float(v_c) if v_c is not None else None,
            voltage_angle_a=float(va_ang) if va_ang is not None else None,
            voltage_angle_b=float(vb_ang) if vb_ang is not None else None,
            voltage_angle_c=float(vc_ang) if vc_ang is not None else None,
            current_a=float(i_a) if i_a is not None else None,
            current_b=float(i_b) if i_b is not None else None,
            current_c=float(i_c) if i_c is not None else None,
            current_angle_a=float(ia_ang) if ia_ang is not None else None,
            current_angle_b=float(ib_ang) if ib_ang is not None else None,
            current_angle_c=float(ic_ang) if ic_ang is not None else None,
            frequency=float(freq) if freq is not None else None,
            rocof=float(rocof) if rocof is not None else None,
            impedance_angle=float(z_ang) if z_ang is not None else None,
            impedance_magnitude=float(z_mag) if z_mag is not None else None,
            status_word=s_word,
            trip_log=trip_log,
            breaker_status=breaker_state,
        )

    @classmethod
    def map_all_relays(cls, raw_data: Dict[str, Any]) -> Dict[str, RelayTelemetry]:
        """Maps all four standard relays R1..R4."""
        return {r: cls.extract_relay_telemetry(r, raw_data) for r in ["R1", "R2", "R3", "R4"]}
