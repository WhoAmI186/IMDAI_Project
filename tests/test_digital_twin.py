"""End-to-End and Scenario Verification Tests for PowerSystemDigitalTwin."""

import json
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.triple_loader import TripleDataLoader
from src.digital_twin.digital_twin import PowerSystemDigitalTwin
from src.digital_twin.schemas import (
    BreakerStatus,
    ComponentStatus,
    ConsistencyStatus,
    DigitalTwinInput,
    PrimaryHypothesis,
)


@pytest.fixture
def digital_twin():
    return PowerSystemDigitalTwin()


def test_output_schema_completeness(digital_twin):
    """Verifies that DigitalTwinOutput conforms strictly to the specified schema."""
    dummy_input = DigitalTwinInput(
        timestamp="t_001",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131620.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130050.0,
            "R1-PM4:I": 400.0, "R2-PM4:I": 395.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.25, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.30, "bus1_voltage_score": 0.1},
        fusion={"fused_score": 0.275, "anomaly_flag": 0},
    )

    out = digital_twin.process_event(dummy_input)
    d = out.to_dict()

    # Required top-level keys per prompt section 12
    assert "timestamp" in d
    assert "grid_state" in d
    assert "anomaly_evidence" in d
    assert "physical_analysis" in d
    assert "component_states" in d
    assert "investigation" in d
    assert "evidence" in d

    # Grid state keys
    assert "topology_state" in d["grid_state"]
    assert "affected_buses" in d["grid_state"]
    assert "affected_lines" in d["grid_state"]
    assert "affected_relays" in d["grid_state"]

    # Physical analysis keys
    assert "physical_consistency" in d["physical_analysis"]
    assert "measurement_consistency" in d["physical_analysis"]
    assert "topology_consistency" in d["physical_analysis"]

    # Investigation keys
    assert "primary_hypothesis" in d["investigation"]
    assert "possible_causes" in d["investigation"]
    assert "confidence" in d["investigation"]
    assert "uncertainty" in d["investigation"]

    # Must be JSON serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0


def test_case_1_normal_noevents_window(digital_twin):
    """Case 1: Normal steady-state operation."""
    normal_input = DigitalTwinInput(
        timestamp="normal_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131620.0,
            "R2-PM1:V": 130050.0, "R3-PM1:V": 130070.0,
            "R1-PM4:I": 395.0, "R2-PM4:I": 393.0,
            "R4-PM4:I": 392.0, "R3-PM4:I": 390.0,
            "R1:F": 60.000, "R2:F": 60.001, "R3:F": 60.000, "R4:F": 59.999,
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.20, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.25},
        fusion={"fused_score": 0.225, "anomaly_flag": 0},
    )

    out = digital_twin.process_event(normal_input)
    assert out.grid_state["topology_state"] == "ALL_LINES_IN_SERVICE"
    assert out.physical_analysis["physical_consistency"] == "consistent"
    assert out.physical_analysis["topology_consistency"] == "consistent"
    assert out.investigation["primary_hypothesis"] == PrimaryHypothesis.NORMAL_OPERATION.value
    assert out.investigation["confidence"] >= 0.95


def test_case_2_topology_consistent_abnormal_state(digital_twin):
    """Case 2: Physical line outage consistent with reported open breaker position."""
    outage_input = DigitalTwinInput(
        timestamp="outage_01",
        telemetry={
            "R1-PM1:V": 131800.0, "R4-PM1:V": 131810.0,
            "R2-PM1:V": 129500.0, "R3-PM1:V": 129520.0,
            # Line 1 is de-energized: 0 A
            "R1-PM4:I": 0.5, "R2-PM4:I": 0.0,
            # Line 2 carries full current: ~780 A
            "R4-PM4:I": 780.0, "R3-PM4:I": 775.0,
            "R1:F": 59.998, "R2:F": 59.998, "R3:F": 59.998, "R4:F": 59.998,
            # BR1 and BR2 report OPEN (trip_log=1, bit 2048)
            "R1:S": 2048, "R2:S": 2048, "R3:S": 0, "R4:S": 0,
            "relay1_log": 1, "relay2_log": 1, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.95, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.90},
        fusion={"fused_score": 0.925, "anomaly_flag": 1},
    )

    out = digital_twin.process_event(outage_input)
    assert out.grid_state["topology_state"] == "LINE_1_OUTAGE"
    assert "Line_1" in out.grid_state["affected_lines"]
    assert out.component_states["Line_1"] == ComponentStatus.DE_ENERGIZED.value
    assert out.component_states["BR1"] == BreakerStatus.OPEN.value
    assert out.physical_analysis["topology_consistency"] == "consistent"
    assert out.investigation["primary_hypothesis"] == PrimaryHypothesis.PHYSICAL_OUTAGE_CONSISTENT.value


def test_case_3_topology_inconsistent_abnormal_state(digital_twin):
    """Case 3: Line 1 current drops to 0A while breakers report CLOSED and buses are energized."""
    inconsistent_input = DigitalTwinInput(
        timestamp="inconsistent_01",
        telemetry={
            "R1-PM1:V": 131500.0, "R4-PM1:V": 131520.0,
            "R2-PM1:V": 129800.0, "R3-PM1:V": 129810.0,
            # Line 1 is 0A
            "R1-PM4:I": 0.0, "R2-PM4:I": 0.0,
            "R4-PM4:I": 390.0, "R3-PM4:I": 388.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            # Breakers report CLOSED!
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.92, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.88},
        fusion={"fused_score": 0.90, "anomaly_flag": 1},
    )

    # First sample: in debounce holdoff (1/2) -> consistent
    out_1 = digital_twin.process_event(inconsistent_input)
    assert out_1.physical_analysis["topology_consistency"] == "consistent"

    # Second sample: persistent contradiction (2/2) -> asserted inconsistent
    out_2 = digital_twin.process_event(inconsistent_input)
    assert out_2.physical_analysis["topology_consistency"] == "inconsistent"
    assert out_2.investigation["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_TOPOLOGY_INCONSISTENCY.value
    assert any("unexpected de-energization" in ev for ev in out_2.evidence)


def test_case_4_sensor_measurement_issue(digital_twin):
    """Case 4: Severe discrepancy on redundant bus voltage sensors while power flow is undisturbed."""
    sensor_fault_input = DigitalTwinInput(
        timestamp="sensor_issue_01",
        telemetry={
            # Bus 1 equipotential violated: R1 reports 131.6 kV, R4 reports 95 kV!
            "R1-PM1:V": 131600.0, "R4-PM1:V": 95000.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130020.0,
            # Line currents are normal and balanced!
            "R1-PM4:I": 395.0, "R2-PM4:I": 392.0,
            "R4-PM4:I": 393.0, "R3-PM4:I": 390.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.85, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.95},
        fusion={"fused_score": 0.90, "anomaly_flag": 1},
    )

    out = digital_twin.process_event(sensor_fault_input)
    assert out.physical_analysis["measurement_consistency"] == "inconsistent"
    assert out.investigation["primary_hypothesis"] == PrimaryHypothesis.SENSOR_MEASUREMENT_ISSUE.value
    assert "Bus_1" in out.grid_state["affected_buses"]


def test_case_5_missing_unknown_telemetry(digital_twin):
    """Case 5: Telemetry is completely stripped or missing."""
    empty_input = DigitalTwinInput(
        timestamp="empty_01",
        telemetry={},
        layer1={"anomaly_score": 0.0, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.0},
        fusion={"fused_score": 0.0, "anomaly_flag": 0},
    )

    out = digital_twin.process_event(empty_input)
    assert out.grid_state["topology_state"] == "UNKNOWN"
    assert out.component_states["Line_1"] == ComponentStatus.UNKNOWN.value
    assert out.component_states["BR1"] == BreakerStatus.UNKNOWN.value
    assert out.investigation["primary_hypothesis"] == PrimaryHypothesis.INSUFFICIENT_EVIDENCE.value


def test_case_6_uncoordinated_trip_cyber_manipulation(digital_twin):
    """Case 6: Breaker BR1 reported OPEN without electrical fault or overcurrent."""
    cyber_trip_input = DigitalTwinInput(
        timestamp="cyber_trip_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131610.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130010.0,
            # Line 1 is 0A
            "R1-PM4:I": 0.0, "R2-PM4:I": 0.0,
            "R4-PM4:I": 395.0, "R3-PM4:I": 392.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            # BR1 tripped by command (relay1_log=1, R1:S=2048)
            "R1:S": 2048, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 1, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.95, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.20},  # No Kirchhoff violation on voltage/continuity
        fusion={"fused_score": 0.85, "anomaly_flag": 1},
    )

    # First sample: in debounce holdoff (1/2) -> consistent
    out_1 = digital_twin.process_event(cyber_trip_input)
    assert out_1.physical_analysis["topology_consistency"] == "consistent"

    # Second sample: persistent contradiction (2/2) -> asserted inconsistent
    out_2 = digital_twin.process_event(cyber_trip_input)
    assert out_2.physical_analysis["topology_consistency"] == "inconsistent"
    assert any("uncoordinated" in ev.lower() or "bilateral isolation" in ev.lower() for ev in out_2.evidence)


def test_case_7_real_data_smoke_test(digital_twin):
    """Case 7: Evaluates Digital Twin on real rows from data13.csv."""
    loader = TripleDataLoader()
    scenario = loader.load_scenario("data13.csv")

    # Take a normal row (row 50 is within NoEvents)
    normal_row = scenario.features_df.iloc[50].to_dict()
    inp_norm = DigitalTwinInput(
        timestamp="data13_row50",
        telemetry=normal_row,
        layer1={"anomaly_score": 0.3, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.3},
        fusion={"fused_score": 0.3, "anomaly_flag": 0},
    )
    out_norm = digital_twin.process_event(inp_norm)
    assert out_norm.grid_state["topology_state"] == "ALL_LINES_IN_SERVICE"

    # Take an attack row (row 500 is within Attack)
    attack_row = scenario.features_df.iloc[500].to_dict()
    inp_atk = DigitalTwinInput(
        timestamp="data13_row500",
        telemetry=attack_row,
        layer1={"anomaly_score": 1.0, "anomaly_flag": 1},
        layer2={"aggregated_score": 1.0},
        fusion={"fused_score": 1.0, "anomaly_flag": 1},
    )
    out_atk = digital_twin.process_event(inp_atk)
    assert out_atk.anomaly_evidence["fusion_flag"] == 1
    assert len(out_atk.evidence) > 0


# ==============================================================================
# 10 NEW TESTS FOR VALIDATED DIGITAL TWIN IMPROVEMENTS
# ==============================================================================

def test_1_ml_anomaly_physical_checks_pass(digital_twin):
    """1. ML anomaly + physical checks pass -> STATISTICAL_DISTURBANCE_NORMAL_TOPOLOGY."""
    digital_twin.reset()
    inp = DigitalTwinInput(
        timestamp="stat_dist_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131620.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130020.0,
            "R1-PM4:I": 400.0, "R2-PM4:I": 395.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.85, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.80},
        fusion={"fused_score": 0.825, "anomaly_flag": 1},
    )
    out = digital_twin.process_event(inp)
    assert out.physical_analysis["physical_consistency"] == "consistent"
    assert out.physical_analysis["topology_consistency"] == "consistent"
    assert out.grid_state["topology_state"] == "ALL_LINES_IN_SERVICE"
    assert out.investigation["primary_hypothesis"] == PrimaryHypothesis.STATISTICAL_DISTURBANCE_NORMAL_TOPOLOGY.value
    assert out.investigation["confidence"] == 0.70


def test_2_ml_anomaly_physical_violation(digital_twin):
    """2. ML anomaly + physical violation -> UNEXPECTED_PHYSICAL_DISCREPANCY."""
    digital_twin.reset()
    inp = DigitalTwinInput(
        timestamp="phys_violation_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131620.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130020.0,
            # Severe Line 1 Current Continuity violation: R1=450A, R2=300A (Diff=150A > 35A tol)
            "R1-PM4:I": 450.0, "R2-PM4:I": 300.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.90, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.95},
        fusion={"fused_score": 0.925, "anomaly_flag": 1},
    )
    out = digital_twin.process_event(inp)
    assert out.physical_analysis["physical_consistency"] == "inconsistent"
    assert out.investigation["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_PHYSICAL_DISCREPANCY.value
    assert out.investigation["confidence"] == 0.85


def test_3_relative_current_load_transfer_no_fault(digital_twin):
    """3. Relative current >= 2 but balanced voltage/unbalance -> NOT ACTIVE_TRANSMISSION_FAULT."""
    from src.digital_twin.telemetry_mapper import RelayTelemetry
    relays = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=131000.0, voltage_b=131000.0, voltage_c=131000.0, current_a=650.0),
        "R2": RelayTelemetry(relay_id="R2", voltage_a=129500.0, voltage_b=129500.0, voltage_c=129500.0, current_a=648.0),
    }
    # Pre-event baseline was 300A -> ratio = 649 / 300 = 2.16x >= 2.0
    # But unbalance = 0V (<1200V) and voltages are nominal (>115kV)
    status = digital_twin.topology_checker.infer_line_status("Line_1", relays, rolling_baseline=300.0)
    assert status == ComponentStatus.ENERGIZED


def test_4_relative_current_unbalance_fault_eligible(digital_twin):
    """4. Relative current >= 2 + unbalance >= 1200 V -> fault logic eligible (FAULT)."""
    from src.digital_twin.telemetry_mapper import RelayTelemetry
    relays = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=131000.0, voltage_b=110000.0, voltage_c=131000.0, current_a=650.0),
        "R2": RelayTelemetry(relay_id="R2", voltage_a=129500.0, voltage_b=109000.0, voltage_c=129500.0, current_a=648.0),
    }
    # Ratio = 649 / 300 = 2.16x AND unbalance = 21,000V >= 1200V
    status = digital_twin.topology_checker.infer_line_status("Line_1", relays, rolling_baseline=300.0)
    assert status == ComponentStatus.FAULT


def test_5_relative_current_undervoltage_fault_eligible(digital_twin):
    """5. Relative current >= 2 + voltage < 115 kV -> fault logic eligible (FAULT)."""
    from src.digital_twin.telemetry_mapper import RelayTelemetry
    relays = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=105000.0, voltage_b=105000.0, voltage_c=105000.0, current_a=650.0),
        "R2": RelayTelemetry(relay_id="R2", voltage_a=104000.0, voltage_b=104000.0, voltage_c=104000.0, current_a=648.0),
    }
    # Ratio = 649 / 300 = 2.16x AND bus voltage = 105kV < 115kV
    status = digital_twin.topology_checker.infer_line_status("Line_1", relays, rolling_baseline=300.0)
    assert status == ComponentStatus.FAULT


def test_6_one_sample_breaker_contradiction_suppressed(digital_twin):
    """6. One-sample breaker contradiction -> no topology inconsistency assertion (holdoff)."""
    digital_twin.reset()
    inp = DigitalTwinInput(
        timestamp="transient_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131610.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130010.0,
            "R1-PM4:I": 400.0, "R2-PM4:I": 395.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            # Single sample open breaker glitch while current flows
            "R1:S": 2048, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 1, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.20, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.20},
        fusion={"fused_score": 0.20, "anomaly_flag": 0},
    )
    out = digital_twin.process_event(inp)
    assert out.physical_analysis["topology_consistency"] == "consistent"


def test_7_two_sample_persistent_contradiction_asserted(digital_twin):
    """7. Two-sample persistent contradiction -> topology inconsistency assertion."""
    digital_twin.reset()
    inp = DigitalTwinInput(
        timestamp="persistent_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131610.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130010.0,
            "R1-PM4:I": 400.0, "R2-PM4:I": 395.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            # Open breaker with active current flowing
            "R1:S": 2048, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 1, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.85, "anomaly_flag": 1},
        layer2={"aggregated_score": 0.20},
        fusion={"fused_score": 0.80, "anomaly_flag": 1},
    )
    # First sample: in holdoff
    out1 = digital_twin.process_event(inp)
    assert out1.physical_analysis["topology_consistency"] == "consistent"

    # Second consecutive sample: asserted inconsistent
    out2 = digital_twin.process_event(inp)
    assert out2.physical_analysis["topology_consistency"] == "inconsistent"
    assert out2.investigation["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_TOPOLOGY_INCONSISTENCY.value


def test_8_counter_resets_after_contradiction_disappears(digital_twin):
    """8. Counter resets after contradiction disappears."""
    digital_twin.reset()
    contra_inp = DigitalTwinInput(
        timestamp="contra_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131610.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130010.0,
            "R1-PM4:I": 400.0, "R2-PM4:I": 395.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            "R1:S": 2048, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 1, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.20, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.20},
        fusion={"fused_score": 0.20, "anomaly_flag": 0},
    )
    normal_inp = DigitalTwinInput(
        timestamp="normal_01",
        telemetry={
            "R1-PM1:V": 131600.0, "R4-PM1:V": 131610.0,
            "R2-PM1:V": 130000.0, "R3-PM1:V": 130010.0,
            "R1-PM4:I": 400.0, "R2-PM4:I": 395.0,
            "R4-PM4:I": 398.0, "R3-PM4:I": 394.0,
            "R1:F": 60.00, "R2:F": 60.00, "R3:F": 60.00, "R4:F": 60.00,
            "R1:S": 0, "R2:S": 0, "R3:S": 0, "R4:S": 0,
            "relay1_log": 0, "relay2_log": 0, "relay3_log": 0, "relay4_log": 0,
        },
        layer1={"anomaly_score": 0.20, "anomaly_flag": 0},
        layer2={"aggregated_score": 0.20},
        fusion={"fused_score": 0.20, "anomaly_flag": 0},
    )
    # Step 1: Contradiction sample 1 -> counter becomes 1, consistent
    out1 = digital_twin.process_event(contra_inp)
    assert out1.physical_analysis["topology_consistency"] == "consistent"

    # Step 2: Normal sample -> counter resets to 0
    out2 = digital_twin.process_event(normal_inp)
    assert out2.physical_analysis["topology_consistency"] == "consistent"
    assert digital_twin.topology_checker.contradiction_counters["BR1_trip_coordination"] == 0

    # Step 3: Contradiction sample again -> counter becomes 1 (holdoff again, not asserted yet)
    out3 = digital_twin.process_event(contra_inp)
    assert out3.physical_analysis["topology_consistency"] == "consistent"


def test_9_counters_independent_between_lines_breakers(digital_twin):
    """9. Counters are independent between breakers/lines."""
    digital_twin.reset()
    tc = digital_twin.topology_checker
    tc.contradiction_counters["Line_1_breaker_consistency"] = 1
    assert tc.contradiction_counters["Line_2_breaker_consistency"] == 0
    assert tc.contradiction_counters["BR4_trip_coordination"] == 0


def test_10_baseline_does_not_include_current_anomalous_sample(digital_twin):
    """10. Baseline does not include current anomalous sample."""
    from src.digital_twin.telemetry_mapper import RelayTelemetry
    tc = digital_twin.topology_checker
    tc.reset()

    # Feed 15 normal observations of 300A
    for _ in range(15):
        relays = {
            "R1": RelayTelemetry(relay_id="R1", voltage_a=131000.0, current_a=300.0),
            "R2": RelayTelemetry(relay_id="R2", voltage_a=130000.0, current_a=300.0),
        }
        tc.infer_line_status("Line_1", relays)

    assert len(tc.baseline_buffers["Line_1"]) == 15
    assert np.median(tc.baseline_buffers["Line_1"]) == 300.0

    # Feed anomalous sample of 1200A (fault) with freeze_baseline=True
    relays_fault = {
        "R1": RelayTelemetry(relay_id="R1", voltage_a=100000.0, current_a=1200.0),
        "R2": RelayTelemetry(relay_id="R2", voltage_a=100000.0, current_a=1200.0),
    }
    status = tc.infer_line_status("Line_1", relays_fault, freeze_baseline=True)
    assert status == ComponentStatus.FAULT

    # Verify 1200A was NOT added to the baseline buffer and baseline is still 300A!
    assert 1200.0 not in tc.baseline_buffers["Line_1"]
    assert np.median(tc.baseline_buffers["Line_1"]) == 300.0

