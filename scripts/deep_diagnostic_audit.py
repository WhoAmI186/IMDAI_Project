"""Forensic Deep Diagnostic Audit of the Power System Digital Twin.

Executes a comprehensive, read-only diagnostic tracing across data13.csv, data14.csv, data15.csv.
Captures every internal signal, physical check result, topology check result, and decision pathway.
Zero production logic, models, thresholds, or topologies are modified.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.triple_loader import TripleDataLoader, TripleScenarioData
from src.digital_twin.digital_twin import PowerSystemDigitalTwin
from src.digital_twin.schemas import (
    BreakerStatus,
    ComponentStatus,
    ConsistencyStatus,
    DigitalTwinInput,
    PrimaryHypothesis,
)
from src.ml.triple_evidence_fusion import TripleEvidenceFusion
from src.ml.triple_layer1_detector import TripleLayer1Detector
from src.ml.triple_layer2_regressors import TripleLayer2Regressors

REPORTS_DIR = WORKSPACE_ROOT / "reports"
MODELS_DIR = WORKSPACE_ROOT / "models"

print("=" * 80)
print("STARTING COMPREHENSIVE DIGITAL TWIN READ-ONLY DIAGNOSTIC AUDIT")
print("=" * 80)

# Load frozen ML models
l1_det = TripleLayer1Detector.load()
l2_reg = TripleLayer2Regressors.load()
fusion = TripleEvidenceFusion(layer1_detector=l1_det, layer2_regressors=l2_reg)

loader = TripleDataLoader()
train_scenarios = loader.load_partition(loader.TRAIN_FILES)
th_fused_p99 = fusion.calibrate_normal_fused_threshold(train_scenarios, percentile=99.0)

# Initialize frozen Digital Twin
digital_twin = PowerSystemDigitalTwin()

test_scenarios = loader.load_partition(loader.TEST_FILES)

# Data containers for sequence-level forensic records
all_records: List[Dict[str, Any]] = []

for scen in test_scenarios:
    print(f"\nProcessing {scen.filename} through ML Pipeline + Digital Twin...")
    ev = fusion.process_scenario(scen, fused_threshold_override=th_fused_p99)

    n_seqs = len(ev.fused_score)
    labels = ev.ground_truth_labels
    features_df = scen.features_df
    end_indices = ev.end_indices

    # Identify transition boundaries in this scenario
    # In MSU/ORNL dataset, marker changes mark event transitions
    marker_changes = np.where(labels[:-1] != labels[1:])[0] + 1
    # Boundary transitions: index where new event block begins

    for i in range(n_seqs):
        csv_row = int(end_indices[i])
        m_cls = str(labels[i])
        row_telemetry = features_df.iloc[csv_row].to_dict()

        l1_ev = {
            "anomaly_score": float(ev.layer1_normalized[i]),
            "anomaly_flag": int(ev.layer1_flag[i]),
            "mse": float(ev.layer1_mse[i]),
        }
        l2_ev = {
            "aggregated_score": float(ev.layer2_top2_mean[i]),
            "mean_score": float(ev.layer2_mean[i]),
            "max_score": float(ev.layer2_max[i]),
            "relationship_scores": {},
        }
        fusion_ev = {
            "fused_score": float(ev.fused_score[i]),
            "anomaly_flag": int(ev.fused_flag[i]),
        }

        event_input = DigitalTwinInput(
            timestamp=f"{scen.filename}_t{i}_row{csv_row}",
            telemetry=row_telemetry,
            layer1=l1_ev,
            layer2=l2_ev,
            fusion=fusion_ev,
        )

        # 1. Map telemetry
        relays = digital_twin.telemetry_mapper.map_all_relays(row_telemetry)

        # 2. Run deterministic physical checks
        phys_results = digital_twin.physical_checker.run_all_checks(relays)

        # 3. Run topology checks
        buses_energized = (
            digital_twin.topology_checker.infer_bus_status("Bus_1", relays).value == "ENERGIZED"
            and digital_twin.topology_checker.infer_bus_status("Bus_2", relays).value == "ENERGIZED"
        )
        topo_results = [
            digital_twin.topology_checker.check_line_breaker_consistency("Line_1", relays, buses_energized),
            digital_twin.topology_checker.check_line_breaker_consistency("Line_2", relays, buses_energized),
        ]
        uncoord_results = digital_twin.topology_checker.check_uncoordinated_trip(
            relays, l1_ev["anomaly_flag"], int(l2_ev["aggregated_score"] >= 1.0)
        )
        topo_results.extend(uncoord_results)

        # 4. State reconstruction
        grid_state, component_states = digital_twin.state_reconstructor.reconstruct_state(relays, phys_results)

        # 5. Event investigation
        phys_analysis, investigation, evidence_statements = digital_twin.event_investigator.investigate(
            grid_state=grid_state,
            component_states=component_states,
            physical_results=phys_results,
            topology_results=topo_results,
            layer1_evidence=l1_ev,
            layer2_evidence=l2_ev,
            fusion_evidence=fusion_ev,
        )

        # Detailed breakdown of which individual physical checks failed
        failed_checks = [r.check_name for r in phys_results if r.status == ConsistencyStatus.INCONSISTENT]
        unknown_checks = [r.check_name for r in phys_results if r.status == ConsistencyStatus.UNKNOWN]

        # Detailed breakdown of which topology checks failed
        failed_topo = [r.check_name for r in topo_results if r.status == ConsistencyStatus.INCONSISTENT]

        # Determine transition status
        # A transition window is defined as within 60 steps following any marker change
        # (the length of the sliding window buffer, during which the causal window contains mixed data)
        dist_to_transitions = [i - t for t in marker_changes if 0 <= (i - t) < 60]
        is_transition = len(dist_to_transitions) > 0
        transition_phase = "TRANSITION_WINDOW" if is_transition else "STEADY_STATE"

        record = {
            "scenario": scen.filename,
            "step": i,
            "csv_row": csv_row,
            "marker": m_cls,  # strictly for post-hoc grouping
            "layer1_score": l1_ev["anomaly_score"],
            "layer1_flag": l1_ev["anomaly_flag"],
            "layer1_mse": l1_ev["mse"],
            "layer2_score": l2_ev["aggregated_score"],
            "layer2_flag": int(l2_ev["aggregated_score"] >= 1.0),
            "fused_score": fusion_ev["fused_score"],
            "fused_flag": fusion_ev["anomaly_flag"],
            "topology_state": grid_state.topology_state,
            "affected_buses": grid_state.affected_buses,
            "affected_lines": grid_state.affected_lines,
            "affected_relays": grid_state.affected_relays,
            "component_states": component_states,
            "physical_consistency": phys_analysis.physical_consistency,
            "measurement_consistency": phys_analysis.measurement_consistency,
            "topology_consistency": phys_analysis.topology_consistency,
            "primary_hypothesis": investigation.primary_hypothesis,
            "confidence": investigation.confidence,
            "failed_physical_checks": failed_checks,
            "num_failed_physical_checks": len(failed_checks),
            "failed_topology_checks": failed_topo,
            "num_failed_topology_checks": len(failed_topo),
            "transition_phase": transition_phase,
            "evidence": evidence_statements,
        }
        all_records.append(record)

df_all = pd.DataFrame(all_records)
print(f"\nTotal evaluated test sequences: {len(df_all)}")
print("Marker distribution:", df_all["marker"].value_counts().to_dict())

# ==============================================================================
# PHASE 1 & 2: REPRODUCE OVERALL INTERPRETATION DISTRIBUTION
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 2: VERIFY REPRODUCED DIGITAL TWIN INTERPRETATION DISTRIBUTION")
print("=" * 80)

summary_dist = pd.crosstab(df_all["primary_hypothesis"], df_all["marker"], normalize="columns") * 100
print(summary_dist.round(2))

# ==============================================================================
# PHASE 3: PHYSICAL-LAW DISCREPANCY ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 3: DETAILED PHYSICAL-LAW DISCREPANCY DECONSTRUCTION")
print("=" * 80)

# Filter samples categorized as "Unexpected physical law discrepancy without clear topology explanation"
pld_df = df_all[df_all["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_PHYSICAL_DISCREPANCY.value]
print(f"Total Physical-Law Discrepancy Samples: {len(pld_df)}")
print("By Marker Class:", pld_df["marker"].value_counts().to_dict())

# Check how many failed ANY physical check vs ZERO physical checks (only fused_flag == 1)
for m_cls in ["NoEvents", "Natural", "Attack"]:
    sub = pld_df[pld_df["marker"] == m_cls]
    n_tot = len(sub)
    if n_tot == 0:
        continue
    zero_checks = (sub["num_failed_physical_checks"] == 0).sum()
    at_least_one = (sub["num_failed_physical_checks"] > 0).sum()
    print(f"\n--- {m_cls} Physical-Law Discrepancy (N={n_tot}) ---")
    print(f"  Zero deterministic physical checks failed (flagged solely by ML fused_flag=1): {zero_checks} ({zero_checks/n_tot*100:.2f}%)")
    print(f"  At least one deterministic physical check failed:                              {at_least_one} ({at_least_one/n_tot*100:.2f}%)")

# Detailed check-by-check failure rates among Physical-Law Discrepancy samples
all_check_names = [
    "bus1_voltage_equipotential",
    "bus2_voltage_equipotential",
    "line1_current_continuity",
    "line2_current_continuity",
    "frequency_synchronization",
    "three_phase_balance",
]

check_breakdown_rows = []
for chk in all_check_names:
    row = {"Check Name": chk}
    for m_cls in ["NoEvents", "Natural", "Attack"]:
        sub = pld_df[pld_df["marker"] == m_cls]
        n_tot = len(sub)
        if chk == "three_phase_balance":
            count = sub["failed_physical_checks"].apply(lambda lst: any("three_phase_balance" in c for c in lst)).sum()
        else:
            count = sub["failed_physical_checks"].apply(lambda lst: chk in lst).sum()
        pct = (count / n_tot * 100.0) if n_tot > 0 else 0.0
        row[m_cls] = f"{count} ({pct:.2f}%)"
    check_breakdown_rows.append(row)

# Also check "Solely ML Triggered" (Zero deterministic physical checks failed)
solely_ml_row = {"Check Name": "Solely ML Triggered (0 physical checks failed)"}
for m_cls in ["NoEvents", "Natural", "Attack"]:
    sub = pld_df[pld_df["marker"] == m_cls]
    n_tot = len(sub)
    count = (sub["num_failed_physical_checks"] == 0).sum()
    pct = (count / n_tot * 100.0) if n_tot > 0 else 0.0
    solely_ml_row[m_cls] = f"{count} ({pct:.2f}%)"
check_breakdown_rows.append(solely_ml_row)

df_check_breakdown = pd.DataFrame(check_breakdown_rows)
print("\nFailure breakdown within Physical-Law Discrepancy category:")
print(df_check_breakdown.to_string(index=False))

# Multiple simultaneous failures
multi_rows = []
for n_fails in [0, 1, 2, 3, 4, "5+"]:
    row = {"Number of Failed Checks": str(n_fails)}
    for m_cls in ["NoEvents", "Natural", "Attack"]:
        sub = pld_df[pld_df["marker"] == m_cls]
        n_tot = len(sub)
        if n_fails == "5+":
            c = (sub["num_failed_physical_checks"] >= 5).sum()
        else:
            c = (sub["num_failed_physical_checks"] == n_fails).sum()
        pct = (c / n_tot * 100.0) if n_tot > 0 else 0.0
        row[m_cls] = f"{c} ({pct:.2f}%)"
    multi_rows.append(row)
print("\nDistribution of simultaneous check failures:")
print(pd.DataFrame(multi_rows).to_string(index=False))

# ==============================================================================
# PHASE 4: TRANSITION VS STEADY-STATE ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 4: TRANSITION VS STEADY-STATE ANALYSIS")
print("=" * 80)

# Check what proportion of Physical-Law Discrepancies fall into Transition Windows (within 60 steps of a marker change)
trans_summary = pd.crosstab(
    pld_df["marker"],
    pld_df["transition_phase"],
    normalize="index"
) * 100
print("Percentage of Physical-Law Discrepancies during Transition vs Steady State:")
print(trans_summary.round(2))

# Also across ALL samples by hypothesis
all_trans_summary = pd.crosstab(
    [df_all["marker"], df_all["transition_phase"]],
    df_all["primary_hypothesis"],
    normalize="index"
) * 100
print("\nHypothesis distribution during Transition vs Steady State (Percentage):")
print(all_trans_summary.round(2))

# ==============================================================================
# PHASE 5: TOPOLOGY INCONSISTENCY ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 5: TOPOLOGY INCONSISTENCY ANALYSIS (Natural: 2.94%, Attack: 5.77%)")
print("=" * 80)

topo_incon_df = df_all[df_all["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_TOPOLOGY_INCONSISTENCY.value]
print(f"Total Topology Inconsistency Samples: {len(topo_incon_df)}")
print("By Marker Class:", topo_incon_df["marker"].value_counts().to_dict())

# What checks caused topology inconsistency?
for m_cls in ["Natural", "Attack"]:
    sub = topo_incon_df[topo_incon_df["marker"] == m_cls]
    n_tot = len(sub)
    print(f"\n--- {m_cls} Topology Inconsistency (N={n_tot}) ---")
    all_failed_topo_checks = [chk for sublist in sub["failed_topology_checks"] for chk in sublist]
    print("Failed Topology Checks:", pd.Series(all_failed_topo_checks).value_counts().to_dict())

    # Check component states in these samples
    sample_cases = sub.head(3)
    for idx, r in sample_cases.iterrows():
        print(f"  Sample {r['scenario']} step {r['step']} (csv_row {r['csv_row']}):")
        print(f"    Line_1={r['component_states']['Line_1']}, Line_2={r['component_states']['Line_2']}")
        print(f"    BR1={r['component_states']['BR1']}, BR2={r['component_states']['BR2']}, BR3={r['component_states']['BR3']}, BR4={r['component_states']['BR4']}")
        print(f"    Evidence: {r['evidence']}")

# ==============================================================================
# PHASE 6: NATURAL "PHYSICAL EVENT CONSISTENT WITH TOPOLOGY"
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 6: NATURAL EVENT 'PHYSICAL EVENT CONSISTENT WITH TOPOLOGY' ANALYSIS (9.02%)")
print("=" * 80)

nat_df = df_all[df_all["marker"] == "Natural"]
nat_consistent_df = nat_df[nat_df["primary_hypothesis"] == PrimaryHypothesis.PHYSICAL_OUTAGE_CONSISTENT.value]
nat_other_df = nat_df[nat_df["primary_hypothesis"] != PrimaryHypothesis.PHYSICAL_OUTAGE_CONSISTENT.value]

print(f"Natural Consistent Outages: {len(nat_consistent_df)} / {len(nat_df)} ({len(nat_consistent_df)/len(nat_df)*100:.2f}%)")
print("Inferred Topology States for Consistent Natural Events:")
print(nat_consistent_df["topology_state"].value_counts().to_dict())

print("\nWhy did the remaining Natural events NOT reach 'Physical event consistent with topology'?")
print("Remaining Natural Events Inferred Topology States:")
print(nat_other_df["topology_state"].value_counts().to_dict())
print("Remaining Natural Events Breaker States (BR1):", nat_other_df["component_states"].apply(lambda d: d.get("BR1")).value_counts().to_dict())
print("Remaining Natural Events Breaker States (BR2):", nat_other_df["component_states"].apply(lambda d: d.get("BR2")).value_counts().to_dict())

# ==============================================================================
# PHASE 7: NORMAL FALSE-POSITIVE ANALYSIS (14.70%)
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 7: NORMAL FALSE-POSITIVE ANALYSIS (14.70% of Normal)")
print("=" * 80)

norm_df = df_all[df_all["marker"] == "NoEvents"]
norm_fp_df = norm_df[norm_df["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_PHYSICAL_DISCREPANCY.value]
norm_clean_df = norm_df[norm_df["primary_hypothesis"] == PrimaryHypothesis.NORMAL_OPERATION.value]

print(f"Total Normal Samples: {len(norm_df)}")
print(f"Normal False Positives (Physical-Law Discrepancy): {len(norm_fp_df)} ({len(norm_fp_df)/len(norm_df)*100:.2f}%)")
print("By Scenario File:")
print(norm_fp_df["scenario"].value_counts().to_dict())

print("\nDid any deterministic physical check fail on these 91 Normal False Positives?")
fp_failed_checks = [chk for sublist in norm_fp_df["failed_physical_checks"] for chk in sublist]
print("Failed Deterministic Physical Checks on Normal False Positives:", pd.Series(fp_failed_checks).value_counts().to_dict())
print("Fused score on Normal False Positives: Mean =", norm_fp_df["fused_score"].mean(), "Min =", norm_fp_df["fused_score"].min(), "Max =", norm_fp_df["fused_score"].max())
print("Layer 1 MSE on Normal False Positives: Mean =", norm_fp_df["layer1_mse"].mean(), "vs Clean Normal Mean =", norm_clean_df["layer1_mse"].mean())

# Timestamp / row clustering of Normal False Positives
for scen in norm_fp_df["scenario"].unique():
    scen_fp = norm_fp_df[norm_fp_df["scenario"] == scen]
    print(f"  {scen}: steps {scen_fp['step'].min()} to {scen_fp['step'].max()} (CSV rows {scen_fp['csv_row'].min()} to {scen_fp['csv_row'].max()})")

# ==============================================================================
# PHASE 8: REPRESENTATIVE CASE STUDIES (5 Normal, 5 Natural, 5 Attack)
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 8: EXTRACTING REPRESENTATIVE CASE STUDIES")
print("=" * 80)

case_studies = {"Normal": [], "Natural": [], "Attack": []}

# 1. Normal Cases:
# Case N1: Clean steady-state in data13
# Case N2: Normal False Positive in data13 (initial phase drift)
# Case N3: Clean steady-state in data14
# Case N4: Clean steady-state in data15
# Case N5: Normal False Positive in data15
c_n1 = norm_clean_df[norm_clean_df["scenario"] == "data13.csv"].iloc[10]
c_n2 = norm_fp_df[norm_fp_df["scenario"] == "data13.csv"].iloc[5]
c_n3 = norm_clean_df[norm_clean_df["scenario"] == "data14.csv"].iloc[5]
c_n4 = norm_clean_df[norm_clean_df["scenario"] == "data15.csv"].iloc[20]
c_n5 = norm_fp_df[norm_fp_df["scenario"] == "data15.csv"].iloc[10]
case_studies["Normal"] = [c_n1.to_dict(), c_n2.to_dict(), c_n3.to_dict(), c_n4.to_dict(), c_n5.to_dict()]

# 2. Natural Cases:
# Case Nat1: Coordinated line outage (Physical event consistent with topology)
# Case Nat2: Active transmission fault (severe overcurrent)
# Case Nat3: Dynamic transient physical discrepancy
# Case Nat4: Topology inconsistency (asymmetric trip during fault)
# Case Nat5: Possible sensor issue during fault
c_nat1 = nat_consistent_df.iloc[0]
c_nat2 = nat_df[nat_df["topology_state"] == "ACTIVE_TRANSMISSION_FAULT"].iloc[0]
c_nat3 = nat_df[nat_df["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_PHYSICAL_DISCREPANCY.value].iloc[10]
c_nat4 = nat_df[nat_df["primary_hypothesis"] == PrimaryHypothesis.UNEXPECTED_TOPOLOGY_INCONSISTENCY.value].iloc[0]
c_nat5 = nat_df[nat_df["primary_hypothesis"] == PrimaryHypothesis.SENSOR_MEASUREMENT_ISSUE.value].iloc[0]
case_studies["Natural"] = [c_nat1.to_dict(), c_nat2.to_dict(), c_nat3.to_dict(), c_nat4.to_dict(), c_nat5.to_dict()]

# 3. Attack Cases:
# Case Atk1: Topology inconsistency (uncoordinated single-ended trip)
# Case Atk2: Topology inconsistency (de-energized line with closed breakers)
# Case Atk3: Sensor measurement issue (FDI on voltage/frequency)
# Case Atk4: Physical discrepancy during steady-state attack
# Case Atk5: Coordinated line outage resulting from attack trip
c_atk1 = topo_incon_df[topo_incon_df["marker"] == "Attack"].iloc[0]
c_atk2 = topo_incon_df[topo_incon_df["marker"] == "Attack"].iloc[min(50, len(topo_incon_df[topo_incon_df["marker"] == "Attack"]) - 1)]
c_atk3 = df_all[(df_all["marker"] == "Attack") & (df_all["primary_hypothesis"] == PrimaryHypothesis.SENSOR_MEASUREMENT_ISSUE.value)].iloc[0]
c_atk4 = pld_df[pld_df["marker"] == "Attack"].iloc[20]
c_atk5 = df_all[(df_all["marker"] == "Attack") & (df_all["primary_hypothesis"] == PrimaryHypothesis.PHYSICAL_OUTAGE_CONSISTENT.value)].iloc[0]
case_studies["Attack"] = [c_atk1.to_dict(), c_atk2.to_dict(), c_atk3.to_dict(), c_atk4.to_dict(), c_atk5.to_dict()]

print(f"Extracted {len(case_studies['Normal'])} Normal, {len(case_studies['Natural'])} Natural, {len(case_studies['Attack'])} Attack cases.")

# ==============================================================================
# PHASE 9: CONFUSION & SEPARATION ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 9: CONFUSION & SEPARATION ANALYSIS")
print("=" * 80)

# Calculate precision/specificity of each DT interpretation for predicting Natural vs Attack vs Normal
separation_table = []
for hyp in df_all["primary_hypothesis"].unique():
    sub = df_all[df_all["primary_hypothesis"] == hyp]
    total_hyp = len(sub)
    n_norm = (sub["marker"] == "NoEvents").sum()
    n_nat = (sub["marker"] == "Natural").sum()
    n_atk = (sub["marker"] == "Attack").sum()

    separation_table.append({
        "Primary Hypothesis": hyp,
        "Total Count": total_hyp,
        "% Normal (Clean)": round(n_norm / total_hyp * 100, 2),
        "% Natural (Fault)": round(n_nat / total_hyp * 100, 2),
        "% Attack (Cyber)": round(n_atk / total_hyp * 100, 2),
        "Dominant Group": "Normal" if n_norm > max(n_nat, n_atk) else ("Natural" if n_nat > n_atk else "Attack"),
    })

df_sep = pd.DataFrame(separation_table).sort_values(by="Total Count", ascending=False)
print(df_sep.to_string(index=False))

# Export machine-readable artifacts
# 1. Full CSV analysis for detailed inspection
df_export = df_all[[
    "scenario", "step", "csv_row", "marker",
    "fused_score", "fused_flag", "layer1_score", "layer2_score",
    "topology_state", "physical_consistency", "measurement_consistency", "topology_consistency",
    "primary_hypothesis", "confidence", "num_failed_physical_checks", "num_failed_topology_checks",
    "transition_phase"
]]
csv_out_path = REPORTS_DIR / "digital_twin_event_analysis.csv"
df_export.to_csv(csv_out_path, index=False)
print(f"\nSaved event analysis CSV to {csv_out_path}")

# 2. Complete diagnostic JSON artifact
diagnostic_json = {
    "audit_title": "Forensic Read-Only Diagnostic Audit of Power System Digital Twin",
    "test_files": loader.TEST_FILES,
    "total_evaluated_sequences": len(df_all),
    "interpretation_distribution_percentage": summary_dist.to_dict(),
    "physical_law_discrepancy_analysis": {
        "total_pld_samples": len(pld_df),
        "check_failure_breakdown": df_check_breakdown.to_dict(orient="records"),
        "simultaneous_failures": df_check_breakdown.to_dict(orient="records"),
    },
    "transition_analysis": {
        "pld_in_transition_pct": trans_summary.to_dict(),
    },
    "topology_inconsistency_analysis": {
        "total_samples": len(topo_incon_df),
        "by_class": topo_incon_df["marker"].value_counts().to_dict(),
    },
    "natural_event_analysis": {
        "consistent_outages_count": len(nat_consistent_df),
        "consistent_outages_pct": len(nat_consistent_df) / len(nat_df) * 100.0,
        "topology_states": nat_df["topology_state"].value_counts().to_dict(),
    },
    "normal_false_positive_analysis": {
        "total_normal_fps": len(norm_fp_df),
        "by_scenario": norm_fp_df["scenario"].value_counts().to_dict(),
        "failed_physical_checks": pd.Series(fp_failed_checks).value_counts().to_dict(),
    },
    "separation_analysis": separation_table,
    "representative_case_studies": case_studies,
}

def json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json_out_path = REPORTS_DIR / "digital_twin_diagnostic.json"
with open(json_out_path, "w") as fp:
    json.dump(diagnostic_json, fp, indent=2, default=json_default)

print(f"Saved diagnostic JSON to {json_out_path}")
print("=" * 80)
print("DIAGNOSTIC AUDIT COMPLETED SUCCESSFULLY.")
print("=" * 80)
