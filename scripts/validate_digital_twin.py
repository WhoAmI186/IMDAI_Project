"""Comprehensive Validation Script for PowerSystemDigitalTwin on Held-Out Test Data."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.triple_loader import TripleDataLoader
from src.digital_twin.digital_twin import PowerSystemDigitalTwin
from src.digital_twin.schemas import PrimaryHypothesis
from src.ml.triple_evidence_fusion import TripleEvidenceFusion
from src.ml.triple_layer1_detector import TripleLayer1Detector
from src.ml.triple_layer2_regressors import TripleLayer2Regressors

REPORTS_DIR = WORKSPACE_ROOT / "reports"
MODELS_DIR = WORKSPACE_ROOT / "models"

print("=" * 80)
print("STARTING DIGITAL TWIN VALIDATION ON HELD-OUT TEST BENCHMARK")
print("=" * 80)

# Initialize frozen ML pipeline and Digital Twin
l1_det = TripleLayer1Detector.load()
l2_reg = TripleLayer2Regressors.load()
fusion = TripleEvidenceFusion(layer1_detector=l1_det, layer2_regressors=l2_reg)

loader = TripleDataLoader()
train_scenarios = loader.load_partition(loader.TRAIN_FILES)
th_fused_p99 = fusion.calibrate_normal_fused_threshold(train_scenarios, percentile=99.0)
print(f"Calibrated Normal Fused Threshold: {th_fused_p99:.4f}")

digital_twin = PowerSystemDigitalTwin()

test_scenarios = loader.load_partition(loader.TEST_FILES)

validation_results = {}
all_hypothesis_by_class = {"NoEvents": {}, "Natural": {}, "Attack": {}}
all_topology_by_class = {"NoEvents": {}, "Natural": {}, "Attack": {}}
all_physical_consistency_by_class = {"NoEvents": {}, "Natural": {}, "Attack": {}}
all_topology_consistency_by_class = {"NoEvents": {}, "Natural": {}, "Attack": {}}

sample_event_records = []

for scen in test_scenarios:
    print(f"\nProcessing {scen.filename} through ML Pipeline + Digital Twin...")
    ev = fusion.process_scenario(scen, fused_threshold_override=th_fused_p99)

    # Process all evaluated sequences through Digital Twin
    n_seqs = len(ev.fused_score)
    labels = ev.ground_truth_labels

    dt_outputs = digital_twin.process_scenario_stream(scen, ev)

    file_hyp_counts = {"NoEvents": {}, "Natural": {}, "Attack": {}}
    file_topo_counts = {"NoEvents": {}, "Natural": {}, "Attack": {}}

    for i, out in enumerate(dt_outputs):
        m_cls = labels[i]
        hyp = out.investigation["primary_hypothesis"]
        topo = out.grid_state["topology_state"]
        p_cons = out.physical_analysis["physical_consistency"]
        t_cons = out.physical_analysis["topology_consistency"]

        # Track per class
        all_hypothesis_by_class[m_cls][hyp] = all_hypothesis_by_class[m_cls].get(hyp, 0) + 1
        all_topology_by_class[m_cls][topo] = all_topology_by_class[m_cls].get(topo, 0) + 1
        all_physical_consistency_by_class[m_cls][p_cons] = all_physical_consistency_by_class[m_cls].get(p_cons, 0) + 1
        all_topology_consistency_by_class[m_cls][t_cons] = all_topology_consistency_by_class[m_cls].get(t_cons, 0) + 1

        file_hyp_counts[m_cls][hyp] = file_hyp_counts[m_cls].get(hyp, 0) + 1
        file_topo_counts[m_cls][topo] = file_topo_counts[m_cls].get(topo, 0) + 1

        # Capture sample records for documentation
        if len(sample_event_records) < 15 and i % 1000 == 50:
            sample_event_records.append({
                "scenario": scen.filename,
                "step": i,
                "csv_row": int(ev.end_indices[i]),
                "ground_truth_marker": m_cls,
                "fused_score": float(ev.fused_score[i]),
                "fused_flag": int(ev.fused_flag[i]),
                "digital_twin_output": out.to_dict(),
            })

    validation_results[scen.filename] = {
        "total_sequences": n_seqs,
        "hypotheses_by_class": file_hyp_counts,
        "topology_states_by_class": file_topo_counts,
    }

print("\n" + "=" * 80)
print("AGGREGATED DIGITAL TWIN INVESTIGATION RESULTS ACROSS HELD-OUT TEST DATA")
print("=" * 80)

for m_cls in ["NoEvents", "Natural", "Attack"]:
    total_m = sum(all_hypothesis_by_class[m_cls].values())
    print(f"\n--- Ground-Truth Class: {m_cls} (Total N={total_m}) ---")
    print("  Primary Hypotheses:")
    for hyp, count in sorted(all_hypothesis_by_class[m_cls].items(), key=lambda x: x[1], reverse=True):
        print(f"    {hyp:70s}: {count:5d} ({count/total_m*100:5.2f}%)")
    print("  Topology Consistency:")
    for tc, count in all_topology_consistency_by_class[m_cls].items():
        print(f"    {tc:20s}: {count:5d} ({count/total_m*100:5.2f}%)")
    print("  Physical Consistency:")
    for pc, count in all_physical_consistency_by_class[m_cls].items():
        print(f"    {pc:20s}: {count:5d} ({count/total_m*100:5.2f}%)")
    print("  Top Inferred Topology States:")
    for ts, count in sorted(all_topology_by_class[m_cls].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {ts:35s}: {count:5d} ({count/total_m*100:5.2f}%)")

# Save machine-readable validation artifact
validation_artifact = {
    "title": "Power System Digital Twin Held-Out Validation Benchmark",
    "test_files": loader.TEST_FILES,
    "total_evaluated_sequences": sum(validation_results[f]["total_sequences"] for f in loader.TEST_FILES),
    "per_file_results": validation_results,
    "aggregated_hypotheses_by_class": all_hypothesis_by_class,
    "aggregated_topology_by_class": all_topology_by_class,
    "aggregated_physical_consistency_by_class": all_physical_consistency_by_class,
    "aggregated_topology_consistency_by_class": all_topology_consistency_by_class,
    "sample_inspected_events": sample_event_records,
}

with open(REPORTS_DIR / "digital_twin_validation.json", "w") as fp:
    json.dump(validation_artifact, fp, indent=2)

print(f"\nSaved machine-readable validation results to {REPORTS_DIR / 'digital_twin_validation.json'}")
print("=" * 80)
