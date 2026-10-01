"""Targeted Evidence Fusion Sensitivity Analysis for Asymmetric Configurations F & G.

Evaluates the two remaining asymmetric threshold configurations:
- Configuration F: L1 P99 + L2 P95 (Temporal Moderate, Physical Aggressive)
- Configuration G: L1 P95 + L2 P99 (Temporal Aggressive, Physical Moderate)

And consolidates all 7 configurations:
- A: L1 P99.5 + L2 P99.5 (Active Baseline)
- B: L1 P99   + L2 P99   (Symmetric Moderate)
- C: L1 P95   + L2 P95   (Symmetric Aggressive)
- D: L1 P99   + L2 P99.5 (Asymmetric Temporal Moderate)
- E: L1 P99.5 + L2 P99   (Asymmetric Physical Moderate)
- F: L1 P99   + L2 P95   (Asymmetric Physical Aggressive)
- G: L1 P95   + L2 P99   (Asymmetric Temporal Aggressive)

CRITICAL INVARIANTS:
- No retraining of any model.
- Thresholds are derived strictly from uncompromised normal operational training data.
- Zero attack data or labels used to derive thresholds.
- Preserves all model checkpoints and active production configurations untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

REPORTS_DIR = WORKSPACE_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "threshold_sensitivity"
MODELS_DIR = WORKSPACE_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.duckdb_loader import get_default_loader
from src.data.process_data import get_default_process_extractor
from src.ml.attack_evaluation import (
    ATTACK_RUNS_CATALOG,
    compute_classification_metrics,
    compute_detection_latency,
    compute_sequence_ground_truth,
    extract_verified_attack_intervals,
)
from src.ml.evidence_fusion import EvidenceFusion, load_active_evidence_fusion
from src.ml.layer1_tcn_detector import Layer1TCNDetector, load_active_layer1_detector
from src.ml.layer2_physical_relationships import Layer2PhysicalRelationships, load_active_layer2_detector
from src.ml.preprocessing import SolarWindPreprocessor
from src.ml.scope_appropriate_evaluation import (
    compute_scope_sequence_labels,
    filter_qualifying_attack_intervals,
)


def extract_telemetry(con: Any, dataset_id: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extracts raw PV and Wind telemetry aligned on nearest timestamp."""
    p_df = con.execute(f"""
        SELECT 
            ts,
            C_on_off AS pv_c_on_off,
            M_temp_air AS pv_m_temp_air,
            M_poa_direct AS pv_m_poa_direct,
            M_wind_speed AS pv_m_wind_speed,
            M_poa_diffuse AS pv_m_poa_diffuse,
            M_cell_temperature AS pv_m_cell_temperature,
            M_inverter_ac_power AS pv_m_inverter_ac_power,
            M_inverter_dc_power AS pv_m_inverter_dc_power
        FROM pv_process_data
        WHERE dataset_id = '{dataset_id}'
        ORDER BY ts
    """).fetchdf()

    w_df = con.execute(f"""
        SELECT 
            ts,
            M_power AS wind_m_power,
            M_pressure AS wind_m_pressure,
            M_wind_speed_a AS wind_m_wind_speed_a,
            M_wind_speed_b AS wind_m_wind_speed_b,
            M_temperature_a AS wind_m_temperature_a,
            M_temperature_b AS wind_m_temperature_b
        FROM wind_process_data
        WHERE dataset_id = '{dataset_id}'
        ORDER BY ts
    """).fetchdf()

    merged = pd.merge_asof(p_df, w_df, on="ts", direction="nearest")
    ts_array = merged["ts"].values
    return merged, ts_array


def run_asymmetric_threshold_sensitivity() -> Dict[str, Any]:
    """Runs targeted evaluation for configurations F & G and consolidates all 7 configurations."""
    print("[Asymmetric Sensitivity] Initializing active detectors...")
    l1_det = load_active_layer1_detector(MODELS_DIR / "tcn_autoencoder_baseline.pt")
    l2_det = load_active_layer2_detector(MODELS_DIR)

    # Load previously computed results for configurations A-E from reports/threshold_sensitivity_analysis.json
    prev_json_path = REPORTS_DIR / "threshold_sensitivity_analysis.json"
    if not prev_json_path.exists():
        raise FileNotFoundError(f"Previous results not found at {prev_json_path}")

    with open(prev_json_path, "r") as f:
        prev_data = json.load(f)

    prev_fusion = prev_data["evidence_fusion_sensitivity"]

    # 1. Normal validation dataset evaluation
    print("[Asymmetric Sensitivity] Extracting normal validation data (20260225_normal)...")
    loader = get_default_loader()
    extractor = get_default_process_extractor()
    raw_normal = extractor.extract_normal_training_set()
    preprocessor = SolarWindPreprocessor(config_name="config_a", train_ratio=0.8)
    train_raw, val_raw, _ = preprocessor.chronological_split(raw_normal)

    l1_val_out = l1_det.process_telemetry(val_raw)
    l1_val_mse = l1_val_out["anomaly_scores"]

    l2_val_raw = l2_det.compute_residuals(val_raw)
    l2_val_res = {r: l2_val_raw[r]["residual"][59:] for r in l2_det.get_relationship_ids()}

    # 2. Attack Campaign Evaluation
    print("[Asymmetric Sensitivity] Extracting multi-agent attack campaign datasets...")
    campaign_records: Dict[str, Dict[str, Any]] = {}

    with loader.get_connection("merged", attach_impact=True) as con:
        for run_name, dataset_id in ATTACK_RUNS_CATALOG.items():
            merged_df, ts_array = extract_telemetry(con, dataset_id)
            seq_end_ts = ts_array[59:]

            steps_df = extract_verified_attack_intervals(con, dataset_id)
            qualifying_steps_df = filter_qualifying_attack_intervals(steps_df)
            labels_dict = compute_sequence_ground_truth(ts_array, seq_end_ts, steps_df, sequence_length=60)
            scope_labels_dict = compute_scope_sequence_labels(seq_end_ts, steps_df, qualifying_steps_df)

            l1_out = l1_det.process_telemetry(merged_df)
            l1_mse = l1_out["anomaly_scores"]

            l2_raw = l2_det.compute_residuals(merged_df)
            l2_res = {r: l2_raw[r]["residual"][59:] for r in l2_det.get_relationship_ids()}

            campaign_records[run_name] = {
                "dataset_id": dataset_id,
                "seq_end_ts": seq_end_ts,
                "steps_df": steps_df,
                "qualifying_steps_df": qualifying_steps_df,
                "y_scope": scope_labels_dict["labels_scope_qualifying"],
                "clean_mask": scope_labels_dict["clean_negatives_mask"],
                "l1_mse": l1_mse,
                "l2_res": l2_res,
            }

    # Scope filtering
    all_y_scope = np.concatenate([c["y_scope"] for c in campaign_records.values()])
    all_clean = np.concatenate([c["clean_mask"] for c in campaign_records.values()])
    scope_filter = all_clean | (all_y_scope == 1)
    y_eval = all_y_scope[scope_filter]

    all_l1_mse = np.concatenate([c["l1_mse"] for c in campaign_records.values()])
    all_l2_res = {r: np.concatenate([c["l2_res"][r] for c in campaign_records.values()]) for r in l2_det.get_relationship_ids()}

    print(f"[Asymmetric Sensitivity] Total in-scope sequences evaluated: {len(y_eval)}, Positives: {int(np.sum(y_eval))}")

    # 3. Target asymmetric configurations
    new_combos = {
        "F (L1 P99 + L2 P95)": ("P99", "P95"),
        "G (L1 P95 + L2 P99)": ("P95", "P99"),
    }

    fusion_results: Dict[str, Any] = {}

    for combo_name, (t1, t2) in new_combos.items():
        print(f"[Asymmetric Sensitivity] Evaluating {combo_name}...")
        th1 = float(l1_det.thresholds[t1])
        th2_dict = {r: float(l2_det.thresholds[r][t2]) for r in l2_det.get_relationship_ids()}

        # Normal validation
        norm_l1_val = np.minimum(l1_val_mse / th1, 1.0)
        flag_l1_val = (l1_val_mse >= th1).astype(int)
        norm_l2_val = np.max(np.column_stack([np.minimum(l2_val_res[r] / th2_dict[r], 1.0) for r in l2_det.get_relationship_ids()]), axis=1)
        flag_l2_val = np.any(np.column_stack([l2_val_res[r] >= th2_dict[r] for r in l2_det.get_relationship_ids()]), axis=1).astype(int)

        fused_val = 0.5 * norm_l1_val + 0.5 * norm_l2_val
        agree_val = flag_l1_val & flag_l2_val

        # Scope sequences
        norm_l1_scope = np.minimum(all_l1_mse[scope_filter] / th1, 1.0)
        norm_l2_scope = np.max(np.column_stack([np.minimum(all_l2_res[r][scope_filter] / th2_dict[r], 1.0) for r in l2_det.get_relationship_ids()]), axis=1)
        fused_scope = 0.5 * norm_l1_scope + 0.5 * norm_l2_scope

        pred_flag = (fused_scope >= 0.784338).astype(int)
        roc_fused = float(roc_auc_score(y_eval, fused_scope))
        pr_fused = float(average_precision_score(y_eval, fused_scope))
        cm = compute_classification_metrics(y_eval, pred_flag)

        det_eps = 0
        total_eps = 0
        latencies = []
        for c in campaign_records.values():
            q_steps = c["qualifying_steps_df"]
            seq_end_ts = c["seq_end_ts"]
            total_eps += len(q_steps)

            c_l1 = np.minimum(c["l1_mse"] / th1, 1.0)
            c_l2 = np.max(np.column_stack([np.minimum(c["l2_res"][r] / th2_dict[r], 1.0) for r in l2_det.get_relationship_ids()]), axis=1)
            c_fused = 0.5 * c_l1 + 0.5 * c_l2

            lat_res = compute_detection_latency(q_steps, seq_end_ts, c_fused, 0.784338)
            det_eps += lat_res["detected_intervals"]
            latencies.extend(lat_res["latencies_array"])

        ep_rate = float(round(det_eps / total_eps * 100.0, 2))
        med_lat = float(round(np.median(latencies), 4)) if latencies else None
        mean_lat = float(round(np.mean(latencies), 4)) if latencies else None

        flag_l1_scope = (all_l1_mse[scope_filter] >= th1).astype(int)
        flag_l2_scope = np.any(np.column_stack([all_l2_res[r][scope_filter] >= th2_dict[r] for r in l2_det.get_relationship_ids()]), axis=1).astype(int)
        agree_scope = flag_l1_scope & flag_l2_scope
        cm_agree = compute_classification_metrics(y_eval, agree_scope)

        det_eps_agree = 0
        lat_agree = []
        for c in campaign_records.values():
            q_steps = c["qualifying_steps_df"]
            seq_end_ts = c["seq_end_ts"]
            c_l1_flag = (c["l1_mse"] >= th1).astype(int)
            c_l2_flag = np.any(np.column_stack([c["l2_res"][r] >= th2_dict[r] for r in l2_det.get_relationship_ids()]), axis=1).astype(int)
            c_agree = (c_l1_flag & c_l2_flag).astype(float)
            lat_res = compute_detection_latency(q_steps, seq_end_ts, c_agree, 0.5)
            det_eps_agree += lat_res["detected_intervals"]
            lat_agree.extend(lat_res["latencies_array"])

        val_fused_exceed = int(np.sum(fused_val >= 0.784338))
        val_fused_exceed_pct = float(round(val_fused_exceed / len(fused_val) * 100.0, 2))
        val_agree_exceed = int(np.sum(agree_val))
        val_agree_exceed_pct = float(round(val_agree_exceed / len(agree_val) * 100.0, 2))

        fusion_results[combo_name] = {
            "combo_name": combo_name,
            "l1_tier": t1,
            "l2_tier": t2,
            "l1_threshold": th1,
            "l2_thresholds": th2_dict,
            "fused_threshold": 0.784338,
            "normal_val_fused_exceed_count": val_fused_exceed,
            "normal_val_fused_exceed_pct": val_fused_exceed_pct,
            "normal_val_agree_exceed_count": val_agree_exceed,
            "normal_val_agree_exceed_pct": val_agree_exceed_pct,
            "roc_auc": float(round(roc_fused, 4)),
            "pr_auc": float(round(pr_fused, 4)),
            "precision": float(round(cm["precision"], 4)),
            "recall": float(round(cm["recall"], 4)),
            "f1": float(round(cm["f1"], 4)),
            "fpr": float(round(cm["fpr"], 4)),
            "fnr": float(round(cm["fnr"], 4)),
            "tp": int(cm["tp"]),
            "fp": int(cm["fp"]),
            "tn": int(cm["tn"]),
            "fn": int(cm["fn"]),
            "episodes_detected": int(det_eps),
            "total_episodes": int(total_eps),
            "episode_detection_rate_pct": ep_rate,
            "median_latency_sec": med_lat,
            "mean_latency_sec": mean_lat,
            "agreement_metrics": {
                "precision": float(round(cm_agree["precision"], 4)),
                "recall": float(round(cm_agree["recall"], 4)),
                "f1": float(round(cm_agree["f1"], 4)),
                "fpr": float(round(cm_agree["fpr"], 4)),
                "fnr": float(round(cm_agree["fnr"], 4)),
                "episodes_detected": int(det_eps_agree),
                "episode_detection_rate_pct": float(round(det_eps_agree / total_eps * 100.0, 2)),
                "median_latency_sec": float(round(np.median(lat_agree), 4)) if lat_agree else None,
                "mean_latency_sec": float(round(np.mean(lat_agree), 4)) if lat_agree else None,
            },
        }

    # Consolidated 7 configurations dictionary
    consolidated_all_7: Dict[str, Any] = {}
    config_keys = [
        ("A", "A (L1 P99.5 + L2 P99.5)"),
        ("B", "B (L1 P99 + L2 P99)"),
        ("C", "C (L1 P95 + L2 P95)"),
        ("D", "D (L1 P99 + L2 P99.5)"),
        ("E", "E (L1 P99.5 + L2 P99)"),
        ("F", "F (L1 P99 + L2 P95)"),
        ("G", "G (L1 P95 + L2 P99)"),
    ]

    for code, full_name in config_keys:
        if full_name in prev_fusion:
            item = prev_fusion[full_name].copy()
            item["config_code"] = code
            consolidated_all_7[code] = item
        elif full_name in fusion_results:
            item = fusion_results[full_name].copy()
            item["config_code"] = code
            consolidated_all_7[code] = item

    # 4. Generate Visualizations
    print("[Asymmetric Sensitivity] Generating publication-quality figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Asymmetric Fusion Comparison (Episode detection & Agreement vs FPR & Normal Exceedance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    labels = ["A\n(P99.5/P99.5)", "B\n(P99/P99)", "C\n(P95/P95)", "D\n(P99/P99.5)", "E\n(P99.5/P99)", "F\n(P99/P95)", "G\n(P95/P99)"]
    x = np.arange(len(labels))
    width = 0.35

    ep_fused = [consolidated_all_7[c]["episode_detection_rate_pct"] for c in ["A", "B", "C", "D", "E", "F", "G"]]
    ep_agree = [consolidated_all_7[c]["agreement_metrics"]["episode_detection_rate_pct"] for c in ["A", "B", "C", "D", "E", "F", "G"]]

    rects1 = ax1.bar(x - width/2, ep_fused, width, label="Fused Score (T=0.784)", color="#2b5c8f", alpha=0.9)
    rects2 = ax1.bar(x + width/2, ep_agree, width, label="Dual Agreement Flag", color="#38a169", alpha=0.9)

    ax1.set_ylabel("Episode Detection Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Attack Episode Coverage Across All 7 Configurations", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_ylim(0, 75)
    ax1.grid(True, alpha=0.3)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 0.8), fontsize=8, ha="center", fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 0.8), fontsize=8, ha="center", fontweight="bold")

    # Panel 2: False Positive Footprint
    fpr_vals = [consolidated_all_7[c]["fpr"] * 100.0 for c in ["A", "B", "C", "D", "E", "F", "G"]]
    val_exc = [consolidated_all_7[c]["normal_val_fused_exceed_pct"] for c in ["A", "B", "C", "D", "E", "F", "G"]]
    val_agree = [consolidated_all_7[c]["normal_val_agree_exceed_pct"] for c in ["A", "B", "C", "D", "E", "F", "G"]]

    ax2.plot(x, fpr_vals, "o-", color="#e53e3e", linewidth=2.2, markersize=8, label="Test Campaign FPR (%)")
    ax2.plot(x, val_exc, "s--", color="#dd6b20", linewidth=2.0, markersize=7, label="Normal Val Exceedance (%)")
    ax2.plot(x, val_agree, "^:", color="#319795", linewidth=2.2, markersize=8, label="Normal Agreement Exceedance (%)")

    ax2.set_ylabel("False Positive / Exceedance Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("False Alarm Penalty Across Configurations", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.set_ylim(0, 65)
    ax2.grid(True, alpha=0.3)

    for i, txt in enumerate(fpr_vals):
        ax2.annotate(f"{txt:.1f}%", (x[i], fpr_vals[i] + 1.2), fontsize=8, ha="center", fontweight="bold", color="#e53e3e")

    plt.tight_layout()
    comp_fig_path = FIGURES_DIR / "asymmetric_fusion_comparison.png"
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"[Asymmetric Sensitivity] Saved comparison figure to {comp_fig_path}")

    # Plot 2: Precision-Recall vs FPR trade-off space for all 7 configs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    recalls = [consolidated_all_7[c]["recall"] * 100.0 for c in ["A", "B", "C", "D", "E", "F", "G"]]
    precisions = [consolidated_all_7[c]["precision"] * 100.0 for c in ["A", "B", "C", "D", "E", "F", "G"]]
    f1s = [consolidated_all_7[c]["f1"] * 100.0 for c in ["A", "B", "C", "D", "E", "F", "G"]]
    colors = ["#718096", "#3182ce", "#e53e3e", "#805ad5", "#319795", "#d69e2e", "#dd6b20"]

    for i, c in enumerate(["A", "B", "C", "D", "E", "F", "G"]):
        ax1.scatter(recalls[i], precisions[i], color=colors[i], s=120, zorder=5, label=f"{c}: {labels[i].replace(chr(10), ' ')}")
        ax1.annotate(f" {c}", (recalls[i] + 0.3, precisions[i] + 0.05), fontsize=10, fontweight="bold")

    ax1.set_xlabel("Sequence Recall (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sequence Precision (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Sequence Precision vs. Recall Trade-off", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    for i, c in enumerate(["A", "B", "C", "D", "E", "F", "G"]):
        ax2.scatter(fpr_vals[i], recalls[i], color=colors[i], s=120, zorder=5, label=f"{c}")
        ax2.annotate(f" {c}", (fpr_vals[i] + 0.3, recalls[i] + 0.3), fontsize=10, fontweight="bold")

    ax2.set_xlabel("Test Scope FPR (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sequence Recall (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Sequence Recall vs. FPR Operating Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pr_fig_path = FIGURES_DIR / "asymmetric_fusion_pr_curve.png"
    plt.savefig(pr_fig_path, dpi=300)
    plt.close()
    print(f"[Asymmetric Sensitivity] Saved PR trade-off figure to {pr_fig_path}")

    # Output JSON serialization
    output_dict = {
        "asymmetric_configurations_evaluated": fusion_results,
        "consolidated_all_7_configurations": consolidated_all_7,
        "figures": [
            str(comp_fig_path.relative_to(WORKSPACE_ROOT)),
            str(pr_fig_path.relative_to(WORKSPACE_ROOT)),
        ],
    }

    asym_json_path = REPORTS_DIR / "threshold_sensitivity_asymmetric.json"
    with open(asym_json_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"[Asymmetric Sensitivity] Saved results JSON to {asym_json_path}")

    return output_dict


if __name__ == "__main__":
    run_asymmetric_threshold_sensitivity()
