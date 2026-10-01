"""Threshold Sensitivity Analysis Module for Smart Grid Anomaly Detection.

Evaluates sensitivity across candidate operational threshold tiers:
- Tier 1: P99.5 (Current Production Baseline)
- Tier 2: P99   (Moderate Sensitivity)
- Tier 3: P95   (Aggressive Sensitivity)

Evaluated across:
1. Layer 1 (Causal TCN Autoencoder)
2. Layer 2 (4 Physical Relationships + Aggregate Max)
3. Evidence Fusion (Configurations A, B, C, D, E)

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
from typing import Any, Dict, List, Optional, Tuple, Union
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


def run_threshold_sensitivity_analysis() -> Dict[str, Any]:
    """Executes the full threshold sensitivity evaluation and generates reports and plots."""
    print("[Threshold Sensitivity] Initializing active detectors...")
    l1_det = load_active_layer1_detector(MODELS_DIR / "tcn_autoencoder_baseline.pt")
    l2_det = load_active_layer2_detector(MODELS_DIR)
    loader = get_default_loader()

    # Step 1: Normal validation distribution
    print("[Threshold Sensitivity] Evaluating normal training and validation splits...")
    extractor = get_default_process_extractor()
    raw_normal = extractor.extract_normal_training_set()
    preprocessor = SolarWindPreprocessor(config_name="config_a", train_ratio=0.8)
    train_raw, val_raw, _ = preprocessor.chronological_split(raw_normal)

    l1_val_out = l1_det.process_telemetry(val_raw)
    l1_val_mse = l1_val_out["anomaly_scores"]

    l2_val_raw = l2_det.compute_residuals(val_raw)
    l2_val_res = {r: l2_val_raw[r]["residual"][59:] for r in l2_det.get_relationship_ids()}

    # Step 2: Campaign ingestion
    print("[Threshold Sensitivity] Ingesting all 5 multi-agent attack campaigns...")
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

    # =========================================================================
    # PART 1: LAYER 1 EVALUATION
    # =========================================================================
    print("[Threshold Sensitivity] Computing Layer 1 sensitivity metrics...")
    l1_results: Dict[str, Any] = {}
    roc_l1 = float(roc_auc_score(y_eval, all_l1_mse[scope_filter]))
    pr_l1 = float(average_precision_score(y_eval, all_l1_mse[scope_filter]))

    for tier in ["P99.5", "P99", "P95"]:
        th_val = float(l1_det.thresholds[tier])
        val_exceed = int(np.sum(l1_val_mse >= th_val))
        val_exceed_pct = float(round(val_exceed / len(l1_val_mse) * 100.0, 2))

        pred_flag = (all_l1_mse[scope_filter] >= th_val).astype(int)
        cm = compute_classification_metrics(y_eval, pred_flag)

        det_eps = 0
        total_eps = 0
        latencies: List[float] = []
        for c in campaign_records.values():
            q_steps = c["qualifying_steps_df"]
            seq_end_ts = c["seq_end_ts"]
            total_eps += len(q_steps)
            lat_res = compute_detection_latency(q_steps, seq_end_ts, c["l1_mse"], th_val)
            det_eps += lat_res["detected_intervals"]
            latencies.extend(lat_res["latencies_array"])

        ep_rate = float(round(det_eps / total_eps * 100.0, 2))
        med_lat = float(round(np.median(latencies), 4)) if latencies else None
        mean_lat = float(round(np.mean(latencies), 4)) if latencies else None

        l1_results[tier] = {
            "tier": tier,
            "threshold_value": th_val,
            "normal_val_exceed_count": val_exceed,
            "normal_val_exceed_pct": val_exceed_pct,
            "roc_auc": float(round(roc_l1, 4)),
            "pr_auc": float(round(pr_l1, 4)),
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
        }

    # =========================================================================
    # PART 2: LAYER 2 RELATIONSHIPS & AGGREGATE EVALUATION
    # =========================================================================
    print("[Threshold Sensitivity] Computing Layer 2 sensitivity metrics...")
    l2_individual_results: Dict[str, Any] = {}

    for rel_id in l2_det.get_relationship_ids():
        rel_meta = l2_det.relationships[rel_id]
        name = rel_meta["name"]
        unit = rel_meta["unit"]
        res_scope = all_l2_res[rel_id][scope_filter]
        roc_rel = float(roc_auc_score(y_eval, res_scope))
        pr_rel = float(average_precision_score(y_eval, res_scope))

        l2_individual_results[rel_id] = {
            "name": name,
            "unit": unit,
            "roc_auc": float(round(roc_rel, 4)),
            "pr_auc": float(round(pr_rel, 4)),
            "tiers": {},
        }

        for tier in ["P99.5", "P99", "P95"]:
            th_val = float(l2_det.thresholds[rel_id][tier])
            val_exceed = int(np.sum(l2_val_res[rel_id] >= th_val))
            val_exceed_pct = float(round(val_exceed / len(l2_val_res[rel_id]) * 100.0, 2))

            pred_flag = (res_scope >= th_val).astype(int)
            cm = compute_classification_metrics(y_eval, pred_flag)

            det_eps = 0
            total_eps = 0
            latencies = []
            for c in campaign_records.values():
                q_steps = c["qualifying_steps_df"]
                seq_end_ts = c["seq_end_ts"]
                total_eps += len(q_steps)
                lat_res = compute_detection_latency(q_steps, seq_end_ts, c["l2_res"][rel_id], th_val)
                det_eps += lat_res["detected_intervals"]
                latencies.extend(lat_res["latencies_array"])

            ep_rate = float(round(det_eps / total_eps * 100.0, 2))
            med_lat = float(round(np.median(latencies), 4)) if latencies else None
            mean_lat = float(round(np.mean(latencies), 4)) if latencies else None

            l2_individual_results[rel_id]["tiers"][tier] = {
                "tier": tier,
                "threshold_value": th_val,
                "unit": unit,
                "normal_val_exceed_count": val_exceed,
                "normal_val_exceed_pct": val_exceed_pct,
                "roc_auc": float(round(roc_rel, 4)),
                "pr_auc": float(round(pr_rel, 4)),
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
            }

    # Layer 2 Aggregate
    l2_aggregate_results: Dict[str, Any] = {}
    for tier in ["P99.5", "P99", "P95"]:
        flags_val = [l2_val_res[r] >= l2_det.thresholds[r][tier] for r in l2_det.get_relationship_ids()]
        agg_val = np.any(np.column_stack(flags_val), axis=1)
        val_exceed = int(np.sum(agg_val))
        val_exceed_pct = float(round(val_exceed / len(agg_val) * 100.0, 2))

        norm_scores_scope = [np.minimum(all_l2_res[r][scope_filter] / l2_det.thresholds[r][tier], 1.0) for r in l2_det.get_relationship_ids()]
        agg_score_scope = np.max(np.column_stack(norm_scores_scope), axis=1)
        pred_flag = (agg_score_scope >= 1.0).astype(int)

        roc_agg = float(roc_auc_score(y_eval, agg_score_scope))
        pr_agg = float(average_precision_score(y_eval, agg_score_scope))
        cm = compute_classification_metrics(y_eval, pred_flag)

        det_eps = 0
        total_eps = 0
        latencies = []
        for c in campaign_records.values():
            q_steps = c["qualifying_steps_df"]
            seq_end_ts = c["seq_end_ts"]
            total_eps += len(q_steps)
            camp_norm = [np.minimum(c["l2_res"][r] / l2_det.thresholds[r][tier], 1.0) for r in l2_det.get_relationship_ids()]
            camp_agg = np.max(np.column_stack(camp_norm), axis=1)
            lat_res = compute_detection_latency(q_steps, seq_end_ts, camp_agg, 1.0)
            det_eps += lat_res["detected_intervals"]
            latencies.extend(lat_res["latencies_array"])

        ep_rate = float(round(det_eps / total_eps * 100.0, 2))
        med_lat = float(round(np.median(latencies), 4)) if latencies else None
        mean_lat = float(round(np.mean(latencies), 4)) if latencies else None

        l2_aggregate_results[tier] = {
            "tier": tier,
            "normal_val_exceed_count": val_exceed,
            "normal_val_exceed_pct": val_exceed_pct,
            "roc_auc": float(round(roc_agg, 4)),
            "pr_auc": float(round(pr_agg, 4)),
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
        }

    # =========================================================================
    # PART 3: EVIDENCE FUSION SENSITIVITY COMBINATIONS
    # =========================================================================
    print("[Threshold Sensitivity] Computing Evidence Fusion combinations...")
    fusion_combos = {
        "A (L1 P99.5 + L2 P99.5)": ("P99.5", "P99.5"),
        "B (L1 P99 + L2 P99)": ("P99", "P99"),
        "C (L1 P95 + L2 P95)": ("P95", "P95"),
        "D (L1 P99 + L2 P99.5)": ("P99", "P99.5"),
        "E (L1 P99.5 + L2 P99)": ("P99.5", "P99"),
    }

    fusion_results: Dict[str, Any] = {}
    for combo_name, (t1, t2) in fusion_combos.items():
        th1 = float(l1_det.thresholds[t1])
        th2_dict = {r: float(l2_det.thresholds[r][t2]) for r in l2_det.get_relationship_ids()}

        norm_l1_val = np.minimum(l1_val_mse / th1, 1.0)
        flag_l1_val = (l1_val_mse >= th1).astype(int)
        norm_l2_val = np.max(np.column_stack([np.minimum(l2_val_res[r] / th2_dict[r], 1.0) for r in l2_det.get_relationship_ids()]), axis=1)
        flag_l2_val = np.any(np.column_stack([l2_val_res[r] >= th2_dict[r] for r in l2_det.get_relationship_ids()]), axis=1).astype(int)

        fused_val = 0.5 * norm_l1_val + 0.5 * norm_l2_val
        agree_val = flag_l1_val & flag_l2_val

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
            "normal_val_fused_exceed_pct": val_fused_exceed_pct,
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
            },
        }

    # =========================================================================
    # PART 4: VISUALIZATIONS GENERATION
    # =========================================================================
    print("[Threshold Sensitivity] Generating publication-quality figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Layer 1 Sensitivity Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    tiers = ["P99.5", "P99", "P95"]
    recalls = [l1_results[t]["recall"] * 100.0 for t in tiers]
    fprs = [l1_results[t]["fpr"] * 100.0 for t in tiers]
    val_exc = [l1_results[t]["normal_val_exceed_pct"] for t in tiers]
    ep_rates = [l1_results[t]["episode_detection_rate_pct"] for t in tiers]

    x = np.arange(len(tiers))
    ax1.plot(x, ep_rates, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Episode Detection Rate (%)")
    ax1.plot(x, recalls, "s--", color="#2ca02c", linewidth=2.0, markersize=7, label="Sequence Recall (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["P99.5\n(Baseline: 0.000970)", "P99\n(Moderate: 0.000895)", "P95\n(Aggressive: 0.000616)"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Detection Rate / Recall (%)", fontsize=11)
    ax1.set_title("Layer 1: Attack Detection vs. Threshold Sensitivity", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_ylim(30, 60)
    for i, txt in enumerate(ep_rates):
        ax1.annotate(f"{txt:.1f}% ({l1_results[tiers[i]]['episodes_detected']}/304)", (x[i], ep_rates[i] + 1.0), fontsize=9, fontweight="bold", ha="center")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, fprs, "d-", color="#d62728", linewidth=2.5, markersize=8, label="Test Scope FPR (%)")
    ax2.plot(x, val_exc, "^--", color="#ff7f0e", linewidth=2.0, markersize=7, label="Normal Validation Exceedance (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["P99.5\n(Baseline)", "P99\n(Moderate)", "P95\n(Aggressive)"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("False Positive / Exceedance Rate (%)", fontsize=11)
    ax2.set_title("Layer 1: False Alarms vs. Threshold Sensitivity", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.set_ylim(15, 55)
    for i, txt in enumerate(fprs):
        ax2.annotate(f"FPR: {txt:.1f}%\nVal: {val_exc[i]:.1f}%", (x[i], fprs[i] + 1.2), fontsize=9, fontweight="bold", ha="center")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig1_path = FIGURES_DIR / "l1_threshold_sensitivity.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # Figure 2: Layer 2 Relationship-Level Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    rel_ids = l2_det.get_relationship_ids()
    rel_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    for idx, rel_id in enumerate(rel_ids):
        ax = axes[idx // 2, idx % 2]
        rdata = l2_individual_results[rel_id]
        tiers_list = ["P99.5", "P99", "P95"]
        r_eps = [rdata["tiers"][t]["episode_detection_rate_pct"] for t in tiers_list]
        r_fpr = [rdata["tiers"][t]["fpr"] * 100.0 for t in tiers_list]
        r_val = [rdata["tiers"][t]["normal_val_exceed_pct"] for t in tiers_list]
        th_labels = [f"{t}\n({rdata['tiers'][t]['threshold_value']:.2f} {rdata['unit']})" for t in tiers_list]

        x_rel = np.arange(len(tiers_list))
        w = 0.28
        b1 = ax.bar(x_rel - w, r_eps, w, label="Episode Det. (%)", color=rel_colors[idx], alpha=0.9)
        b2 = ax.bar(x_rel, r_fpr, w, label="Scope FPR (%)", color="#e377c2", alpha=0.8)
        b3 = ax.bar(x_rel + w, r_val, w, label="Normal Val Exc. (%)", color="#bcbd22", alpha=0.8)

        ax.set_xticks(x_rel)
        ax.set_xticklabels(th_labels, fontsize=9, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.set_title(f"{rdata['name']} ({rel_id})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, max(max(r_eps), max(r_fpr), max(r_val), 10) * 1.25)
        ax.grid(True, axis="y", alpha=0.3)

        for b in b1:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Layer 2: Individual Physical Relationship Sensitivity Comparison", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig2_path = FIGURES_DIR / "l2_relationship_comparison.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    # Figure 3: Evidence Fusion Configuration Comparison
    fig, (ax_fused, ax_agree) = plt.subplots(1, 2, figsize=(15, 6))
    c_names = list(fusion_combos.keys())
    short_names = ["A\n(P99.5/P99.5)", "B\n(P99/P99)", "C\n(P95/P95)", "D\n(P99/P99.5)", "E\n(P99.5/P99)"]

    f_eps = [fusion_results[c]["episode_detection_rate_pct"] for c in c_names]
    f_fpr = [fusion_results[c]["fpr"] * 100.0 for c in c_names]
    f_val = [fusion_results[c]["normal_val_fused_exceed_pct"] for c in c_names]

    x_c = np.arange(len(c_names))
    w = 0.26
    ax_fused.bar(x_c - w, f_eps, w, label="Episode Det. (%)", color="#1f77b4", alpha=0.9)
    ax_fused.bar(x_c, f_fpr, w, label="Scope FPR (%)", color="#d62728", alpha=0.85)
    ax_fused.bar(x_c + w, f_val, w, label="Normal Val Exc. (%)", color="#ff7f0e", alpha=0.85)
    ax_fused.set_xticks(x_c)
    ax_fused.set_xticklabels(short_names, fontsize=9, fontweight="bold")
    ax_fused.set_ylabel("Rate (%)", fontsize=11)
    ax_fused.set_title("Evidence Fusion (Continuous Score, T=0.7843)", fontsize=12, fontweight="bold")
    ax_fused.legend(loc="upper left", fontsize=9)
    ax_fused.set_ylim(0, 75)
    ax_fused.grid(True, axis="y", alpha=0.3)
    for i, h in enumerate(f_eps):
        ax_fused.annotate(f"{h:.1f}%", xy=(x_c[i] - w, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Agreement Plot
    a_eps = [fusion_results[c]["agreement_metrics"]["episode_detection_rate_pct"] for c in c_names]
    a_fpr = [fusion_results[c]["agreement_metrics"]["fpr"] * 100.0 for c in c_names]
    a_val = [fusion_results[c]["normal_val_agree_exceed_pct"] for c in c_names]

    ax_agree.bar(x_c - w, a_eps, w, label="Dual Episode Det. (%)", color="#2ca02c", alpha=0.9)
    ax_agree.bar(x_c, a_fpr, w, label="Dual Scope FPR (%)", color="#9467bd", alpha=0.85)
    ax_agree.bar(x_c + w, a_val, w, label="Normal Val Agree Exc. (%)", color="#8c564b", alpha=0.85)
    ax_agree.set_xticks(x_c)
    ax_agree.set_xticklabels(short_names, fontsize=9, fontweight="bold")
    ax_agree.set_ylabel("Rate (%)", fontsize=11)
    ax_agree.set_title("Cross-Layer Agreement (Both L1 & L2 Flag)", fontsize=12, fontweight="bold")
    ax_agree.legend(loc="upper left", fontsize=9)
    ax_agree.set_ylim(0, 55)
    ax_agree.grid(True, axis="y", alpha=0.3)
    for i, h in enumerate(a_eps):
        ax_agree.annotate(f"{h:.1f}%", xy=(x_c[i] - w, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Evidence Fusion Sensitivity Across Threshold Configurations", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig3_path = FIGURES_DIR / "fusion_configuration_comparison.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    # Figure 4: Tradeoff Scatter (Episode Detection vs. FPR)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot_items = [
        ("L1 P99.5", l1_results["P99.5"]["fpr"]*100, l1_results["P99.5"]["episode_detection_rate_pct"], "#1f77b4", "s"),
        ("L1 P99", l1_results["P99"]["fpr"]*100, l1_results["P99"]["episode_detection_rate_pct"], "#1f77b4", "^"),
        ("L1 P95", l1_results["P95"]["fpr"]*100, l1_results["P95"]["episode_detection_rate_pct"], "#1f77b4", "v"),
        ("L2 Agg P99.5", l2_aggregate_results["P99.5"]["fpr"]*100, l2_aggregate_results["P99.5"]["episode_detection_rate_pct"], "#2ca02c", "s"),
        ("L2 Agg P99", l2_aggregate_results["P99"]["fpr"]*100, l2_aggregate_results["P99"]["episode_detection_rate_pct"], "#2ca02c", "^"),
        ("L2 Agg P95", l2_aggregate_results["P95"]["fpr"]*100, l2_aggregate_results["P95"]["episode_detection_rate_pct"], "#2ca02c", "v"),
        ("Fusion A (P99.5/P99.5)", fusion_results["A (L1 P99.5 + L2 P99.5)"]["fpr"]*100, fusion_results["A (L1 P99.5 + L2 P99.5)"]["episode_detection_rate_pct"], "#ff7f0e", "s"),
        ("Fusion B (P99/P99)", fusion_results["B (L1 P99 + L2 P99)"]["fpr"]*100, fusion_results["B (L1 P99 + L2 P99)"]["episode_detection_rate_pct"], "#ff7f0e", "^"),
        ("Fusion C (P95/P95)", fusion_results["C (L1 P95 + L2 P95)"]["fpr"]*100, fusion_results["C (L1 P95 + L2 P95)"]["episode_detection_rate_pct"], "#ff7f0e", "v"),
        ("Agreement A", fusion_results["A (L1 P99.5 + L2 P99.5)"]["agreement_metrics"]["fpr"]*100, fusion_results["A (L1 P99.5 + L2 P99.5)"]["agreement_metrics"]["episode_detection_rate_pct"], "#d62728", "s"),
        ("Agreement B", fusion_results["B (L1 P99 + L2 P99)"]["agreement_metrics"]["fpr"]*100, fusion_results["B (L1 P99 + L2 P99)"]["agreement_metrics"]["episode_detection_rate_pct"], "#d62728", "^"),
        ("Agreement C", fusion_results["C (L1 P95 + L2 P95)"]["agreement_metrics"]["fpr"]*100, fusion_results["C (L1 P95 + L2 P95)"]["agreement_metrics"]["episode_detection_rate_pct"], "#d62728", "v"),
    ]

    for label, fpr_val, ep_val, color, marker in plot_items:
        ax.scatter(fpr_val, ep_val, color=color, marker=marker, s=120, edgecolors="black", linewidth=1.2, zorder=5)
        ax.annotate(label, (fpr_val + 0.6, ep_val + 0.5), fontsize=8.5, fontweight="bold")

    ax.set_xlabel("False Positive Rate (%) on Unattacked Scope Sequences", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attack Episode Detection Rate (%) [out of 304]", fontsize=11, fontweight="bold")
    ax.set_title("Operational Tradeoff: Attack Episode Coverage vs. False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_xlim(15, 65)
    ax.set_ylim(20, 95)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig4_path = FIGURES_DIR / "precision_recall_sensitivity.png"
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    # Save JSON summary
    final_payload = {
        "evaluation_standard": "Strategy D (Sequence End Timestamp t_59)",
        "campaigns_evaluated": list(ATTACK_RUNS_CATALOG.keys()),
        "total_test_sequences": len(all_y_scope),
        "total_qualifying_episodes": 304,
        "layer1_sensitivity": l1_results,
        "layer2_individual_sensitivity": l2_individual_results,
        "layer2_aggregate_sensitivity": l2_aggregate_results,
        "evidence_fusion_sensitivity": fusion_results,
        "figures": [
            str(fig1_path.relative_to(WORKSPACE_ROOT)),
            str(fig2_path.relative_to(WORKSPACE_ROOT)),
            str(fig3_path.relative_to(WORKSPACE_ROOT)),
            str(fig4_path.relative_to(WORKSPACE_ROOT)),
        ],
    }

    json_path = REPORTS_DIR / "threshold_sensitivity_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"[Threshold Sensitivity] Saved JSON report to {json_path}")
    print(f"[Threshold Sensitivity] Saved figures to {FIGURES_DIR}")

    return final_payload


if __name__ == "__main__":
    run_threshold_sensitivity_analysis()
