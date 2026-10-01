"""Controlled Experiment: Layer 2 Physical-Relationship Aggregation Comparison.

Evaluates alternative aggregation methods for Layer 2 normalized physical-relationship scores
inside Evidence Fusion:
1. MAX: max(s_1, s_2, s_3, s_4) [Current Baseline]
2. MEAN: (s_1 + s_2 + s_3 + s_4) / 4
3. TOP-2 MEAN: (s_(1) + s_(2)) / 2
4. TOP-3 MEAN: (s_(1) + s_(2) + s_(3)) / 3
5. WEIGHTED MEAN (Equal Weights): 0.25 * s_1 + 0.25 * s_2 + 0.25 * s_3 + 0.25 * s_4 (Sanity check)
6. WEIGHTED MEAN (Domain Balanced): 0.25 each (PV domain 0.5, Wind domain 0.5)

CRITICAL INVARIANTS:
- No retraining of any model.
- Architecture remains: Layer 1 + Layer 2 in parallel -> Evidence Fusion.
- Checkpoints untouched.
- Active thresholds untouched: Layer 1 = P99 (0.00089546), Layer 2 = P95 (7.0729 kW, 0.5411 m/s, 0.8566 C, 10.3477 C).
- Fixed 0.5 / 0.5 L1-L2 fusion weights.
- Strategy D evaluation on 153,196 sequences, 304 qualifying impactful episodes across 5 campaigns.
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

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

REPORTS_DIR = WORKSPACE_ROOT / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
FIGURES_DIR = EXPERIMENTS_DIR / "figures" / "layer2_aggregation"
MODELS_DIR = WORKSPACE_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

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


def compute_layer2_aggregations(scores_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes all candidate Layer 2 aggregation methods from an (N, 4) matrix of normalized scores.
    
    Args:
        scores_matrix: (N, 4) array where each column is min(|res_k| / th_k, 1.0)
    
    Returns:
        Dict mapping method name to 1D array of shape (N,)
    """
    # 1. MAX
    max_agg = np.max(scores_matrix, axis=1)

    # 2. MEAN
    mean_agg = np.mean(scores_matrix, axis=1)

    # Sorted descending along axis 1
    sorted_scores = np.sort(scores_matrix, axis=1)[:, ::-1]

    # 3. TOP-2 MEAN
    top2_agg = np.mean(sorted_scores[:, :2], axis=1)

    # 4. TOP-3 MEAN
    top3_agg = np.mean(sorted_scores[:, :3], axis=1)

    # 5. WEIGHTED MEAN (Equal weights = 0.25 each)
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    weighted_agg = np.dot(scores_matrix, weights)

    return {
        "max": max_agg,
        "mean": mean_agg,
        "top2_mean": top2_agg,
        "top3_mean": top3_agg,
        "weighted_equal": weighted_agg,
    }


def run_experiment() -> Dict[str, Any]:
    print("[L2 Aggregation Experiment] Initializing active detectors...")
    l1_det = load_active_layer1_detector(MODELS_DIR / "tcn_autoencoder_baseline.pt")
    l2_det = load_active_layer2_detector(MODELS_DIR)

    th1 = float(l1_det.primary_threshold)  # P99 = 0.00089546
    th2_dict = {r: float(l2_det.thresholds[r]["P95"]) for r in l2_det.get_relationship_ids()}
    rel_ids = l2_det.get_relationship_ids()

    print(f"  Layer 1 Active Threshold (P99): {th1:.8f}")
    print(f"  Layer 2 Active Thresholds (P95): {th2_dict}")

    loader = get_default_loader()
    extractor = get_default_process_extractor()

    # 1. Normal Train and Validation Extraction
    print("[L2 Aggregation Experiment] Extracting normal dataset (20260225_normal)...")
    raw_normal = extractor.extract_normal_training_set()
    preprocessor = SolarWindPreprocessor(config_name="config_a", train_ratio=0.8)
    train_raw, val_raw, _ = preprocessor.chronological_split(raw_normal)

    # Normal train scores for threshold calibration
    l1_train_out = l1_det.process_telemetry(train_raw)
    l1_train_mse = l1_train_out["anomaly_scores"]
    l2_train_raw = l2_det.compute_residuals(train_raw)
    l2_train_matrix = np.column_stack([
        np.minimum(l2_train_raw[r]["residual"][59:] / th2_dict[r], 1.0)
        for r in rel_ids
    ])
    norm_l1_train = np.minimum(l1_train_mse / th1, 1.0)
    l2_train_aggs = compute_layer2_aggregations(l2_train_matrix)

    # Normal val scores for exceedance evaluation
    l1_val_out = l1_det.process_telemetry(val_raw)
    l1_val_mse = l1_val_out["anomaly_scores"]
    l2_val_raw = l2_det.compute_residuals(val_raw)
    l2_val_matrix = np.column_stack([
        np.minimum(l2_val_raw[r]["residual"][59:] / th2_dict[r], 1.0)
        for r in rel_ids
    ])
    norm_l1_val = np.minimum(l1_val_mse / th1, 1.0)
    l2_val_aggs = compute_layer2_aggregations(l2_val_matrix)

    # Compute normal training percentiles for each aggregation method
    method_keys = ["max", "mean", "top2_mean", "top3_mean", "weighted_equal"]
    method_titles = {
        "max": "MAX (Current Baseline)",
        "mean": "MEAN (All 4)",
        "top2_mean": "TOP-2 MEAN",
        "top3_mean": "TOP-3 MEAN",
        "weighted_equal": "WEIGHTED MEAN (Equal)",
    }

    calibrated_th_p995: Dict[str, float] = {}
    calibrated_th_p99: Dict[str, float] = {}
    calibrated_th_p95: Dict[str, float] = {}

    for m in method_keys:
        fused_train_m = 0.5 * norm_l1_train + 0.5 * l2_train_aggs[m]
        calibrated_th_p995[m] = float(np.percentile(fused_train_m, 99.5))
        calibrated_th_p99[m] = float(np.percentile(fused_train_m, 99.0))
        calibrated_th_p95[m] = float(np.percentile(fused_train_m, 95.0))
        print(f"  Calibrated Normal Fused Thresholds for {m}: P95={calibrated_th_p95[m]:.4f} | P99={calibrated_th_p99[m]:.4f} | P99.5={calibrated_th_p995[m]:.4f}")

    # 2. Attack Campaigns Ingestion
    print("[L2 Aggregation Experiment] Ingesting all 5 multi-agent attack campaigns...")
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
            l2_matrix = np.column_stack([
                np.minimum(l2_raw[r]["residual"][59:] / th2_dict[r], 1.0)
                for r in rel_ids
            ])

            l2_aggs = compute_layer2_aggregations(l2_matrix)
            c_norm_l1 = np.minimum(l1_mse / th1, 1.0)

            campaign_records[run_name] = {
                "dataset_id": dataset_id,
                "seq_end_ts": seq_end_ts,
                "steps_df": steps_df,
                "qualifying_steps_df": qualifying_steps_df,
                "y_scope": scope_labels_dict["labels_scope_qualifying"],
                "clean_mask": scope_labels_dict["clean_negatives_mask"],
                "norm_l1": c_norm_l1,
                "l2_aggs": l2_aggs,
            }

    # Pooled evaluation sequences
    all_y_scope = np.concatenate([c["y_scope"] for c in campaign_records.values()])
    all_clean = np.concatenate([c["clean_mask"] for c in campaign_records.values()])
    scope_filter = all_clean | (all_y_scope == 1)
    y_eval = all_y_scope[scope_filter]

    all_norm_l1 = np.concatenate([c["norm_l1"] for c in campaign_records.values()])[scope_filter]
    all_l2_aggs = {
        m: np.concatenate([c["l2_aggs"][m] for c in campaign_records.values()])[scope_filter]
        for m in method_keys
    }

    total_qualifying_episodes = sum(len(c["qualifying_steps_df"]) for c in campaign_records.values())
    print(f"[L2 Aggregation Experiment] Total in-scope sequences: {len(y_eval)}, Positives: {int(np.sum(y_eval))}, Total qualifying episodes: {total_qualifying_episodes}")

    # 3. Evaluate Each Aggregation Method
    results: Dict[str, Any] = {}

    for m in method_keys:
        print(f"[L2 Aggregation Experiment] Evaluating {m}...")
        fused_score = 0.5 * all_norm_l1 + 0.5 * all_l2_aggs[m]
        fused_val = 0.5 * norm_l1_val + 0.5 * l2_val_aggs[m]

        # Ranking metrics (independent of threshold)
        roc_auc = float(roc_auc_score(y_eval, fused_score))
        pr_auc = float(average_precision_score(y_eval, fused_score))

        # We evaluate under two critical threshold paradigms:
        # Paradigm A: Fixed baseline decision threshold T = 0.784338
        # Paradigm B: Normal-calibrated threshold T = P99.5 of that method's normal training fused score

        th_fixed = 0.784338
        th_calibrated = calibrated_th_p995[m]

        paradigms = {
            "fixed_baseline_threshold": th_fixed,
            "calibrated_p995_threshold": th_calibrated,
        }

        method_eval: Dict[str, Any] = {
            "method_key": m,
            "title": method_titles[m],
            "roc_auc": float(round(roc_auc, 4)),
            "pr_auc": float(round(pr_auc, 4)),
            "calibrated_normal_p995_threshold": float(round(th_calibrated, 6)),
            "calibrated_normal_p99_threshold": float(round(calibrated_th_p99[m], 6)),
            "calibrated_normal_p95_threshold": float(round(calibrated_th_p95[m], 6)),
            "evaluations": {},
        }

        for p_name, th in paradigms.items():
            pred_flags = (fused_score >= th).astype(int)
            cm = compute_classification_metrics(y_eval, pred_flags)

            val_exceed_count = int(np.sum(fused_val >= th))
            val_exceed_pct = float(round(val_exceed_count / len(fused_val) * 100.0, 2))

            # Episode latency and count
            det_eps = 0
            latencies = []
            for c in campaign_records.values():
                q_steps = c["qualifying_steps_df"]
                seq_end_ts = c["seq_end_ts"]
                c_fused = 0.5 * c["norm_l1"] + 0.5 * c["l2_aggs"][m]
                lat_res = compute_detection_latency(q_steps, seq_end_ts, c_fused, th)
                det_eps += lat_res["detected_intervals"]
                latencies.extend(lat_res["latencies_array"])

            ep_rate = float(round(det_eps / total_qualifying_episodes * 100.0, 2))
            med_lat = float(round(np.median(latencies), 4)) if latencies else None
            mean_lat = float(round(np.mean(latencies), 4)) if latencies else None

            method_eval["evaluations"][p_name] = {
                "threshold_applied": float(round(th, 6)),
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
                "total_episodes": int(total_qualifying_episodes),
                "episode_detection_rate_pct": ep_rate,
                "median_latency_sec": med_lat,
                "mean_latency_sec": mean_lat,
                "normal_val_exceed_count": val_exceed_count,
                "normal_val_exceed_pct": val_exceed_pct,
            }

        results[m] = method_eval

    # 4. Generate Visualizations
    print("[L2 Aggregation Experiment] Generating diagnostic figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: ROC and PR Curves Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {
        "max": "#1f77b4",
        "mean": "#ff7f0e",
        "top2_mean": "#2ca02c",
        "top3_mean": "#d62728",
        "weighted_equal": "#9467bd",
    }
    linestyles = {
        "max": "-",
        "mean": "--",
        "top2_mean": "-.",
        "top3_mean": ":",
        "weighted_equal": (0, (3, 1, 1, 1)),
    }

    for m in ["max", "mean", "top2_mean", "top3_mean"]:
        fused = 0.5 * all_norm_l1 + 0.5 * all_l2_aggs[m]
        fpr_arr, tpr_arr, _ = roc_curve(y_eval, fused)
        prec_arr, rec_arr, _ = precision_recall_curve(y_eval, fused)

        ax1.plot(fpr_arr, tpr_arr, label=f"{method_titles[m]} (AUC={results[m]['roc_auc']:.4f})",
                 color=colors[m], linestyle=linestyles[m], linewidth=2.0)
        ax2.plot(rec_arr, prec_arr, label=f"{method_titles[m]} (AP={results[m]['pr_auc']:.4f})",
                 color=colors[m], linestyle=linestyles[m], linewidth=2.0)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("ROC Curves Across Layer 2 Aggregation Methods", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9.5)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Precision-Recall Curves Across Layer 2 Aggregation Methods", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "l2_aggregation_roc_pr.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved ROC/PR figure to {fig1_path}")

    # Figure 2: Multi-Metric Comparison (Episode Detection, F1, Recall, FPR under Calibrated P99.5 Threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_methods = ["max", "mean", "top2_mean", "top3_mean", "weighted_equal"]
    x = np.arange(len(plot_methods))
    width = 0.35

    # Fixed threshold bars
    ep_fixed = [results[m]["evaluations"]["fixed_baseline_threshold"]["episode_detection_rate_pct"] for m in plot_methods]
    ep_cal = [results[m]["evaluations"]["calibrated_p995_threshold"]["episode_detection_rate_pct"] for m in plot_methods]

    rects1 = ax1.bar(x - width/2, ep_fixed, width, label="Fixed Baseline Threshold (T=0.784)", color="#2b5c8f", alpha=0.85)
    rects2 = ax1.bar(x + width/2, ep_cal, width, label="Normal Calibrated P99.5 Threshold", color="#38a169", alpha=0.85)

    ax1.set_ylabel("Attack Episode Detection Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Attack Episode Detection Across Aggregation Methods", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([method_titles[m].replace(" ", "\n") for m in plot_methods], fontsize=9, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.set_ylim(0, 70)
    ax1.grid(True, alpha=0.3)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 0.8), fontsize=8, ha="center", fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 0.8), fontsize=8, ha="center", fontweight="bold")

    # Panel 2: F1 and FPR under Fixed Baseline Threshold
    f1_fixed = [results[m]["evaluations"]["fixed_baseline_threshold"]["f1"] * 100.0 for m in plot_methods]
    rec_fixed = [results[m]["evaluations"]["fixed_baseline_threshold"]["recall"] * 100.0 for m in plot_methods]
    fpr_fixed = [results[m]["evaluations"]["fixed_baseline_threshold"]["fpr"] * 100.0 for m in plot_methods]

    ax2.plot(x, rec_fixed, "s--", color="#2ca02c", linewidth=2.2, markersize=8, label="Sequence Recall (%)")
    ax2.plot(x, fpr_fixed, "d-", color="#d62728", linewidth=2.2, markersize=8, label="Test Campaign FPR (%)")
    ax2.plot(x, f1_fixed, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="F1 Score (x100)")

    ax2.set_ylabel("Metric Value (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Detection vs. False Alarm Burden (Fixed Threshold)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([method_titles[m].replace(" ", "\n") for m in plot_methods], fontsize=9, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9.5)
    ax2.set_ylim(0, 65)
    ax2.grid(True, alpha=0.3)

    for i, txt in enumerate(rec_fixed):
        ax2.annotate(f"{txt:.1f}%", (x[i], rec_fixed[i] + 1.2), fontsize=8, ha="center", fontweight="bold", color="#2ca02c")

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "l2_aggregation_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved comparison figure to {fig2_path}")

    # Output JSON serialization
    output_dict = {
        "experiment_title": "Controlled Experiment: Layer 2 Physical Aggregation Comparison",
        "active_configuration_evaluated": {
            "layer1": "Causal TCN Autoencoder (P99 = 0.00089546)",
            "layer2": "4 XGBoost Physical Regressors (P95 thresholds)",
            "evidence_fusion_weights": "0.5 L1 + 0.5 L2",
        },
        "evaluation_standard": "Strategy D (Sequence End Timestamp t_59)",
        "total_test_sequences": len(y_eval),
        "total_qualifying_episodes": total_qualifying_episodes,
        "aggregation_results": results,
        "figures": [
            str(fig1_path.relative_to(WORKSPACE_ROOT)),
            str(fig2_path.relative_to(WORKSPACE_ROOT)),
        ],
    }

    out_json_path = EXPERIMENTS_DIR / "layer2_aggregation_comparison.json"
    with open(out_json_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"  Saved experiment JSON to {out_json_path}")

    return output_dict


if __name__ == "__main__":
    run_experiment()
