"""Held-Out Benchmark Evaluation of Triple Dataset Pipeline (data13, data14, data15)."""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.triple_loader import TripleDataLoader, TripleScenarioData
from src.ml.triple_evidence_fusion import TripleEvidenceFusion
from src.ml.triple_layer1_detector import TripleLayer1Detector
from src.ml.triple_layer2_regressors import TripleLayer2Regressors

print("=" * 70)
print("EVALUATION OF TRIPLE DATASET PIPELINE ON HELD-OUT TEST FILES")
print("=" * 70)

loader = TripleDataLoader()
test_scenarios = loader.load_partition(loader.TEST_FILES)

# Initialize models
l1_det = TripleLayer1Detector.load(
    checkpoint_path=MODELS_DIR / "triple_tcn_autoencoder.pt",
    scaler_path=MODELS_DIR / "triple_scaler.json",
    metadata_path=MODELS_DIR / "triple_layer1_metadata.json",
)
l2_reg = TripleLayer2Regressors.load(MODELS_DIR)

# Calibrate normal fused threshold strictly on training data
train_scenarios = loader.load_partition(loader.TRAIN_FILES)
fusion = TripleEvidenceFusion(
    layer1_detector=l1_det,
    layer2_regressors=l2_reg,
    weight_l1=0.5,
    weight_l2=0.5,
)
th_fused_p99 = fusion.calibrate_normal_fused_threshold(train_scenarios, percentile=99.0)
th_fused_p995 = fusion.calibrate_normal_fused_threshold(train_scenarios, percentile=99.5)
th_fused_p95 = fusion.calibrate_normal_fused_threshold(train_scenarios, percentile=95.0)

print(f"Calibrated Normal Fused Thresholds: P95={th_fused_p95:.4f}, P99={th_fused_p99:.4f}, P99.5={th_fused_p995:.4f}")

# Process test scenarios
test_evidence_list = [fusion.process_scenario(scen, fused_threshold_override=th_fused_p99) for scen in test_scenarios]

# Concatenate all test outputs
all_l1_scores = np.concatenate([ev.layer1_normalized for ev in test_evidence_list])
all_l2_scores = np.concatenate([ev.layer2_top2_mean for ev in test_evidence_list])
all_fused_scores = np.concatenate([ev.fused_score for ev in test_evidence_list])
all_labels = np.concatenate([ev.ground_truth_labels for ev in test_evidence_list])
all_files = np.concatenate([[ev.scenario_name] * len(ev.ground_truth_labels) for ev in test_evidence_list])

total_test_samples = len(all_labels)
print(f"\nTotal Held-Out Test Sequences: {total_test_samples}")
print("Test Label Distribution:", pd.Series(all_labels).value_counts().to_dict())

# Define binary ground truth:
# Primary Paradigm A: Attack (Positive=1) vs Clean Normal (Negative=0; NoEvents only)
# Primary Paradigm B: Attack (Positive=1) vs All Non-Attack (NoEvents + Natural = 0)
# Secondary: Natural Disturbance Analysis (flag rate on Natural)

# Let's compute metrics under both paradigms
def evaluate_stream(scores: np.ndarray, labels: np.ndarray, threshold: float, binary_mode: str = "attack_vs_clean"):
    if binary_mode == "attack_vs_clean":
        mask = labels != "Natural"
        y_true = (labels[mask] == "Attack").astype(int)
        y_score = scores[mask]
    elif binary_mode == "attack_vs_all":
        y_true = (labels == "Attack").astype(int)
        y_score = scores
    elif binary_mode == "disturbance_vs_normal":
        # Evaluates whether system flags ANY physical anomaly (Attack OR Natural) vs Normal
        y_true = (labels != "NoEvents").astype(int)
        y_score = scores
    else:
        raise ValueError(binary_mode)

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f2 = float(5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
    }

# 1. Sequence-Level Evaluations across Layer 1, Layer 2, and Evidence Fusion
eval_summary = {}

for name, scores, th in [
    ("Layer 1 (TCN-AE)", all_l1_scores, 1.0),
    ("Layer 2 (TOP-2 MEAN)", all_l2_scores, 1.0),
    ("Evidence Fusion (P99 th)", all_fused_scores, th_fused_p99),
    ("Evidence Fusion (P95 th)", all_fused_scores, th_fused_p95),
]:
    # Mode A: Attack vs Clean NoEvents (strict anomaly detection against uncompromised normal baseline)
    res_clean = evaluate_stream(scores, all_labels, threshold=th, binary_mode="attack_vs_clean")
    # Mode B: Attack vs All (including Natural disturbances as negative)
    res_all = evaluate_stream(scores, all_labels, threshold=th, binary_mode="attack_vs_all")
    # Mode C: Physical Disturbance (Attack + Natural) vs NoEvents
    res_dist = evaluate_stream(scores, all_labels, threshold=th, binary_mode="disturbance_vs_normal")

    eval_summary[name] = {
        "attack_vs_clean_normal": res_clean,
        "attack_vs_all": res_all,
        "disturbance_vs_normal": res_dist,
    }
    print(f"\n--- {name} (Threshold={th:.4f}) ---")
    print(f"  [Attack vs Clean Normal] F1={res_clean['f1']:.4f}, F2={res_clean['f2']:.4f}, Prec={res_clean['precision']:.4f}, Rec={res_clean['recall']:.4f}, FPR={res_clean['fpr']:.4f}, PR-AUC={res_clean['pr_auc']:.4f}")
    print(f"  [Attack vs All]          F1={res_all['f1']:.4f}, F2={res_all['f2']:.4f}, Prec={res_all['precision']:.4f}, Rec={res_all['recall']:.4f}, FPR={res_all['fpr']:.4f}, PR-AUC={res_all['pr_auc']:.4f}")

# 2. Natural Fault Analysis
print("\n" + "=" * 70)
print("NATURAL FAULT & CLASS BEHAVIOR ANALYSIS")
print("=" * 70)
class_breakdown = {}
for m_class in ["NoEvents", "Natural", "Attack"]:
    m_mask = all_labels == m_class
    c_count = int(m_mask.sum())
    flag_l1 = int(np.sum(all_l1_scores[m_mask] >= 1.0))
    flag_l2 = int(np.sum(all_l2_scores[m_mask] >= 1.0))
    flag_fused_p99 = int(np.sum(all_fused_scores[m_mask] >= th_fused_p99))
    flag_fused_p95 = int(np.sum(all_fused_scores[m_mask] >= th_fused_p95))

    class_breakdown[m_class] = {
        "total_samples": c_count,
        "flagged_l1": flag_l1,
        "pct_l1": round(flag_l1 / c_count * 100, 2),
        "flagged_l2": flag_l2,
        "pct_l2": round(flag_l2 / c_count * 100, 2),
        "flagged_fused_p99": flag_fused_p99,
        "pct_fused_p99": round(flag_fused_p99 / c_count * 100, 2),
        "flagged_fused_p95": flag_fused_p95,
        "pct_fused_p95": round(flag_fused_p95 / c_count * 100, 2),
    }
    print(f"Class '{m_class}' (N={c_count}):")
    print(f"  Flagged by Layer 1:         {flag_l1}/{c_count} ({class_breakdown[m_class]['pct_l1']}%)")
    print(f"  Flagged by Layer 2:         {flag_l2}/{c_count} ({class_breakdown[m_class]['pct_l2']}%)")
    print(f"  Flagged by Fusion (P99 th): {flag_fused_p99}/{c_count} ({class_breakdown[m_class]['pct_fused_p99']}%)")
    print(f"  Flagged by Fusion (P95 th): {flag_fused_p95}/{c_count} ({class_breakdown[m_class]['pct_fused_p95']}%)")

# 3. Episode-Level Evaluation
print("\n" + "=" * 70)
print("ATTACK EPISODE DETECTION & LATENCY EVALUATION")
print("=" * 70)
# Extract attack episodes in test scenarios
episode_results = []
for ev in test_evidence_list:
    labels = ev.ground_truth_labels
    scores = ev.fused_score
    th = th_fused_p99

    # Find contiguous Attack blocks
    is_attack = (labels == "Attack").astype(int)
    diffs = np.diff(np.pad(is_attack, (1, 1), "constant"))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for ep_idx, (s, e) in enumerate(zip(starts, ends)):
        ep_len = e - s
        ep_scores = scores[s:e]
        flags = ep_scores >= th
        detected = bool(np.any(flags))
        if detected:
            first_det_step = int(np.where(flags)[0][0])
        else:
            first_det_step = None

        episode_results.append({
            "scenario": ev.scenario_name,
            "episode_id": f"{ev.scenario_name}_attack_ep{ep_idx+1}",
            "start_step": int(s),
            "end_step": int(e),
            "length_steps": int(ep_len),
            "detected": detected,
            "first_detection_latency_steps": first_det_step,
        })

total_episodes = len(episode_results)
detected_episodes = sum(1 for ep in episode_results if ep["detected"])
latencies = [ep["first_detection_latency_steps"] for ep in episode_results if ep["detected"] and ep["first_detection_latency_steps"] is not None]

print(f"Total Test Attack Episodes: {total_episodes}")
print(f"Detected Attack Episodes: {detected_episodes}/{total_episodes} ({detected_episodes/total_episodes*100:.2f}%)")
if latencies:
    print(f"Detection Latency (steps): Median = {np.median(latencies):.1f} steps, Mean = {np.mean(latencies):.2f} steps, Min = {min(latencies)}, Max = {max(latencies)}")
    print(f"At estimated 30 Hz sampling rate: Median = {np.median(latencies)/30.0:.3f} s, Mean = {np.mean(latencies)/30.0:.3f} s")

# Save full JSON evaluation
full_eval_artifact = {
    "evaluation_title": "Held-Out Benchmark Evaluation of Triple Synchrophasor Dataset",
    "test_files": loader.TEST_FILES,
    "total_test_sequences": total_test_samples,
    "label_distribution": pd.Series(all_labels).value_counts().to_dict(),
    "sequence_level_evaluations": eval_summary,
    "class_behavior_breakdown": class_breakdown,
    "episode_level_evaluation": {
        "total_attack_episodes": total_episodes,
        "detected_attack_episodes": detected_episodes,
        "detection_rate_pct": round(detected_episodes / total_episodes * 100.0, 2),
        "median_latency_steps": float(np.median(latencies)) if latencies else None,
        "mean_latency_steps": float(np.mean(latencies)) if latencies else None,
        "median_latency_sec_at_30hz": float(np.median(latencies) / 30.0) if latencies else None,
        "episodes": episode_results,
    },
}

with open(REPORTS_DIR / "triple_evaluation.json", "w") as fp:
    json.dump(full_eval_artifact, fp, indent=2)

print("\nSaved evaluation results to reports/triple_evaluation.json")
print("=" * 70)
