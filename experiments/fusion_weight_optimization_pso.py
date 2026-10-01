"""Controlled Experiment: Unsupervised Layer 1 vs Layer 2 Fusion-Weight Optimization using PSO.

Objective:
    Determine data-driven weights w1 and w2 for Evidence Fusion:
        S_fused = w1 * L1 + w2 * L2_top2
    subject to:
        w1 + w2 = 1, w1 in [0, 1], w2 in [0, 1]

CRITICAL CONSTRAINTS & ARCHITECTURAL INVARIANTS:
1. TOP-2 MEAN is FINAL and FROZEN:
       L2_top2 = (s_(1) + s_(2)) / 2
   Do NOT redo, modify, compare, or optimize Layer 2 aggregation.
2. Layer 1 is FROZEN:
   - Causal TCN Autoencoder (Config A, 14 features, sequence length 60)
   - Active threshold P99 = 0.0008954601059667766
   - Checkpoint: models/tcn_autoencoder_baseline.pt
   - Checkpoint SHA-256: 0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22
3. Layer 2 is FROZEN:
   - Four unsupervised XGBoost physical-relationship regressors
   - Active thresholds P95: PV inverter (7.0729 kW), Wind anemometer (0.5411 m/s),
     Wind temp (0.8566 C), PV thermal (10.3477 C)
4. STRICT ZERO-LEAKAGE UNSUPERVISED OPTIMIZATION:
   - The PSO optimizer uses ONLY uncompromised normal operational training data (20260225_normal).
   - Zero attack data, attack labels, attack timestamps, or attack metrics (F1/F2/PR-AUC)
     are accessed during optimization.
5. EVALUATION:
   - Supervised metrics (F1, F2, PR-AUC, ROC-AUC, FPR, FNR, Episode Detection / 304, Latency)
     are computed exclusively POST-OPTIMIZATION on held-out multi-agent attack campaigns.
6. PRODUCTION WEIGHTS:
   - Active production weights remain 0.5/0.5 until formal decision review.
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
FIGURES_DIR = EXPERIMENTS_DIR / "figures" / "fusion_weights"
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


class UnsupervisedPSOOptimizer:
    """1D Particle Swarm Optimization for Unsupervised Fusion Weight Selection.

    Optimizes w1 in [0, 1] where w2 = 1 - w1, evaluated strictly on uncompromised
    normal training telemetry.
    """

    def __init__(
        self,
        norm_l1: np.ndarray,
        norm_l2_top2: np.ndarray,
        nominal_threshold: float = 0.784338,
        alpha: float = 1.0,
        beta: float = 20.0,
        gamma: float = 0.02,
        n_particles: int = 30,
        n_iterations: int = 40,
        w_inertia: float = 0.7298,
        c1: float = 1.49618,
        c2: float = 1.49618,
        v_max: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.norm_l1 = norm_l1
        self.norm_l2_top2 = norm_l2_top2
        self.nominal_threshold = nominal_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_inertia = w_inertia
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.seed = seed

    def evaluate_fitness(self, w1: float) -> Tuple[float, Dict[str, float]]:
        """Calculates unsupervised multi-criteria loss J(w1) on normal data.

        Formula:
            J(w1) = alpha * Dispersion(S) + beta * TailRisk(S) + gamma * Imbalance(w1)

        where:
            S = w1 * L1 + (1 - w1) * L2_top2
            Dispersion(S) = std(S)
            TailRisk(S) = mean(max(0, S - T0)^2)
            Imbalance(w1) = 4 * (w1 - 0.5)^2
        """
        s = w1 * self.norm_l1 + (1.0 - w1) * self.norm_l2_top2
        dispersion = float(np.std(s))
        tail_excess = np.maximum(0.0, s - self.nominal_threshold)
        tail_risk = float(np.mean(tail_excess ** 2))
        imbalance = float(4.0 * ((w1 - 0.5) ** 2))

        total_loss = self.alpha * dispersion + self.beta * tail_risk + self.gamma * imbalance
        components = {
            "dispersion": dispersion,
            "tail_risk": tail_risk,
            "imbalance": imbalance,
            "weighted_dispersion": self.alpha * dispersion,
            "weighted_tail_risk": self.beta * tail_risk,
            "weighted_imbalance": self.gamma * imbalance,
            "total_loss": total_loss,
        }
        return total_loss, components

    def optimize(self) -> Dict[str, Any]:
        """Executes Particle Swarm Optimization and returns convergence trajectory."""
        np.random.seed(self.seed)

        # Initialize particles uniformly across [0, 1]
        positions = np.random.uniform(0.0, 1.0, self.n_particles)
        velocities = np.random.uniform(-self.v_max, self.v_max, self.n_particles)

        pbest_pos = positions.copy()
        pbest_val = np.array([self.evaluate_fitness(p)[0] for p in positions])

        gbest_idx = int(np.argmin(pbest_val))
        gbest_pos = float(pbest_pos[gbest_idx])
        gbest_val = float(pbest_val[gbest_idx])

        history: List[Dict[str, Any]] = []
        particle_trajectories: List[np.ndarray] = [positions.copy()]

        # Record initial state (Iteration 0)
        history.append({
            "iteration": 0,
            "gbest_w1": gbest_pos,
            "gbest_w2": 1.0 - gbest_pos,
            "gbest_fitness": gbest_val,
            "swarm_mean_fitness": float(np.mean(pbest_val)),
            "swarm_std_fitness": float(np.std(pbest_val)),
            "swarm_mean_position": float(np.mean(positions)),
            "swarm_std_position": float(np.std(positions)),
        })

        for it in range(1, self.n_iterations + 1):
            r1 = np.random.uniform(0.0, 1.0, self.n_particles)
            r2 = np.random.uniform(0.0, 1.0, self.n_particles)

            # Velocity update equation
            velocities = (
                self.w_inertia * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            velocities = np.clip(velocities, -self.v_max, self.v_max)

            # Position update equation
            positions = np.clip(positions + velocities, 0.0, 1.0)
            particle_trajectories.append(positions.copy())

            # Evaluate fitness
            current_vals = np.array([self.evaluate_fitness(p)[0] for p in positions])

            # Update personal bests
            better_mask = current_vals < pbest_val
            pbest_pos[better_mask] = positions[better_mask]
            pbest_val[better_mask] = current_vals[better_mask]

            # Update global best
            if np.min(pbest_val) < gbest_val:
                gbest_idx = int(np.argmin(pbest_val))
                gbest_pos = float(pbest_pos[gbest_idx])
                gbest_val = float(pbest_val[gbest_idx])

            history.append({
                "iteration": it,
                "gbest_w1": gbest_pos,
                "gbest_w2": 1.0 - gbest_pos,
                "gbest_fitness": gbest_val,
                "swarm_mean_fitness": float(np.mean(current_vals)),
                "swarm_std_fitness": float(np.std(current_vals)),
                "swarm_mean_position": float(np.mean(positions)),
                "swarm_std_position": float(np.std(positions)),
            })

        best_loss, best_components = self.evaluate_fitness(gbest_pos)

        return {
            "gbest_w1": gbest_pos,
            "gbest_w2": 1.0 - gbest_pos,
            "gbest_fitness": gbest_val,
            "fitness_components": best_components,
            "history": history,
            "particle_trajectories": particle_trajectories,
            "pso_hyperparameters": {
                "seed": self.seed,
                "n_particles": self.n_particles,
                "n_iterations": self.n_iterations,
                "w_inertia": self.w_inertia,
                "c1": self.c1,
                "c2": self.c2,
                "v_max": self.v_max,
                "alpha_dispersion": self.alpha,
                "beta_tail_risk": self.beta,
                "gamma_imbalance": self.gamma,
                "nominal_threshold": self.nominal_threshold,
            },
        }


def run_experiment() -> Dict[str, Any]:
    print("=" * 70)
    print("CONTROLLED EXPERIMENT: UNSUPERVISED FUSION-WEIGHT OPTIMIZATION (PSO)")
    print("=" * 70)

    # 1. Initialize Active Detectors
    print("\n[Step 1] Loading frozen active detectors...")
    l1_det = load_active_layer1_detector(MODELS_DIR / "tcn_autoencoder_baseline.pt")
    l2_det = load_active_layer2_detector(MODELS_DIR)

    th1 = float(l1_det.primary_threshold)  # P99 = 0.0008954601059667766
    th2_dict = {r: float(l2_det.thresholds[r]["P95"]) for r in l2_det.get_relationship_ids()}
    rel_ids = l2_det.get_relationship_ids()

    print(f"  Layer 1 Active Threshold (P99): {th1:.10f}")
    print(f"  Layer 2 Active Thresholds (P95): {th2_dict}")

    loader = get_default_loader()
    extractor = get_default_process_extractor()

    # 2. Extract Normal Operation Data (20260225_normal)
    print("\n[Step 2] Ingesting uncompromised normal operational telemetry (20260225_normal)...")
    raw_normal = extractor.extract_normal_training_set()
    preprocessor = SolarWindPreprocessor(config_name="config_a", train_ratio=0.8)
    train_raw, val_raw, _ = preprocessor.chronological_split(raw_normal)

    # Compute Layer 1 and Layer 2 normalized signals on Normal Train split
    l1_train_mse = l1_det.process_telemetry(train_raw)["anomaly_scores"]
    l2_train_raw = l2_det.compute_residuals(train_raw)
    l2_train_matrix = np.column_stack([
        np.minimum(l2_train_raw[r]["residual"][59:] / th2_dict[r], 1.0)
        for r in rel_ids
    ])
    sorted_l2_train = np.sort(l2_train_matrix, axis=1)[:, ::-1]
    norm_l2_top2_train = np.mean(sorted_l2_train[:, :2], axis=1)
    norm_l1_train = np.minimum(l1_train_mse / th1, 1.0)

    # Compute Layer 1 and Layer 2 normalized signals on Normal Val split
    l1_val_mse = l1_det.process_telemetry(val_raw)["anomaly_scores"]
    l2_val_raw = l2_det.compute_residuals(val_raw)
    l2_val_matrix = np.column_stack([
        np.minimum(l2_val_raw[r]["residual"][59:] / th2_dict[r], 1.0)
        for r in rel_ids
    ])
    sorted_l2_val = np.sort(l2_val_matrix, axis=1)[:, ::-1]
    norm_l2_top2_val = np.mean(sorted_l2_val[:, :2], axis=1)
    norm_l1_val = np.minimum(l1_val_mse / th1, 1.0)

    print(f"  Normal Train sequences: N={len(norm_l1_train)}")
    print(f"    L1 (P99-norm): mean={np.mean(norm_l1_train):.4f}, std={np.std(norm_l1_train):.4f}, P95={np.percentile(norm_l1_train, 95):.4f}")
    print(f"    L2 (P95-norm): mean={np.mean(norm_l2_top2_train):.4f}, std={np.std(norm_l2_top2_train):.4f}, P95={np.percentile(norm_l2_top2_train, 95):.4f}")
    corr_normal = float(np.corrcoef(norm_l1_train, norm_l2_top2_train)[0, 1])
    print(f"    Cross-layer Pearson correlation (train): r={corr_normal:.4f}")

    print(f"  Normal Validation sequences: N={len(norm_l1_val)}")
    print(f"    L1 (P99-norm): mean={np.mean(norm_l1_val):.4f}, std={np.std(norm_l1_val):.4f}")
    print(f"    L2 (P95-norm): mean={np.mean(norm_l2_top2_val):.4f}, std={np.std(norm_l2_top2_val):.4f}")

    # 3. Unsupervised Weight Optimization via PSO
    print("\n[Step 3] Running strictly unsupervised PSO optimization on normal training telemetry...")
    pso = UnsupervisedPSOOptimizer(
        norm_l1=norm_l1_train,
        norm_l2_top2=norm_l2_top2_train,
        nominal_threshold=0.784338,
        alpha=1.0,
        beta=20.0,
        gamma=0.02,
        n_particles=30,
        n_iterations=40,
        w_inertia=0.7298,
        c1=1.49618,
        c2=1.49618,
        v_max=0.1,
        seed=42,
    )
    pso_res = pso.optimize()

    w1_opt = float(pso_res["gbest_w1"])
    w2_opt = float(pso_res["gbest_w2"])
    fit_opt = float(pso_res["gbest_fitness"])

    print(f"  PSO Convergence completed successfully:")
    print(f"  -> Optimal Frozen Weights: w1* = {w1_opt:.5f}, w2* = {w2_opt:.5f}")
    print(f"  -> Unsupervised Fitness J(w1*): {fit_opt:.6f}")
    print(f"     Components: Dispersion={pso_res['fitness_components']['dispersion']:.6f} "
          f"| TailRisk={pso_res['fitness_components']['tail_risk']:.6f} "
          f"| Imbalance={pso_res['fitness_components']['imbalance']:.6f}")

    # 4. Evaluation of Landscape across w1 in [0, 1]
    print("\n[Step 4] Mapping unsupervised loss landscape over grid of w1 in [0, 1]...")
    w_grid = np.linspace(0.0, 1.0, 201)
    grid_losses: List[float] = []
    grid_dispersion: List[float] = []
    grid_tail_risk: List[float] = []
    grid_imbalance: List[float] = []

    for w in w_grid:
        _, comps = pso.evaluate_fitness(w)
        grid_losses.append(comps["total_loss"])
        grid_dispersion.append(comps["dispersion"])
        grid_tail_risk.append(comps["tail_risk"])
        grid_imbalance.append(comps["imbalance"])

    # 5. Sanity Check on Normal Validation Data
    print("\n[Step 5] Generalization check on held-out normal validation split...")
    T0 = 0.784338
    val_configs = {
        "Baseline 0.5/0.5": 0.5,
        "PSO Optimal": w1_opt,
        "Layer 1 Only (1.0/0.0)": 1.0,
        "Layer 2 Only (0.0/1.0)": 0.0,
    }
    val_stats: Dict[str, Dict[str, float]] = {}
    for cfg_name, w1 in val_configs.items():
        s_val = w1 * norm_l1_val + (1.0 - w1) * norm_l2_top2_val
        val_exceed_count = int(np.sum(s_val >= T0))
        val_exceed_pct = float(round(val_exceed_count / len(s_val) * 100.0, 2))
        val_stats[cfg_name] = {
            "w1": float(round(w1, 5)),
            "w2": float(round(1.0 - w1, 5)),
            "mean": float(round(np.mean(s_val), 4)),
            "std": float(round(np.std(s_val), 4)),
            "p95": float(round(np.percentile(s_val, 95.0), 4)),
            "p99": float(round(np.percentile(s_val, 99.0), 4)),
            "p995": float(round(np.percentile(s_val, 99.5), 4)),
            "exceed_count_t0": val_exceed_count,
            "exceed_pct_t0": val_exceed_pct,
        }
        print(f"  {cfg_name}: Mean={val_stats[cfg_name]['mean']:.4f}, Std={val_stats[cfg_name]['std']:.4f}, "
              f"P95={val_stats[cfg_name]['p95']:.4f}, Exceedance over T0: {val_exceed_count}/{len(s_val)} ({val_exceed_pct}%)")

    # 6. Held-Out Multi-Agent Attack Campaigns Ingestion & Evaluation
    print("\n[Step 6] Ingesting all 5 multi-agent attack campaigns for frozen supervised evaluation...")
    campaign_records: Dict[str, Dict[str, Any]] = {}

    with loader.get_connection("merged", attach_impact=True) as con:
        for run_name, dataset_id in ATTACK_RUNS_CATALOG.items():
            merged_df, ts_array = extract_telemetry(con, dataset_id)
            seq_end_ts = ts_array[59:]

            steps_df = extract_verified_attack_intervals(con, dataset_id)
            qualifying_steps_df = filter_qualifying_attack_intervals(steps_df)
            scope_labels_dict = compute_scope_sequence_labels(seq_end_ts, steps_df, qualifying_steps_df)

            l1_out = l1_det.process_telemetry(merged_df)
            l1_mse = l1_out["anomaly_scores"]

            l2_raw = l2_det.compute_residuals(merged_df)
            l2_matrix = np.column_stack([
                np.minimum(l2_raw[r]["residual"][59:] / th2_dict[r], 1.0)
                for r in rel_ids
            ])
            sorted_l2 = np.sort(l2_matrix, axis=1)[:, ::-1]
            l2_top2 = np.mean(sorted_l2[:, :2], axis=1)
            c_norm_l1 = np.minimum(l1_mse / th1, 1.0)

            campaign_records[run_name] = {
                "dataset_id": dataset_id,
                "seq_end_ts": seq_end_ts,
                "steps_df": steps_df,
                "qualifying_steps_df": qualifying_steps_df,
                "y_scope": scope_labels_dict["labels_scope_qualifying"],
                "clean_mask": scope_labels_dict["clean_negatives_mask"],
                "norm_l1": c_norm_l1,
                "norm_l2_top2": l2_top2,
            }

    # Pooled evaluation sequences
    all_y_scope = np.concatenate([c["y_scope"] for c in campaign_records.values()])
    all_clean = np.concatenate([c["clean_mask"] for c in campaign_records.values()])
    scope_filter = all_clean | (all_y_scope == 1)
    y_eval = all_y_scope[scope_filter]

    all_norm_l1 = np.concatenate([c["norm_l1"] for c in campaign_records.values()])[scope_filter]
    all_norm_l2_top2 = np.concatenate([c["norm_l2_top2"] for c in campaign_records.values()])[scope_filter]

    total_qualifying_episodes = sum(len(c["qualifying_steps_df"]) for c in campaign_records.values())
    print(f"  Total evaluation sequences (in-scope): N={len(y_eval)}, Positives={int(np.sum(y_eval))}, Clean Negatives={int(np.sum(y_eval == 0))}")
    print(f"  Total qualifying impactful attack episodes across 5 campaigns: {total_qualifying_episodes}")

    # 7. Comparative Evaluation across Frozen Configurations
    configs_to_evaluate = {
        "pso_optimized": {
            "name": "PSO Optimized",
            "w1": w1_opt,
            "w2": w2_opt,
            "is_baseline": False,
        },
        "baseline_equal": {
            "name": "Current Baseline (0.5/0.5)",
            "w1": 0.5,
            "w2": 0.5,
            "is_baseline": True,
        },
        "diagnostic_l1_only": {
            "name": "Diagnostic: Layer 1 Only (1.0/0.0)",
            "w1": 1.0,
            "w2": 0.0,
            "is_baseline": True,
        },
        "diagnostic_l2_only": {
            "name": "Diagnostic: Layer 2 Only (0.0/1.0)",
            "w1": 0.0,
            "w2": 1.0,
            "is_baseline": True,
        },
    }

    eval_results: Dict[str, Any] = {}

    print("\n[Step 7] Evaluating frozen configurations on held-out attack benchmark...")
    for cfg_key, cfg_info in configs_to_evaluate.items():
        w1 = cfg_info["w1"]
        w2 = cfg_info["w2"]
        fused_scores = w1 * all_norm_l1 + w2 * all_norm_l2_top2

        # Overall ranking metrics
        roc_auc = float(roc_auc_score(y_eval, fused_scores))
        pr_auc = float(average_precision_score(y_eval, fused_scores))

        # Nominal operational threshold classification metrics
        preds = (fused_scores >= T0).astype(int)
        cm = compute_classification_metrics(y_eval, preds)
        p = float(cm["precision"])
        r = float(cm["recall"])
        f1 = float(cm["f1"])
        f2 = float((5.0 * p * r) / (4.0 * p + r)) if (4.0 * p + r) > 0 else 0.0
        fpr = float(cm["fpr"])
        fnr = float(cm["fnr"])

        # Episode detection latency and rate
        det_eps = 0
        latencies: List[float] = []
        per_campaign_metrics: Dict[str, Dict[str, Any]] = {}

        for c_name, c_data in campaign_records.items():
            q_steps = c_data["qualifying_steps_df"]
            seq_end_ts = c_data["seq_end_ts"]
            c_fused = w1 * c_data["norm_l1"] + w2 * c_data["norm_l2_top2"]
            lat_res = compute_detection_latency(q_steps, seq_end_ts, c_fused, T0)
            det_eps += lat_res["detected_intervals"]
            latencies.extend(lat_res["latencies_array"])

            # Per-campaign sequence metrics
            c_mask = c_data["clean_mask"] | (c_data["y_scope"] == 1)
            c_y = c_data["y_scope"][c_mask]
            c_pred = (c_fused[c_mask] >= T0).astype(int)
            c_cm = compute_classification_metrics(c_y, c_pred)
            c_p = float(c_cm["precision"])
            c_r = float(c_cm["recall"])
            c_f1 = float(c_cm["f1"])
            c_f2 = float((5.0 * c_p * c_r) / (4.0 * c_p + c_r)) if (4.0 * c_p + c_r) > 0 else 0.0

            per_campaign_metrics[c_name] = {
                "sequences": len(c_y),
                "positives": int(np.sum(c_y)),
                "precision": float(round(c_p, 4)),
                "recall": float(round(c_r, 4)),
                "f1": float(round(c_f1, 4)),
                "f2": float(round(c_f2, 4)),
                "fpr": float(round(c_cm["fpr"], 4)),
                "episodes_detected": int(lat_res["detected_intervals"]),
                "total_episodes": len(q_steps),
                "episode_rate_pct": float(round(lat_res["detected_intervals"] / len(q_steps) * 100.0, 2)) if len(q_steps) > 0 else 0.0,
                "median_latency_sec": float(round(np.median(lat_res["latencies_array"]), 4)) if lat_res["latencies_array"] else None,
                "mean_latency_sec": float(round(np.mean(lat_res["latencies_array"]), 4)) if lat_res["latencies_array"] else None,
            }

        ep_rate = float(round(det_eps / total_qualifying_episodes * 100.0, 2))
        med_lat = float(round(np.median(latencies), 4)) if latencies else None
        mean_lat = float(round(np.mean(latencies), 4)) if latencies else None

        eval_results[cfg_key] = {
            "name": cfg_info["name"],
            "w1": float(round(w1, 5)),
            "w2": float(round(w2, 5)),
            "pr_auc": float(round(pr_auc, 4)),
            "roc_auc": float(round(roc_auc, 4)),
            "precision": float(round(p, 4)),
            "recall": float(round(r, 4)),
            "f1": float(round(f1, 4)),
            "f2": float(round(f2, 4)),
            "fpr": float(round(fpr, 4)),
            "fnr": float(round(fnr, 4)),
            "tp": int(cm["tp"]),
            "fp": int(cm["fp"]),
            "tn": int(cm["tn"]),
            "fn": int(cm["fn"]),
            "episodes_detected": int(det_eps),
            "total_episodes": int(total_qualifying_episodes),
            "episode_detection_rate_pct": ep_rate,
            "median_latency_sec": med_lat,
            "mean_latency_sec": mean_lat,
            "per_campaign": per_campaign_metrics,
        }

        print(f"  {cfg_info['name']} (w1={w1:.4f}, w2={w2:.4f}):")
        print(f"    F1={f1:.4f} | F2={f2:.4f} | PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | Precision={p:.4f} | Recall={r:.4f} | FPR={fpr:.4f}")
        print(f"    Episodes Detected: {det_eps}/{total_qualifying_episodes} ({ep_rate}%) | Median Latency={med_lat}s | Mean Latency={mean_lat}s")

    # 8. Visualizations Generation
    print("\n[Step 8] Generating publication-grade visualization figures in reports/experiments/figures/fusion_weights/...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: PSO Convergence Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    iters = [h["iteration"] for h in pso_res["history"]]
    gbest_fits = [h["gbest_fitness"] for h in pso_res["history"]]
    swarm_means = [h["swarm_mean_fitness"] for h in pso_res["history"]]
    gbest_w1s = [h["gbest_w1"] for h in pso_res["history"]]
    swarm_mean_w1s = [h["swarm_mean_position"] for h in pso_res["history"]]

    ax1.plot(iters, gbest_fits, "b-o", linewidth=2, markersize=4, label="Global Best Fitness")
    ax1.plot(iters, swarm_means, "orange", linestyle="--", linewidth=1.5, label="Swarm Mean Fitness")
    ax1.set_title("PSO Fitness Convergence (Unsupervised Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Fitness J(w1)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(iters, gbest_w1s, "g-s", linewidth=2, markersize=4, label="Global Best w1")
    ax2.plot(iters, swarm_mean_w1s, "purple", linestyle="--", linewidth=1.5, label="Swarm Mean w1")
    ax2.axhline(0.5, color="red", linestyle=":", label="Heuristic Baseline (w1=0.5)")
    ax2.set_title("PSO Position Convergence (w1 Parameter)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Weight w1", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "pso_convergence_curve.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  [Fig 1] Saved: {fig1_path.name}")

    # Figure 2: Fitness vs Weight Landscape
    plt.figure(figsize=(10, 6))
    plt.plot(w_grid, grid_losses, "b-", linewidth=2.5, label="Total Loss J(w1)")
    plt.plot(w_grid, grid_dispersion, "g--", linewidth=1.5, label="Normal Score Dispersion (std)")
    plt.plot(w_grid, np.array(grid_tail_risk) * 20.0, "r-.", linewidth=1.5, label=r"Tail Risk Penalty ($20 \times \mathbb{E}[excess^2]$)")
    plt.plot(w_grid, np.array(grid_imbalance) * 0.02, "m:", linewidth=1.5, label=r"Branch Imbalance Penalty ($0.02 \times 4(w_1-0.5)^2$)")
    plt.axvline(w1_opt, color="blue", linestyle="--", linewidth=2, label=f"PSO Optimal w1* = {w1_opt:.4f}")
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Baseline w1 = 0.5")
    plt.title("Unsupervised Fitness Function Landscape vs Layer 1 Weight (w1)", fontsize=13, fontweight="bold")
    plt.xlabel("Layer 1 Fusion Weight (w1)", fontsize=11)
    plt.ylabel("Loss / Objective Value", fontsize=11)
    plt.legend(loc="upper center", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fitness_vs_weight_landscape.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  [Fig 2] Saved: {fig2_path.name}")

    # Figure 3: Normal Validation Fused Score Distribution
    plt.figure(figsize=(10, 5.5))
    s_val_pso = w1_opt * norm_l1_val + w2_opt * norm_l2_top2_val
    s_val_base = 0.5 * norm_l1_val + 0.5 * norm_l2_top2_val
    plt.hist(s_val_base, bins=50, density=True, alpha=0.5, color="steelblue", label=f"Baseline 0.5/0.5 (std={np.std(s_val_base):.4f})")
    plt.hist(s_val_pso, bins=50, density=True, alpha=0.5, color="forestgreen", label=f"PSO Optimal {w1_opt:.3f}/{w2_opt:.3f} (std={np.std(s_val_pso):.4f})")
    plt.axvline(T0, color="darkred", linestyle="--", linewidth=2, label=f"Decision Threshold T0={T0:.4f}")
    plt.title("Normal Validation Fused-Score Empirical Density Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Fused Anomaly Evidence Score S_fused", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "normal_fused_score_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  [Fig 3] Saved: {fig3_path.name}")

    # Figure 4: F1 Comparison
    plt.figure(figsize=(9, 5))
    cfg_names = [eval_results[k]["name"] for k in eval_results]
    f1_vals = [eval_results[k]["f1"] for k in eval_results]
    colors = ["#2ca02c", "#1f77b4", "#7f7f7f", "#bcbd22"]
    bars = plt.bar(cfg_names, f1_vals, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.001, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.title("Held-Out Attack Sequence F1-Score Comparison", fontsize=13, fontweight="bold")
    plt.ylabel("F1 Score (Harmonic Mean)", fontsize=11)
    plt.ylim(0, max(f1_vals) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "f1_comparison.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  [Fig 4] Saved: {fig4_path.name}")

    # Figure 5: F2 Comparison
    plt.figure(figsize=(9, 5))
    f2_vals = [eval_results[k]["f2"] for k in eval_results]
    bars = plt.bar(cfg_names, f2_vals, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.title("Held-Out Attack Sequence F2-Score Comparison (Recall-Emphasized)", fontsize=13, fontweight="bold")
    plt.ylabel("F2 Score (Recall-Weighted)", fontsize=11)
    plt.ylim(0, max(f2_vals) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "f2_comparison.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"  [Fig 5] Saved: {fig5_path.name}")

    # Figure 6: Precision-Recall Curves & PR-AUC Comparison
    plt.figure(figsize=(9, 6))
    for k, col in zip(["pso_optimized", "baseline_equal", "diagnostic_l1_only", "diagnostic_l2_only"], colors):
        w1_k = eval_results[k]["w1"]
        w2_k = eval_results[k]["w2"]
        s_k = w1_k * all_norm_l1 + w2_k * all_norm_l2_top2
        prec_curve, rec_curve, _ = precision_recall_curve(y_eval, s_k)
        plt.plot(rec_curve, prec_curve, color=col, linewidth=2, label=f"{eval_results[k]['name']} (PR-AUC={eval_results[k]['pr_auc']:.4f})")
    plt.title("Precision-Recall Curves across Fusion Weight Configurations", fontsize=13, fontweight="bold")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig6_path = FIGURES_DIR / "pr_auc_comparison.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"  [Fig 6] Saved: {fig6_path.name}")

    # Figure 7: Recall vs FPR Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, col in zip(["pso_optimized", "baseline_equal", "diagnostic_l1_only", "diagnostic_l2_only"], colors):
        w1_k = eval_results[k]["w1"]
        w2_k = eval_results[k]["w2"]
        s_k = w1_k * all_norm_l1 + w2_k * all_norm_l2_top2
        fpr_curve, tpr_curve, _ = roc_curve(y_eval, s_k)
        ax1.plot(fpr_curve, tpr_curve, color=col, linewidth=2, label=f"{eval_results[k]['name']} (AUC={eval_results[k]['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("ROC Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Scatter of operating points at T0
    for k, col in zip(["pso_optimized", "baseline_equal", "diagnostic_l1_only", "diagnostic_l2_only"], colors):
        ax2.scatter(eval_results[k]["fpr"], eval_results[k]["recall"], color=col, s=150, zorder=5, label=f"{eval_results[k]['name']}")
        ax2.annotate(
            f"{eval_results[k]['name']}\n(FPR={eval_results[k]['fpr']:.3f}, Rec={eval_results[k]['recall']:.3f})",
            (eval_results[k]["fpr"], eval_results[k]["recall"]),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
        )
    ax2.set_title("Operating Point at Nominal Threshold T0 = 0.784338", fontsize=12, fontweight="bold")
    ax2.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax2.set_ylabel("Recall (True Positive Rate)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig7_path = FIGURES_DIR / "recall_vs_fpr_comparison.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"  [Fig 7] Saved: {fig7_path.name}")

    # Figure 8: Attack Episode Detection Comparison
    plt.figure(figsize=(10, 5.5))
    ep_counts = [eval_results[k]["episodes_detected"] for k in eval_results]
    ep_rates = [eval_results[k]["episode_detection_rate_pct"] for k in eval_results]
    bars = plt.bar(cfg_names, ep_counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar, rate in zip(bars, ep_rates):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 3, f"{yval}/{total_qualifying_episodes}\n({rate:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.title("Attack Episode Coverage Comparison (304 Qualifying Multi-Agent Episodes)", fontsize=13, fontweight="bold")
    plt.ylabel("Episodes Detected", fontsize=11)
    plt.ylim(0, 304 * 1.25)
    plt.axhline(304, color="black", linestyle=":", alpha=0.7, label="Total Qualifying Episodes (304)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig8_path = FIGURES_DIR / "episode_detection_comparison.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"  [Fig 8] Saved: {fig8_path.name}")

    # Figure 9: Per-Campaign Comparison (PSO vs 0.5/0.5 Baseline)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    campaign_keys = list(ATTACK_RUNS_CATALOG.keys())
    x = np.arange(len(campaign_keys))
    width = 0.35

    pso_f1s = [eval_results["pso_optimized"]["per_campaign"][c]["f1"] for c in campaign_keys]
    base_f1s = [eval_results["baseline_equal"]["per_campaign"][c]["f1"] for c in campaign_keys]

    ax1.bar(x - width / 2, base_f1s, width, label="Baseline 0.5/0.5", color="#1f77b4", edgecolor="black")
    ax1.bar(x + width / 2, pso_f1s, width, label=f"PSO Optimal ({w1_opt:.3f}/{w2_opt:.3f})", color="#2ca02c", edgecolor="black")
    ax1.set_title("Per-Campaign F1-Score Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(campaign_keys, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("F1 Score", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    pso_ep_rates = [eval_results["pso_optimized"]["per_campaign"][c]["episode_rate_pct"] for c in campaign_keys]
    base_ep_rates = [eval_results["baseline_equal"]["per_campaign"][c]["episode_rate_pct"] for c in campaign_keys]

    ax2.bar(x - width / 2, base_ep_rates, width, label="Baseline 0.5/0.5", color="#1f77b4", edgecolor="black")
    ax2.bar(x + width / 2, pso_ep_rates, width, label=f"PSO Optimal ({w1_opt:.3f}/{w2_opt:.3f})", color="#2ca02c", edgecolor="black")
    ax2.set_title("Per-Campaign Episode Detection Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(campaign_keys, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Episode Detection Rate (%)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig9_path = FIGURES_DIR / "per_campaign_comparison.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"  [Fig 9] Saved: {fig9_path.name}")

    # 9. Structure Final Results Artifact
    output_data: Dict[str, Any] = {
        "experiment_title": "Unsupervised Evidence Fusion Weight Optimization using Particle Swarm Optimization (PSO)",
        "timestamp": "2026-10-01",
        "invariants": {
            "layer1_model": "Causal TCN Autoencoder (Config A, 14 features, seq_len=60)",
            "layer1_checkpoint": "models/tcn_autoencoder_baseline.pt",
            "layer1_sha256": "0DAAB40B4096F704CD9DAEA8F703D892DEF7F63D93D8DFC37F58B516FF4A0F22",
            "layer1_threshold_p99": th1,
            "layer2_models": "Four Unsupervised XGBoost Regressors (PV inverter, Wind anemometer, Wind temp, PV thermal)",
            "layer2_thresholds_p95": th2_dict,
            "layer2_aggregation": "TOP-2 MEAN (Strictly Frozen)",
            "active_production_weights": {"w1": 0.5, "w2": 0.5},
        },
        "pso_optimization": {
            "dataset_used": "20260225_normal (strictly uncompromised normal operational training split)",
            "normal_train_samples": len(norm_l1_train),
            "normal_val_samples": len(norm_l1_val),
            "cross_layer_normal_correlation": corr_normal,
            "pso_hyperparameters": pso_res["pso_hyperparameters"],
            "optimal_weights": {
                "w1": w1_opt,
                "w2": w2_opt,
            },
            "unsupervised_fitness": fit_opt,
            "fitness_components": pso_res["fitness_components"],
            "convergence_history": pso_res["history"],
        },
        "normal_validation_check": val_stats,
        "held_out_attack_evaluations": eval_results,
        "figures_generated": [
            str(fig1_path.name),
            str(fig2_path.name),
            str(fig3_path.name),
            str(fig4_path.name),
            str(fig5_path.name),
            str(fig6_path.name),
            str(fig7_path.name),
            str(fig8_path.name),
            str(fig9_path.name),
        ],
    }

    json_path = REPORTS_DIR / "experiments" / "fusion_weight_optimization_pso.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Step 9] Serialized full experiment results to {json_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("=" * 70)
    return output_data


if __name__ == "__main__":
    run_experiment()
