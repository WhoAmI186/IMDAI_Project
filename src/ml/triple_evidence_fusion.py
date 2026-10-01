"""Evidence Fusion Module for the Triple Synchrophasor Dataset.

Combines independent evidence sources:
- Layer 1: Temporal Anomaly Evidence (Triple Causal TCN Autoencoder sequence reconstruction error)
- Layer 2: Physical Inconsistency Evidence (Four XGBoost physical relationship residuals with TOP-2 MEAN aggregation)

ARCHITECTURAL PRINCIPLE:
    Synchrophasor Telemetry
              │
        ┌─────┴─────┐
        ▼           ▼
     Layer 1     Layer 2
    (TCN-AE)   (Physical)
        │           │
        └─────┬─────┘
              ▼
       Evidence Fusion
              ▼
        Digital Twin
              ▼
          RAG + LLM
              ▼
          Dashboard

Layer 1 and Layer 2 are independent parallel evidence sources that DO NOT feed into each other.
Evidence Fusion is the FIRST component that combines their outputs:
    S_fused = w1 * L1 + w2 * L2_top2  (w1=0.5, w2=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_ROOT / "models"

from src.data.triple_loader import TripleScenarioData
from src.ml.triple_layer1_detector import TripleLayer1Detector
from src.ml.triple_layer2_regressors import TripleLayer2Regressors


@dataclass
class TripleFusedEvidence:
    """Container for complete explainable fused evidence combining Layer 1 and Layer 2."""

    scenario_name: str
    end_indices: np.ndarray  # Original row indices corresponding to sequence endpoints
    ground_truth_labels: np.ndarray  # 'Attack', 'Natural', 'NoEvents'
    layer1_mse: np.ndarray  # Raw reconstruction MSE
    layer1_normalized: np.ndarray  # min(MSE / threshold_L1, 1.0)
    layer1_threshold: float
    layer1_flag: np.ndarray
    layer2_top2_mean: np.ndarray  # Active TOP-2 MEAN physical score
    layer2_mean: np.ndarray
    layer2_max: np.ndarray
    fused_score: np.ndarray  # w1 * L1 + w2 * L2_top2
    fused_threshold: float
    fused_flag: np.ndarray  # fused_score >= fused_threshold
    agreement_flag: np.ndarray  # L1_flag == 1 AND any(L2_flags) == 1
    weight_l1: float = 0.5
    weight_l2: float = 0.5


class TripleEvidenceFusion:
    """Transparent rule-based evidence fusion engine for Triple synchrophasor dataset."""

    def __init__(
        self,
        layer1_detector: Optional[TripleLayer1Detector] = None,
        layer2_regressors: Optional[TripleLayer2Regressors] = None,
        weight_l1: float = 0.5,
        weight_l2: float = 0.5,
        fused_threshold: Optional[float] = None,
    ) -> None:
        self.layer1 = layer1_detector or TripleLayer1Detector.load()
        self.layer2 = layer2_regressors or TripleLayer2Regressors.load()
        self.weight_l1 = float(weight_l1)
        self.weight_l2 = float(weight_l2)
        self.fused_threshold = fused_threshold

    def calibrate_normal_fused_threshold(
        self,
        normal_train_scenarios: List[TripleScenarioData],
        percentile: float = 99.0,
    ) -> float:
        """Calibrates fused decision threshold strictly on normal training telemetry."""
        fused_scores = []
        for scen in normal_train_scenarios:
            # We filter only contiguous NoEvents sequences
            ev = self.process_scenario(scen)
            normal_mask = ev.ground_truth_labels == "NoEvents"
            if normal_mask.sum() > 0:
                fused_scores.append(ev.fused_score[normal_mask])

        if not fused_scores:
            raise ValueError("No normal sequences found for fused threshold calibration.")

        all_normal_fused = np.concatenate(fused_scores)
        th = float(np.percentile(all_normal_fused, percentile))
        self.fused_threshold = th
        return th

    def process_scenario(
        self,
        scenario: TripleScenarioData,
        fused_threshold_override: Optional[float] = None,
    ) -> TripleFusedEvidence:
        """Processes a single scenario end-to-end through Layer 1, Layer 2, and Evidence Fusion."""
        # 1. Layer 1 inference (generates sliding sequences L=60, stride=1)
        l1_res = self.layer1.predict_scenario(scenario)
        l1_norm = l1_res["normalized_scores"]
        l1_mse = l1_res["sequence_mse"]
        l1_flags = l1_res["anomaly_flags"]
        l1_th = l1_res["threshold"]
        end_indices = l1_res["end_indices"]
        labels = l1_res["labels"]

        # 2. Layer 2 inference (point-in-time residuals, sliced [59:] for exact causal alignment)
        l2_res = self.layer2.predict_scenario(scenario, start_idx=self.layer1.model.sequence_length - 1)
        l2_top2 = l2_res["aggregated_score"]
        l2_mean = l2_res["mean_score"]
        l2_max = l2_res["max_score"]

        # Ensure exact length alignment
        n_samples = min(len(l1_norm), len(l2_top2))
        l1_norm = l1_norm[:n_samples]
        l1_mse = l1_mse[:n_samples]
        l1_flags = l1_flags[:n_samples]
        l2_top2 = l2_top2[:n_samples]
        l2_mean = l2_mean[:n_samples]
        l2_max = l2_max[:n_samples]
        labels = labels[:n_samples]
        end_indices = end_indices[:n_samples]

        # 3. Fuse scores: S = w1 * L1 + w2 * L2_top2
        fused_score = self.weight_l1 * l1_norm + self.weight_l2 * l2_top2
        th = fused_threshold_override or self.fused_threshold or 0.85
        fused_flag = (fused_score >= th).astype(int)

        # Cross-layer agreement: Layer 1 flagged AND at least one Layer 2 relationship flagged
        l2_any_flag = (l2_max >= 1.0).astype(int)
        agreement = (l1_flags & l2_any_flag).astype(int)

        return TripleFusedEvidence(
            scenario_name=scenario.filename,
            end_indices=end_indices,
            ground_truth_labels=labels,
            layer1_mse=l1_mse,
            layer1_normalized=l1_norm,
            layer1_threshold=l1_th,
            layer1_flag=l1_flags,
            layer2_top2_mean=l2_top2,
            layer2_mean=l2_mean,
            layer2_max=l2_max,
            fused_score=fused_score,
            fused_threshold=th,
            fused_flag=fused_flag,
            agreement_flag=agreement,
            weight_l1=self.weight_l1,
            weight_l2=self.weight_l2,
        )
