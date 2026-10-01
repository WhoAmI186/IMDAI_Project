"""Layer 2 Physical Relationships Module for Triple Synchrophasor Dataset.

Models four high-confidence physical conservation laws across the power transmission grid:
1. Bus 1 Voltage Redundancy: R1-PM1:V -> R4-PM1:V (Bus 1 potential equivalence, Volts)
2. Bus 2 Voltage Redundancy: R2-PM1:V -> R3-PM1:V (Bus 2 potential equivalence, Volts)
3. Line 1 Current Conservation: R1-PM4:I -> R2-PM4:I (Series current continuity, Amperes)
4. Line 2 Current Conservation: R4-PM4:I -> R3-PM4:I (Series current continuity, Amperes)

Uses frozen unsupervised XGBoost regressors trained strictly on uncompromised NoEvents operational telemetry.
Aggregates normalized residuals using the finalized TOP-2 MEAN strategy:
    L2_top2 = (s_(1) + s_(2)) / 2
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.triple_loader import TripleScenarioData

TRIPLE_RELATIONSHIPS_CONFIG: Dict[str, Dict[str, Any]] = {
    "bus1_voltage": {
        "name": "Bus 1 Voltage Redundancy",
        "input_feature": "R1-PM1:V",
        "target_feature": "R4-PM1:V",
        "unit": "V",
        "physical_principle": "Kirchhoff's Voltage Law (Equipotential Bus 1 potential equivalence between R1 and R4)",
    },
    "bus2_voltage": {
        "name": "Bus 2 Voltage Redundancy",
        "input_feature": "R2-PM1:V",
        "target_feature": "R3-PM1:V",
        "unit": "V",
        "physical_principle": "Equipotential Bus 2 potential equivalence between R2 and R3",
    },
    "line1_current": {
        "name": "Line 1 Current Conservation",
        "input_feature": "R1-PM4:I",
        "target_feature": "R2-PM4:I",
        "unit": "A",
        "physical_principle": "Kirchhoff's Current Law (Transmission Line 1 series current conservation)",
    },
    "line2_current": {
        "name": "Line 2 Current Conservation",
        "input_feature": "R4-PM4:I",
        "target_feature": "R3-PM4:I",
        "unit": "A",
        "physical_principle": "Kirchhoff's Current Law (Transmission Line 2 series current conservation)",
    },
}


class TripleLayer2Regressors:
    """Manages four frozen XGBoost physical regressors and TOP-2 MEAN aggregation."""

    def __init__(
        self,
        config: Dict[str, Dict[str, Any]] = TRIPLE_RELATIONSHIPS_CONFIG,
        threshold_key: str = "P95",
    ) -> None:
        self.config = config
        self.threshold_key = threshold_key
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}

    def fit_normal_training_set(
        self,
        train_scenarios: List[TripleScenarioData],
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Trains all four XGBoost regressors strictly on uncompromised NoEvents telemetry."""
        # Extract and pool normal training rows from data1..data10
        normal_dfs = []
        for scen in train_scenarios:
            mask = scen.marker_series == "NoEvents"
            if mask.sum() > 0:
                normal_dfs.append(scen.features_df.loc[mask])

        pooled_normal = pd.concat(normal_dfs, ignore_index=True)
        print(f"[TripleLayer2] Training 4 XGBoost regressors on {len(pooled_normal)} normal samples...")

        training_metrics = {}

        for rel_id, rel_info in self.config.items():
            in_col = rel_info["input_feature"]
            tgt_col = rel_info["target_feature"]

            X = pooled_normal[[in_col]].to_numpy(dtype=np.float32)
            y = pooled_normal[tgt_col].to_numpy(dtype=np.float32)

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_seed,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X, y)
            self.models[rel_id] = model

            # Compute residuals on normal training data
            preds = model.predict(X)
            residuals = np.abs(y - preds)

            th_dict = {
                "P95": float(np.percentile(residuals, 95.0)),
                "P99": float(np.percentile(residuals, 99.0)),
                "P99.5": float(np.percentile(residuals, 99.5)),
                "P99.9": float(np.percentile(residuals, 99.9)),
            }
            self.thresholds[rel_id] = th_dict

            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae = float(np.mean(residuals))
            training_metrics[rel_id] = {
                "rmse": rmse,
                "mae": mae,
                "thresholds": th_dict,
                "active_threshold_p95": th_dict["P95"],
            }
            print(f"  {rel_id} ({rel_info['name']}): MAE={mae:.4f} {rel_info['unit']}, RMSE={rmse:.4f}, P95={th_dict['P95']:.4f}")

        return training_metrics

    def compute_residuals(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Computes physical residuals and normalized anomaly scores for each relationship."""
        out: Dict[str, Dict[str, np.ndarray]] = {}

        for rel_id, rel_info in self.config.items():
            model = self.models[rel_id]
            in_col = rel_info["input_feature"]
            tgt_col = rel_info["target_feature"]
            th = self.thresholds[rel_id][self.threshold_key]

            X = df[[in_col]].to_numpy(dtype=np.float32)
            y = df[tgt_col].to_numpy(dtype=np.float32)
            preds = model.predict(X)
            raw_res = np.abs(y - preds)
            signed_res = y - preds
            norm_score = np.minimum(raw_res / th, 1.0)
            flag = (raw_res >= th).astype(int)

            out[rel_id] = {
                "raw_residual": raw_res,
                "signed_residual": signed_res,
                "normalized_score": norm_score,
                "anomaly_flag": flag,
                "threshold": np.full(len(df), th),
            }

        return out

    def aggregate_layer2(self, rel_outputs: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregates individual normalized relationship scores via strictly TOP-2 MEAN.

        Formula:
            L2_top2 = (s_(1) + s_(2)) / 2
        """
        score_matrix = np.column_stack([rel_outputs[r]["normalized_score"] for r in self.config])
        sorted_scores = np.sort(score_matrix, axis=1)[:, ::-1]
        top2_mean = np.mean(sorted_scores[:, :2], axis=1)
        mean_score = np.mean(score_matrix, axis=1)
        max_score = np.max(score_matrix, axis=1)
        return top2_mean, mean_score, max_score

    def predict_scenario(self, scenario: TripleScenarioData, start_idx: int = 59) -> Dict[str, Any]:
        """Predicts scenario and aligns temporally with Layer 1 causal sliding window (start_idx=59)."""
        df = scenario.features_df
        rel_outs = self.compute_residuals(df)
        top2_mean, mean_score, max_score = self.aggregate_layer2(rel_outs)

        # Slice from start_idx: to match Layer 1 window end indices
        aligned_top2 = top2_mean[start_idx:]
        aligned_mean = mean_score[start_idx:]
        aligned_max = max_score[start_idx:]
        aligned_labels = scenario.marker_series.values[start_idx:]
        aligned_indices = scenario.original_indices[start_idx:]

        # Align individual relationships
        aligned_rels = {}
        for r, data in rel_outs.items():
            aligned_rels[r] = {k: v[start_idx:] for k, v in data.items()}

        return {
            "aggregated_score": aligned_top2,
            "mean_score": aligned_mean,
            "max_score": aligned_max,
            "labels": aligned_labels,
            "end_indices": aligned_indices,
            "relationships": aligned_rels,
        }

    def save(self, models_dir: Union[str, Path] = MODELS_DIR) -> None:
        """Saves four XGBoost JSON models and metadata."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        for rel_id, model in self.models.items():
            model_path = models_dir / f"triple_layer2_{rel_id}_xgb.json"
            model.save_model(str(model_path))

        meta = {
            "module": "TripleLayer2Regressors",
            "threshold_key": self.threshold_key,
            "thresholds": self.thresholds,
            "relationships": self.config,
            "aggregation_strategy": "top2_mean",
        }
        with open(models_dir / "triple_layer2_metadata.json", "w") as fp:
            json.dump(meta, fp, indent=2)

    @classmethod
    def load(cls, models_dir: Union[str, Path] = MODELS_DIR) -> TripleLayer2Regressors:
        """Loads four frozen XGBoost models and metadata."""
        models_dir = Path(models_dir)
        with open(models_dir / "triple_layer2_metadata.json", "r") as fp:
            meta = json.load(fp)

        inst = cls(config=meta["relationships"], threshold_key=meta.get("threshold_key", "P95"))
        inst.thresholds = meta["thresholds"]

        for rel_id in inst.config:
            model_path = models_dir / f"triple_layer2_{rel_id}_xgb.json"
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            inst.models[rel_id] = model

        return inst
