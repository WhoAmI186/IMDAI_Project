"""Layer 1 Temporal Anomaly Detector for the Triple Synchrophasor Dataset.

Combines TripleTCNAutoencoder and TriplePreprocessor to produce standardized temporal anomaly evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_ROOT / "models"

from src.data.triple_loader import TripleScenarioData
from src.ml.triple_preprocessor import TriplePreprocessor, TripleStandardScaler
from src.ml.triple_tcn_autoencoder import TripleTCNAutoencoder


class TripleLayer1Detector:
    """Standardized Layer 1 temporal anomaly detector using Causal TCN reconstruction error."""

    def __init__(
        self,
        model: Optional[TripleTCNAutoencoder] = None,
        preprocessor: Optional[TriplePreprocessor] = None,
        thresholds: Optional[Dict[str, float]] = None,
        primary_threshold_key: str = "P99",
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.thresholds = thresholds or {}
        self.primary_threshold_key = primary_threshold_key

    @property
    def primary_threshold(self) -> float:
        if self.primary_threshold_key not in self.thresholds:
            raise KeyError(f"Primary threshold key '{self.primary_threshold_key}' not found in thresholds: {self.thresholds}")
        return self.thresholds[self.primary_threshold_key]

    def calibrate_thresholds(
        self,
        normal_train_sequences: np.ndarray,
        percentiles: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calculates candidate anomaly thresholds strictly from normal training reconstruction errors."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before threshold calibration.")

        if percentiles is None:
            percentiles = {"P95": 95.0, "P99": 99.0, "P99.5": 99.5, "P99.9": 99.9}

        mse_scores = self.model.compute_sequence_mse(normal_train_sequences)
        calibrated = {key: float(np.percentile(mse_scores, p)) for key, p in percentiles.items()}
        self.thresholds = calibrated
        return calibrated

    def predict_scenario(self, scenario: TripleScenarioData) -> Dict[str, Any]:
        """Runs end-to-end Layer 1 anomaly detection on a single scenario."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model and preprocessor must be initialized.")

        # Generate continuous sliding windows across scenario
        batch = self.preprocessor.create_scenario_sequences(scenario)
        if len(batch) == 0:
            return {
                "sequence_mse": np.empty((0,)),
                "normalized_scores": np.empty((0,)),
                "anomaly_flags": np.empty((0,), dtype=int),
                "labels": np.empty((0,), dtype=object),
                "end_indices": np.empty((0,), dtype=int),
                "threshold": self.primary_threshold,
            }

        mse_scores = self.model.compute_sequence_mse(batch.sequences)
        th = self.primary_threshold
        norm_scores = np.minimum(mse_scores / th, 1.0)
        flags = (mse_scores >= th).astype(int)

        return {
            "sequence_mse": mse_scores,
            "normalized_scores": norm_scores,
            "anomaly_flags": flags,
            "labels": batch.labels,
            "end_indices": batch.end_indices,
            "threshold": th,
            "threshold_key": self.primary_threshold_key,
        }

    def save_metadata(self, path: Union[str, Path] = MODELS_DIR / "triple_layer1_metadata.json") -> None:
        """Saves metadata, threshold configuration, and training summary."""
        meta = {
            "model_type": "TripleTCNAutoencoder",
            "checkpoint_path": "models/triple_tcn_autoencoder.pt",
            "sequence_length": self.model.sequence_length if self.model else 60,
            "n_features": self.model.n_features if self.model else 16,
            "feature_names": self.preprocessor.feature_names if self.preprocessor else [],
            "thresholds": self.thresholds,
            "primary_threshold_key": self.primary_threshold_key,
            "primary_threshold_value": self.primary_threshold if self.thresholds else None,
        }
        with open(path, "w") as fp:
            json.dump(meta, fp, indent=2)

    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path] = MODELS_DIR / "triple_tcn_autoencoder.pt",
        scaler_path: Union[str, Path] = MODELS_DIR / "triple_scaler.json",
        metadata_path: Union[str, Path] = MODELS_DIR / "triple_layer1_metadata.json",
        device: Optional[Union[str, Any]] = None,
    ) -> TripleLayer1Detector:
        """Loads frozen detector with checkpoint, scaler, and threshold metadata."""
        model = TripleTCNAutoencoder.load_checkpoint(checkpoint_path, device=device)
        scaler = TripleStandardScaler.load(scaler_path)
        preprocessor = TriplePreprocessor(sequence_length=model.sequence_length, feature_names=scaler.feature_names)
        preprocessor.scaler = scaler
        preprocessor.is_fitted = True

        with open(metadata_path, "r") as fp:
            meta = json.load(fp)

        return cls(
            model=model,
            preprocessor=preprocessor,
            thresholds=meta["thresholds"],
            primary_threshold_key=meta.get("primary_threshold_key", "P99"),
        )
