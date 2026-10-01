"""Preprocessing and Sequence Generation Module for the Triple Synchrophasor Dataset.

Enforces:
1. Strict selection of the 16 approved Layer 1 physical features.
2. StandardScaler fitted exclusively on uncompromised NoEvents operational telemetry from data1..data10.
3. Strict campaign/file-aware sequence generation (L=60, stride=1) without crossing file boundaries.
4. Clean separation of target labels and metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.triple_loader import TripleDataLoader, TripleScenarioData

# The 16 Authoritative Physical Features for Layer 1
LAYER1_FEATURES_16: List[str] = [
    # Substation 1 (Bus 1) Voltage Phasor
    "R1-PM1:V",   # 1. Bus 1 Voltage Magnitude (Relay 1)
    "R1-PA1:VH",  # 2. Bus 1 Voltage Phase Angle (Relay 1)
    "R4-PM1:V",   # 3. Bus 1 Voltage Magnitude (Relay 4 Redundancy)

    # Substation 2 (Bus 2) Voltage Phasor
    "R2-PM1:V",   # 4. Bus 2 Voltage Magnitude (Relay 2)
    "R2-PA1:VH",  # 5. Bus 2 Voltage Phase Angle (Relay 2)
    "R3-PM1:V",   # 6. Bus 2 Voltage Magnitude (Relay 3 Redundancy)

    # Line 1 Current Phasor
    "R1-PM4:I",   # 7. Line 1 Sending Current Magnitude (Relay 1)
    "R1-PA4:IH",  # 8. Line 1 Sending Current Phase Angle (Relay 1)
    "R2-PM4:I",   # 9. Line 1 Receiving Current Magnitude (Relay 2)
    "R2-PA4:IH",  # 10. Line 1 Receiving Current Phase Angle (Relay 2)

    # Line 2 Current Phasor
    "R4-PM4:I",   # 11. Line 2 Sending Current Magnitude (Relay 4)
    "R3-PM4:I",   # 12. Line 2 Receiving Current Magnitude (Relay 3)

    # System Frequency & ROCOF
    "R1:F",       # 13. Electrical Frequency at Substation 1
    "R1:DF",      # 14. Rate of Change of Frequency (ROCOF) at Substation 1
    "R2:F",       # 15. Electrical Frequency at Substation 2

    # Apparent Impedance Angle
    "R1-PA:ZH",   # 16. Apparent Impedance Phase Angle (Relay 1)
]


@dataclass
class SequenceBatch:
    """Standardized container for sequential sliding window batches."""

    sequences: np.ndarray  # Shape: (N, L=60, D=16)
    labels: np.ndarray  # Shape: (N,) string array ('Attack', 'Natural', 'NoEvents')
    file_origins: np.ndarray  # Shape: (N,) string array (e.g. 'data1.csv')
    end_indices: np.ndarray  # Shape: (N,) original CSV row index at sequence end
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.sequences)


class TripleStandardScaler:
    """StandardScaler with deterministic serialization for 16-channel synchrophasors."""

    def __init__(self, feature_names: List[str] = LAYER1_FEATURES_16) -> None:
        self.feature_names = feature_names
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

    def fit(self, X: pd.DataFrame) -> TripleStandardScaler:
        """Fits scaler strictly on uncompromised normal operational training telemetry."""
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            raise KeyError(f"Missing required features for scaling: {missing}")

        data = X[self.feature_names].to_numpy(dtype=np.float64)
        self.n_samples_seen_ = len(data)
        self.mean_ = np.mean(data, axis=0)
        self.var_ = np.var(data, axis=0)
        # Avoid division by zero with small epsilon
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.scale_ < 1e-8] = 1.0

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame using frozen normal scale statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TripleStandardScaler must be fitted before transforming.")

        data = X[self.feature_names].to_numpy(dtype=np.float64)
        return (data - self.mean_) / self.scale_

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transforms scaled array back to physical power system units."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TripleStandardScaler must be fitted before inverse transform.")
        return (X_scaled * self.scale_) + self.mean_

    def save(self, file_path: Union[str, Path] = MODELS_DIR / "triple_scaler.json") -> None:
        """Serializes scaling statistics to JSON."""
        state = {
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
            "var": self.var_.tolist() if self.var_ is not None else [],
            "n_samples_seen": self.n_samples_seen_,
        }
        with open(file_path, "w") as fp:
            json.dump(state, fp, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path] = MODELS_DIR / "triple_scaler.json") -> TripleStandardScaler:
        """Loads frozen scaling statistics from JSON."""
        with open(file_path, "r") as fp:
            state = json.load(fp)

        scaler = cls(feature_names=state["feature_names"])
        scaler.mean_ = np.array(state["mean"], dtype=np.float64)
        scaler.scale_ = np.array(state["scale"], dtype=np.float64)
        scaler.var_ = np.array(state["var"], dtype=np.float64)
        scaler.n_samples_seen_ = state["n_samples_seen"]
        return scaler


class TriplePreprocessor:
    """End-to-end preprocessor for the Triple Synchrophasor Dataset."""

    def __init__(
        self,
        sequence_length: int = 60,
        stride: int = 1,
        feature_names: List[str] = LAYER1_FEATURES_16,
    ) -> None:
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_names = feature_names
        self.scaler = TripleStandardScaler(feature_names=self.feature_names)
        self.is_fitted = False

    def fit_normal_training_set(self, train_scenarios: List[TripleScenarioData]) -> TriplePreprocessor:
        """Fits scaler strictly on 'NoEvents' rows from training scenarios (data1..data10).

        Zero attack or natural disturbance observations are used for fitting.
        """
        normal_dfs = []
        for scen in train_scenarios:
            no_events_mask = scen.marker_series == "NoEvents"
            if no_events_mask.sum() > 0:
                normal_dfs.append(scen.features_df.loc[no_events_mask, self.feature_names])

        if not normal_dfs:
            raise ValueError("No 'NoEvents' rows found in training scenarios.")

        pooled_normal = pd.concat(normal_dfs, ignore_index=True)
        self.scaler.fit(pooled_normal)
        self.is_fitted = True
        return self

    def create_scenario_sequences(
        self,
        scenario: TripleScenarioData,
        filter_marker: Optional[str] = None,
    ) -> SequenceBatch:
        """Generates sliding sequences for a single scenario file.

        CRITICAL INTEGRITY CONSTRAINTS:
        - Never crosses file boundaries.
        - Each window ending at index t contains contiguous timesteps [t - L + 1 .. t].
        - If filter_marker is provided (e.g. 'NoEvents'), sequences are restricted strictly
          to contiguous intervals of that marker without bridging across event transitions.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on normal training data before generating sequences.")

        scaled_features = self.scaler.transform(scenario.features_df)
        markers = scenario.marker_series.values
        orig_indices = scenario.original_indices
        L = self.sequence_length
        step = self.stride

        seq_list: List[np.ndarray] = []
        label_list: List[str] = []
        file_list: List[str] = []
        end_idx_list: List[int] = []

        total_rows = len(scaled_features)

        if filter_marker is not None:
            # Group into contiguous intervals of filter_marker
            is_target = (markers == filter_marker).astype(int)
            # Find contiguous blocks
            diffs = np.diff(np.pad(is_target, (1, 1), "constant"))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            for start, end in zip(starts, ends):
                block_len = end - start
                if block_len >= L:
                    for i in range(start + L, end + 1, step):
                        seq_list.append(scaled_features[i - L : i])
                        label_list.append(markers[i - 1])
                        file_list.append(scenario.filename)
                        end_idx_list.append(orig_indices[i - 1])
        else:
            # Full continuous sliding window across the entire file
            for i in range(L, total_rows + 1, step):
                seq_list.append(scaled_features[i - L : i])
                label_list.append(markers[i - 1])
                file_list.append(scenario.filename)
                end_idx_list.append(orig_indices[i - 1])

        if seq_list:
            X_seq = np.array(seq_list, dtype=np.float32)
            y_labels = np.array(label_list, dtype=object)
            f_origins = np.array(file_list, dtype=object)
            e_indices = np.array(end_idx_list, dtype=int)
        else:
            X_seq = np.empty((0, L, len(self.feature_names)), dtype=np.float32)
            y_labels = np.empty((0,), dtype=object)
            f_origins = np.empty((0,), dtype=object)
            e_indices = np.empty((0,), dtype=int)

        return SequenceBatch(
            sequences=X_seq,
            labels=y_labels,
            file_origins=f_origins,
            end_indices=e_indices,
            feature_names=self.feature_names,
        )

    def process_partition(
        self,
        scenarios: List[TripleScenarioData],
        filter_marker: Optional[str] = None,
    ) -> SequenceBatch:
        """Processes multiple scenario files, concatenating sequences while strictly preserving per-file integrity."""
        all_seqs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_files: List[np.ndarray] = []
        all_end_indices: List[np.ndarray] = []

        for scen in scenarios:
            batch = self.create_scenario_sequences(scen, filter_marker=filter_marker)
            if len(batch) > 0:
                all_seqs.append(batch.sequences)
                all_labels.append(batch.labels)
                all_files.append(batch.file_origins)
                all_end_indices.append(batch.end_indices)

        if all_seqs:
            X_concat = np.concatenate(all_seqs, axis=0)
            y_concat = np.concatenate(all_labels, axis=0)
            f_concat = np.concatenate(all_files, axis=0)
            e_concat = np.concatenate(all_end_indices, axis=0)
        else:
            X_concat = np.empty((0, self.sequence_length, len(self.feature_names)), dtype=np.float32)
            y_concat = np.empty((0,), dtype=object)
            f_concat = np.empty((0,), dtype=object)
            e_concat = np.empty((0,), dtype=int)

        return SequenceBatch(
            sequences=X_concat,
            labels=y_concat,
            file_origins=f_concat,
            end_indices=e_concat,
            feature_names=self.feature_names,
        )
