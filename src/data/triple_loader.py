"""Triple Dataset Loader for MSU/ORNL Power System Synchrophasor Benchmark.

Responsible for:
- Discovering and ingesting data1.csv through data15.csv from dataset/triple/
- Preserving source file identity and chronological sequence ordering
- Enforcing strict data sanitization:
    * Detecting and removing corrupted rows in data4.csv (indices 3974–3981)
    * Detecting and deduplicating buffer repeats in data8.csv (indices 4287–4294)
- Isolating the ground-truth target label ('marker') from input telemetry features to prevent data leakage
- Providing clean partition-aware data structures for training, validation, and testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "dataset" / "triple"


@dataclass
class CleaningReport:
    """Record of data sanitization operations performed during loading."""

    total_raw_rows: int = 0
    total_clean_rows: int = 0
    corrupted_rows_removed: int = 0
    corrupted_rows_details: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_rows_removed: int = 0
    duplicate_rows_details: List[Dict[str, Any]] = field(default_factory=list)
    files_processed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_raw_rows": self.total_raw_rows,
            "total_clean_rows": self.total_clean_rows,
            "corrupted_rows_removed": self.corrupted_rows_removed,
            "corrupted_rows_details": self.corrupted_rows_details,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "duplicate_rows_details": self.duplicate_rows_details,
            "files_processed": self.files_processed,
        }


@dataclass
class TripleScenarioData:
    """Container for a single scenario file with clean features, metadata, and separated labels."""

    filename: str
    features_df: pd.DataFrame  # Physical/telemetry features (128 columns; NO marker)
    marker_series: pd.Series  # Target event label ('Attack', 'Natural', 'NoEvents')
    original_indices: np.ndarray  # Original row indices in raw CSV for auditability
    raw_row_count: int
    clean_row_count: int


class TripleDataLoader:
    """DataLoader for the 15-file MSU/ORNL Power System synchrophasor dataset."""

    TRAIN_FILES = [f"data{i}.csv" for i in range(1, 11)]  # data1.csv .. data10.csv
    VAL_FILES = ["data11.csv", "data12.csv"]  # data11.csv, data12.csv
    TEST_FILES = ["data13.csv", "data14.csv", "data15.csv"]  # data13.csv, data14.csv, data15.csv

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Triple dataset directory not found at: {self.data_dir}")

        self.cleaning_report = CleaningReport()

    def discover_files(self) -> List[Path]:
        """Discovers all 15 data files sorted in natural numerical order."""
        files = sorted(
            self.data_dir.glob("data*.csv"),
            key=lambda p: int(p.stem.replace("data", "")) if p.stem.replace("data", "").isdigit() else 999,
        )
        if len(files) != 15:
            raise ValueError(f"Expected 15 CSV files in {self.data_dir}, found {len(files)}")
        return files

    def sanitize_scenario(self, df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Sanitizes raw scenario DataFrame without altering original files on disk.

        Actions:
        1. In data4.csv: detect and remove the 8 corrupted rows (indices 3974–3981).
        2. In data8.csv: detect and deduplicate the 8 contiguous duplicate rows (indices 4287–4294).
        3. Strictly separate 'marker' column from physical telemetry features.
        """
        raw_rows = len(df)
        df_work = df.copy()
        df_work["_orig_idx"] = np.arange(raw_rows)

        # 1. Check for data4.csv corrupted rows (known column shift)
        if filename == "data4.csv":
            corrupt_mask = df_work["_orig_idx"].between(3974, 3981)
            n_corrupt = int(corrupt_mask.sum())
            if n_corrupt > 0:
                self.cleaning_report.corrupted_rows_removed += n_corrupt
                self.cleaning_report.corrupted_rows_details.append({
                    "file": filename,
                    "row_indices": df_work.loc[corrupt_mask, "_orig_idx"].tolist(),
                    "reason": "Column-shift data acquisition corruption in testbed recording",
                })
                df_work = df_work.loc[~corrupt_mask].copy()

        # 2. Check for data8.csv duplicate rows
        if filename == "data8.csv":
            # Identify duplicated feature rows excluding metadata
            feature_cols = [c for c in df_work.columns if c != "_orig_idx"]
            dup_mask = df_work.duplicated(subset=feature_cols, keep="first")
            n_dups = int(dup_mask.sum())
            if n_dups > 0:
                self.cleaning_report.duplicate_rows_removed += n_dups
                self.cleaning_report.duplicate_rows_details.append({
                    "file": filename,
                    "row_indices": df_work.loc[dup_mask, "_orig_idx"].tolist(),
                    "reason": "Contiguous identical logger buffer repeat",
                })
                df_work = df_work.loc[~dup_mask].copy()

        # 3. Separate target marker to make accidental feature leakage impossible
        if "marker" not in df_work.columns:
            raise KeyError(f"'marker' column not found in {filename}")

        marker_series = df_work["marker"].copy()
        orig_indices = df_work["_orig_idx"].values
        features_df = df_work.drop(columns=["marker", "_orig_idx"]).copy()

        # Reset working index
        features_df.reset_index(drop=True, inplace=True)
        marker_series.reset_index(drop=True, inplace=True)

        return features_df, marker_series, orig_indices

    def load_scenario(self, filename: str) -> TripleScenarioData:
        """Loads and sanitizes an individual scenario CSV file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")

        raw_df = pd.read_csv(file_path)
        raw_count = len(raw_df)
        self.cleaning_report.total_raw_rows += raw_count

        features_df, marker_series, orig_indices = self.sanitize_scenario(raw_df, filename)
        clean_count = len(features_df)
        self.cleaning_report.total_clean_rows += clean_count
        self.cleaning_report.files_processed.append(filename)

        return TripleScenarioData(
            filename=filename,
            features_df=features_df,
            marker_series=marker_series,
            original_indices=orig_indices,
            raw_row_count=raw_count,
            clean_row_count=clean_count,
        )

    def load_partition(self, filenames: List[str]) -> List[TripleScenarioData]:
        """Loads a list of scenario files preserving per-file integrity."""
        return [self.load_scenario(fname) for fname in filenames]

    def load_all_scenarios(self) -> Dict[str, TripleScenarioData]:
        """Loads and sanitizes all 15 scenarios, returning a dict keyed by filename."""
        files = self.discover_files()
        scenarios: Dict[str, TripleScenarioData] = {}
        for f in files:
            scenarios[f.name] = self.load_scenario(f.name)
        return scenarios

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Returns structured JSON-serializable cleaning statistics."""
        return self.cleaning_report.to_dict()
