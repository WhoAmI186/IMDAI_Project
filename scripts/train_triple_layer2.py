"""Train Triple Layer 2 Physical XGBoost Regressors on NoEvents telemetry."""

import json
from pathlib import Path
import sys
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.triple_loader import TripleDataLoader
from src.ml.triple_layer2_regressors import TripleLayer2Regressors

print("=" * 60)
print("TRAINING TRIPLE LAYER 2 PHYSICAL XGBOOST REGRESSORS")
print("=" * 60)

loader = TripleDataLoader()
train_scenarios = loader.load_partition(loader.TRAIN_FILES)
val_scenarios = loader.load_partition(loader.VAL_FILES)

layer2 = TripleLayer2Regressors(threshold_key="P95")
train_metrics = layer2.fit_normal_training_set(train_scenarios)
layer2.save(MODELS_DIR)
print(f"Layer 2 models and metadata saved to {MODELS_DIR}")

# Evaluate on Validation Normal telemetry (data11..data12 NoEvents)
val_normal_dfs = []
for scen in val_scenarios:
    mask = scen.marker_series == "NoEvents"
    if mask.sum() > 0:
        val_normal_dfs.append(scen.features_df.loc[mask])
import pandas as pd
val_normal_pooled = pd.concat(val_normal_dfs, ignore_index=True)

val_res_out = layer2.compute_residuals(val_normal_pooled)
top2_val, mean_val, max_val = layer2.aggregate_layer2(val_res_out)

print(f"\nValidation Normal Telemetry: N={len(val_normal_pooled)}")
print(f"  TOP-2 MEAN Score: Mean={np.mean(top2_val):.4f}, Std={np.std(top2_val):.4f}, Max={np.max(top2_val):.4f}")

val_exceed = {}
for r in layer2.config:
    raw_res = val_res_out[r]["raw_residual"]
    th = layer2.thresholds[r]["P95"]
    exc_count = int(np.sum(raw_res >= th))
    exc_pct = round(exc_count / len(raw_res) * 100.0, 2)
    val_exceed[r] = {"exceed_count": exc_count, "exceed_pct": exc_pct, "p95_threshold": th}
    print(f"  {r} exceedance over P95 ({th:.4f}): {exc_count}/{len(raw_res)} ({exc_pct}%)")

report_data = {
    "training_metrics": train_metrics,
    "validation_normal_exceedance": val_exceed,
    "val_top2_mean_stats": {
        "mean": float(np.mean(top2_val)),
        "std": float(np.std(top2_val)),
        "p95": float(np.percentile(top2_val, 95.0)),
    },
}

with open(REPORTS_DIR / "triple_layer2_training.json", "w") as fp:
    json.dump(report_data, fp, indent=2)

print("Saved Layer 2 training report to reports/triple_layer2_training.json")
print("=" * 60)
