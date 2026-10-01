"""Train Triple Layer 1 Causal TCN Autoencoder on NoEvents telemetry."""

import json
from pathlib import Path
import numpy as np
import torch

import sys
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.triple_loader import TripleDataLoader
from src.ml.triple_layer1_detector import TripleLayer1Detector
from src.ml.triple_preprocessor import TriplePreprocessor
from src.ml.triple_tcn_autoencoder import TripleTCNAutoencoder

print("=" * 60)
print("TRAINING TRIPLE LAYER 1 CAUSAL TCN AUTOENCODER")
print("=" * 60)

loader = TripleDataLoader()
train_scenarios = loader.load_partition(loader.TRAIN_FILES)
val_scenarios = loader.load_partition(loader.VAL_FILES)

preprocessor = TriplePreprocessor(sequence_length=60, stride=1)
preprocessor.fit_normal_training_set(train_scenarios)
preprocessor.scaler.save(MODELS_DIR / "triple_scaler.json")
print(f"Scaler fitted on {preprocessor.scaler.n_samples_seen_} normal training samples and saved.")

# Extract normal sequences for unsupervised training
train_batch = preprocessor.process_partition(train_scenarios, filter_marker="NoEvents")
val_batch = preprocessor.process_partition(val_scenarios, filter_marker="NoEvents")

X_train = train_batch.sequences
X_val = val_batch.sequences
print(f"Normal Training Sequences: {X_train.shape}")
print(f"Normal Validation Sequences: {X_val.shape}")

# Initialize TCN
model = TripleTCNAutoencoder(
    sequence_length=60,
    n_features=16,
    hidden_channels=32,
    latent_channels=16,
    kernel_size=3,
    dilations=[1, 2, 4, 8],
    dropout=0.0,
    learning_rate=0.001,
    random_seed=42,
)
print(f"Model Architecture Initialized. Parameters: {model.count_parameters():,}")

# Train
history = model.fit_model(
    X_train=X_train,
    X_val=X_val,
    epochs=40,
    batch_size=64,
    patience=8,
    verbose=True,
)

# Save checkpoint
ckpt_path = MODELS_DIR / "triple_tcn_autoencoder.pt"
model.save_checkpoint(ckpt_path)
print(f"Model checkpoint saved to: {ckpt_path}")

# Threshold Calibration strictly on normal training sequences
detector = TripleLayer1Detector(
    model=model,
    preprocessor=preprocessor,
    primary_threshold_key="P99",
)
thresholds = detector.calibrate_thresholds(X_train)
print("\n--- Calibrated Normal Reconstruction Thresholds ---")
for k, v in thresholds.items():
    print(f"  {k}: {v:.8f}")

detector.save_metadata(MODELS_DIR / "triple_layer1_metadata.json")
print("Metadata saved to models/triple_layer1_metadata.json")

# Evaluate normal validation sequences
val_mse = model.compute_sequence_mse(X_val)
val_exceed_p99 = np.sum(val_mse >= thresholds["P99"])
val_exceed_pct = val_exceed_p99 / len(val_mse) * 100.0
print(f"Normal Validation Sequences: N={len(val_mse)}")
print(f"  Val MSE: mean={np.mean(val_mse):.8f}, std={np.std(val_mse):.8f}")
print(f"  Exceedance at P99 ({thresholds['P99']:.8f}): {val_exceed_p99}/{len(val_mse)} ({val_exceed_pct:.2f}%)")

training_summary = {
    "model_type": "TripleTCNAutoencoder",
    "checkpoint": "models/triple_tcn_autoencoder.pt",
    "parameters": model.count_parameters(),
    "sequence_length": 60,
    "n_features": 16,
    "epochs_trained": len(history["train_loss"]),
    "best_epoch": history["best_epoch"],
    "final_train_loss": history["train_loss"][-1],
    "best_val_loss": min(history["val_loss"]),
    "normal_train_sequences": len(X_train),
    "normal_val_sequences": len(X_val),
    "thresholds": thresholds,
    "primary_threshold": thresholds["P99"],
    "val_exceedance_p99_pct": round(val_exceed_pct, 2),
    "history": history,
}

with open(REPORTS_DIR / "triple_layer1_training.json", "w") as fp:
    json.dump(training_summary, fp, indent=2)

print("Training summary saved to reports/triple_layer1_training.json")
print("=" * 60)
