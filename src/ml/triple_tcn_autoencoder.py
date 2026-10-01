"""PyTorch Causal TCN Autoencoder for Triple Synchrophasor Process Anomaly Detection.

Input: (N, L=60, D=16) -> Causal TCN Encoder -> Latent Bottleneck -> Causal TCN Decoder -> Output: (N, L=60, D=16)

Receptive Field:
    Kernel size k = 3.
    Dilations = [1, 2, 4, 8].
    Receptive field = 1 + 2 * 2 * (1 + 2 + 4 + 8) = 61 timesteps,
    spanning the full 60-timestep window.

Causality:
    Every 1D convolution uses left-padding of (kernel_size - 1) * dilation and zero right-padding.
    Timestep t depends strictly on tau <= t. Zero future leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_ROOT / "models"


class CausalConv1d(nn.Module):
    """1D causal convolution with strict left-padding and right-cropping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalResidualBlock(nn.Module):
    """Residual block of two dilated causal conv layers with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        return out + residual


class TripleTCNAutoencoder(nn.Module):
    """Causal TCN Autoencoder tailored for the 16-channel synchrophasor dataset."""

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 16,
        hidden_channels: int = 32,
        latent_channels: int = 16,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        random_seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4, 8]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._set_deterministic_seeds(self.random_seed)

        # 1. Causal Encoder Path: 16 -> 32 -> 32 -> 32 -> 32 -> 16
        enc_blocks = []
        curr_in = self.n_features
        for d in self.dilations:
            enc_blocks.append(
                TemporalResidualBlock(
                    in_channels=curr_in,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=d,
                    dropout=self.dropout,
                )
            )
            curr_in = self.hidden_channels
        self.encoder_blocks = nn.Sequential(*enc_blocks)

        # Latent bottleneck projection
        self.bottleneck = nn.Conv1d(
            in_channels=self.hidden_channels,
            out_channels=self.latent_channels,
            kernel_size=1,
        )

        # 2. Causal Decoder Path: 16 -> 32 -> 32 -> 32 -> 32 -> 16
        dec_blocks = []
        curr_in = self.latent_channels
        for d in self.dilations:
            dec_blocks.append(
                TemporalResidualBlock(
                    in_channels=curr_in,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=d,
                    dropout=self.dropout,
                )
            )
            curr_in = self.hidden_channels
        self.decoder_blocks = nn.Sequential(*dec_blocks)

        # Output projection back to 16 physical channels
        self.output_proj = nn.Conv1d(
            in_channels=self.hidden_channels,
            out_channels=self.n_features,
            kernel_size=1,
        )

        self.to(self.device)

    def _set_deterministic_seeds(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length=60, n_features=16)
        x_trans = x.transpose(1, 2)  # -> (batch_size, 16, 60)
        enc = self.encoder_blocks(x_trans)
        latent = self.bottleneck(enc)
        dec = self.decoder_blocks(latent)
        out = self.output_proj(dec)
        return out.transpose(1, 2)  # -> (batch_size, 60, 16)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit_model(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Trains autoencoder strictly on uncompromised normal operational sequences."""
        self._set_deterministic_seeds(self.random_seed)
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_tensor = torch.tensor(X_val, dtype=torch.float32)
            val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_weights = None

        if verbose:
            print(f"[TripleTCN] Training on {len(X_train)} normal sequences | Params: {self.count_parameters():,}")

        for ep in range(1, epochs + 1):
            self.train()
            train_losses = []
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for (batch_v,) in val_loader:
                        batch_v = batch_v.to(self.device)
                        recon_v = self(batch_v)
                        v_loss = criterion(recon_v, batch_v)
                        val_losses.append(v_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    history["best_epoch"] = ep
                    epochs_no_improve = 0
                    best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if verbose and (ep % 5 == 0 or ep == 1):
                    print(f"  Epoch {ep:2d}/{epochs} - Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"[TripleTCN] Early stopping triggered at epoch {ep} (Best Val: {best_val_loss:.6f})")
                    break
            else:
                if verbose and (ep % 5 == 0 or ep == 1):
                    print(f"  Epoch {ep:2d}/{epochs} - Train MSE: {avg_train_loss:.6f}")

        if best_weights is not None:
            self.load_state_dict(best_weights)

        return history

    def compute_sequence_mse(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Computes mean squared reconstruction error for each sequence."""
        self.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        mse_scores = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon = self(batch_x)
                # Sequence MSE: mean over (sequence_length, n_features)
                diff = (batch_x - recon) ** 2
                seq_mse = diff.mean(dim=(1, 2)).cpu().numpy()
                mse_scores.append(seq_mse)

        return np.concatenate(mse_scores, axis=0)

    def save_checkpoint(self, path: Union[str, Path] = MODELS_DIR / "triple_tcn_autoencoder.pt") -> None:
        """Saves weights and architecture hyperparams."""
        checkpoint = {
            "state_dict": self.state_dict(),
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "random_seed": self.random_seed,
            "param_count": self.count_parameters(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path] = MODELS_DIR / "triple_tcn_autoencoder.pt",
        device: Optional[Union[str, torch.device]] = None,
    ) -> TripleTCNAutoencoder:
        checkpoint = torch.load(path, map_location="cpu" if device is None else device)
        model = cls(
            sequence_length=checkpoint["sequence_length"],
            n_features=checkpoint["n_features"],
            hidden_channels=checkpoint["hidden_channels"],
            latent_channels=checkpoint["latent_channels"],
            kernel_size=checkpoint["kernel_size"],
            dilations=checkpoint["dilations"],
            dropout=checkpoint.get("dropout", 0.0),
            learning_rate=checkpoint.get("learning_rate", 0.001),
            random_seed=checkpoint.get("random_seed", 42),
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
