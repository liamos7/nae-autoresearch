"""
train.py — NAE Phase 2 (Energy-Based) Training
================================================
This is the ONLY file the agent modifies.

It contains all hyperparameters, the NAE energy training loop, Langevin sampling,
and the model architecture. The agent can change anything here.

The evaluation harness (evaluate.py) is READ-ONLY and defines the metric.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# HYPERPARAMETERS — The agent should tune these
# ============================================================================

# --- Model ---
LATENT_DIM = 20
SPHERICAL = True

# --- Optimizer ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = "AdamW"  # Options: "AdamW", "Adam", "SGD"
BETAS = (0.9, 0.999)

# --- LR Schedule ---
WARMUP_FRACTION = 0.10      # fraction of total steps for linear warmup
USE_COSINE_DECAY = True

# --- Training ---
BATCH_SIZE = 128
EPOCHS = 25                 # shortened for faster iteration in autoresearch
SEED = 42
GRAD_CLIP = 0.1

# --- NAE Energy Training (Phase 2) ---
GAMMA = 5e-1                # weight for energy^2 regularization
NEG_LAMBDA = 1.0            # weight on negative energy term
L2_WEIGHT = 1e-8            # L2 weight regularization
TEMPERATURE = 1.0
TEMPERATURE_TRAINABLE = False

# --- Langevin Monte Carlo: Latent chain ---
Z_STEP_SIZE = 0.005
Z_NOISE_STD = 0.005
Z_STEPS = 15
Z_USE_METROPOLIS = False

# --- Langevin Monte Carlo: Data chain ---
X_STEP_SIZE = 0.001
X_NOISE_STD = 0.0005
X_STEPS = 10
X_USE_ANNEALING = True
X_ANNEALING_DECAY = 0.5     # how much to reduce step size over the chain

# --- Replay Buffer ---
BUFFER_SIZE = 10000
BUFFER_PROB = 0.95           # probability of drawing from buffer vs fresh noise
BUFFER_REINIT_PROB = 0.05    # probability of reinitializing a buffer entry from noise

# --- Gradient clipping for LMC ---
LMC_GRAD_CLIP = 0.01        # clip LMC gradients as in Du & Mordatch 2019

# --- Early stopping / stability ---
MAX_ENERGY_RATIO = 100.0     # abort if |neg_energy/pos_energy| exceeds this
NAN_PATIENCE = 5             # abort after this many consecutive NaN steps

# ============================================================================
# ENCODER / DECODER — The agent can modify architecture
# ============================================================================

# --- MNIST / FashionMNIST architecture (input: 1x28x28) ---

class Encoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(128, latent_dim),
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 8 * 4 * 4),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Unflatten(dim=1, unflattened_size=(8, 4, 4)),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, out_channels, 3, stride=1, padding="valid"),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


# --- CICADA architecture (input: 1x18x14, LHC calorimeter data) ---

class CicadaEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(14 * 10 * 16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
        )
    def forward(self, x):
        return self.net(x)


class CicadaDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 14 * 10 * 16),
            nn.LeakyReLU(),
            nn.Unflatten(1, (16, 14, 10)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=0),
        )
    def forward(self, x):
        return self.net(x)


def build_encoder_decoder(dataset):
    """Build dataset-appropriate encoder/decoder pair."""
    if dataset == "CICADA":
        return CicadaEncoder(LATENT_DIM), CicadaDecoder(LATENT_DIM)
    elif dataset == "CIFAR10":
        return Encoder(LATENT_DIM, in_channels=3), Decoder(LATENT_DIM, out_channels=3)
    else:  # MNIST, FMNIST
        return Encoder(LATENT_DIM, in_channels=1), Decoder(LATENT_DIM, out_channels=1)


# ============================================================================
# NAE WITH ENERGY TRAINING
# ============================================================================

class NAEWithEnergyTraining(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Temperature
        temp_log = np.log(TEMPERATURE)
        if TEMPERATURE_TRAINABLE:
            self.register_parameter('temperature_log', nn.Parameter(torch.tensor(temp_log, dtype=torch.float)))
        else:
            self.register_buffer('temperature_log', torch.tensor(temp_log, dtype=torch.float))
        
        # Replay buffer state
        self._replay_buffer = None
        self._buffer_ptr = 0

    @property
    def temperature(self):
        return torch.exp(self.temperature_log)

    def normalize_z(self, z):
        if SPHERICAL:
            return z / z.view(len(z), -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
        return z

    def encode(self, x):
        return self.normalize_z(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def energy(self, x):
        """Normalized L2 reconstruction error per dimension."""
        recon = self(x)
        D = float(np.prod(x.shape[1:]))
        error = ((x - recon) ** 2).view(len(x), -1).sum(dim=1)
        return error / D

    def energy_from_z(self, z):
        """Energy of decoded latent (for LMC in latent space)."""
        x_recon = self.decode(z)
        # re-encode and decode to get reconstruction error
        recon = self(x_recon)
        D = float(np.prod(x_recon.shape[1:]))
        error = ((x_recon - recon) ** 2).view(len(x_recon), -1).sum(dim=1)
        return error / D

    # ------------------------------------------------------------------
    # Replay Buffer
    # ------------------------------------------------------------------

    def seed_buffer(self, train_loader, device):
        """Seed buffer with encoded training data for stable initialization."""
        self.eval()
        z_list = []
        with torch.no_grad():
            for x, _ in train_loader:
                z = self.encode(x.to(device))
                z_list.append(z.cpu())
                if sum(z.shape[0] for z in z_list) >= BUFFER_SIZE:
                    break
        z_all = torch.cat(z_list, dim=0)[:BUFFER_SIZE]
        if z_all.size(0) < BUFFER_SIZE:
            extra = torch.randn(BUFFER_SIZE - z_all.size(0), LATENT_DIM)
            if SPHERICAL:
                extra = F.normalize(extra, dim=-1)
            z_all = torch.cat([z_all, extra], dim=0)
        self._replay_buffer = z_all.detach()
        self.train()

    def _sample_latent_init(self, batch_size, device):
        if self._replay_buffer is None:
            z = torch.randn(BUFFER_SIZE, LATENT_DIM)
            if SPHERICAL:
                z = F.normalize(z, dim=-1)
            self._replay_buffer = z

        idx = torch.randint(0, BUFFER_SIZE, (batch_size,))
        buffer_samples = self._replay_buffer[idx].to(device)

        fresh = torch.randn(batch_size, LATENT_DIM, device=device)
        if SPHERICAL:
            fresh = F.normalize(fresh, dim=-1)

        use_buffer = (torch.rand(batch_size, device=device) < BUFFER_PROB).unsqueeze(-1)
        return torch.where(use_buffer, buffer_samples, fresh).detach()

    def _update_buffer(self, z_final):
        z_final = z_final.detach().cpu()
        bs = len(z_final)
        end = self._buffer_ptr + bs
        if end <= BUFFER_SIZE:
            self._replay_buffer[self._buffer_ptr:end] = z_final
        else:
            first = BUFFER_SIZE - self._buffer_ptr
            self._replay_buffer[self._buffer_ptr:] = z_final[:first]
            self._replay_buffer[:end - BUFFER_SIZE] = z_final[first:]
        self._buffer_ptr = end % BUFFER_SIZE

    # ------------------------------------------------------------------
    # Langevin Monte Carlo (On-Manifold Initialization)
    # ------------------------------------------------------------------

    def langevin_sample(self, x_init):
        """Two-stage OMI sampling: latent chain → data chain."""
        batch_size = x_init.shape[0]
        device = x_init.device

        # Stage 1: Latent Chain
        z = self._sample_latent_init(batch_size, device).requires_grad_(True)
        for step in range(Z_STEPS):
            e = self.energy(self.decode(z)).sum()
            grad = torch.autograd.grad(e, z)[0]
            
            # Clip LMC gradients
            if LMC_GRAD_CLIP > 0:
                grad = grad.clamp(-LMC_GRAD_CLIP, LMC_GRAD_CLIP)
            
            with torch.no_grad():
                z = z - Z_STEP_SIZE * grad + torch.randn_like(z) * Z_NOISE_STD
                if SPHERICAL:
                    z = F.normalize(z, dim=-1)
            z = z.detach().requires_grad_(True)

        self._update_buffer(z)

        # Stage 2: Data Chain
        x = self.decode(z).detach().requires_grad_(True)
        for step in range(X_STEPS):
            e = self.energy(x).sum()
            grad = torch.autograd.grad(e, x)[0]
            
            if LMC_GRAD_CLIP > 0:
                grad = grad.clamp(-LMC_GRAD_CLIP, LMC_GRAD_CLIP)
            
            with torch.no_grad():
                step_size = X_STEP_SIZE
                if X_USE_ANNEALING:
                    step_size *= (1.0 - (step / max(X_STEPS, 1)) * X_ANNEALING_DECAY)
                noise_std = X_NOISE_STD
                x = x - step_size * grad + torch.randn_like(x) * noise_std
                x = x.clamp(0, 1)  # Safe for normalized image data; agent can remove for CICADA
            x = x.detach().requires_grad_(True)

        return x.detach()

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def train_step(self, x, optimizer):
        optimizer.zero_grad()

        # Positive energy (training data)
        pos_energy = self.energy(x)

        # Negative samples via OMI + LMC
        x_neg = self.langevin_sample(x)
        neg_energy = self.energy(x_neg)

        # Contrastive divergence loss
        cd_loss = pos_energy.mean() - NEG_LAMBDA * neg_energy.mean()

        # Energy regularization (prevent divergence)
        reg_loss = GAMMA * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())

        # Weight decay
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * L2_WEIGHT

        loss = (cd_loss + reg_loss) / self.temperature + l2_loss

        if torch.isnan(loss):
            return None  # signal NaN

        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        return {
            'loss': loss.item(),
            'pos_energy': pos_energy.mean().item(),
            'neg_energy': neg_energy.mean().item(),
            'energy_diff': (pos_energy.mean() - neg_energy.mean()).item(),
        }


# ============================================================================
# LR SCHEDULE
# ============================================================================

def get_scheduler(optimizer, total_steps):
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if USE_COSINE_DECAY:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(dataset, holdout_class, pretrained_path, data_root, output_dir):
    """
    Run NAE Phase 2 training. Returns a dict of metrics for evaluate.py to consume.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build model ---
    encoder, decoder = build_encoder_decoder(dataset)
    model = NAEWithEnergyTraining(encoder, decoder).to(device)

    # --- Load pretrained AE weights (Phase 1) ---
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    # --- Data ---
    # Import from fastad (assumed to be on PYTHONPATH or in parent directory)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fastad.datasets import get_loaders
    train_loader, val_loader = get_loaders(
        hold_out_classes=holdout_class, batch_size=BATCH_SIZE,
        ds_name=dataset, n_max=50000, root=data_root,
    )

    # --- Seed replay buffer ---
    model.seed_buffer(train_loader, device)

    # --- Optimizer ---
    if OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                       weight_decay=WEIGHT_DECAY, betas=BETAS)
    elif OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY, betas=BETAS)
    elif OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(optimizer, total_steps)

    # --- Training ---
    history = {
        'train_loss': [], 'pos_energy': [], 'neg_energy': [],
        'val_auc': [], 'val_loss': [],
    }
    best_val_auc = 0.0
    best_model_state = None
    nan_count = 0
    collapsed = False

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for x, y in train_loader:
            x = x.to(device)
            result = model.train_step(x, optimizer)

            if result is None:
                nan_count += 1
                if nan_count >= NAN_PATIENCE:
                    collapsed = True
                    break
                continue
            else:
                nan_count = 0

            scheduler.step()
            epoch_losses.append(result['loss'])

            # Stability check
            if abs(result['neg_energy']) > MAX_ENERGY_RATIO * max(abs(result['pos_energy']), 1e-6):
                collapsed = True
                break

        if collapsed:
            print(f"COLLAPSED at epoch {epoch+1}")
            break

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        history['train_loss'].append(avg_loss)
        history['pos_energy'].append(result['pos_energy'] if result else float('nan'))
        history['neg_energy'].append(result['neg_energy'] if result else float('nan'))

        # --- Validation (compute AUC) ---
        model.eval()
        all_scores = []
        all_labels = []
        val_losses = []

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                scores = model.energy(val_x)
                all_scores.append(scores.cpu())
                all_labels.append(val_y)
                # val loss = mean energy on inliers
                inlier_mask = (val_y == 0)
                if inlier_mask.any():
                    val_losses.append(scores[inlier_mask.to(device)].mean().item())

        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()

        from sklearn.metrics import roc_auc_score
        try:
            # AUC: higher energy for holdout (label=1) is good
            has_pos = (all_labels == 1).any()
            has_neg = (all_labels == 0).any()
            if has_pos and has_neg:
                auc = roc_auc_score(all_labels > 0, all_scores)
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        val_loss = np.mean(val_losses) if val_losses else float('nan')
        history['val_auc'].append(auc)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | "
              f"pos_E={history['pos_energy'][-1]:.4f} | neg_E={history['neg_energy'][-1]:.4f} | "
              f"val_auc={auc:.4f} | val_loss={val_loss:.4f}")

        if auc > best_val_auc:
            best_val_auc = auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - start_time

    # --- Save best model ---
    if best_model_state is not None and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "model_best.pkl")
        torch.save({'model_state': best_model_state, 'val_auc': best_val_auc}, save_path)

    # --- Final metrics ---
    metrics = {
        'best_val_auc': best_val_auc,
        'final_val_auc': history['val_auc'][-1] if history['val_auc'] else 0.0,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else float('nan'),
        'final_pos_energy': history['pos_energy'][-1] if history['pos_energy'] else float('nan'),
        'final_neg_energy': history['neg_energy'][-1] if history['neg_energy'] else float('nan'),
        'collapsed': collapsed,
        'epochs_completed': len(history['train_loss']),
        'elapsed_seconds': elapsed,
        'energy_stable': not collapsed and not any(np.isnan(history['train_loss'])),
    }

    # Write metrics for evaluate.py
    metrics_path = os.path.join(output_dir or '.', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CICADA",
                        choices=["MNIST", "FMNIST", "CIFAR10", "CICADA"])
    parser.add_argument("--holdout-class", type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated list of holdout classes (e.g. '1,2')")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="/scratch/network/lo8603/thesis/fast-ad/data/h5_files/")
    parser.add_argument("--output-dir", type=str, default="./autoresearch_output")
    args = parser.parse_args()

    # Parse holdout class(es)
    holdout_str = args.holdout_class
    if ',' in holdout_str:
        holdout = [int(x.strip()) for x in holdout_str.split(',')]
    else:
        holdout = int(holdout_str)

    metrics = train(args.dataset, holdout, args.pretrained_path,
                    args.data_root, args.output_dir)

    print("\n=== FINAL METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")