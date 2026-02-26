"""
Training Script for Complex-Valued Diffusion Models
Trains ComplexUNet and RealUNet with real-time logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time
import datetime
import os
import json

from models import ComplexUNet, RealUNet, DiffusionSchedule


def setup_device():
    """Setup device with MPS fallback to CPU"""
    if torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            # Test MPS with a small tensor
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor
            print(f"✓ Using MPS device")
            return device
        except Exception as e:
            print(f"⚠ MPS failed ({e}), falling back to CPU")
            return torch.device("cpu")
    else:
        print(f"✓ Using CPU device")
        return torch.device("cpu")


def log_message(message, log_file="training.log"):
    """Write message to log file with timestamp and flush immediately"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()  # Critical for real-time tail -f support


def load_training_data():
    """Load and prepare training data"""
    print("Loading training data...")
    
    data = np.load('train_data.npz')
    h_true = torch.from_numpy(data['h_true']).float()  # [N, 2, K]
    h_ls = torch.from_numpy(data['h_ls']).float()      # [N, 2, K]
    pilot_mask = torch.from_numpy(data['pilot_mask']).bool()  # [K]
    
    print(f"Training data shapes:")
    print(f"  h_true: {h_true.shape}")
    print(f"  h_ls: {h_ls.shape}")
    print(f"  pilot_mask: {pilot_mask.shape}")
    
    return h_true, h_ls, pilot_mask


def create_data_loaders(h_true, h_ls, batch_size=64, val_split=0.2):
    """Create train/validation data loaders"""
    N = h_true.shape[0]
    val_size = int(N * val_split)
    train_size = N - val_size
    
    # Random split
    indices = torch.randperm(N)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(h_true[train_indices], h_ls[train_indices])
    val_dataset = TensorDataset(h_true[val_indices], h_ls[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    return train_loader, val_loader


def add_noise(h_true, t, schedule, device):
    """
    Add noise to clean channel according to diffusion schedule
    
    Args:
        h_true: [B, 2, K] clean channel
        t: [B] timesteps
        schedule: DiffusionSchedule object
        device: torch device
    
    Returns:
        h_noisy: [B, 2, K] noisy channel
        noise: [B, 2, K] added noise
    """
    B, _, K = h_true.shape
    
    # Get noise schedule parameters
    params = schedule.get_noise_schedule(t, device)
    alpha_cumprod = params['alpha_cumprod'].view(B, 1, 1)
    
    # Generate complex Gaussian noise
    noise = torch.randn_like(h_true, device=device)
    
    # Add noise: h_t = sqrt(alpha_cumprod) * h_0 + sqrt(1 - alpha_cumprod) * noise
    h_noisy = torch.sqrt(alpha_cumprod) * h_true + torch.sqrt(1 - alpha_cumprod) * noise
    
    return h_noisy, noise


def compute_nmse_db(pred, target):
    """Compute NMSE in dB"""
    mse = torch.mean((pred - target) ** 2)
    signal_power = torch.mean(target ** 2)
    nmse = mse / (signal_power + 1e-8)
    nmse_db = 10 * torch.log10(nmse + 1e-8)
    return nmse_db.item()


def train_epoch(model, train_loader, optimizer, schedule, device, pilot_mask):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # pilot_mask: [K] bool → [1, 1, K] float for broadcasting to [B, 1, K]
    mask_base = pilot_mask.float().to(device).view(1, 1, -1)

    for h_true_batch, h_ls_batch in train_loader:
        h_true_batch = h_true_batch.to(device)
        h_ls_batch   = h_ls_batch.to(device)
        B = h_true_batch.shape[0]

        # Sample random timesteps
        t = torch.randint(0, schedule.T, (B,), device=device)

        # Add noise to clean channel
        h_noisy, noise = add_noise(h_true_batch, t, schedule, device)

        # Build 5-channel input: [h_noisy (2ch), h_ls (2ch), pilot_mask (1ch)]
        # pilot_mask tells the model which h_ls values are real pilot estimates vs. interpolated
        mask_ch = mask_base.expand(B, 1, -1)  # [B, 1, K]
        model_input = torch.cat([h_noisy, h_ls_batch, mask_ch], dim=1)  # [B, 5, K]

        # Predict noise
        optimizer.zero_grad()
        predicted_noise = model(model_input, t)
        
        # MSE loss on noise prediction
        loss = nn.MSELoss()(predicted_noise, noise)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, schedule, device, pilot_mask):
    """Validate model using denoising NMSE"""
    model.eval()
    total_nmse = 0
    num_batches = 0

    mask_base = pilot_mask.float().to(device).view(1, 1, -1)

    with torch.no_grad():
        for h_true_batch, h_ls_batch in val_loader:
            h_true_batch = h_true_batch.to(device)
            h_ls_batch   = h_ls_batch.to(device)
            B = h_true_batch.shape[0]

            # Use moderate noise level for validation (t = T//2)
            t = torch.full((B,), schedule.T // 2, device=device)
            h_noisy, noise = add_noise(h_true_batch, t, schedule, device)

            # Build 5-channel input
            mask_ch = mask_base.expand(B, 1, -1)
            model_input = torch.cat([h_noisy, h_ls_batch, mask_ch], dim=1)  # [B, 5, K]

            # Predict noise and compute denoised estimate
            predicted_noise = model(model_input, t)
            params = schedule.get_noise_schedule(t, device)
            alpha_cumprod = params['alpha_cumprod'].view(B, 1, 1)
            
            # Approximate denoised signal
            h_pred = (h_noisy - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            # Compute NMSE vs ground truth
            nmse_db = compute_nmse_db(h_pred, h_true_batch)
            total_nmse += nmse_db
            num_batches += 1
    
    return total_nmse / num_batches


def train_model(model, model_name, train_loader, val_loader, schedule, device,
                pilot_mask=None, epochs=300, lr=1e-3, patience=30, warmup_epochs=20):
    """
    Train a single model with warmup + cosine-annealing LR and early stopping.

    LR schedule:
      [0, warmup_epochs)  : linear warm-up from lr/10 → lr
      [warmup_epochs, end): cosine annealing from lr → 1e-5

    Returns:
        best_val_nmse: Best validation NMSE achieved
        total_time: Total training time in seconds
    """

    log_message(f"=== Starting {model_name} Training ===")
    log_message(f"Model: {model_name}")
    log_message(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log_message(f"Device: {device}")
    log_message(f"Epochs: {epochs}, LR: {lr}, Warmup: {warmup_epochs}, Patience: {patience}")
    log_message("STATUS: running")

    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Warmup: linear from lr/10 to lr over warmup_epochs
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    # Cosine annealing from lr to eta_min over remaining epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )

    best_val_nmse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        train_loss = train_epoch(model, train_loader, optimizer, schedule, device, pilot_mask)

        # Validation
        val_nmse = validate(model, val_loader, schedule, device, pilot_mask)

        # Advance LR schedule (step-per-epoch, not metric-based)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        improved = False
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            patience_counter = 0
            improved = True
            
            # Save best model
            checkpoint_path = f"{model_name.lower()}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_nmse': val_nmse,
                'train_loss': train_loss
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # Compute epoch time
        epoch_time = time.time() - epoch_start
        
        # Log progress
        log_line = (f"Epoch {epoch+1:3d}/{epochs} | "
                   f"train_loss: {train_loss:.6f} | "
                   f"val_nmse: {val_nmse:6.2f} dB | "
                   f"best: {best_val_nmse:6.2f} dB | "
                   f"patience: {patience_counter:2d}/{patience} | "
                   f"lr: {current_lr:.2e} | "
                   f"time: {epoch_time:4.0f}s")
        
        if improved:
            log_line += " | ** saved best model **"
        
        log_message(log_line)
        print(f"[{model_name}] {log_line}")
        
        # Early stopping
        if patience_counter >= patience:
            log_message(f"Early stopping at epoch {epoch+1}")
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    log_message(f"{model_name} training complete | best_val_nmse: {best_val_nmse:.2f} dB | total_time: {total_time:.0f}s")
    
    return best_val_nmse, total_time


def main():
    """Main training loop"""
    
    # Initialize log file
    log_file = "training.log"
    if os.path.exists(log_file):
        os.remove(log_file)  # Start fresh
    
    print("=== Complex-Valued Diffusion Training ===")
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = setup_device()
    
    # Load data
    h_true, h_ls, pilot_mask = load_training_data()
    train_loader, val_loader = create_data_loaders(h_true, h_ls, batch_size=64)
    
    # Initialize diffusion schedule
    schedule = DiffusionSchedule(T=500)
    
    # Training configuration
    epochs = 300   # increased: conditional model needs more epochs to learn h_ls mapping
    lr = 1e-3
    patience = 30  # increased: give more room before early stopping
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: 64")
    print(f"  Diffusion steps: {schedule.T}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Device: {device}")
    
    # Models to train: 5-channel input = h_noisy[2] + h_ls[2] + pilot_mask[1]
    models_to_train = [
        ("ComplexUNet", ComplexUNet(K=64, in_channels=5)),
        ("RealUNet",    RealUNet(K=64,    in_channels=5)),
    ]
    
    results = {}
    
    # Train each model
    for model_name, model in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"{'='*50}")
        
        model = model.to(device)
        
        try:
            best_val_nmse, total_time = train_model(
                model, model_name, train_loader, val_loader, schedule, device,
                pilot_mask=pilot_mask,
                epochs=epochs, lr=lr, patience=patience, warmup_epochs=20
            )
            
            results[model_name] = {
                'best_val_nmse': best_val_nmse,
                'total_time': total_time,
                'status': 'completed'
            }
            
            print(f"✓ {model_name} training completed")
            print(f"  Best validation NMSE: {best_val_nmse:.2f} dB")
            print(f"  Total time: {total_time:.0f} seconds")
            
        except Exception as e:
            print(f"✗ {model_name} training failed: {e}")
            log_message(f"{model_name} training failed: {e}")
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Final summary
    log_message("STATUS: completed")
    log_message("All models trained successfully.")
    
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    
    for model_name, result in results.items():
        if result['status'] == 'completed':
            print(f"{model_name:12s}: {result['best_val_nmse']:6.2f} dB ({result['total_time']:4.0f}s)")
        else:
            print(f"{model_name:12s}: FAILED")
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete! Check training.log for detailed progress.")
    print(f"✓ Model checkpoints saved: complex_unet.pt, real_unet.pt")
    print(f"✓ Results saved: training_results.json")
    print(f"\nNext steps:")
    print(f"  1. Run: python evaluate.py")
    print(f"  2. Monitor: tail -f training.log")


if __name__ == "__main__":
    main()