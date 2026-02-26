"""
Evaluation Script for Complex-Valued Diffusion Channel Estimation
Evaluates all methods: LS, MMSE, SimpleDNN, RealUNet, ComplexUNet
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import time
import datetime
import warnings
from models import ComplexUNet, RealUNet, SimpleDNN, DiffusionSchedule


def log_message(message, log_file="training.log"):
    """Append message to training.log with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()


def setup_device():
    """Setup device with MPS fallback"""
    if torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor
            return device
        except:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def compute_nmse_db(pred, target):
    """Compute NMSE in dB between complex predictions and targets"""
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Convert to complex if needed
    if pred.shape[1] == 2:  # [N, 2, K] format
        pred_complex = pred[:, 0, :] + 1j * pred[:, 1, :]
        target_complex = target[:, 0, :] + 1j * target[:, 1, :]
    else:
        pred_complex = pred
        target_complex = target
    
    mse = np.mean(np.abs(pred_complex - target_complex)**2)
    signal_power = np.mean(np.abs(target_complex)**2)
    nmse = mse / (signal_power + 1e-12)
    nmse_db = 10 * np.log10(nmse + 1e-12)
    
    return nmse_db


def compute_ber(h_pred, h_true, snr_db, modulation='QPSK'):
    """
    Compute BER after Zero-Forcing equalization
    
    Args:
        h_pred: [N, 2, K] predicted channel
        h_true: [N, 2, K] true channel  
        snr_db: SNR in dB
        modulation: 'QPSK' or '16QAM'
    
    Returns:
        ber: Bit error rate
    """
    if torch.is_tensor(h_pred):
        h_pred = h_pred.detach().cpu().numpy()
    if torch.is_tensor(h_true):
        h_true = h_true.detach().cpu().numpy()
    
    N, _, K = h_pred.shape
    
    # Convert to complex
    h_pred_complex = h_pred[:, 0, :] + 1j * h_pred[:, 1, :]
    h_true_complex = h_true[:, 0, :] + 1j * h_true[:, 1, :]
    
    # Generate random data symbols
    if modulation == 'QPSK':
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        bits_per_symbol = 2
    else:  # 16QAM
        constellation = np.array([
            1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j, 3-1j, 3-3j,
            -1+1j, -1+3j, -3+1j, -3+3j, -1-1j, -1-3j, -3-1j, -3-3j
        ]) / np.sqrt(10)
        bits_per_symbol = 4
    
    total_errors = 0
    total_bits = 0
    
    for n in range(N):
        # Generate random symbols
        data_indices = np.random.randint(0, len(constellation), K)
        data_symbols = constellation[data_indices]
        
        # Simulate transmission through true channel
        received = h_true_complex[n, :] * data_symbols
        
        # Add AWGN
        signal_power = np.mean(np.abs(received)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = (np.random.randn(K) + 1j * np.random.randn(K)) * np.sqrt(noise_power / 2)
        received_noisy = received + noise
        
        # Zero-forcing equalization with predicted channel
        h_pred_safe = h_pred_complex[n, :] + 1e-8  # Avoid division by zero
        equalized = received_noisy / h_pred_safe
        
        # Symbol detection
        distances = np.abs(equalized[:, None] - constellation[None, :])
        detected_indices = np.argmin(distances, axis=1)
        
        # Count bit errors
        for k in range(K):
            true_bits = format(data_indices[k], f'0{bits_per_symbol}b')
            detected_bits = format(detected_indices[k], f'0{bits_per_symbol}b')
            
            bit_errors = sum(t != d for t, d in zip(true_bits, detected_bits))
            total_errors += bit_errors
            total_bits += bits_per_symbol
    
    ber = total_errors / total_bits
    return ber


def ddpm_reverse_sampling(model, h_ls, pilot_mask, schedule, device, num_steps=200):
    """
    DDIM (deterministic) reverse sampling, conditioned on h_ls and pilot_mask.

    Key improvements over the original DDPM+RePaint approach:

    1. No RePaint: The model already receives h_ls via direct concatenation (channels 3-4).
       RePaint's pilot replacement created training-inference distribution mismatch —
       the model was trained on clean h_t values but at inference RePaint injected
       noisy LS observations into pilot positions, producing inputs never seen during training.

    2. DDIM (eta=0) instead of DDPM: Removes the stochastic sigma*noise term at each step.
       DDPM accumulates random noise over 50+ steps; DDIM is deterministic and gives
       better quality with the same number of steps.

    3. 200 steps instead of 50: Each step jumps 2-3 timesteps (vs. 10 for 50 steps),
       reducing the DDIM subsampling approximation error significantly.

    Args:
        model: Trained diffusion model
        h_ls: [N, 2, K] LS channel estimates (normalized)
        pilot_mask: [K] boolean mask for pilot positions (on device)
        schedule: DiffusionSchedule
        device: torch device
        num_steps: Number of DDIM reverse steps (200 gives good quality/speed tradeoff)

    Returns:
        h_pred: [N, 2, K] predicted channels (normalized space)
    """
    model.eval()
    N, _, K = h_ls.shape

    # Start from pure Gaussian noise
    h_t = torch.randn(N, 2, K, device=device, dtype=h_ls.dtype)

    if pilot_mask.device != h_t.device:
        pilot_mask = pilot_mask.to(device)

    # Pilot mask as a constant conditioning channel [N, 1, K]
    mask_ch = pilot_mask.float().view(1, 1, -1).expand(N, 1, -1)

    # Linearly spaced timesteps from T-1 → 0 (subsampled DDIM trajectory)
    timesteps = torch.linspace(schedule.T - 1, 0, num_steps).long()

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = torch.full((N,), t.item(), device=device)

            # 5-channel conditional input: [h_t (2ch), h_ls (2ch), pilot_mask (1ch)]
            model_input = torch.cat([h_t, h_ls, mask_ch], dim=1)  # [N, 5, K]
            predicted_noise = model(model_input, t_batch)

            params = schedule.get_noise_schedule(t_batch, device)
            alpha_cumprod = params['alpha_cumprod'].view(N, 1, 1)

            # Estimate clean signal x0 from current noisy sample
            # Clamp to ±5σ: at high t, 1/sqrt(alpha_cumprod) amplifies errors,
            # but true normalized channel values are O(1) (std=1 after normalization)
            pred_h0 = (h_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            pred_h0 = torch.clamp(pred_h0, -5.0, 5.0)

            if t > 0:
                # DDIM deterministic update (eta=0, no stochastic noise added):
                #   h_{t-1} = sqrt(ᾱ_{t-1}) * pred_h0 + sqrt(1-ᾱ_{t-1}) * ε_θ
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
                params_prev = schedule.get_noise_schedule(
                    torch.full((N,), t_prev.item(), device=device), device
                )
                alpha_cumprod_prev = params_prev['alpha_cumprod'].view(N, 1, 1)

                h_t = (torch.sqrt(alpha_cumprod_prev) * pred_h0
                       + torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise)
            else:
                # Final step (t=0): return the clean estimate directly
                h_t = pred_h0

    return h_t


def compute_mmse_estimate(h_ls, R_hh, noise_var, pilot_mask):
    """
    Compute MMSE channel estimate
    
    Args:
        h_ls: [N, 2, K] LS estimates
        R_hh: [2K, 2K] channel covariance matrix
        noise_var: Noise variance
        pilot_mask: [K] pilot positions
    
    Returns:
        h_mmse: [N, 2, K] MMSE estimates
    """
    N, _, K = h_ls.shape
    
    # Convert to stacked real format [Re; Im]
    h_ls_stacked = np.concatenate([h_ls[:, 0, :], h_ls[:, 1, :]], axis=1)  # [N, 2K]
    
    # Create observation matrix (pilot positions only)
    P = len(pilot_mask.nonzero()[0])
    H_pilot = np.zeros((2*P, 2*K))
    
    pilot_indices = pilot_mask.nonzero()[0]
    for i, p_idx in enumerate(pilot_indices):
        H_pilot[i, p_idx] = 1  # Real part
        H_pilot[i+P, p_idx+K] = 1  # Imaginary part
    
    # MMSE formula: R_hh @ H^T @ (H @ R_hh @ H^T + noise_var * I)^{-1} @ y
    R_yy = H_pilot @ R_hh @ H_pilot.T + noise_var * np.eye(2*P)
    W = R_hh @ H_pilot.T @ np.linalg.pinv(R_yy)
    
    h_mmse_stacked = np.zeros_like(h_ls_stacked)
    
    for n in range(N):
        # Extract pilot observations
        y_pilot = np.concatenate([
            h_ls_stacked[n, pilot_indices],
            h_ls_stacked[n, pilot_indices + K]
        ])
        
        # MMSE estimate
        h_mmse_stacked[n, :] = W @ y_pilot
    
    # Convert back to [N, 2, K] format
    h_mmse = np.stack([
        h_mmse_stacked[:, :K],
        h_mmse_stacked[:, K:]
    ], axis=1)
    
    return h_mmse


def estimate_channel_covariance(h_true_train):
    """Estimate channel covariance from training data"""
    N, _, K = h_true_train.shape
    
    # Convert to stacked format [Re; Im]
    h_stacked = np.concatenate([h_true_train[:, 0, :], h_true_train[:, 1, :]], axis=1)  # [N, 2K]
    
    # Remove mean and compute covariance
    h_mean = np.mean(h_stacked, axis=0)
    h_centered = h_stacked - h_mean
    R_hh = np.cov(h_centered.T)
    
    return R_hh


def train_simple_dnn(h_true, h_ls, device, epochs=200, batch_size=64):
    """Train SimpleDNN on the fly with mini-batch SGD"""
    from torch.utils.data import DataLoader, TensorDataset
    print("Training SimpleDNN...")

    model = SimpleDNN(K=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Convert to tensors
    h_true_tensor = torch.from_numpy(h_true).float().to(device)
    h_ls_tensor = torch.from_numpy(h_ls).float().to(device)

    dataset = TensorDataset(h_ls_tensor, h_true_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 40 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}: loss = {epoch_loss / len(loader):.6f}")

    return model


def validate_results(results, snr_values):
    """Validate results and print warnings for suspicious values"""
    print("\n--- Result Validation ---")
    
    methods = list(results['nmse_db'].keys())
    
    for method in methods:
        nmse_values = results['nmse_db'][method]
        
        # Check for monotonic decrease with SNR
        non_monotonic = False
        for i in range(1, len(nmse_values)):
            if nmse_values[i] > nmse_values[i-1] + 1.0:  # Allow 1dB tolerance
                non_monotonic = True
                break
        
        if non_monotonic:
            warnings.warn(f"{method}: NMSE does not decrease monotonically with SNR")
        
        # Check for NaN/Inf
        if any(np.isnan(v) or np.isinf(v) for v in nmse_values):
            warnings.warn(f"{method}: Contains NaN or Inf values")
        
        # Check for constant values
        if len(set(np.round(nmse_values, 1))) == 1:
            warnings.warn(f"{method}: NMSE values are constant across SNR")
    
    # Check MMSE vs LS
    for i, snr in enumerate(snr_values):
        if results['nmse_db']['MMSE'][i] > results['nmse_db']['LS'][i]:
            warnings.warn(f"SNR={snr}dB: MMSE worse than LS")
    
    print("✓ Validation complete")
    
def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    """Main evaluation loop"""
    
    log_message("=== Starting Evaluation ===")
    print("=== Complex-Valued Diffusion Evaluation ===")
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = setup_device()
    
    # Parameters
    SNR_values = [5, 10, 15, 20, 25]
    K = 64
    
    print(f"Device: {device}")
    print(f"SNR values: {SNR_values} dB")
    
    # Load training data for MMSE covariance estimation and normalization constant
    print("Loading training data for MMSE...")
    train_data = np.load('train_data.npz')
    h_true_train = train_data['h_true']
    h_ls_train   = train_data['h_ls']
    pilot_mask   = train_data['pilot_mask']
    # norm_std: the training-set std used to normalize h_true and h_ls.
    # Must be applied to test h_ls before feeding to the model, and used to
    # denormalize the model output before NMSE/BER computation.
    norm_std = float(train_data['norm_std']) if 'norm_std' in train_data else 1.0
    print(f"Normalization std (from training): {norm_std:.4f}")

    # Estimate channel covariance for MMSE (training data is already normalized)
    print("Estimating channel covariance...")
    R_hh = estimate_channel_covariance(h_true_train)

    # Train SimpleDNN
    simple_dnn = train_simple_dnn(h_true_train, h_ls_train, device)

    # Load diffusion models
    print("Loading diffusion models...")
    schedule = DiffusionSchedule(T=500)

    try:
        complex_unet = ComplexUNet(K=K, in_channels=5).to(device)
        checkpoint = torch.load('complexunet.pt', map_location=device)
        complex_unet.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded ComplexUNet")
    except Exception as e:
        print(f"✗ Failed to load ComplexUNet: {e}")
        complex_unet = None

    try:
        real_unet = RealUNet(K=K, in_channels=5).to(device)
        checkpoint = torch.load('realunet.pt', map_location=device)
        real_unet.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded RealUNet")
    except Exception as e:
        print(f"✗ Failed to load RealUNet: {e}")
        real_unet = None
    
    # Initialize results
    results = {
        'snr_values': SNR_values,
        'nmse_db': {'LS': [], 'MMSE': [], 'DNN': [], 'RealUNet': [], 'ComplexUNet': []},
        'ber': {'LS': [], 'MMSE': [], 'DNN': [], 'RealUNet': [], 'ComplexUNet': []}
    }
    
    # Evaluate at each SNR
    for snr_db in tqdm(SNR_values, desc="Evaluating SNR"):
        print(f"\n--- SNR = {snr_db} dB ---")
        
        # Load test data (raw, un-normalized)
        test_data = np.load(f'test_data_{snr_db}dB.npz')
        h_true    = test_data['h_true']   # [N, 2, K], raw channel values
        h_ls      = test_data['h_ls']     # [N, 2, K], raw LS estimates
        noise_var = test_data['noise_var']

        N = h_true.shape[0]
        print(f"Test samples: {N}")

        # Normalize h_ls by the training norm_std before feeding to diffusion models.
        # MMSE and DNN also use normalized data (they were trained on it).
        # Raw h_true/h_ls are kept for final NMSE/BER computation (scale-invariant NMSE).
        h_ls_norm = h_ls / norm_std   # [N, 2, K] — normalized for model input

        # Method 1: LS Estimator (evaluated on raw data — scale-invariant NMSE)
        nmse_ls = compute_nmse_db(h_ls, h_true)
        ber_ls  = compute_ber(h_ls, h_true, snr_db)
        results['nmse_db']['LS'].append(nmse_ls)
        results['ber']['LS'].append(ber_ls)

        # Method 2: MMSE Estimator
        # R_hh was estimated on normalized training data, h_ls_norm is on same scale
        h_mmse_norm = compute_mmse_estimate(h_ls_norm, R_hh, noise_var / norm_std**2, pilot_mask)
        h_mmse = h_mmse_norm * norm_std   # denormalize for NMSE/BER
        nmse_mmse = compute_nmse_db(h_mmse, h_true)
        ber_mmse  = compute_ber(h_mmse, h_true, snr_db)
        results['nmse_db']['MMSE'].append(nmse_mmse)
        results['ber']['MMSE'].append(ber_mmse)

        # Method 3: SimpleDNN (trained on normalized data → input/output in normalized space)
        h_ls_norm_tensor = torch.from_numpy(h_ls_norm).float().to(device)
        with torch.no_grad():
            h_dnn_norm = simple_dnn(h_ls_norm_tensor)
        h_dnn = h_dnn_norm.cpu().numpy() * norm_std   # denormalize
        nmse_dnn = compute_nmse_db(h_dnn, h_true)
        ber_dnn  = compute_ber(h_dnn, h_true, snr_db)
        results['nmse_db']['DNN'].append(nmse_dnn)
        results['ber']['DNN'].append(ber_dnn)

        # Method 4: RealUNet (conditional diffusion — h_ls normalized, output denormalized)
        if real_unet is not None:
            h_ls_norm_tensor  = torch.from_numpy(h_ls_norm).float().to(device)
            pilot_mask_tensor = torch.from_numpy(pilot_mask).to(device)
            h_real_unet_norm = ddpm_reverse_sampling(
                real_unet, h_ls_norm_tensor, pilot_mask_tensor, schedule, device
            )
            h_real_unet = h_real_unet_norm.cpu().numpy() * norm_std   # denormalize
            nmse_real = compute_nmse_db(h_real_unet, h_true)
            ber_real  = compute_ber(h_real_unet, h_true, snr_db)
        else:
            nmse_real = float('nan')
            ber_real  = float('nan')
        results['nmse_db']['RealUNet'].append(nmse_real)
        results['ber']['RealUNet'].append(ber_real)

        # Method 5: ComplexUNet (conditional diffusion — same normalization as RealUNet)
        if complex_unet is not None:
            h_ls_norm_tensor  = torch.from_numpy(h_ls_norm).float().to(device)
            pilot_mask_tensor = torch.from_numpy(pilot_mask).to(device)
            h_complex_unet_norm = ddpm_reverse_sampling(
                complex_unet, h_ls_norm_tensor, pilot_mask_tensor, schedule, device
            )
            h_complex_unet = h_complex_unet_norm.cpu().numpy() * norm_std   # denormalize
            nmse_complex = compute_nmse_db(h_complex_unet, h_true)
            ber_complex  = compute_ber(h_complex_unet, h_true, snr_db)
        else:
            nmse_complex = float('nan')
            ber_complex  = float('nan')
        results['nmse_db']['ComplexUNet'].append(nmse_complex)
        results['ber']['ComplexUNet'].append(ber_complex)
        
        # Log progress
        log_line = (f"SNR={snr_db:2d}dB | "
                   f"LS:{nmse_ls:6.2f} "
                   f"MMSE:{nmse_mmse:6.2f} "
                   f"DNN:{nmse_dnn:6.2f} "
                   f"Real:{nmse_real:6.2f} "
                   f"Complex:{nmse_complex:6.2f} dB")
        log_message(log_line)
        print(f"  NMSE: {log_line}")
    
    # Validate results
    validate_results(results, SNR_values)
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2, default=convert_to_serializable)
    
    log_message("STATUS: evaluation_completed")
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'SNR (dB)':>8s} {'LS':>8s} {'MMSE':>8s} {'DNN':>8s} {'RealUNet':>10s} {'ComplexUNet':>12s}")
    print("-" * 80)
    
    for i, snr in enumerate(SNR_values):
        print(f"{snr:8d} "
              f"{results['nmse_db']['LS'][i]:8.2f} "
              f"{results['nmse_db']['MMSE'][i]:8.2f} "
              f"{results['nmse_db']['DNN'][i]:8.2f} "
              f"{results['nmse_db']['RealUNet'][i]:10.2f} "
              f"{results['nmse_db']['ComplexUNet'][i]:12.2f}")
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to results.json")
    print(f"✓ Check training.log for detailed progress")


if __name__ == "__main__":
    main()