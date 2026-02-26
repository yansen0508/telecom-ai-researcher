"""
OFDM Channel Data Generation for Complex-Valued Diffusion Training
Generates Rayleigh fading channels with exponential PDP and QPSK pilots
"""

import numpy as np
import os
from tqdm import tqdm


def generate_channel_taps(L=8, tau=2.0, N_samples=1000):
    """
    Generate complex channel taps with exponential power delay profile
    
    Args:
        L: Number of channel taps
        tau: PDP decay constant
        N_samples: Number of channel realizations
    
    Returns:
        h_taps: [N_samples, L] complex channel taps
    """
    # Exponential PDP: P[l] = exp(-l/tau)
    pdp = np.exp(-np.arange(L) / tau)
    pdp = pdp / np.sum(pdp)  # Normalize to unit total power
    
    # Generate i.i.d. complex Gaussian taps
    h_real = np.random.randn(N_samples, L) * np.sqrt(pdp / 2)
    h_imag = np.random.randn(N_samples, L) * np.sqrt(pdp / 2)
    h_taps = h_real + 1j * h_imag
    
    return h_taps


def channel_taps_to_frequency(h_taps, K=64):
    """
    Convert channel taps to frequency response via DFT
    
    Args:
        h_taps: [N_samples, L] channel taps
        K: Number of subcarriers
    
    Returns:
        H: [N_samples, K] channel frequency response
    """
    N_samples, L = h_taps.shape
    
    # Zero-pad to K samples and take DFT
    h_padded = np.zeros((N_samples, K), dtype=complex)
    h_padded[:, :L] = h_taps
    
    H = np.fft.fft(h_padded, axis=1)
    
    return H


def generate_qpsk_symbols(N_samples, K, pilot_positions):
    """
    Generate QPSK symbols for pilots and data
    
    Args:
        N_samples: Number of OFDM symbols
        K: Number of subcarriers
        pilot_positions: Indices of pilot subcarriers
    
    Returns:
        X: [N_samples, K] transmitted symbols (complex)
    """
    # QPSK constellation: {±1±j}/sqrt(2)
    qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    # Generate random QPSK symbols
    symbols_idx = np.random.randint(0, 4, size=(N_samples, K))
    X = qpsk_constellation[symbols_idx]
    
    return X


def add_awgn(Y_clean, snr_db):
    """
    Add AWGN to received signal
    
    Args:
        Y_clean: [N_samples, K] noiseless received signal
        snr_db: SNR in dB
    
    Returns:
        Y_noisy: [N_samples, K] noisy received signal
        noise_var: Noise variance
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(Y_clean)**2)
    
    # Calculate noise variance from SNR
    snr_linear = 10**(snr_db / 10)
    noise_var = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    noise_real = np.random.randn(*Y_clean.shape) * np.sqrt(noise_var / 2)
    noise_imag = np.random.randn(*Y_clean.shape) * np.sqrt(noise_var / 2)
    noise = noise_real + 1j * noise_imag
    
    Y_noisy = Y_clean + noise
    
    return Y_noisy, noise_var


def ls_channel_estimation(Y, X, pilot_positions, K):
    """
    Least Squares channel estimation with linear interpolation
    
    Args:
        Y: [N_samples, K] received signal
        X: [N_samples, K] transmitted symbols
        pilot_positions: Indices of pilot subcarriers
        K: Number of subcarriers
    
    Returns:
        H_ls: [N_samples, K] LS channel estimates
    """
    N_samples = Y.shape[0]
    H_ls = np.zeros((N_samples, K), dtype=complex)
    
    # LS estimation at pilot positions
    H_ls_pilots = Y[:, pilot_positions] / X[:, pilot_positions]
    
    # Linear interpolation to all subcarriers.
    # np.interp only handles real values, so split complex into Re/Im explicitly.
    x_all = np.arange(K)
    for n in range(N_samples):
        re_interp = np.interp(x_all, pilot_positions, H_ls_pilots[n, :].real, period=K)
        im_interp = np.interp(x_all, pilot_positions, H_ls_pilots[n, :].imag, period=K)
        H_ls[n, :] = re_interp + 1j * im_interp
    
    return H_ls


def generate_ofdm_data(N_samples, K=64, P=16, L=8, snr_db=10, seed=None):
    """
    Generate complete OFDM channel estimation dataset
    
    Args:
        N_samples: Number of OFDM symbols
        K: Number of subcarriers
        P: Number of pilots
        L: Number of channel taps
        snr_db: SNR in dB
        seed: Random seed for reproducibility
    
    Returns:
        dict with keys: 'h_true', 'h_ls', 'pilot_mask', 'snr_db'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Comb-type pilot pattern: every 4th subcarrier
    pilot_spacing = K // P
    pilot_positions = np.arange(0, K, pilot_spacing)[:P]
    pilot_mask = np.zeros(K, dtype=bool)
    pilot_mask[pilot_positions] = True
    
    print(f"Generating {N_samples} OFDM symbols with K={K}, P={P}, L={L}, SNR={snr_db}dB")
    
    # Generate channel taps and frequency response
    h_taps = generate_channel_taps(L=L, tau=2.0, N_samples=N_samples)
    h_true = channel_taps_to_frequency(h_taps, K=K)
    
    # Generate transmitted symbols
    X = generate_qpsk_symbols(N_samples, K, pilot_positions)
    
    # Noiseless received signal
    Y_clean = h_true * X
    
    # Add AWGN
    Y_noisy, noise_var = add_awgn(Y_clean, snr_db)
    
    # LS channel estimation
    h_ls = ls_channel_estimation(Y_noisy, X, pilot_positions, K)
    
    return {
        'h_true': h_true,
        'h_ls': h_ls,
        'pilot_mask': pilot_mask,
        'pilot_positions': pilot_positions,
        'snr_db': snr_db,
        'noise_var': noise_var
    }


def convert_to_real_format(h_complex):
    """
    Convert complex channel to [Re, Im] format for PyTorch
    
    Args:
        h_complex: [N, K] complex channel
    
    Returns:
        h_real: [N, 2, K] real format
    """
    N, K = h_complex.shape
    h_real = np.zeros((N, 2, K), dtype=np.float32)
    h_real[:, 0, :] = h_complex.real  # Real part
    h_real[:, 1, :] = h_complex.imag  # Imaginary part
    
    return h_real


def main():
    """Generate training and test datasets"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    K = 64          # Subcarriers
    P = 16          # Pilots
    L = 8           # Channel taps
    N_train_per_snr = 2000  # Training samples per SNR (5 SNRs → 10000 total)
    N_test = 500    # Test samples per SNR (increased from 100 for reliable metrics)

    SNR_values = [5, 10, 15, 20, 25]  # SNR range in dB (no 0dB)

    print("=== OFDM Channel Data Generation ===")
    print(f"K={K} subcarriers, P={P} pilots, L={L} taps")
    print(f"Training: {N_train_per_snr} samples/SNR × {len(SNR_values)} SNRs = {N_train_per_snr * len(SNR_values)} total")
    print(f"Test: {N_test} samples/SNR × {len(SNR_values)} SNRs = {N_test * len(SNR_values)} total")
    print(f"SNR values: {SNR_values} dB")

    # Generate training data (stratified across SNR levels)
    print("\n--- Generating Training Data ---")
    h_true_all = []
    h_ls_all = []
    pilot_mask_ref = None
    pilot_positions_ref = None

    for snr_db in tqdm(SNR_values, desc="Train SNR"):
        data = generate_ofdm_data(
            N_samples=N_train_per_snr,
            K=K, P=P, L=L,
            snr_db=snr_db,
            seed=42 + snr_db
        )
        h_true_all.append(convert_to_real_format(data['h_true']))
        h_ls_all.append(convert_to_real_format(data['h_ls']))
        if pilot_mask_ref is None:
            pilot_mask_ref = data['pilot_mask']
            pilot_positions_ref = data['pilot_positions']
        print(f"  SNR={snr_db}dB: {N_train_per_snr} samples generated")

    # Concatenate and shuffle to mix SNR levels
    h_true_train = np.concatenate(h_true_all, axis=0)  # [N_total, 2, 64]
    h_ls_train = np.concatenate(h_ls_all, axis=0)      # [N_total, 2, 64]

    shuffle_idx = np.random.permutation(len(h_true_train))
    h_true_train = h_true_train[shuffle_idx]
    h_ls_train = h_ls_train[shuffle_idx]

    # --- Normalization ---
    # DDPM assumes data with unit variance. h_true has std ~0.707 (Re/Im each ~0.5 variance).
    # We normalize by the training std so the diffusion schedule is properly calibrated.
    # CRITICAL: h_ls is normalized by the SAME std as h_true, not its own std.
    # This ensures the RePaint conditioning and model input are on the same scale.
    norm_std = float(h_true_train.std())
    h_true_train = h_true_train / norm_std
    h_ls_train   = h_ls_train   / norm_std

    print(f"\nNormalization: std = {norm_std:.4f} (target: 1.0 after normalization)")
    print(f"  h_true std after norm: {h_true_train.std():.4f}")
    print(f"  h_ls   std after norm: {h_ls_train.std():.4f}")

    # Save training data with normalization constant
    np.savez_compressed(
        'train_data.npz',
        h_true=h_true_train,
        h_ls=h_ls_train,
        pilot_mask=pilot_mask_ref,
        pilot_positions=pilot_positions_ref,
        norm_std=np.float32(norm_std)   # saved for test-time denormalization
    )

    print(f"✓ Saved train_data.npz")
    print(f"  h_true shape: {h_true_train.shape}")
    print(f"  h_ls shape:   {h_ls_train.shape}")
    print(f"  pilot_mask shape: {pilot_mask_ref.shape}")
    print(f"  norm_std: {norm_std:.6f}")

    # Generate test data for each SNR
    print("\n--- Generating Test Data ---")
    for snr_db in tqdm(SNR_values, desc="Test SNR"):
        test_data = generate_ofdm_data(
            N_samples=N_test,
            K=K, P=P, L=L,
            snr_db=snr_db,
            seed=100 + snr_db  # Different seed from training
        )
        
        # Convert to real format
        h_true_test = convert_to_real_format(test_data['h_true'])
        h_ls_test = convert_to_real_format(test_data['h_ls'])
        
        # Save test data (raw, unnormalized — evaluate.py applies training norm_std)
        filename = f'test_data_{snr_db}dB.npz'
        np.savez_compressed(
            filename,
            h_true=h_true_test,
            h_ls=h_ls_test,
            pilot_mask=test_data['pilot_mask'],
            pilot_positions=test_data['pilot_positions'],
            snr_db=test_data['snr_db'],
            noise_var=test_data['noise_var'],
            norm_std=np.float32(norm_std)   # same constant as training set
        )
        
        print(f"  ✓ Saved {filename}")
    
    # Validation checks
    print("\n--- Validation Checks ---")

    # Check pilot pattern
    print(f"Pilot positions: {pilot_positions_ref}")
    print(f"Pilot spacing: {pilot_positions_ref[1] - pilot_positions_ref[0]}")
    print(f"Number of pilots: {np.sum(pilot_mask_ref)}")

    # Check training data shape
    print(f"Training h_true shape: {h_true_train.shape} (expect [5000, 2, 64])")
    print(f"Training h_ls shape: {h_ls_train.shape} (expect [5000, 2, 64])")
    
    # Check test data NMSE trends
    print("\nTest data LS NMSE vs SNR:")
    for snr_db in SNR_values:
        test_data = np.load(f'test_data_{snr_db}dB.npz')
        h_true_complex = test_data['h_true'][:, 0, :] + 1j * test_data['h_true'][:, 1, :]
        h_ls_complex = test_data['h_ls'][:, 0, :] + 1j * test_data['h_ls'][:, 1, :]
        
        nmse = np.mean(np.abs(h_ls_complex - h_true_complex)**2) / np.mean(np.abs(h_true_complex)**2)
        nmse_db = 10 * np.log10(nmse)
        
        print(f"  SNR={snr_db:2d}dB: LS NMSE = {nmse_db:6.2f} dB")
    
    print("\n=== Data Generation Complete ===")
    print("Generated files:")
    print("  - train_data.npz")
    for snr_db in SNR_values:
        print(f"  - test_data_{snr_db}dB.npz")


if __name__ == "__main__":
    main()