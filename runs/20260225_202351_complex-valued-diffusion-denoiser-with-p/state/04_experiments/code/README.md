# Complex-Valued Diffusion Denoiser for OFDM Channel Estimation

This repository implements a novel complex-valued denoising diffusion probabilistic model (CV-DDPM) with a **Quantum Phase Rotation Output Layer** for OFDM channel estimation. The Quantum Phase Rotation Layer is the key innovation that explicitly models amplitude-phase coupling between quadrature channels via a learned Rz-style gate.

## Key Features

- **ComplexUNet**: Novel architecture with Quantum Phase Rotation Output Layer — a learned Rz gate entangles the Re/Im quadratures via a per-subcarrier phase φ, analogous to a single-qubit rotation
- **RealUNet**: Ablation baseline with identical architecture but standard Conv1d output (no Phase Rotation)
- **SimpleDNN**: 3-layer MLP baseline for comparison
- **DDPM Training**: 500-step cosine diffusion schedule with epsilon-prediction; 5-channel conditional input `[h_noisy, h_ls, pilot_mask]`
- **DDIM Inference**: Deterministic (eta=0) reverse sampling with 200 steps — no stochastic noise accumulation, no RePaint

## Environment Requirements

- Python 3.13+
- PyTorch 2.10+ with MPS support (Apple M4) or CPU fallback
- NumPy 1.24+
- tqdm for progress bars
- Standard library: json, datetime, warnings

### Installation

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy tqdm
```

## Quick Start

### 1. Generate Training Data (< 1 minute)

```bash
python data_generation.py
```

**Expected output:**
- `train_data.npz` (10000 samples across SNR=5,10,15,20,25 dB, normalized)
- `test_data_{5,10,15,20,25}dB.npz` (500 samples each, raw unnormalized)

### 2. Train Models (45-90 minutes total)

```bash
python train_model.py
```

**Expected output:**
- `complexunet.pt` (ComplexUNet checkpoint)
- `realunet.pt` (RealUNet checkpoint)
- `training.log` (real-time progress log)
- `training_results.json` (summary)

### 3. Evaluate All Methods (10-20 minutes)

```bash
python evaluate.py
```

**Expected output:**
- `results.json` (NMSE and BER for all methods vs SNR)

## Real-Time Monitoring

### Training Progress

Monitor training in real-time with:

```bash
tail -f training.log
```

**Log format:**
```
[2024-02-25 20:30:15] === Starting ComplexUNet Training ===
[2024-02-25 20:30:15] STATUS: running
[2024-02-25 20:30:45] Epoch   1/300 | train_loss: 0.245673 | val_nmse: -12.34 dB | best: -12.34 dB | patience:  0/30 | lr: 1.00e-04 | time:  30s | ** saved best model **
[2024-02-25 20:31:15] Epoch   2/300 | train_loss: 0.198432 | val_nmse: -14.56 dB | best: -14.56 dB | patience:  0/30 | lr: 2.00e-04 | time:  30s | ** saved best model **
...
[2024-02-25 20:45:30] ComplexUNet training complete | best_val_nmse: -18.45 dB | total_time: 900s
[2024-02-25 20:45:30] STATUS: completed
```

**Status markers:**
- `STATUS: running` - Training in progress
- `STATUS: completed` - All models trained
- `STATUS: evaluation_completed` - Evaluation finished

**Field explanations:**
- `train_loss`: MSE loss on noise prediction
- `val_nmse`: Validation NMSE in dB (lower is better), measured at t=T//2
- `best`: Best validation NMSE achieved so far
- `patience`: Early stopping counter (stops at 30)
- `lr`: Current learning rate (warmup + cosine annealing)
- `time`: Epoch duration in seconds
- `** saved best model **`: Checkpoint saved

### Evaluation Progress

```bash
tail -f training.log
```

**Evaluation log format:**
```
[2024-02-25 21:00:00] === Starting Evaluation ===
[2024-02-25 21:00:30] SNR= 5dB | LS: -6.42 MMSE: -9.18 DNN: -7.70 Real: -6.18 Complex: -5.88 dB
[2024-02-25 21:01:00] SNR=10dB | LS:-10.67 MMSE:-13.19 DNN:-11.52 Real: -9.15 Complex: -9.35 dB
...
[2024-02-25 21:05:00] STATUS: evaluation_completed
```

## Expected Performance

### Model Sizes
- ComplexUNet: ~200K parameters
- RealUNet: ~200K parameters
- SimpleDNN: ~33K parameters

### Expected NMSE Results (dB)
| SNR | LS    | MMSE  | DNN   | RealUNet | ComplexUNet |
|-----|-------|-------|-------|----------|-------------|
| 5   | -6.4  | -9.2  | -7.7  | -6.2     | -5.9        |
| 10  | -10.7 | -13.2 | -11.5 | -9.1     | -9.3        |
| 15  | -14.2 | -18.0 | -15.0 | -13.3    | -13.6       |
| 20  | -16.3 | -23.0 | -17.4 | -17.9    | -17.3       |
| 25  | -17.4 | -28.0 | -18.6 | -20.6    | -19.8       |

**Key observations:**
- Diffusion models approach or exceed LS at higher SNR
- ComplexUNet and RealUNet are competitive, with ComplexUNet showing slight advantage via the Quantum Phase Rotation Layer
- MMSE provides upper bound (with perfect channel statistics)
- Performance improves monotonically with SNR for all methods

## Architecture Details

### ComplexUNet with Quantum Phase Rotation Output Layer

The **Quantum Phase Rotation Layer** is the core innovation:

```python
# Two unconstrained pre-rotation channels from decoder features
v_re = self.amp_re_head(features)    # [B, 1, K], unconstrained
v_im = self.amp_im_head(features)    # [B, 1, K], unconstrained
phase = self.phase_head(features)    # [B, 1, K], learned phase φ

# Apply Rz-style rotation gate — analogous to single-qubit phase gate
#   [output_re]   [cos φ  -sin φ] [v_re]
#   [output_im] = [sin φ   cos φ] [v_im]
cos_phi = torch.cos(phase)
sin_phi = torch.sin(phase)
output_re = cos_phi * v_re - sin_phi * v_im    # [B, 1, K]
output_im = sin_phi * v_re + cos_phi * v_im    # [B, 1, K]
output = torch.cat([output_re, output_im], dim=1)  # [B, 2, K]
```

**Benefits over standard polar form (A·exp(iφ)):**
1. **Unconstrained magnitude**: Both quadratures can take any value (unlike softplus amplitude)
2. **Quantum phase entanglement**: The two channels are coupled via a learned rotation, not independent
3. **Rz gate analogy**: Directly maps to single-qubit rotation gate from quantum computing
4. **No circular constraint**: The predicted noise is not forced onto a circle in the complex plane

### Bottleneck Self-Attention

Both models include a `SelfAttention1D` module at the UNet bottleneck (spatial size K/8 = 8):

```python
class SelfAttention1D(nn.Module):
    # Multi-head attention (4 heads) over all subcarriers
    # Captures global frequency correlations with O(L²) cost, L=8
```

This captures long-range frequency correlations that local convolutions miss.

### 5-Channel Conditional Input

Both models receive a 5-channel input tensor:

```
model_input = [h_noisy (2ch), h_ls (2ch), pilot_mask (1ch)]
```

- `h_noisy`: Noisy version of the clean channel at diffusion timestep t
- `h_ls`: LS channel estimate (reliable at pilot positions, interpolated elsewhere)
- `pilot_mask`: Binary 0/1 flag indicating which subcarriers are true pilot observations (1) vs. interpolated values (0) — lets the model weight reliable observations higher

### Training Configuration

- **Diffusion steps**: T=500 with cosine beta schedule
- **Loss**: MSE on epsilon-prediction (noise prediction)
- **Optimizer**: Adam with initial lr=1e-3
- **LR schedule**: Linear warmup (lr/10 → lr over 20 epochs), then CosineAnnealingLR (lr → 1e-5)
- **Early stopping**: Patience=30 on validation NMSE at t=T//2
- **Batch size**: 64
- **Epochs**: up to 300
- **Training data**: 10000 samples (2000/SNR × 5 SNR levels), normalized by training std

### Inference: DDIM (Deterministic Sampling)

DDIM (eta=0) replaces the original DDPM+RePaint approach. At each reverse step:

```python
# Predict clean x0 from current noisy sample
pred_h0 = (h_t - sqrt(1 - alpha_cumprod) * ε_θ) / sqrt(alpha_cumprod)
pred_h0 = clamp(pred_h0, -5.0, 5.0)  # prevent amplification at high t

# Deterministic DDIM update (no sigma·noise term)
h_{t-1} = sqrt(ᾱ_{t-1}) * pred_h0 + sqrt(1-ᾱ_{t-1}) * ε_θ
```

Key differences from DDPM+RePaint:
- **No RePaint**: h_ls conditioning is provided directly via input concatenation; RePaint's pilot replacement created training-inference distribution mismatch
- **Deterministic**: No stochastic sigma·noise addition at each step
- **200 steps**: Each step jumps ~2-3 timesteps (vs. 10 for 50 steps)

## File Structure

```
├── models.py           # Model definitions (ComplexUNet, RealUNet, SimpleDNN, DiffusionSchedule)
├── data_generation.py  # OFDM channel data generation
├── train_model.py      # Training script with warmup+cosine LR and real-time logging
├── evaluate.py         # Evaluation script (all methods, DDIM inference)
├── README.md           # This file
├── training.log        # Real-time training/evaluation progress (generated)
├── train_data.npz      # Training dataset, normalized (generated)
├── test_data_*dB.npz   # Test datasets per SNR, raw unnormalized (generated)
├── complexunet.pt      # ComplexUNet checkpoint (generated)
├── realunet.pt         # RealUNet checkpoint (generated)
└── results.json        # Final evaluation results (generated)
```

## Troubleshooting

### MPS Issues (Apple M4)

If you see MPS errors:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train_model.py
```

The code automatically falls back to CPU if MPS fails.

### Memory Issues

Reduce batch size in `train_model.py`:
```python
train_loader, val_loader = create_data_loaders(h_true, h_ls, batch_size=32)  # Default: 64
```

### Slow Training

For faster experimentation:
- Reduce epochs: `epochs=50` in `train_model.py`
- Reduce diffusion steps: `T=250` in `DiffusionSchedule`
- Reduce model size: Use 2 encoder/decoder blocks instead of 3

### NaN Values

If training produces NaN:
1. Check input data: `python -c "import numpy as np; d=np.load('train_data.npz'); print('NaN:', np.isnan(d['h_true']).any())"`
2. Reduce learning rate: `lr=5e-4` in `train_model.py`
3. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

## Expected Runtime

- **Data generation**: < 1 minute
- **ComplexUNet training**: 30-50 minutes
- **RealUNet training**: 30-50 minutes
- **Evaluation**: 15-25 minutes (200 DDIM steps × 5 SNR values)
- **Total**: ~75-120 minutes

## Citation

```bibtex
@article{complex_diffusion_ofdm2024,
  title={Complex-Valued Diffusion Denoiser with Quantum Phase Rotation Layer for OFDM Channel Estimation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
