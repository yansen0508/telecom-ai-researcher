"""
Complex-Valued Diffusion Models for OFDM Channel Estimation
Models: DiffusionSchedule, ComplexUNet, RealUNet, SimpleDNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DiffusionSchedule:
    """Cosine beta schedule for T=500 diffusion steps"""
    
    def __init__(self, T=500, s=0.008):
        self.T = T
        self.s = s
        
        # Cosine schedule
        t = torch.linspace(0, T, T + 1)
        f_t = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
        
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip to prevent numerical issues
        self.betas = torch.clamp(betas, 0.0001, 0.9999)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        
    def get_noise_schedule(self, t, device):
        """Get noise schedule parameters for timestep t"""
        if isinstance(t, int):
            t = torch.tensor([t])
        t = t.cpu()
        
        return {
            'alpha_t': self.alphas[t].to(device),
            'alpha_cumprod': self.alphas_cumprod[t].to(device),
            'beta_t': self.betas[t].to(device),
            'posterior_variance': self.posterior_variance[t].to(device)
        }


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Args:
            t: [batch_size] tensor of timesteps
        Returns:
            [batch_size, dim] time embeddings
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention for 1D feature maps [B, C, L].
    Placed at the UNet bottleneck to capture global frequency correlation
    across all subcarriers with O(L²) cost (L=K/8=8 at bottleneck, negligible).
    Uses pre-norm (LayerNorm before attention) for training stability.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        x_seq = x.permute(0, 2, 1)           # [B, L, C]
        x_norm = self.norm(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return (x_seq + attn_out).permute(0, 2, 1)  # residual, back to [B, C, L]


class ComplexUNet(nn.Module):
    """
    Complex-valued U-Net with Quantum Phase Rotation Output Layer.
    Input: [h_noisy (2ch), h_ls (2ch), pilot_mask (1ch)] = 5 channels.

    Conditioning:
      - h_ls: LS channel estimate (interpolated from pilots) at every denoising step.
      - pilot_mask: binary 0/1 flag indicating which subcarriers are true pilot observations
        vs. interpolated values. This lets the model weight reliable pilot positions higher.

    Bottleneck: SelfAttention1D captures global frequency correlation across all subcarriers.

    Output layer: Quantum Phase Rotation Gate — a learned Rz-style rotation entangles
    the two quadrature channels while keeping each individually unconstrained.
    """

    def __init__(self, in_channels=5, out_channels=2, K=64):
        super().__init__()
        self.K = K
        
        # Time embedding
        time_dim = 128
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, 32, time_dim)
        self.enc2 = self._make_layer(32, 64, time_dim)
        self.enc3 = self._make_layer(64, 128, time_dim)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        # Self-attention at bottleneck (spatial size = K/8 = 8 for K=64):
        # captures global frequency correlation across all subcarriers cheaply.
        self.bottleneck_attn = SelfAttention1D(128, num_heads=4)

        # Decoder with skip connections
        self.dec3 = self._make_layer(128 + 128, 64, time_dim)  # +skip from enc3
        self.dec2 = self._make_layer(64 + 64, 32, time_dim)    # +skip from enc2
        self.dec1 = self._make_layer(32 + 32, 32, time_dim)    # +skip from enc1

        # Quantum Phase Rotation Output Layer - KEY NOVELTY
        # Two independent feature channels are "entangled" via a learned phase rotation,
        # analogous to a single-qubit Rz gate: U(φ)|ψ⟩ = [cos φ, -sin φ; sin φ, cos φ] |ψ⟩
        self.amp_re_head = nn.Conv1d(32, 1, 1)  # Pre-rotation Re channel (unconstrained)
        self.amp_im_head = nn.Conv1d(32, 1, 1)  # Pre-rotation Im channel (unconstrained)
        self.phase_head = nn.Conv1d(32, 1, 1)   # Learned phase coupling φ
        
    def _make_layer(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'conv': nn.Conv1d(in_ch, out_ch, 3, padding=1),
            'norm': nn.GroupNorm(min(8, out_ch), out_ch),
            'time_proj': nn.Linear(time_dim, out_ch),
            'activation': nn.SiLU(),
        })
    
    def _apply_layer(self, x, t_emb, layer):
        """Apply layer with time embedding injection"""
        x = layer['conv'](x)
        x = layer['norm'](x)
        
        # Add time embedding
        t_proj = layer['time_proj'](t_emb)[:, :, None]  # [B, C, 1]
        x = x + t_proj
        
        x = layer['activation'](x)
        return x
        
    def forward(self, x, t):
        """
        Args:
            x: [B, 5, K] = cat([h_noisy, h_ls, pilot_mask], dim=1)
            t: [B] timesteps
        Returns:
            [B, 2, K] predicted noise (Re/Im) with Quantum Phase Rotation output layer
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_dim]

        # Encoder with skip connections
        enc1_out = self._apply_layer(x, t_emb, self.enc1)          # [B, 32, K]
        enc1_down = F.avg_pool1d(enc1_out, 2)                      # [B, 32, K/2]

        enc2_out = self._apply_layer(enc1_down, t_emb, self.enc2)  # [B, 64, K/2]
        enc2_down = F.avg_pool1d(enc2_out, 2)                      # [B, 64, K/4]

        enc3_out = self._apply_layer(enc2_down, t_emb, self.enc3)  # [B, 128, K/4]
        enc3_down = F.avg_pool1d(enc3_out, 2)                      # [B, 128, K/8]

        # Bottleneck + self-attention (captures global subcarrier correlation)
        bottleneck_out = self.bottleneck(enc3_down)                # [B, 128, K/8]
        bottleneck_out = self.bottleneck_attn(bottleneck_out)      # [B, 128, K/8]
        
        # Decoder with skip connections
        dec3_up = F.interpolate(bottleneck_out, size=enc3_out.shape[-1], mode='linear')
        dec3_in = torch.cat([dec3_up, enc3_out], dim=1)           # [B, 256, K/4]
        dec3_out = self._apply_layer(dec3_in, t_emb, self.dec3)   # [B, 64, K/4]
        
        dec2_up = F.interpolate(dec3_out, size=enc2_out.shape[-1], mode='linear')
        dec2_in = torch.cat([dec2_up, enc2_out], dim=1)           # [B, 128, K/2]
        dec2_out = self._apply_layer(dec2_in, t_emb, self.dec2)   # [B, 32, K/2]
        
        dec1_up = F.interpolate(dec2_out, size=enc1_out.shape[-1], mode='linear')
        dec1_in = torch.cat([dec1_up, enc1_out], dim=1)           # [B, 64, K]
        features = self._apply_layer(dec1_in, t_emb, self.dec1)   # [B, 32, K]
        
        # Quantum Phase Rotation Output Layer - CORE INNOVATION
        # Independent pre-rotation channels (unconstrained, unlike softplus amplitude)
        v_re = self.amp_re_head(features)                          # [B, 1, K]
        v_im = self.amp_im_head(features)                          # [B, 1, K]
        phase = self.phase_head(features)                          # [B, 1, K], learned phase

        # Apply Rz-style rotation: entangles v_re and v_im via learned phase φ
        #   [output_re]   [cos φ  -sin φ] [v_re]
        #   [output_im] = [sin φ   cos φ] [v_im]
        # This models quantum phase interaction between the two quadratures.
        # Unlike the polar form A*exp(iφ), both channels are fully unconstrained,
        # so the predicted noise magnitude is not restricted to a fixed circle.
        cos_phi = torch.cos(phase)
        sin_phi = torch.sin(phase)
        output_re = cos_phi * v_re - sin_phi * v_im                # [B, 1, K]
        output_im = sin_phi * v_re + cos_phi * v_im                # [B, 1, K]

        output = torch.cat([output_re, output_im], dim=1)          # [B, 2, K]

        return output


class RealUNet(nn.Module):
    """
    Real-valued U-Net (ablation baseline).
    Same architecture as ComplexUNet — 5-channel input, bottleneck attention —
    but uses a standard Conv1d output instead of the Quantum Phase Rotation Layer.
    This isolates the contribution of the phase-rotation innovation.
    """

    def __init__(self, in_channels=5, out_channels=2, K=64):
        super().__init__()
        self.K = K
        
        # Time embedding (identical to ComplexUNet)
        time_dim = 128
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder (identical to ComplexUNet)
        self.enc1 = self._make_layer(in_channels, 32, time_dim)
        self.enc2 = self._make_layer(32, 64, time_dim)
        self.enc3 = self._make_layer(64, 128, time_dim)
        
        # Bottleneck + self-attention (same as ComplexUNet)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.bottleneck_attn = SelfAttention1D(128, num_heads=4)

        # Decoder (identical to ComplexUNet)
        self.dec3 = self._make_layer(128 + 128, 64, time_dim)
        self.dec2 = self._make_layer(64 + 64, 32, time_dim)
        self.dec1 = self._make_layer(32 + 32, 32, time_dim)

        # Standard output layer (NO Phase Interaction Layer — ablation baseline)
        self.final_conv = nn.Conv1d(32, out_channels, 1)
        
    def _make_layer(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'conv': nn.Conv1d(in_ch, out_ch, 3, padding=1),
            'norm': nn.GroupNorm(min(8, out_ch), out_ch),
            'time_proj': nn.Linear(time_dim, out_ch),
            'activation': nn.SiLU(),
        })
    
    def _apply_layer(self, x, t_emb, layer):
        """Apply layer with time embedding injection"""
        x = layer['conv'](x)
        x = layer['norm'](x)
        
        # Add time embedding
        t_proj = layer['time_proj'](t_emb)[:, :, None]
        x = x + t_proj
        
        x = layer['activation'](x)
        return x
        
    def forward(self, x, t):
        """
        Args:
            x: [B, 5, K] = cat([h_noisy, h_ls, pilot_mask], dim=1)
            t: [B] timesteps
        Returns:
            [B, 2, K] predicted noise (standard Conv1d output — no Phase Rotation)
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Encoder (identical to ComplexUNet)
        enc1_out = self._apply_layer(x, t_emb, self.enc1)
        enc1_down = F.avg_pool1d(enc1_out, 2)

        enc2_out = self._apply_layer(enc1_down, t_emb, self.enc2)
        enc2_down = F.avg_pool1d(enc2_out, 2)

        enc3_out = self._apply_layer(enc2_down, t_emb, self.enc3)
        enc3_down = F.avg_pool1d(enc3_out, 2)

        # Bottleneck + self-attention
        bottleneck_out = self.bottleneck(enc3_down)
        bottleneck_out = self.bottleneck_attn(bottleneck_out)
        
        # Decoder (identical to ComplexUNet)
        dec3_up = F.interpolate(bottleneck_out, size=enc3_out.shape[-1], mode='linear')
        dec3_in = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_out = self._apply_layer(dec3_in, t_emb, self.dec3)
        
        dec2_up = F.interpolate(dec3_out, size=enc2_out.shape[-1], mode='linear')
        dec2_in = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_out = self._apply_layer(dec2_in, t_emb, self.dec2)
        
        dec1_up = F.interpolate(dec2_out, size=enc1_out.shape[-1], mode='linear')
        dec1_in = torch.cat([dec1_up, enc1_out], dim=1)
        features = self._apply_layer(dec1_in, t_emb, self.dec1)
        
        # Standard output (NO Phase Interaction Layer)
        output = self.final_conv(features)  # [B, 2, K] directly
        
        return output


class SimpleDNN(nn.Module):
    """
    3-layer MLP baseline for channel estimation
    Direct mapping from LS estimate to refined estimate
    """
    
    def __init__(self, K=64):
        super().__init__()
        self.K = K
        
        # 3-layer MLP
        self.layers = nn.Sequential(
            nn.Linear(2 * K, 256),  # Input: [Re, Im] flattened
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * K),  # Output: [Re, Im] flattened
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 2, K] LS channel estimate
        Returns:
            [B, 2, K] refined channel estimate
        """
        B, _, K = x.shape
        x_flat = x.view(B, -1)  # [B, 2*K]
        out_flat = self.layers(x_flat)  # [B, 2*K]
        output = out_flat.view(B, 2, K)  # [B, 2, K]
        
        return output


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model parameter counts
    complex_unet = ComplexUNet(K=64)
    real_unet = RealUNet(K=64)
    simple_dnn = SimpleDNN(K=64)
    
    print(f"ComplexUNet parameters: {count_parameters(complex_unet):,}")
    print(f"RealUNet parameters: {count_parameters(real_unet):,}")
    print(f"SimpleDNN parameters: {count_parameters(simple_dnn):,}")
    
    # Test forward pass
    batch_size = 4
    K = 64
    T = 500
    
    x = torch.randn(batch_size, 2, K)
    t = torch.randint(1, T, (batch_size,))
    
    with torch.no_grad():
        complex_out = complex_unet(x, t)
        real_out = real_unet(x, t)
        dnn_out = simple_dnn(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"ComplexUNet output shape: {complex_out.shape}")
    print(f"RealUNet output shape: {real_out.shape}")
    print(f"SimpleDNN output shape: {dnn_out.shape}")
    
    # Test diffusion schedule
    schedule = DiffusionSchedule(T=500)
    print(f"\nDiffusion schedule T={schedule.T}")
    print(f"Beta range: {schedule.betas.min():.6f} - {schedule.betas.max():.6f}")
    print(f"Alpha_cumprod range: {schedule.alphas_cumprod.min():.6f} - {schedule.alphas_cumprod.max():.6f}")