#!/usr/bin/env python3
"""WAN VAE accuracy with real pretrained weights: BF16 and FP8 vs PyTorch.

Downloads only the VAE subfolder from Wan-AI/Wan2.1-T2V-1.3B-Diffusers
(the smallest WAN model, ~300MB VAE weights vs ~30GB for 14B).
Runs decode on real latents drawn from a standard normal (as the diffusion
process would produce) and measures output accuracy vs PyTorch reference.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from kernels.conv3d_implicit_8wave import conv3d_implicit_8wave  # noqa: E402
from kernels.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8  # noqa: E402


def metrics(y, ref):
    y, ref = y.float(), ref.float()
    err = (y - ref).abs()
    rel = err.mean() / ref.abs().mean().clamp_min(1e-6)
    sig = ref.pow(2).mean()
    noise = err.pow(2).mean()
    snr = 10 * (sig / noise.clamp_min(1e-12)).log10()
    cos = (y.flatten() @ ref.flatten()) / (y.norm() * ref.norm()).clamp_min(1e-12)
    return {"max_abs": err.max().item(), "mean_abs": err.mean().item(),
            "rel%": rel.item() * 100, "cosine": cos.item(), "SNR_dB": snr.item()}


def fmt(m):
    return (f"max={m['max_abs']:.4f}  mean={m['mean_abs']:.5f}"
            f"  rel={m['rel%']:.3f}%  cos={m['cosine']:.6f}  SNR={m['SNR_dB']:.1f}dB")


def main():
    print("Loading WAN VAE (real pretrained weights)...", flush=True)
    print("Downloading from Wan-AI/Wan2.1-T2V-1.3B-Diffusers (VAE only)...", flush=True)

    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    z_ch = vae.config.z_dim
    print(f"VAE z_dim={z_ch}  config: base_dim={vae.config.base_dim} dim_mult={vae.config.dim_mult}")

    orig = WanCausalConv3d.forward

    def make_fly(kernel, cvec):
        def fwd(self, x, cache_x=None):
            pad = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                pad[4] -= cache_x.shape[2]
            x = F.pad(x, pad)
            c = self.in_channels
            if c % cvec:
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
                w = F.pad(self.weight, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
            else:
                w = self.weight
            return kernel(x, w, bias=self.bias, stride=self.stride, padding=0)
        return fwd

    fly_bf16 = make_fly(conv3d_implicit_8wave, 8)
    fly_fp8 = make_fly(conv3d_implicit_8wave_fp8, 16)

    # Test across multiple latent sizes (480p-ish and 720p-ish).
    test_cases = [
        ("480p  latent(6,60,104)", 6, 60, 104),
        ("720p  latent(6,90,160)", 6, 90, 160),
    ]

    for desc, frames, lat_h, lat_w in test_cases:
        print(f"\n{'='*60}")
        print(f"  {desc}  z=(1,{z_ch},{frames},{lat_h},{lat_w})")
        torch.manual_seed(42)
        z = torch.randn(1, z_ch, frames, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)

        # PyTorch reference
        WanCausalConv3d.forward = orig
        with torch.no_grad():
            ref_out = vae.decode(z)
            ref = (ref_out.sample if hasattr(ref_out, "sample") else ref_out).float()
        torch.cuda.synchronize()
        print(f"  decoded video: {tuple(ref.shape)}")
        print(f"  ref stats: mean={ref.mean():.3f} std={ref.std():.3f} "
              f"min={ref.min():.3f} max={ref.max():.3f}")

        # FlyDSL BF16
        WanCausalConv3d.forward = fly_bf16
        with torch.no_grad():
            out_bf16 = vae.decode(z)
            out_bf16 = (out_bf16.sample if hasattr(out_bf16, "sample") else out_bf16).float()
        torch.cuda.synchronize()
        m = metrics(out_bf16, ref)
        print(f"\n  FlyDSL BF16: {fmt(m)}")
        verdict = "✓ PASS" if m["cosine"] > 0.999 else ("⚠ OK" if m["cosine"] > 0.98 else "✗ FAIL")
        print(f"           -> {verdict} (cosine threshold: >0.999 pass, >0.98 ok)")

        # FlyDSL FP8
        WanCausalConv3d.forward = fly_fp8
        with torch.no_grad():
            out_fp8 = vae.decode(z)
            out_fp8 = (out_fp8.sample if hasattr(out_fp8, "sample") else out_fp8).float()
        torch.cuda.synchronize()
        m = metrics(out_fp8, ref)
        print(f"  FlyDSL FP8 : {fmt(m)}")
        verdict = "✓ PASS" if m["cosine"] > 0.99 else ("⚠ MARGINAL" if m["cosine"] > 0.98 else "✗ LIKELY VISIBLE ARTIFACT")
        print(f"           -> {verdict} (cosine threshold: >0.99 pass, >0.98 marginal)")

        WanCausalConv3d.forward = orig

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
