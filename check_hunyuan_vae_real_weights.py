#!/usr/bin/env python3
"""HunyuanVideo VAE accuracy with real pretrained weights: BF16 and FP8 vs PyTorch."""

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
    print("Loading HunyuanVideo VAE (real pretrained weights)...", flush=True)

    from diffusers import AutoencoderKLHunyuanVideo
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoCausalConv3d

    # diffusers/HunyuanVideo-vae is the standalone VAE (~400MB)
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    z_ch = vae.config.latent_channels
    print(f"VAE latent_channels={z_ch}  block_out_channels={vae.config.block_out_channels}")

    orig = HunyuanVideoCausalConv3d.forward

    def make_fly(kernel, cvec):
        def fwd(self, x):
            x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
            c = self.conv.in_channels
            if c % cvec:
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
                w = F.pad(self.conv.weight, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
            else:
                w = self.conv.weight
            return kernel(x, w, bias=self.conv.bias, stride=self.conv.stride, padding=0)
        return fwd

    fly_bf16 = make_fly(conv3d_implicit_8wave, 8)
    fly_fp8 = make_fly(conv3d_implicit_8wave_fp8, 16)

    test_cases = [
        ("720p  latent(5,90,160)", 5, 90, 160),
    ]

    for desc, frames, lat_h, lat_w in test_cases:
        print(f"\n{'='*60}")
        print(f"  {desc}  z=(1,{z_ch},{frames},{lat_h},{lat_w})")
        torch.manual_seed(42)
        z = torch.randn(1, z_ch, frames, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)

        HunyuanVideoCausalConv3d.forward = orig
        with torch.no_grad():
            ref_out = vae.decode(z)
            ref = (ref_out.sample if hasattr(ref_out, "sample") else ref_out).float()
        torch.cuda.synchronize()
        print(f"  decoded video: {tuple(ref.shape)}")
        print(f"  ref stats: mean={ref.mean():.3f} std={ref.std():.3f} "
              f"min={ref.min():.3f} max={ref.max():.3f}")

        for tag, fwd in [("FlyDSL BF16", fly_bf16), ("FlyDSL FP8 ", fly_fp8)]:
            HunyuanVideoCausalConv3d.forward = fwd
            with torch.no_grad():
                out = vae.decode(z)
                out = (out.sample if hasattr(out, "sample") else out).float()
            torch.cuda.synchronize()
            m = metrics(out, ref)
            print(f"\n  {tag}: {fmt(m)}")
            if tag.startswith("FlyDSL BF16"):
                verdict = "✓ PASS" if m["cosine"] > 0.999 else ("⚠ OK" if m["cosine"] > 0.98 else "✗ FAIL")
            else:
                verdict = "✓ PASS" if m["cosine"] > 0.99 else ("⚠ MARGINAL" if m["cosine"] > 0.98 else "✗ LIKELY VISIBLE ARTIFACT")
            print(f"             -> {verdict}")

        HunyuanVideoCausalConv3d.forward = orig

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
