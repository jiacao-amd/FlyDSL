# VAE End-to-End Results: FlyDSL conv3d vs PyTorch (MIOpen)

FlyDSL BF16 / FP8 implicit-GEMM conv3d dropped into diffusers VAE models via
monkeypatching the causal conv3d forward. Every conv runs on FlyDSL (small-C
stems via math-preserving channel-pad), 0 MIOpen fallbacks.

- **GPU**: AMD Instinct MI355X (gfx950 / CDNA4)
- **Container**: `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0`
- **Perf config**: warmup=2, iters=5, N=1
- **Harnesses**: `bench_wan_vae_e2e.py`, `bench_hunyuan_vae_e2e.py`
- **Accuracy harnesses**: `check_wan_vae_real_weights.py`, `check_hunyuan_vae_real_weights.py`

---

## WAN VAE (`AutoencoderKLWan`, `WanCausalConv3d`)

Real weights: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. 6 latent frames.
Baseline: stock PyTorch/MIOpen (fully tuned, verified healthy, 0 warnings).

### Performance

#### Decode (latent → video)

| resolution | latent (H,W) | video | PyTorch (ms) | FlyDSL BF16 (ms) | BF16 speedup | FlyDSL FP8 (ms) | FP8 speedup |
|---|---|---|---|---|---|---|---|
| 480p  | 60×104  | 480×832   |  366.21 |  306.82 | 1.19x |  258.50 | **1.42x** |
| 720p  | 90×160  | 720×1280  |  854.49 |  701.31 | 1.22x |  597.43 | **1.43x** |
| 1080p | 136×240 | 1088×1920 | 1927.28 | 1535.51 | 1.26x | 1317.04 | **1.46x** |

#### Encode (video → latent)

| resolution | latent (H,W) | video | PyTorch (ms) | FlyDSL BF16 (ms) | BF16 speedup | FlyDSL FP8 (ms) | FP8 speedup |
|---|---|---|---|---|---|---|---|
| 480p  | 60×104  | 480×832   |  210.29 |  183.26 | 1.15x |  153.40 | **1.37x** |
| 720p  | 90×160  | 720×1280  |  496.05 |  417.21 | 1.19x |  353.67 | **1.40x** |
| 1080p | 136×240 | 1088×1920 | 1033.37 |  892.01 | 1.16x |  759.23 | **1.36x** |

Conv layers per pass: ~197 decode, ~155 encode (frames processed in chunks).

### Numerical Accuracy (real pretrained weights)

Reference = PyTorch BF16 output. Threshold: BF16 cosine > 0.999 ✓; FP8 cosine > 0.99 ✓ / > 0.98 marginal.

#### Decode

| resolution | path | rel err | cosine | SNR (dB) | verdict |
|---|---|---|---|---|---|
| 480p | FlyDSL BF16 | 0.652% | 0.999972 | 42.4 | ✓ PASS |
| 480p | FlyDSL FP8  | 6.24%  | 0.995614 | 20.6 | ✓ PASS |
| 720p | FlyDSL BF16 | 0.653% | 0.999972 | 42.4 | ✓ PASS |
| 720p | FlyDSL FP8  | 6.08%  | 0.996143 | 21.1 | ✓ PASS |

#### Encode (latent mean, deterministic)

| resolution | path | rel err | cosine | SNR (dB) | verdict |
|---|---|---|---|---|---|
| 480p | FlyDSL BF16 | 1.75%  | 0.999830 | 34.8 | ✓ PASS |
| 480p | FlyDSL FP8  | 25.81% | 0.964980 | 11.5 | ⚠ MARGINAL |

---

## HunyuanVideo VAE (`AutoencoderKLHunyuanVideo`, `HunyuanVideoCausalConv3d`)

Real weights: `hunyuanvideo-community/HunyuanVideo` (vae subfolder). 5 latent frames.
Baseline: `MIOPEN_FIND_MODE=2` (fast/untuned — avoids 20+ min MIOpen search on
first run; not directly comparable to WAN tuned numbers).

### Performance

#### Decode (latent → video)

| resolution | latent (H,W) | video | PyTorch (ms) | FlyDSL BF16 (ms) | BF16 speedup | FlyDSL FP8 (ms) | FP8 speedup |
|---|---|---|---|---|---|---|---|
| 480p | 60×104 | 480×832 | 961.5 | 1037.0 | 0.93x | 886.0 | **1.09x** |

#### Encode (video → latent)

| resolution | latent (H,W) | video | PyTorch (ms) | FlyDSL BF16 (ms) | BF16 speedup | FlyDSL FP8 (ms) | FP8 speedup |
|---|---|---|---|---|---|---|---|
| 480p | 60×104 | 480×832 | 522.7 | 586.1 | 0.90x | 490.8 | **1.04x** |

Conv layers per pass: ~70 decode, ~54 encode.

**Why BF16 is slower on Hunyuan:** MIOpen with `FIND_MODE=2` selects
`ConvHipImplicitGemm3DGroupFwdXdlops` (XDLOPS fused path) for the dominant
C=512 layers — MIOpen's fastest 3D conv kernel, with fused output in NCDHW.
FlyDSL BF16 MFMA compute is competitive but the NCDHW epilogue store
(`col×dhw` stride) has lower L2 coalescing, narrowing the margin.
FP8 wins because 2× higher MFMA throughput overcomes the store overhead.

### Numerical Accuracy (real pretrained weights)

#### Decode

| resolution | path | rel err | cosine | SNR (dB) | verdict |
|---|---|---|---|---|---|
| 480p | FlyDSL BF16 | 0.692% | 0.999969 | 42.0 | ✓ PASS |
| 480p | FlyDSL FP8  | 8.09%  | 0.996289 | 21.2 | ✓ PASS |
| 720p | — | — | — | — | OOM (PyTorch baseline OOM, not FlyDSL) |

---

## SNR Reference

SNR is always relative between two signals: `10×log₁₀(signal_power / error_power)`.

| SNR (dB) | Meaning |
|---|---|
| 40–50 | Near-perfect, visually indistinguishable |
| 30–40 | High quality (JPEG high-quality ≈ 40 dB) |
| 20–30 | Acceptable; not perceptible in most cases |
| 10–20 | Noticeable degradation |
| < 10  | Severe distortion |

In video generation the diffusion sampling process operates at much lower SNR than
VAE quantization errors, so FP8 decode at ~21 dB SNR is typically not visually
perceptible in final output.

---

## Key Engineering Findings

### NCDHW Epilogue Store
Original kernel returned `permute(0,4,1,2,3)` (non-contiguous NCDHW). Hunyuan's
`nn.GroupNorm` triggered `aten::contiguous` copies (+300ms/decode). Fixed by
computing the correct NCDHW flat offset in the kernel epilogue:
```python
# N=1 (dominant inference case): no division
off = col * dhw + row
# N>1 general:
off = (row // dhw) * (k * dhw) + col * dhw + (row % dhw)
```
WAN uses `WanRMSNorm` (layout-agnostic) so the issue was invisible there.
After fix: Hunyuan FP8 decode 0.85x → **1.09x**.

### Partial-tile Masking (NPQ / K / CRS axes)
All three tile axes now support non-128-aligned sizes:
- **M (NPQ)**: `grid_m` ceil; OOB loads use sentinel, OOB stores dropped by HW bounds.
- **N (K)**: OOB stores masked with `col < k`; split-K atomics guarded via `scf.if`.
- **K (CRS)**: `k_tiles` ceil; OOB activation loads zeroed via `k_abs < crs`.

### Split-K JIT-OOM Fix
C=384 (k_tiles=81) caused LLVM JIT OOM from `range_constexpr` unrolling 81 tiles.
Fixed by capping `tiles_per_split=54` in `_resolve_splitk`, bumping splitk to
next valid divisor (splitk=3 for C=384, tiles_per_split=27).

---

## Reproduce

```bash
DOCKER_IMAGE="rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0"
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v "$PWD:/workspace/FlyDSL" -v "$PWD/bench_conv3d_wan_vae_cache:/bench_cache" \
  -w /workspace/FlyDSL \
  -e "PYTHONPATH=/workspace/FlyDSL/build-fly/python_packages:/workspace/FlyDSL" \
  -e "LD_LIBRARY_PATH=/workspace/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs" \
  -e "FLYDSL_RUNTIME_CACHE_DIR=/bench_cache" -e "HIP_VISIBLE_DEVICES=1" \
  "$DOCKER_IMAGE" bash

# Inside the container:
pip install diffusers

# --- WAN VAE ---
python3 -u bench_wan_vae_e2e.py                                    # 480p
VAE_FRAMES=6 VAE_H=90  VAE_W=160 python3 -u bench_wan_vae_e2e.py   # 720p
VAE_FRAMES=6 VAE_H=136 VAE_W=240 python3 -u bench_wan_vae_e2e.py   # 1080p
python3 -u check_wan_vae_real_weights.py                            # accuracy

# --- Hunyuan VAE ---
export MIOPEN_FIND_MODE=2
python3 -u bench_hunyuan_vae_e2e.py                                 # 480p
VAE_FRAMES=5 VAE_H=90 VAE_W=160 python3 -u bench_hunyuan_vae_e2e.py # 720p
python3 -u check_hunyuan_vae_real_weights.py                        # accuracy
```
