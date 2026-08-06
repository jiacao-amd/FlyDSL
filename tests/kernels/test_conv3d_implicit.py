#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 implicit-GEMM conv3d kernel.

Compares ``conv3d_implicit`` against ``torch.nn.functional.conv3d`` on
NCDHW/OIDHW bf16 inputs across stride/padding/dilation and M%TILE_M / K%TILE_N
tail paths. Any channel count, spatial extent, and group count is supported.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_implicit import conv3d_implicit

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
# mfma_f32_16x16x32_bf16 is only available on CDNA4 (gfx95x)
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.startswith("gfx95")),
    reason=f"conv3d BF16 needs mfma_f32_16x16x32_bf16 (CDNA4 gfx95x), got {_ARCH}",
)


# (N, C, T, H, W, K), kernel 3x3x3. Covers stride/padding and tile-tail paths.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,t,h,w,k,stride,padding",
    [
        (1, 32, 8, 16, 16, 64, 1, 0),
        (1, 32, 9, 17, 17, 96, 1, 1),
        (2, 64, 6, 18, 18, 192, 1, 1),
        (1, 32, 10, 20, 20, 64, 2, 1),
        # Partial K-tile: C=16 -> CRS=432, 432 % TILE_K(32) = 16 (masked).
        (1, 16, 6, 16, 20, 16, 1, 1),
        (1, 16, 4, 12, 16, 384, 1, 1),
        (1, 3, 4, 12, 12, 32, 1, 1),
        (1, 12, 4, 12, 12, 32, 1, 1),
        (2, 5, 4, 10, 14, 48, 1, 1),
        (1, 6, 3, 11, 11, 32, 1, 1),
    ],
)
def test_conv3d_vs_torch(n, c, t, h, w, k, stride, padding):
    torch.manual_seed(2000 + h + w + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,padding",
    [
        ((1, 3, 3), (0, 1, 1)),
        ((3, 1, 1), (1, 0, 0)),
    ],
)
def test_conv3d_factorized_filters_vs_torch(kernel_shape, padding):
    """Cover the spatial-only and temporal-only filter dispatch paths."""
    torch.manual_seed(3100 + sum(kernel_shape))
    n, c, t, h, w, k = 1, 64, 6, 18, 20, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=1, padding=padding)
    y_ref = F.conv3d(x, weight, stride=1, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Dilation stretches the im2col gather without changing the GEMM K axis. These cover the
# general 3D address path: isotropic and anisotropic dilation, dilation combined with
# stride/padding/bias, taps that fall out of range on most rows, and the K-tile / N-tile
# tails. Filters are chosen so the padded input still admits a >= 1 output extent.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,t,h,w,k,kernel_shape,stride,padding,dilation",
    [
        (1, 32, 8, 16, 16, 64, (3, 3, 3), 1, 2, 2),  # output extent preserved
        (1, 32, 8, 16, 16, 64, (3, 3, 3), 1, 0, 2),  # unpadded -> output shrinks
        (2, 64, 6, 18, 18, 128, (3, 3, 3), 1, (1, 2, 3), (1, 2, 3)),  # anisotropic
        (1, 32, 10, 20, 20, 64, (3, 3, 3), 2, 2, 2),  # strided and dilated
        (1, 16, 6, 16, 20, 16, (3, 3, 3), 1, 2, 2),  # CRS % TILE_K != 0, K < TILE_N
        (1, 3, 5, 12, 12, 32, (3, 3, 3), 1, 2, 2),  # C padded to the gather width
        (1, 64, 5, 9, 9, 64, (3, 3, 3), 1, 4, 4),  # wide taps: most reads masked off
        (1, 32, 6, 14, 14, 96, (2, 2, 2), 1, 1, 2),  # even filter
        (1, 64, 4, 10, 10, 64, (1, 1, 1), 1, 0, 3),  # 1x1x1: dilation is a no-op
    ],
)
def test_conv3d_dilation_vs_torch(n, c, t, h, w, k, kernel_shape, stride, padding, dilation):
    torch.manual_seed(3300 + c + k + sum(kernel_shape))
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding, dilation=dilation)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Split-K reduces through an fp32 (npq, k) staging buffer and only reshapes to
# (N, K, Do, Ho, Wo) at the end, so the dilated output extents have to reach that path.
@_skip_non_cdna4
@pytest.mark.parametrize("dilation", [1, 2])
def test_conv3d_dilation_splitk_vs_torch(dilation):
    torch.manual_seed(3500 + dilation)
    n, c, t, h, w, k = 1, 64, 8, 16, 16, 64
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, padding=dilation, dilation=dilation, splitk=2)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), padding=dilation, dilation=dilation)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Dilation and groups are independent axes of the address math -- dilation stretches the
# tap offsets, groups shift the channel base -- so they have to be exercised together,
# including the per-group channel pad (C/groups % 8 != 0) and per-group N tail.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,t,h,w,k,groups,padding,dilation",
    [
        (1, 64, 6, 16, 16, 128, 2, 2, 2),  # CG=32, KG=64, both aligned
        (2, 128, 6, 14, 14, 256, 4, 2, 2),  # CG=32, KG=64
        (1, 256, 6, 12, 12, 128, 8, 2, 2),  # KG=16 -> N tile under-fills
        (1, 12, 6, 12, 12, 32, 4, 2, 2),  # CG=3 -> per-group pad to 8
        (1, 24, 6, 10, 10, 40, 8, 2, 2),  # CG=3, KG=5 -> pad + tail together
        (1, 64, 6, 16, 16, 128, 2, (1, 2, 3), (1, 2, 3)),  # anisotropic
        (1, 16, 6, 10, 10, 16, 16, 2, 2),  # depthwise: CG=1, KG=1
    ],
)
def test_conv3d_grouped_dilation_vs_torch(n, c, t, h, w, k, groups, padding, dilation):
    torch.manual_seed(3600 + c + k + groups)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, padding=padding, dilation=dilation, groups=groups)
    y_ref = F.conv3d(x, weight, padding=padding, dilation=dilation, groups=groups)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Dilation must survive the forced-tile path as well as the default heuristic.
@_skip_non_cdna4
@pytest.mark.parametrize("tile", [(128, 128, 2, 4), (256, 128, 2, 4), (64, 64, 2, 2)])
def test_conv3d_dilation_tile_configs(tile):
    torch.manual_seed(3700 + sum(tile))
    n, c, t, h, w, k = 2, 64, 6, 18, 18, 192
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, padding=2, dilation=2, tile=tile)
    y_ref = F.conv3d(x, weight, padding=2, dilation=2)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Rejected up front rather than silently computing a wrong / empty result.
@_skip_non_cdna4
def test_conv3d_invalid_dilation():
    x = torch.randn((1, 16, 4, 8, 8), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((32, 16, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="dilation must be >= 1"):
        conv3d_implicit(x, weight, padding=1, dilation=0)
    # 5*(3-1)+1 = 11 taps span more than the unpadded extent of 8.
    with pytest.raises(AssertionError, match="dilated filter is larger than the padded input"):
        conv3d_implicit(x, weight, padding=0, dilation=5)


# Dilated factorized filters. The (T,1,1) cases keep Do==D so they still take the
# temporal-only fast path, where dilation folds into the per-tap temporal delta.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,padding,dilation",
    [
        ((3, 1, 1), (2, 0, 0), (2, 1, 1)),  # temporal-only fast path, Do == D
        ((5, 1, 1), (4, 0, 0), (2, 1, 1)),  # temporal-only fast path, wider filter
        ((3, 1, 1), (0, 0, 0), (2, 1, 1)),  # temporal-only, Do < D -> general path
        ((1, 3, 3), (0, 2, 2), (1, 2, 2)),  # spatial-only
    ],
)
def test_conv3d_dilated_factorized_filters_vs_torch(kernel_shape, padding, dilation):
    torch.manual_seed(3400 + sum(kernel_shape) + sum(dilation))
    n, c, t, h, w, k = 1, 64, 8, 18, 20, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=1, padding=padding, dilation=dilation)
    y_ref = F.conv3d(x, weight, stride=1, padding=padding, dilation=dilation)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize("c", [16, 64])
def test_conv3d_runtime_k_loop_short_problems(c):
    """Exercise one- and two-K-tile runtime-pipeline epilogues."""
    torch.manual_seed(3200 + c)
    n, t, h, w, k = 1, 3, 8, 8, 64
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 1, 1, 1), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight)
    y_ref = F.conv3d(x, weight)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Tile-size sweep: each forced (TILE_M, TILE_N, WAVE_M, WAVE_N) must stay correct.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "tile",
    [
        (128, 128, 2, 4),  # default
        (128, 256, 2, 4),
        (256, 128, 2, 4),
        (256, 256, 2, 4),
        (256, 256, 4, 4),
        (128, 128, 4, 2),
        (64, 128, 1, 4),
        (64, 64, 2, 2),
    ],
)
def test_conv3d_tile_configs(tile):
    torch.manual_seed(4000 + sum(tile))
    n, c, t, h, w, k, stride, padding = 2, 64, 6, 18, 18, 192, 1, 1
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding, tile=tile)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
def test_conv3d_autotune(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "at"))
    from kernels.conv import conv3d_autotune

    conv3d_autotune._MEM_CACHE.clear()

    torch.manual_seed(4242)
    n, c, t, h, w, k = 1, 128, 6, 40, 40, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=1, padding=1, autotune=True)
    y_ref = F.conv3d(x, weight, stride=1, padding=1)
    torch.cuda.synchronize()
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)

    # A tile was chosen and persisted; the second call must hit the cache.
    assert len(conv3d_autotune._MEM_CACHE) == 1
    calls = {"n": 0}
    orig = conv3d_autotune.do_bench

    def _counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(conv3d_autotune, "do_bench", _counting)
    y2 = conv3d_implicit(x, weight, stride=1, padding=1, autotune=True)
    torch.cuda.synchronize()
    assert torch.allclose(y2, y_ref, rtol=2e-2, atol=2e-2)
    assert calls["n"] == 0  # cached, no re-benchmark


# 2D conv via the depth-1 degenerate path through the 3D kernel.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,stride,padding",
    [
        ((3, 3), 1, 1),
        ((1, 1), 1, 0),  # 1x1 -> temporal_only_fast-style vectorized epilogue
        ((5, 5), 1, 2),
        ((3, 3), 2, 1),
    ],
)
def test_conv2d_vs_torch(kernel_shape, stride, padding):
    torch.manual_seed(5000 + sum(kernel_shape) + stride + padding)
    n, c, h, w, k = 2, 64, 24, 28, 128
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv2d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# 2D dilation. _conv2d_impl widens (dh, dw) to (1, dh, dw) for the degenerate depth axis,
# so a per-axis dilation tuple must land on the right two axes.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,stride,padding,dilation",
    [
        ((3, 3), 1, 2, 2),
        ((3, 3), 1, 0, 2),  # unpadded -> output shrinks
        ((3, 3), 2, 2, 2),  # strided and dilated
        ((5, 5), 1, 4, 2),
        ((3, 3), 1, (1, 2), (1, 2)),  # anisotropic
        ((1, 1), 1, 0, 3),  # 1x1: dilation is a no-op
    ],
)
def test_conv2d_dilation_vs_torch(kernel_shape, stride, padding, dilation):
    torch.manual_seed(5100 + sum(kernel_shape) + stride)
    n, c, h, w, k = 2, 64, 24, 28, 128
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    y_ref = F.conv2d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding, dilation=dilation)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Unaligned channel counts and spatial extents.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "c,h,w,k,kernel_shape,stride,padding",
    [
        (3, 32, 32, 64, (3, 3), 1, 1),
        (3, 24, 28, 32, (7, 7), 2, 3),
        (1, 24, 24, 32, (3, 3), 1, 1),
        (12, 16, 16, 32, (3, 3), 1, 1),
        (64, 33, 33, 64, (3, 3), 2, 0),
        (128, 17, 17, 64, (3, 3), 2, 0),
        (6, 21, 21, 32, (3, 3), 1, 1),
    ],
)
def test_conv2d_unaligned_channels_and_spatial(c, h, w, k, kernel_shape, stride, padding):
    torch.manual_seed(7000 + c + h + k)
    x = torch.randn((1, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=stride, padding=padding)
    y_ref = F.conv2d(x, weight, stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# The transpose needs C aligned to the vector width; S may be anything.
@_skip_non_cdna4
@pytest.mark.parametrize("c", [8, 16, 64, 512])
@pytest.mark.parametrize("h,w", [(3, 3), (5, 5), (17, 15), (33, 33), (8, 8)])
def test_transpose_unaligned_spatial(c, h, w):
    from kernels.conv.conv3d_implicit import _ncdhw_to_ndhwc

    torch.manual_seed(8000 + c + h * w)
    x = torch.randn((1, c, 1, h, w), device="cuda", dtype=torch.bfloat16)

    got = _ncdhw_to_ndhwc(x, torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(got, x.permute(0, 2, 3, 4, 1).contiguous())


# 1D conv via the depth/height-1 degenerate path through the 3D kernel.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "s,stride,padding",
    [
        (3, 1, 1),
        (1, 1, 0),
        (5, 2, 2),
    ],
)
def test_conv1d_vs_torch(s, stride, padding):
    torch.manual_seed(6000 + s + stride + padding)
    n, c, w, k = 2, 64, 96, 128
    x = torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, s), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv1d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# 1D dilation. _conv1d_impl takes a scalar or a 1-tuple and widens it to (1, 1, dw).
@_skip_non_cdna4
@pytest.mark.parametrize(
    "s,stride,padding,dilation",
    [
        (3, 1, 2, 2),
        (3, 1, 0, 2),  # unpadded -> output shrinks
        (5, 2, 4, 2),  # strided and dilated
        (3, 1, (4,), (4,)),  # tuple form
    ],
)
def test_conv1d_dilation_vs_torch(s, stride, padding, dilation):
    torch.manual_seed(6100 + s + stride)
    n, c, w, k = 2, 64, 96, 128
    x = torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, s), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    y_ref = F.conv1d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding, dilation=dilation)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Grouped conv. Groups fold onto the N grid axis, one tile never spanning two groups, so
# these cover the per-group channel pad (C/groups % 8 != 0) and the per-group N tail
# (K/groups % TILE_N != 0) as well as the plain aligned case.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,t,h,w,k,groups,stride,padding",
    [
        (1, 64, 6, 16, 16, 128, 2, 1, 1),  # CG=32, KG=64, both aligned
        (2, 128, 4, 14, 14, 256, 4, 1, 1),  # CG=32, KG=64
        (1, 256, 4, 12, 12, 128, 8, 1, 0),  # CG=32, KG=16 -> N tile under-fills
        (1, 64, 8, 20, 20, 64, 2, 2, 1),  # strided
        (1, 32, 4, 12, 12, 96, 4, 1, 1),  # CG=8 exactly at the vector width
        (1, 12, 4, 12, 12, 32, 4, 1, 1),  # CG=3 -> per-group pad to 8
        (1, 24, 4, 10, 10, 40, 8, 1, 1),  # CG=3, KG=5 -> pad + tail together
        (1, 96, 4, 10, 10, 48, 3, 1, 1),  # CG=32, KG=16, groups not a power of two
        (1, 40, 3, 9, 9, 20, 5, 1, 1),  # CG=8, KG=4
        (1, 16, 4, 8, 8, 16, 16, 1, 1),  # depthwise: CG=1, KG=1
        (1, 32, 3, 8, 8, 64, 32, 1, 1),  # depthwise with multiplier 2
    ],
)
def test_conv3d_grouped_vs_torch(n, c, t, h, w, k, groups, stride, padding):
    torch.manual_seed(9000 + c + k + groups)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=stride, padding=padding, groups=groups)
    y_ref = F.conv3d(x, weight, stride=stride, padding=padding, groups=groups)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Bias is indexed by the global out-channel, so it must survive the group remap.
@_skip_non_cdna4
@pytest.mark.parametrize("groups", [1, 2, 4, 8])
def test_conv3d_grouped_bias_vs_torch(groups):
    torch.manual_seed(9100 + groups)
    n, c, t, h, w, k = 1, 64, 5, 14, 14, 96
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, padding=1, groups=groups)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), padding=1, groups=groups)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Grouped 1x1x1 skips the ungrouped torch.matmul fast path and runs the kernel.
@_skip_non_cdna4
@pytest.mark.parametrize("groups", [2, 4])
def test_conv3d_grouped_1x1x1_vs_torch(groups):
    torch.manual_seed(9200 + groups)
    n, c, t, h, w, k = 1, 128, 4, 12, 12, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, 1, 1, 1), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, groups=groups)
    y_ref = F.conv3d(x, weight, groups=groups)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# groups reaches _conv3d_impl through the degenerate-5D wrappers' **kwargs.
@_skip_non_cdna4
@pytest.mark.parametrize("groups", [2, 4, 16])
def test_conv2d_grouped_vs_torch(groups):
    torch.manual_seed(9300 + groups)
    n, c, h, w, k = 2, 64, 20, 24, 128
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, padding=1, groups=groups)
    y_ref = F.conv2d(x, weight, bias=bias.to(torch.bfloat16), padding=1, groups=groups)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize("groups", [2, 8])
def test_conv1d_grouped_vs_torch(groups):
    torch.manual_seed(9400 + groups)
    n, c, w, k = 2, 64, 96, 128
    x = torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, padding=1, groups=groups)
    y_ref = F.conv1d(x, weight, padding=1, groups=groups)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Mismatched shapes must fail fast rather than compute something wrong.
@_skip_non_cdna4
def test_conv3d_grouped_invalid_shapes():
    x = torch.randn((1, 12, 4, 8, 8), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="not divisible by groups"):
        conv3d_implicit(x, torch.randn((16, 3, 3, 3, 3), device="cuda", dtype=torch.bfloat16), groups=5)
    with pytest.raises(AssertionError, match="weight in-channels"):
        conv3d_implicit(x, torch.randn((16, 12, 3, 3, 3), device="cuda", dtype=torch.bfloat16), groups=4)
