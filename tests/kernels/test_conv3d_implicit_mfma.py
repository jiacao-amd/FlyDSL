#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 implicit-MFMA conv3d kernel.

Compares against ``torch.nn.functional.conv3d`` on NCDHW/OIDHW bf16 inputs for
the symmetric-padding path, and against a manual causal reference (front-padded
time + same spatial padding) for the Wan ``CausalConv3d`` path.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels.conv3d_implicit_mfma import causal_conv3d_implicit_mfma, conv3d_implicit_mfma

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


# (N, C, T, H, W, K), kernel 3x3x3. Covers stride/padding and M%TILE_M paths.
@pytest.mark.parametrize(
    "n,c,t,h,w,k,stride,padding",
    [
        (1, 32, 8, 16, 16, 64, 1, 0),
        (1, 32, 9, 17, 17, 96, 1, 1),
        (2, 64, 6, 18, 18, 192, 1, 1),
        (1, 32, 10, 20, 20, 64, 2, 1),
        (1, 3, 9, 32, 32, 96, 1, 1),  # small-C (RGB conv_in): C=3, crs=81 not %32
        (1, 6, 7, 18, 18, 64, 1, 1),  # small-C with a_vec=2: C=6
    ],
)
def test_conv3d_vs_torch(n, c, t, h, w, k, stride, padding):
    torch.manual_seed(2000 + h + w + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit_mfma(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Wan-like CausalConv3d(C, K, 3, padding=1): C/K in the VAE 96-384+ range.
@pytest.mark.parametrize(
    "n,c,t,h,w,k",
    [
        (1, 96, 5, 16, 16, 96),
        (1, 96, 7, 16, 16, 192),
    ],
)
def test_causal_conv3d_vs_reference(n, c, t, h, w, k):
    torch.manual_seed(3000 + c + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = causal_conv3d_implicit_mfma(x, weight, stride=1, padding=1)

    # Reference: causal temporal pad (front 2, back 0), same spatial pad (1,1).
    x_pad = F.pad(x, (1, 1, 1, 1, 2, 0))
    y_ref = F.conv3d(x_pad, weight, stride=1, padding=0)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)
