#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 implicit-MFMA conv2d kernel.

Compares ``conv2d_implicit_mfma(x, w)`` against ``torch.nn.functional.conv2d``
on the same NCHW/OIHW bf16 inputs (stride 1, no padding). ``torch.allclose``
also catches NaN/inf.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels.conv2d_implicit_mfma import conv2d_implicit_mfma

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


# (N, C, H, W, K, R, S); covers both npq % TILE_M == 0 and != 0 paths.
@pytest.mark.parametrize(
    "n,c,h,w,k,r,s",
    [
        (8, 32, 62, 62, 64, 3, 3),
        (8, 32, 122, 122, 64, 3, 3),
        (8, 32, 242, 242, 64, 3, 3),
        (8, 32, 482, 482, 64, 3, 3),
    ],
)
def test_conv2d_implicit_mfma(n, c, h, w, k, r, s):
    torch.manual_seed(1000 + h + w)
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, r, s), device="cuda", dtype=torch.bfloat16)

    y = conv2d_implicit_mfma(x, weight)
    y_ref = F.conv2d(x, weight)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
