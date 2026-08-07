# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Double-buffered implicit-GEMM conv3d (BF16).

x: (N, C, D, H, W) bf16 NCDHW, weight: (K, C/groups, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16. Supports stride, padding, dilation, bias, groups,
and split-K.
"""

import functools
import os
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops
from kernels.common.mem_ops import buffer_atomic_add

TILE_K = 32
STAGES = 2
WARP_SIZE = 64

MFMA_M = 16
MFMA_N = 16
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

LDG_VEC = 8

BF16_BYTES = 2

DEFAULT_TILE = (128, 128, 2, 4)

# Applied around both tracing and flyc.compile so the hinted and the fast-dispatch
# paths lower identically. Keys are the ones rocm.py reads: waves_per_eu, maxnreg,
# fast_fp_math, unsafe_fp_math, llvm_options.
#
# Deliberately empty. Swept on gfx950 over five shapes (g in {1,2,4,8}, small and
# large): fast_fp_math is a wash -- MFMA does the arithmetic and the epilogue is a
# single f32->bf16 convert, so there is nothing to reassociate -- and waves_per_eu
# of 1/2 measured 3%/11% slower, 4 within noise. Left as the hook the launchers
# already thread through, not as a tuning knob that currently pays.
CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)


def _dispatch(exe, *args, stream=None):
    """Run a builder's launcher, pre-compiling on first use.

    ``exe.compile(...)`` both compiles and executes, and hands back a
    ``CompiledFunction`` whose call path skips signature binding and cache lookup
    (~5 us vs ~35 us for the @flyc.jit wrapper). Cached on the launcher, which is
    itself memoized per problem shape by the builder's lru_cache.
    """
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = exe.compile(*args, stream=stream)
        return
    cf(*args, _as_stream(stream))


def _autotune_enabled():
    return os.environ.get("FLYDSL_CONV3D_AUTOTUNE", "0").lower() in ("1", "true", "yes")


_WEIGHT_CACHE = {}


def _pad_channels(c):
    return (c + LDG_VEC - 1) // LDG_VEC * LDG_VEC


def _prep_weight(w, k, kt, kh, kw, c):
    key = id(w)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is w:
        return ent[1]
    cp = _pad_channels(c)
    wsrc = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cp - c)) if cp != c else w
    wk = wsrc.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * cp)
    _WEIGHT_CACHE[key] = (weakref.ref(w), wk)
    return wk


TR_TILE = 64
TR_VEC = 8
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
_TR_PAD = 8
_TR_LDS_S = TR_TILE + _TR_PAD


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc(n, c, s):
    """Transpose flat (N, C, S) -> (N, S, C) (S == T*H*W). Requires c%8==0."""
    grid_s = (s + TR_TILE - 1) // TR_TILE
    grid_c = (c + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16
    BIG = (n * c * s) > 0x7FFFFFFF

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        # max_size: an exact num_records would zero the whole straddling tail read.
        in_rsrc = buffer_ops.create_buffer_resource(inp)
        out_rsrc = buffer_ops.create_buffer_resource(out)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]).peek()

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
        if const_expr(BIG):
            in_base_elem = fx.Index(nb) * fx.Index(c) * fx.Index(s) + fx.Index(c0) * fx.Index(s) + fx.Index(s0)
            in_addr = fx.Int64(buffer_ops.extract_base_index(inp)) + fx.Int64(in_base_elem) * fx.Int64(2)
            in_rsrc = buffer_ops.create_buffer_resource_from_addr(in_addr)
            out_base_elem = fx.Index(nb) * fx.Index(s) * fx.Index(c) + fx.Index(s0) * fx.Index(c) + fx.Index(c0)
            out_addr = fx.Int64(buffer_ops.extract_base_index(out)) + fx.Int64(out_base_elem) * fx.Int64(2)
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(out_addr)
        else:
            in_base = nb * c * s
            out_base = nb * s * c

        def lds_store_vec8(elem_offset, value):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            ptr = buffer_ops.create_llvm_ptr(base, address_space=3)
            llvm.StoreOp(value, ptr, alignment=16)

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=BF16Ty)

        # Read: coalesced vec8 along contiguous S -> LDS[c_local][s_local].
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // _TR_VPL
            sv = (lin % _TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            valid = (cc < c) & (ss < s)
            if const_expr(BIG):
                g = fx.Int32(rc * s + sv)
            else:
                g = fx.Int32(in_base + cc * s + ss)
            safe = arith.select(valid, g, fx.Int32(0))
            v = buffer_ops.buffer_load(in_rsrc, safe, vec_width=TR_VEC, dtype=elem_ty)
            lds_store_vec8(rc * _TR_LDS_S + sv, v)

        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True)

        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // _TR_VPL
            cv = (lin % _TR_VPL) * TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            scalars = [lds_load_scalar((cv + j) * _TR_LDS_S + rs) for j in range_constexpr(TR_VEC)]
            vv = fx.Vector.from_elements(scalars, dtype=elem_ty)
            valid = (ss < s) & (cc < c)
            if valid:
                if const_expr(BIG):
                    go = fx.Int32(rs * c + cv)
                else:
                    go = fx.Int32(out_base + ss * c + cc)
                buffer_ops.buffer_store(vv, out_rsrc, go)

    @flyc.jit
    def launch_transpose(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(
            grid=(grid_s, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_transpose(out, inp, stream=_as_stream(stream))

    def _compile(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_transpose, out, inp, _as_stream(stream))

    _launch.compile = _compile
    return _launch


def _ncdhw_to_ndhwc(x, stream):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch."""
    n, c, t, h, w = x.shape
    s = t * h * w
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % TR_VEC == 0):
        return x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((n, t, h, w, c), device=x.device, dtype=x.dtype)
    exe = compile_transpose_ncdhw_ndhwc(n, c, s)
    _dispatch(exe, out, x, stream=torch.cuda.current_stream() if stream is None else stream)
    return out


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit(
    n,
    c,
    d,
    h,
    w,
    k,
    kt,
    kh,
    kw,
    st,
    sh,
    sw,
    pt,
    ph,
    pw,
    dt=1,
    dh=1,
    dw=1,
    has_bias=False,
    splitk=1,
    tile=DEFAULT_TILE,
    wgm=1,
    groups=1,
):
    TILE_M, TILE_N, WAVE_M, WAVE_N = tile
    BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE
    # Per-wave MFMA grid (flat acc[mi * MI_N + ni]); WARP_M/N is the per-wave tile span.
    MI_M = TILE_M // WAVE_M // MFMA_M
    MI_N = TILE_N // WAVE_N // MFMA_N
    N_ACC = MI_M * MI_N
    WARP_M = MI_M * MFMA_M
    WARP_N = MI_N * MFMA_N
    BLOCK_VECS = LDG_VEC * BLOCK_THREADS
    LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
    LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS

    # `c` is the padded TOTAL channel count and stays the NDHWC row stride. CGP is the
    # per-group channel count and is what the GEMM K axis decomposes against; the two
    # coincide only when groups == 1.
    CGP = c // groups
    KG = k // groups

    assert TILE_K == 32
    assert TILE_M % (WAVE_M * MFMA_M) == 0, f"TILE_M={TILE_M} not divisible by WAVE_M*16"
    assert TILE_N % (WAVE_N * MFMA_N) == 0, f"TILE_N={TILE_N} not divisible by WAVE_N*16"
    assert (TILE_M * TILE_K) % BLOCK_VECS == 0, f"A tile {TILE_M}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert (TILE_N * TILE_K) % BLOCK_VECS == 0, f"B tile {TILE_N}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert LDG_A_COUNT >= 1 and LDG_B_COUNT >= 1
    assert c % groups == 0, f"c={c} not divisible by groups={groups}"
    assert k % groups == 0, f"k={k} not divisible by groups={groups}"
    assert CGP % LDG_VEC == 0, f"c/groups={CGP} must be a multiple of LDG_VEC={LDG_VEC}; use _conv3d_impl to pad"
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"

    # Dilation only stretches the filter's footprint; the K axis (CRS) is unchanged.
    do = (d + 2 * pt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = CGP * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K

    BIG_IN = (n * c * d * h * w) > 0x7FFFFFFF
    BIG_OUT = (n * k * do * ho * wo * BF16_BYTES) > 0x7FFFFFFF

    X_BYTES = n * c * d * h * w * BF16_BYTES
    W_BYTES = k * crs * BF16_BYTES
    OOB_SENTINEL_ELEM = 0x7FFFFF80  # *2 = 0xFFFFFF00 bytes (~4.2950 GB), just under 2^32
    OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES
    BIG_IN_NR = 0x80000000  # 2 GB num_records for the rebased BIG_IN resource
    assert W_BYTES < OOB_SENTINEL_BYTES, f"weight {W_BYTES}B exceeds limit {OOB_SENTINEL_BYTES}B"
    assert X_BYTES < OOB_SENTINEL_BYTES or BIG_IN, f"input {X_BYTES}B exceeds limit"
    BIG_IN_N1 = BIG_IN and n == 1
    BIG_IN_NM = BIG_IN and n > 1
    X_SAMPLE_BYTES = c * d * h * w * BF16_BYTES

    # A tile must never straddle a group boundary -- every column in it shares one A tile in
    # LDS, and different groups need different input channels. So the N grid is
    # over-provisioned per group and the per-group tail is masked.
    tiles_per_group = (KG + TILE_N - 1) // TILE_N
    n_tail = KG % TILE_N != 0
    grid_n = groups * tiles_per_group

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    # Software-pipeline depth. 4 stages is optimal across all shapes on gfx950 --
    # even short-K, memory-bound 3x1x1 depends more (not less) on deep prefetch to
    # hide DMA latency; a shallower pipeline measured slower (2/3/4-stage A/B).
    PIPE_STAGES = 4

    LDS_A_SIZE = PIPE_STAGES * TILE_M * TILE_K
    LDS_B_SIZE = PIPE_STAGES * TILE_N * TILE_K

    grid_m = (npq + TILE_M - 1) // TILE_M
    WGM = max(1, int(wgm))
    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16
    temporal_only_fast = (
        kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and do == d
        and ho == h
        and wo == w
    )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        w_rsrc = buffer_ops.create_buffer_resource(weight, num_records_bytes=W_BYTES)
        if const_expr(not BIG_IN):
            x_rsrc = buffer_ops.create_buffer_resource(x, num_records_bytes=X_BYTES)
        y_rsrc = buffer_ops.create_buffer_resource(y)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias)

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        if const_expr(WGM > 1):
            pid = fx.Index(fx.block_idx.x) + fx.Index(fx.block_idx.y) * fx.Index(grid_m)
            blocks_per_swizzle = fx.Index(WGM * grid_n)
            swizzle_id = pid // blocks_per_swizzle
            first_m = swizzle_id * fx.Index(WGM)
            swizzle_rows = fx.Index(grid_m) - first_m
            swizzle_rows = fx.Index(arith.select(swizzle_rows < fx.Index(WGM), swizzle_rows, fx.Index(WGM)))
            local = pid % blocks_per_swizzle
            m_offset = fx.Index(first_m + (local % swizzle_rows)) * TILE_M
            n_tile = fx.Index(local // swizzle_rows)
        else:
            m_offset = fx.block_idx.x * TILE_M
            n_tile = fx.block_idx.y
        # n_offset is the GLOBAL output-channel base (drives the B row and the store),
        # n_local is the base within this group (drives every tail check), and ch_base is
        # this group's first input channel. All three are block-uniform.
        if const_expr(groups > 1):
            gi = n_tile // tiles_per_group
            n_local = (n_tile % tiles_per_group) * TILE_N
            n_offset = gi * KG + n_local
            ch_base = gi * CGP
        else:
            n_offset = n_tile * TILE_N
            n_local = n_offset
        if const_expr(use_splitk):
            k_off = fx.block_idx.z * (tiles_per_split * TILE_K)
        else:
            k_off = 0

        if const_expr(BIG_IN_N1):
            nbase = m_offset // dhw
            ot_base0 = (m_offset % dhw) // hw_o
            base_t = ot_base0 - fx.Index(pt)
            base_t = arith.select(base_t < fx.Index(0), fx.Index(0), base_t)
            x_base_elem = ((nbase * fx.Index(d) + base_t) * fx.Index(h) + fx.Index(0)) * fx.Index(w) * fx.Index(c)
            x_addr = fx.Int64(buffer_ops.extract_base_index(x)) + fx.Int64(x_base_elem) * fx.Int64(2)
            x_rsrc = buffer_ops.create_buffer_resource_from_addr(x_addr, num_records_bytes=BIG_IN_NR)
        if const_expr(BIG_IN_NM):
            x_base_addr = fx.Int64(buffer_ops.extract_base_index(x))

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // WAVE_N
        wave_n = wid % WAVE_N

        lane_m = lane % MFMA_M
        lane_n = lane % MFMA_N
        lane_k_a = lane // MFMA_M * MFMA_A_VALUES
        lane_k_b = lane // MFMA_N * MFMA_B_VALUES
        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(8, elem_ty)

        acc0 = Vec.filled(MFMA_C_VALUES, 0.0, fx.Float32)
        acc = [acc0 for _ in range_constexpr(N_ACC)]

        def barrier(vmcnt=0, lgkmcnt=None):
            waits = []
            if vmcnt is not None:
                waits.append(f"vmcnt({vmcnt})")
            if lgkmcnt is not None:
                waits.append(f"lgkmcnt({lgkmcnt})")
            pre = ("s_waitcnt " + " ".join(waits) + "\n\t") if waits else ""
            llvm.InlineAsmOp(None, [], f"{pre}s_barrier", "", has_side_effects=True)

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def a_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def in_range(v, hi):
            return (v >= 0) & (v < fx.Index(hi))

        def dil(tap, factor):
            # Filter tap -> input offset. Kept off the multiply when undilated.
            scaled = tap * factor if const_expr(factor != 1) else tap
            return scaled

        # ---- Per-thread row decomposition (loop-invariant across K) ----
        _row_dec = []  # per-i tuple of precomputed row terms
        for i in range_constexpr(LDG_A_COUNT):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + local_m
            row_valid = row < fx.Index(npq)
            if const_expr(temporal_only_fast):
                out_t = (row // hw_o) % d
                _row_dec.append((local_k, row, row_valid, out_t))
            else:
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                in_t0 = ot * st - pt
                in_h0 = oh * sh - ph
                in_w0 = ow * sw - pw
                if const_expr(BIG_IN_N1):
                    di = n_idx - nbase
                    _row_dec.append((local_k, row_valid, di, in_t0, in_h0, in_w0))
                elif const_expr(BIG_IN_NM):
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))
                else:
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))

        SCALAR_K = CGP % TILE_K == 0

        # ---- 3D im2col address math ----
        # The K axis decomposes against CGP (per-group channels) while every g_off below
        # keeps `c` (padded total channels) as the NDHWC row stride. `cc` is the absolute
        # input channel: the group base plus the offset within the group.
        def _a_addr(i, kbase_i, cc_base, ckk_base):
            dec = _row_dec[i]
            local_k = dec[0]
            k_abs = kbase_i + fx.Index(local_k)
            if const_expr(SCALAR_K):
                cc = cc_base + fx.Index(local_k)  # cc_base already carries ch_base
            else:
                cc = k_abs % CGP
                if const_expr(groups > 1):
                    cc = ch_base + cc
            k_valid = k_abs < fx.Index(crs)
            if const_expr(temporal_only_fast):
                _, row, row_valid, out_t = dec
                kt_i = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                temporal_delta = dil(kt_i, dt) - pt
                in_t = out_t + temporal_delta
                valid = row_valid & k_valid & in_range(in_t, d)
                if const_expr(BIG_IN_N1):
                    g_off = ((row + temporal_delta * hw_o) - (fx.Index(nbase) * dhw + base_t * hw_o)) * c + cc
                else:
                    g_off = (row + temporal_delta * hw_o) * c + cc
            else:
                ckk = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                if const_expr(BIG_IN_N1):
                    _, row_valid, di, in_t0, in_h0, in_w0 = dec
                    in_t = in_t0 + dil(kt_i, dt)
                    in_h = in_h0 + dil(kh_i, dh)
                    in_w = in_w0 + dil(kw_i, dw)
                    valid = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
                    g_off = (((di * d + (in_t - base_t)) * h + in_h) * w + in_w) * c + cc
                elif const_expr(BIG_IN_NM):
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t = in_t0 + dil(kt_i, dt)
                    in_h = in_h0 + dil(kh_i, dh)
                    in_w = in_w0 + dil(kw_i, dw)
                    valid = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
                    g_off = ((in_t * h + in_h) * w + in_w) * c + cc
                    return fx.Int32(g_off), valid, n_idx
                else:
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t = in_t0 + dil(kt_i, dt)
                    in_h = in_h0 + dil(kh_i, dh)
                    in_w = in_w0 + dil(kw_i, dw)
                    valid = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
                    g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
            return fx.Int32(g_off), valid

        def _b_addr(i, k_base):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(local_n)
            g_off = fx.Int32(col * crs + (fx.Index(k_base) + fx.Index(local_k)))
            # Tail is per group: the N grid is over-provisioned to groups*tiles_per_group.
            col_valid = ((n_local + fx.Index(local_n)) < fx.Index(KG)) if const_expr(n_tail) else None
            return g_off, col_valid

        # ---- global -> LDS DMA copy, masking via OOB routing ----
        DMA_BYTES = LDG_VEC * BF16_BYTES  # 16
        OOB_ELEM = fx.Int32(OOB_SENTINEL_ELEM)

        def _lds_dma_ptr(lds_array, stage_tile, i):
            off_elems = fx.Index(stage_tile) + (fx.Index(tid) + fx.Index(i * BLOCK_THREADS)) * fx.Index(LDG_VEC)
            base_bytes = off_elems * fx.Index(BF16_BYTES)
            addr = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(base_bytes)
            addr = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, addr.ir_value()))
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), addr)

        def _dma_to_lds(rsrc, lds_ptr, voff_elem):
            voff_b = (voff_elem * fx.Int32(BF16_BYTES)).ir_value()
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                arith.constant(DMA_BYTES, type=T.i32),
                voff_b,
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            )

        def _load_a(stage, k_base):
            kbase_i = fx.Index(k_base)
            cc_base = ckk_base = None
            if const_expr(SCALAR_K):
                # Loop-invariant and block-uniform, so folding ch_base in here costs no
                # per-load instructions on the SCALAR_K path.
                cc_base = kbase_i % CGP
                if const_expr(groups > 1):
                    cc_base = ch_base + cc_base
                ckk_base = kbase_i // CGP
            stage_tile = fx.Index(stage) * TILE_M * TILE_K
            for i in range_constexpr(LDG_A_COUNT):
                if const_expr(BIG_IN_NM):
                    addr_ret = _a_addr(i, kbase_i, cc_base, ckk_base)
                    g_off_i, valid, n_idx_i = addr_ret
                    sample_addr = x_base_addr + fx.Int64(n_idx_i) * fx.Int64(X_SAMPLE_BYTES)
                    x_rsrc_i = buffer_ops.create_buffer_resource_from_addr(sample_addr, num_records_bytes=BIG_IN_NR)
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_rsrc_i, _lds_dma_ptr(a_lds, stage_tile, i), voff)
                else:
                    g_off_i, valid = _a_addr(i, kbase_i, cc_base, ckk_base)
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_rsrc, _lds_dma_ptr(a_lds, stage_tile, i), voff)

        def _load_b(stage, k_base):
            stage_tile = fx.Index(stage) * TILE_N * TILE_K
            for i in range_constexpr(LDG_B_COUNT):
                g_off, col_valid = _b_addr(i, k_base)
                if const_expr(n_tail):
                    voff = fx.Int32(arith.select(col_valid, g_off, OOB_ELEM))
                else:
                    voff = g_off
                _dma_to_lds(w_rsrc, _lds_dma_ptr(b_lds, stage_tile, i), voff)

        # ---- single-vec ds_read (LDS -> register), indexed by per-wave MFMA row ----
        def read_a_vec(stage, mi):
            a_row = wave_m * WARP_M + mi * MFMA_M + lane_m
            return lds_load_vec8(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(lane_k_a)))

        def read_b_vec(stage, ni):
            b_row = wave_n * WARP_N + ni * MFMA_N + lane_n
            return lds_load_vec8(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(lane_k_b)))

        def mfma_one(a_frag, b_frag, c_frag):
            out = mfma_fn(
                T.vec(MFMA_C_VALUES, T.f32),
                [a_frag, b_frag, c_frag, 0, 0, 0],
            )
            rocdl.sched_mfma(1)
            return out

        def read_a_frags(stage):
            frags = [read_a_vec(stage, mi) for mi in range_constexpr(MI_M)]
            rocdl.sched_dsrd(MI_M)
            return frags

        def read_b_frags(stage):
            frags = [read_b_vec(stage, ni) for ni in range_constexpr(MI_N)]
            rocdl.sched_dsrd(MI_N)
            return frags

        def do_compute(acc_values, a_frag_values, b_frag_values):
            rocdl.s_setprio(1)
            for mi in range_constexpr(MI_M):
                for ni in range_constexpr(MI_N):
                    idx = mi * MI_N + ni
                    acc_values[idx] = mfma_one(a_frag_values[mi], b_frag_values[ni], acc_values[idx])
            rocdl.s_setprio(0)
            return acc_values

        # global->LDS software pipeline
        # ---- prologue: fill the pipeline with the first PREFETCH tiles' DMAs ----
        PREFETCH = PIPE_STAGES - 1
        for s in range_constexpr(PREFETCH):
            if const_expr(s < tiles_per_split):
                _load_a(s, k_off + s * TILE_K)
                _load_b(s, k_off + s * TILE_K)
        LDG_PER_TILE = LDG_A_COUNT + LDG_B_COUNT

        # ---- main loop: wait oldest tile, read frags, launch tile PREFETCH ahead, compute ----
        for kt_idx in range_constexpr(tiles_per_split):
            cur = kt_idx % PIPE_STAGES
            inflight_tiles = min(PREFETCH - 1, tiles_per_split - 1 - kt_idx)
            barrier(vmcnt=inflight_tiles * LDG_PER_TILE, lgkmcnt=0)
            a_frags = read_a_frags(cur)
            b_frags = read_b_frags(cur)
            nxt = kt_idx + PREFETCH
            if const_expr(nxt < tiles_per_split):
                _load_a(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                _load_b(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)
            acc = do_compute(acc, a_frags, b_frags)

        _row_chk = npq % TILE_M != 0
        _need_chk = _row_chk or n_tail
        _vec_store = (n == 1) and (not use_splitk) and (dhw % MFMA_C_VALUES == 0) and (not BIG_OUT)

        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(buffer_ops.extract_base_index(y))

        def _big_store(off_nk_i64, value):
            addr = y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            llvm.StoreOp(value.ir_value() if hasattr(value, "ir_value") else value, ptr, alignment=2)

        # col_loc is the column within its group; the tail check is per group because the N
        # grid is over-provisioned. At groups == 1 it is the same value as col.
        def _valid_raw(row, col_loc):
            if const_expr(_row_chk and n_tail):
                return arith.andi(row < fx.Index(npq), col_loc < fx.Index(KG))
            if const_expr(_row_chk):
                v = row < fx.Index(npq)
                return arith.andi(v, v)
            v = col_loc < fx.Index(KG)
            return arith.andi(v, v)

        def store_acc():
            for mi in range_constexpr(MI_M):
                row_base = m_offset + wave_m * WARP_M + mi * MFMA_M + c_m_vec
                for ni in range_constexpr(MI_N):
                    col_off = fx.Index(wave_n * WARP_N + ni * MFMA_N + c_n)
                    col = n_offset + col_off
                    col_loc = (n_local + col_off) if const_expr(groups > 1) else col
                    a = Vec(acc[mi * MI_N + ni])
                    if const_expr(has_bias and not use_splitk):
                        col_i = fx.Int32(col)  # bias is indexed by the global out-channel
                        if const_expr(n_tail):
                            col_i = arith.select(col_loc < fx.Index(KG), col_i, fx.Int32(0))
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))

                    if const_expr(_vec_store):
                        row0 = fx.Index(row_base)
                        off_nk0 = col * dhw + row0

                        def _emit_vec():
                            vals = []
                            for i in range_constexpr(MFMA_C_VALUES):
                                cval = (a[i] + bias_val) if const_expr(has_bias) else a[i]
                                vals.append(cval.to(elem_ty))
                            v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                            buffer_ops.buffer_store(v4, y_rsrc, off_nk0)

                        if const_expr(_need_chk):
                            if _valid_raw(row0, col_loc):
                                _emit_vec()
                        else:
                            _emit_vec()
                        continue

                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Index(row_base + i)
                        off_sk = row * k + col

                        if const_expr(n == 1):
                            off_nk = col * dhw + row
                        else:
                            n_idx = row // dhw
                            sp = row % dhw
                            off_nk = n_idx * (k * dhw) + col * dhw + sp

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = fx.Int32(off_sk * 4)
                                z0 = fx.Int32(0)
                                buffer_atomic_add(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
                                if const_expr(BIG_OUT):
                                    _big_store(fx.Int64(off_nk), cval)
                                else:
                                    buffer_ops.buffer_store(cval, y_rsrc, off_nk)

                        if const_expr(_need_chk):
                            if _valid_raw(row, col_loc):
                                _emit()
                        else:
                            _emit()

        store_acc()

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_implicit_kernel(y, x, weight, bias).launch(
            grid=(grid_m, grid_n, splitk), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    def _launch(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch(y, x, weight, bias, stream=_as_stream(stream))

    def _compile(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch, y, x, weight, bias, _as_stream(stream))

    _launch.compile = _compile
    return _launch


def _resolve_splitk(splitk, npq, crs, k, device, tile=DEFAULT_TILE, groups=1):
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        kg = k // groups
        base = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        if (
            npq < 4096
            or k_tiles < 16
            or kg % tile_n != 0
            or npq % tile_m != 0
            or crs % TILE_K != 0
            or npq * k * 4 > 0x7FFFFFFF
        ):
            sk = 1
        else:
            try:
                num_cu = torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                num_cu = 256
            if base >= (3 * num_cu) // 4:
                sk = 1
            else:
                sk = min(4, max(1, num_cu // base), k_tiles)
    else:
        sk = max(1, splitk)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def _conv3d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    splitk=None,
    stream=None,
    tile=None,
    autotune=None,
):
    n, c, d, h, w = x.shape
    k, wc, kt, kh, kw = weight.shape
    groups = int(groups)
    assert groups >= 1, f"groups must be >= 1, got {groups}"
    assert c % groups == 0, f"in-channels {c} not divisible by groups {groups}"
    assert k % groups == 0, f"out-channels {k} not divisible by groups {groups}"
    assert wc == c // groups, f"weight in-channels {wc} != C/groups = {c // groups}"
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    st, sh, sw = (stride, stride, stride) if isinstance(stride, int) else stride
    pt, ph, pw = (padding, padding, padding) if isinstance(padding, int) else padding
    dt, dh, dw = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
    assert min(dt, dh, dw) >= 1, f"dilation must be >= 1, got {(dt, dh, dw)}"

    # 1x1x1 fast path: y[n,k,dhw] = sum_c weight[k,c] * x[n,c,dhw] — pure channel GEMM.
    # Grouped 1x1x1 is block-diagonal, so it goes through the kernel instead.
    if (
        groups == 1
        and kt == 1
        and kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and pt == 0
        and ph == 0
        and pw == 0
    ):
        wm = weight.reshape(k, c)
        if n == 1:
            y = torch.matmul(wm, x.reshape(c, d * h * w)).reshape(n, k, d, h, w)
        else:
            y = torch.matmul(wm, x.reshape(n, c, d * h * w)).reshape(n, k, d, h, w)
        if bias is not None:
            y = y + bias.to(y.dtype).view(1, k, 1, 1, 1)
        return y

    do = (d + 2 * pt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    assert min(do, ho, wo) >= 1, f"dilated filter is larger than the padded input: output ({do}, {ho}, {wo})"
    npq = n * do * ho * wo

    # An empty batch would launch the transpose with grid.z == 0 and the conv with grid.x == 0;
    # both return hipErrorInvalidValue and leave the HIP context unusable. Return the empty
    # output torch produces instead, before any launch.
    if n == 0:
        return torch.empty((0, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)

    # Zero-pad C to the gather's vector width; padded channels see zero weights. The pad is
    # PER GROUP, since the gather vectorizes along channels and must not cross into the next
    # group. _prep_weight pads the weight's own C the same way, so the two stay aligned.
    cg = c // groups
    cgp = _pad_channels(cg)
    if cgp != cg:
        x = torch.nn.functional.pad(x.reshape(n, groups, cg, d, h, w), (0, 0, 0, 0, 0, 0, 0, cgp - cg))
        x = x.reshape(n, groups * cgp, d, h, w)
    c = groups * cgp
    crs = cgp * kt * kh * kw

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else torch.empty(1, device=x.device, dtype=torch.float32)

    x_ndhwc = _ncdhw_to_ndhwc(x, stream)
    w_packed = _prep_weight(weight, k, kt, kh, kw, wc)

    shape = (n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, dt, dh, dw, has_bias, groups)

    def _run(the_tile, the_wgm=1):
        sk = _resolve_splitk(splitk, npq, crs, k, x.device, the_tile, groups)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            y = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        exe = compile_conv3d_implicit(
            n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, dt, dh, dw, has_bias, sk, the_tile, the_wgm, groups
        )
        _dispatch(exe, y, x_ndhwc, w_packed, bias_arg, stream=launch_stream)
        return y, sk

    if tile is not None:
        chosen_tile = tuple(tile)
        chosen_wgm = 1
    elif autotune or (autotune is None and _autotune_enabled()):
        from kernels.conv.conv3d_autotune import BF16_CANDIDATES, WGM_VALUES, autotune_conv3d

        candidates = [(t, w) for t in BF16_CANDIDATES for w in WGM_VALUES]
        best = autotune_conv3d("bf16", shape, "bf16", candidates, x.device, lambda tw: _run(tw[0], tw[1])[0])
        chosen_tile, chosen_wgm = best
    else:
        # A tile never spans groups, so a 128-wide N tile sits mostly empty when K/groups is
        # small; drop to the narrowest legal candidate. Autotune picks properly when enabled.
        chosen_tile = (64, 64, 2, 2) if (groups > 1 and k // groups < DEFAULT_TILE[1]) else DEFAULT_TILE
        chosen_wgm = 1

    y, sk = _run(chosen_tile, chosen_wgm)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        # The split-K accumulator is the raw GEMM shape, i.e. NDHWC, so the permute below is
        # a metadata-only relabel and leaves the data channels-last. Materialize contiguous
        # NCDHW so the memory format does not depend on whether split-K ran. copy_ folds the
        # f32 -> bf16 cast into the transpose, keeping this to the single pass the cast cost
        # anyway.
        out = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        out.copy_(y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3))
        return out
    return y


def _conv2d_impl(x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d expects (N,C,H,W) / (K,C,R,S)"
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    n, c, h, w = x.shape
    k, wc, r, s = weight.shape
    x5 = x.reshape(n, c, 1, h, w)
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl(x5, w5, bias=bias, stride=(1, sh, sw), padding=(0, ph, pw), dilation=(1, dh, dw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl(x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d expects (N,C,W) / (K,C,S)"
    sw = stride if isinstance(stride, int) else stride[0]
    pw = padding if isinstance(padding, int) else padding[0]
    dw = dilation if isinstance(dilation, int) else dilation[0]
    n, c, w = x.shape
    k, wc, s = weight.shape
    x5 = x.reshape(n, c, 1, 1, w)
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl(x5, w5, bias=bias, stride=(1, 1, sw), padding=(0, 0, pw), dilation=(1, 1, dw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit(x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs):
    """Main implicit-GEMM conv entry; dispatches 1D/2D/3D by filter rank.

    Rank is taken from the filter (weight.dim() - 2): 3 -> 3D (N,C,D,H,W)/(K,C,T,R,S),
    2 -> 2D (N,C,H,W)/(K,C,R,S), 1 -> 1D (N,C,W)/(K,C,S). Like torch, ``x`` may also
    be unbatched -- (C,D,H,W) / (C,H,W) / (C,W), one rank below the filter -- in which
    case the output is unbatched too. True 3D calls run the implementation directly;
    2D/1D reshape to the degenerate 5D case. stride/padding/dilation/bias and extra
    kwargs (groups, splitk, tile, autotune, stream) forward to the chosen path.

    ``dilation`` follows torch semantics: it spaces the filter taps by that factor
    over the input, shrinking the output to
    ``(D + 2*pad - dilation*(T-1) - 1)//stride + 1`` per axis. It costs nothing in the
    GEMM -- the K axis is still C/groups*T*R*S -- it only stretches the im2col gather,
    so a dilated filter reads a wider input footprint per output row and gets less
    reuse out of cache than the same filter undilated.

    ``groups`` follows torch semantics: C and K must both be divisible by it and the
    weight's channel dim is C/groups. Groups map onto the N grid axis, one tile never
    spanning two groups, so efficiency tracks how well K/groups fills TILE_N. Measured
    on gfx950 vs torch/MIOpen, moderate cardinality wins across the board (1.5-2.0x for
    K/groups in [8, 256]). True depthwise (groups == C, so C/groups == 1) is the one
    weak case at ~0.5x: C/groups=1 pads to the gather's 8-wide vector, wasting 7/8 of
    the K axis, while K/groups=1 leaves all but one column of the N tile masked.
    Narrower tiles recover little there -- depthwise wants its own kernel, not this
    single-GEMM mapping.
    """
    spatial_rank = weight.dim() - 2
    if spatial_rank not in (1, 2, 3):
        raise ValueError(f"conv3d_implicit supports 1D/2D/3D; got filter rank {weight.dim()}")
    # An unbatched (C, *spatial) input runs as a batch of one and loses the dim again on
    # the way out, matching torch.
    unbatched = x.dim() == weight.dim() - 1
    if unbatched:
        x = x.unsqueeze(0)
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    impl = {3: _conv3d_impl, 2: _conv2d_impl, 1: _conv1d_impl}[spatial_rank]
    y = impl(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, **kwargs)
    return y.squeeze(0) if unbatched else y
