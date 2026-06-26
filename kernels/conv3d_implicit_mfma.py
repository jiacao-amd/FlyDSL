import functools
import os
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

TILE_M = 128
TILE_N = 64
TILE_K = 32
STAGES = 2

BLOCK_M_WARPS = 4
BLOCK_N_WARPS = 1
WARP_SIZE = 64
BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE

MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

WARP_M = TILE_M // BLOCK_M_WARPS
WARP_N = TILE_N // BLOCK_N_WARPS
WARP_M_STEPS = WARP_M // MFMA_M
WARP_N_STEPS = WARP_N // MFMA_N
WARP_K_STEPS = TILE_K // MFMA_K

LDG_VEC = 8
BLOCK_VECS = LDG_VEC * BLOCK_THREADS
LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS
LDS_A_SIZE = STAGES * TILE_M * TILE_K
LDS_B_SIZE = STAGES * TILE_N * TILE_K


_WEIGHT_CACHE = {}


def _prep_weight(w, k, kt, kh, kw, c):
    key = id(w)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is w:
        return ent[1]
    wk = w.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * c)
    _WEIGHT_CACHE[key] = (weakref.ref(w), wk)
    return wk


def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


TR_TILE = 64
TR_VEC = 8  
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC  
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)  
_TR_PAD = 8 
_TR_LDS_S = TR_TILE + _TR_PAD


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc(n, c, s):
    """Transpose flat (N, C, S) -> (N, S, C) (S == T*H*W). Requires c%8==0, s%8==0."""
    grid_s = (s + TR_TILE - 1) // TR_TILE
    grid_c = (c + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]).peek()

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(TR_VEC, elem_ty)

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
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
            g = arith.index_cast(T.i32, in_base + cc * s + ss)
            safe = arith.select(valid, g, arith.constant(0, type=T.i32))
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
            vv = Vec.from_elements(scalars, dtype=elem_ty)
            valid = arith.andi(ss < s, cc < c)
            store_if = scf.IfOp(valid, results_=[], has_else=False)
            with ir.InsertionPoint(store_if.then_block):
                go = arith.index_cast(T.i32, out_base + ss * c + cc)
                buffer_ops.buffer_store(vv, out_rsrc, go)
                scf.YieldOp([])

    @flyc.jit
    def launch_transpose(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(
            grid=(grid_s, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    return launch_transpose


def _ncdhw_to_ndhwc(x, stream):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch."""
    n, c, t, h, w = x.shape
    s = t * h * w
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % 8 == 0 and s % 8 == 0):
        return x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((n, t, h, w, c), device=x.device, dtype=x.dtype)
    exe = compile_transpose_ncdhw_ndhwc(n, c, s)
    _run_compiled(exe, out, x, torch.cuda.current_stream() if stream is None else stream)
    return out


@functools.lru_cache(maxsize=64)
def compile_conv3d_implicit_mfma(
    n,
    c,
    t,
    h,
    w,
    k,
    kt,
    kh,
    kw,
    st,
    sh,
    sw,
    pad_t_lo,
    pad_t_hi,
    pad_h_lo,
    pad_h_hi,
    pad_w_lo,
    pad_w_hi,
    has_bias=False,
    splitk=1,
):
    to = (t + pad_t_lo + pad_t_hi - kt) // st + 1
    ho = (h + pad_h_lo + pad_h_hi - kh) // sh + 1
    wo = (w + pad_w_lo + pad_w_hi - kw) // sw + 1
    hw = ho * wo
    thw = to * hw
    m_total = n * thw
    crs = c * kt * kh * kw

    _forced_n = os.environ.get("CONV3D_TILE_N")
    _wide_n = m_total >= 50000 or (hw >= 16384 and k >= 256)
    if _forced_n:
        TILE_N = int(_forced_n)
    elif _wide_n:
        TILE_N = min((128, 192), key=lambda tn: ((k + tn - 1) // tn, tn))
    else:
        TILE_N = 64
    WARP_N = TILE_N // BLOCK_N_WARPS
    WARP_N_STEPS = WARP_N // MFMA_N
    LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS
    LDS_B_SIZE = STAGES * TILE_N * TILE_K

    a_vec = 1
    for v in (8, 4, 2):
        if c % v == 0:
            a_vec = v
            break
    assert (TILE_M * TILE_K) % (a_vec * BLOCK_THREADS) == 0
    ldg_a_count = TILE_M * TILE_K // (a_vec * BLOCK_THREADS)
    k_tail = crs % TILE_K != 0

    grid_m = (m_total + TILE_M - 1) // TILE_M
    grid_n = (k + TILE_N - 1) // TILE_N
    k_tiles = (crs + TILE_K - 1) // TILE_K

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = (k_tiles + splitk - 1) // splitk
    use_splitk = splitk > 1
    mask_k = k_tail or use_splitk
    bias_in_kernel = has_bias and not use_splitk
    out_ty = fx.Float32 if use_splitk else fx.BFloat16

    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_mfma_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(weight, max_size=True)
        y_rsrc = buffer_ops.create_buffer_resource(y, max_size=True)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=True)
        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        pid_m = fx.block_idx.x
        pid_n = fx.block_idx.y
        m_offset = pid_m * TILE_M
        n_offset = pid_n * TILE_N
        # First k-tile this split-K block reduces (k_off==0 when splitk==1).
        if const_expr(use_splitk):
            k_off = fx.block_idx.z * (tiles_per_split * TILE_K)
        else:
            k_off = 0

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // BLOCK_N_WARPS
        wave_n = wid % BLOCK_N_WARPS

        lane_m = lane % MFMA_M
        lane_n = lane % MFMA_N
        lane_k_a = lane // MFMA_M * MFMA_A_VALUES
        lane_k_b = lane // MFMA_N * MFMA_B_VALUES
        warp_m = wave_m * WARP_M
        warp_n = wave_n * WARP_N

        acc0 = arith.constant_vector(0.0, T.vec(MFMA_C_VALUES, T.f32))
        accs = [acc0 for _ in range_constexpr(WARP_M_STEPS * WARP_N_STEPS)]

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(LDG_VEC, elem_ty)

        zero8 = arith.constant_vector(0.0, Vec8Ty.ir_type)
        zero1 = fx.BFloat16(0.0)

        class AVecTy:
            ir_type = Vec.make_type(a_vec, elem_ty) if a_vec > 1 else elem_ty.ir_type

        zero_a = zero8 if a_vec == 8 else (arith.constant_vector(0.0, AVecTy.ir_type) if a_vec > 1 else zero1)

        def barrier(lgkmcnt=None):
            wait = "s_waitcnt vmcnt(0)"
            if lgkmcnt is not None:
                wait += f" lgkmcnt({lgkmcnt})"
            llvm.InlineAsmOp(None, [], f"{wait}\n\ts_barrier", "", has_side_effects=True)

        def lds_ptr_at(lds_array, byte_offset):
            lds_base = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(byte_offset)
            return buffer_ops.create_llvm_ptr(lds_base, address_space=3)

        def lds_store(lds_array, elem_offset, value, width):
            ptr = lds_ptr_at(lds_array, elem_offset * 2)
            llvm.StoreOp(value, ptr, alignment=min(16, width * 2))

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def in_range(v, hi):
            return (v >= 0) & (v < hi)

        def gather_raw(rsrc, elem_offset, valid, width):
            off_i32 = arith.index_cast(T.i32, elem_offset)
            safe = arith.select(valid, off_i32, arith.constant(0, type=T.i32))
            return buffer_ops.buffer_load(rsrc, safe, vec_width=width, dtype=elem_ty)

        def gather_a(k_base):
            out = []
            for i in range_constexpr(ldg_a_count):
                linear = (tid + i * BLOCK_THREADS) * a_vec
                local_m = linear // TILE_K
                local_k = linear % TILE_K
                row = m_offset + local_m
                row_valid = row < m_total

                n_idx = row // thw
                rem = row % thw
                ot = rem // hw
                rem2 = rem % hw
                oh = rem2 // wo
                ow = rem2 % wo

                k_abs = local_k + k_base
                cc = k_abs % c
                ckk = k_abs // c
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh

                in_t = ot * st + kt_i - pad_t_lo
                in_h = oh * sh + kh_i - pad_h_lo
                in_w = ow * sw + kw_i - pad_w_lo

                valid = row_valid & in_range(in_t, t) & in_range(in_h, h) & in_range(in_w, w)
                if const_expr(mask_k):
                    # zero the A tail / split-K overhang so the (possibly garbage)
                    # B tail multiplies out
                    valid = valid & (k_abs < crs)

                g_off = (((n_idx * t + in_t) * h + in_h) * w + in_w) * c + cc
                raw = gather_raw(x_rsrc, g_off, valid, a_vec)
                out.append((raw, valid, local_m * TILE_K + local_k))
            return out

        def gather_b(k_base):
            out = []
            for i in range_constexpr(LDG_B_COUNT):
                linear = (tid + i * BLOCK_THREADS) * LDG_VEC
                local_n = linear // TILE_K
                local_k = linear % TILE_K
                col = n_offset + local_n
                col_valid = col < k

                g_off = col * crs + (local_k + k_base)
                raw = gather_raw(w_rsrc, g_off, col_valid, LDG_VEC)
                out.append((raw, col_valid, local_n * TILE_K + local_k))
            return out

        def commit_a(stage, gathered):
            base = stage * TILE_M * TILE_K
            for raw, valid, off in gathered:
                lds_store(a_lds, base + off, arith.select(valid, raw, zero_a), a_vec)

        def commit_b(stage, gathered):
            base = stage * TILE_N * TILE_K
            for raw, valid, off in gathered:
                lds_store(b_lds, base + off, arith.select(valid, raw, zero8), LDG_VEC)

        def compute_stage(stage, old_accs):
            a_base = stage * TILE_M * TILE_K
            b_base = stage * TILE_N * TILE_K
            new_accs = [a for a in old_accs]
            for kk in range_constexpr(WARP_K_STEPS):
                k_inner = kk * MFMA_K
                for wm in range_constexpr(WARP_M_STEPS):
                    a_row = warp_m + wm * MFMA_M + lane_m
                    a_col = k_inner + lane_k_a
                    a_frag = lds_load_vec8(a_lds, a_base + a_row * TILE_K + a_col)
                    for wn in range_constexpr(WARP_N_STEPS):
                        b_row = warp_n + wn * MFMA_N + lane_n
                        b_col = k_inner + lane_k_b
                        b_frag = lds_load_vec8(b_lds, b_base + b_row * TILE_K + b_col)
                        idx = wm * WARP_N_STEPS + wn
                        new_accs[idx] = mfma_fn(
                            T.vec(MFMA_C_VALUES, T.f32),
                            [a_frag, b_frag, new_accs[idx], 0, 0, 0],
                        )
            return new_accs

        # Prologue: stage 0 holds this block's first k-tile (k_off).
        commit_a(0, gather_a(k_off))
        commit_b(0, gather_b(k_off))
        barrier(lgkmcnt=0)

        stage = 0
        for kt_idx in range_constexpr(tiles_per_split):
            if const_expr(kt_idx + 1 < tiles_per_split):
                next_a = gather_a(k_off + (kt_idx + 1) * TILE_K)
                next_b = gather_b(k_off + (kt_idx + 1) * TILE_K)
            rocdl.s_setprio(1)
            accs = compute_stage(stage, accs)
            rocdl.s_setprio(0)
            if const_expr(kt_idx + 1 < tiles_per_split):
                n_stage = (stage + 1) % STAGES
                commit_a(n_stage, next_a)
                commit_b(n_stage, next_b)
                barrier(lgkmcnt=0)
                stage = n_stage

        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N
        for wm in range_constexpr(WARP_M_STEPS):
            for wn in range_constexpr(WARP_N_STEPS):
                col = n_offset + warp_n + wn * MFMA_N + c_n
                col_valid = col < k
                acc = Vec(accs[wm * WARP_N_STEPS + wn])
                if const_expr(bias_in_kernel):
                    bias_off = arith.select(col_valid, arith.index_cast(T.i32, col), arith.constant(0, type=T.i32))
                    bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, bias_off, vec_width=1, dtype=fx.Float32))
                for i in range_constexpr(MFMA_C_VALUES):
                    row = m_offset + warp_m + wm * MFMA_M + c_m_vec + i
                    row_valid = row < m_total
                    valid = arith.andi(row_valid, col_valid)
                    store_if = scf.IfOp(valid, results_=[], has_else=False)
                    with ir.InsertionPoint(store_if.then_block):
                        if const_expr(use_splitk):
                            off_bytes = arith.index_cast(T.i32, (row * k + col) * 4)
                            zero_i32 = arith.constant(0, type=T.i32)
                            rocdl.raw_ptr_buffer_atomic_fadd(acc[i], y_rsrc, off_bytes, zero_i32, zero_i32)
                        else:
                            if const_expr(n == 1):
                                out_off = col * thw + row
                            else:
                                out_off = (row // thw) * (k * thw) + col * thw + (row % thw)
                            if const_expr(bias_in_kernel):
                                buffer_ops.buffer_store((acc[i] + bias_val).to(out_ty), y_rsrc, out_off)
                            else:
                                buffer_ops.buffer_store(acc[i].to(out_ty), y_rsrc, out_off)
                        scf.YieldOp([])

    @flyc.jit
    def launch_conv3d_implicit_mfma(
        y: fx.Tensor, x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)
    ):
        conv3d_implicit_mfma_kernel(y, x, w, bias).launch(
            grid=(grid_m, grid_n, splitk),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_conv3d_implicit_mfma


def _choose_splitk(m_total, k, crs, device):
    """Pick a split-K factor for block-starved shapes (small M, large K).

    Returns 1 (no split-K) once the base grid already fills a wave of CUs;
    otherwise aims for ~4 waves of blocks (capped at 4). ``CONV3D_SPLITK`` env
    var forces a fixed value (for benchmarking / autotuning).
    """
    forced = os.environ.get("CONV3D_SPLITK")
    if forced is not None:
        return max(1, int(forced))

    grid_m = (m_total + TILE_M - 1) // TILE_M
    grid_n = (k + TILE_N - 1) // TILE_N
    base = grid_m * grid_n
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if base <= 0:
        return 1
    try:
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        num_cu = 256
    # Only split when the base grid does not even fill one wave of CUs; otherwise
    # the atomic-reduction overhead outweighs the occupancy gain.
    if base >= num_cu:
        return 1
    # Aim for ~4 waves of blocks; cap at 4 (returns diminish past that), and
    # prefer a divisor of k_tiles so the last split has no wasted overhang tiles.
    sk = min(4, max(1, (4 * num_cu) // base), k_tiles)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return max(1, sk)


def _normalize_3(v):
    if isinstance(v, int):
        return (v, v, v)
    assert len(v) == 3, f"expected int or length-3 tuple, got {v!r}"
    return tuple(v)


def conv3d_implicit_mfma(
    x: torch.Tensor,
    w: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    stream=None,
) -> torch.Tensor:
    """``F.conv3d``-style entry (symmetric padding).

    x: (N, C, T, H, W); w: (K, C, kt, kh, kw); returns (N, K, To, Ho, Wo).
    ``stride`` / ``padding`` are int or length-3 (t, h, w) tuples; padding is
    symmetric (low == high) to match ``torch.nn.functional.conv3d``.
    """
    st, sh, sw = _normalize_3(stride)
    pt, ph, pw = _normalize_3(padding)
    return _conv3d_padded(x, w, bias, (st, sh, sw), (pt, pt, ph, ph, pw, pw), stream)


def causal_conv3d_implicit_mfma(
    x: torch.Tensor,
    w: torch.Tensor,
    bias=None,
    stride=1,
    padding=1,
    stream=None,
) -> torch.Tensor:
    """Wan ``CausalConv3d`` semantics: temporal pad (2*pt, 0), spatial same pad.

    x: (N, C, T, H, W); w: (K, C, kt, kh, kw); returns (N, K, To, Ho, Wo).
    """
    st, sh, sw = _normalize_3(stride)
    pt, ph, pw = _normalize_3(padding)
    pads = (2 * pt, 0, ph, ph, pw, pw)
    return _conv3d_padded(x, w, bias, (st, sh, sw), pads, stream)


def _conv3d_padded(x, w, bias, strides, pads, stream):
    n, c, t, h, width = x.shape
    k, wc, kt, kh, kw = w.shape
    assert c == wc, f"in-channel mismatch: x has {c}, w has {wc}"
    assert (
        x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    ), f"only bfloat16 supported, got x={x.dtype}, w={w.dtype}"
    st, sh, sw = strides
    pad_t_lo, pad_t_hi, pad_h_lo, pad_h_hi, pad_w_lo, pad_w_hi = pads
    to = (t + pad_t_lo + pad_t_hi - kt) // st + 1
    ho = (h + pad_h_lo + pad_h_hi - kh) // sh + 1
    wo = (width + pad_w_lo + pad_w_hi - kw) // sw + 1

    bias_f32 = None
    if bias is not None:
        assert bias.numel() == k, f"bias must have {k} elements, got {bias.numel()}"
        bias_f32 = bias.to(torch.float32).contiguous()
        bias_arg = bias_f32
    else:
        bias_arg = torch.empty(1, device=x.device, dtype=torch.float32)

    x_ndhwc = _ncdhw_to_ndhwc(x, stream)
    # w (K, C, kt, kh, kw) -> (K, kt, kh, kw, C) -> (K, kt*kh*kw*C); cached per weight.
    w_kthwc = _prep_weight(w, k, kt, kh, kw, c)

    m_total = n * to * ho * wo
    crs = c * kt * kh * kw
    splitk = _choose_splitk(m_total, k, crs, x.device)
    if splitk > 1:
        y = torch.zeros((m_total, k), device=x.device, dtype=torch.float32)
    else:
        y = torch.empty((n, k, to, ho, wo), device=x.device, dtype=x.dtype)

    exe = compile_conv3d_implicit_mfma(
        n,
        c,
        t,
        h,
        width,
        k,
        kt,
        kh,
        kw,
        st,
        sh,
        sw,
        pad_t_lo,
        pad_t_hi,
        pad_h_lo,
        pad_h_hi,
        pad_w_lo,
        pad_w_hi,
        bias is not None,
        splitk,
    )
    _run_compiled(exe, y, x_ndhwc, w_kthwc, bias_arg, torch.cuda.current_stream() if stream is None else stream)
    if splitk > 1:
        if bias is not None:
            y = y + bias_f32.view(1, k)
        return y.to(x.dtype).view(n, to, ho, wo, k).permute(0, 4, 1, 2, 3)
    return y
