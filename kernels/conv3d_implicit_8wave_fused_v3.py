import functools
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

# 3D 8-wave implicit-GEMM conv3d, v3 experiment.
#
# Based on conv3d_implicit_8wave_fused.py, but keeps this as a separate file so
# the old 8-wave kernel remains untouched.  This variant adds more explicit
# scheduling around the next-tile VMEM gathers and removes avoidable zero-fill in
# the no-split-K wrapper path.
#
# conv3d (bf16, gfx950) with opus-style ds_read<->MFMA phase fusion (read/compute
# interleaving).
# Supports stride + padding + arbitrary kernel. k == TILE_N (128), TILE_K = 32,
# crs = c*kt*kh*kw must be a multiple of TILE_K, c a multiple of 8.

TILE_M = 128
TILE_N = 128
TILE_K = 32
STAGES = 2

WAVE_M = 2
WAVE_N = 4
WARP_SIZE = 64
BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE  # 512

MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
QM_STEPS = HALF_M // WAVE_M // MFMA_M  # 2
QN_STEPS = HALF_N // WAVE_N // MFMA_N  # 1
N_SUB = QM_STEPS * QN_STEPS

LDG_VEC = 8
BLOCK_VECS = LDG_VEC * BLOCK_THREADS
LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS
LDS_A_SIZE = STAGES * TILE_M * TILE_K
LDS_B_SIZE = STAGES * TILE_N * TILE_K


def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


@functools.lru_cache(maxsize=64)
def compile_conv3d_implicit_8wave_fused_v3(
    n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias=False, splitk=1
):
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (w + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    k_tiles = crs // TILE_K

    assert c % LDG_VEC == 0
    assert crs % TILE_K == 0
    assert LDG_A_COUNT == 1 and LDG_B_COUNT == 1

    n_tail = k % TILE_N != 0
    grid_n = (k + TILE_N - 1) // TILE_N

    if (k % TILE_N != 0) or (npq % TILE_M != 0):
        splitk = 1
    splitk = max(1, min(splitk, k_tiles))
    while k_tiles % splitk != 0:
        splitk -= 1
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    grid_m = (npq + TILE_M - 1) // TILE_M
    elem_ty = fx.BFloat16
    out_ty = fx.Float32 if use_splitk else fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_8wave_fused_v3_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(weight, max_size=True)
        y_rsrc = buffer_ops.create_buffer_resource(y, max_size=True)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=True)

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        pid = fx.block_idx.x
        m_offset = pid * TILE_M
        n_offset = fx.block_idx.y * TILE_N
        if const_expr(use_splitk):
            k_off = fx.block_idx.z * (tiles_per_split * TILE_K)
        else:
            k_off = 0

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

        acc0 = arith.constant_vector(0.0, T.vec(MFMA_C_VALUES, T.f32))
        acc00 = [acc0 for _ in range_constexpr(N_SUB)]
        acc01 = [acc0 for _ in range_constexpr(N_SUB)]
        acc10 = [acc0 for _ in range_constexpr(N_SUB)]
        acc11 = [acc0 for _ in range_constexpr(N_SUB)]

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(8, elem_ty)

        zero8 = arith.constant_vector(0.0, Vec8Ty.ir_type)

        def barrier(vmcnt=0, lgkmcnt=None):
            waits = []
            if vmcnt is not None:
                waits.append(f"vmcnt({vmcnt})")
            if lgkmcnt is not None:
                waits.append(f"lgkmcnt({lgkmcnt})")
            pre = ("s_waitcnt " + " ".join(waits) + "\n\t") if waits else ""
            llvm.InlineAsmOp(None, [], f"{pre}s_barrier", "", has_side_effects=True)

        def lds_ptr_at(lds_array, byte_offset):
            lds_base = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(byte_offset)
            return buffer_ops.create_llvm_ptr(lds_base, address_space=3)

        def lds_store_vec8(lds_array, elem_offset, value):
            llvm.StoreOp(value, lds_ptr_at(lds_array, elem_offset * 2), alignment=16)

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def a_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def in_range(v, hi):
            return (v >= 0) & (v < fx.Index(hi))

        # ---- 3D im2col gather (global -> registers) ----
        def gather_a(k_base):
            linear = tid * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + local_m
            row_valid = row < fx.Index(npq)
            n_idx = row // dhw
            rem = row % dhw
            ot = rem // hw_o
            rem2 = rem % hw_o
            oh = rem2 // wo
            ow = rem2 % wo
            k_abs = fx.Index(k_base) + fx.Index(local_k)
            cc = k_abs % c
            ckk = k_abs // c
            kw_i = ckk % kw
            ckk2 = ckk // kw
            kh_i = ckk2 % kh
            kt_i = ckk2 // kh
            in_t = ot * st + kt_i - pt
            in_h = oh * sh + kh_i - ph
            in_w = ow * sw + kw_i - pw
            valid = row_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
            g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
            g_off_i = arith.index_cast(T.i32, g_off)
            safe = arith.select(valid, g_off_i, arith.constant(0, type=T.i32))
            raw = buffer_ops.buffer_load(x_rsrc, safe, vec_width=8, dtype=elem_ty)
            return (raw, valid, local_m * TILE_K + local_k)

        def gather_b(k_base):
            linear = tid * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(local_n)
            g_off = arith.index_cast(T.i32, col * crs + (fx.Index(k_base) + fx.Index(local_k)))
            if const_expr(n_tail):
                col_valid = col < fx.Index(k)
                safe = arith.select(col_valid, g_off, arith.constant(0, type=T.i32))
                raw = buffer_ops.buffer_load(w_rsrc, safe, vec_width=8, dtype=elem_ty)
                return (raw, col_valid, local_n * TILE_K + local_k)
            raw = buffer_ops.buffer_load(w_rsrc, g_off, vec_width=8, dtype=elem_ty)
            return (raw, None, local_n * TILE_K + local_k)

        def commit_a(stage, vo):
            raw, valid, off = vo
            val = arith.select(valid, raw, zero8)  # mask consumed here (hidden behind MFMAs)
            lds_store_vec8(a_lds, fx.Index(stage) * TILE_M * TILE_K + off, val)

        def commit_b(stage, vo):
            raw, valid, off = vo
            val = raw if const_expr(valid is None) else arith.select(valid, raw, zero8)
            lds_store_vec8(b_lds, fx.Index(stage) * TILE_N * TILE_K + off, val)

        # ---- single-vec ds_read (LDS -> register) ----
        def read_a_vec(stage, m_half, wm):
            a_row = m_half * HALF_M + wave_m * (HALF_M // WAVE_M) + wm * MFMA_M + lane_m
            return lds_load_vec8(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(lane_k_a)))

        def read_b_vec(stage, n_half, wn):
            b_row = n_half * HALF_N + wave_n * (HALF_N // WAVE_N) + wn * MFMA_N + lane_n
            return lds_load_vec8(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(lane_k_b)))

        def a_thunks(stage, m_half):
            return [(lambda st=stage, mh=m_half, wm=wm: read_a_vec(st, mh, wm)) for wm in range_constexpr(QM_STEPS)]

        def b_thunks(stage, n_half):
            return [(lambda st=stage, nh=n_half, wn=wn: read_b_vec(st, nh, wn)) for wn in range_constexpr(QN_STEPS)]

        # ---- fused phase: issue `thunks` (next-op ds_read / commit) BETWEEN the
        #      current quadrant's MFMAs -> 读算读算 ----
        def mma_phase(a_frags, b_frags, acc, thunks, kind="dsrd"):
            out = [a for a in acc]
            nt = len(thunks)
            n_jobs = QM_STEPS * QN_STEPS
            surplus = nt - n_jobs if nt > n_jobs else 0
            res = [None] * nt
            sched_thunk = rocdl.sched_dsrd if const_expr(kind == "dsrd") else rocdl.sched_dswr
            for i in range_constexpr(surplus):
                res[i] = thunks[i]()
            if const_expr(surplus > 0):
                sched_thunk(surplus)
            ti = surplus
            for wm in range_constexpr(QM_STEPS):
                for wn in range_constexpr(QN_STEPS):
                    if const_expr(ti < nt):
                        res[ti] = thunks[ti]()
                        ti += 1
                        sched_thunk(1)  # force this ds_read/write before the mfma
                    idx = wm * QN_STEPS + wn
                    out[idx] = mfma_fn(
                        T.vec(MFMA_C_VALUES, T.f32),
                        [a_frags[wm], b_frags[wn], out[idx], 0, 0, 0],
                    )
                    rocdl.sched_mfma(1)  # ... then exactly one mfma (读算读算)
            return out, res

        # ---- prologue ----
        commit_a(0, gather_a(k_off))
        commit_b(0, gather_b(k_off))
        barrier(lgkmcnt=0)

        stage = 0
        for kt_idx in range_constexpr(tiles_per_split):
            if const_expr(kt_idx + 1 < tiles_per_split):
                na = gather_a(k_off + (kt_idx + 1) * TILE_K)
                nb = gather_b(k_off + (kt_idx + 1) * TILE_K)
                rocdl.sched_vmem(2)
            n_stage = (stage + 1) % STAGES

            a0 = [read_a_vec(stage, 0, wm) for wm in range_constexpr(QM_STEPS)]
            b0 = [read_b_vec(stage, 0, wn) for wn in range_constexpr(QN_STEPS)]
            rocdl.sched_dsrd(QM_STEPS + QN_STEPS)  # account for the a0/b0 reads

            rocdl.s_setprio(1)
            acc00, b1 = mma_phase(a0, b0, acc00, b_thunks(stage, 1))
            acc01, a1 = mma_phase(a0, b1, acc01, a_thunks(stage, 1))
            if const_expr(kt_idx + 1 < tiles_per_split):
                acc10, _ = mma_phase(
                    a1, b0, acc10,
                    [(lambda ns=n_stage, v=na: commit_a(ns, v)), (lambda ns=n_stage, v=nb: commit_b(ns, v))],
                    kind="dswr",
                )
            else:
                acc10, _ = mma_phase(a1, b0, acc10, [])
            acc11, _ = mma_phase(a1, b1, acc11, [])
            rocdl.s_setprio(0)

            if const_expr(kt_idx + 1 < tiles_per_split):
                barrier(lgkmcnt=0)
                stage = n_stage

        # ---- epilogue: invalid (row/col tail) lanes are GUARDED out with scf.if
        # (OOB-redirect is unreliable -- stores fault past the buffer, atomics
        # serialize). Clean shapes (no tail) take the unguarded fast path.
        from flydsl._mlir.dialects import scf
        from flydsl._mlir import ir

        _row_chk = npq % TILE_M != 0
        _need_chk = _row_chk or n_tail

        def _valid_raw(row, col):
            if const_expr(_row_chk and n_tail):
                return arith.andi(row < fx.Index(npq), col < fx.Index(k))
            if const_expr(_row_chk):
                rc = row < fx.Index(npq)
                return arith.andi(rc, rc)
            cc = col < fx.Index(k)
            return arith.andi(cc, cc)

        def store_quad(acc, m_half, n_half):
            for wm in range_constexpr(QM_STEPS):
                row_base = m_offset + m_half * HALF_M + wave_m * (HALF_M // WAVE_M) + wm * MFMA_M + c_m_vec
                for wn in range_constexpr(QN_STEPS):
                    col = n_offset + fx.Index(n_half * HALF_N + wave_n * (HALF_N // WAVE_N) + wn * MFMA_N + c_n)
                    a = Vec(acc[wm * QN_STEPS + wn])
                    if const_expr(has_bias and not use_splitk):
                        col_i = arith.index_cast(T.i32, col)
                        if const_expr(n_tail):
                            col_i = arith.select(col < fx.Index(k), col_i, arith.constant(0, type=T.i32))
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))
                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Index(row_base + i)
                        off = row * k + col

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = arith.index_cast(T.i32, off * 4)
                                z0 = arith.constant(0, type=T.i32)
                                rocdl.raw_ptr_buffer_atomic_fadd(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
                                buffer_ops.buffer_store(cval, y_rsrc, off)

                        if const_expr(_need_chk):
                            store_if = scf.IfOp(_valid_raw(row, col), results_=[], has_else=False)
                            with ir.InsertionPoint(store_if.then_block):
                                _emit()
                                scf.YieldOp([])
                        else:
                            _emit()

        store_quad(acc00, 0, 0)
        store_quad(acc01, 0, 1)
        store_quad(acc10, 1, 0)
        store_quad(acc11, 1, 1)

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_8wave_fused_v3_kernel(y, x, weight, bias).launch(
            grid=(grid_m, grid_n, splitk), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch


def _choose_splitk(npq, crs, k, device):
    """Auto split-K dispatch: split only when the base grid is block-starved
    (small M / large crs); aim for ~4 waves of blocks, prefer a divisor of
    k_tiles. CONV3D_8W_SPLITK overrides."""
    forced = os.environ.get("CONV3D_8W_SPLITK")
    if forced is not None:
        return max(1, int(forced))
    grid_m = (npq + TILE_M - 1) // TILE_M
    grid_n = (k + TILE_N - 1) // TILE_N
    base = grid_m * grid_n
    k_tiles = crs // TILE_K
    # Only split when the problem is non-trivial (atomic + f32-convert overhead
    # would otherwise dominate), the reduction is deep enough to be worth
    # splitting, and the base grid is clearly block-starved.
    if npq < 4096 or k_tiles < 16:
        return 1
    if k % TILE_N != 0 or npq % TILE_M != 0:  # atomic OOB-redirect breaks on tails
        return 1
    try:
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        num_cu = 256
    if base >= (3 * num_cu) // 4:  # base grid already (nearly) fills the machine
        return 1
    sk = min(4, max(1, num_cu // base), k_tiles)  # aim to roughly fill the CUs
    while sk > 1 and k_tiles % sk != 0:  # prefer a divisor (no overhang)
        sk -= 1
    return sk


def conv3d_implicit_8wave_fused_v3(x, weight, bias=None, stride=1, padding=0, splitk=None, stream=None):
    # x: (N,C,D,H,W) bf16, weight: (K,C,T,R,S) bf16. splitk=None -> auto-dispatch.
    n, c, d, h, w = x.shape
    k, wc, kt, kh, kw = weight.shape
    assert c == wc
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    st, sh, sw = (stride, stride, stride) if isinstance(stride, int) else stride
    pt, ph, pw = (padding, padding, padding) if isinstance(padding, int) else padding
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (w + 2 * pw - kw) // sw + 1
    npq = n * do * ho * wo
    crs = c * kt * kh * kw

    sk = _choose_splitk(npq, crs, k, x.device) if splitk is None else max(1, splitk)
    k_tiles = crs // TILE_K
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    use_splitk = sk > 1

    # Fast fused NCDHW->NDHWC transpose + cached weight permute (reuse prod's
    # helpers) instead of torch permute+contiguous per call.
    from kernels.conv3d_implicit_mfma import _ncdhw_to_ndhwc, _prep_weight

    x_ndhwc = _ncdhw_to_ndhwc(x, stream)
    w_packed = _prep_weight(weight, k, kt, kh, kw, c)
    if use_splitk:
        y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
    else:
        y = torch.empty((npq, k), device=x.device, dtype=torch.bfloat16)
    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else torch.empty(1, device=x.device, dtype=torch.float32)
    exe = compile_conv3d_implicit_8wave_fused_v3(n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias, sk)
    _run_compiled(exe, y, x_ndhwc, w_packed, bias_arg, torch.cuda.current_stream() if stream is None else stream)
    if use_splitk:
        if has_bias:
            y = y + bias_arg.view(1, k)
        y = y.to(torch.bfloat16)
    # (N*Do*Ho*Wo, K) -> (N, K, Do, Ho, Wo)
    return y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3)
