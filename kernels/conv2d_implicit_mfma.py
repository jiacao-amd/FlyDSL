import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

TILE_M = 256
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
DMA_BYTES = 16
BLOCK_VECS = LDG_VEC * BLOCK_THREADS
LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS
LDG_C_COUNT = TILE_M * TILE_N // BLOCK_VECS
LDS_A_SIZE = STAGES * TILE_M * TILE_K
LDS_B_SIZE = STAGES * TILE_N * TILE_K


def swizzle_xor16(row, col_in_bytes):
    return col_in_bytes ^ ((row % (TILE_K * 2 // 16)) * 16)


def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


@functools.lru_cache(maxsize=32)
def compile_conv2d_implicit_mfma(n, c, h, width, k, r, s, has_bias=False):
    p = h - r + 1
    q = width - s + 1
    npq = n * p * q
    crs = c * r * s
    k_tiles = crs // TILE_K
    grid_m = (npq + TILE_M - 1) // TILE_M

    assert c % LDG_VEC == 0
    assert crs % TILE_K == 0
    assert k == TILE_N

    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv2d_implicit_mfma_kernel(y: fx.Tensor, x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor):
        # x is NHWC, w is KRSC, y is (N*P*Q, K), bias is (K,) f32 (unused if has_bias=False).
        x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(w, max_size=True)
        y_rsrc = buffer_ops.create_buffer_resource(y, max_size=True)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=True)
        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()
        c_lds = a_lds

        tid = fx.thread_idx.x
        pid = fx.block_idx.x
        m_offset = pid * TILE_M

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
            ir_type = Vec.make_type(8, elem_ty)

        def barrier(vmcnt=0, lgkmcnt=None):
            wait = f"s_waitcnt vmcnt({vmcnt})"
            if lgkmcnt is not None:
                wait += f" lgkmcnt({lgkmcnt})"
            llvm.InlineAsmOp(None, [], f"{wait}\n\ts_barrier", "", has_side_effects=True)

        def dma_warp_offset():
            return rocdl.readfirstlane(
                T.i64,
                arith.index_cast(T.i64, fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True)),
            )

        def buffer_load_to_lds(rsrc, lds_ptr, global_offset):
            llvm.InlineAsmOp(
                None,
                [lds_ptr, global_offset, rsrc],
                "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds",
                "s,v,s",
                has_side_effects=True,
            )

        def lds_ptr_at(lds_array, byte_offset):
            lds_base = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(byte_offset)
            return buffer_ops.create_llvm_ptr(lds_base, address_space=3)

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def a_lds_offset(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_offset(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def c_lds_offset(row, col):
            return row * TILE_N + col

        def x_offset(n_idx, h_idx, w_idx, c_idx):
            return ((n_idx * h + h_idx) * width + w_idx) * c + c_idx

        def w_offset(k_idx, crs_idx):
            return k_idx * crs + crs_idx

        def y_offset(row, col):
            return row * k + col

        def load_a_to_lds(k_base, stage):
            warp_lds_off = dma_warp_offset()
            for i in range_constexpr(LDG_A_COUNT):
                linear = tid * LDG_VEC + i * BLOCK_VECS
                local_m = linear // TILE_K
                local_k = linear % TILE_K
                row = m_offset + local_m
                n_idx = row // (p * q)
                pq = row % (p * q)
                pp = pq // q
                qq = pq % q
                col_bytes = swizzle_xor16(local_m, local_k * 2)
                k_abs = fx.Index(k_base) + fx.Index(col_bytes // 2)
                rs = k_abs // c
                rr = rs // s
                ss = rs % s
                cc = k_abs % c

                g_off = x_offset(n_idx, pp + rr, qq + ss, cc) * 2
                g_off = arith.index_cast(T.i32, g_off)

                if const_expr(i == 0):
                    lds_off = a_lds_offset(stage, fx.Index(0), fx.Index(0)) * 2
                    lds_ptr_base = lds_ptr_at(a_lds, lds_off)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_lds_off)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES)

                if const_expr(npq % TILE_M == 0):
                    buffer_load_to_lds(x_rsrc, lds_ptr, g_off)
                elif row < fx.Index(npq):
                    buffer_load_to_lds(x_rsrc, lds_ptr, g_off)

        def load_b_to_lds(k_base, stage):
            warp_lds_off = dma_warp_offset()
            for i in range_constexpr(LDG_B_COUNT):
                linear = tid * LDG_VEC + i * BLOCK_VECS
                local_n = linear // TILE_K
                local_k = linear % TILE_K
                col_bytes = swizzle_xor16(local_n, local_k * 2)

                g_off = w_offset(fx.Index(local_n), fx.Index(k_base) + fx.Index(col_bytes // 2)) * 2
                g_off = arith.index_cast(T.i32, g_off)

                if const_expr(i == 0):
                    lds_off = b_lds_offset(stage, fx.Index(0), fx.Index(0)) * 2
                    lds_ptr_base = lds_ptr_at(b_lds, lds_off)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_lds_off)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES)

                buffer_load_to_lds(w_rsrc, lds_ptr, g_off)

        def compute_stage(stage, old_accs):
            new_accs = [a for a in old_accs]
            for kk in range_constexpr(WARP_K_STEPS):
                k_inner = kk * MFMA_K
                for wm in range_constexpr(WARP_M_STEPS):
                    a_row = warp_m + wm * MFMA_M + lane_m
                    a_col = swizzle_xor16(a_row, (k_inner + lane_k_a) * 2) // 2
                    a_frag = lds_load_vec8(a_lds, a_lds_offset(stage, fx.Index(a_row), fx.Index(a_col)))
                    for wn in range_constexpr(WARP_N_STEPS):
                        b_row = warp_n + wn * MFMA_N + lane_n
                        b_col = swizzle_xor16(b_row, (k_inner + lane_k_b) * 2) // 2
                        b_frag = lds_load_vec8(b_lds, b_lds_offset(stage, fx.Index(b_row), fx.Index(b_col)))
                        idx = wm * WARP_N_STEPS + wn
                        new_accs[idx] = mfma_fn(
                            T.vec(MFMA_C_VALUES, T.f32),
                            [a_frag, b_frag, new_accs[idx], 0, 0, 0],
                        )
            return new_accs

        for preload in range_constexpr(STAGES - 1):
            preload_k = preload * TILE_K
            load_b_to_lds(preload_k, preload)
            load_a_to_lds(preload_k, preload)
        barrier()

        stage = 0
        for kt in range_constexpr(k_tiles - (STAGES - 1)):
            write_stage = (stage + STAGES - 1) % STAGES
            next_k = (kt + STAGES - 1) * TILE_K
            barrier()
            load_b_to_lds(next_k, write_stage)
            load_a_to_lds(next_k, write_stage)
            rocdl.s_setprio(1)
            accs = compute_stage(stage, accs)
            rocdl.s_setprio(0)
            stage = (stage + 1) % STAGES

        for tail in range_constexpr(STAGES - 1):
            barrier()
            rocdl.s_setprio(1)
            accs = compute_stage(stage, accs)
            rocdl.s_setprio(0)
            stage = (stage + 1) % STAGES

        barrier()

        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N
        for wm in range_constexpr(WARP_M_STEPS):
            row = warp_m + wm * MFMA_M + c_m_vec
            for wn in range_constexpr(WARP_N_STEPS):
                col = warp_n + wn * MFMA_N + c_n
                acc = Vec(accs[wm * WARP_N_STEPS + wn])
                if const_expr(has_bias):
                    bias_val = fx.Float32(
                        buffer_ops.buffer_load(bias_rsrc, fx.Int32(col), vec_width=1, dtype=fx.Float32)
                    )
                for i in range_constexpr(MFMA_C_VALUES):
                    if const_expr(has_bias):
                        c_val = (acc[i] + bias_val).to(elem_ty)
                    else:
                        c_val = acc[i].to(elem_ty)
                    fx.ptr_store(c_val, c_lds.ptr + fx.Int32(c_lds_offset(fx.Index(row + i), fx.Index(col))))

        # Drain the C stores (lgkmcnt) before reading C back: the store uses
        # c_lds.ptr while the load recasts to u8, so the compiler can't see the
        # aliasing and would otherwise omit the wait.
        barrier(lgkmcnt=0)

        for i in range_constexpr(LDG_C_COUNT):
            linear = tid * LDG_VEC + i * BLOCK_VECS
            local_m = linear // TILE_N
            local_n = linear % TILE_N
            global_m = m_offset + local_m
            if const_expr(npq % TILE_M == 0):
                vals = lds_load_vec8(c_lds, c_lds_offset(fx.Index(local_m), fx.Index(local_n)))
                buffer_ops.buffer_store(vals, y_rsrc, y_offset(global_m, local_n))
            elif global_m < fx.Index(npq):
                buffer_ops.buffer_store(
                    lds_load_vec8(c_lds, c_lds_offset(fx.Index(local_m), fx.Index(local_n))),
                    y_rsrc,
                    y_offset(global_m, local_n),
                )

    @flyc.jit
    def launch_conv2d_implicit_mfma(
        y: fx.Tensor, x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)
    ):
        conv2d_implicit_mfma_kernel(
            y,
            x,
            w,
            bias,
        ).launch(
            grid=(grid_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_conv2d_implicit_mfma


def conv2d_implicit_mfma_(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias=None, stream=None):
    # Low-level entry: x is NHWC, w is KRSC, y is (N*P*Q, K) pre-allocated.
    # bias is an optional float32 (K,) tensor.
    n, h, width, c = x.shape
    k, r, s, wc = w.shape
    assert c == wc
    assert x.dtype == torch.bfloat16, f"only bfloat16 supported, got {x.dtype}"
    has_bias = bias is not None
    if has_bias:
        assert bias.dtype == torch.float32, f"bias must be float32, got {bias.dtype}"
        bias_arg = bias
    else:
        bias_arg = torch.empty(1, device=x.device, dtype=torch.float32)  # unused placeholder
    exe = compile_conv2d_implicit_mfma(n, c, h, width, k, r, s, has_bias)
    _run_compiled(exe, y, x, w, bias_arg, torch.cuda.current_stream() if stream is None else stream)


def conv2d_implicit_mfma(x: torch.Tensor, w: torch.Tensor, bias=None, stream=None) -> torch.Tensor:
    n, c, h, width = x.shape
    k, wc, r, s = w.shape
    assert c == wc, f"in-channel mismatch: x has {c}, w has {wc}"
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, (
        f"only bfloat16 supported, got x={x.dtype}, w={w.dtype}"
    )
    p = h - r + 1
    q = width - s + 1

    bias_f32 = None
    if bias is not None:
        assert bias.numel() == k, f"bias must have {k} elements, got {bias.numel()}"
        bias_f32 = bias.to(torch.float32).contiguous()

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    w_krsc = w.permute(0, 2, 3, 1).contiguous()
    y = torch.empty((n * p * q, k), device=x.device, dtype=x.dtype)
    conv2d_implicit_mfma_(y, x_nhwc, w_krsc, bias_f32, stream)
    # (N*P*Q, K) -> (N, K, P, Q) as a channels-last view (zero-copy).
    return y.view(n, p, q, k).permute(0, 3, 1, 2)
