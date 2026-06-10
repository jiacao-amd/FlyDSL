import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.tensor_shim import GTensor, STensor, _run_compiled


TILE_M = 96
TILE_N = 64
TILE_K = 32
STAGES = 3

BLOCK_M_WARPS = 2
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


def swizzle_xor16(row, col_in_bytes):
    return col_in_bytes ^ ((row % (TILE_K * 2 // 16)) * 16)


@functools.lru_cache(maxsize=32)
def compile_conv2d_implicit_mfma(n, c, h, width, k, r, s):
    p = h - r + 1
    q = width - s + 1
    npq = n * p * q
    crs = c * r * s
    k_tiles = crs // TILE_K
    grid_m = npq // TILE_M

    assert c % LDG_VEC == 0
    assert crs % TILE_K == 0
    assert npq % TILE_M == 0
    assert k == TILE_N

    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name=f"conv_smem_{n}_{c}_{h}_{width}_{k}")
    a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = a_offset + STAGES * TILE_M * TILE_K * 2
    b_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = b_offset + STAGES * TILE_N * TILE_K * 2
    c_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = c_offset + TILE_M * TILE_N * 2

    def make_lds():
        base = allocator.get_base()
        a = STensor(
            SmemPtr(base, a_offset, T.f16, shape=(STAGES * TILE_M * TILE_K,)),
            T.f16,
            shape=(STAGES, TILE_M, TILE_K),
        )
        b = STensor(
            SmemPtr(base, b_offset, T.f16, shape=(STAGES * TILE_N * TILE_K,)),
            T.f16,
            shape=(STAGES, TILE_N, TILE_K),
        )
        c_tile = STensor(
            SmemPtr(base, c_offset, T.f16, shape=(TILE_M * TILE_N,)),
            T.f16,
            shape=(TILE_M, TILE_N),
        )
        return a, b, c_tile

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv2d_implicit_mfma_kernel(y: fx.Tensor, x: fx.Tensor, w: fx.Tensor):
        # x is NHWC, w is KRSC, y is (N*P*Q, K).
        x_g = GTensor(x, dtype=T.f16, shape=(n, h, width, c))
        w_g = GTensor(w, dtype=T.f16, shape=(k, crs))
        y_g = GTensor(y, dtype=T.f16, shape=(npq, k))
        a_lds, b_lds, c_lds = make_lds()

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

        def barrier(vmcnt=0):
            llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier", "", has_side_effects=True)

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

                g_off = x_g.linear_offset((n_idx, pp + rr, qq + ss, cc)) * 2
                g_off = arith.index_cast(T.i32, g_off)

                if const_expr(i == 0):
                    lds_off = a_lds.linear_offset((fx.Index(stage), fx.Index(0), fx.Index(0))) * 2
                    lds_base = memref.extract_aligned_pointer_as_index(a_lds.memptr) + lds_off
                    lds_ptr_base = buffer_ops.create_llvm_ptr(arith.index_cast(T.i64, lds_base), address_space=3)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_lds_off)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES)

                buffer_load_to_lds(x_g.rsrc, lds_ptr, g_off)

        def load_b_to_lds(k_base, stage):
            warp_lds_off = dma_warp_offset()
            for i in range_constexpr(LDG_B_COUNT):
                linear = tid * LDG_VEC + i * BLOCK_VECS
                local_n = linear // TILE_K
                local_k = linear % TILE_K
                col_bytes = swizzle_xor16(local_n, local_k * 2)

                g_off = w_g.linear_offset((fx.Index(local_n), fx.Index(k_base) + fx.Index(col_bytes // 2))) * 2
                g_off = arith.index_cast(T.i32, g_off)

                if const_expr(i == 0):
                    lds_off = b_lds.linear_offset((fx.Index(stage), fx.Index(0), fx.Index(0))) * 2
                    lds_base = memref.extract_aligned_pointer_as_index(b_lds.memptr) + lds_off
                    lds_ptr_base = buffer_ops.create_llvm_ptr(arith.index_cast(T.i64, lds_base), address_space=3)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_lds_off)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES)

                buffer_load_to_lds(w_g.rsrc, lds_ptr, g_off)

        def compute_stage(stage, old_accs):
            new_accs = [a for a in old_accs]
            sidx = fx.Index(stage)
            for kk in range_constexpr(WARP_K_STEPS):
                k_inner = kk * MFMA_K
                for wm in range_constexpr(WARP_M_STEPS):
                    a_row = warp_m + wm * MFMA_M + lane_m
                    a_col = swizzle_xor16(a_row, (k_inner + lane_k_a) * 2) // 2
                    a_frag = a_lds.vec_load((sidx, fx.Index(a_row), fx.Index(a_col)), MFMA_A_VALUES)
                    for wn in range_constexpr(WARP_N_STEPS):
                        b_row = warp_n + wn * MFMA_N + lane_n
                        b_col = swizzle_xor16(b_row, (k_inner + lane_k_b) * 2) // 2
                        b_frag = b_lds.vec_load((sidx, fx.Index(b_row), fx.Index(b_col)), MFMA_B_VALUES)
                        idx = wm * WARP_N_STEPS + wn
                        new_accs[idx] = rocdl.mfma_f32_16x16x32_f16(
                            T.vec(MFMA_C_VALUES, T.f32),
                            [a_frag, b_frag, new_accs[idx], 0, 0, 0],
                        )
            return new_accs

        def hot_loop_scheduler():
            for _ in range_constexpr(LDG_B_COUNT + LDG_A_COUNT):
                rocdl.sched_vmem(1)
            for _ in range_constexpr(WARP_K_STEPS):
                for _ in range_constexpr(WARP_N_STEPS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(WARP_M_STEPS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(WARP_M_STEPS):
                    rocdl.sched_mfma(WARP_N_STEPS)
            rocdl.sched_barrier(0)

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
            hot_loop_scheduler()
            stage = (stage + 1) % STAGES

        for tail in range_constexpr(STAGES - 1):
            barrier()
            rocdl.s_setprio(1)
            accs = compute_stage(stage, accs)
            rocdl.s_setprio(0)
            hot_loop_scheduler()
            stage = (stage + 1) % STAGES

        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N
        for wm in range_constexpr(WARP_M_STEPS):
            row = warp_m + wm * MFMA_M + c_m_vec
            for wn in range_constexpr(WARP_N_STEPS):
                col = warp_n + wn * MFMA_N + c_n
                acc = accs[wm * WARP_N_STEPS + wn]
                for i in range_constexpr(MFMA_C_VALUES):
                    c_lds[fx.Index(row + i), fx.Index(col)] = vector.extract(
                        acc, static_position=[i], dynamic_position=[]
                    ).truncf(T.f16)

        barrier()

        for i in range_constexpr(LDG_C_COUNT):
            linear = tid * LDG_VEC + i * BLOCK_VECS
            local_m = linear // TILE_N
            local_n = linear % TILE_N
            vals = c_lds.vec_load((fx.Index(local_m), fx.Index(local_n)), LDG_VEC)
            y_g.vec_store((m_offset + local_m, local_n), vals, LDG_VEC)

    @flyc.jit
    def launch_conv2d_implicit_mfma(y: fx.Tensor, x: fx.Tensor, w: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        conv2d_implicit_mfma_kernel(y, x, w).launch(
            grid=(grid_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_conv2d_implicit_mfma


def conv2d_implicit_mfma_(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor, stream=None):
    n, h, width, c = x.shape
    k, r, s, wc = w.shape
    assert c == wc
    exe = compile_conv2d_implicit_mfma(n, c, h, width, k, r, s)
    _run_compiled(exe, y, x, w, torch.cuda.current_stream() if stream is None else stream)
