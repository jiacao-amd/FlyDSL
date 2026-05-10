# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Raw-B 8-wave ping-pong FP8 GEMM with row-wise fp32 scaling.

- FP8 E4M3 inputs, row-wise fp32 scales, bf16/fp16 output.
- A and B are raw row-major / k-major tensors: A[M, K], B[N, K].
- Fixed 256x256x128 tile.
- Two LDS buffers are used as pong/ping.
- No half-wave producer/consumer specialization in the hot loop.

All waves execute the same instruction order. wave_m/wave_n are used only for
address partitioning. The hot loop interleaves LDS reads, global-to-LDS copies,
MFMA, and barriers in a fixed branch-free schedule.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.mfma_preshuffle_pipeline import swizzle_xor16


_NUM_WAVES = 8
_WAVE_SIZE = 64
_TOTAL_THREADS = _NUM_WAVES * _WAVE_SIZE

_TILE_M = 256
_TILE_N = 256
_TILE_K = 128
_HALF_MN = 128
_HALF_TILE_BYTES = _HALF_MN * _TILE_K
_STAGE_BYTES = 4 * _HALF_TILE_BYTES

_A0_OFF = 0
_B0_OFF = _HALF_TILE_BYTES
_A1_OFF = 2 * _HALF_TILE_BYTES
_B1_OFF = 3 * _HALF_TILE_BYTES


def compile_preshuffle_gemm_fp8_8wave(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    out_dtype: str = "bf16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = True,
    dsrd_preload: int = -1,
    dvmem_preload: int = -1,
    aonly_wave_pingpong: bool = False,
):
    """Compile the raw-B 8-wave FP8 ping-pong GEMM.

    Signature:
      launch(c, a_fp8, b_raw_fp8, scale_a_f32, scale_b_f32,
             unused_bias, M, N, stream)
    """

    del M, N, dsrd_preload, dvmem_preload, aonly_wave_pingpong

    if int(tile_m) != _TILE_M or int(tile_n) != _TILE_N or int(tile_k) != _TILE_K:
        raise ValueError("8-wave FP8 ping-pong only supports 256x256x128 tiles")
    if int(lds_stage) != 2:
        raise ValueError("8-wave FP8 ping-pong only supports lds_stage=2")
    if bool(use_cshuffle_epilog):
        raise ValueError("8-wave FP8 ping-pong does not support CShuffle")
    if not bool(use_async_copy):
        raise ValueError("8-wave FP8 ping-pong requires async copy")
    if int(K) < 2 * _TILE_K or int(K) % (2 * _TILE_K) != 0:
        raise ValueError("K must be a multiple of 256 and at least 256")
    if out_dtype not in ("bf16", "fp16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'fp16', got {out_dtype!r}")

    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="fp8_8wave_smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="fp8_8wave_smem1")

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _STAGE_BYTES
    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _STAGE_BYTES

    Vec = fx.Vector
    fp8_dtype = fx.Float8E4M3FN
    out_type = fx.BFloat16 if out_dtype == "bf16" else fx.Float16

    def _raw_value(v):
        return v.ir_value() if const_expr(hasattr(v, "ir_value")) else v

    def _wait_vmem_and_barrier(vmcnt: int = 0):
        llvm.InlineAsmOp(
            None, [], f"s_waitcnt vmcnt({int(vmcnt)})\ns_barrier", "", has_side_effects=True
        )

    def _barrier():
        llvm.InlineAsmOp(None, [], "s_barrier", "", has_side_effects=True)

    def _sched_barrier():
        return

    def _mfma_f32_f8(a_reg, b_reg, acc, result_type):
        return Vec(llvm.inline_asm(
            result_type,
            [_raw_value(a_reg), _raw_value(b_reg), _raw_value(acc)],
            "v_mfma_f32_16x16x128_f8f6f4 $0, $1, $2, $3",
            "=v,v,v,v",
            has_side_effects=False,
            is_align_stack=False,
        ))

    def _mfma_tile(accs_in, a_regs, b_regs, mfma_res_ty):
        accs = list(accs_in)
        for mi in range_constexpr(4):
            for ni in range_constexpr(2):
                idx = mi * 2 + ni
                accs[idx] = _mfma_f32_f8(a_regs[mi], b_regs[ni], accs[idx], mfma_res_ty)
        return accs

    def _mfma_all_waves(accs_in, a_regs, b_regs, mfma_res_ty):
        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)", "", has_side_effects=True)
        llvm.InlineAsmOp(None, [], "s_setprio 1", "", has_side_effects=True)
        accs_out = _mfma_tile(accs_in, a_regs, b_regs, mfma_res_ty)
        llvm.InlineAsmOp(None, [], "s_setprio 0", "", has_side_effects=True)
        return accs_out

    def _mfma_all_waves_ready(accs_in, a_regs, b_regs, mfma_res_ty):
        llvm.InlineAsmOp(None, [], "s_setprio 1", "", has_side_effects=True)
        accs_out = _mfma_tile(accs_in, a_regs, b_regs, mfma_res_ty)
        llvm.InlineAsmOp(None, [], "s_setprio 0", "", has_side_effects=True)
        return accs_out

    def _raw_copy_slab_to_lds(
        rsrc,
        lds_buffer,
        lds_region_off: int,
        soffset,
        voffset0,
        voffset1,
        wave_id,
    ):
        from flydsl._mlir.dialects import memref as memref_dialect

        lds_base = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
        lds_ptr_base = buffer_ops.create_llvm_ptr(fx.Int64(lds_base), address_space=3)
        wave_offset = rocdl.readfirstlane(
            fx.Int64.ir_type,
            fx.Int64(wave_id * fx.Index(_WAVE_SIZE * 16)),
        )
        lds_ptr0 = buffer_ops.get_element_ptr(
            lds_ptr_base, wave_offset, static_byte_offset=int(lds_region_off)
        )
        lds_ptr1 = buffer_ops.get_element_ptr(lds_ptr0, static_byte_offset=64 * _TILE_K)

        for lds_ptr, voffset in ((lds_ptr0, voffset0), (lds_ptr1, voffset1)):
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                fx.Int32(16),
                fx.Int32(voffset),
                fx.Int32(soffset),
                fx.Int32(0),
                fx.Int32(1),
            )

    def _slab_soffset(base_k, half: int, tile_base):
        return (tile_base + fx.Index(half * _HALF_MN)) * fx.Index(K) + base_k

    def _local_voffset(row_local, col_swz):
        return row_local * fx.Index(K) + col_swz

    def _dma_half(rsrc, lds_buffer, base_k, lds_off: int, tile_base, half: int,
                  wave_id, copy_row_base, copy_col, k_blocks16):
        row0 = copy_row_base
        row1 = copy_row_base + fx.Index(64)
        col0 = swizzle_xor16(row0, copy_col, k_blocks16)
        col1 = swizzle_xor16(row1, copy_col, k_blocks16)
        _raw_copy_slab_to_lds(
            rsrc,
            lds_buffer,
            lds_off,
            _slab_soffset(base_k, half, tile_base),
            _local_voffset(row0, col0),
            _local_voffset(row1, col1),
            wave_id,
        )

    def _dma_full_stage(a_rsrc, b_rsrc, lds_buffer, base_k, bx_m, by_n, *copy_args):
        _dma_half(b_rsrc, lds_buffer, base_k, _B0_OFF, by_n, 0, *copy_args)
        _dma_half(a_rsrc, lds_buffer, base_k, _A0_OFF, bx_m, 0, *copy_args)
        _dma_half(b_rsrc, lds_buffer, base_k, _B1_OFF, by_n, 1, *copy_args)
        _dma_half(a_rsrc, lds_buffer, base_k, _A1_OFF, bx_m, 1, *copy_args)

    def _dma_stage_b0_a0_b1(a_rsrc, b_rsrc, lds_buffer, base_k, bx_m, by_n, *copy_args):
        _dma_half(b_rsrc, lds_buffer, base_k, _B0_OFF, by_n, 0, *copy_args)
        _dma_half(a_rsrc, lds_buffer, base_k, _A0_OFF, bx_m, 0, *copy_args)
        _dma_half(b_rsrc, lds_buffer, base_k, _B1_OFF, by_n, 1, *copy_args)

    def _lds_load_packs(lds_buffer, region_off: int, row_local, col_base, k_blocks16):
        col_swz = swizzle_xor16(row_local, col_base, k_blocks16)
        idx = fx.Index(region_off) + row_local * fx.Index(_TILE_K) + col_swz
        loaded = Vec.load(Vec.make_type(16, fp8_dtype), lds_buffer, [idx])
        as_i64x2 = Vec(loaded).bitcast(fx.Int64)
        return as_i64x2[0].ir_value(), as_i64x2[1].ir_value()

    def _load_a_regs(lds_buffer, half: int, wave_m, lane_mod_16, col_base, k_blocks16):
        region_off = _A0_OFF if half == 0 else _A1_OFF
        regs = []
        for mi in range_constexpr(4):
            row = wave_m * fx.Index(64) + fx.Index(mi * 16) + lane_mod_16
            a0, a1 = _lds_load_packs(lds_buffer, region_off, row, col_base, k_blocks16)
            a2, a3 = _lds_load_packs(
                lds_buffer, region_off, row, col_base + fx.Index(64), k_blocks16
            )
            regs.append(Vec.from_elements([a0, a1, a2, a3], fx.Int64).bitcast(fx.Int32))
        return regs

    def _load_b_regs(lds_buffer, half: int, wave_n, lane_mod_16, col_base, k_blocks16):
        region_off = _B0_OFF if half == 0 else _B1_OFF
        regs = []
        for ni in range_constexpr(2):
            row = wave_n * fx.Index(32) + fx.Index(ni * 16) + lane_mod_16
            b0, b1 = _lds_load_packs(lds_buffer, region_off, row, col_base, k_blocks16)
            b2, b3 = _lds_load_packs(
                lds_buffer, region_off, row, col_base + fx.Index(64), k_blocks16
            )
            regs.append(Vec.from_elements([b0, b1, b2, b3], fx.Int64).bitcast(fx.Int32))
        return regs

    def _compute_k_block(
        acc0,
        acc1,
        acc2,
        acc3,
        cur,
        nxt,
        k_base,
        a_rsrc,
        b_rsrc,
        bx_m,
        by_n,
        copy_args,
        a_load_args,
        b_load_args,
        mfma_res_ty,
    ):
        next_base = k_base + fx.Index(2 * _TILE_K)

        b0_regs = _load_b_regs(cur, 0, *b_load_args)
        a0_regs = _load_a_regs(cur, 0, *a_load_args)
        _dma_half(a_rsrc, nxt, k_base + fx.Index(_TILE_K), _A1_OFF, bx_m, 1, *copy_args)

        acc0 = _mfma_all_waves(acc0, a0_regs, b0_regs, mfma_res_ty)
        _sched_barrier()

        b1_regs = _load_b_regs(cur, 1, *b_load_args)
        _dma_half(b_rsrc, cur, next_base, _B0_OFF, by_n, 0, *copy_args)
        _barrier()

        acc1 = _mfma_all_waves(acc1, a0_regs, b1_regs, mfma_res_ty)

        a1_regs = _load_a_regs(cur, 1, *a_load_args)
        _dma_half(a_rsrc, cur, next_base, _A0_OFF, bx_m, 0, *copy_args)

        acc2 = _mfma_all_waves(acc2, a1_regs, b0_regs, mfma_res_ty)
        _sched_barrier()

        _dma_half(b_rsrc, cur, next_base, _B1_OFF, by_n, 1, *copy_args)
        _wait_vmem_and_barrier(7)

        acc3 = _mfma_all_waves_ready(acc3, a1_regs, b1_regs, mfma_res_ty)
        return acc0, acc1, acc2, acc3

    def _store_quadrant(
        accs_in,
        a_half: int,
        b_half: int,
        c_rsrc,
        scale_a_rsrc,
        scale_b_rsrc,
        bx_m,
        by_n,
        wave_m,
        wave_n,
        lane_div_16,
        lane_mod_16,
        c_n,
    ):
        s_a_vecs = []
        row_bases = []
        for mi in range_constexpr(4):
            row_base = (
                bx_m
                + fx.Index(a_half * _HALF_MN)
                + wave_m * fx.Index(64)
                + fx.Index(mi * 16)
                + lane_div_16 * fx.Index(4)
            )
            row_bases.append(row_base)
            s_a_vecs.append(
                Vec(buffer_ops.buffer_load(
                    scale_a_rsrc, row_base, vec_width=4, dtype=fx.Float32
                ))
            )

        s_b_vals = []
        col_vals = []
        for ni in range_constexpr(2):
            col = (
                by_n
                + fx.Index(b_half * _HALF_MN)
                + wave_n * fx.Index(32)
                + fx.Index(ni * 16)
                + lane_mod_16
            )
            col_vals.append(col)
            s_b_vals.append(buffer_ops.buffer_load(scale_b_rsrc, col, vec_width=1, dtype=fx.Float32))

        for mi in range_constexpr(4):
            for ni in range_constexpr(2):
                acc = Vec(accs_in[mi * 2 + ni])
                for ii in range_constexpr(4):
                    row = row_bases[mi] + fx.Index(ii)
                    val = acc[ii] * (s_a_vecs[mi][ii] * s_b_vals[ni])
                    buffer_ops.buffer_store(out_type(val), c_rsrc, row * c_n + col_vals[ni])

    @flyc.kernel(known_block_size=[_TOTAL_THREADS, 1, 1])
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        _arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        c_m = fx.Index(i32_m)
        c_n = fx.Index(i32_n)
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx_m = bx * _TILE_M
        by_n = by * _TILE_N

        lds_pong = SmemPtr(
            allocator_pong.get_base(), lds_pong_offset, fp8_dtype.ir_type,
            shape=(_STAGE_BYTES,)
        ).get()
        lds_ping = SmemPtr(
            allocator_ping.get_base(), lds_ping_offset, fp8_dtype.ir_type,
            shape=(_STAGE_BYTES,)
        ).get()

        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=fx.Int64(c_m * K)
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=fx.Int64(c_n * K)
        )
        c_rsrc = buffer_ops.create_buffer_resource(
            arg_c, max_size=False, num_records_bytes=fx.Int64(c_m * c_n * 2)
        )
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=False)

        wave_lane = fx.idx2crd(
            tx, fx.make_layout((_NUM_WAVES, _WAVE_SIZE), (_WAVE_SIZE, 1))
        )
        wave_id = fx.get(wave_lane, 0)
        lane_id = fx.get(wave_lane, 1)

        lane16 = fx.idx2crd(lane_id, fx.make_layout((4, 16), (16, 1)))
        lane_div_16 = fx.get(lane16, 0)
        lane_mod_16 = fx.get(lane16, 1)

        wave_m = wave_id // fx.Index(4)
        wave_n = wave_id % fx.Index(4)
        copy_row_base = (
            (wave_id // fx.Index(2)) * fx.Index(16)
            + (((wave_id % fx.Index(2)) * fx.Index(64) + lane_id) // fx.Index(8))
        )
        copy_col = (lane_id % fx.Index(8)) * fx.Index(16)
        k_blocks16 = fx.Index(_TILE_K // 16)
        col_base = lane_div_16 * fx.Index(16)

        copy_args = (wave_id, copy_row_base, copy_col, k_blocks16)
        a_load_args = (wave_m, lane_mod_16, col_base, k_blocks16)
        b_load_args = (wave_n, lane_mod_16, col_base, k_blocks16)
        store_args = (
            c_rsrc, scale_a_rsrc, scale_b_rsrc, bx_m, by_n,
            wave_m, wave_n, lane_div_16, lane_mod_16, c_n,
        )

        acc_init = Vec.filled(4, 0.0, fx.Float32)
        mfma_res_ty = Vec.make_type(4, fx.Float32)
        init_state = [acc_init] * 32
        results = init_state

        _dma_full_stage(a_rsrc, b_rsrc, lds_pong, fx.Index(0), bx_m, by_n, *copy_args)
        _wait_vmem_and_barrier(7)

        _dma_stage_b0_a0_b1(
            a_rsrc, b_rsrc, lds_ping, fx.Index(_TILE_K), bx_m, by_n, *copy_args
        )
        _wait_vmem_and_barrier(7)

        for k_pair_base, state in range(0, K - (2 * _TILE_K), 2 * _TILE_K, init=init_state):
            acc0 = list(state[0:8])
            acc1 = list(state[8:16])
            acc2 = list(state[16:24])
            acc3 = list(state[24:32])

            acc0, acc1, acc2, acc3 = _compute_k_block(
                acc0, acc1, acc2, acc3,
                lds_pong, lds_ping, k_pair_base,
                a_rsrc, b_rsrc, bx_m, by_n, copy_args,
                a_load_args, b_load_args, mfma_res_ty,
            )
            acc0, acc1, acc2, acc3 = _compute_k_block(
                acc0, acc1, acc2, acc3,
                lds_ping, lds_pong, k_pair_base + fx.Index(_TILE_K),
                a_rsrc, b_rsrc, bx_m, by_n, copy_args,
                a_load_args, b_load_args, mfma_res_ty,
            )

            results = yield list(acc0) + list(acc1) + list(acc2) + list(acc3)

        acc0 = list(results[0:8])
        acc1 = list(results[8:16])
        acc2 = list(results[16:24])
        acc3 = list(results[24:32])
        tail_k = fx.Index(K - (2 * _TILE_K))
        cur = lds_pong
        nxt = lds_ping

        b0_regs = _load_b_regs(cur, 0, *b_load_args)
        a0_regs = _load_a_regs(cur, 0, *a_load_args)
        _dma_half(a_rsrc, nxt, tail_k + fx.Index(_TILE_K), _A1_OFF, bx_m, 1, *copy_args)

        acc0 = _mfma_all_waves(acc0, a0_regs, b0_regs, mfma_res_ty)
        _sched_barrier()

        b1_regs = _load_b_regs(cur, 1, *b_load_args)

        acc1 = _mfma_all_waves(acc1, a0_regs, b1_regs, mfma_res_ty)

        a1_regs = _load_a_regs(cur, 1, *a_load_args)
        _wait_vmem_and_barrier(4)

        acc2 = _mfma_all_waves(acc2, a1_regs, b0_regs, mfma_res_ty)

        b0_regs = _load_b_regs(nxt, 0, *b_load_args)

        acc3 = _mfma_all_waves_ready(acc3, a1_regs, b1_regs, mfma_res_ty)
        _sched_barrier()

        cur = lds_ping

        a0_regs = _load_a_regs(cur, 0, *a_load_args)
        _wait_vmem_and_barrier(0)

        acc0 = _mfma_all_waves(acc0, a0_regs, b0_regs, mfma_res_ty)
        _store_quadrant(acc0, 0, 0, *store_args)

        b1_regs = _load_b_regs(cur, 1, *b_load_args)
        _sched_barrier()

        acc1 = _mfma_all_waves(acc1, a0_regs, b1_regs, mfma_res_ty)
        _store_quadrant(acc1, 0, 1, *store_args)

        a1_regs = _load_a_regs(cur, 1, *a_load_args)

        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)", "", has_side_effects=True)
        llvm.InlineAsmOp(None, [], "s_setprio 1", "", has_side_effects=True)
        acc2 = _mfma_tile(acc2, a1_regs, b0_regs, mfma_res_ty)
        acc3 = _mfma_tile(acc3, a1_regs, b1_regs, mfma_res_ty)
        llvm.InlineAsmOp(None, [], "s_setprio 0", "", has_side_effects=True)

        _store_quadrant(acc2, 1, 0, *store_args)
        _store_quadrant(acc3, 1, 1, *store_args)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        launcher = kernel_gemm(
            arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, i32_m, i32_n
        )
        if const_expr(waves_per_eu is not None):
            waves = int(waves_per_eu)
            if const_expr(waves >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            fx.Int32.ir_type, waves
                        )
        launcher.launch(
            grid=(i32_m // _TILE_M, i32_n // _TILE_N, 1),
            block=(_TOTAL_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gemm


def compile_preshuffle_gemm_fp8_8wave_aonly_pingpong(**kwargs):
    kwargs["aonly_wave_pingpong"] = True
    return compile_preshuffle_gemm_fp8_8wave(**kwargs)


__all__ = [
    "compile_preshuffle_gemm_fp8_8wave",
    "compile_preshuffle_gemm_fp8_8wave_aonly_pingpong",
]
