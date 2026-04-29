# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compact 8-wave preshuffle FP8 GEMM with row-wise fp32 scaling.

This file is intentionally narrow:
- A/B are FP8 E4M3 values.
- scale_a and scale_b are row-wise fp32 vectors.
- output is fp16 or bf16.
- A uses a two-buffer LDS ping-pong pipeline with async global-to-LDS copies.
- B is read from the existing preshuffled layout.

It does not contain FP4/MXFP8/int8 paths, CShuffle, or fused epilogues.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    swizzle_xor16,
    tile_chunk_coord_i32,
)


_NUM_WAVES = 8
_WAVE_SIZE = 64
_TOTAL_THREADS = _NUM_WAVES * _WAVE_SIZE

_TILE_PRELOAD_DEFAULT = (0, 0)
_TILE_PRELOAD_TABLE = {
    # (num_waves, tile_m, tile_n, tile_k): (dsrd_preload, dvmem_preload)
    (8, 128, 512, 128): (6, 4),
    (8, 256, 256, 128): (4, 4),
}


def _get_preload(tile_m: int, tile_n: int, tile_k: int):
    return _TILE_PRELOAD_TABLE.get(
        (_NUM_WAVES, int(tile_m), int(tile_n), int(tile_k)),
        _TILE_PRELOAD_DEFAULT,
    )


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
):
    """Compile an 8-wave FP8 preshuffle GEMM with row-wise fp32 scales.

    Signature:
      launch(c, a_fp8, b_preshuffled_fp8, scale_a_f32, scale_b_f32,
             unused_bias, M, N, stream)
    """

    gpu_arch = get_hip_arch()
    if str(gpu_arch) != "gfx950":
        raise RuntimeError(f"8-wave row-wise FP8 GEMM currently requires gfx950, got {gpu_arch}")
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}")
    if int(lds_stage) != 2:
        raise NotImplementedError("compact 8-wave FP8 kernel only supports lds_stage=2")
    if not bool(use_async_copy):
        raise NotImplementedError("compact 8-wave FP8 kernel only supports async A copies")
    if bool(use_cshuffle_epilog):
        raise NotImplementedError("compact 8-wave FP8 kernel does not support CShuffle")
    if int(tile_k) % 128 != 0:
        raise ValueError(f"tile_k must be divisible by 128, got tile_k={tile_k}")
    if int(K) % int(tile_k) != 0:
        raise ValueError(f"K must be divisible by tile_k, got K={K}, tile_k={tile_k}")
    if (int(K) // int(tile_k)) < 2 or (int(K) // int(tile_k)) % 2 != 0:
        raise NotImplementedError(
            "compact 8-wave FP8 kernel currently requires an even number of K tiles >= 2"
        )
    if int(tile_n) % _NUM_WAVES != 0:
        raise ValueError(f"tile_n must be divisible by {_NUM_WAVES}, got {tile_n}")
    if (int(tile_n) // _NUM_WAVES) % 16 != 0:
        raise ValueError(f"tile_n/{_NUM_WAVES} must be divisible by 16, got {tile_n}")

    if dsrd_preload < 0 or dvmem_preload < 0:
        computed_dsrd, computed_dvmem = _get_preload(tile_m, tile_n, tile_k)
        if dsrd_preload < 0:
            dsrd_preload = computed_dsrd
        if dvmem_preload < 0:
            dvmem_preload = computed_dvmem

    elem_bytes = 1
    tile_k_bytes = int(tile_k)
    bytes_a_per_tile = int(tile_m) * int(tile_k)
    bytes_b_per_tile = int(tile_n) * int(tile_k)
    if bytes_a_per_tile % _TOTAL_THREADS != 0:
        raise ValueError(
            f"tile_m*tile_k must be divisible by {_TOTAL_THREADS}, "
            f"got tile_m={tile_m}, tile_k={tile_k}"
        )
    if bytes_b_per_tile % _TOTAL_THREADS != 0:
        raise ValueError(
            f"tile_n*tile_k must be divisible by {_TOTAL_THREADS}, "
            f"got tile_n={tile_n}, tile_k={tile_k}"
        )

    a_load_bytes = 16
    a_async_load_bytes = 16
    a_async_load_dword = a_async_load_bytes // 4
    bytes_per_thread_a = bytes_a_per_tile // _TOTAL_THREADS
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by 16")

    b_load_bytes = 16
    bytes_per_thread_b = bytes_b_per_tile // _TOTAL_THREADS
    if bytes_per_thread_b % b_load_bytes != 0:
        raise ValueError(f"bytes_per_thread_b ({bytes_per_thread_b}) must be divisible by 16")

    num_b_loads = bytes_per_thread_b // b_load_bytes
    num_a_lds_load = bytes_a_per_tile // _WAVE_SIZE // a_load_bytes
    num_a_async_loads = bytes_per_thread_a // a_async_load_bytes

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    lds_tile_bytes = int(tile_m) * int(tile_k)
    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + lds_tile_bytes
    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + lds_tile_bytes

    out_is_bf16 = out_dtype == "bf16"
    Vec = fx.Vector

    def _fp8_dtype():
        return fx.Float8E4M3FN

    def _elem_type():
        return _fp8_dtype().ir_type

    def _vec16_type():
        return Vec.make_type(16, _fp8_dtype())

    def _out_dtype():
        return fx.BFloat16 if out_is_bf16 else fx.Float16

    def _out_elem_type():
        return _out_dtype().ir_type

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
        from flydsl._mlir.dialects import memref as memref_dialect

        c_m = fx.Index(i32_m)
        c_n = fx.Index(i32_n)
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx_m = bx * tile_m
        by_n = by * tile_n

        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_a_pong = SmemPtr(
            base_ptr_pong, lds_pong_offset, _elem_type(), shape=(tile_m * tile_k,)
        ).get()
        lds_a_ping = SmemPtr(
            base_ptr_ping, lds_ping_offset, _elem_type(), shape=(tile_m * tile_k,)
        ).get()

        a_nrec = fx.Int64(c_m * K)
        c_nrec = fx.Int64(c_m * c_n * 2)
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=a_nrec
        )
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(
            arg_c, max_size=False, num_records_bytes=c_nrec
        )
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        wave_lane_layout = fx.make_layout((_NUM_WAVES, _WAVE_SIZE), (_WAVE_SIZE, 1))
        wave_lane = fx.idx2crd(tx, wave_lane_layout)
        wave_id = fx.get(wave_lane, 0)
        lane_id = fx.get(wave_lane, 1)

        lane16_layout = fx.make_layout((4, 16), (16, 1))
        lane16 = fx.idx2crd(lane_id, lane16_layout)
        lane_div_16 = fx.get(lane16, 0)
        lane_mod_16 = fx.get(lane16, 1)

        row_a_lds = lane_mod_16
        col_offset_base_bytes = lane_div_16 * 16
        m_repeat = tile_m // 16
        k_unroll = tile_k // 64
        n_per_wave = tile_n // _NUM_WAVES
        num_acc_n = n_per_wave // 16
        n_tile_base = wave_id * n_per_wave

        acc_init = Vec.filled(4, 0.0, fx.Float32)
        mfma_res_ty = Vec.make_type(4, fx.Float32)

        # Preshuffled B layout:
        #   (N/16, K/64, 4, 16, 16)
        #   strides keep four K-lanes packed under each 16-column group.
        n0_val = N // 16
        k0_val = K // 64
        stride_nlane = 16
        stride_klane = 16 * stride_nlane
        stride_k0 = 4 * stride_klane
        stride_n0 = k0_val * stride_k0
        b_dword_stride_n0 = stride_n0 // 4
        b_dword_stride_k0 = stride_k0 // 4
        b_dword_stride_klane = stride_klane // 4
        b_dword_stride_nlane = stride_nlane // 4
        b_dword_stride_k0_c = fx.Index(b_dword_stride_k0)

        n_blk_list = []
        n_intra_list = []
        b_n_full_dword = []
        for ni in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + (ni * 16) + lane_mod_16
            n_blk = global_n // 16
            n_intra = global_n % 16
            n_blk_list.append(n_blk)
            n_intra_list.append(n_intra)
            b_n_full_dword.append(
                n_blk * fx.Index(b_dword_stride_n0)
                + n_intra * fx.Index(b_dword_stride_nlane)
                + lane_div_16 * fx.Index(b_dword_stride_klane)
            )

        def extract_b_packs(b16):
            b_i64x2 = Vec(b16).bitcast(fx.Int64)
            return b_i64x2[0].ir_value(), b_i64x2[1].ir_value()

        def load_b_single(k_dword_offset, ni: int):
            dword_idx = fx.Int32(b_n_full_dword[ni] + k_dword_offset)
            b_vec4 = buffer_ops.buffer_load(
                b_rsrc, dword_idx, vec_width=4, dtype=fx.Int32
            )
            b16 = Vec(b_vec4).bitcast(_fp8_dtype())
            return extract_b_packs(b16)

        def load_b_tile(base_k):
            k0_base = base_k // fx.Index(64)
            k_dwords = []
            for ku in range_constexpr(k_unroll):
                k_dwords.append((k0_base + ku) * b_dword_stride_k0_c)

            packs0_per_ku = [[] for _ in range(k_unroll)]
            packs1_per_ku = [[] for _ in range(k_unroll)]
            for ni in range_constexpr(num_acc_n):
                for ku in range_constexpr(k_unroll):
                    b0, b1 = load_b_single(k_dwords[ku], ni)
                    packs0_per_ku[ku].append(b0)
                    packs1_per_ku[ku].append(b1)

            b_tile = []
            for ku in range_constexpr(k_unroll):
                b_tile.append((packs0_per_ku[ku], packs1_per_ku[ku]))
            return b_tile

        k_blocks16 = fx.Index(tile_k // 16)
        lds_k_dim_c = fx.Index(tile_k)

        def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
            col_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            idx_a16 = curr_row_a_lds * lds_k_dim_c + col_swz
            loaded_a16 = Vec.load(_vec16_type(), lds_buffer, [idx_a16])
            a_i64x2 = Vec(loaded_a16).bitcast(fx.Int64)
            return a_i64x2[0].ir_value(), a_i64x2[1].ir_value()

        tile_k_dwords = tile_k // 4
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        c4 = fx.Index(4)
        tx_i32_base = tx * c4
        tx_i32_async_base = tx * a_async_load_dword

        def a_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=_TOTAL_THREADS,
                layout_tile_div4=layout_a_tile_div4,
            )

        def load_a_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                fx.vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=a_rsrc,
                vec_elems=16,
                elem_bytes=elem_bytes,
            )

        def load_a_tile(base_k_div4):
            parts = []
            for i in range_constexpr(bytes_per_thread_a // a_load_bytes):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                row_a_global = bx_m + row_a_local
                idx_i32 = row_a_global * fx.Index(K // 4) + base_k_div4 + col_a_local_i32
                parts.append(Vec(load_a_16(idx_i32)).bitcast(fx.Int32))
            return parts

        def a_tile_chunk_coord_i32_async(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_async_base,
                i=i,
                total_threads=_TOTAL_THREADS,
                layout_tile_div4=layout_a_tile_div4,
                chunk_i32=a_async_load_dword,
            )

        def dma_a_tile_to_lds(base_k_div4, lds_buffer):
            wave_offset = rocdl.readfirstlane(
                fx.Int64.ir_type,
                fx.Int64(wave_id * fx.Index(_WAVE_SIZE * a_async_load_bytes)),
            )
            lds_base = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
            lds_ptr_base = buffer_ops.create_llvm_ptr(fx.Int64(lds_base), address_space=3)
            lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, wave_offset)

            for i in range_constexpr(num_a_async_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32_async(i)
                col_a_swz = swizzle_xor16(row_a_local, col_a_local_i32 * c4, k_blocks16)
                row_a_global = bx_m + row_a_local
                global_byte_idx = row_a_global * fx.Index(K) + base_k_div4 * c4 + col_a_swz
                if const_expr(i > 0):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=_TOTAL_THREADS * a_async_load_bytes,
                    )
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    lds_ptr,
                    fx.Int32(a_async_load_bytes),
                    fx.Int32(global_byte_idx),
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(1),
                )

        def prefetch_a_to_lds(base_k, lds_buffer):
            dma_a_tile_to_lds(base_k // fx.Index(4), lds_buffer)

        def prefetch_a_tile(base_k):
            return load_a_tile(base_k // fx.Index(4))

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            return Vec.from_elements([x0, x1, x2, x3], fx.Int64).bitcast(fx.Int32)

        def prefetch_output_scales():
            s_b_vals = []
            for ni in range_constexpr(num_acc_n):
                col = by_n + n_tile_base + (ni * 16) + lane_mod_16
                s_b_vals.append(
                    buffer_ops.buffer_load(scale_b_rsrc, col, vec_width=1, dtype=fx.Float32)
                )

            s_a_vecs = []
            row_off_base = lane_div_16 * 4
            for mi in range_constexpr(m_repeat):
                row_base = bx_m + (mi * 16) + row_off_base
                s_a_vec = buffer_ops.buffer_load(
                    scale_a_rsrc, row_base, vec_width=4, dtype=fx.Float32
                )
                s_a_vecs.append(Vec(s_a_vec))
            return s_a_vecs, s_b_vals

        def compute_tile(accs_in, b_tile_in, lds_buffer, *, is_last_tile=False, a0_prefetch=None):
            scales = None
            if const_expr(is_last_tile):
                scales = prefetch_output_scales()

            accs = list(accs_in)
            c0_i64 = fx.Int64(0)
            for ku128 in range_constexpr(k_unroll // 2):
                ku0 = ku128 * 2
                ku1 = ku0 + 1
                b0_packs0, b0_packs1 = b_tile_in[ku0]
                b1_packs0, b1_packs1 = b_tile_in[ku1]
                col_base0 = col_offset_base_bytes + (ku0 * 64)
                col_base1 = col_offset_base_bytes + (ku1 * 64)

                for mi in range_constexpr(m_repeat):
                    curr_row = row_a_lds + (mi * 16)
                    if const_expr((a0_prefetch is not None) and (ku0 == 0) and (mi == 0)):
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row, col_base0, lds_buffer)
                    a2, a3 = lds_load_packs_k64(curr_row, col_base1, lds_buffer)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b0_packs0[ni],
                            b0_packs1[ni],
                            b1_packs0[ni],
                            b1_packs1[ni],
                        )
                        acc_idx = mi * num_acc_n + ni
                        accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [
                                a128,
                                b128,
                                accs[acc_idx],
                                0,
                                0,
                                0,
                                0x7F7F7F7F,
                                0,
                                0x7F7F7F7F,
                            ],
                        )
            return accs, scales

        def store_output(final_accs, scales):
            s_a_vecs, s_b_vals = scales
            col_base = by_n + n_tile_base + lane_mod_16
            for mi in range_constexpr(m_repeat):
                s_a_vec = s_a_vecs[mi]
                row_base = bx_m + (mi * 16) + lane_div_16 * 4
                for ii in range_constexpr(4):
                    row = row_base + fx.Index(ii)
                    s_a = Vec(s_a_vec)[ii]
                    idx_base = row * c_n + col_base
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        val = Vec(final_accs[acc_idx])[ii]
                        out_val = _out_dtype()(val * (s_a * s_b_vals[ni]))
                        buffer_ops.buffer_store(out_val, c_rsrc, idx_base + (ni * 16))

        def build_scheduler(numer: int, denom: int):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_total = (tile_k // 128) * m_repeat * num_acc_n
            dsrd_preload_eff = min(int(dsrd_preload), num_a_lds_load)
            dvmem_preload_eff = min(int(dvmem_preload), num_b_loads + num_a_async_loads)
            dsrd_remaining = num_a_lds_load - dsrd_preload_eff
            vmem_remaining = num_b_loads + num_a_async_loads - dvmem_preload_eff
            dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
            vmem_schedule = build_scheduler(vmem_remaining, mfma_total)

            idx_ds_read = dsrd_preload_eff
            idx_gmem_load = dvmem_preload_eff
            if const_expr(dvmem_preload_eff):
                rocdl.sched_vmem(dvmem_preload_eff)
            if const_expr(dsrd_preload_eff):
                rocdl.sched_dsrd(dsrd_preload_eff)
            for mfma_idx in range_constexpr(mfma_total):
                rocdl.sched_mfma(1)
                n_dsrd = dsrd_schedule[mfma_idx]
                if const_expr(n_dsrd and idx_ds_read < num_a_lds_load):
                    if const_expr(idx_ds_read + n_dsrd > num_a_lds_load):
                        n_dsrd = num_a_lds_load - idx_ds_read
                    if const_expr(n_dsrd):
                        rocdl.sched_dsrd(n_dsrd)
                        idx_ds_read += n_dsrd

                n_vmem = vmem_schedule[mfma_idx]
                if const_expr(n_vmem and idx_gmem_load < (num_b_loads + num_a_async_loads)):
                    if const_expr(idx_gmem_load + n_vmem > (num_b_loads + num_a_async_loads)):
                        n_vmem = num_b_loads + num_a_async_loads - idx_gmem_load
                    if const_expr(n_vmem):
                        rocdl.sched_vmem(n_vmem)
                        idx_gmem_load += n_vmem
            rocdl.sched_barrier(0)

        def flatten_b_tile(b_tile):
            flat = []
            for packs0, packs1 in b_tile:
                flat.extend(packs0)
                flat.extend(packs1)
            return flat

        def unflatten_b_tile(flat):
            b_tile = []
            idx = 0
            for _ in range_constexpr(k_unroll):
                packs0 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                packs1 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                b_tile.append((packs0, packs1))
            return b_tile

        n_accs = num_acc_n * m_repeat
        n_btile = k_unroll * 2 * num_acc_n

        def pack_state(accs, b_flat, a0_prefetch):
            return list(accs) + list(b_flat) + [a0_prefetch[0], a0_prefetch[1]]

        def unpack_state(vals):
            accs = list(vals[:n_accs])
            b_flat = list(vals[n_accs:n_accs + n_btile])
            a0 = vals[n_accs + n_btile]
            a1 = vals[n_accs + n_btile + 1]
            return accs, b_flat, (a0, a1)

        def prefetch_a0_pack(lds_buffer):
            return lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)

        def pingpong_body(k_iv, inner_state):
            accs, b_flat, a0_prefetch = unpack_state(inner_state)
            b_tile_pong = unflatten_b_tile(b_flat)

            next_k = k_iv + tile_k
            prefetch_a_to_lds(next_k, lds_a_ping)
            b_tile_ping = load_b_tile(next_k)
            accs, _ = compute_tile(
                accs, b_tile_pong, lds_a_pong, a0_prefetch=a0_prefetch
            )
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

            next_k = k_iv + (tile_k * 2)
            prefetch_a_to_lds(next_k, lds_a_pong)
            b_tile_pong_new = load_b_tile(next_k)
            accs, _ = compute_tile(
                accs, b_tile_ping, lds_a_ping, a0_prefetch=a0_prefetch_ping
            )
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_pong_new = prefetch_a0_pack(lds_a_pong)

            return pack_state(accs, flatten_b_tile(b_tile_pong_new), a0_prefetch_pong_new)

        rocdl.sched_barrier(0)

        k0 = fx.Index(0)
        b_tile0 = load_b_tile(k0)
        prefetch_a_to_lds(k0, lds_a_pong)
        gpu.barrier()

        accs = [acc_init] * n_accs
        a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)
        init_state = pack_state(accs, flatten_b_tile(b_tile0), a0_prefetch_pong)

        results = init_state
        c_k_stop = K - (tile_k * 3)
        for iv, state in range(0, c_k_stop, tile_k * 2, init=init_state):
            results = yield pingpong_body(iv, state)

        accs, b_flat, a0_prefetch = unpack_state(results)
        b_tile_pong = unflatten_b_tile(b_flat)

        last_k = fx.Index(K - tile_k)
        b_tile_ping = load_b_tile(last_k)
        prefetch_a_to_lds(last_k, lds_a_ping)
        accs, _ = compute_tile(
            accs, b_tile_pong, lds_a_pong, a0_prefetch=a0_prefetch
        )
        hot_loop_scheduler()
        rocdl.s_waitcnt(num_b_loads)
        gpu.barrier()
        a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

        final_accs, scales = compute_tile(
            accs,
            b_tile_ping,
            lds_a_ping,
            is_last_tile=True,
            a0_prefetch=a0_prefetch_ping,
        )
        store_output(final_accs, scales)

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
        from flydsl._mlir import ir

        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n
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
            grid=(gx, gy, 1),
            block=(_TOTAL_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gemm


__all__ = ["compile_preshuffle_gemm_fp8_8wave"]
