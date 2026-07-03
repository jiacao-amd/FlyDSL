#!/bin/bash
# Run bench_conv3d_8wave.py one shape at a time to avoid MIOpen OOM.
set -e

IMAGE=rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0
COMMON=(
    --rm --device=/dev/kfd --device=/dev/dri --group-add video
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
    -v /home/jiacao/amd-inference:/workspace/amd-inference
    -v /home/jiacao/aiter:/workspace/aiter
    -v /home/jiacao/FlyDSL:/workspace/FlyDSL
    -w /workspace/FlyDSL
    -e HIP_VISIBLE_DEVICES=0
    -e PYTHONPATH=/workspace/FlyDSL/build-fly/python_packages:/workspace/FlyDSL
    -e LD_LIBRARY_PATH=/workspace/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs
    -e FLYDSL_RUNTIME_ENABLE_CACHE=0
)

docker run "${COMMON[@]}" -e BENCH_HEADER_ONLY=1 "$IMAGE" python bench_conv3d_8wave.py 2>/dev/null

N_SHAPES=7
for idx in $(seq 0 $((N_SHAPES - 1))); do
    docker run "${COMMON[@]}" -e BENCH_IDX="$idx" "$IMAGE" python bench_conv3d_8wave.py 2>/dev/null
    sleep 3
done
echo ""
