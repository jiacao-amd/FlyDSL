#!/usr/bin/env bash
# Run bench_conv3d_wan_vae.py one shape at a time in the ROCm container.
# Each invocation gets a fresh process so compilation memory is released.
# Usage: bash bench_conv3d_wan_vae_runner.sh [WARMUP] [ITERS]

set -uo pipefail

WARMUP=${1:-5}
ITERS=${2:-20}
NUM_SHAPES=9  # must match len(SHAPES) in bench_conv3d_wan_vae.py (0..8)

DOCKER_IMAGE="rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0"
WORKSPACE="/home/jiacao/FlyDSL"
DOCKER_WORKSPACE="/workspace/FlyDSL"

CACHE_DIR="${WORKSPACE}/bench_conv3d_wan_vae_cache"
mkdir -p "${CACHE_DIR}"

COMMON_DOCKER_ARGS=(
  --rm
  --device=/dev/kfd --device=/dev/dri
  --group-add video
  --ipc=host
  -v "${WORKSPACE}:${DOCKER_WORKSPACE}"
  -v "${CACHE_DIR}:/bench_cache"
  -w "${DOCKER_WORKSPACE}"
  -e "PYTHONPATH=${DOCKER_WORKSPACE}/build-fly/python_packages:${DOCKER_WORKSPACE}"
  -e "LD_LIBRARY_PATH=${DOCKER_WORKSPACE}/build-fly/python_packages/flydsl/_mlir/_mlir_libs"
  -e "FLYDSL_RUNTIME_CACHE_DIR=/bench_cache"
  -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}"
  -e "BENCH_WARMUP=${WARMUP}"
  -e "BENCH_ITERS=${ITERS}"
)

# Print header
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e BENCH_HEADER_ONLY=1 \
  "${DOCKER_IMAGE}" \
  python3 bench_conv3d_wan_vae.py

# Run each shape in its own container (fresh memory per invocation).
# Sleep 2s between runs so the previous container fully releases GPU state.
for i in $(seq 0 $((NUM_SHAPES - 1))); do
  sleep 5
  result=$(docker run "${COMMON_DOCKER_ARGS[@]}" \
    -e "BENCH_IDX=${i}" \
    "${DOCKER_IMAGE}" \
    python3 bench_conv3d_wan_vae.py 2>&1)
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    # retry once after a longer wait
    sleep 10
    result=$(docker run "${COMMON_DOCKER_ARGS[@]}" \
      -e "BENCH_IDX=${i}" \
      "${DOCKER_IMAGE}" \
      python3 bench_conv3d_wan_vae.py 2>&1)
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
      echo "  [shape $i]: failed after retry (exit $exit_code) — run standalone with BENCH_IDX=$i"
    else
      echo "$result"
    fi
  else
    echo "$result"
  fi
done

echo ""
echo "Notes:"
echo "  'skip' = kernel constraints not met (C%8/CRS%32 for BF16; C%128/etc for FP8)"
echo "  'err'  = eligible but kernel raised an error"
echo "  BF16/PT, FP8/PT = PT_time / FlyDSL_time ratio (>1 = FlyDSL faster)"
echo "  FP8 requires CDNA4 (gfx95x), C%16==0 (partial M/N/K tiles are masked)"
