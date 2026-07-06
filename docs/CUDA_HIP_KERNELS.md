# Writing GPU kernels for both CUDA and HIP

SCAMP's GPU kernels compile for NVIDIA (CUDA) and AMD (HIP/ROCm) from one
source tree. The compatibility shim `src/common/cuda_to_hip.h` aliases the
CUDA spellings to their HIP equivalents, so most host and kernel code needs no
`#ifdef`. The one place the two platforms genuinely differ is the **warp
width**: a warp is 32 lanes on NVIDIA and on RDNA AMD parts (gfx10xx/11xx/12xx,
"wave32"), but 64 lanes on CDNA AMD parts (gfx90a, gfx94x, "wave64"). Any code
that bakes in 32 is silently wrong on wave64.

The shim exposes two symbols to write against instead of the literal 32:

- `kWarpSize` -- the warp width. In device code it is 64 under `__GFX9__`
  (CDNA) and 32 otherwise (RDNA, CUDA). In host code it is a compile-time
  upper bound; query the real width at runtime with
  `hipGetDeviceProperties(...).warpSize`.
- `SCAMP_FULL_WARP_MASK` -- the all-lanes shuffle mask: `0xffffffffffffffffULL`
  (64-bit) on HIP, `0xffffffff` on CUDA.

When you add or edit a kernel, walk this checklist. Each item is a real hazard
from porting SCAMP's kernels; the "portable" column is the pattern already used
in `kernels_compute.h`, `kernels_compute_shfl.h`, and `kernels_impl_shfl.h`.

## Checklist

### 1. Shuffle masks

- **Trap:** a literal `0xffffffff` mask passed to `__shfl_*_sync`. That is a
  32-bit value; on a 64-lane wave it names only the low half of the warp, so
  the upper 32 lanes drop out of the exchange and results are wrong.
- **Portable:** use `SCAMP_FULL_WARP_MASK`. It is 64-bit wide on HIP and 32-bit
  on CUDA, so `__shfl_down_sync(SCAMP_FULL_WARP_MASK, x, delta)` addresses the
  whole warp on both. Never write a hex mask literal in a kernel.

### 2. Reduction strides

- **Trap:** a warp reduction that starts at `delta = 16` (i.e. 32/2) and halves
  down to 1. It only reduces 32 lanes; on wave64 the top 32 lanes' partials are
  never folded in.
- **Portable:** start the butterfly at `kWarpSize / 2` and halve to 1:
  `for (int delta = kWarpSize / 2; delta >= 1; delta /= 2) sum +=
  __shfl_down_sync(SCAMP_FULL_WARP_MASK, sum, delta);`. This yields strides
  16,8,4,2,1 on wave32 and 32,16,8,4,2,1 on wave64 automatically.

### 3. Lane-index math

- **Trap:** `threadIdx.x & 31` for the lane-in-warp and `threadIdx.x >> 5` for
  the warp-id. The `& 31` mask and `>> 5` shift hard-code a 32-lane warp; on
  wave64 the lane index wraps at 32 and the warp-id doubles.
- **Portable:** `warpln = threadIdx.x % kWarpSize`, `warpid = threadIdx.x /
  kWarpSize`. For the "previous lane" wrap use
  `(warpln + kWarpSize - 1) % kWarpSize`, not `(warpln + 31) & 31`. A first-lane
  or last-lane test becomes `warpln == 0` / `warpln == kWarpSize - 1`, never a
  compare against 31.

### 4. Warp-count math

- **Trap:** `BLOCKSZ / 32` (or `* 32`) to derive warps-per-block or block size
  from a warp count. On wave64 a block has half as many warps, so this
  over-counts by 2x and mis-sizes every per-warp structure.
- **Portable:** `warps_per_block = BLOCKSZ / kWarpSize` and
  `BLOCKSZ = warps_per_block * kWarpSize`. Guard the divisibility with
  `static_assert(BLOCKSZ % kWarpSize == 0, ...)`. Note the direction: a wave64
  block yields *fewer, wider* warps than the same block on wave32.

### 5. Host-sized dynamic shared memory

- **Trap:** the host computes a dynamic-smem size for a per-warp region using
  the *device's* warp width, or using the wave64 width as an upper bound. The
  per-warp region (e.g. the cross-warp hand-off buffer, `2 * warps_per_block`
  scalars) grows as the warp width *shrinks*, because a smaller warp means more
  warps per block. Sizing it with the largest warp width **under-allocates** on
  a wave32 device and the kernel writes past the smem region.
- **Portable:** size any per-warp dynamic-smem region on the host with the
  *smallest* possible warp width, `kMinWarpSize = 32`, so the allocation bounds
  the device's use for any runtime warp width. See `get_smem_shfl` in
  `src/core/gpu_kernel/kernel_gpu_utils.cu`:
  `warps_per_block = blocksz / kMinWarpSize`. Over-allocating a few slots on
  wave64 is harmless; under-allocating corrupts memory.

## Before you commit a kernel change

- Grep your diff for `32`, `31`, `0xffffffff`, `>> 5`, `<< 5`, `& 31`. Every
  hit in warp-related code should be `kWarpSize` / `SCAMP_FULL_WARP_MASK` /
  `kMinWarpSize` instead.
- The `build-hip` CI job compiles the HIP path for both a wave64 arch (gfx90a)
  and a wave32 arch (gfx1100); a change that breaks one width fails there.
- CI cannot launch AMD kernels (no GPU on the runners). Correctness on AMD
  still needs a manual run of the integration suite on a ROCm GPU, the same way
  CUDA changes need a manual GPU run (see CONTRIBUTING.md, Testing).
