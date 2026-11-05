// Saber paged attention, CLEANROOM-2 (CR2-2c): the FN124 QSA fp8 gather prefill with the fully-masked sub-tile skip, its negative control, and a word-diff counter.
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/flashnext_qsa124.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_prefill.h"

namespace {
using namespace sattn;
// Skip::Wave: an m-tile is skipped when no row of the wave sees any of its keys (the identity).
// Skip::HalfVote: the negative control -- only the first 8 keys of the m-tile vote, so it must differ.
template <Skip SK>
using SkipPrefill = PrefillSpec<12, true, false, false, true, SK, false>;
}  // namespace

extern "C" __global__ void __launch_bounds__(768)
fn124_qsa_attn_prefill_h256_gqa12_fp8kv_skip(sattn::Params args, sattn::UnionKeys qa) {
  prefill_body<SkipPrefill<Skip::Wave>>(args, qa);
}

extern "C" __global__ void __launch_bounds__(768)
fn124_qsa_attn_prefill_h256_gqa12_fp8kv_skipctl(sattn::Params args, sattn::UnionKeys qa) {
  prefill_body<SkipPrefill<Skip::HalfVote>>(args, qa);
}

// Adds the number of 32-bit words where a and b differ (n words) to *out: a grid-strided count per lane, an
// XOR-butterfly sum over the wave, then one atomic per wave that saw a difference.
extern "C" __global__ void __launch_bounds__(256)
fn124_cmp_u32(const uint32_t* a, const uint32_t* b, long n, uint32_t* out) {
  uint32_t diff = 0;
  const long step = (long)gridDim.x * blockDim.x;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) diff += a[i] != b[i] ? 1u : 0u;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) diff += __shfl_xor(diff, m, 32);
  if ((threadIdx.x & 31) == 0 && diff) atomicAdd(out, diff);
}
