// CMP27L (Saber's own, 2026-10-28): the R2 flush's tree advance for EVERY GDN layer in one launch
// -- the `--engine-b1-arms adv` arm's code object. Its own object so `p27_tree_gdn` keeps its
// bytes. `tg_advance` and the body are `p27_tree_gdn.hip`'s line for line; built -ffp-contract=off
// like it (saber-kbuild's `p27_tree_`).
#include <hip/hip_runtime.h>
#include "sgdn_prims.h"

#define TG_K   128              // head_k
#define TG_V   128              // head_v
#define TG_HK   64              // k elements per thread (two threads per v row)
#define TG_TM   10              // rows a sequence
#define TG_REC_U    0
#define TG_REC_K    (TG_TM * TG_V)
#define TG_REC_P    (TG_REC_K + TG_TM * TG_K)
#define TG_REC_A    (TG_REC_P + 16)

// p27_tree_gdn.hip's tg_advance, verbatim.
__device__ __forceinline__ bool tg_advance(float* h, const float* __restrict__ rec, int acc,
                                           int row, int k0) {
  if (acc < 0 || acc >= TG_TM) return false;
  const float Pa = rec[TG_REC_P + acc];
  const unsigned wa = (unsigned)((const int*)rec)[TG_REC_A + acc];
  const float ea = __expf(Pa);
#pragma unroll
  for (int j = 0; j < TG_HK; ++j) h[j] *= ea;
#pragma unroll 1
  for (int c = 0; c <= acc; ++c) {
    if (!((wa >> c) & 1u)) continue;
    const float coef = __expf(Pa - rec[TG_REC_P + c]) * rec[TG_REC_U + c * TG_V + row];
    const float* kc = rec + TG_REC_K + c * TG_K + k0;
#pragma unroll
    for (int j = 0; j < TG_HK; j += 4) {
      const float4 w = *(const float4*)(kc + j);
      h[j] += coef * w.x; h[j + 1] += coef * w.y; h[j + 2] += coef * w.z; h[j + 3] += coef * w.w;
    }
  }
  return true;
}

// CMP27L (Saber's own, 2026-10-28): the flush's advance for EVERY GDN layer in ONE launch. Grid
// (layers, H, sequences): `state_base` [layers] / `ts_base` [layers] are each layer's state pool
// and record (the same strides in every layer); the body is `p27_gdn_tree_advance_impl`'s, line
// for line, at (sequence blockIdx.z, head blockIdx.y), so every stored byte is the per-layer
// launch's.
extern "C" __global__ __launch_bounds__(TG_V * 2)
void p27_gdn_tree_advance_layers_f16s_k128_v128(
    const unsigned long long* __restrict__ state_base, const unsigned long long* __restrict__ ts_base,
    long st_slot, long st_head, const int* __restrict__ slot, const int* __restrict__ slot_out,
    long ts_seq, long ts_head, const int* __restrict__ acc) {
  unsigned short* __restrict__ state = (unsigned short*)state_base[blockIdx.x];
  const float* __restrict__ ts = (const float*)ts_base[blockIdx.x];
  const int n = blockIdx.z, hv = blockIdx.y;
  const int si = slot[n], so = slot_out[n];
  if (si <= 0 || so <= 0) return;
  const int tid = threadIdx.x, row = tid >> 1, k0 = (tid & 1) * TG_HK;
  float h[TG_HK];
  const size_t off = (size_t)hv * st_head + (size_t)row * TG_K + k0;
  sgdn::state_load<sgdn::kStateF16, TG_HK>(h, state + (size_t)si * st_slot + off);
  const float* rec = ts + (size_t)n * ts_seq + (size_t)hv * ts_head;
  const bool moved = tg_advance(h, rec, acc[n], row, k0);
  if (moved || so != si) sgdn::state_store<sgdn::kStateF16, TG_HK>(state + (size_t)so * st_slot + off, h);
}
