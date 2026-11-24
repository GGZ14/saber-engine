// INK33: the Inkling 2 block drafter's draft CONFIDENCE (docs/INKLING2.md section 9 INK33), for
// the confidence-gated draft depth (`--ink-mtp-gate`, an ARM, default off). Own object, so
// ink_draft / ink_dr2 keep their bytes. The drafter's numerics are free under the drawn-row walk
// (a draft moves acceptance only); these bodies are ink_dr_fold / ink_dr_argmax_map verbatim plus
// one quantity: the top-1 probability of the scored vocabulary, p1 = 1 / sum_v exp(lg_v - lg_max).
// The softcap bounds every logit to (-30, 30), so each workgroup sums exp(lg - 30) (in [e^-60, 1]:
// no overflow, no f32 underflow) and the reduction needs no running max.
//
//   ink_dr6_fold_p        ink_dr_fold + psum[blockIdx.x] = sum over its 64 rows of exp(lg - cap)
//   ink_dr6_argmax_map_p  ink_dr_argmax_map + *p1 = exp(best - cap) / sum_i psum[i]
#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

__device__ __forceinline__ float wsum32(float v) {
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor(v, o, 32);
  return v;
}

// grid ceil(V / 64), 256 lanes = 8 waves x 8 rows: lane l of a wave sums 8 of the 256 columns
extern "C" __global__ __launch_bounds__(256) void ink_dr6_fold_p(
    const unsigned short* __restrict__ fw, const unsigned short* __restrict__ fb,
    const float* __restrict__ base, const float* __restrict__ lat, int V, float cap,
    float* __restrict__ part, float* __restrict__ psum) {
  __shared__ float ls[256];
  __shared__ float bv[8];
  __shared__ int bi[8];
  __shared__ float bs[8];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  ls[tid] = lat[tid];
  __syncthreads();
  float best = -INFINITY;
  int besti = 0x7fffffff;
  float se = 0.f;
  for (int k = 0; k < 8; ++k) {
    const int v = blockIdx.x * 64 + w * 8 + k;
    if (v >= V) break;
    const unsigned short* row = fw + (long)v * 256;
    float d = 0.f;
    for (int e = 0; e < 8; ++e) d = d + bf2f(row[lane * 8 + e]) * ls[lane * 8 + e];
    d = wsum32(d);
    float lg = (base[v] + bf2f(fb[v])) + d;
    lg = tanhf(lg / cap) * cap;
    se = se + __expf(lg - cap);
    if (lg > best) {
      best = lg;
      besti = v;
    }
  }
  if (lane == 0) {
    bv[w] = best;
    bi[w] = besti;
    bs[w] = se;
  }
  __syncthreads();
  if (tid == 0) {
    float b = bv[0];
    int ib = bi[0];
    float s = bs[0];
    for (int i = 1; i < 8; ++i) {
      s = s + bs[i];
      if (bv[i] > b || (bv[i] == b && bi[i] < ib)) {
        b = bv[i];
        ib = bi[i];
      }
    }
    part[2 * blockIdx.x] = b;
    part[2 * blockIdx.x + 1] = __int_as_float(ib);
    psum[blockIdx.x] = s;
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_dr6_argmax_map_p(
    const float* __restrict__ part, int n, const unsigned* __restrict__ map, int* __restrict__ tok,
    const float* __restrict__ psum, float cap, float* __restrict__ p1) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  __shared__ float bs[256];
  const int tid = threadIdx.x;
  float b = -INFINITY;
  int ib = 0x7fffffff;
  float s = 0.f;
  for (int i = tid; i < n; i += 256) {
    const float v = part[2 * i];
    const int k = __float_as_int(part[2 * i + 1]);
    s = s + psum[i];
    if (v > b || (v == b && k < ib)) {
      b = v;
      ib = k;
    }
  }
  bv[tid] = b;
  bi[tid] = ib;
  bs[tid] = s;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (tid < st) {
      const float v = bv[tid + st];
      const int k = bi[tid + st];
      bs[tid] = bs[tid] + bs[tid + st];
      if (v > bv[tid] || (v == bv[tid] && k < bi[tid])) {
        bv[tid] = v;
        bi[tid] = k;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    *tok = (int)map[bi[0]];
    *p1 = __expf(bv[0] - cap) / bs[0];
  }
}
