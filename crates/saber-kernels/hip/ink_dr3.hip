// INK20: the Inkling 2 block drafter for B sequences in ONE pass (docs/INKLING2.md section 9 INK20).
// Saber's own; its own object so ink_draft / ink_dr2 keep their bytes. Each body is its ink_draft /
// ink_dr2 parent's arithmetic in the parent's order with a SEQUENCE index added: sequence b's block
// rows are rows b*8 .. b*8 + 7 of the pass, its anchor position ppos[b], its context cache the slot
// slots[b]'s (`cstride` bytes / 4 a slot), its GRU state / latent / partials / tokens at a per-sequence
// stride. (The drafter's numerics only move acceptance: gated on the MTP law + acceptance.)
//
//   ink_dr3_hnr          ink_dr_hnr, position ppos[row / rows] + add + row % rows
//   ink_dr3_attn         ink_dr_attn, sequence row / 8's context + block keys / values
//   ink_dr3_gather       dst[b][i] = src[slots[b]][i] (the f_rnn rows of the pass's slots)
//   ink_dr3_gru          ink_dr_gru, grid (256, B)
//   ink_dr3_fold         ink_dr_fold, grid (ceil(V / 64), B)
//   ink_dr3_argmax_map   ink_dr_argmax_map, grid B
#include <hip/hip_runtime.h>

#define DR_D 2304
#define DR_HD 128
#define DR_MAXS 4224

__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

__device__ __forceinline__ float wsum32(float v) {
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor(v, o, 32);
  return v;
}

__device__ __forceinline__ float bsum256(float v, float* red) {
  v = wsum32(v);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = 0.f;
  for (int i = 0; i < 8; ++i) s = s + red[i];
  return s;
}

__device__ __forceinline__ float bmax256(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
  for (int i = 1; i < 8; ++i) s = fmaxf(s, red[i]);
  return s;
}

// grid (B * rows, heads), 64 lanes
extern "C" __global__ __launch_bounds__(64) void ink_dr3_hnr(float* __restrict__ x, int ld,
                                                             const unsigned short* __restrict__ gain,
                                                             const int* __restrict__ ppos, int add,
                                                             int rows, float eps) {
  __shared__ float red[2];
  const int j = threadIdx.x;
  float* p = x + (long)blockIdx.x * ld + blockIdx.y * DR_HD;
  const float v1 = p[j], v2 = p[j + 64];
  float s = wsum32(v1 * v1 + v2 * v2);
  if ((j & 31) == 0) red[j >> 5] = s;
  __syncthreads();
  s = red[0] + red[1];
  const float r = 1.0f / sqrtf(s / 128.0f + eps);
  const float y1 = (v1 * r) * bf2f(gain[j]);
  const float y2 = (v2 * r) * bf2f(gain[j + 64]);
  const int b = (int)blockIdx.x / rows, row = (int)blockIdx.x % rows;
  const double pos = (double)(ppos[b] + add + row);
  const double inv = pow(1.0e6, -(double)j / 64.0);
  const double ang = pos * inv;
  const float c = (float)cos(ang), sn = (float)sin(ang);
  p[j] = y1 * c - y2 * sn;
  p[j + 64] = y2 * c + y1 * sn;
}

// grid (B * 8 block rows, 24 heads), 256 lanes
extern "C" __global__ __launch_bounds__(256) void ink_dr3_attn(
    const float* __restrict__ q, int ldq, const float* __restrict__ kc0, const float* __restrict__ vc0,
    long cstride, const int* __restrict__ slots, const float* __restrict__ bk,
    const float* __restrict__ bv, int ldb, const int* __restrict__ ppos, int window,
    float* __restrict__ o, int ldo, float scale) {
  __shared__ float qs[DR_HD];
  __shared__ float sc[DR_MAXS];
  __shared__ float red[8];
  __shared__ float acc2[2][DR_HD];
  const int row = blockIdx.x, h = blockIdx.y, kvh = h / 3, tid = threadIdx.x;
  const int b = row / 8;
  const float* kc = kc0 + (long)slots[b] * cstride;
  const float* vc = vc0 + (long)slots[b] * cstride;
  const int p = ppos[b];
  const int lo = window > 0 ? max(0, p - window) : 0;
  const int nctx = p - lo, S = nctx + 8;
  if (tid < DR_HD) qs[tid] = q[(long)row * ldq + h * DR_HD + tid];
  __syncthreads();
  float mx = -INFINITY;
  for (int j = tid; j < S; j += 256) {
    const float* kr = j < nctx ? kc + ((long)(lo + j) * 8 + kvh) * DR_HD
                               : bk + (long)(b * 8 + j - nctx) * ldb + kvh * DR_HD;
    float d = 0.f;
    for (int e = 0; e < DR_HD; ++e) d = d + qs[e] * kr[e];
    d = d * scale;
    sc[j] = d;
    mx = fmaxf(mx, d);
  }
  mx = bmax256(mx, red);
  float sum = 0.f;
  for (int j = tid; j < S; j += 256) {
    const float e = expf(sc[j] - mx);
    sc[j] = e;
    sum = sum + e;
  }
  sum = bsum256(sum, red);
  __syncthreads();
  const int d = tid & 127, half = tid >> 7;
  float a = 0.f;
  for (int j = half; j < S; j += 2) {
    const float* vr = j < nctx ? vc + ((long)(lo + j) * 8 + kvh) * DR_HD
                               : bv + (long)(b * 8 + j - nctx) * ldb + kvh * DR_HD;
    a = a + sc[j] * vr[d];
  }
  acc2[half][d] = a;
  __syncthreads();
  if (tid < DR_HD) o[(long)row * ldo + h * DR_HD + tid] = (acc2[0][tid] + acc2[1][tid]) / sum;
}

// grid (ceil(n / 256), B)
extern "C" __global__ __launch_bounds__(256) void ink_dr3_gather(const float* __restrict__ src,
                                                                 long stride,
                                                                 const int* __restrict__ slots,
                                                                 float* __restrict__ dst, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x, b = blockIdx.y;
  if (i < n) dst[(long)b * n + i] = src[(long)slots[b] * stride + i];
}

// grid (256, B): ink_dr_gru for sequence b (state / latent [B][256], tokens [B][16], ca rows at
// ca_stride, cb rows of 768)
extern "C" __global__ __launch_bounds__(256) void ink_dr3_gru(
    const float* __restrict__ s_in0, const unsigned short* __restrict__ w1,
    const int* __restrict__ ptok0, const unsigned short* __restrict__ wlo, int ldw,
    const float* __restrict__ ca0, long ca_stride, const float* __restrict__ cb0,
    const unsigned short* __restrict__ bias, float* __restrict__ s_out0, float* __restrict__ lat0) {
  __shared__ float red[8];
  const int j = blockIdx.x, tid = threadIdx.x, b = blockIdx.y;
  const float* s_in = s_in0 + b * 256;
  const float* ca = ca0 + b * ca_stride;
  const float* cb = cb0 + b * 768;
  const long tok = ptok0[b * 16];
  const float in0 = s_in[tid], in1 = bf2f(w1[tok * 256 + tid]);
  float r[3];
  for (int g = 0; g < 3; ++g) {
    const unsigned short* wr = wlo + (long)(g * 256 + j) * ldw;
    r[g] = bsum256(in0 * bf2f(wr[tid]) + in1 * bf2f(wr[256 + tid]), red);
  }
  if (tid == 0) {
    float raw[3];
    for (int g = 0; g < 3; ++g) {
      const int k = g * 256 + j;
      raw[g] = ((r[g] + ca[k]) + cb[k]) + bf2f(bias[k]);
    }
    const float gate = 1.0f / (1.0f + expf(-raw[0]));
    const float cand = tanhf(raw[1]);
    s_out0[b * 256 + j] = gate * s_in[j] + (1.0f - gate) * cand;
    lat0[b * 256 + j] = tanhf(raw[2]);
  }
}

// grid (ceil(V / 64), B)
extern "C" __global__ __launch_bounds__(256) void ink_dr3_fold(
    const unsigned short* __restrict__ fw, const unsigned short* __restrict__ fb,
    const float* __restrict__ base0, long base_stride, const float* __restrict__ lat0, int V,
    float cap, float* __restrict__ part0, long part_stride) {
  __shared__ float ls[256];
  __shared__ float bv[8];
  __shared__ int bi[8];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, b = blockIdx.y;
  const float* base = base0 + b * base_stride;
  float* part = part0 + b * part_stride;
  ls[tid] = lat0[b * 256 + tid];
  __syncthreads();
  float best = -INFINITY;
  int besti = 0x7fffffff;
  for (int k = 0; k < 8; ++k) {
    const int v = blockIdx.x * 64 + w * 8 + k;
    if (v >= V) break;
    const unsigned short* row = fw + (long)v * 256;
    float d = 0.f;
    for (int e = 0; e < 8; ++e) d = d + bf2f(row[lane * 8 + e]) * ls[lane * 8 + e];
    d = wsum32(d);
    float lg = (base[v] + bf2f(fb[v])) + d;
    lg = tanhf(lg / cap) * cap;
    if (lg > best) {
      best = lg;
      besti = v;
    }
  }
  if (lane == 0) {
    bv[w] = best;
    bi[w] = besti;
  }
  __syncthreads();
  if (tid == 0) {
    float bb = bv[0];
    int ib = bi[0];
    for (int i = 1; i < 8; ++i)
      if (bv[i] > bb || (bv[i] == bb && bi[i] < ib)) {
        bb = bv[i];
        ib = bi[i];
      }
    part[2 * blockIdx.x] = bb;
    part[2 * blockIdx.x + 1] = __int_as_float(ib);
  }
}

// grid B
extern "C" __global__ __launch_bounds__(256) void ink_dr3_argmax_map(
    const float* __restrict__ part0, long part_stride, int n, const unsigned* __restrict__ map,
    int* __restrict__ tok0, int tok_stride) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  const int tid = threadIdx.x, b = blockIdx.x;
  const float* part = part0 + b * part_stride;
  float bb = -INFINITY;
  int ib = 0x7fffffff;
  for (int i = tid; i < n; i += 256) {
    const float v = part[2 * i];
    const int k = __float_as_int(part[2 * i + 1]);
    if (v > bb || (v == bb && k < ib)) {
      bb = v;
      ib = k;
    }
  }
  bv[tid] = bb;
  bi[tid] = ib;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) {
      const float v = bv[tid + s];
      const int k = bi[tid + s];
      if (v > bv[tid] || (v == bv[tid] && k < bi[tid])) {
        bv[tid] = v;
        bi[tid] = k;
      }
    }
    __syncthreads();
  }
  if (tid == 0) tok0[b * tok_stride] = (int)map[bi[0]];
}
