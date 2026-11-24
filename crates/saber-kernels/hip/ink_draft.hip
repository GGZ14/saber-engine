// INK12: the Inkling 2 block drafter (`mtp.safetensors`, docs/INKLING2.md section 6) + the verify
// rollback of the 192 conv states. Saber's own arithmetic, f32, built -ffp-contract=off; gated
// against tools/capture/ink12/dspark_ref.py (the host f32 oracle) under an envelope (the sums run
// in another order than numpy's).
//
//   ink_dr_copy        dst[i] = src[i] (n f32; the taps of a pass into the drafter's tap planes)
//   ink_dr_mixnorm     o[t] = rmsnorm_g(sum_i a_i taps[i][t])            (the DFlare feature mix)
//   ink_dr_hnr         per (row, head): rmsnorm_g over 128 then rope (half split, theta 1e6) at
//                      position *ppos + add + row, in place
//   ink_dr_attn        one (block row, head): softmax over [ctx keys lo..p-1 ; the 8 block keys]
//                      (no mask: the block is bidirectional), GQA 3
//   ink_dr_silu_mul    g = silu(g) * u
//   ink_dr_gru         the Markov head's GRU step (rank 256) over [state ; w1[*ptok]]
//   ink_dr_fold        logits = softcap(base + fold_b + fold_w . latent) -> per-workgroup argmax
//   ink_dr_argmax      the partials' argmax (first index on ties) -> *tok
//   ink_dr_hist        hist = [state (3 rows) ; x (T rows)] of one conv (the verify's history)
//   ink_dr_commit      state = hist rows keep .. keep + 2 for every logged conv (the rollback)
#include <hip/hip_runtime.h>

#define DR_D 2304
#define DR_HD 128
#define DR_MAXS 4224

__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

__device__ __forceinline__ float wsum32(float v) {
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor(v, o, 32);
  return v;
}

// 256-lane sum (8 waves of 32)
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

extern "C" __global__ __launch_bounds__(256) void ink_dr_copy(const float* __restrict__ src,
                                                              float* __restrict__ dst, long n) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) dst[i] = src[i];
}

extern "C" __global__ __launch_bounds__(256) void ink_dr_mixnorm(
    const float* __restrict__ taps, long plane, int row0, float a0, float a1, float a2, float a3,
    const unsigned short* __restrict__ gain, float* __restrict__ o, float eps) {
  __shared__ float m[DR_D];
  __shared__ float red[8];
  const long t = row0 + blockIdx.x;
  float ss = 0.f;
  for (int i = threadIdx.x; i < DR_D; i += 256) {
    const long b = t * DR_D + i;
    float v = taps[b] * a0;
    v = v + taps[plane + b] * a1;
    v = v + taps[2 * plane + b] * a2;
    v = v + taps[3 * plane + b] * a3;
    m[i] = v;
    ss = ss + v * v;
  }
  const float r = 1.0f / sqrtf(bsum256(ss, red) / (float)DR_D + eps);
  for (int i = threadIdx.x; i < DR_D; i += 256)
    o[(long)blockIdx.x * DR_D + i] = (m[i] * r) * bf2f(gain[i]);
}

// grid (rows, heads), 64 lanes: lane j owns elements j and j + 64 (the rope pair)
extern "C" __global__ __launch_bounds__(64) void ink_dr_hnr(float* __restrict__ x, int ld,
                                                            const unsigned short* __restrict__ gain,
                                                            const int* __restrict__ ppos, int add,
                                                            float eps) {
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
  const double pos = (double)(*ppos + add + (int)blockIdx.x);
  const double inv = pow(1.0e6, -(double)j / 64.0);
  const double ang = pos * inv;
  const float c = (float)cos(ang), sn = (float)sin(ang);
  p[j] = y1 * c - y2 * sn;
  p[j + 64] = y2 * c + y1 * sn;
}

// grid (8 block rows, 24 heads), 256 lanes. kc / vc: the context cache [pos][8][128] f32 (rows by
// absolute position); bk / bv: the block's keys / values [8][ldb]; q / o: [8][ldq]
extern "C" __global__ __launch_bounds__(256) void ink_dr_attn(
    const float* __restrict__ q, int ldq, const float* __restrict__ kc, const float* __restrict__ vc,
    const float* __restrict__ bk, const float* __restrict__ bv, int ldb, const int* __restrict__ ppos,
    int window, float* __restrict__ o, int ldo, float scale) {
  __shared__ float qs[DR_HD];
  __shared__ float sc[DR_MAXS];
  __shared__ float red[8];
  __shared__ float acc2[2][DR_HD];
  const int row = blockIdx.x, h = blockIdx.y, kvh = h / 3, tid = threadIdx.x;
  const int p = *ppos;
  const int lo = window > 0 ? max(0, p - window) : 0;
  const int nctx = p - lo, S = nctx + 8;
  if (tid < DR_HD) qs[tid] = q[(long)row * ldq + h * DR_HD + tid];
  __syncthreads();
  float mx = -INFINITY;
  for (int j = tid; j < S; j += 256) {
    const float* kr = j < nctx ? kc + ((long)(lo + j) * 8 + kvh) * DR_HD
                               : bk + (long)(j - nctx) * ldb + kvh * DR_HD;
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
                               : bv + (long)(j - nctx) * ldb + kvh * DR_HD;
    a = a + sc[j] * vr[d];
  }
  acc2[half][d] = a;
  __syncthreads();
  if (tid < DR_HD) o[(long)row * ldo + h * DR_HD + tid] = (acc2[0][tid] + acc2[1][tid]) / sum;
}

extern "C" __global__ __launch_bounds__(256) void ink_dr_silu_mul(float* __restrict__ g,
                                                                  const float* __restrict__ u, long n) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const float x = g[i];
    g[i] = (x / (1.0f + expf(-x))) * u[i];
  }
}

// grid 256 (one state index j a workgroup), 256 lanes: the three 512-long rows j, 256 + j, 512 + j
// of W[:, :512] over [s ; w1[tok]], + ctx A row (out part) + ctx B (the f_rnn part) + bias
extern "C" __global__ __launch_bounds__(256) void ink_dr_gru(
    const float* __restrict__ s_in, const unsigned short* __restrict__ w1, const int* __restrict__ ptok,
    const unsigned short* __restrict__ wlo, int ldw, const float* __restrict__ ca,
    const float* __restrict__ cb, const unsigned short* __restrict__ bias, float* __restrict__ s_out,
    float* __restrict__ lat) {
  __shared__ float red[8];
  const int j = blockIdx.x, tid = threadIdx.x;
  const long tok = *ptok;
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
    s_out[j] = gate * s_in[j] + (1.0f - gate) * cand;
    lat[j] = tanhf(raw[2]);
  }
}

// grid ceil(V / 64), 256 lanes = 8 waves x 8 rows: lane l of a wave sums 8 of the 256 columns
extern "C" __global__ __launch_bounds__(256) void ink_dr_fold(
    const unsigned short* __restrict__ fw, const unsigned short* __restrict__ fb,
    const float* __restrict__ base, const float* __restrict__ lat, int V, float cap,
    float* __restrict__ part) {
  __shared__ float ls[256];
  __shared__ float bv[8];
  __shared__ int bi[8];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  ls[tid] = lat[tid];
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
    float b = bv[0];
    int ib = bi[0];
    for (int i = 1; i < 8; ++i)
      if (bv[i] > b || (bv[i] == b && bi[i] < ib)) {
        b = bv[i];
        ib = bi[i];
      }
    part[2 * blockIdx.x] = b;
    part[2 * blockIdx.x + 1] = __int_as_float(ib);
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_dr_argmax(const float* __restrict__ part, int n,
                                                                int* __restrict__ tok) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  const int tid = threadIdx.x;
  float b = -INFINITY;
  int ib = 0x7fffffff;
  for (int i = tid; i < n; i += 256) {
    const float v = part[2 * i];
    const int k = __float_as_int(part[2 * i + 1]);
    if (v > b || (v == b && k < ib)) {
      b = v;
      ib = k;
    }
  }
  bv[tid] = b;
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
  if (tid == 0) *tok = bi[0];
}

// grid (ceil(W / 256), 3 + T): hist row r < 3 = st row r, else x row r - 3
extern "C" __global__ __launch_bounds__(256) void ink_dr_hist(const float* __restrict__ st,
                                                              const float* __restrict__ x, int ldx,
                                                              int W, float* __restrict__ hist) {
  const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
  if (c >= W) return;
  hist[(long)r * W + c] = r < 3 ? st[(long)r * W + c] : x[(long)(r - 3) * ldx + c];
}

// grid (ceil(maxW / 256), n): entry e = (hist, dst, W) as three u64 words
extern "C" __global__ __launch_bounds__(256) void ink_dr_commit(const unsigned long long* __restrict__ tab,
                                                                int keep) {
  const unsigned long long* e = tab + 3 * blockIdx.y;
  const float* hist = (const float*)e[0];
  float* dst = (float*)e[1];
  const int W = (int)e[2];
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= W) return;
  for (int r = 0; r < 3; ++r) dst[(long)r * W + c] = hist[(long)(keep + r) * W + c];
}

// the gate's rows: lg[v] = softcap(base + fold_b + fold_w . latent) (ink_dr_fold's arithmetic)
extern "C" __global__ __launch_bounds__(256) void ink_dr_fold_rows(
    const unsigned short* __restrict__ fw, const unsigned short* __restrict__ fb,
    const float* __restrict__ base, const float* __restrict__ lat, int V, float cap,
    float* __restrict__ out) {
  __shared__ float ls[256];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  ls[tid] = lat[tid];
  __syncthreads();
  for (int k = 0; k < 8; ++k) {
    const int v = blockIdx.x * 64 + w * 8 + k;
    if (v >= V) break;
    const unsigned short* row = fw + (long)v * 256;
    float d = 0.f;
    for (int e = 0; e < 8; ++e) d = d + bf2f(row[lane * 8 + e]) * ls[lane * 8 + e];
    d = wsum32(d);
    const float lg = (base[v] + bf2f(fb[v])) + d;
    if (lane == 0) out[v] = tanhf(lg / cap) * cap;
  }
}
