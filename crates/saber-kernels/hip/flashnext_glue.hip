// Flash-Next (qwen4_exp) attention / PLE / MTP glue kernels (workstream FN6).
//
// NOT sliced from a reference: upstream runs these ops as torch compositions and Triton kernels
// (`qwen4_exp/amd/{qsa.py, indexer_qsa.py, ops/qsa.py, ple_layer.py, mtp.py}` in
// `localhost/saber-base:0.29.0-test2`). The math is upstream's, written down op by op in
// docs/LEDGER.md "FN6" and in `tools/capture/fn6/fn6_ref.py` (the numpy SPEC every kernel here is
// gated against bit for bit, plus float64 / torch-bf16 transcriptions of upstream it is bounded
// against). Inside each op the order is Saber's own and fixed:
//
//   fn_kv_append_ring_h256_bf16   bf16 K|V pages (the reference's [block][kv head][slot][K 256 | V 256]
//                                 layout) + the indexer's raw-key ring (the last R rows of each
//                                 sequence's step) -- copies, no arithmetic
//   fn_idx_q_compress_h128        the QSA indexer: 4 q heads Gemma-normed + roped (for FN9's
//                                 scorer), and at every row that completes a group of 4 the
//                                 mean-pooled raw key (ring + this step's rows), Gemma-normed and
//                                 roped at the group's FIRST position -> the compressed cache row
//   fn_ple_gate_{x,combine}       PLE: grouped norms of key / query, the per-stream gate
//                                 sigmoid(sign(g) sqrt|g|), gated value, the conv's input norm
//                                 (`combine` also materializes the delayed HC combine = FN5's
//                                 fn_hc_combine bytes, the stream the PLE adds into)
//   fn_ple_conv                   the dilated short conv (kernel 4, dilation 3), silu, the PLE
//                                 output and its add into the 10,240-wide stream, and the conv
//                                 state's next history (ping-pong slots, a drop count for rolled
//                                 back verify rows)
//   fn_ple_rows_fp8               the GATED fp8 table option: e4m3 rows x one table scale -> bf16
//   fn_mtp_norms                  MTP's pre_fc_norm_embedding (2,560) and pre_fc_norm_hidden
//                                 (ONE norm over the whole 10,240 stream), one launch
//   fn_mtp_add                    x[t, s] = bf16(fc_embedding[t] + fc_hidden[t, s])
//   fn_moe_silu_mul_bf16_t256     the MTP experts' SiluAndMul, bf16 out, expert-major rows
//
// Built with -ffp-contract=off (saber-kbuild's EXTRA_FLAGS): every fp32 multiply and add rounds
// on its own, so the host spec reproduces each one as its own IEEE op. Loads past a live range are
// CLAMPED, never predicated (the predicated-staging law).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef unsigned int u32;
union G8 { uint4 u4; __bf16 h[8]; };
union G4 { uint2 u2; __bf16 h[4]; };

__device__ __forceinline__ float bf(float x) { return (float)(__bf16)x; }   // one RNE rounding

// attn_glue.hip's rsqrt_exact: correctly rounded 1/sqrt, via fp64 (ATen's torch.rsqrt agrees on
// every input; v_rsq_f32 does not).
__device__ __forceinline__ float rsqrt_exact(float v) { return (float)(1.0 / sqrt((double)v)); }

// attn_glue.hip's bfmul: Triton's bf16 rope multiply (v_dot2_bf16_bf16 with a zeroed lane)
// TRUNCATES the exact product of two bf16 values to bf16.
__device__ __forceinline__ float bfmul(float a, float b) {
  return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, a * b) & 0xffff0000u);
}

// flashnext_hc.hip's fn_hexp / fn_sigmoid / fn_silu, VERBATIM (hc_ref.hexp op for op): clamp,
// Cody-Waite with a 16-bit ln2_hi, degree-7 Horner, ldexpf.
__device__ __forceinline__ float fn_hexp(float t) {
  t = fminf(fmaxf(t, -80.f), 80.f);
  const float kn = rintf(t * (float)1.4426950408889634);
  float r = t - kn * (float)0.693145751953125;
  r = r - kn * (float)1.4286068203094173e-06;
  float p = (float)(1.0 / 5040);
  p = p * r + (float)(1.0 / 720);
  p = p * r + (float)(1.0 / 120);
  p = p * r + (float)(1.0 / 24);
  p = p * r + (float)(1.0 / 6);
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  return ldexpf(p, (int)kn);
}
__device__ __forceinline__ float fn_sigmoid(float v) { return 1.f / (1.f + fn_hexp(-v)); }
__device__ __forceinline__ float fn_silu(float v) { return v * fn_sigmoid(v); }

#define FG_H 2560             // hidden per stream
#define FG_S 4                // hc_count
#define FG_W (FG_H * FG_S)    // 10,240
#define FG_EPS 1e-6f

// =============================================================================== 1. KV APPEND
// grid (m, nkv), 128 threads. Row t: `slot[t] < 0` = padding (nothing written, the reference server's marker);
// else block = slot / 16, offset = slot % 16, and head h's K (256 bf16) and V (256 bf16) land at
// kv + block * block_stride + h * head_stride + offset * page_stride (elements), K then V -- the
// saber paged layout `saber_attn_*` reads (`kv_block_stride`, `kv_head_stride`, 2 * head_dim per
// slot). kv head 0's workgroup also stores the INDEXER's raw key row (128 bf16 at `ikey`) and its
// three rope positions into the ring when `ring_slot[t] >= 0` (the host's circular slot map: the
// last R rows of each sequence's step, docs/EXEC.md section 13).
extern "C" __global__ __launch_bounds__(128) void fn_kv_append_ring_h256_bf16(
    const __bf16 *__restrict__ key, long key_stride, const __bf16 *__restrict__ value,
    long value_stride, __bf16 *__restrict__ kv, const long *__restrict__ slot, long block_stride,
    long page_stride, long head_stride, const __bf16 *__restrict__ ikey, long ikey_stride,
    __bf16 *__restrict__ ring, long *__restrict__ ring_pos, const long *__restrict__ ring_slot,
    const long *__restrict__ pos, int m) {
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const long s = slot[t];
  if (s >= 0) {
    const long blk = s / 16, off = s % 16;
    const u32 *kp = (const u32 *)(key + (long)t * key_stride + (long)h * 256);
    const u32 *vp = (const u32 *)(value + (long)t * value_stride + (long)h * 256);
    u32 *dst = (u32 *)(kv + blk * block_stride + (long)h * head_stride + off * page_stride);
    dst[tid] = kp[tid];
    dst[128 + tid] = vp[tid];
  }
  if (h != 0) return;
  const long rs = ring_slot[t];
  if (rs < 0) return;
  if (tid < 64) ((u32 *)(ring + rs * 128))[tid] = ((const u32 *)(ikey + (long)t * ikey_stride))[tid];
  if (tid < 3) ring_pos[rs * 3 + tid] = pos[(long)tid * m + t];
}

// =============================================================================== 2. INDEXER
// One WAVE per (row, head) of 128: lane l owns elements [4l, 4l + 4). attn_glue's
// `qk_norm_rope_wave_impl` arithmetic at HD 128 / RD 64 (the drafter's QKR128 order, RED 5 --
// ATen's): sum of squares ((a0 + a1) + a2) + a3 per lane (one vec4 round at HD 128), then the
// ADJACENT tree `v += shfl_down(v, o)` for o = 1..16; inv = rsqrt_exact(sum / 128 + eps);
// n = bf16((x * inv) * (w + 1)); rope over dims [0, 64) (neox pairs i, i + 32, lanes 0..15; the
// partner is lane ^ 8): o = rtz(n * c) -/+ rtz(n' * s), rounded once to bf16; mrope interleaved
// [11, 11, 10] (the section of rotary index i: h if i % 3 == 1 and i <= 3 sec_h, w if i % 3 == 2
// and i <= 3 sec_w, else t) at the given position row.
__device__ __forceinline__ void fg_norm_rope128(float x[4], const __bf16 *__restrict__ w,
                                                const __bf16 *__restrict__ cos_sin,
                                                const long p3[3], int lane, int sec_h,
                                                int sec_w) {
  float a0 = x[0] * x[0], a1 = x[1] * x[1], a2 = x[2] * x[2], a3 = x[3] * x[3];
  float sum = ((a0 + a1) + a2) + a3;
  for (int o = 1; o < 32; o <<= 1) sum += __shfl_down(sum, o);
  float inv = 0.f;
  if (lane == 0) inv = rsqrt_exact(__shfl(sum, 0) / 128.f + FG_EPS);
  inv = __shfl(inv, 0);
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n[j] = bf((x[j] * inv) * ((float)w[4 * lane + j] + 1.f));
  float other[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) other[j] = __shfl_xor(n[j], 8);
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = (lane & 7) * 4 + j;
      const int r = i % 3;
      const int sec = (r == 1 && i <= 3 * sec_h) ? 1 : ((r == 2 && i <= 3 * sec_w) ? 2 : 0);
      const long p = p3[sec];
      const float c = (float)cos_sin[p * 64 + i];
      const float sn = (float)cos_sin[p * 64 + 32 + i];
      const float t = bfmul(other[j], sn);
      n[j] = bfmul(n[j], c) + (lane < 8 ? -t : t);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = n[j];
}

// grid (m), 5 waves: waves 0..3 = the four q heads -> q_out [m, 4, 128]; wave 4 = the group row.
// Row metadata (host-built, the step's layout): row_seq[t] = its sequence, row_lpos[t] = its
// logical position; per sequence seq_row0 / seq_pos0 = the step's first row and its position,
// ring_row0[s] = the sequence's ring base row (R rows each), block_table [S][max_blocks].
// A row completes a group iff (lpos + 1) % 4 == 0; the group's members are lpos-3..lpos, taken
// from THIS step's raw keys (qk[., 512..640)) when >= seq_pos0, else from the ring (written by
// an earlier step's append; this launch never writes the ring, so no member is read while it is
// written). acc = (((0 + k0) + k1) + k2) + k3 in fp32, pooled = bf16(acc * 0.25) (upstream's
// `accumulator / 4` stored into a bf16 tensor), then k_layernorm + rope at the FIRST member's
// three positions -> comp[(page * 4 + g % 4) * 128], page = block_table[seq][g / 4] (4 groups of
// 4 tokens per 16-token KV block: the compressed cache shares the main KV block table).
extern "C" __global__ __launch_bounds__(160) void fn_idx_q_compress_h128(
    const __bf16 *__restrict__ qk, long qk_stride, const __bf16 *__restrict__ qw,
    const __bf16 *__restrict__ kw, const __bf16 *__restrict__ cos_sin, const long *__restrict__ pos,
    const int *__restrict__ row_seq, const long *__restrict__ row_lpos,
    const int *__restrict__ seq_row0, const long *__restrict__ seq_pos0,
    const int *__restrict__ ring_row0, const __bf16 *__restrict__ ring,
    const long *__restrict__ ring_pos, int ring_rows, const int *__restrict__ block_table,
    int max_blocks, __bf16 *__restrict__ q_out, __bf16 *__restrict__ comp, int m, int sec_h,
    int sec_w) {
  const int t = blockIdx.x, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (wave < 4) {
    G4 v;
    v.u2 = *(const uint2 *)(qk + (long)t * qk_stride + wave * 128 + 4 * lane);
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = (float)v.h[j];
    const long p3[3] = {pos[t], pos[(long)m + t], pos[2 * (long)m + t]};
    fg_norm_rope128(x, qw, cos_sin, p3, lane, sec_h, sec_w);
    G4 o;
#pragma unroll
    for (int j = 0; j < 4; ++j) o.h[j] = (__bf16)x[j];
    *(uint2 *)(q_out + ((long)t * 4 + wave) * 128 + 4 * lane) = o.u2;
    return;
  }
  const long lp = row_lpos[t];
  if (lp < 3 || ((lp + 1) & 3) != 0) return;
  const int s = row_seq[t];
  const long p0 = seq_pos0[s];
  const int r0 = seq_row0[s];
  const long rbase = (long)ring_row0[s] * ring_rows;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < 4; ++j) {
    const long pj = lp - 3 + j;
    const __bf16 *src = pj >= p0 ? qk + (long)(r0 + (pj - p0)) * qk_stride + 512
                                 : ring + (rbase + pj % ring_rows) * 128;
    G4 v;
    v.u2 = *(const uint2 *)(src + 4 * lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = acc[e] + (float)v.h[e];
  }
  float x[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) x[e] = bf(acc[e] * 0.25f);
  const long pf = lp - 3;
  long p3[3];
  if (pf >= p0) {
    const long rr = r0 + (pf - p0);
    p3[0] = pos[rr]; p3[1] = pos[(long)m + rr]; p3[2] = pos[2 * (long)m + rr];
  } else {
    const long rr = rbase + pf % ring_rows;
    p3[0] = ring_pos[rr * 3]; p3[1] = ring_pos[rr * 3 + 1]; p3[2] = ring_pos[rr * 3 + 2];
  }
  fg_norm_rope128(x, kw, cos_sin, p3, lane, sec_h, sec_w);
  const long g = lp >> 2;
  const long page = block_table[(long)s * max_blocks + (g >> 2)];
  G4 o;
#pragma unroll
  for (int e = 0; e < 4; ++e) o.h[e] = (__bf16)x[e];
  *(uint2 *)(comp + (page * 4 + (g & 3)) * 128 + 4 * lane) = o.u2;
}

// =============================================================================== 3. PLE
// The grouped RMS norm of one 2,560 stream, one WAVE: lane l owns elements c * 256 + 8l + j
// (c 0..9, j 0..7), the sum of squares sequential in (c, j) order, then a 32-lane xor butterfly
// (o = 16, 8, 4, 2, 1); rr = rsqrt_exact(ss / 2560 + eps). Upstream's Qwen4ExpPLEGroupedNorm:
// out = bf16((x * rr) * (1 + w)) -- two products, fp32 (normalized, then the affine).
__device__ __forceinline__ float fg_wave_rr(const float v[10][8]) {
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < 10; ++c)
#pragma unroll
    for (int j = 0; j < 8; ++j) ss = ss + v[c][j] * v[c][j];
  for (int o = 16; o; o >>= 1) ss = ss + __shfl_xor(ss, o);
  return rsqrt_exact(ss / 2560.f + FG_EPS);
}

__device__ __forceinline__ void fg_ld8(const __bf16 *p, float o[8]) {
  G8 v;
  v.u4 = *(const uint4 *)p;
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = (float)v.h[j];
}
__device__ __forceinline__ void fg_st8(__bf16 *p, const float o[8]) {
  G8 v;
#pragma unroll
  for (int j = 0; j < 8; ++j) v.h[j] = (__bf16)o[j];
  *(uint4 *)p = v.u4;
}

// ONE workgroup per token, 8 waves (256 threads).
//   phase 1  waves 0..3: key stream s = wave -> kn = norm_key(key_s) (bf16, LDS);
//            waves 4..7: query stream s = wave - 4 -> the stream x_s (COMBINE: x_s =
//            bf16(res_s + blk * 2 sigmoid(inj_s / 4)), FN5's `fn_hc_combine` arithmetic, written
//            to `xs`), qn = norm_query(x_s) (bf16, LDS)
//   phase 2  waves 0..3: g_s = bf16(sum_i bf16(kn_i * qn_i)) in the lane order above + butterfly;
//            g2 = bf16(g * (1 / sqrt(2560))); a = bf16(sqrt(max(|g2|, bf16(1e-6))));
//            gate_s = bf16(sigmoid(sign(g2) * a))            (every torch bf16 op's rounding)
//   phase 3  waves 0..3: gv = bf16(gate_s * value) -> `gv`; xc = norm_conv(gv) -> `xc`
template <bool COMBINE>
__device__ __forceinline__ void fg_ple_gate(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ blk, const __bf16 *__restrict__ inj,
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    const __bf16 *__restrict__ wk, const __bf16 *__restrict__ wq, const __bf16 *__restrict__ wc,
    __bf16 *__restrict__ xs, __bf16 *__restrict__ gv, __bf16 *__restrict__ xc) {
  __shared__ __bf16 s_kn[FG_W];
  __shared__ __bf16 s_qn[FG_W];
  __shared__ float s_gate[FG_S];
  const int t = blockIdx.x, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int s = wave & 3;
  float v[10][8];
  if (wave < 4) {
    const __bf16 *src = key + (long)t * FG_W + s * FG_H;
#pragma unroll
    for (int c = 0; c < 10; ++c) fg_ld8(src + c * 256 + 8 * lane, v[c]);
  } else {
    const __bf16 *src = x + (long)t * FG_W + s * FG_H;
    if (COMBINE) {
      const float gain = 2.f * fn_sigmoid((float)inj[(long)t * 4 + s] * 0.25f);
#pragma unroll
      for (int c = 0; c < 10; ++c) {
        float r[8], b[8];
        fg_ld8(src + c * 256 + 8 * lane, r);
        fg_ld8(blk + (long)t * FG_H + c * 256 + 8 * lane, b);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[c][j] = bf(r[j] + b[j] * gain);
        fg_st8(xs + (long)t * FG_W + s * FG_H + c * 256 + 8 * lane, v[c]);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 10; ++c) fg_ld8(src + c * 256 + 8 * lane, v[c]);
    }
  }
  {
    const float rr = fg_wave_rr(v);
    const __bf16 *w = (wave < 4 ? wk : wq) + s * FG_H;
    __bf16 *dst = (wave < 4 ? s_kn : s_qn) + s * FG_H;
#pragma unroll
    for (int c = 0; c < 10; ++c) {
      float wf[8], o[8];
      fg_ld8(w + c * 256 + 8 * lane, wf);
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = bf((v[c][j] * rr) * (1.f + wf[j]));
      fg_st8(dst + c * 256 + 8 * lane, o);
    }
  }
  __syncthreads();
  if (wave >= 4) return;
  {
    float d = 0.f;
#pragma unroll
    for (int c = 0; c < 10; ++c) {
      float a[8], b[8];
      fg_ld8(s_kn + s * FG_H + c * 256 + 8 * lane, a);
      fg_ld8(s_qn + s * FG_H + c * 256 + 8 * lane, b);
#pragma unroll
      for (int j = 0; j < 8; ++j) d = d + bf(a[j] * b[j]);
    }
    for (int o = 16; o; o >>= 1) d = d + __shfl_xor(d, o);
    const float g = bf(d);
    const float g2 = bf(g * (float)(1.0 / 50.596442562694072));   // 1 / sqrt(2560)
    const float mag = bf(sqrtf(fmaxf(fabsf(g2), (float)(__bf16)1e-6f)));
    const float sgn = g2 > 0.f ? 1.f : (g2 < 0.f ? -1.f : 0.f);
    if (lane == 0) s_gate[s] = bf(fn_sigmoid(sgn * mag));
  }
  // s_gate[s] is written and read by the same wave only.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  const float gate = s_gate[s];
#pragma unroll
  for (int c = 0; c < 10; ++c) {
    float val[8];
    fg_ld8(value + (long)t * FG_H + c * 256 + 8 * lane, val);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[c][j] = bf(gate * val[j]);
    fg_st8(gv + (long)t * FG_W + s * FG_H + c * 256 + 8 * lane, v[c]);
  }
  const float rr = fg_wave_rr(v);
  const __bf16 *w = wc + s * FG_H;
#pragma unroll
  for (int c = 0; c < 10; ++c) {
    float wf[8], o[8];
    fg_ld8(w + c * 256 + 8 * lane, wf);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = bf((v[c][j] * rr) * (1.f + wf[j]));
    fg_st8(xc + (long)t * FG_W + s * FG_H + c * 256 + 8 * lane, o);
  }
}

extern "C" __global__ __launch_bounds__(256) void fn_ple_gate_x(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    const __bf16 *__restrict__ wk, const __bf16 *__restrict__ wq, const __bf16 *__restrict__ wc,
    __bf16 *__restrict__ gv, __bf16 *__restrict__ xc) {
  fg_ple_gate<false>(x, nullptr, nullptr, key, value, wk, wq, wc, nullptr, gv, xc);
}

extern "C" __global__ __launch_bounds__(256) void fn_ple_gate_combine(
    const __bf16 *__restrict__ res, const __bf16 *__restrict__ blk, const __bf16 *__restrict__ inj,
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    const __bf16 *__restrict__ wk, const __bf16 *__restrict__ wq, const __bf16 *__restrict__ wc,
    __bf16 *__restrict__ xs, __bf16 *__restrict__ gv, __bf16 *__restrict__ xc) {
  fg_ple_gate<true>(res, blk, inj, key, value, wk, wq, wc, xs, gv, xc);
}

// The dilated short conv, its silu, the PLE output and the add into the stream.
// grid (m, 5), 256 threads, 8 channels a thread (5 x 256 x 8 = 10,240).
// Row t of sequence s (r = t - seq_row0[s], the step's rows of s are contiguous):
//   x(k) = xc[t - k]                         if r - k >= 0 (this step's conv input)
//        = st_in[s][H - drop[s] - (k - r)]  otherwise, when has_init[s] (else 0)
//   c    = bf16((((w0 x(9) + w1 x(6)) + w2 x(3)) + w3 x(0)))     (F.conv1d, kernel 4, dilation 3:
//                                                                 w[j] multiplies x(t - 3(3 - j)))
//   ple  = bf16(gv + bf16(silu(c)));  out = bf16(xs + ple)          (upstream: hidden + ple(...))
// The conv state is a HISTORY of H = 9 + spec rows per sequence, two slots (in / out, ping-pong):
// st_out[s][r'] = the conv input at position end - H + r' (end = the step's last position + 1),
// from this step's rows or, for the rows before it, from st_in (the row of chunk row 0 copies
// those). `drop[s]` = the rows of the previous step's history that were rejected (verify rows
// past the accepted prefix): the valid history then ends at H - 1 - drop.
extern "C" __global__ __launch_bounds__(256) void fn_ple_conv(
    const __bf16 *__restrict__ xc, const __bf16 *__restrict__ gv, const __bf16 *__restrict__ xs,
    const __bf16 *__restrict__ cw, const int *__restrict__ row_seq,
    const int *__restrict__ seq_row0, const int *__restrict__ seq_rows,
    const __bf16 *__restrict__ st_in, __bf16 *__restrict__ st_out, const int *__restrict__ drop,
    const int *__restrict__ has_init, int hist, __bf16 *__restrict__ out,
    __bf16 *__restrict__ ple) {
  const int t = blockIdx.x;
  const int ch = ((int)blockIdx.y * 256 + (int)threadIdx.x) * 8;
  const int s = row_seq[t];
  const int r = t - seq_row0[s], len = seq_rows[s];
  const int dr = drop[s];
  const bool init = has_init[s] != 0;
  const long sbase = (long)s * hist * FG_W;
  float x[4][8];                            // x[j] = the input at t - 3 (3 - j)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k = 3 * (3 - j);
    if (r - k >= 0) {
      fg_ld8(xc + (long)(t - k) * FG_W + ch, x[j]);
    } else if (init) {
      fg_ld8(st_in + sbase + (long)(hist - dr - (k - r)) * FG_W + ch, x[j]);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) x[j][e] = 0.f;
    }
  }
  float w[8][4];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    G4 v;
    v.u2 = *(const uint2 *)(cw + (long)(ch + e) * 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) w[e][j] = (float)v.h[j];
  }
  float g[8], xv[8], o[8], p[8];
  fg_ld8(gv + (long)t * FG_W + ch, g);
  fg_ld8(xs + (long)t * FG_W + ch, xv);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float a = w[e][0] * x[0][e];
    a = a + w[e][1] * x[1][e];
    a = a + w[e][2] * x[2][e];
    a = a + w[e][3] * x[3][e];
    const float c = bf(a);
    p[e] = bf(g[e] + bf(fn_silu(c)));
    o[e] = bf(xv[e] + p[e]);
  }
  fg_st8(out + (long)t * FG_W + ch, o);
  if (ple != nullptr) fg_st8(ple + (long)t * FG_W + ch, p);
  // the next history: this row if it is one of the last H of its sequence's step
  if (r >= len - hist) fg_st8(st_out + sbase + (long)(hist - len + r) * FG_W + ch, x[3]);
  if (r == 0) {
    for (int ro = 0; ro < hist - len; ++ro) {
      const int k = hist - len - ro;          // positions before this step's first row
      // A row older than the old history's valid window (drop > 0 shortens it by `drop`) is not
      // known: zero. It is never read -- the next step needs 9 + (its drop <= len - 1) rows back,
      // and len + hist - drop >= 9 + len - 1 whenever drop <= spec (hard hold 2 finding: the
      // unguarded copy read row -2 of the slot and faulted).
      const int src = hist - dr - k;
      float h8[8];
      if (init && src >= 0) fg_ld8(st_in + sbase + (long)src * FG_W + ch, h8);
      else {
#pragma unroll
        for (int e = 0; e < 8; ++e) h8[e] = 0.f;
      }
      fg_st8(st_out + sbase + (long)ro * FG_W + ch, h8);
    }
  }
}

// The GATED fp8 table (qwen_fp8, docs/EXEC.md sections 10 / 14): 16 rows x 160 e4m3 per token
// staged in the same [t][head][160] order, ONE f32 per-TABLE scale beside the buffer (FN7's
// `PleIndex::table_scale`, the checkpoint's bf16 value widened): emb = bf16(f32(e4m3) * scale).
// One thread per 8 bytes; grid ceil(m * 2560 / 8 / 256).
extern "C" __global__ __launch_bounds__(256) void fn_ple_rows_fp8(
    const unsigned char *__restrict__ rows, const float *__restrict__ scale,
    __bf16 *__restrict__ emb, int m) {
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  if (g >= (long)m * (FG_H / 8)) return;
  const float sc = scale[0];
  const uint2 raw = ((const uint2 *)rows)[g];
  float o[8];
#pragma unroll
  for (int e = 0; e < 8; e += 2) {
    const u32 w = e < 4 ? raw.x : raw.y;
    const int sh = (e & 3) * 8;
    typedef float v2f __attribute__((ext_vector_type(2)));
    const v2f f = __builtin_amdgcn_cvt_pk_f32_fp8(w >> sh, false);
    o[e] = bf(f[0] * sc);
    o[e + 1] = bf(f[1] * sc);
  }
  fg_st8(emb + g * 8, o);
}

// =============================================================================== 4. MTP
// The reference server's GemmaRMSNorm (forward_static): out = bf16((x * rsqrt(mean(x^2) + eps)) * (1 + w)).
// ONE workgroup per row, 256 threads; thread t owns elements c * 2048 + 8t + j (c < W / 2048,
// j 0..7), summed sequentially in (c, j) order; a 32-lane xor butterfly per wave; the 8 wave sums
// added in wave order by thread 0. Rows [0, m) = the embedding (W 2,560, `we`), rows [m, 2m) =
// the target's pre-final-mixer stream (W 10,240, `wh` -- ONE norm over all four streams).
#define FG_MAXC 5
__device__ __forceinline__ void fg_row_norm(const __bf16 *__restrict__ x,
                                            const __bf16 *__restrict__ w, __bf16 *__restrict__ o,
                                            int width) {
  __shared__ float s_part[8];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int nc = width / 2048;
  float v[FG_MAXC][8];
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < FG_MAXC; ++c) {
    if (c < nc) {
      fg_ld8(x + c * 2048 + 8 * tid, v[c]);
#pragma unroll
      for (int j = 0; j < 8; ++j) ss = ss + v[c][j] * v[c][j];
    }
  }
  for (int o2 = 16; o2; o2 >>= 1) ss = ss + __shfl_xor(ss, o2);
  if (lane == 0) s_part[wave] = ss;
  __syncthreads();
  float tot = s_part[0];
#pragma unroll
  for (int q = 1; q < 8; ++q) tot = tot + s_part[q];
  const float rr = rsqrt_exact(tot / (float)width + FG_EPS);
#pragma unroll
  for (int c = 0; c < FG_MAXC; ++c) {
    if (c < nc) {
      float wf[8], out[8];
      fg_ld8(w + c * 2048 + 8 * tid, wf);
#pragma unroll
      for (int j = 0; j < 8; ++j) out[j] = bf((v[c][j] * rr) * (1.f + wf[j]));
      fg_st8(o + c * 2048 + 8 * tid, out);
    }
  }
}

// The embedding's width is 2,560 = 1.25 x 2048: handled as ONE 2,048 chunk + a 512 tail read by
// threads 0..63, summed after the chunk (c = 1) -- still one fixed order per thread.
__device__ __forceinline__ void fg_row_norm_2560(const __bf16 *__restrict__ x,
                                                 const __bf16 *__restrict__ w,
                                                 __bf16 *__restrict__ o) {
  __shared__ float s_part[8];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  float v0[8], v1[8];
  fg_ld8(x + 8 * tid, v0);
  const bool tail = tid < 64;
  fg_ld8(x + 2048 + 8 * (tail ? tid : 63), v1);
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) ss = ss + v0[j] * v0[j];
  if (tail) {
#pragma unroll
    for (int j = 0; j < 8; ++j) ss = ss + v1[j] * v1[j];
  }
  for (int o2 = 16; o2; o2 >>= 1) ss = ss + __shfl_xor(ss, o2);
  if (lane == 0) s_part[wave] = ss;
  __syncthreads();
  float tot = s_part[0];
#pragma unroll
  for (int q = 1; q < 8; ++q) tot = tot + s_part[q];
  const float rr = rsqrt_exact(tot / 2560.f + FG_EPS);
  float wf[8], out[8];
  fg_ld8(w + 8 * tid, wf);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = bf((v0[j] * rr) * (1.f + wf[j]));
  fg_st8(o + 8 * tid, out);
  if (tail) {
    fg_ld8(w + 2048 + 8 * tid, wf);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = bf((v1[j] * rr) * (1.f + wf[j]));
    fg_st8(o + 2048 + 8 * tid, out);
  }
}

extern "C" __global__ __launch_bounds__(256) void fn_mtp_norms(
    const __bf16 *__restrict__ emb, const __bf16 *__restrict__ we, __bf16 *__restrict__ emb_n,
    const __bf16 *__restrict__ hid, const __bf16 *__restrict__ wh, __bf16 *__restrict__ hid_n,
    int m) {
  const int row = blockIdx.x;
  if (row < m) {
    fg_row_norm_2560(emb + (long)row * FG_H, we, emb_n + (long)row * FG_H);
  } else {
    const int r = row - m;
    fg_row_norm(hid + (long)r * FG_W, wh, hid_n + (long)r * FG_W, FG_W);
  }
}

// x[t][s * 2560 + i] = bf16(fe[t][i] + fh[t * 4 + s][i]); fh = fc_hidden over the normed stream
// viewed [4m, 2560] (token-major, stream inner -- `view(T, hc, H)`). One thread per 8 elements.
extern "C" __global__ __launch_bounds__(256) void fn_mtp_add(
    const __bf16 *__restrict__ fe, const __bf16 *__restrict__ fh, __bf16 *__restrict__ x, int m) {
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  if (g >= (long)m * (FG_W / 8)) return;
  const long t = g / (FG_W / 8);
  const int e = (int)(g - t * (FG_W / 8)) * 8, i = e % FG_H;
  float a[8], b[8], o[8];
  fg_ld8(fe + t * FG_H + i, a);
  fg_ld8(fh + t * FG_W + e, b);
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = bf(a[j] + b[j]);
  fg_st8(x + t * FG_W + e, o);
}

// The MTP experts' SiluAndMul on expert-major rows (FN4's row law: `meta[1]` live padded rows,
// `row_tok[row] < 0` a pad row, skipped). act = bf16(silu(g) * u), silu = g * sigmoid(g) on
// fn_hexp; gu row = [gate 640 | up 640] (the pack's merged gate_up, gate rows first).
extern "C" __global__ __launch_bounds__(256) void fn_moe_silu_mul_bf16_t256(
    const __bf16 *__restrict__ gu, __bf16 *__restrict__ act, int inter,
    const int *__restrict__ row_tok, const int *__restrict__ meta) {
  const int row = blockIdx.x;
  if (row >= meta[1] || row_tok[row] < 0) return;
  for (int i = threadIdx.x * 8; i < inter; i += 256 * 8) {
    float g[8], u[8], o[8];
    fg_ld8(gu + (long)row * 2 * inter + i, g);
    fg_ld8(gu + (long)row * 2 * inter + inter + i, u);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = bf(fn_silu(g[j]) * u[j]);
    fg_st8(act + (long)row * inter + i, o);
  }
}

// =============================================================================== 5. FN21 (prefill)
// BIT-EXACT restructurings of the two narrow glue kernels, NEW symbols beside the shipped ones
// (the shipped `fn_idx_q_compress_h128` and the 27B's `qk_norm_rope_wg8_h256_rot64_mi1_red5`, which
// Flash-Next calls at 12 q / 1 kv, are untouched). Their ISA: every wave walked a chain of serial
// round trips -- the head, then (after the reduction and the norm) the weight one 2-byte load per
// element, then per rotary element a position load, a wait, the cos / sin loads, a wait -- and ONE
// fp64 correctly rounded rsqrt per head (~27 fp64 instructions on one lane; gfx1201's fp64 is a
// small fraction of its fp32 rate, and a wave pays for the instruction, not for the lane). What
// fixes the bytes, kept exactly: the per-lane sum of squares ((a0 + a1) + a2) + a3 (two vec4 blocks
// accumulated first at HD 256), the ADJACENT shuffle tree o = 1..16 whose lane-0 value is the
// head's sum, inv = rsqrt_exact(sum / HD + eps), n = bf((x * inv) * (w + 1)), the rope
// rtz(n c) -/+ rtz(n' s) rounded once, the section select, the pooled key's (((0 + k0) + k1) + k2)
// + k3 and bf(acc * 0.25). What changes: every load is issued before any arithmetic that does not
// feed it (positions -> cos / sin are the only dependent pair), the weight is loaded as vectors,
// and SEVERAL heads' rsqrts run as ONE fp64 sequence in several lanes (the same per-lane IEEE ops).

// cos / sin of this lane's four rotary dims i = (lane & 7) * 4 + j at the interleaved mrope
// section's position (lanes 16..31 load lane & 7's harmlessly; they hold no rotary dim).
__device__ __forceinline__ void fg_rope_ops(const __bf16 *__restrict__ cos_sin, const long p3[3],
                                            int lane, int rd, int sec_h, int sec_w, float c[4],
                                            float sn[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = (lane & 7) * 4 + j;
    const int r = i % 3;
    const int sec = (r == 1 && i <= 3 * sec_h) ? 1 : ((r == 2 && i <= 3 * sec_w) ? 2 : 0);
    const long p = sec == 0 ? p3[0] : (sec == 1 ? p3[1] : p3[2]);
    c[j] = (float)cos_sin[p * rd + i];
    sn[j] = (float)cos_sin[p * rd + rd / 2 + i];
  }
}
// n = bf((x * inv) * w1) (w1 = w + 1), then the neox rope over the first vec4 block (lanes < 16).
__device__ __forceinline__ void fg_apply4(float x[4], float inv, const float w1[4],
                                          const float c[4], const float sn[4], int lane) {
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n[j] = bf((x[j] * inv) * w1[j]);
  float other[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) other[j] = __shfl_xor(n[j], 8);
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float t = bfmul(other[j], sn[j]);
      n[j] = bfmul(n[j], c[j]) + (lane < 8 ? -t : t);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = n[j];
}
// the head's sum: ((a0 + a1) + a2) + a3 per lane, then the adjacent tree (lane 0 = the head's)
__device__ __forceinline__ float fg_tree(float sum) {
  for (int o = 1; o < 32; o <<= 1) sum += __shfl_down(sum, o);
  return sum;
}

// qk_norm_rope at head_dim 256 / rotary 64 / interleaved mrope (attn_glue's
// `qk_norm_rope_wave_impl<256, 64, true, 256>`, the 27B symbol Flash-Next calls): the same
// arguments, grid ceil(m (nq + nkv) / 8), 256 threads, one (token, head) pair per wave. The eight
// waves' sums meet in LDS and wave 0 runs the eight fp64 rsqrts as one sequence.
extern "C" __global__ __launch_bounds__(256) void fn_qk_norm_rope_h256_rot64_mi1_v2(
    const __bf16 *__restrict__ qkv, const __bf16 *__restrict__ qw, const __bf16 *__restrict__ kw,
    const __bf16 *__restrict__ cos_sin, const long *__restrict__ pos, __bf16 *__restrict__ q_out,
    __bf16 *__restrict__ k_out, int m, int nq, int nkv, int qkv_n, float eps, int sec_t,
    int sec_h, int sec_w) {
  constexpr int HD = 256, RD = 64, W = 32;
  __shared__ float s_sum[8], s_inv[8];
  const int lane = (int)(threadIdx.x & 31), wave = (int)(threadIdx.x >> 5);
  const int heads = nq + nkv;
  const long npair = (long)m * heads;
  const long pr = (long)blockIdx.x * 8 + wave;
  const bool live = pr < npair;
  const long pair = live ? pr : npair - 1;       // a dead wave still meets the barriers
  const int tok = (int)(pair / heads), head = (int)(pair - (long)tok * heads);
  const bool is_k = head >= nq;
  const int lh = is_k ? head - nq : head;
  const __bf16 *in = qkv + (long)tok * qkv_n
                   + (is_k ? (long)nq * 2 * HD + (long)lh * HD : (long)lh * 2 * HD);
  const __bf16 *w = is_k ? kw : qw;
  __bf16 *out = is_k ? k_out + (long)tok * nkv * HD + (long)lh * HD
                     : q_out + (long)tok * nq * HD + (long)lh * HD;
  const long p3[3] = {pos[tok], pos[(long)m + tok], pos[2 * (long)m + tok]};
  G4 vin[2], vw[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    vin[r].u2 = ((const uint2 *)(in + (r * W + lane) * 4))[0];
    vw[r].u2 = ((const uint2 *)(w + (r * W + lane) * 4))[0];
  }
  float c[4], sn[4];
  fg_rope_ops(cos_sin, p3, lane, RD, sec_h, sec_w, c, sn);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float v0 = (float)vin[r].h[0], v1 = (float)vin[r].h[1];
    const float v2 = (float)vin[r].h[2], v3 = (float)vin[r].h[3];
    a0 += v0 * v0;
    a1 += v1 * v1;
    a2 += v2 * v2;
    a3 += v3 * v3;
  }
  const float sum = fg_tree(((a0 + a1) + a2) + a3);
  if (lane == 0) s_sum[wave] = sum;
  __syncthreads();
  if (wave == 0 && lane < 8) s_inv[lane] = rsqrt_exact(s_sum[lane] / (float)HD + eps);
  __syncthreads();
  const float inv = s_inv[wave];
  float x[2][4], w1[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[r][j] = (float)vin[r].h[j];
      w1[r][j] = (float)vw[r].h[j] + 1.f;
    }
  fg_apply4(x[0], inv, w1[0], c, sn, lane);      // block 0 holds the rotary dims
  float n1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n1[j] = bf((x[1][j] * inv) * w1[1][j]);
  if (!live) return;
  G4 o0, o1;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    o0.h[j] = (__bf16)x[0][j];
    o1.h[j] = (__bf16)n1[j];
  }
  ((uint2 *)(out + lane * 4))[0] = o0.u2;
  ((uint2 *)(out + (W + lane) * 4))[0] = o1.u2;
}

// The QSA indexer (section 2's contract, the same arguments) at FOUR ROWS a workgroup: grid
// ceil(m / 4), 256 threads; wave 2r = the four q heads of row 4 bx + r (the four heads' loads in
// flight together, one cos / sin gather for all four, their rsqrts as one fp64 sequence in lanes
// 0..3), wave 2r + 1 = that row's compressed group (when it completes one), its row metadata,
// members, positions, weight and page loaded as early as their dependencies allow.
extern "C" __global__ __launch_bounds__(256) void fn_idx_q_compress_h128_v2(
    const __bf16 *__restrict__ qk, long qk_stride, const __bf16 *__restrict__ qw,
    const __bf16 *__restrict__ kw, const __bf16 *__restrict__ cos_sin, const long *__restrict__ pos,
    const int *__restrict__ row_seq, const long *__restrict__ row_lpos,
    const int *__restrict__ seq_row0, const long *__restrict__ seq_pos0,
    const int *__restrict__ ring_row0, const __bf16 *__restrict__ ring,
    const long *__restrict__ ring_pos, int ring_rows, const int *__restrict__ block_table,
    int max_blocks, __bf16 *__restrict__ q_out, __bf16 *__restrict__ comp, int m, int sec_h,
    int sec_w) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int t = (int)blockIdx.x * 4 + (wave >> 1);
  if (t >= m) return;
  if ((wave & 1) == 0) {
    const long p3[3] = {pos[t], pos[(long)m + t], pos[2 * (long)m + t]};
    G4 v[4], vw;
#pragma unroll
    for (int h = 0; h < 4; ++h) v[h].u2 = *(const uint2 *)(qk + (long)t * qk_stride + h * 128 + 4 * lane);
    vw.u2 = *(const uint2 *)(qw + 4 * lane);
    float c[4], sn[4];
    fg_rope_ops(cos_sin, p3, lane, 64, sec_h, sec_w, c, sn);
    float x[4][4], sum[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
#pragma unroll
      for (int j = 0; j < 4; ++j) x[h][j] = (float)v[h].h[j];
      const float a0 = x[h][0] * x[h][0], a1 = x[h][1] * x[h][1];
      const float a2 = x[h][2] * x[h][2], a3 = x[h][3] * x[h][3];
      sum[h] = fg_tree(((a0 + a1) + a2) + a3);
    }
    const float s0 = __shfl(sum[0], 0), s1 = __shfl(sum[1], 0);
    const float s2 = __shfl(sum[2], 0), s3 = __shfl(sum[3], 0);
    const int hl = lane & 3;
    const float mine = hl == 0 ? s0 : (hl == 1 ? s1 : (hl == 2 ? s2 : s3));
    const float invl = rsqrt_exact(mine / 128.f + FG_EPS);
    float w1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w1[j] = (float)vw.h[j] + 1.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      fg_apply4(x[h], __shfl(invl, h), w1, c, sn, lane);
      G4 o;
#pragma unroll
      for (int j = 0; j < 4; ++j) o.h[j] = (__bf16)x[h][j];
      *(uint2 *)(q_out + ((long)t * 4 + h) * 128 + 4 * lane) = o.u2;
    }
    return;
  }
  const long lp = row_lpos[t];
  if (lp < 3 || ((lp + 1) & 3) != 0) return;
  const int s = row_seq[t];
  G4 vw;
  vw.u2 = *(const uint2 *)(kw + 4 * lane);
  const long g = lp >> 2;
  const long page = block_table[(long)s * max_blocks + (g >> 2)];
  const long p0 = seq_pos0[s];
  const int r0 = seq_row0[s];
  const long rbase = (long)ring_row0[s] * ring_rows;
  G4 mv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const long pj = lp - 3 + j;
    const __bf16 *src = pj >= p0 ? qk + (long)(r0 + (pj - p0)) * qk_stride + 512
                                 : ring + (rbase + pj % ring_rows) * 128;
    mv[j].u2 = *(const uint2 *)(src + 4 * lane);
  }
  const long pf = lp - 3;
  long p3[3];
  if (pf >= p0) {
    const long rr = r0 + (pf - p0);
    p3[0] = pos[rr]; p3[1] = pos[(long)m + rr]; p3[2] = pos[2 * (long)m + rr];
  } else {
    const long rr = rbase + pf % ring_rows;
    p3[0] = ring_pos[rr * 3]; p3[1] = ring_pos[rr * 3 + 1]; p3[2] = ring_pos[rr * 3 + 2];
  }
  float c[4], sn[4];
  fg_rope_ops(cos_sin, p3, lane, 64, sec_h, sec_w, c, sn);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = acc[e] + (float)mv[j].h[e];
  float x[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) x[e] = bf(acc[e] * 0.25f);
  const float a0 = x[0] * x[0], a1 = x[1] * x[1], a2 = x[2] * x[2], a3 = x[3] * x[3];
  const float sum = fg_tree(((a0 + a1) + a2) + a3);
  const float inv = rsqrt_exact(__shfl(sum, 0) / 128.f + FG_EPS);
  float w1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) w1[j] = (float)vw.h[j] + 1.f;
  fg_apply4(x, inv, w1, c, sn, lane);
  G4 o;
#pragma unroll
  for (int e = 0; e < 4; ++e) o.h[e] = (__bf16)x[e];
  *(uint2 *)(comp + (page * 4 + (g & 3)) * 128 + 4 * lane) = o.u2;
}
