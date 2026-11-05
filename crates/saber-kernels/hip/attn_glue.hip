// The three attention-layer ops the P1 decoder block still computed in torch, or did not model at
// all. Unlike every other plugin in this crate, this file is NOT sliced from a reference .hip by a
// generator: prod has no HIP source for any of these. Each one is a Triton kernel or a torch
// composition, so the device code here is written from the reference server's reference semantics and gated
// bit-exact against what prod actually runs. What prod runs, op by op:
//
//   qk_norm_rope   GemmaRMSNorm(q)/GemmaRMSNorm(k) in ATen  +  the reference server's `triton_mrope` Triton
//                  kernel. The FUSED Triton kernel the reference server ships for exactly this fusion
//                  (`fused_qk_rmsnorm_rope_gate`) is gated on `current_platform.is_cuda()`, which
//                  is FALSE on ROCm -- so prod takes the eager path and never runs it.
//   attn_gate_quant  `attn_output * torch.sigmoid(gate)` (two ATen kernels) followed by
//                  `scaled_fp8_quant(..., use_per_token_if_dynamic=True)`, i.e. the reference server's
//                  `dynamic_per_token_scaled_fp8_quant` HIP kernel, inside `mxfp4_linear`.
//   kv_append      the reference server's `triton_reshape_and_cache_flash` Triton kernel (NOT
//                  `_custom_ops.reshape_and_cache_flash`): `TritonAttentionImpl.do_kv_cache_update`
//                  picks it, and SABERAttentionImpl inherits that method unchanged.
//
// No host code, no getenv: every choice prod makes from a config value is an explicit symbol here
// and a row in rules/attn_glue_gfx1201.toml.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef unsigned int u32;
typedef unsigned long long u64;
union Vec8 { uint4 u4; __bf16 h[8]; };
union Vec4 { uint2 u2; __bf16 h[4]; };

// ---- shared precision primitives -------------------------------------------------------------
// Every rounding below is a place where prod rounds; none is a convenience. A bf16 round that
// belongs in the middle of an expression and is left out does not fail loudly, it fails at the
// last bit of a few elements in ten thousand.
__device__ __forceinline__ float bf16r(float v) { return (float)(__bf16)v; }

// ATen's `rsqrt`, which is CORRECTLY ROUNDED. This is the last thing the rope gate turned on and
// the least obvious. `rsqrtf()` in HIP device code lowers to `v_rsq_f32` -- OCML's `rsqrt_f32` is
// `llvm.amdgcn.rsq.f32` with a denormal rescale and NO refinement, so the function name is not
// the question -- and that instruction is a ~1-ULP approximation. `torch.rsqrt` is not: over 2^22
// random inputs it agreed with an fp64 reference on every single one, while `1.0f/sqrtf(x)`
// disagreed on 23% of them. Computing it in fp64 is what matches, and it is free here: the
// variance is one value per workgroup, so this is one fp64 divide per 256 threads.
//
// What it was worth: with `v_rsq_f32` the rope gate stopped at 15 of 40 cases differing, by 1 to
// 29 bytes out of up to 29 MB. The failures are always a rotary PAIR of one head -- a 1-ULP `inv`
// moves one normed element across a rounding boundary, and that element feeds both outputs of its
// pair while never being stored itself, which is why it is invisible in the pass-through tail.
__device__ __forceinline__ float rsqrt_exact(float v) {
  return (float)(1.0 / sqrt((double)v));
}

// A bf16 MULTIPLY, the way the hardware does it and the way Triton emits it. Triton keeps the
// rope arithmetic in bf16 tensors (`arith.mulf : tensor<16x32xbf16>` in its own TTIR) and the
// AMDGPU backend lowers each such multiply to `v_dot2_bf16_bf16` with the second lane zeroed --
// 20 of them in the cached `_triton_mrope_forward.amdgcn`.
//
// **That instruction TRUNCATES its bf16 result; it does not round to nearest.** Modelling the
// product as `(__bf16)(a * c)`, which is the obvious reading of "a bf16 multiply" and what
// torch's own bf16 multiply does, disagrees with the reference on 17% of the rotary elements;
// modelling the whole expression in fp32 with one rounding disagrees on 19%. Truncating agrees on
// every one of 196608. (The SUM is round-to-nearest-even -- the `lshr 16; and 1; add3 0x7fff`
// sequence in the same ISA -- so only the products truncate, and only they may.)
//
// The truncation is a mask rather than the `__builtin_amdgcn_fdot2_bf16_bf16` intrinsic on
// purpose. Both operands are bf16 values, so their product has at most 16 significant bits and is
// EXACT in fp32; dropping the low 16 bits of that exact product is what the instruction does, by
// definition, with no dependence on the builtin lowering the way one expects. (It did not: a
// first attempt through the intrinsic missed every rotary element.)
__device__ __forceinline__ float bfmul(float a, float b) {
  return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, a * b) & 0xffff0000u);
}

// The reference server `scaled_fp8_conversion<false>`: clamp then the ROCm hardware convert. Note the DIVISION --
// the reference divides by the scale here, it does not multiply by a reciprocal, and on some rows
// the two differ in the last bit of the e4m3 result. (The MXFP4 producers in this crate multiply
// by the reciprocal, because THEIR reference, the reference's own kernel, does.)
__device__ __forceinline__ u32 e4m3_pair(float a, float b, float scale) {
  a = fminf(fmaxf(a / scale, -448.f), 448.f);
  b = fminf(fmaxf(b / scale, -448.f), 448.f);
  return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu;
}

// The KV append's convert has NO clamp, deliberately: `triton_reshape_and_cache_flash` divides and
// lets `tl.store` cast, and that cast is this same hardware convert, which saturates on its own.
// Clamping first is identical for every finite input and DIFFERENT for a NaN -- `fminf(NaN, 448)`
// is 448, while the convert propagates the NaN. Nothing in a KV cache should be NaN; a port that
// quietly turned one into 448 would be the kind of difference this gate exists to catch.
__device__ __forceinline__ u32 e4m3_pair_raw(float a, float b, float scale) {
  return __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0u, false) & 0xffffu;
}

// ---- 1. qk_norm_rope ------------------------------------------------------------------------
//
// One launch for what prod spends five on (two ATen RMS norms, the `cos_sin_cache[positions]`
// gather, two chunks, and the Triton mrope). The gather is folded in, so this kernel takes the
// CACHE and the positions rather than a materialised [3, T, rd/2] cos/sin pair.
//
// Arithmetic contract, in prod's own order:
//   GemmaRMSNorm (upstream.ir.ops.rms_norm native -- the compiled C++ kernel is NOT eligible because its
//   `supports_args` needs weight.dtype == x.dtype and GemmaRMSNorm hands it `weight.float()+1`):
//     xf   = f32(x)                                 (bf16 -> f32, exact)
//     var  = mean(xf^2)                             <- fp32, ORDER MATTERS, see RED below
//     n    = bf16( (xf * rsqrt(var + eps)) * (f32(w) + 1) )
//   triton_mrope, is_neox_style=1 (pairs i and i+rd/2), all arithmetic in the CACHE's dtype (bf16):
//     o1 = bf16( bf16(n1*cos) - bf16(n2*sin) )
//     o2 = bf16( bf16(n2*cos) + bf16(n1*sin) )
//   and [rd, hd) passes through as the normed value.
//
// The mrope section select, from the Triton kernel's own masks over i in [0, rd/2):
//   MI=1 (mrope_interleaved, prod's config):  h if i%3==1 and i <= 3*sec_h; w if i%3==2 and
//                                             i <= 3*sec_w; t otherwise.
//   MI=0 (chunked):                           t for i < sec_t, h for the next sec_h, else w.
// For a text-only request the three position rows are equal and the select cannot be observed;
// it is still a decision the config makes, so it is a symbol and a rules row, not a branch.
//
// RED: the fp32 variance reduction ORDER. torch's `mean(-1)` is a kernel whose summation order is
// its own; a port that sums in a different order agrees to ~1e-7 relative, which survives the
// bf16 round almost always -- "almost" being a few elements in 10^5. The order is therefore a
// gated choice, not an implementation detail: three orders are compiled and the rules name the
// one that is bit-exact, with the other two standing as controls that the gate can tell them
// apart.
//   RED=0  serial over 0..hd-1, one accumulator
//   RED=1  four accumulators striding 4 (a vec4 load's lanes), combined (a0+a1)+(a2+a3)
//   RED=2  pairwise tree over the whole head
//   RED=3  ATen's shape at W = hd/4 threads: one vec4 each, ((a0+a1)+a2)+a3, descending tree
//   RED=4  ATen's shape at W = hd/8 threads: two vec4s each, same combine and tree
//   RED=5  ATen's ACTUAL order: vec4, 32 threads, left-assoc combine, ADJACENT-pairing tree
template <int HD, int RD, bool MI, int RED, bool GATED = true>
__device__ __forceinline__ void qk_norm_rope_impl(
    const __bf16 *__restrict__ qkv,      // [M, qkv_n] : per q head [q|gate], then k, then v
    const __bf16 *__restrict__ qw,       // [HD] GemmaRMSNorm weight for q (the raw w, not 1+w)
    const __bf16 *__restrict__ kw,       // [HD]
    const __bf16 *__restrict__ cos_sin,  // [max_pos, RD] : [cos(RD/2) | sin(RD/2)]
    const long *__restrict__ pos,        // [3, M] int64 mrope positions (t, h, w rows)
    __bf16 *__restrict__ q_out,          // [M, nq*HD]
    __bf16 *__restrict__ k_out,          // [M, nkv*HD]
    int m, int nq, int nkv, int qkv_n, float eps, int sec_t, int sec_h, int sec_w) {
  const int tok = blockIdx.x;
  const int head = blockIdx.y;                 // [0, nq) q heads, then [nq, nq+nkv) k heads
  const int tid = threadIdx.x;                 // HD threads, one per element of the head
  const bool is_k = head >= nq;
  const int lh = is_k ? head - nq : head;
  // GATED is the TARGET's layout, and it is the default so every symbol below is unchanged:
  // `attn_output_gate` doubles q_proj, so each q head occupies [q | gate] and the k block starts
  // after nq * 2 * HD. The DFLASH2 DRAFTER has no output gate -- its qkv is a plain [q | k | v] --
  // so its q heads are HD apart and k starts at nq * HD.
  //
  // This parameter is the whole kernel side of serving the drafter's rope. The rest of it is
  // already reachable: the drafter's rope is PLAIN (no mrope_section in its config) and the MI=0
  // arm's `sec = i < sec_t ? 0 : ...` sends every element to section 0 when `sec_t >= RD/2`, which
  // IS a per-token position. What is missing is on the RULES side -- `QkBand` selects on a token
  // count and carries no `head_dim`, so a row for a 128-wide head would be selectable for the
  // target at a large M -- so no entry point is instantiated here yet. When that dimension exists,
  // the drafter's symbol is one `QKR`-style macro line: <128, 128, false, 5, false>.
  constexpr int QSTRIDE = GATED ? 2 * HD : HD;
  const __bf16 *in = qkv + (long)tok * qkv_n
                   + (is_k ? (long)nq * QSTRIDE + (long)lh * HD : (long)lh * QSTRIDE);
  const __bf16 *w = is_k ? kw : qw;
  __bf16 *out = is_k ? k_out + (long)tok * nkv * HD + (long)lh * HD
                     : q_out + (long)tok * nq * HD + (long)lh * HD;

  const float x = (float)in[tid];
  // --- variance -------------------------------------------------------------------------------
  __shared__ float red[HD];
  __shared__ float part[4];
  __shared__ float tree[HD / 4];
  red[tid] = x * x;
  __syncthreads();
  if constexpr (RED == 0) {
    if (tid == 0) {
      float s = 0.f;
      for (int i = 0; i < HD; ++i) s += red[i];
      part[0] = s;
    }
    __syncthreads();
  } else if constexpr (RED == 1) {
    if (tid < 4) {
      float s = 0.f;
      for (int i = tid; i < HD; i += 4) s += red[i];
      part[tid] = s;                           // a scratch of its own: writing into `red` would
    }                                          // race the other three lanes still reading it
    __syncthreads();
    if (tid == 0) part[0] = (part[0] + part[1]) + (part[2] + part[3]);
    __syncthreads();
  } else if constexpr (RED == 2) {
    for (int o = HD >> 1; o; o >>= 1) {
      if (tid < o) red[tid] += red[tid + o];
      __syncthreads();
    }
    if (tid == 0) part[0] = red[0];
    __syncthreads();
  } else if constexpr (RED == 5) {
    // ATen's ACTUAL order, found by sweeping (vector width x threads per row x tree shape)
    // against `x.pow(2).sum(-1)` on this device: four floats at a time into four accumulators,
    // 32 threads cooperating over a 256-wide row (so two vec4s each), the four accumulators
    // combined LEFT-ASSOCIATIVELY, and then an ADJACENT-pairing tree over the 32 partials --
    // which is exactly `warp_reduce`'s `v += shfl_down(v, o)` for o = 1, 2, 4, 8, 16. Zero of
    // 4096 rows differ; every other shape in the sweep differed on 1.6-93% of them.
    //
    // 32 threads is one wave on RDNA4, so the tree needs no LDS and no barrier -- this order is
    // both the bit-exact one and the cheapest of the five.
    constexpr int W = 32, VEC = 4, ROUNDS = HD / (W * VEC);
    if (tid < W) {
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int r = 0; r < ROUNDS; ++r) {
        const int b = (r * W + tid) * VEC;
        a0 += red[b];
        a1 += red[b + 1];
        a2 += red[b + 2];
        a3 += red[b + 3];
      }
      float v = ((a0 + a1) + a2) + a3;
      for (int o = 1; o < W; o <<= 1) v += __shfl_down(v, o);
      if (tid == 0) part[0] = v;
    }
    __syncthreads();
  } else {
    // Two more ATen-shaped candidates, kept as controls: same four accumulators, but W = hd/4 or
    // hd/8 threads and the DESCENDING (tid, tid + offset) tree rather than the adjacent one.
    // Both are plausible readings of `ReduceConfig` and neither is bit-exact.
    constexpr int W = (RED == 3) ? (HD / 4) : (HD / 8);
    constexpr int NV = HD / 4;
    if (tid < W) {
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      for (int v = tid; v < NV; v += W) {
        a0 += red[4 * v];
        a1 += red[4 * v + 1];
        a2 += red[4 * v + 2];
        a3 += red[4 * v + 3];
      }
      tree[tid] = ((a0 + a1) + a2) + a3;
    }
    __syncthreads();
    for (int o = W >> 1; o; o >>= 1) {
      if (tid < o) tree[tid] += tree[tid + o];
      __syncthreads();
    }
    if (tid == 0) part[0] = tree[0];
    __syncthreads();
  }
  const float inv = rsqrt_exact(part[0] / (float)HD + eps);
  const float n = bf16r(x * inv * ((float)w[tid] + 1.f));

  // --- rope over [0, RD), pass-through over [RD, HD) -------------------------------------------
  // Every thread reaches the barrier: the pass-through lanes store AFTER it, not before. An early
  // `return` here would leave a divergent __syncthreads, which on gfx12 is a hang, not a warning.
  __shared__ float pair[RD];
  if (tid < RD) pair[tid] = n;
  __syncthreads();
  if (tid >= RD) { out[tid] = (__bf16)n; return; }
  const int half = RD / 2;
  const int i = tid < half ? tid : tid - half;   // rotary index within the pair
  int sec;
  if constexpr (MI) {
    const int r = i % 3;
    sec = (r == 1 && i <= 3 * sec_h) ? 1 : ((r == 2 && i <= 3 * sec_w) ? 2 : 0);
  } else {
    sec = i < sec_t ? 0 : (i < sec_t + sec_h ? 1 : 2);
  }
  const long p = pos[(long)sec * m + tok];
  const float c = (float)cos_sin[p * RD + i];
  const float sn = (float)cos_sin[p * RD + half + i];
  const float a = pair[i], b = pair[half + i];
  // Truncating products, a round-to-nearest-even sum: see `bfdot`. `-ffp-contract=off` is in
  // saber-kbuild's EXTRA_FLAGS for this line -- with contraction on, hipcc would fuse the sum
  // into an FMA over one of the products and keep it at full width, which is neither what the
  // hardware dot2 produces nor what Triton emits.
  out[tid] = (__bf16)(tid < half ? bfmul(a, c) - bfmul(b, sn) : bfmul(b, c) + bfmul(a, sn));
}

// ---- 1b. qk_norm_rope, one WAVE per head -----------------------------------------------------
//
// Same op, same bytes, a different decomposition -- and the reason it exists is the one case the
// first round's timing lost: `qk_rope` at M=2048 on the TP=2 shard, at ratio 1.377. The kernel
// above gives one 256-thread workgroup to one (token, head), which at M=2048 is 28672 workgroups
// each running a reduction in which 32 of its 256 lanes take part, through LDS, behind three
// barriers.
//
// The fix is not to widen the reduction -- the reduction ORDER is ATen's and may not change (see
// RED=5) -- but to notice what that order already is: four accumulators over two vec4s, in 32
// lanes, combined by an adjacent-pairing shuffle tree. That is *exactly one wave* on RDNA4. So
// give each wave a whole head and put eight of them in a workgroup: all 256 lanes now take part,
// in eight independent reductions, and the kernel needs **no LDS and no barrier at all**.
//
// The element map falls out of ATen's order and costs nothing: lane l of a wave owns vec4 blocks
// l and l+32, i.e. elements [4l, 4l+4) and [128+4l, 128+4l+4) -- the same eight it needs for the
// reduction, the norm and the store, as two 8-byte loads and two 8-byte stores.
//
// The rope pairs dim i with dim i+32, both inside the FIRST vec4 block, so lane l holds one side
// and lane l^8 holds the other: a single `__shfl_xor(n, 8)` replaces the LDS staging. Lanes 16-31
// of each wave hold only dims >= 64, which pass through. And the two rope arms turn out to be one
// expression with a sign, because `o1 = rtz(a*c) - rtz(b*s)` at dim i and `o2 = rtz(b*c) +
// rtz(a*s)` at dim i+32 are both `rtz(mine*c) +- rtz(theirs*s)`.
//
// Bit-exactness is preserved BY CONSTRUCTION, not by luck: the reduction is the same 32 lanes
// doing the same four accumulations in the same order with the same shuffle tree, and everything
// after it is per-element.
template <int HD, int RD, bool MI, int THREADS>
__device__ __forceinline__ void qk_norm_rope_wave_impl(
    const __bf16 *__restrict__ qkv, const __bf16 *__restrict__ qw,
    const __bf16 *__restrict__ kw, const __bf16 *__restrict__ cos_sin,
    const long *__restrict__ pos, __bf16 *__restrict__ q_out, __bf16 *__restrict__ k_out,
    int m, int nq, int nkv, int qkv_n, float eps, int sec_t, int sec_h, int sec_w) {
  // WAVES is a template constant and not `blockDim.x >> 5` on purpose: reading blockDim pulls the
  // COV5 implicit kernarg block in, which took this symbol's kernarg_segment_size from 88 to 344
  // and made the code object depend on a caller filling args the descriptor does not describe.
  constexpr int W = 32, VEC = 4, PER = HD / W, BLK = PER / VEC;   // 8 per lane, 2 vec4 blocks
  constexpr int WAVES = THREADS / W;
  const int lane = (int)(threadIdx.x & (W - 1));
  const int wave = (int)(threadIdx.x >> 5);
  const int heads = nq + nkv;
  const long pair = (long)blockIdx.x * WAVES + wave;
  // A plain early return, which this kernel may do because it has no barrier to strand anyone at.
  if (pair >= (long)m * heads) return;
  const int tok = (int)(pair / heads), head = (int)(pair - (long)tok * heads);
  const bool is_k = head >= nq;
  const int lh = is_k ? head - nq : head;
  const __bf16 *in = qkv + (long)tok * qkv_n
                   + (is_k ? (long)nq * 2 * HD + (long)lh * HD : (long)lh * 2 * HD);
  const __bf16 *w = is_k ? kw : qw;
  __bf16 *out = is_k ? k_out + (long)tok * nkv * HD + (long)lh * HD
                     : q_out + (long)tok * nq * HD + (long)lh * HD;

  Vec4 vin[BLK];
#pragma unroll
  for (int r = 0; r < BLK; ++r) vin[r].u2 = ((const uint2 *)(in + (r * W + lane) * VEC))[0];

  // --- variance: ATen's order, and here it is literally one wave ------------------------------
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int r = 0; r < BLK; ++r) {
    const float v0 = (float)vin[r].h[0], v1 = (float)vin[r].h[1];
    const float v2 = (float)vin[r].h[2], v3 = (float)vin[r].h[3];
    a0 += v0 * v0;
    a1 += v1 * v1;
    a2 += v2 * v2;
    a3 += v3 * v3;
  }
  float sum = ((a0 + a1) + a2) + a3;
  for (int o = 1; o < W; o <<= 1) sum += __shfl_down(sum, o);
  float inv = 0.f;
  if (lane == 0) inv = rsqrt_exact(__shfl(sum, 0) / (float)HD + eps);
  inv = __shfl(inv, 0);

  // --- norm -----------------------------------------------------------------------------------
  float n[PER];
#pragma unroll
  for (int r = 0; r < BLK; ++r) {
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const int d = (r * W + lane) * VEC + j;
      n[r * VEC + j] = bf16r((float)vin[r].h[j] * inv * ((float)w[d] + 1.f));
    }
  }

  // --- rope over the first vec4 block; the second is always >= 128 and passes through ---------
  // The exchange is unconditional so every lane of the wave takes part in the shuffle; only
  // lanes 0-15 hold a rotary dim and use the result.
  float other[VEC];
#pragma unroll
  for (int j = 0; j < VEC; ++j) other[j] = __shfl_xor(n[j], 8);
  if (lane < RD / VEC) {
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const int i = (lane & (RD / VEC / 2 - 1)) * VEC + j;    // rotary index within the pair
      int sec;
      if constexpr (MI) {
        const int r = i % 3;
        sec = (r == 1 && i <= 3 * sec_h) ? 1 : ((r == 2 && i <= 3 * sec_w) ? 2 : 0);
      } else {
        sec = i < sec_t ? 0 : (i < sec_t + sec_h ? 1 : 2);
      }
      const long p = pos[(long)sec * m + tok];
      const float c = (float)cos_sin[p * RD + i];
      const float sn = (float)cos_sin[p * RD + RD / 2 + i];
      const float t = bfmul(other[j], sn);
      n[j] = bfmul(n[j], c) + (lane < RD / VEC / 2 ? -t : t);
    }
  }

#pragma unroll
  for (int r = 0; r < BLK; ++r) {
    Vec4 vo;
#pragma unroll
    for (int j = 0; j < VEC; ++j) vo.h[j] = (__bf16)n[r * VEC + j];
    ((uint2 *)(out + (r * W + lane) * VEC))[0] = vo.u2;
  }
}

// ---- 2. attn_gate_quant ----------------------------------------------------------------------
//
// `attn_output * torch.sigmoid(gate)` then the per-token e4m3 quant o_proj's A side wants. The
// gate is NOT copied out of qkv first: prod's `gate.reshape(...)` materialises a whole [M, nq*HD]
// bf16 tensor per attention layer purely because a chunk of a view is not contiguous. Here the
// kernel reads it in place at a stride, and that copy is gone.
//
//   sig = bf16( 1 / (1 + expf(-f32(g))) )        ATen sigmoid: fp32 math, bf16 result
//   p   = bf16( f32(core) * f32(sig) )           ATen mul: fp32 math, bf16 result
//   sc  = max(amax|f32(p)| / 448, 1/(448*512))   the reference server dynamic_per_token_scaled_fp8_quant
//   q   = cvt_e4m3( clamp(f32(p) / sc, +-448) )  <- divide, per scaled_fp8_conversion<false>
//
// TILED writes the fragment-tiled A layout the A-tiled prefill GEMM reads, the same decision
// `add_rms_quant_t256_tiled{0,1}` makes and at the same band edge.
template <int T, bool TILED>
__device__ __forceinline__ void attn_gate_quant_impl(
    const __bf16 *__restrict__ core,   // [M, N] attention output, contiguous
    const __bf16 *__restrict__ gate,   // the [q|gate] block: head h's gate at +h*2*hd + hd
    unsigned char *__restrict__ q, float *__restrict__ scale,
    int n, int hd, int gs) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  core += (long)row * n;
  gate += (long)row * gs;
  if (!TILED) q += (long)row * n;
  const int mt = row >> 4, lrow = row & 15, kst = n >> 4;
  const int nv = n >> 3;                          // groups of 8
  constexpr int MAXG = 3;                         // widest N this op sees is 6144 (TP=1): 3 groups
  float sv[MAXG][8];
  float amax = 0.f;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * T;
    if (g >= nv) break;
    Vec8 vc;
    vc.u4 = ((const uint4 *)core)[g];
    // `gate` is the [q|gate] BLOCK's base, not the gate's: the gate half of head h starts at
    // h*2*hd + hd, and the kernel adds that itself so the descriptor can bind the whole tensor
    // the projection wrote. The 8 lanes are contiguous there because hd is a multiple of 8.
    const int e = g << 3, h = e / hd, o = e - h * hd;
    Vec8 vg;
    vg.u4 = ((const uint4 *)(gate + (long)h * 2 * hd + hd + o))[0];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      // ATen sigmoid: opmath fp32, the ACCURATE expf (__ocml_exp_f32), then one bf16 round.
      const float sig = bf16r(1.f / (1.f + expf(-(float)vg.h[j])));
      const float p = bf16r((float)vc.h[j] * sig);
      sv[ng][j] = p;
      amax = fmaxf(amax, fabsf(p));
    }
  }
  __shared__ float lds[T / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < T / 32) {
    float v = lds[tid];
    for (int o = T / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  // DIVIDE by 448, as `dynamic_per_token_scaled_fp8_quant` does. The MXFP4 producers multiply by
  // the reciprocal because the reference's own kernel does; this op's reference is the reference server's, and
  // `amax * (1/448)` differs from `amax / 448` in the last bit often enough to move a whole row's
  // e4m3 codes. The first gate run failed on `scale` at offset 0 for exactly this.
  const float sc = fmaxf(lds[0] / 448.f, 1.f / (448.f * 512.f));
  if (tid == 0) scale[row] = sc;
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * T;
    if (g >= nv) break;
    u32 lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) lo |= e4m3_pair(sv[mg][j], sv[mg][j + 1], sc) << (j * 8);
#pragma unroll
    for (int j = 4; j < 8; j += 2) hi |= e4m3_pair(sv[mg][j], sv[mg][j + 1], sc) << ((j - 4) * 8);
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

// ---- 3. kv_append ----------------------------------------------------------------------------
//
// The reference server's `triton_reshape_and_cache_flash` with FP8_KV_CACHE=1 and USE_HEAD_MAJOR_LAYOUT=0, which
// is what `--kv-cache-dtype fp8` selects on this box. The paged cache arrives as one tensor in
// logical [num_blocks, num_kv_heads, block_size, 2*head_size] order; the impl transposes to
// [blocks, slots, heads, 2*hs] and splits the last dim, so K lives at [..., :hs] and V at
// [..., hs:] of the SAME storage, and `value_cache_ptr = key_cache_ptr + hs`.
//
//   slot < 0  -> the token is padding and nothing is written (a real case under cudagraph
//                padding, and the one branch in the kernel that is not arithmetic)
//   k_q = cvt_e4m3( f32(k) / *k_scale )      no clamp: the reference divides and lets the store's
//   v_q = cvt_e4m3( f32(v) / *v_scale )      implicit cast saturate, which is what the hardware
//                                            convert does anyway
//
// One launch writes BOTH K and V, as the reference kernel does; `n_launches` is 1.
template <int HD, int BS>
__device__ __forceinline__ void kv_append_fp8_impl(
    const __bf16 *__restrict__ key,     // [T, nkv, HD]
    const __bf16 *__restrict__ value,   // [T, nkv, HD]
    unsigned char *__restrict__ kv,     // the paged cache, K base; V is kv + HD elements
    const long *__restrict__ slot,      // [T] int64
    const float *__restrict__ k_scale, const float *__restrict__ v_scale,
    long key_stride, long value_stride,
    long block_stride, long page_stride, long head_stride) {
  const int tok = blockIdx.x;
  const int h = blockIdx.y;
  const long s = slot[tok];
  if (s < 0) return;                       // padding token, ignored -- the reference's early out
  const long blk = s / BS, off = s % BS;
  const int tid = threadIdx.x;             // HD/2 threads: each writes one packed e4m3 pair
  const __bf16 *kp = key + (long)tok * key_stride + (long)h * HD;
  const __bf16 *vp = value + (long)tok * value_stride + (long)h * HD;
  unsigned char *base = kv + blk * block_stride + off * page_stride + (long)h * head_stride;
  const int d = tid * 2;
  const u32 kq = e4m3_pair_raw((float)kp[d], (float)kp[d + 1], *k_scale);
  const u32 vq = e4m3_pair_raw((float)vp[d], (float)vp[d + 1], *v_scale);
  *(unsigned short *)(base + d) = (unsigned short)kq;
  *(unsigned short *)(base + HD + d) = (unsigned short)vq;
}

// ---- the entry points ------------------------------------------------------------------------
// One symbol per decision prod makes from a config value or a band edge, never a runtime branch.
// The templates are wrapped BY VALUE for scalars and by pointer for buffers; nothing here is a
// by-const-ref wrapper, because none of these takes a struct (the saber lesson, LEDGER
// "Timing -- saber": a template wrapped by value on a 136-byte struct spilled 8 more bytes of
// scratch per thread than the reference and cost 3%).
#define QKR(name, MI, RED)                                                                        \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *qkv, const __bf16 *qw, const __bf16 *kw, const __bf16 *cos_sin,               \
      const long *pos, __bf16 *q_out, __bf16 *k_out, int m, int nq, int nkv, int qkv_n,           \
      float eps, int sec_t, int sec_h, int sec_w) {                                               \
    qk_norm_rope_impl<256, 64, MI, RED>(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq, nkv,      \
                                        qkv_n, eps, sec_t, sec_h, sec_w);                         \
  }
// Both mrope layouts at every variance order. The order is a GATED choice (see the kernel), and
// the gate sweeps it with `gate_glue.sh --reduction-sweep`, which resolves the same cases under a
// rules file per order -- so every order has to exist for both layouts or the sweep would be
// comparing a layout change as well.
QKR(qk_norm_rope_h256_rot64_mi1_red0, true, 0)
QKR(qk_norm_rope_h256_rot64_mi1_red1, true, 1)
QKR(qk_norm_rope_h256_rot64_mi1_red2, true, 2)
QKR(qk_norm_rope_h256_rot64_mi1_red3, true, 3)
QKR(qk_norm_rope_h256_rot64_mi1_red4, true, 4)
QKR(qk_norm_rope_h256_rot64_mi1_red5, true, 5)
QKR(qk_norm_rope_h256_rot64_mi0_red0, false, 0)
QKR(qk_norm_rope_h256_rot64_mi0_red1, false, 1)
QKR(qk_norm_rope_h256_rot64_mi0_red2, false, 2)
QKR(qk_norm_rope_h256_rot64_mi0_red3, false, 3)
QKR(qk_norm_rope_h256_rot64_mi0_red4, false, 4)
QKR(qk_norm_rope_h256_rot64_mi0_red5, false, 5)

// The DRAFTER's per-layer q/k norm + rope. Its own macro rather than a QKR argument, because the
// kernel runs ONE THREAD PER ELEMENT OF THE HEAD: at HD=128 the workgroup is 128 wide, not 256,
// and `__launch_bounds__` has to say so or every launch wastes half its threads on a barrier they
// cannot leave. `red[HD]` and `tree[HD/4]` shrink with it.
//
// <128, 128, false, 5, false> is the drafter, read off its config.json rather than assumed:
// `head_dim` 128 with no partial_rotary_factor, so ROTARY_DIM == HD == 128; `rope_type: "default"`,
// so PLAIN, and MI=0 with `sec_t >= RD/2` sends every element to position row 0; no
// `attn_output_gate`, so GATED=false and the qkv is a plain [q|k|v]. RED=5 is ATen's order, which
// the reduction sweep settled once for every band.
#define QKR128(name, MI, GATED)                                                                   \
  extern "C" __global__ __launch_bounds__(128) void name(                                         \
      const __bf16 *qkv, const __bf16 *qw, const __bf16 *kw, const __bf16 *cos_sin,               \
      const long *pos, __bf16 *q_out, __bf16 *k_out, int m, int nq, int nkv, int qkv_n,           \
      float eps, int sec_t, int sec_h, int sec_w) {                                               \
    qk_norm_rope_impl<128, 128, MI, 5, GATED>(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq,     \
                                              nkv, qkv_n, eps, sec_t, sec_h, sec_w);              \
  }
QKR128(qk_norm_rope_h128_rot128_mi0_red5, false, false)
#undef QKR128

// The wave-per-head decomposition, at ATen's reduction order only: the sweep settled that
// question once, and a second family of controls would be five more symbols proving the same
// thing. `wg8` is how many heads one workgroup carries, which is `block / 32`.
#define QKRW(name, MI)                                                                            \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *qkv, const __bf16 *qw, const __bf16 *kw, const __bf16 *cos_sin,               \
      const long *pos, __bf16 *q_out, __bf16 *k_out, int m, int nq, int nkv, int qkv_n,           \
      float eps, int sec_t, int sec_h, int sec_w) {                                               \
    qk_norm_rope_wave_impl<256, 64, MI, 256>(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq,     \
                                             nkv, qkv_n, eps, sec_t, sec_h, sec_w);                         \
  }
QKRW(qk_norm_rope_wg8_h256_rot64_mi1_red5, true)
QKRW(qk_norm_rope_wg8_h256_rot64_mi0_red5, false)
#undef QKRW
#undef QKR

#define AGQ(name, TILED)                                                                          \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *core, const __bf16 *gate, unsigned char *q, float *scale,                     \
      int n, int hd, int gs) {                                                                    \
    attn_gate_quant_impl<256, TILED>(core, gate, q, scale, n, hd, gs);                            \
  }
AGQ(attn_gate_quant_t256_tiled0, false)
AGQ(attn_gate_quant_t256_tiled1, true)
#undef AGQ

extern "C" __global__ __launch_bounds__(128) void kv_append_h256_b16_fp8(
    const __bf16 *key, const __bf16 *value, unsigned char *kv, const long *slot,
    const float *k_scale, const float *v_scale, long key_stride, long value_stride,
    long block_stride, long page_stride, long head_stride) {
  kv_append_fp8_impl<256, 16>(key, value, kv, slot, k_scale, v_scale, key_stride, value_stride,
                              block_stride, page_stride, head_stride);
}

// ---- 4. qk_norm_rope + kv_append, ONE dispatch (ws-h38, fusion-map row 7) ---------------------
//
// The decode step ran `qk_norm_rope` and then `kv_append` as two nodes, and the append reads only
// what the rope wrote plus the projection's V. The fused symbols below are the two ops' OWN code,
// called one after the other in one workgroup -- `qk_norm_rope_impl` / `qk_norm_rope_wave_impl`
// unchanged (so q_out and k_out are the same bytes by construction, and every existing symbol's
// ISA is untouched), then `kv_append_row`, which is `kv_append_fp8_impl`'s body with the
// (token, kv head) and the thread stride passed in instead of read off the grid.
//
// WHY NO GRID BARRIER. The two grids are ROW-COMPATIBLE: kv_append's work item (token t, kv head
// h) reads exactly one row of k_out, (t, h), and that row is written by exactly one work item of
// the rope -- (t, nq + h), a K head -- in both decompositions (the head kernel's workgroup
// (blockIdx.x = t, blockIdx.y = nq + h); the wave kernel's wave `t * (nq + nkv) + nq + h`). So the
// append for (t, h) runs INSIDE the rope's workgroup for (t, nq + h), after it, and reads back only
// that workgroup's own stores: one `__syncthreads()` (workgroup-scope release/acquire, which orders
// global stores across the waves of one workgroup) is the whole synchronisation. V is read in place
// out of the projection, which no op of this step writes. q heads append nothing.
//
// Arithmetic: `kv_append` quantises the bf16 k it READS from k_out, i.e. the rope's bf16-rounded
// output; reading the same buffer back gives the same bf16 values, the same `e4m3_pair_raw`
// (divide by *k_scale / *v_scale, no clamp, hardware convert) and the same slot law
// (`slot < 0` -> nothing, blk = s / BS, off = s % BS). Nothing is recomputed.
//
// DROPV = true is the NEGATIVE CONTROL (`--fusion-control rope-append-drop`): the V half is never
// written, which a gate that poisons the step's KV slots before the fused arm must read MISMATCH.
template <int HD, int BS, bool DROPV>
__device__ __forceinline__ void kv_append_row(
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    unsigned char *__restrict__ kv, const long *__restrict__ slot,
    const float *__restrict__ k_scale, const float *__restrict__ v_scale,
    long key_stride, long value_stride, long block_stride, long page_stride, long head_stride,
    int tok, int h, int t0, int nt) {
  const long s = slot[tok];
  if (s < 0) return;                       // padding token, ignored -- the reference's early out
  const long blk = s / BS, off = s % BS;
  const __bf16 *kp = key + (long)tok * key_stride + (long)h * HD;
  const __bf16 *vp = value + (long)tok * value_stride + (long)h * HD;
  unsigned char *base = kv + blk * block_stride + off * page_stride + (long)h * head_stride;
  for (int t = t0; t < HD / 2; t += nt) {  // one packed e4m3 pair of K and one of V per step
    const int d = t * 2;
    const u32 kq = e4m3_pair_raw((float)kp[d], (float)kp[d + 1], *k_scale);
    *(unsigned short *)(base + d) = (unsigned short)kq;
    if (!DROPV) {
      const u32 vq = e4m3_pair_raw((float)vp[d], (float)vp[d + 1], *v_scale);
      *(unsigned short *)(base + HD + d) = (unsigned short)vq;
    }
  }
}

// The per-head decomposition: workgroup (t, head), HD threads. The rope returns from its impl on
// every path (its pass-through lanes `return` from the FUNCTION, not the kernel), so every thread
// reaches the barrier.
template <int HD, int RD, bool MI, int RED, int BS, bool DROPV>
__device__ __forceinline__ void qk_rope_append_head_impl(
    const __bf16 *qkv, const __bf16 *qw, const __bf16 *kw, const __bf16 *cos_sin, const long *pos,
    __bf16 *q_out, __bf16 *k_out, int m, int nq, int nkv, int qkv_n, float eps, int sec_t,
    int sec_h, int sec_w, const __bf16 *value, unsigned char *kv, const long *slot,
    const float *k_scale, const float *v_scale, long key_stride, long value_stride,
    long block_stride, long page_stride, long head_stride) {
  qk_norm_rope_impl<HD, RD, MI, RED>(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq, nkv, qkv_n,
                                     eps, sec_t, sec_h, sec_w);
  __syncthreads();
  const int head = blockIdx.y;
  if (head < nq) return;
  kv_append_row<HD, BS, DROPV>(k_out, value, kv, slot, k_scale, v_scale, key_stride, value_stride,
                               block_stride, page_stride, head_stride, blockIdx.x, head - nq,
                               threadIdx.x, HD);
}

// The wave decomposition: wave w of workgroup b is pair b * WAVES + w; a K-head pair appends its
// own row with its own 32 lanes, four pairs of dims each.
template <int HD, int RD, bool MI, int THREADS, int BS, bool DROPV>
__device__ __forceinline__ void qk_rope_append_wave_impl(
    const __bf16 *qkv, const __bf16 *qw, const __bf16 *kw, const __bf16 *cos_sin, const long *pos,
    __bf16 *q_out, __bf16 *k_out, int m, int nq, int nkv, int qkv_n, float eps, int sec_t,
    int sec_h, int sec_w, const __bf16 *value, unsigned char *kv, const long *slot,
    const float *k_scale, const float *v_scale, long key_stride, long value_stride,
    long block_stride, long page_stride, long head_stride) {
  qk_norm_rope_wave_impl<HD, RD, MI, THREADS>(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq, nkv,
                                              qkv_n, eps, sec_t, sec_h, sec_w);
  __syncthreads();
  constexpr int W = 32, WAVES = THREADS / W;
  const int lane = (int)(threadIdx.x & (W - 1));
  const int wave = (int)(threadIdx.x >> 5);
  const int heads = nq + nkv;
  const long pair = (long)blockIdx.x * WAVES + wave;
  if (pair >= (long)m * heads) return;
  const int tok = (int)(pair / heads), head = (int)(pair - (long)tok * heads);
  if (head < nq) return;
  kv_append_row<HD, BS, DROPV>(k_out, value, kv, slot, k_scale, v_scale, key_stride, value_stride,
                               block_stride, page_stride, head_stride, tok, head - nq, lane, W);
}

#define QKRA(name, KIND, MI, DROPV)                                                               \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *qkv, const __bf16 *qw, const __bf16 *kw, const __bf16 *cos_sin,               \
      const long *pos, __bf16 *q_out, __bf16 *k_out, int m, int nq, int nkv, int qkv_n,           \
      float eps, int sec_t, int sec_h, int sec_w, const __bf16 *value, unsigned char *kv,         \
      const long *slot, const float *k_scale, const float *v_scale, long key_stride,              \
      long value_stride, long block_stride, long page_stride, long head_stride) {                 \
    KIND(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq, nkv, qkv_n, eps, sec_t, sec_h, sec_w,    \
         value, kv, slot, k_scale, v_scale, key_stride, value_stride, block_stride, page_stride,  \
         head_stride);                                                                            \
  }
#define QKRA_HEAD(MI, DROPV) qk_rope_append_head_impl<256, 64, MI, 5, 16, DROPV>
#define QKRA_WAVE(MI, DROPV) qk_rope_append_wave_impl<256, 64, MI, 256, 16, DROPV>
QKRA(qk_rope_append_h256_rot64_mi1_red5_b16_fp8, QKRA_HEAD(true, false), true, false)
QKRA(qk_rope_append_h256_rot64_mi0_red5_b16_fp8, QKRA_HEAD(false, false), false, false)
QKRA(qk_rope_append_wg8_h256_rot64_mi1_red5_b16_fp8, QKRA_WAVE(true, false), true, false)
QKRA(qk_rope_append_wg8_h256_rot64_mi0_red5_b16_fp8, QKRA_WAVE(false, false), false, false)
// The negative controls, at the config's (interleaved) layout only.
QKRA(qk_rope_append_h256_rot64_mi1_red5_b16_fp8_dropv, QKRA_HEAD(true, true), true, true)
QKRA(qk_rope_append_wg8_h256_rot64_mi1_red5_b16_fp8_dropv, QKRA_WAVE(true, true), true, true)
#undef QKRA_WAVE
#undef QKRA_HEAD
#undef QKRA
