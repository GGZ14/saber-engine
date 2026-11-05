// The drafter's CONTEXT-KV store: the two ops it calls as `upstream::` C++ kernels rather than
// through a module.
//
// `precompute_and_store_context_kv` (`qwen3_dflash.py:585`) fills the drafter's KV cache from the
// TARGET's hidden states, once a step, and it is the one place in the drafter that reaches past a
// module and invokes `torch.ops._C` directly:
//
//   ops.rms_norm(normed, context_states, hidden_norm_weight, eps)      -- one launch, 1 group
//   F.linear(...)                                                       -- the fused KV GEMM
//   ops.rms_norm(all_k_normed, all_k, k_norm_weights, eps)              -- one launch, L GROUPS
//   ops.rotary_embedding(pos, all_k_flat, None, head_size, cache, neox) -- one launch, K only
//
// WHY THIS IS ITS OWN TRANSLATION UNIT, and not rows in `dflash_mlp.hip` or an `attn_glue`
// instantiation. Every other norm and rope on the drafter's path has an INDUCTOR-COMPILED TRITON
// reference, because the serve sets `custom_ops: [..., "none"]` and the modules take their native
// path. These two call sites bypass that toggle, so their reference is the C++ kernel in
// `csrc/libtorch_stable/layernorm_kernels.hip` and `pos_encoding_kernels.hip` -- a different
// program with different roundings, read here off the gfx1201 code object inside prod's own
// `_C_stable_libtorch.abi3.so` (`llvm-objdump`, 2026-09-21). Two concrete consequences:
//
//   * `attn_glue` is compiled `-ffp-contract=off`, because ITS reference is Triton and Triton
//     emits a separate `mulf` and `subf` for `a*c - b*s`. The C++ rope CONTRACTS -- its ISA is
//     `v_mul_f32` + `v_fma_f32` -- so the same file cannot host both, and the flag is TU-wide.
//     (The same trap the LEDGER records for `silu_quant`'s `exp`.)
//   * `dflash_mlp`'s `dflash_rms_norm` is the INDUCTOR lowering of `ir.ops.rms_norm`: a Triton
//     reduction over a fused region inductor keeps in fp32. The C++ kernel below does a CUB block
//     reduction. Both round ONCE (F29 removed the second rounding from `dflash_mlp`'s pair: the
//     `(bf16)(x * rsqrt) * weight` of the PYTHON and of this C++ kernel's own sibling
//     `fused_add_rms_norm` is not what the compiled drafter runs), so what separates them now is
//     the reduction and the `rsqrt`, not the epilogue -- and that is still two answers on the
//     same input, so this is a second kernel and not a second caller of the first.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// A 16-byte bf16 vector, for the layout copy at the bottom of this file. Named for this
// translation unit so it cannot collide with another plugin's.
typedef __bf16 bf16x8_ctx __attribute__((ext_vector_type(8)));

#define DC_MAX_BLOCK 1024
#define DC_WAVE        32

__device__ __forceinline__ float dc_f32(__bf16 v) { return (float)v; }

// ---- the block reduction the reference uses ----------------------------------------------------
//
// NOT `upstream::blockReduceSum`'s shfl_XOR ladder, and the difference is the whole gate: the ISA
// carries five `ds_bpermute_b32` at lane+1, +2, +4, +8, +16 each guarded by
// `lane + offset < valid`, then one LDS round trip, then the same five over the warp aggregates.
// That is `cub::BlockReduce<float, 1024>`'s WARP_REDUCTIONS algorithm -- a shfl_DOWN tree with an
// ascending offset and a validity guard -- and for fp32 addition the order is the answer.
//
// `valid` is `blockDim.x`, which CUB is handed as `num_valid`; the second stage's is the warp
// count. For a power-of-two lane count this is the balanced binary tree in element order, and a
// lane whose partial is an exact zero contributes nothing, which is why the result does not depend
// on the block width so long as one thread covers at most one vector.
__device__ __forceinline__ float dc_wave_reduce(float v, int valid, int lane) {
#pragma unroll
  for (int off = 1; off < DC_WAVE; off <<= 1) {
    const float o = __shfl_down(v, off, DC_WAVE);
    if (lane + off < valid) v += o;
  }
  return v;
}

// ---- `upstream::rms_norm_kernel<bf16, WIDTH, NDIM, true>` -------------------------------------------
//
// One workgroup per row. `weight` is [groups, h] and the row's group is `row / rows_per_group` --
// the drafter's K-norm stacks the five layers' `k_norm.weight` into one contiguous [L, head_dim]
// buffer so the per-layer norm runs as a single grouped launch over [L, num_ctx, nkv, head_dim],
// and the hidden norm is the same kernel with one group.
//
// THE EPILOGUE, and it is the one that reads off the ISA rather than the source:
//
//     out = bf16( x * rsqrt(mean + eps) * weight )
//
// with BOTH multiplies in fp32 and ONE conversion, where the SOURCE of `fused_add_rms_norm` and
// of `ir.ops.rms_norm` writes `((scalar_t)(x * s)) * weight` and rounds twice. The ISA is
// explicit: `v_mul_f32 v10, v6, v10` then `v_mul_f32 v10, v10, v22` and only then the RTNE
// `v_add3_u32 ..., 0x7fff` pair. (`dflash_mlp`'s pair rounds once as well, for the other reason:
// inductor elides the intermediate round when it decomposes the custom op. Two references, one
// answer -- which is a coincidence and not a rule, and each is read off its own program.)
//
// `rsqrtf`, not `__builtin_amdgcn_rsqf`: the reference's sequence is the OCML wrapper (scale by
// 2^24 below the smallest normal, `v_rsq_f32`, scale back by 2^12). The two agree for every
// normal argument and `mean + eps >= eps`, but the wrapper is what the reference compiled and
// there is no reason to write the other one.
//
// The division is a real division. `variance / (float)h` compiles to the IEEE
// `v_div_scale`/`v_rcp`/`v_div_fmas`/`v_div_fixup` expansion in both builds; the reciprocal
// multiply that `silu_quant`'s scale needs is the opposite case and is not this one.
template <int W>
__device__ __forceinline__ void dc_rms_body(const __bf16 *__restrict__ x,
                                            const __bf16 *__restrict__ weight,
                                            __bf16 *__restrict__ out, int h, int rows_per_group,
                                            float eps) {
  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int nthreads = (int)blockDim.x;
  const int lane = tid & (DC_WAVE - 1);
  const int wid = tid / DC_WAVE;
  const int nwaves = (nthreads + DC_WAVE - 1) / DC_WAVE;
  const int nvec = h / W;
  const __bf16 *xr = x + (long)row * h;
  const __bf16 *wr = weight + (long)(row / rows_per_group) * h;
  __bf16 *orow = out + (long)row * h;

  __shared__ float wave_sums[DC_MAX_BLOCK / DC_WAVE];
  __shared__ float s_rsqrt;

  // Pass 1: the sum of squares, one fp32 accumulator per thread, `W` contiguous elements at a
  // time IN ELEMENT ORDER. `fmaf` because the reference's loop body is `v_fmac_f32` per element.
  float acc = 0.f;
  for (int i = tid; i < nvec; i += nthreads) {
    const __bf16 *p = xr + (long)i * W;
#pragma unroll
    for (int j = 0; j < W; ++j) {
      const float v = dc_f32(p[j]);
      acc = fmaf(v, v, acc);
    }
  }
  acc = dc_wave_reduce(acc, nthreads - wid * DC_WAVE, lane);
  if (lane == 0) {
    wave_sums[wid] = acc;
  }
  __syncthreads();
  if (tid < DC_WAVE) {
    // A lane past the wave count is read as an exact zero rather than out of uninitialised LDS;
    // the reference reads the slot and masks the add, which is the same value and one more load.
    float a = (lane < nwaves) ? wave_sums[lane] : 0.f;
    a = dc_wave_reduce(a, nwaves, lane);
    if (lane == 0) {
      s_rsqrt = rsqrtf(a / (float)h + eps);
    }
  }
  __syncthreads();
  const float s = s_rsqrt;

  // Pass 2: re-read the row -- the reference does too -- and round ONCE.
  for (int i = tid; i < nvec; i += nthreads) {
    const __bf16 *p = xr + (long)i * W;
    const __bf16 *q = wr + (long)i * W;
    __bf16 *o = orow + (long)i * W;
#pragma unroll
    for (int j = 0; j < W; ++j) {
      o[j] = (__bf16)(dc_f32(p[j]) * s * dc_f32(q[j]));
    }
  }
}

// The five vector widths the reference instantiates (`vectorize_read_with_alignment<W, ...>` is in
// the code object at W = 1, 2, 4, 8 and 16 for every dtype). Which one the host picks is a fact
// about prod's launcher, not about this kernel, so all five are here: the gate names the one that
// reproduces prod's bytes and the other four are its negative controls -- they must differ, and a
// round in which they do not is a round in which the gate proved nothing.
#define DC_RMS(W)                                                                                  \
  extern "C" __global__ __launch_bounds__(DC_MAX_BLOCK) void dflash_rms_norm_grouped_w##W(         \
      const __bf16 *__restrict__ x, const __bf16 *__restrict__ weight, __bf16 *__restrict__ out,   \
      int rows, int h, int rows_per_group, float eps) {                                            \
    if ((int)blockIdx.x >= rows) return;                                                           \
    dc_rms_body<W>(x, weight, out, h, rows_per_group, eps);                                        \
  }

DC_RMS(1)
DC_RMS(2)
DC_RMS(4)
DC_RMS(8)
DC_RMS(16)
#undef DC_RMS

// ---- `upstream::rotary_embedding_kernel<bf16, bf16, true>`, K only and in place ---------------------
//
// `precompute_and_store_context_kv` views the normed context K as [L * num_ctx, kv] and applies
// ONE rope to all five layers at once, with the positions repeated L times, IN PLACE, and with
// `key = nullptr` so K rides in the kernel's query slot. That is why this is an entry point of its
// own rather than an `attn_glue` band row: `qk_norm_rope` norms first and writes q and k to
// separate outputs, and this call has already normed and has no q.
//
// NEOX, because the drafter's `rotary_emb.is_neox_style` is true: the pair is (i, i + rot_dim/2)
// and both halves take `cos[i]` / `sin[i]`, against GPT-J's interleaved (2i, 2i+1).
//
// THE ARITHMETIC, read off the ISA rather than off the source, because the source's shape does not
// say where the compiler contracted:
//
//     t1 = sin * y ;  out_x = fma(x, cos, -t1)
//     t2 = cos * y ;  out_y = fma(sin, x,  t2)
//
// One rounding per output, and one of the two products in each expression kept as a separate
// `v_mul_f32`. Written with `fmaf` and an explicit temporary so the result does not depend on
// which multiply hipcc would have chosen to fuse -- the flag is TU-wide and this file's other
// kernels must not inherit a decision made for this one.
//
// `cos_sin_cache` is bf16 here: the caller casts it to the K tensor's dtype before the call
// (`if cos_sin_cache.dtype != all_k_flat.dtype: cos_sin_cache.to(...)`), so the cache's own fp32
// values are rounded ONCE, before the kernel, and this kernel reads bf16.
extern "C" __global__ __launch_bounds__(512) void dflash_rope_neox_k_inplace(
    const long *__restrict__ positions,   // [rows], int64 as the reference server stores them
    __bf16 *__restrict__ k,               // [rows, stride], rotated IN PLACE
    const __bf16 *__restrict__ cos_sin,   // [max_position, rot_dim], cos then sin
    int rows, int heads, int head_size, int rot_dim, int stride) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const int half = rot_dim / 2;
  const int nq = heads * half;
  const __bf16 *cache = cos_sin + positions[row] * (long)rot_dim;
  __bf16 *base = k + (long)row * stride;
  for (int i = (int)threadIdx.x; i < nq; i += (int)blockDim.x) {
    const int head = i / half;
    const int ro = i - head * half;
    __bf16 *arr = base + (long)head * head_size;
    const float c = dc_f32(cache[ro]);
    const float s = dc_f32(cache[half + ro]);
    const float xv = dc_f32(arr[ro]);
    const float yv = dc_f32(arr[half + ro]);
    const float t1 = s * yv;
    const float t2 = c * yv;
    arr[ro] = (__bf16)fmaf(xv, c, -t1);
    arr[half + ro] = (__bf16)fmaf(s, xv, t2);
  }
}

// ---- the context store's LAYOUT copy ------------------------------------------------------------
//
// `all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()` -- an ATen copy
// that computes NOTHING, and a launch prod makes. The plan called it "a fusion to take rather than
// a kernel to port" for three rounds and left it absent; that reading was half right and half a
// gap. It IS a fusion to take -- a Saber store would have the projection write layer-major -- but
// while the projection is `bgemm_bf16` writing `[num_ctx, L * 2 * kv_size]` row-major, the copy is
// the only thing between it and the K norm, and an absent op in a CHAIN contributes zeros.
//
// So it is here, as the permutation it is. Nothing is computed, nothing is rounded, and the gate
// against prod's own tensor is exact BY CONSTRUCTION rather than by arithmetic -- which is also
// why its control is an axis swap and not a rounding.
//
//   out[c][l][t][h][d] = in[t][l][c][h][d],   c in {0,1} (K then V), l < L, t < num_ctx
//
// The innermost `nkv * hd` elements are contiguous on BOTH sides, so one workgroup copies one such
// run and the grid is `2 * L * num_ctx`. 1024 bf16 at TP=1, 512 at TP=2; `vec` is 8 elements
// (16 B) per thread, so 128 threads cover a TP=1 run in one pass.
extern "C" __global__ __launch_bounds__(256) void dflash_ctx_kv_layout(
    const __bf16 *__restrict__ in,   // [num_ctx, L, 2, nkv, hd]
    __bf16 *__restrict__ out,        // [2, L, num_ctx, nkv, hd]
    int num_ctx, int layers, int nkv, int hd) {
  const int run = nkv * hd;                       // the contiguous tail, both sides
  const int g = (int)blockIdx.x;
  const int t = g % num_ctx;
  const int l = (g / num_ctx) % layers;
  const int c = g / (num_ctx * layers);
  if (c >= 2) return;
  const __bf16 *src = in + ((((long)t * layers + l) * 2) + c) * run;
  __bf16 *dst = out + (((long)c * layers + l) * num_ctx + t) * run;
  for (int e = (int)threadIdx.x * 8; e < run; e += (int)blockDim.x * 8) {
    if (e + 8 <= run) {
      *(bf16x8_ctx *)(dst + e) = *(const bf16x8_ctx *)(src + e);
    } else {
      for (int i = e; i < run; ++i) dst[i] = src[i];
    }
  }
}

// CONTROL: the axis order a reader would write from the `.permute(2, 1, 0, 3, 4)` call without
// tracing which index the VIEW put where -- layer-major and context-minor swapped. It reads and
// writes exactly the same bytes in a different order, so it can only be told from the route by the
// reference, which is the point of a control here.
extern "C" __global__ __launch_bounds__(256) void dflash_ctx_kv_layout_swapped(
    const __bf16 *__restrict__ in, __bf16 *__restrict__ out,
    int num_ctx, int layers, int nkv, int hd) {
  const int run = nkv * hd;
  const int g = (int)blockIdx.x;
  const int t = g % num_ctx;
  const int l = (g / num_ctx) % layers;
  const int c = g / (num_ctx * layers);
  if (c >= 2) return;
  const __bf16 *src = in + ((((long)l * num_ctx + t) * 2) + c) * run;
  __bf16 *dst = out + (((long)c * layers + l) * num_ctx + t) * run;
  for (int e = (int)threadIdx.x * 8; e < run; e += (int)blockDim.x * 8) {
    if (e + 8 <= run) {
      *(bf16x8_ctx *)(dst + e) = *(const bf16x8_ctx *)(src + e);
    } else {
      for (int i = e; i < run; ++i) dst[i] = src[i];
    }
  }
}
