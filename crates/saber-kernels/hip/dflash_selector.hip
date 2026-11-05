// The DFlash2 drafter's candidate selector: `_score_edges` as one kernel.
//
// Prod runs this as `qwen3_dflash2.py:181`, inside a `@support_torch_compile` module, which the
// census saw as two codebook gathers, a broadcast multiply, a batched bf16 GEMM through hipBLASLt
// and a bias add. What it computes, for each (sequence b, block position l):
//
//   pred_id[p] = l == 0 ? anchor[b] : candidate_ids[b][l-1][p]
//   A[p][r]    = bf16( predecessor_codebook[pred_id[p]][r] * hidden[b][l][r] )
//   out[p][c]  = f32( bf16( sum over r of A[p][r] * successor_codebook[cand[b][l][c]][r] )
//                     + unary_logits[b][l][c] )
//
// R is `selector_rank` = 256 and P = C = `selector_top_k` = 16, so ONE WAVE does all of it: the
// 16x16 output tile is a single WMMA fragment and the 256-long reduction is 16 k-steps of it. The
// gathers never become tensors -- a lane simply reads the codebook row its own index names -- so
// the two [vocab, 256] tables are touched for 32 rows per position instead of being indexed into
// an intermediate.
//
// THE ROUNDINGS ARE THE GATE, and all three are forced by the reference being a LIBRARY GEMM with
// torch ops on either side: `A` is a real bf16 tensor because hipBLASLt has to read it, the
// einsum's result is a real bf16 tensor because hipBLASLt writes bf16, and only then is the bias
// added. Accumulating the bias inside the fp32 accumulator would be a different answer.
//
// AND SO ARE THE ELEMENT WIDTHS -- this kernel's second defect and the more expensive one. Prod's
// tensors are `torch.topk`'s `int64` candidate ids, an `f32` `unary_logits` (the exact rerank's
// output) and therefore an `f32` score tensor, because `f32 + bf16` is `f32` in torch and that add
// is the last thing `_score_edges` does. Until 2026-09-21 this signature said `i32` candidates,
// `bf16` unary and `bf16` out -- the widths `capture_dflash_sel_std.py` happened to build, which is
// the walk's `int32` defect (LEDGER F round 22) in a second place: a kernel and its reference
// agreeing at widths the serve does not use. Measured on the host against the r23 capture, the
// three wrong widths are the WHOLE of that gate's red row: emulating this kernel byte for byte in
// CPU numpy reproduces 5007 / 5377 / 5133 of 7168 differing bytes exactly, and restoring the three
// widths takes it to 0 (LEDGER F round 24).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));

#define DS_R      256   // selector_rank
#define DS_K       16   // selector_top_k, and the WMMA tile in both M and N
#define DS_STEPS  (DS_R / 16)
#define DS_UNROLL   8   // k-steps in flight

__device__ __forceinline__ floatx8 ds_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

extern "C" __global__ __launch_bounds__(32) void dflash_selector_score_r256_k16(
    const __bf16 *__restrict__ pred_cb,    // [vocab, R]
    const __bf16 *__restrict__ succ_cb,    // [vocab, R]
    const long long *__restrict__ cand,    // [B, L, K] candidate ids, torch.topk's INT64
    const int *__restrict__ anchor,        // [B] the anchor token per sequence, i32 `input_ids`
    const float *__restrict__ unary,       // [B, L, K] the exact rerank's f32 logits
    const __bf16 *__restrict__ hidden,     // [B, L, R] the projection's output
    float *__restrict__ out,               // [B, L, K, K] f32: bf16 dot + f32 bias is f32
    int bl, int l_len, int r, int k_top) {
  const int pos = (int)blockIdx.x;                 // one workgroup per (b, l)
  if (pos >= bl) return;
  const int lane = (int)threadIdx.x;
  const int row = lane & 15, half = lane >> 4;
  const int l = pos % l_len;

  // The gather, as an index rather than a tensor. `predecessor_ids` is the anchor at l == 0 and
  // the PREVIOUS position's candidates otherwise -- prod's `cat(anchor.expand(...), cand[:, :-1])`.
  const long pid =
      l == 0 ? (long)anchor[pos / l_len] : (long)cand[(long)(pos - 1) * k_top + row];
  const long cid = (long)cand[(long)pos * k_top + row];
  const __bf16 *pr = pred_cb + (long)pid * r + half * 8;
  const __bf16 *sr = succ_cb + (long)cid * r + half * 8;
  const __bf16 *hr = hidden + (long)pos * r + half * 8;

  floatx8 acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;

  for (int k0 = 0; k0 < DS_STEPS; k0 += DS_UNROLL) {
    bf16x8 pv[DS_UNROLL], sv[DS_UNROLL], hv[DS_UNROLL];
#pragma unroll
    for (int u = 0; u < DS_UNROLL; ++u) {
      pv[u] = *(const bf16x8 *)(pr + (k0 + u) * 16);
      sv[u] = *(const bf16x8 *)(sr + (k0 + u) * 16);
      hv[u] = *(const bf16x8 *)(hr + (k0 + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < DS_UNROLL; ++u) {
      bf16x8 av;
#pragma unroll
      for (int e = 0; e < 8; ++e) av[e] = (__bf16)((float)pv[u][e] * (float)hv[u][e]);
      acc = ds_wmma(av, sv[u], acc);
    }
  }

  // Accumulator element `e` of lane `l` is row (l >> 4) * 8 + e, column l & 15 -- so `p` comes out
  // of the element index and `c` out of the lane.
  //
  // THE BIAS IS INDEXED BY `c`, NOT `p`. `unary_logits[:, :, None]` makes a [B, L, 1, top_k] tensor
  // out of a [B, L, top_k] one, so it broadcasts along the PREDECESSOR axis and lines up with the
  // successor axis -- the unary score belongs to the token the edge arrives at. Reading it at `p`
  // is a transpose of the bias and it type-checks perfectly.
  const int c = row;
  const float bias = unary[(long)pos * k_top + c];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int p = half * 8 + e;
    const float g = (float)(__bf16)acc[e];         // hipBLASLt writes bf16; the bias sees THAT
    // ... and the SUM is f32 and stays f32. `unary_logits[:, :, None] + einsum(...)` is
    // `f32 + bf16`, which torch types f32, which is why `walk.scores` is an f32 tensor.
    out[((long)pos * k_top + p) * k_top + c] = g + bias;
  }
}
