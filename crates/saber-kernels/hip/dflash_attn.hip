// The DFlash2 drafter's attention.
//
// NOT a port, and not reachable by instantiating the saber one either: `saber_attn` is templated on
// HEAD_DIM and GQA, so the drafter's 128 and 4 would be a new instantiation, but its KV layout is
// ONE paged buffer with K and V interleaved per token (`(kk % BS) * (2 * HEAD_DIM)`, V at
// `+ HEAD_DIM`) while the drafter reads the reference server's FLASH layout -- separate `[blocks, block_size,
// kv_heads, head_size]` caches -- and the file contains no notion of a sliding window at all. Two
// of the four differences are absent semantics, not parameters. So this is written from the reference server's
// Triton `kernel_unified_attention` and gated bit-exact against it.
//
// PROD'S CONFIGURATION, every line of it read out of source or the trace rather than assumed:
//   causal = FALSE          `spec_decode/dflash.py:79` sets `dflash_causal = not
//                           dflash_has_any_non_causal(...)`, and the checkpoint says
//                           `is_causal: false`. The window is therefore SYMMETRIC.
//   SLIDING_WINDOW = 2048   the backend passes `window_size = (sliding_window - 1, 0)` and the
//                           launcher takes `1 + window_size[0]`
//   BLOCK_M 16, BLOCK_Q 4   `16 if num_queries_per_kv <= 16`, then `BLOCK_M // 4`
//   TILE_SIZE 32            and this line was WRONG until 2026-09-21. It read
//                           `_get_tile_size(128, 2048, element_size, is_prefill=False)` -- 16 for
//                           bf16 KV, 32 for fp8 -- but `unified_attention` computes BOTH tiles and
//                           hands the 2D path `TILE_SIZE_PREFILL`, which is 32 at every head size
//                           that is not Gemma3's AND AT EVERY ELEMENT SIZE (`is_prefill=True`
//                           returns 32 before it ever looks at `element_size`;
//                           `triton_unified_attention.py:784-799`, and the 2D branch at :1077-1079
//                           is the only reader). The fp8 instantiation is `t32` and right by the
//                           fp8 branch; `bf16kv_t16` was the WRONG tile for this path and is kept
//                           only as the tile CONTROL -- the bf16 route binds `bf16kv_t32` since
//                           2026-09-21 and `DflashRules::validate` refuses any attention row whose
//                           symbol does not end in the 2D tile. MEASURED against prod's own live
//                           call (F round 24): the fp8 kernel forced to TILE 16 differs in
//                           14,306 bytes of 983,040 and the reference forced to 16 in 14,295,
//                           against 30 at TILE 32.
//   HEAD_SIZE 128, GQA 4    32 q heads / 8 kv, halved by TP=2
//   2D, no segments         the census contains NO `reduce_segments`, and prod's own captured
//                           metadata settles why: `use_3d` is false whenever `max_seqlen_q > 1`,
//                           and the drafter's block is 8 query rows at every step. Reading
//                           `seq_threshold_3D = 64` as a sequence-LENGTH bound -- and concluding
//                           "SHORT-CONTEXT drafting only" -- was wrong: it is a sequence COUNT,
//                           and the captured steps run 270 / 286 / 273 keys on the 2D path. A
//                           batch of more than `seq_threshold_3D` SEQUENCES would need the segment
//                           phase; a long context does not.
//   everything else off     no sinks, no alibi, no softcap, no qq_bias, no mm_prefix, no R-SWA,
//                           no chunked attention, no tensor descriptors
//
// ONE SEQUENCE PER LAUNCH, OR A WALK -- and the file carries both, as two symbols. The
// one-sequence entry points (`..._t32`, `..._t16`) are what every gate in this tree byte-compared
// against prod's own live call: the program id IS the local q-block index, the length is
// `seq_lens[0]` and the block table has no row stride. The `..._t32_ns` pair added in ws-s11
// takes `q_per_seq` and walks `seq = qblock * BLOCK_Q / q_per_seq` -- exact, because the drafter's
// block is a multiple of BLOCK_Q -- so a batched proposal is ONE launch instead of `reqs`. `NS` is
// a template parameter, so the one-sequence instantiations are the same programs they were.
//
// GRID. The reference's is `(q.shape[0] // BLOCK_Q + num_seqs) x num_kv_heads` = 3 x 4 for the
// drafter's 8 query rows and one sequence, at 128 threads -- 12 programs of 4 waves. This kernel
// launches 3 workgroups of 4 waves and gives **one wave each (q_block, kv_head)**, so it runs the
// same 12 independent pieces of work with each piece's arithmetic entirely inside one wave. That is
// deliberate: every cross-wave split of this loop would re-associate either the head reduction or
// the running softmax, and there is no way to be bit-exact through a re-association.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define DA_BM    16    // BLOCK_M
#define DA_BQ     4    // BLOCK_Q = BLOCK_M / num_queries_per_kv
#define DA_NQPK   4    // num_queries_per_kv
#define DA_HEAD 128    // HEAD_SIZE
#define DA_NW     4    // waves per workgroup; wave w takes kv head w
#define DA_NT   (DA_NW * 32)
#define DA_NF   (DA_HEAD / 16)   // 8 output fragments of 16 head dims

__device__ __forceinline__ floatx8 da_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// `tl.exp`, READ OUT OF THE REFERENCE'S OWN IR -- and the reading was WRONG until 2026-09-21.
//
// The old note said Triton lowers `tl.exp` to `llvm.exp.f32` and inferred that `__builtin_expf`
// would therefore get the same expansion. It does not. Triton's AMD backend rewrites `math.exp`
// into exp2 BEFORE LLVM sees it: the cached `.llir` for prod's own specialisation carries
//
//     %a = fmul float %x, 0x3FF7154760000000      ; float(log2 e) = 0x1.715476p+0f
//     %b = call float @llvm.exp2.f32(float %a)
//
// at `triton_attention_helpers.py:433` (`P = tl.exp(S - m_j[:, None])`) and `:437` (`alpha`), and
// the `.amdgcn` expands THAT to the denormal-safe exp2 and nothing else:
//
//     v_cmp_gt_f32 0xc2fc0000(-126), x ; v_cndmask 0 / 0x42800000(64) ; v_fmac 0x3fb8aa3b, x
//     v_exp_f32 ; v_ldexp_f32 by 0 / -64
//
// `__builtin_expf` is a DIFFERENT function: it is `llvm.exp.f32`, whose AMDGPU expansion range-
// reduces with a two-term log2(e) (`v_rndne_f32`, `v_fma 0x3fb8aa3b`, `v_fmac 0x32a5705f` -- the
// low half of log2 e -- then `v_exp_f32` + `v_ldexp_f32` by the integer part, plus overflow and
// underflow clamps at 0x42b17218 / 0xc2ce8ed0). It is the more accurate of the two and its last
// bit is not the reference's. Compiling this file both ways and diffing the ISA is how that was
// settled, on the host, before any card time: the form below emits the reference's six
// instructions, `__builtin_expf` emits the other eleven.
//
// This one line was the whole of the kernel's 30-byte residual against prod (LEDGER F round 25).
// `exp2f()` would be ocml's library function, a third answer again.
__device__ __forceinline__ float da_exp(float x) {
  return __builtin_exp2f(x * 0x1.715476p+0f);
}

// e4m3 -> fp32 with the HARDWARE convert, one instruction per byte. This is safe in a way the
// rerank's decoder was not: fp8 -> fp32 is LOSSLESS, so there is no rounding to match and any
// correct decode gives the reference's bits -- which is why the hardware instruction may be used
// here while the rerank had to spell the arithmetic out. It takes the byte from a dword by index,
// so a four-byte group costs four converts and no shifts.
//
// The software version this replaced -- mantissa/exponent arithmetic with an `exp2` per byte -- was
// worth 60 us of the fp8 arm's time on the same values.
template <int BYTE>
__device__ __forceinline__ float da_fp8x(unsigned int packed) {
  return __builtin_amdgcn_cvt_f32_fp8(packed, BYTE);
}

// THE REDUCTION LAYOUT IS THE REFERENCE'S, and it is not the WMMA accumulator's. The disassembly
// reduces each row with EIGHT in-register `v_max3_num_f32` and then ONE `v_permlanex16_b32` -- so a
// lane holds eight values ALONG THE REDUCED AXIS (the KV tile) for ONE query row, and the two
// 16-lane halves hold the two halves of the tile. That is the A-fragment layout, which the WMMA
// accumulator has to be transposed into (the reference does the same thing: its `ds_store_b128` /
// `ds_load_b128` pairs are exactly this conversion).
//
// Matching it is not optional for the SUM: an in-lane sequential fold of 8 values plus one
// cross-half add is a different tree from a 16-lane butterfly, and the draft head's rerank cost a
// gate round to precisely this distinction. The MAX is order-independent, so only the sum's shape
// matters, but both are written the same way here so the code says which layout it is in.
template <int N>
__device__ __forceinline__ float da_tile_max(const float *v) {
  float m = v[0];
#pragma unroll
  for (int i = 1; i < N; ++i) m = fmaxf(m, v[i]);
  return fmaxf(m, __shfl_xor(m, 16, 32));
}
// THIS FOLD IS THE REFERENCE'S TREE AT TILE 32, read out of the compiled kernel on 2026-09-21 and
// no longer inferred from the tile-16 note. Prod's own specialisation (bf16 q/out, e4m3 KV,
// BLOCK_M 16, TILE 32, 4 warps, gfx1201 wave32) reduces `tl.sum(P, axis=1)` in the WMMA
// accumulator layout -- the `.ttgir` reduces `tensor<16x32xf32, #ttg.amd_wmma<{version = 2,
// isTranspose = true, warpsPerCTA = [4, 1]}>` over axis 1, which is 2 N-reps x 8 elements = SIXTEEN
// values per lane along the reduced axis, one query row per lane, the two 16-lane halves holding
// the two halves of nothing (the reps are in-lane; the halves are the two column groups of each
// rep). The `.llir` and the `.amdgcn` then say the shape exactly:
//
//     %a = fadd float %v0, %v1 ; %b = fadd float %v2, %a ; ... 15 adds, accumulator on the RIGHT
//     v_permlanex16_b32 (ONE) ; one final v_add_f32
//
// which is this function: a sequential left-to-right fold of the lane's sixteen values and one
// cross-half add. Operand order inside an `fadd` is not a difference (IEEE addition is
// commutative bit for bit), the association is what matters, and it matches. `v_sumtree` (a
// balanced pairwise fold) was F24's evidence that the tree is a real lever -- it moved the
// residual from 30 to 32 bytes -- and the residual was never this fold: it was `da_exp`, see
// above. Changing this to any other tree will re-open a gate that is now 0 of 983,040.
template <int N>
__device__ __forceinline__ float da_tile_sum(const float *v) {
  float s = v[0];
#pragma unroll
  for (int i = 1; i < N; ++i) s += v[i];       // the fold order, left to right
  return s + __shfl_xor(s, 16, 32);            // ... then the one cross-half add
}

// The gfx12 w32 layouts, from this tree's own WMMA code: an A or B fragment holds row `l & 15` with
// k elements `(l >> 4) * 8 .. +8`; accumulator element `e` of lane `l` is row `(l >> 4) * 8 + e`,
// column `l & 15`.
#define DA_ACC_ROW(lane, e) (((lane) >> 4) * 8 + (e))

// LDS per wave, and every one of the three pieces is a LAYOUT CONVERSION the arithmetic forces:
//   sS   the score tile out of the WMMA accumulator layout (8 rows x 1 tile-column per lane) into
//        The reference's reduction layout (1 row x 8 tile-columns per lane);
//   sA   the per-row alpha back the other way, because the accumulator it rescales is in the
//        accumulator layout;
//   sV   the value tile from [t][d] into [d][t], because the second dot needs d as its B-fragment
//        row and t as its k axis.
// `+ 4` / `+ 8` on the row strides is the usual bank skew.
#define DA_SSTR (32 + 4)
#define DA_VSTR (32 + 8)
// In FLOAT units: the score tile and the alpha row are fp32, the value tile is bf16 (hence / 2).
#define DA_LDS_PER_WAVE (DA_BM * DA_SSTR + DA_BM + (DA_HEAD * DA_VSTR) / 2)

template <int TILE, bool KV_FP8, bool NS>
__device__ __forceinline__ void dflash_attn_impl(
    const __bf16 *__restrict__ q,          // [rows, num_q_heads, HEAD]
    const void *__restrict__ kc,           // [blocks, block_size, num_kv_heads, HEAD]
    const void *__restrict__ vc,           //  "
    const int *__restrict__ block_table,   // [num_seqs, max_blocks]
    const int *__restrict__ seq_lens,      // [num_seqs]
    const float *__restrict__ k_scale,     // fp8 dequant, [1, kv_heads] but the reference reads
    const float *__restrict__ v_scale,     // element 0 only -- a SCALAR load per tile
    __bf16 *__restrict__ out,              // [rows, num_q_heads, HEAD]
    int rows, int num_q_heads, int num_kv_heads, int block_size, int bt_stride,
    int window, float scale, int q_per_seq, float *lds) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  // WAVE `w` OF GRID ROW `y` TAKES KV HEAD `y * DA_NW + w`, and the second grid dimension is what
  // makes this kernel serve a rank with more kv heads than a workgroup has waves.
  //
  // It was `kvh = wave` until 2026-09-21, which is correct at the drafter's TP=2 shape (4 kv
  // heads, 4 waves) and SILENTLY HALF-COMPUTES at TP=1, where the checkpoint's 8 kv heads meet
  // the same four waves: heads 4..7 were never visited and their slice of `out` kept whatever was
  // in it. `resolve` could not see it either -- its GQA check is `r * gqa == n`, which 8 x 4 = 32
  // satisfies. Eight waves in one workgroup is not the alternative: LDS is `DA_NW *
  // DA_LDS_PER_WAVE` floats = 50,432 bytes at four, and twice that does not fit.
  //
  // At `num_kv_heads <= DA_NW` the grid's y extent is 1 and `blockIdx.y` is 0, so this is the
  // same program the TP=2 gate byte-compared -- the change cannot move a number that gate took.
  const int kvh = (int)blockIdx.y * DA_NW + wave;
  if (kvh >= num_kv_heads) return;
  const int gblock = (int)blockIdx.x;
  // ---- THE SEQUENCE WALK, and why it is a division rather than a binary search ---------------
  //
  // The reference resolves `(seq_idx, q_block_local_idx)` with a binary search over
  // `cu_seqlens_q`, because the reference server's batch is RAGGED -- every request contributes its own number of
  // query rows. The drafter's is not: every request in a proposal contributes exactly the block's
  // rows, `q_per_seq`, and the block is a multiple of BLOCK_Q. So the search collapses to one
  // division, EXACTLY (not approximately): `seq = qblock * BLOCK_Q / q_per_seq` and the local
  // block index is `qblock - seq * (q_per_seq / BLOCK_Q)`. A ragged drafter would need the
  // prefix-sum walk back; a rectangular one that pretended to be ragged would be doing a search
  // whose answer it already knows.
  //
  // `NS` is a TEMPLATE parameter and not a runtime branch, so the one-sequence instantiations --
  // the three this tree has gated against prod's own live call -- compile to exactly the program
  // they compiled to before: `seq` 0, `start` 0, `seq_lens[0]`, the block table at offset 0, and
  // the program id IS the local q-block index. That is what makes the batched symbol an ADDITION
  // rather than an edit to a gated kernel.
  //
  // The grid's x extent is the reference's `q.shape[0] / BLOCK_Q + num_seqs`, and the trailing
  // `num_seqs` blocks land past `rows` and early-return here, before any table is read -- which
  // is why `seq_lens[seq]` below is safe at the last program.
  const int qps = NS ? q_per_seq : rows;
  const int seq = NS ? (gblock * DA_BQ) / qps : 0;
  const int start = seq * qps;
  if (start >= rows) return;
  const int qblock = NS ? gblock - seq * (qps / DA_BQ) : gblock;
  const int qlen = qps, seq_len = seq_lens[seq];
  block_table += (long)seq * bt_stride;
  if (qblock * DA_BQ >= qlen) return;
  const int context_len = seq_len - qlen;

  const int c = lane & 15, h = lane >> 4;
  float *sS = lds + wave * DA_LDS_PER_WAVE;
  float *sAl = sS + DA_BM * DA_SSTR;
  __bf16 *sV = (__bf16 *)(sAl + DA_BM);

  // ---- Q, once: 8 A-fragments, row `lane & 15` of the [BLOCK_M, HEAD] tile -------------------
  bf16x8 qf[DA_HEAD / 16];
  {
    const int m = c;
    const int qpos = qblock * DA_BQ + m / DA_NQPK;
    const int qhead = kvh * DA_NQPK + m % DA_NQPK;
    const bool live = qpos < qlen && qhead < num_q_heads;
    const __bf16 *qp = q + ((long)(start + (live ? qpos : 0)) * num_q_heads
                           + (live ? qhead : 0)) * DA_HEAD + h * 8;
#pragma unroll
    for (int ks = 0; ks < DA_HEAD / 16; ++ks) {
      bf16x8 t = *(const bf16x8 *)(qp + ks * 16);
      if (!live)
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = (__bf16)0.f;
      qf[ks] = t;
    }
  }

  floatx8 acc[DA_NF];
#pragma unroll
  for (int f = 0; f < DA_NF; ++f)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[f][e] = 0.f;
  // M and L are per QUERY ROW and live in the reduction layout, where a lane owns exactly one row
  // (`row = lane & 15`, replicated across the two halves). `init_softmax_M` with no sinks is -inf,
  // and L starts at ONE, not zero.
  float Mrow = -INFINITY, Lrow = 1.0f;

  // ---- the tile loop bounds, which the sliding window prunes ---------------------------------
  // Non-causal, so `max_seq_prefix_len` is the whole sequence and the window is symmetric:
  // `first = q_abs - WIN + 1`, `last = min(context + qpos_hi + WIN - 1, seq_len - 1)`.
  const int qpos_lo = qblock * DA_BQ;
  const int qpos_hi = min(qpos_lo + (DA_BM - 1) / DA_NQPK, qlen - 1);
  const int num_tiles = (seq_len + TILE - 1) / TILE;
  const int first_key = context_len + qpos_lo - window + 1;
  const int last_key = min(context_len + qpos_hi + window - 1, seq_len - 1);
  const int lo = max(0, first_key < 0 ? 0 : first_key / TILE);
  const int hi = min(last_key / TILE + 1, num_tiles);

  // `_cast_kv_tile` at KV_QUANT_MODE 1 does `tl.load(tensor_scale)` -- element 0 of each, even
  // though prod passes `[1, kv_heads]` tensors. Two SEPARATE buffers, which is how prod calls it;
  // the first draft read `kv_scale[0]` and `kv_scale[1]` out of one, and would have taken k's scale
  // for v against prod's real arguments.
  const float ks_v = KV_FP8 ? k_scale[0] : 1.f;
  const float vs_v = KV_FP8 ? v_scale[0] : 1.f;

  for (int j = lo; j < hi; ++j) {
    // ---- S = scale * (Q . K^T): B-fragments straight out of the cache. The flash layout is
    // slot-major with the head dimension contiguous, which IS the B-fragment layout, so no staging.
    floatx8 sacc[TILE / 16];
#pragma unroll
    for (int u = 0; u < TILE / 16; ++u) {
#pragma unroll
      for (int e = 0; e < 8; ++e) sacc[u][e] = 0.f;
      const int t_abs = j * TILE + u * 16 + c;
      const int tb = t_abs < seq_len ? t_abs : 0;
      const long blk = block_table[tb / block_size];
      const long slot = tb % block_size;
      const long base = ((blk * block_size + slot) * num_kv_heads + kvh) * DA_HEAD;
#pragma unroll
      for (int ks = 0; ks < DA_HEAD / 16; ++ks) {
        bf16x8 kf;
        if (KV_FP8) {
          // ONE eight-byte load, then decode: eight scalar byte loads is what took the fp8 arm to
          // 3x the reference on identical work.
          const uint2_t raw =
              *(const uint2_t *)((const unsigned char *)kc + base + ks * 16 + h * 8);
          // `_cast_kv_tile` at KV_QUANT_MODE 1 with a bf16 Q: fp8 -> fp32, times the tensor scale,
          // then ROUNDED TO BF16 before the dot.
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            kf[w * 4 + 0] = (__bf16)(da_fp8x<0>(raw[w]) * ks_v);
            kf[w * 4 + 1] = (__bf16)(da_fp8x<1>(raw[w]) * ks_v);
            kf[w * 4 + 2] = (__bf16)(da_fp8x<2>(raw[w]) * ks_v);
            kf[w * 4 + 3] = (__bf16)(da_fp8x<3>(raw[w]) * ks_v);
          }
        } else {
          kf = *(const bf16x8 *)((const __bf16 *)kc + base + ks * 16 + h * 8);
        }
        if (t_abs >= seq_len)
#pragma unroll
          for (int i = 0; i < 8; ++i) kf[i] = (__bf16)0.f;
        sacc[u] = da_wmma(qf[ks], kf, sacc[u]);
      }
    }

    // ---- into the reduction layout, masking on the way -----------------------------------------
    // `S += score_scale * tl.dot(Q, K)` scales the FINISHED dot product, and the mask replaces a
    // masked score with -inf rather than zero.
#pragma unroll
    for (int u = 0; u < TILE / 16; ++u)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = DA_ACC_ROW(lane, e);
        const int qpos = qblock * DA_BQ + m / DA_NQPK;
        const int qhead = kvh * DA_NQPK + m % DA_NQPK;
        const int qabs = context_len + qpos;
        const int t_abs = j * TILE + u * 16 + c;
        // Non-causal: `seq_offset < seq_len`, AND a window on BOTH sides.
        const bool keep = qpos < qlen && qhead < num_q_heads && t_abs < seq_len
                          && (qabs - t_abs) < window && (t_abs - qabs) < window;
        sS[m * DA_SSTR + u * 16 + c] = keep ? scale * sacc[u][e] : -INFINITY;
      }
    __builtin_amdgcn_s_waitcnt(0);            // the wave's own LDS writes, before it reads them back

    // ---- the running softmax, in the reference's layout ----------------------------------------
    // A lane holds HALF the tile -- `TILE / 2` columns of one query row -- and the paired lane
    // (`lane ^ 16`) holds the other half. That is what the single `permlanex16` in the reference's
    // reduction means, and declaring this vector `TILE` long instead read past the score tile in
    // LDS: every output element differed and some were NaN.
    constexpr int NV = TILE / 2;
    float sv[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) sv[i] = sS[c * DA_SSTR + (i / 8) * 16 + h * 8 + (i % 8)];
    const float rm = da_tile_max<NV>(sv);
    float m_j = fmaxf(Mrow, rm);
    // A whole row can be masked out under a window; the reference replaces the resulting -inf with
    // zero rather than letting `exp` produce a NaN.
    m_j = m_j > -INFINITY ? m_j : 0.f;
    float pv[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) pv[i] = da_exp(sv[i] - m_j);
    const float l_j = da_tile_sum<NV>(pv);
    const float alpha = da_exp(Mrow - m_j);
    Lrow = Lrow * alpha + l_j;
    Mrow = m_j;
    // alpha is per QUERY ROW and the accumulator is in the accumulator layout, so it has to cross
    // back. Lane `c` of the low half owns row `c`.
    if (h == 0) sAl[c] = alpha;
    __builtin_amdgcn_s_waitcnt(0);

    // ---- V into [d][t], masked per KEY ---------------------------------------------------------
    // The reference computes `dist` from `qpos_lo` alone -- one decision per key, applied to whole
    // V rows -- not per query row.
#pragma unroll
    for (int u = 0; u < TILE / 16; ++u) {
      const int t = u * 16 + c;
      const int t_abs = j * TILE + t;
      const int dist = context_len + qpos_lo - t_abs;
      const bool keep = t_abs < seq_len && dist < window && dist > -window;
      const int tb = t_abs < seq_len ? t_abs : 0;
      const long blk = block_table[tb / block_size];
      const long slot = tb % block_size;
      const long base = ((blk * block_size + slot) * num_kv_heads + kvh) * DA_HEAD;
#pragma unroll
      for (int d0 = 0; d0 < DA_HEAD; d0 += 32) {
        const int d = d0 + h * 16;
        if (KV_FP8) {
          const uint4_t raw = *(const uint4_t *)((const unsigned char *)vc + base + d);
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            const float f0 = da_fp8x<0>(raw[w]) * vs_v, f1 = da_fp8x<1>(raw[w]) * vs_v;
            const float f2 = da_fp8x<2>(raw[w]) * vs_v, f3 = da_fp8x<3>(raw[w]) * vs_v;
            sV[(d + w * 4 + 0) * DA_VSTR + t] = keep ? (__bf16)f0 : (__bf16)0.f;
            sV[(d + w * 4 + 1) * DA_VSTR + t] = keep ? (__bf16)f1 : (__bf16)0.f;
            sV[(d + w * 4 + 2) * DA_VSTR + t] = keep ? (__bf16)f2 : (__bf16)0.f;
            sV[(d + w * 4 + 3) * DA_VSTR + t] = keep ? (__bf16)f3 : (__bf16)0.f;
          }
        } else {
          const bf16x8 lo = *(const bf16x8 *)((const __bf16 *)vc + base + d);
          const bf16x8 hi = *(const bf16x8 *)((const __bf16 *)vc + base + d + 8);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            sV[(d + i) * DA_VSTR + t] = keep ? lo[i] : (__bf16)0.f;
            sV[(d + 8 + i) * DA_VSTR + t] = keep ? hi[i] : (__bf16)0.f;
          }
        }
      }
    }
    __builtin_amdgcn_s_waitcnt(0);

    // ---- acc = acc * alpha, then acc += P . V ---------------------------------------------------
    // P is ALREADY an A fragment: a lane holds one query row's eight tile columns, which is exactly
    // `row = lane & 15, k = (lane >> 4) * 8 .. +8`. That is the layout the softmax just worked in,
    // so unlike the first draft nothing transposes P.
#pragma unroll
    for (int f = 0; f < DA_NF; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[f][e] *= sAl[DA_ACC_ROW(lane, e)];
#pragma unroll
    for (int u = 0; u < TILE / 16; ++u) {
      bf16x8 af;
#pragma unroll
      for (int i = 0; i < 8; ++i) af[i] = (__bf16)pv[u * 8 + i];
#pragma unroll
      for (int f = 0; f < DA_NF; ++f) {
        const bf16x8 bf = *(const bf16x8 *)(sV + (f * 16 + c) * DA_VSTR + u * 16 + h * 8);
        acc[f] = da_wmma(af, bf, acc[f]);
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
  }

  // ---- epilogue: one division by L, then a bf16 store ----------------------------------------
  // L is per query row in the reduction layout; the accumulator needs it per accumulator row, so it
  // crosses back the same way alpha did.
  if (h == 0) sAl[c] = Lrow;
  __builtin_amdgcn_s_waitcnt(0);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int m = DA_ACC_ROW(lane, e);
    const int qpos = qblock * DA_BQ + m / DA_NQPK;
    const int qhead = kvh * DA_NQPK + m % DA_NQPK;
    if (qpos >= qlen || qhead >= num_q_heads) continue;
    const float linv = sAl[m];
    __bf16 *op = out + ((long)(start + qpos) * num_q_heads + qhead) * DA_HEAD + c;
#pragma unroll
    for (int f = 0; f < DA_NF; ++f) op[f * 16] = (__bf16)(acc[f][e] / linv);
  }
}

// THE BF16-CACHE ENTRY POINT OF RECORD. `t32`, because the 2D path's tile is `TILE_SIZE_PREFILL`
// and that is 32 whatever the KV dtype is -- the element-size branch in `_get_tile_size` is on the
// `is_prefill=False` arm, which only the 3D path reads and the drafter never takes. It shares the
// LDS budget with the fp8 instantiation exactly (`DA_SSTR`/`DA_VSTR` are already sized for a
// 32-wide tile, which is why `t16` over-allocated rather than the reverse): both report
// `group_segment_fixed_size = 50432`.
extern "C" __global__ __launch_bounds__(DA_NT) void dflash_attn_h128_gqa4_bf16kv_t32(
    const __bf16 *__restrict__ q, const __bf16 *__restrict__ kc, const __bf16 *__restrict__ vc,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    const float *__restrict__ k_scale, const float *__restrict__ v_scale,
    __bf16 *__restrict__ out,
    int rows, int num_q_heads, int num_kv_heads, int block_size, int bt_stride, int window,
    float scale) {
  __shared__ float lds[DA_NW * DA_LDS_PER_WAVE];
  dflash_attn_impl<32, false, false>(q, kc, vc, block_table, seq_lens, k_scale, v_scale, out,
                                     rows, num_q_heads, num_kv_heads, block_size, bt_stride,
                                     window, scale, 0, lds);
}

// THE TILE CONTROL, and not a route. It is the same kernel at the 3D path's tile, kept compiled so
// a gate can put the wrong tile back and watch it fire (F24 measured 14,306 bytes of 983,040 on
// the fp8 side of exactly this substitution). `DflashRules::validate` refuses to BIND it: an
// attention row whose symbol does not end in `_t32` is a rules error by name.
extern "C" __global__ __launch_bounds__(DA_NT) void dflash_attn_h128_gqa4_bf16kv_t16(
    const __bf16 *__restrict__ q, const __bf16 *__restrict__ kc, const __bf16 *__restrict__ vc,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    const float *__restrict__ k_scale, const float *__restrict__ v_scale,
    __bf16 *__restrict__ out,
    int rows, int num_q_heads, int num_kv_heads, int block_size, int bt_stride, int window,
    float scale) {
  __shared__ float lds[DA_NW * DA_LDS_PER_WAVE];
  dflash_attn_impl<16, false, false>(q, kc, vc, block_table, seq_lens, k_scale, v_scale, out,
                                     rows, num_q_heads, num_kv_heads, block_size, bt_stride,
                                     window, scale, 0, lds);
}

extern "C" __global__ __launch_bounds__(DA_NT) void dflash_attn_h128_gqa4_fp8kv_t32(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const int *__restrict__ block_table,
    const int *__restrict__ seq_lens, const float *__restrict__ k_scale,
    const float *__restrict__ v_scale, __bf16 *__restrict__ out, int rows, int num_q_heads, int num_kv_heads, int block_size,
    int bt_stride, int window, float scale) {
  __shared__ float lds[DA_NW * DA_LDS_PER_WAVE];
  dflash_attn_impl<32, true, false>(q, kc, vc, block_table, seq_lens, k_scale, v_scale, out, rows,
                                    num_q_heads, num_kv_heads, block_size, bt_stride, window,
                                    scale, 0, lds);
}

// ---- THE BATCHED ENTRY POINTS: `num_seqs` SEQUENCES IN ONE LAUNCH (ws-s11) ---------------------
//
// One extra kernarg -- `q_per_seq`, the drafter's block -- and one extra template instantiation.
// `num_seqs` itself is NOT a kernarg: it is `rows / q_per_seq` and the grid carries it (the x
// extent is `rows / BLOCK_Q + num_seqs`), so two numbers that must agree are one number. A frame
// that disagreed with the grid would compute a sequence nobody launched.
//
// WHY THIS SYMBOL EXISTS. Until this round the engine expressed a batched proposal as `reqs`
// LAUNCHES of the one-sequence symbol at static per-request offsets -- bit-identical by
// construction, and `5 * (reqs - 1)` extra nodes in the captured backbone (ws-s10). The launches
// are the same work; the NODES are not free, and the batch was the one site of nineteen that did
// not enter the grid. It is one site now, and the one-sequence symbols above are untouched --
// which is what makes their gate against prod's own live call still THEIR gate.
//
// It is a rules row of its own (`attn_fp8kv_ns`), so nothing picks this symbol by accident: the
// one-request shape still binds the one-sequence kernel and the batched shapes bind this one,
// `DflashOp::attn_for` is the only chooser, and `resolve` refuses the one-sequence symbol a shape
// with more than one sequence BY NAME rather than serving request 0's attention `reqs` times.
extern "C" __global__ __launch_bounds__(DA_NT) void dflash_attn_h128_gqa4_bf16kv_t32_ns(
    const __bf16 *__restrict__ q, const __bf16 *__restrict__ kc, const __bf16 *__restrict__ vc,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    const float *__restrict__ k_scale, const float *__restrict__ v_scale,
    __bf16 *__restrict__ out,
    int rows, int num_q_heads, int num_kv_heads, int block_size, int bt_stride, int window,
    float scale, int q_per_seq) {
  __shared__ float lds[DA_NW * DA_LDS_PER_WAVE];
  dflash_attn_impl<32, false, true>(q, kc, vc, block_table, seq_lens, k_scale, v_scale, out, rows,
                                    num_q_heads, num_kv_heads, block_size, bt_stride, window,
                                    scale, q_per_seq, lds);
}

extern "C" __global__ __launch_bounds__(DA_NT) void dflash_attn_h128_gqa4_fp8kv_t32_ns(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const int *__restrict__ block_table,
    const int *__restrict__ seq_lens, const float *__restrict__ k_scale,
    const float *__restrict__ v_scale, __bf16 *__restrict__ out, int rows, int num_q_heads,
    int num_kv_heads, int block_size, int bt_stride, int window, float scale, int q_per_seq) {
  __shared__ float lds[DA_NW * DA_LDS_PER_WAVE];
  dflash_attn_impl<32, true, true>(q, kc, vc, block_table, seq_lens, k_scale, v_scale, out, rows,
                                   num_q_heads, num_kv_heads, block_size, bt_stride, window, scale,
                                   q_per_seq, lds);
}
