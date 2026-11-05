#!/usr/bin/env python3
"""FN148: derive `crates/saber-kernels/hip/moe_fn148.hip` -- the NVFP4 (e2m1 + E4M3 g16 scale + f32 global)
routed-expert GEMM bodies that read the FP4-Local xtier v2 record with NO requant.

  fn148_a16_nvfp4_ref_g{0,1}   THE ORACLE: bf16 A times the exact fp32 dequant w = (e2m1(code) * E4M3(s)) * gs,
                               K in order, `acc = acc + a * w` (fp-contract off), one bf16 rounding. FN4's A16
                               control (`moe_route.hip`) with the E4M3 g16 scale + the global; a numpy f32 program
                               reproduces it bit for bit (tools/capture/fn148/fn148_ref.py).
  fn148_w4a8b_*                FORM (b), the exact-weight arm: the shipped kMag W4A8 decode body
                               (`moe_grouped_tp2.hip`'s `moe_grouped_w4a8_impl`, the served WPERM + NT staging)
                               with the weight bytes = the plain e2m1 codes (kMag[0]) and the whole E4M3 scale
                               FMA'd per 16-deep k-group into the accumulator:
                               tile = wmma(a8, e2m1, 0); acc = fma(tile, E4M3(s[g, n]), acc)  (8 VALU / WMMA);
                               epilogue acc * gs * As. The scale's exponent rides the f32 FMA operand, so no
                               2^e fold into kMag is needed (the FMA operand has 4 significant bits either way).
  fn148_w4a8c_*                FORM (c), the production candidate (lossy, W4A8 bytes): the same body with the
                               kMag table built PER g16 GROUP from the E4M3 scale -- entry i = e4m3_rne(e2m1_i *
                               E4M3(s) * 2^r) -- r = the split's per-row power-of-two shift (6 * max_s * 2^r <= 448,
                               maximal; one prologue pass over the split's scale bytes); epilogue
                               ((acc * 2^-r) * gs) * As (the global out of the table: FN143's model with gs moved
                               to the epilogue).
Every other line -- A staging, the WPERM fragment walk, the WMMA chain, the split-K partials, the fused
reduce -- is the shipped body's text (anchored substitutions, each asserted to hit exactly once).
  python3 tools/gen_fn148_nvfp4.py          # write
  python3 tools/gen_fn148_nvfp4.py --check  # the committed file == the derivation
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "crates/saber-kernels/hip/moe_grouped_tp2.hip")
OUT = os.path.join(REPO, "crates/saber-kernels/hip/moe_fn148.hip")

CELLS = (  # (dbk, ks, tm, gather)
    [(128, ks, tm, 1) for ks in (1, 2, 4) for tm in (1, 2, 4)]
    + [(128, ks, tm, 0) for ks in (1, 2) for tm in (1, 2, 4)]
    + [(64, ks, 1, 0) for ks in (1, 2)]
)


def sub(text, old, new, count=1):
    n = text.count(old)
    assert n == count, (n, count, old[:80])
    return text.replace(old, new)


def e4m3_f32_table():
    import struct
    vals = []
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            u = 0x7FC00000 | (0x80000000 if b & 0x80 else 0)
        else:
            v = s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
            u = struct.unpack("<I", struct.pack("<f", v))[0]
        vals.append(u)
    rows = []
    for i in range(0, 256, 8):
        rows.append("  " + ", ".join(f"0x{u:08X}u" for u in vals[i:i + 8]) + ",")
    return "\n".join(rows)


def derive():
    src = open(SRC).read()
    lo = src.index("static __device__ __forceinline__ unsigned int wq_dec")
    hi = src.index("// ---------------------------------------------------------------- the producer's preamble")
    body = src[lo:hi].rstrip() + "\n"
    # --- the impl's name, its FORM parameter and the gs split argument
    body = sub(body, "template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool GATHER, bool DIRECT = true,\n"
                     "          bool NT = false>\n__device__ __forceinline__ void moe_grouped_w4a8_impl(",
               "template <int FORM, int DWN, int DBK, int DKS, int DTM, bool WPERM, bool GATHER, bool DIRECT = true,\n"
               "          bool NT = false>\n__device__ __forceinline__ void fn148_nvfp4_impl(")
    body = sub(body, "    __bf16 *__restrict__ C, int N, int K, int Pmax) {\n  constexpr int BND",
               "    __bf16 *__restrict__ C, int N, int K, int Pmax, int gsplit) {\n"
               "  static_assert(WPERM, \"FN148: the v2 record's w is in permute_w order\");\n"
               "  static_assert(FORM == 1 || FORM == 2, \"FN148: FORM 1 = (b), 2 = (c)\");\n  constexpr int BND")
    # --- the record: w, ws (E4M3 [K/16, N]), gs (f32 x 2 gate_up / x 1 down) replace w, ws (e8m0), wref
    body = sub(body, "  const unsigned char *__restrict__ Wref =\n"
                     "      (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 2];\n",
               "  const float *__restrict__ Gs = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];\n")
    # --- FORM 2's prologue: the split's per-row shift from the max scale byte over its k-groups
    body = sub(body, "  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);\n",
               "  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);\n"
               "  __shared__ int s_rsh[BND];   // FORM 2: the split's per-row shift r\n"
               "  if constexpr (FORM == 2) {\n"
               "    static_assert(DNTHREADS % (BND / 4) == 0, \"FN148: the shift pass's thread map\");\n"
               "    const int g_lo = s_lo * (DBK / 16), g_hi = min(K / 16, s_hi * (DBK / 16));\n"
               "    const int c4 = (tid % (BND / 4)) * 4, gq = tid / (BND / 4);   // 4 rows a thread, 8 groups a pass\n"
               "    const int nr4 = n0 + c4 < N - 4 ? n0 + c4 : N - 4;              // N % 4 == 0 (u32 aligned)\n"
               "    unsigned int m0 = 0, m1 = 0, m2 = 0, m3 = 0;\n"
               "#pragma unroll 4\n"
               "    for (int g = g_lo + gq; g < g_hi; g += DNTHREADS / (BND / 4)) {\n"
               "      const unsigned int w = *(const unsigned int *)&Ws[(size_t)g * N + nr4];\n"
               "      m0 = max(m0, w & 0x7Fu); m1 = max(m1, (w >> 8) & 0x7Fu);\n"
               "      m2 = max(m2, (w >> 16) & 0x7Fu); m3 = max(m3, (w >> 24) & 0x7Fu);\n"
               "    }\n"
               "    // the 8-way max goes through sW (free until the first W stage, behind the barriers below)\n"
               "    static_assert((DNTHREADS / (BND / 4)) * BND <= BND * DWSTR, \"FN148: the max scratch fits sW\");\n"
               "    unsigned char *s_mx = sW;\n"
               "    s_mx[gq * BND + c4] = (unsigned char)m0; s_mx[gq * BND + c4 + 1] = (unsigned char)m1;\n"
               "    s_mx[gq * BND + c4 + 2] = (unsigned char)m2; s_mx[gq * BND + c4 + 3] = (unsigned char)m3;\n"
               "    __syncthreads();\n"
               "    int r = 0;\n"
               "    if (tid < BND) {\n"
               "      unsigned int mb = 0;\n"
               "#pragma unroll\n"
               "      for (int j = 0; j < DNTHREADS / (BND / 4); ++j) mb = max(mb, (unsigned int)s_mx[j * BND + tid]);\n"
               "      r = fn148_row_shift((int)mb);\n"
               "    }\n"
               "    __syncthreads();\n"
               "    if (tid < BND) s_rsh[tid] = r;\n"
               "    __syncthreads();\n"
               "  }\n")
    # --- the row-order staging and its branch are dead under WPERM (static_assert): drop them
    a = body.index("  // Row-order (checkpoint layout) W staging from a packed 8-byte word per (iteration, thread).")
    b = body.index("  auto stage_w_wperm = [&](int k0) {")
    body = body[:a] + body[b:]
    a = body.index("  } else {\n    for (int s = s_lo; s < s_hi; ++s) {\n      const int k0 = s * DBK;\n      stage_a(k0);\n      stage_w_row(")
    b = body.index("  const int n = n0 + wave * 16 + col;")
    body = body[:a] + "  }\n\n" + body[b:]
    # --- WPERM staging: the table per (row, 16-k group) instead of kMag[wref - e8m0]
    body = sub(body, "        const int blk = (k0 + kst * 16) / 32;\n",
               "        const int grp = kstep0 + kst;   // the NVFP4 16-k group\n")
    body = sub(body, "          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];\n"
                     "          d = d < 0 ? 0 : (d > 15 ? 15 : d);\n"
                     "          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];\n",
               "          unsigned int t0, t1;\n"
               "          fn148_table<FORM>(Ws[(size_t)grp * N + gc + q], s_rsh[fn148_ri(gc + q - n0, BND)], t0, t1);\n")
    # --- the WMMA step: FORM 1 = a fresh tile per 16-k group, FMA'd by the group's scale
    body = sub(body, "  auto mma_slab = [&]() {\n    __syncthreads();\n",
               "  const int ncol = n0 + wave * 16 + col < N - 1 ? n0 + wave * 16 + col : N - 1;\n"
               "  auto mma_slab = [&](int k0) {\n"
               "    float scv[DBK / 16];\n"
               "    if constexpr (FORM == 1) {\n"
               "#pragma unroll\n"
               "      for (int step = 0; step < DBK / 16; ++step)\n"
               "        scv[step] = fn148_e4m3_hw(Ws[(size_t)(k0 / 16 + step) * N + ncol]);\n"
               "    }\n"
               "    __syncthreads();\n")
    body = sub(body, "#pragma unroll\n      for (int i = 0; i < DTM; ++i)\n"
                     "        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);\n",
               "      if constexpr (FORM == 1) {\n"
               "        const floatx8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n"
               "#pragma unroll\n"
               "        for (int i = 0; i < DTM; ++i) {\n"
               "          const floatx8 t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, z);\n"
               "#pragma unroll\n"
               "          for (int e = 0; e < 8; ++e) acc[i][e] = __builtin_fmaf(t[e], scv[step], acc[i][e]);\n"
               "        }\n"
               "      } else {\n"
               "#pragma unroll\n      for (int i = 0; i < DTM; ++i)\n"
               "        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);\n"
               "      }\n")
    body = sub(body, "      stage_w_wperm(k0);\n      mma_slab();\n", "      stage_w_wperm(k0);\n      mma_slab(k0);\n")
    # --- epilogues: the global (gate rows | up rows) replaces 2^(wref-127); FORM 2 undoes its shift first
    body = sub(body, "      const float rf = __int_as_float((int)Wref[n] << 23);\n",
               "      const float rf = Gs[n < gsplit ? 0 : 1];\n")
    body = sub(body, "C[(size_t)(row0 + m) * N + n] = (__bf16)(acc[i][e] * rf * As[s_arow[m]]);",
               "C[(size_t)(row0 + m) * N + n] = (__bf16)(fn148_pre<FORM>(acc[i][e], s_rsh, n - n0, BND) * rf * As[s_arow[m]]);")
    body = sub(body, "        if (m < M) P[((size_t)ks * Pmax + row0 + m) * N + n] = acc[i][e];",
               "        if (m < M) P[((size_t)ks * Pmax + row0 + m) * N + n] = fn148_pre<FORM>(acc[i][e], s_rsh, n - n0, BND);")
    body = sub(body, "    const float rf = __int_as_float((int)Wref[nn] << 23);\n",
               "    const float rf = Gs[nn < gsplit ? 0 : 1];\n")
    assert "Wref" not in body and "kMag[d]" not in body, "a Wref / kMag[d] use survived"

    head = f'''// GENERATED by tools/gen_fn148_nvfp4.py -- do not edit by hand, re-run the deriver.
//
// FN148: the NVFP4 (FP4-Local) routed-expert GEMM bodies over the xtier v2 record (`w` permute_w e2m1,
// `ws` E4M3 g16 [K/16, N], `gs` f32 x 2 gate_up (gate rows < gsplit, up rows after) / x 1 down) -- no requant.
//   fn148_a16_nvfp4_ref_g{{0,1}}  the oracle (exact fp32 dequant, bf16 A, K in order, fp-contract off)
//   fn148_w4a8b_*                FORM (b): plain e2m1 bytes + the E4M3 scale FMA'd per 16-k group (exact weights)
//   fn148_w4a8c_*                FORM (c): kMag table per g16 group = e4m3(e2m1 * s * 2^r), r per row and split
// The W4A8 bodies are `moe_grouped_tp2.hip`'s `moe_grouped_w4a8_impl` (the served WPERM + NT decode body)
// by anchored substitution; see the generator's docstring for each change.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

#define DEC_DWN 8   // waves per decode block; the N tile is DEC_DWN*16 = 128 wide
#define DEC_MTILE 16
#define DEC_PAD 8

// E4M3 (OCP fn) byte -> f32, exact (generated; 0x7F / 0xFF = NaN).
__device__ __constant__ unsigned int kFn148E4m3[256] = {{
{e4m3_f32_table()}
}};
static __device__ __forceinline__ float fn148_e4m3_f32(unsigned char b) {{
  return __uint_as_float(kFn148E4m3[b]);
}}
// The forms' convert: v_cvt_f32_fp8 (exact for every OCP e4m3 byte; the byte law vs the table-built twin checks it).
static __device__ __forceinline__ float fn148_e4m3_hw(unsigned char b) {{
  return __builtin_amdgcn_cvt_f32_fp8((int)b, 0);
}}
// The per-row shift r of FORM (c): the largest r with 6 * E4M3(mb) * 2^r <= 448 (mb = the max scale byte,
// sign masked); 0 for an all-zero row. Exact: the f32 exponent then one compare of an exact power-of-two scaling.
static __device__ __forceinline__ int fn148_row_shift(int mb) {{
  const float v = 6.f * fn148_e4m3_f32((unsigned char)mb);
  if (!(v > 0.f)) return 0;
  int r = 8 - (int)((__float_as_uint(v) >> 23) & 0xFF) + 127;   // v is a normal f32 (>= 6 * 2^-9)
  if (__builtin_ldexpf(v, r) > 448.f) r -= 1;
  return r;
}}
static __device__ __forceinline__ int fn148_ri(int r, int bnd) {{ return r < 0 ? 0 : (r >= bnd ? bnd - 1 : r); }}
// The 8 magnitude bytes of a (row, 16-k group): FORM 1 = the plain e2m1 codes (kMag[0]); FORM 2 =
// e4m3_rne(m_i * E4M3(s) * 2^r) by the hardware convert (exact products: m_i * s has <= 5 significant bits).
template <int FORM>
static __device__ __forceinline__ void fn148_table(unsigned char sb, int r, unsigned int &t0, unsigned int &t1) {{
  if constexpr (FORM == 1) {{
    t0 = kMag[0][0];
    t1 = kMag[0][1];
  }} else {{
    const float s = __builtin_ldexpf(fn148_e4m3_hw(sb), r);
    int w0 = __builtin_amdgcn_cvt_pk_fp8_f32(0.f, 0.5f * s, 0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(s, 1.5f * s, w0, true);
    int w1 = __builtin_amdgcn_cvt_pk_fp8_f32(2.f * s, 3.f * s, 0, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(4.f * s, 6.f * s, w1, true);
    t0 = (unsigned int)w0;
    t1 = (unsigned int)w1;
  }}
}}
// FORM 2 undoes its row shift (an exact power of two) before the global / the partial.
template <int FORM>
static __device__ __forceinline__ float fn148_pre(float v, const int *s_rsh, int r, int bnd) {{
  if constexpr (FORM == 2) return __builtin_ldexpf(v, -s_rsh[fn148_ri(r, bnd)]);
  else return v;
}}

'''
    oracle = '''
// ---------------------------------------------------------------- THE ORACLE (exact fp32 dequant, W4A16)
// C[row0 + m, n] = bf16( sum_k A[arow(m), k] * ((e2m1(code(n, k)) * E4M3(ws[k/16, n])) * gs[n >= gsplit]) ),
// K in order, every product and sum its own IEEE op (the object builds with -ffp-contract=off).
__device__ __constant__ float kFn148E2M1[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                                -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
template <bool GATHER>
__device__ __forceinline__ void fn148_a16_impl(
    const __bf16 *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,
    const int *__restrict__ meta, __bf16 *__restrict__ C, int N, int K, int gsplit) {
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  const int n = blockIdx.x * 16 + (threadIdx.x & 15);
  const int m = 16 * blockIdx.z + (threadIdx.x >> 4);
  if (m >= M) return;
  const unsigned char *W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const unsigned char *Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  const float *Gs = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];
  const float gs = Gs[n < gsplit ? 0 : 1];
  const int arow = GATHER ? row_tok[row0 + m] : row0 + m;
  const __bf16 *a = A + (size_t)arow * K;
  const int KS = K / 16, nt = n >> 4, nr = n & 15;
  float acc = 0.f, sc = 0.f;
  for (int k = 0; k < K; ++k) {
    if ((k & 15) == 0) sc = fn148_e4m3_f32(Ws[(size_t)(k >> 4) * N + n]);
    const size_t off = ((((size_t)nt * KS + (k >> 4)) * 2 + ((k >> 3) & 1)) * 16 + nr) * 4 + ((k & 7) >> 1);
    const int code = (W[off] >> ((k & 1) * 4)) & 0xF;
    const float w = (kFn148E2M1[code] * sc) * gs;
    acc = acc + (float)a[k] * w;
  }
  C[(size_t)(row0 + m) * N + n] = (__bf16)acc;
}
#define FN148_A16(name, G)                                                                          \\
  extern "C" __global__ __launch_bounds__(256) void name(                                         \\
      const __bf16 *__restrict__ A, const unsigned long long *__restrict__ xptr,                  \\
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                     \\
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                         \\
      const int *__restrict__ meta, __bf16 *__restrict__ C, int N, int K, int gsplit) {           \\
    fn148_a16_impl<G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, C, N, K, gsplit); \\
  }
FN148_A16(fn148_a16_nvfp4_ref_g1, true)
FN148_A16(fn148_a16_nvfp4_ref_g0, false)
#undef FN148_A16
'''
    wr = ["\n// ---------------------------------------------------------------- the W4A8 cells"]
    for form, tag in ((1, "b"), (2, "c")):
        for dbk, ks, tm, g in CELLS:
            nm = f"fn148_w4a8{tag}_bk{dbk}_ks{ks}_tm{tm}_g{g}"
            wr.append(f'''extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void {nm}(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,
    const int *__restrict__ meta, const float *__restrict__ As, float *__restrict__ P,
    int *__restrict__ cnt, __bf16 *__restrict__ C, int N, int K, int Pmax, int gsplit) {{
  fn148_nvfp4_impl<{form}, DEC_DWN, {dbk}, {ks}, {tm}, true, {"true" if g else "false"}, true, true>(
      A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, As, P, cnt, C, N, K, Pmax, gsplit);
}}''')
    return head + body + oracle + "\n".join(wr) + "\n"


def main():
    text = derive()
    if "--check" in sys.argv:
        ok = os.path.exists(OUT) and open(OUT).read() == text
        print(f"gen_fn148_nvfp4: {OUT} {'== the derivation' if ok else 'DIFFERS from the derivation'}")
        return 0 if ok else 1
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} B, {2 * len(CELLS) + 2} kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
