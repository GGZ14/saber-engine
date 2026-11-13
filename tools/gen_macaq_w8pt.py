#!/usr/bin/env python3
"""MACaQ: generate `crates/saber-kernels/hip/macaq_w8.hip` -- the PER-TOKEN-scale twins of the shipped fp8
W8A8 bodies, for the 27B form-c W8 promotions (`--macaq-w8`: out_proj / o_proj / down_proj on the
MACaQ-Hadamard pack).

The 27B producers (`mxfp4_quant` / `mxfp4_had` / `attn_glue[_had]`) write the W4 path's activation: row-major
e4m3 `A [M, K]` + ONE f32 scale a row (`scale[row]`, the per-token law). The shipped W8 bodies read a scale a
128-wide k tile (`As [M, K/128]`). The twins below are those bodies verbatim (anchored substitution of the
shipped text) with the ONE change: the k tile's activation scale is the row's (`As[row]` for
`As[row * (K / 128) + kt]`). Every other operation -- the 8-step WMMA chain from zero per 128-deep k tile, the
weight's own 128 x 128 block scale, `acc += tile * (as * bs)` in kt order, one bf16 rounding at the store -- is
the parent's, so a twin == its parent called with `As[r][kt] = As_tok[r]` for every kt, byte for byte (the
oracle's law (a)).

  macaq_w8pt_n64_ldc      `fn111_t8_a8w8_n64_ldc`   (flashnext_t8141 `t8_gemm_a8w8_impl` at BN 64), M <= 16
  macaq_w8pt_n64_pf4_ldc  `fn141_t8_a8w8_n64_pf4_ldc` (the register ring), M <= 16; the ring's
                          [row][kt] scale table becomes [row] (the per-token scale needs no kt axis) and its
                          k-tile cap KTMAX 32 -> 160 (the 27B down at TP=1 is K 17,408 = 136 tiles)
  macaq_w8pt_d2w4_ldc     `fn_t8p_a8w8_d2w4_ldc`    (flashnext_trunk8p `t8p_impl`), 17..32 rows
  macaq_w8pt_d4w4_ldc     `fn_t8p_a8w8_d4w4_ldc`, 33..64 rows
  macaq_w8pt_p4w8_ldc     `fn_t8p_a8w8_p4w8_ldc`, the prefill tile (rows fastest)

Signature = the parents' (`abi::SIG_FN_T8_A8W8_LDC`: a, w, as, ws, c, m, n, k, ldc). Built with the parents'
flags (default contraction, saber-dev:latest -- the image that reproduces flashnext_t8141 / trunk8p).

    python3 tools/gen_macaq_w8pt.py          # write
    python3 tools/gen_macaq_w8pt.py --check  # exit 1 if the committed file differs
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIP = ROOT / "crates" / "saber-kernels" / "hip"
T141 = HIP / "flashnext_t8141.hip"
T8P = HIP / "flashnext_trunk8p.hip"
OUT = HIP / "macaq_w8.hip"


def sub1(text, old, new):
    n = text.count(old)
    assert n == 1, f"anchor occurs {n} times: {old[:90]!r}"
    return text.replace(old, new)


def slice_between(text, head, tail):
    i = text.index(head)
    j = text.index(tail, i + len(head))
    return text[i:j]


def build():
    t141 = T141.read_text()
    t8p = T8P.read_text()
    # ---- A: the helpers + t8_gemm_a8w8_impl (= fn111_t8_a8w8_n64_ldc's body at BN 64)
    a = slice_between(t141, "#include <hip/hip_runtime.h>",
                      "// ============================================================ bf16, one split")
    a = sub1(a, "__device__ __forceinline__ void t8_gemm_a8w8_impl(",
             "__device__ __forceinline__ void w8pt_gemm_impl(")
    a = sub1(a, "const float *__restrict__ As,          // [M, K/128] fp32 block scales",
             "const float *__restrict__ As,          // [M] fp32 per-token scales")
    a = sub1(a, "const float as = row < M ? As[(long)row * (K / 128) + kt] : 0.f;",
             "const float as = row < M ? As[row] : 0.f;")
    # ---- B: the ring (= fn141_t8_a8w8_n64_pf4_ldc's body), its scale table per row
    b = slice_between(t141, "__device__ __forceinline__ void fn141_lds_barrier()",
                      "// ---- the merged projection")
    b = sub1(b, "void fn141_t8_ring_impl(", "void w8pt_ring_impl(")
    b = sub1(b, """  constexpr int KTMAX = 32;
  __shared__ float s_as[BM * KTMAX];  // [row][kt] at stride KT (BM * KT <= BM * KTMAX = SV * NT)
  __shared__ float s_ws[KTMAX];""", """  constexpr int KTMAX = 160;          // K <= 20,480 (the route checks K / BK <= KTMAX)
  __shared__ float s_as[BM];          // the per-token scale, one a row
  __shared__ float s_ws[KTMAX];""")
    b = sub1(b, """  constexpr int SV = BM * KTMAX / NT;
  float sv[SV];
#pragma unroll
  for (int j = 0; j < SV; ++j) {
    const int i = tid + j * NT, r = min(i / KT, BM - 1), k = i - (i / KT) * KT;
    sv[j] = As[(long)min(m0 + r, M - 1) * KT + k];
  }
  const float wv = Ws[(long)(n0 / 128) * KT + min(tid, KT - 1)];""", """  const float sv = As[min(m0 + min(tid, BM - 1), M - 1)];
  constexpr int WV = (KTMAX + NT - 1) / NT;
  float wv[WV];
#pragma unroll
  for (int j = 0; j < WV; ++j) wv[j] = Ws[(long)(n0 / 128) * KT + min(tid + j * NT, KT - 1)];""")
    b = sub1(b, """#pragma unroll
  for (int j = 0; j < SV; ++j) {
    const int i = tid + j * NT, r = i / KT;
    s_as[i] = m0 + r < M ? sv[j] : 0.f;  // every slot written (slots past BM * KT unread)
  }
  s_ws[min(tid, KT - 1)] = wv;  // threads past KT store the last value again (same bytes)""",
             """  if (tid < BM) s_as[tid] = m0 + tid < M ? sv : 0.f;
#pragma unroll
  for (int j = 0; j < WV; ++j) s_ws[min(tid + j * NT, KT - 1)] = wv[j];  // past KT: the last value again""")
    b = sub1(b, "const float as = s_as[dg_acc_row(lane, e) * KT + kt];",
             "const float as = s_as[dg_acc_row(lane, e)];")
    # ---- C: t8p_impl + its helpers (= fn_t8p_a8w8_{d2w4,d4w4,p4w8}_ldc's body)
    c = slice_between(t8p, "#define P_BK 128", "#define T8P_SIG")
    c = sub1(c, "__device__ __forceinline__ void t8p_impl(", "__device__ __forceinline__ void w8pt_p_impl(")
    c = sub1(c, "const float *__restrict__ As,          // [M, K/128] fp32",
             "const float *__restrict__ As,          // [M] fp32 per-token")
    c = sub1(c, "sbuf = As[(long)srow * kb + kt];", "sbuf = As[srow];")
    head = """// GENERATED by tools/gen_macaq_w8pt.py from flashnext_t8141.hip + flashnext_trunk8p.hip -- do not edit;
// re-run the script.
//
// MACaQ (`--macaq-w8`): the per-token-scale twins of the shipped fp8 W8A8 bodies for the 27B form-c W8
// promotions. Each twin is its parent verbatim except the activation scale of a k tile, which is the row's
// per-token scale (`As[row]`, the W4 path producers' law) instead of a scale a 128-wide k tile; so a twin ==
// its parent on `As[r][kt] = As_tok[r]`, byte for byte.
"""
    wrap = """
#define W8PT_SIG                                                                                \\
  (const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,                    \\
   const float *__restrict__ as, const float *__restrict__ ws, __bf16 *__restrict__ c, int m,   \\
   int n, int k, int ldc)

// <= 16 rows: grid N / 64 x ceil(M / 16), 128 threads
extern "C" __global__ __launch_bounds__(DG_NT) void macaq_w8pt_n64_ldc W8PT_SIG {
  w8pt_gemm_impl<DG_BM, 64, DG_BK, DG_NW>(a, w, as, ws, c, m, n, k, (long)ldc);
}
extern "C" __global__ __launch_bounds__(DG_NT) void macaq_w8pt_n64_pf4_ldc W8PT_SIG {
  w8pt_ring_impl<DG_BM, 64, DG_BK, DG_NW, 4>(a, w, as, ws, c, m, n, k, (long)ldc);
}

#define W8PT_DEF(NAME, MF, MH, NW, MFAST, NT)                                                   \\
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME W8PT_SIG {                         \\
    w8pt_p_impl<MF, MH, NW, MFAST, NT>(a, w, as, ws, c, m, n, k, (long)ldc);                    \\
  }
// decode / verify: one row tile, grid N / 64 x ceil(M / (16 MF)), 128 threads
W8PT_DEF(macaq_w8pt_d2w4_ldc, 2, 2, 4, false, true)
W8PT_DEF(macaq_w8pt_d4w4_ldc, 4, 2, 4, false, true)
// prefill: rows fastest, grid ceil(M / 64) x N / 128, 256 threads
W8PT_DEF(macaq_w8pt_p4w8_ldc, 4, 4, 8, true, false)
"""
    return head + a + "\n" + b + "\n" + c + wrap


def main():
    text = build()
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != text:
            print(f"{OUT} differs from the generator's output", file=sys.stderr)
            return 1
        print(f"{OUT}: up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
