#!/usr/bin/env python3
"""FN164: generate the SHIPPED code object source `crates/saber-kernels/hip/flashnext_t8far.hip`
(`--fn-mixer-ar-push on|b1`): the W8A8 out / o projection GEMV of a TP = 2 decode / verify step with
the MIXER-OUT all-reduce FUSED into its epilogue (FN137's `_far` template on the trunk GEMV).

Every body is the shipped text, verbatim, by anchored substitution:
  * `flashnext_t8141.hip` up to its first kernel wrapper (every helper of `flashnext_trunk8` it carries
    + FN141's register ring) -- `t8_gemm_a8w8_impl` (= `fn111_t8_a8w8_n64_ldc`'s body at BN 64) and
    `fn141_t8_ring_impl` (= `fn141_t8_a8w8_n64_pf4_ldc`'s);
  * `flashnext_trunk8p.hip`'s `t8p_impl` (+ its helpers) (= `fn_t8p_a8w8_d{2,4}w4_ldc`'s).
Each `_far` impl keeps the parent's k loop and its store of the PARTIAL (`c`, unchanged), then stages
the same bf16 words (one rounding, the store's) in LDS and runs the exchange: the workgroup's 64
columns x M rows are pushed to the PEER's private window slot (16-byte words), drained
(`s_wait_storecnt 0`, the served drain 3), the workgroup's flag released (system scope), the peer's
same-workgroup flag acquire-spun (a WAIT bounded at 40 s of the realtime clock, FN137b h2's law), and
the SUM written with the reference's `vadd<bf16, 8>` arithmetic (fp32 add, one bf16 rounding = the exact
one-shot's bytes). The slot is the workgroup's CALL parity (FN114d's law per workgroup): each of the
40 workgroups keeps its own call counter (`ctr[blockIdx.x]`, advanced by its thread 0 at entry, only
that workgroup of each call touches it, the calls are stream-ordered), so slot j of workgroup w is
rewritten two calls later, after the peer's workgroup w passed call s+1's flag, which follows its
whole call-s kernel.

Symbols (`--fn-mixer-ar-push`): `fn164_far_<parent>{,_ctl,_chk}` for the parents
`fn111_t8_a8w8_n64_ldc` (M <= 16), `fn141_t8_a8w8_n64_pf4_ldc` (M <= 16, `--fn-k141 t8`),
`fn_t8p_a8w8_d2w4_ldc` (17..32), `fn_t8p_a8w8_d4w4_ldc` (33..64): the parent's 9 arguments + the
by-value `FarArgs` (FN137's 72-byte struct, `abi::STRUCT_FAR_ARGS`); the parent's grid (N / 64 x 1)
and 128 threads; `_ctl` = no spin (the control), `_chk` = the form + the flag-range and late-word
checks. + the oracle's skew kernel `fn164_delay`. Built with the parents' flags (no
-ffp-contract=off: the k loop contracts exactly as there).

    python3 tools/gen_fn164_t8far.py          # write
    python3 tools/gen_fn164_t8far.py --check  # exit 1 if the committed file differs
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIP = ROOT / "crates" / "saber-kernels" / "hip"
T141 = HIP / "flashnext_t8141.hip"
T8P = HIP / "flashnext_trunk8p.hip"
OUT = HIP / "flashnext_t8far.hip"


def sub1(text, old, new):
    n = text.count(old)
    assert n == 1, f"anchor occurs {n} times: {old[:90]!r}"
    return text.replace(old, new)


def slice_between(text, head, tail):
    i = text.index(head)
    j = text.index(tail, i + len(head))
    return text[i:j]


EXCHANGE = r'''
// ================================================================ FN164: the mixer-out exchange
// FMODE 0 = the form, 1 = the CONTROL (no spin), 2 = the RACE CHECKER `_chk` (the form + the
// flag-range and late-word checks). FN137's channel struct, field for field (abi::STRUCT_FAR_ARGS).
struct FarArgs {
  int4 *peer;               // the PEER's window (its slot 0; slot 1 at + stride16)
  const int4 *my;           // this rank's window (the peer writes it)
  unsigned *pflags;         // the peer's 40 workgroup flags
  unsigned *mflags;         // this rank's 40 workgroup flags (the peer writes them)
  unsigned *ctr;            // the 40 workgroups' call counters (this rank's, plain memory)
  int4 *red;                // the SUM (M x 2,560 bf16): the all-reduce's output
  unsigned *diag;           // [0] spin timeouts; _chk: [1] peer flag outside {s, s+1}, [2] a
                            // payload word that changed between two reads after the flag
  unsigned long long *stamp;  // unused (0): the served form never stamps
  int stride16;             // a slot, in 16-byte words
  int pad;
};
__device__ __forceinline__ void t8far_rel(unsigned *p, unsigned v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned t8far_acq(const unsigned *p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// saber_ar_exact.hip's `vadd<__hip_bfloat16, 8>`, verbatim arithmetic: fp32 sum, one bf16 rounding.
__device__ __forceinline__ int4 t8far_vadd(const int4 &a, const int4 &b) {
  int4 r;
  const __hip_bfloat16 *pa = reinterpret_cast<const __hip_bfloat16 *>(&a);
  const __hip_bfloat16 *pb = reinterpret_cast<const __hip_bfloat16 *>(&b);
  __hip_bfloat16 *pr = reinterpret_cast<__hip_bfloat16 *>(&r);
#pragma unroll
  for (int j = 0; j < 8; ++j) pr[j] = (__hip_bfloat16)((float)pa[j] + (float)pb[j]);
  return r;
}
// The oracle's skew source: one workgroup spins `ticks` of the 100 MHz realtime clock.
extern "C" __global__ void fn164_delay(unsigned long long ticks) {
  const unsigned long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
}
// Thread 0 advances this workgroup's call counter at entry (the load issues before the k loop).
__device__ __forceinline__ void t8far_begin(const FarArgs &fa, unsigned &s_sh) {
  if (threadIdx.x == 0) {
    const unsigned s = fa.ctr[blockIdx.x] + 1u;
    fa.ctr[blockIdx.x] = s;
    s_sh = s;
  }
}
// The workgroup's exchange over its 64 columns x (M - m0, <= BM) rows: the bf16 partial words in
// LDS `sp` ([BM][64], the store's roundings), pushed, flagged, the peer's awaited, summed.
template <int FMODE, int BM, int NT>
__device__ __forceinline__ void t8far_exchange(const __bf16 *sp, const int m0, const int M,
                                               const FarArgs &fa, const unsigned &s_sh) {
  constexpr int W16 = 2560 / 8;   // 16-byte words a row of the SUM (N = ldc = 2,560)
  constexpr int NI = (BM * 8 + NT - 1) / NT;
  __syncthreads();   // the partial tile in LDS + the call index
  const unsigned s = s_sh;
  const int wg = (int)blockIdx.x;
  const int items = min(BM, M - m0) * 8;
  const int slot = (int)(s & 1u);
  int4 *ps = fa.peer + (size_t)slot * fa.stride16;
  const int4 *ms = fa.my + (size_t)slot * fa.stride16;
  const int4 *s16 = reinterpret_cast<const int4 *>(sp);
  int4 o[NI];
#pragma unroll
  for (int j = 0; j < NI; ++j) {
    const int i = (int)threadIdx.x + j * NT;
    if (i < items) {
      const int k = (m0 + (i >> 3)) * W16 + wg * 8 + (i & 7);
      o[j] = s16[i];
      ps[k] = o[j];
    }
  }
  asm volatile("s_wait_storecnt 0x0" ::: "memory");   // the served drain (3)
  __syncthreads();
  if (threadIdx.x == 0) {
    t8far_rel(&fa.pflags[wg], s);
    if (FMODE != 1) {
      const unsigned long long t0 = wall_clock64();
      while (t8far_acq(&fa.mflags[wg]) < s) {
        // 40 s of the 100 MHz realtime clock: a peer KFD-queue eviction is a WAIT, never a
        // timeout (FN137b h2's law; the shipped one-shot bounds its spin at the same order)
        if (wall_clock64() - t0 > 4000000000ull) { atomicAdd(&fa.diag[0], 1u); break; }
      }
    }
    if (FMODE == 2) {   // _chk: the peer is at this call or the next one, never further
      const unsigned v = t8far_acq(&fa.mflags[wg]);
      if (v != s && v != s + 1u) atomicAdd(&fa.diag[1], 1u);
    }
  }
  __syncthreads();   // the served acq = 0: no fence after
#pragma unroll
  for (int j = 0; j < NI; ++j) {
    const int i = (int)threadIdx.x + j * NT;
    if (i < items) {
      const int k = (m0 + (i >> 3)) * W16 + wg * 8 + (i & 7);
      const int4 pv = ms[k];
      fa.red[k] = t8far_vadd(o[j], pv);
      if (FMODE == 2) {   // _chk: the landed word must not change after the flag (a late write)
        __builtin_amdgcn_s_sleep(4);
        const unsigned *pw = reinterpret_cast<const unsigned *>(&ms[k]);
        const unsigned w0 = __hip_atomic_load(pw + 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const unsigned w1 = __hip_atomic_load(pw + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const unsigned w2 = __hip_atomic_load(pw + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const unsigned w3 = __hip_atomic_load(pw + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if (w0 != (unsigned)pv.x || w1 != (unsigned)pv.y || w2 != (unsigned)pv.z || w3 != (unsigned)pv.w)
          atomicAdd(&fa.diag[2], 1u);
      }
    }
  }
}
'''


def far_n64(src):
    """`t8_gemm_a8w8_impl` -> `t8_gemm_a8w8_far_impl` (one fragment a wave at BN 64)."""
    t = slice_between(src, "template <int BM, int BN, int BK, int NW>\n__device__ __forceinline__ void t8_gemm_a8w8_impl(",
                      "\n// ====")
    t = sub1(t, "template <int BM, int BN, int BK, int NW>\n__device__ __forceinline__ void t8_gemm_a8w8_impl(",
             "template <int BM, int BN, int BK, int NW, int FMODE>\n__device__ __forceinline__ void t8_gemm_a8w8_far_impl(")
    t = sub1(t, "    int M, int N, int K, long LDC) {\n  constexpr int NT = NW * 32;",
             "    int M, int N, int K, long LDC, const FarArgs fa) {\n  constexpr int NT = NW * 32;")
    t = sub1(t, "  if (m0 >= M) return;\n  const int ktiles = K / BK;\n",
             "  if (m0 >= M) return;\n"
             "  __shared__ unsigned t8far_s;   // FN164: this workgroup's call index\n"
             "  t8far_begin(fa, t8far_s);\n"
             "  const int ktiles = K / BK;\n")
    t = sub1(t,
             "#pragma unroll\n  for (int i = 0; i < MYF; ++i)\n"
             "    dg_store_acc(acc[i], C, n0 + (wave + i * NW) * 16, m0, M, N, LDC, lane);\n}\n",
             "#pragma unroll\n  for (int i = 0; i < MYF; ++i)\n"
             "    dg_store_acc(acc[i], C, n0 + (wave + i * NW) * 16, m0, M, N, LDC, lane);\n"
             "  // FN164: the same words (the store's rounding) staged for the exchange\n"
             "  static_assert(BN == 64, \"the exchange's 40 workgroups of 64 columns\");\n"
             "  __shared__ __bf16 t8far_p[BM * BN];\n"
             "#pragma unroll\n  for (int i = 0; i < MYF; ++i)\n"
             "#pragma unroll\n    for (int e = 0; e < 8; ++e)\n"
             "      t8far_p[dg_acc_row(lane, e) * BN + (wave + i * NW) * 16 + (lane & 15)] = (__bf16)acc[i][e];\n"
             "  t8far_exchange<FMODE, BM, NT>(t8far_p, m0, M, fa, t8far_s);\n}\n")
    return t


def far_ring(src):
    """`fn141_t8_ring_impl` -> `fn141_t8_ring_far_impl`."""
    t = slice_between(src, "template <int BM, int BN, int BK, int NW, int PF>\n__device__ __forceinline__ void fn141_t8_ring_impl(",
                      "\n// ---- ")
    t = sub1(t, "template <int BM, int BN, int BK, int NW, int PF>\n__device__ __forceinline__ void fn141_t8_ring_impl(",
             "template <int BM, int BN, int BK, int NW, int PF, int FMODE>\n__device__ __forceinline__ void fn141_t8_ring_far_impl(")
    t = sub1(t, "    int M, int N, int K, long LDC) {\n  static_assert(BM == 16 && BN == 16 * NW",
             "    int M, int N, int K, long LDC, const FarArgs fa) {\n  static_assert(BM == 16 && BN == 16 * NW")
    t = sub1(t, "  if (m0 >= M) return;\n  const int KT = K / BK;\n",
             "  if (m0 >= M) return;\n"
             "  __shared__ unsigned t8far_s;   // FN164: this workgroup's call index\n"
             "  t8far_begin(fa, t8far_s);\n"
             "  const int KT = K / BK;\n")
    t = sub1(t, "  dg_store_acc(acc, C, n0 + wave * 16, m0, M, N, LDC, lane);\n}\n",
             "  dg_store_acc(acc, C, n0 + wave * 16, m0, M, N, LDC, lane);\n"
             "  // FN164: the same words (the store's rounding) staged for the exchange\n"
             "  static_assert(BN == 64, \"the exchange's 40 workgroups of 64 columns\");\n"
             "  __shared__ __bf16 t8far_p[BM * BN];\n"
             "#pragma unroll\n  for (int e = 0; e < 8; ++e)\n"
             "    t8far_p[dg_acc_row(lane, e) * BN + wave * 16 + (lane & 15)] = (__bf16)acc[e];\n"
             "  t8far_exchange<FMODE, BM, NT>(t8far_p, m0, M, fa, t8far_s);\n}\n")
    return t


def far_t8p(src):
    """trunk8p's helpers + `t8p_impl` -> `t8p_far_impl` (decode tiles: column tiles on x)."""
    helpers = slice_between(src, "#define P_BK 128\n", "template <int MF, int MH, int NW, bool MFAST, bool NT>\n")
    t = slice_between(src, "template <int MF, int MH, int NW, bool MFAST, bool NT>\n__device__ __forceinline__ void t8p_impl(",
                      "#define T8P_SIG")
    t = sub1(t, "template <int MF, int MH, int NW, bool MFAST, bool NT>\n__device__ __forceinline__ void t8p_impl(",
             "template <int MF, int MH, int NW, bool MFAST, bool NT, int FMODE>\n__device__ __forceinline__ void t8p_far_impl(")
    t = sub1(t, "    int M, int N, int K, long LDC) {\n  constexpr int BM = 16 * MF;",
             "    int M, int N, int K, long LDC, const FarArgs fa) {\n  constexpr int BM = 16 * MF;")
    t = sub1(t, "  const int kb = K / P_BK;\n",
             "  const int kb = K / P_BK;\n"
             "  static_assert(!MFAST && NW == 4, \"FN164: the decode tiles (40 column workgroups on x)\");\n"
             "  __shared__ unsigned t8far_s;   // FN164: this workgroup's call index\n"
             "  t8far_begin(fa, t8far_s);\n")
    t = sub1(t,
             "      if (row < M) C[(long)row * LDC + col] = (__bf16)acc[i][e];\n    }\n}\n",
             "      if (row < M) C[(long)row * LDC + col] = (__bf16)acc[i][e];\n    }\n"
             "  // FN164: the same words (the store's rounding) staged for the exchange\n"
             "  __shared__ __bf16 t8far_p[BM * 64];\n"
             "#pragma unroll\n  for (int i = 0; i < MF; ++i)\n"
             "#pragma unroll\n    for (int e = 0; e < 8; ++e)\n"
             "      t8far_p[(i * 16 + (lane >> 4) * 8 + e) * 64 + wave * 16 + (lane & 15)] = (__bf16)acc[i][e];\n"
             "  t8far_exchange<FMODE, BM, NTH>(t8far_p, m0, M, fa, t8far_s);\n}\n")
    return helpers, t


MODES = ((0, ""), (1, "_ctl"), (2, "_chk"))
SIG = ("    const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,\n"
       "    const float *__restrict__ as, const float *__restrict__ ws, __bf16 *__restrict__ c,\n"
       "    int m, int n, int k, int ldc, FarArgs fa")
PARENTS = (
    ("fn111_t8_a8w8_n64_ldc", "t8_gemm_a8w8_far_impl<DG_BM, 64, DG_BK, DG_NW, {fm}>"),
    ("fn141_t8_a8w8_n64_pf4_ldc", "fn141_t8_ring_far_impl<DG_BM, 64, DG_BK, DG_NW, 4, {fm}>"),
    ("fn_t8p_a8w8_d2w4_ldc", "t8p_far_impl<2, 2, 4, false, true, {fm}>"),
    ("fn_t8p_a8w8_d4w4_ldc", "t8p_far_impl<4, 2, 4, false, true, {fm}>"),
)


def gen(t141, t8p):
    first = t141.index('extern "C" __global__')
    prefix = t141[:first].rstrip() + "\n"
    assert "t8_gemm_a8w8_impl(" in prefix and "fn141_t8_ring_impl(" in prefix
    helpers, p = far_t8p(t8p)
    wr = []
    for parent, impl in PARENTS:
        for fm, suf in MODES:
            wr.append(f'extern "C" __global__ __launch_bounds__(128) void fn164_far_{parent}{suf}(\n{SIG}) {{\n'
                      f"  {impl.format(fm=fm)}(a, w, as, ws, c, m, n, k, (long)ldc, fa);\n}}\n")
    head = ("// GENERATED by tools/gen_fn164_t8far.py -- do not edit by hand, re-run the generator.\n"
            "// FN164 (`--fn-mixer-ar-push`): the W8A8 out / o projection with the mixer-out all-reduce\n"
            "// fused into its epilogue. The text up to the FN164 marker below is `flashnext_t8141.hip`\n"
            "// up to its first kernel wrapper, verbatim, then `flashnext_trunk8p.hip`'s helpers verbatim;\n"
            "// the `_far` impls are the parents' text by anchored substitution (the FN164 lines marked).\n")
    return (head + prefix + "\n// ---- flashnext_trunk8p.hip's helpers, verbatim\n" + helpers + EXCHANGE + "\n"
            + far_n64(prefix) + "\n" + far_ring(prefix) + "\n" + p + "\n" + "\n".join(wr))


def main():
    txt = gen(T141.read_text(), T8P.read_text())
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != txt:
            print("DIFFERS: flashnext_t8far.hip")
            sys.exit(1)
        print("flashnext_t8far.hip == generator")
        return
    OUT.write_text(txt)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(txt)} B)")


if __name__ == "__main__":
    main()
