#!/usr/bin/env python3
"""FN137 Phase B: generate the SHIPPED code object source `crates/saber-kernels/hip/moe_grouped_far.hip`
(`--fn-moe-ar-push on`): FN80b-1's half-width fused back half `moe_back_h2_impl` (sliced from the
committed, generated `moe_grouped_tp2.hip`) with the MoE-out all-reduce FUSED into its combine: after
every combine slice (the unchanged `mx_combine_slice` call) the slice's workgroup pushes its bf16
partial words to the PEER's private window slot, drains (`s_wait_storecnt 0`, the served drain 3),
releases the slice's flag (system scope), acquire-spins on the peer's same-slice flag and writes the
slice of the SUM with the reference's `vadd<bf16, 8>` arithmetic (fp32 add, one bf16 rounding: the exact
one-shot's bytes). The slot is the channel's CALL parity (FN114d's law): the ticket-0 workgroup
advances the channel counter and publishes the call index before its `sgf` release; every slice
reads it after its acquire of `sgf`.

The file is: `moe_grouped_tp2.hip`'s text up to its first kernel wrapper (every helper the back half
calls, verbatim -- no shipped symbol is re-instantiated here, so `moe_grouped_tp2` keeps its bytes
and the new object holds only FN137's symbols) + the exchange helpers + the far impl (anchored
substitution of the shipped impl, the FN137 lines marked) + the wrappers
`fn137_moe_back_h2_far_gu{4,1}_dn{2,1}_p1{,_ctl,_chk}` and the oracle's skew kernel `fn137_delay`.
Phase A's probe (`tools/capture/fn137/fn137_probe.cpp`) loads the same symbol names.

    python3 tools/gen_fn137b_far.py          # write
    python3 tools/gen_fn137b_far.py --check  # exit 1 if the committed file differs
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIP = ROOT / "crates" / "saber-kernels" / "hip"
TP2 = HIP / "moe_grouped_tp2.hip"
OUT = HIP / "moe_grouped_far.hip"


def sub1(text, old, new):
    n = text.count(old)
    assert n == 1, f"anchor occurs {n} times: {old[:80]!r}"
    return text.replace(old, new)


def slice_between(text, head, tail):
    i = text.index(head)
    j = text.index(tail, i)
    return text[i:j]


FAR_HELPERS = r'''
// ---------------------------------------------------------------- FN137: the exchange (FMODE 0 the form,
// 1 the CONTROL: no spin, 2 the RACE CHECKER `_chk`: the form + the flag-range and late-word checks)
struct FarArgs {
  int4 *peer;               // the PEER's window (its slot 0; slot 1 at + stride16)
  const int4 *my;           // this rank's window (the peer writes it)
  unsigned *pflags;         // the peer's 20 slice flags
  unsigned *mflags;         // this rank's 20 slice flags (the peer writes them)
  unsigned *ctr;            // [0] the channel's call counter, [1] this call's index
  int4 *red;                // the SUM (T x 2,560 bf16): the all-reduce's output
  unsigned *diag;           // [0] spin timeouts; _chk: [1] peer flag outside {s, s+1}, [2] a
                            // payload word that changed between two reads after the flag
  unsigned long long *stamp;  // timing only: [call % 64][21][8] s_memrealtime
  int stride16;             // a slot, in 16-byte words
  int pad;
};
__device__ __forceinline__ void far_rel(unsigned *p, unsigned v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned far_acq(const unsigned *p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// saber_ar_exact.hip's `vadd<__hip_bfloat16, 8>`, verbatim arithmetic: fp32 sum, one bf16 rounding.
__device__ __forceinline__ int4 far_vadd(const int4 &a, const int4 &b) {
  int4 r;
  const __hip_bfloat16 *pa = reinterpret_cast<const __hip_bfloat16 *>(&a);
  const __hip_bfloat16 *pb = reinterpret_cast<const __hip_bfloat16 *>(&b);
  __hip_bfloat16 *pr = reinterpret_cast<__hip_bfloat16 *>(&r);
#pragma unroll
  for (int j = 0; j < 8; ++j) pr[j] = (__hip_bfloat16)((float)pa[j] + (float)pb[j]);
  return r;
}
__device__ __forceinline__ unsigned long long far_now() { return wall_clock64(); }
// The oracle's skew source: one workgroup spins `ticks` of the 100 MHz realtime clock.
extern "C" __global__ void fn137_delay(unsigned long long ticks) {
  const unsigned long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
}
__device__ __forceinline__ void far_st(const FarArgs &fa, unsigned s, int row, int j) {
  if (fa.stamp && threadIdx.x == 0) fa.stamp[((size_t)(s & 63u) * 21 + row) * 8 + j] = far_now();
}
// The ticket-0 workgroup: advance the channel's call counter, publish the index (before `sgf`).
__device__ __forceinline__ unsigned far_call_begin(const FarArgs &fa) {
  __shared__ unsigned s_far;
  if (threadIdx.x == 0) {
    const unsigned s = fa.ctr[0] + 1u;
    fa.ctr[0] = s;
    fa.ctr[1] = s;
    s_far = s;
  }
  __syncthreads();
  const unsigned s = s_far;
  far_st(fa, s, 20, 0);
  return s;
}
__device__ __forceinline__ unsigned far_call_read(const FarArgs &fa) {
  return __builtin_amdgcn_readfirstlane(__hip_atomic_load(&fa.ctr[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
}
// The slice's exchange: the words `mx_combine_slice` wrote for slice nb (its own thread mapping),
// pushed, flagged, the peer's awaited, summed.
template <int FMODE>
__device__ __forceinline__ void far_exchange(const int nb, const int T, const __bf16 *__restrict__ out,
                                             const FarArgs &fa, const unsigned s) {
  constexpr int W16 = 2560 / 8;   // 16-byte words a row
  const int slot = (int)(s & 1u);
  int4 *ps = fa.peer + (size_t)slot * fa.stride16;
  const int4 *ms = fa.my + (size_t)slot * fa.stride16;
  far_st(fa, s, nb, 0);
  int4 o[4];   // T <= 64: T * 16 <= 4 x 256 items a thread, static indices (no scratch)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
    if (i < T * 16) {
      const int k = (i >> 4) * W16 + nb * 16 + (i & 15);
      o[j] = ((const int4 *)out)[k];
      ps[k] = o[j];
    }
  }
  far_st(fa, s, nb, 1);
  asm volatile("s_wait_storecnt 0x0" ::: "memory");   // the served drain (3)
  __syncthreads();
  far_st(fa, s, nb, 2);
  if (threadIdx.x == 0) {
    far_rel(&fa.pflags[nb], s);
    if (fa.stamp) fa.stamp[((size_t)(s & 63u) * 21 + nb) * 8 + 3] = far_now();
    if (FMODE != 1) {
      const unsigned long long t0 = far_now();
      while (far_acq(&fa.mflags[nb]) < s) {
        // 40 s of the 100 MHz realtime clock: a peer KFD-queue eviction (FN121: > 50 ms served steps)
        // is a WAIT, never a timeout (FN137b h2: a 50 ms bound timed out 19 times on a served boot);
        // the shipped one-shot bounds its spin at SABER_AR_SPIN_MAX polls, the same order
        if (far_now() - t0 > 4000000000ull) { atomicAdd(&fa.diag[0], 1u); break; }
      }
    }
    if (FMODE == 2) {   // _chk: the peer is at this call or the next one, never further
      const unsigned v = far_acq(&fa.mflags[nb]);
      if (v != s && v != s + 1u) atomicAdd(&fa.diag[1], 1u);
    }
  }
  __syncthreads();                                     // the served acq = 0: no fence after
  far_st(fa, s, nb, 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
    if (i < T * 16) {
      const int k = (i >> 4) * W16 + nb * 16 + (i & 15);
      const int4 pv = ms[k];
      fa.red[k] = far_vadd(o[j], pv);
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
  if (fa.stamp) {
    __syncthreads();
    far_st(fa, s, nb, 5);
  }
}
'''


MODES = ((0, ""), (1, "_ctl"), (2, "_chk"))


def gen(src):
    first = src.index('extern "C" __global__')
    prefix = src[:first].rstrip() + "\n"
    assert "moe_back_h2_impl(" in prefix
    impl = slice_between(src, "template <int KSG, int KSD, int PFD>\n__device__ __forceinline__ void moe_back_h2_impl(",
                         'extern "C" __global__')
    t = impl
    t = sub1(t, "template <int KSG, int KSD, int PFD>\n__device__ __forceinline__ void moe_back_h2_impl(",
             "template <int KSG, int KSD, int PFD, int FMODE>\n__device__ __forceinline__ void moe_back_h2_far_impl(")
    t = sub1(t, "    const int tc) {\n  constexpr int NBG = 640 / 128",
             "    const int tc, const FarArgs fa) {\n  constexpr int NBG = 640 / 128")
    t = sub1(t,
             "    if (ntile == 0) {   // no expert rows on this rank: the combine is its (all-remote) zero sum\n"
             "      for (int nb = 0; nb < NBD; ++nb) mx_combine_slice(nb, T, Y, topk_w, route_pos, sg, meta, out);\n",
             "    const unsigned fs = far_call_begin(fa);   // FN137: the call index, before `sgf`\n"
             "    if (ntile == 0) {   // no expert rows on this rank: the combine is its (all-remote) zero sum\n"
             "      for (int nb = 0; nb < NBD; ++nb) {\n"
             "        mx_combine_slice(nb, T, Y, topk_w, route_pos, sg, meta, out);\n"
             "        far_exchange<FMODE>(nb, T, out, fa, fs);   // FN137\n"
             "      }\n")
    t = sub1(t,
             "  mx_combine_slice(nb, T, Y, topk_w, route_pos, sg, meta, out);\n  if (tid == 0 && atomicAdd(cdone, 1) == NBD - 1) {",
             "  mx_combine_slice(nb, T, Y, topk_w, route_pos, sg, meta, out);\n"
             "  far_exchange<FMODE>(nb, T, out, fa, far_call_read(fa));   // FN137\n"
             "  if (tid == 0 && atomicAdd(cdone, 1) == NBD - 1) {")
    params = ("    const unsigned char *__restrict__ A, const float *__restrict__ As,\n"
              "    const unsigned long long *__restrict__ xgu, const unsigned long long *__restrict__ xdn,\n"
              "    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,\n"
              "    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,\n"
              "    const int *__restrict__ meta, float *__restrict__ Pgu, float *__restrict__ Pdn,\n"
              "    int *__restrict__ cgu, int *__restrict__ cdn, __bf16 *__restrict__ GU,\n"
              "    unsigned char *__restrict__ Q, float *__restrict__ QS, __bf16 *__restrict__ Y,\n"
              "    const float *__restrict__ topk_w, const int *__restrict__ route_pos,\n"
              "    const __bf16 *__restrict__ wsg, const __bf16 *__restrict__ bi, __bf16 *__restrict__ sg,\n"
              "    __bf16 *__restrict__ out, int *__restrict__ sync, int T, int Pmax, int tc, FarArgs fa")
    call = ("A, As, xgu, xdn, tile_expert, tile_row0, tile_rows, row_tok, meta, Pgu, Pdn, cgu, cdn, GU, Q, QS, Y,\n"
            "      topk_w, route_pos, wsg, bi, sg, out, sync, T, Pmax, tc, fa")
    wr = []
    for gk, dk in ((4, 2), (1, 1)):
        for fm, suf in MODES:
            wr.append(f'extern "C" __global__ __launch_bounds__(256) void fn137_moe_back_h2_far_gu{gk}_dn{dk}_p1{suf}(\n'
                      f"{params}) {{\n  moe_back_h2_far_impl<{gk}, {dk}, 1, {fm}>({call});\n}}\n")
    head = ("// GENERATED by tools/gen_fn137b_far.py -- do not edit by hand, re-run the generator.\n"
            "// FN137 Phase B (`--fn-moe-ar-push on`): the half-width fused back half with the MoE-out\n"
            "// all-reduce fused into its combine slices. The text up to the FN137 marker below is\n"
            "// `moe_grouped_tp2.hip` up to its first kernel wrapper, verbatim (generated by\n"
            "// tools/gen_moe_grouped.py), so every helper lowers from the shipped text.\n")
    return head + prefix + FAR_HELPERS + "\n" + t + "\n".join(wr)


def main():
    txt = gen(TP2.read_text())
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != txt:
            print("DIFFERS: moe_grouped_far.hip")
            sys.exit(1)
        print("moe_grouped_far.hip == generator")
        return
    OUT.write_text(txt)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(txt)} B)")


if __name__ == "__main__":
    main()
