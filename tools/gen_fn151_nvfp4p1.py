#!/usr/bin/env python3
"""FN151: derive `crates/saber-kernels/hip/moe_fn151.hip` -- FN148's FORM (b) (exact NVFP4 weights: plain e2m1
bytes, `tile = wmma(a8, e2m1, 0); acc = fma(tile, E4M3(s_g), acc)` per 16-deep k-group, epilogue x gs x As) in
the SERVED `_p1` class: `moe_grouped_w4a8_x_impl`'s WPERM + NT PFD-1 register double buffer (slab s + 1's W words,
scale bytes and A row in registers while slab s multiplies), the same slab partition, the same split-K partials and
fused reduce. Byte law: every output byte == FN148's `fn148_w4a8b_*` cell of the same (bk, ks, tm, g).

What changes vs the shipped `_p1` (MXFP4) block:
  * no `wref` prologue and no kMag[d] lookup: the fold table is the constant plain e2m1 magnitude pair
    (kMag[0] = 0x3c383000 / 0x4c484440 as immediates) -> the staging has NO dependent table reads;
  * the prefetched scale word is the E4M3 g16 pair of the slot's two rows at the slot's 16-k group (one u16);
    `pf_put` stores it into a [BND][DBK/16] byte table in LDS beside sW (SC bit 1 clear), and the WMMA lane reads
    its column's DBK/16 scale bytes with ONE ds_load (b64 at bk128, b32 at bk64) after the slab barrier and
    converts each with v_cvt_f32_fp8 (byte select) -- or (SC bit 1 set) the lane prefetches its own column's
    bytes from global one slab ahead (the attribution arm);
  * the live-row ladder (SC bit 0): the accumulator element e of a lane holds rows e (lanes 0-15) and 8 + e
    (lanes 16-31) of a 16-row tile, so a tile of M <= 4 rows only needs elements e < M -> the slab loop is
    instantiated at NE = 1 / 2 / 4 / 8 live elements (wave-uniform branch on the tile's M, once a launch) and the
    per-WMMA scale FMAs drop from 8 to NE; at NE < 8 the tiles i >= 1 are wholly dead (M <= 4 < 16) and are
    skipped. Dead rows are never stored (every C / P store and the reduce test m < M), so the live bytes are the
    8-FMA body's (the NE = 8 instantiation == FN148's chain op for op).
Source: `moe_grouped_far.hip` (FN137b's text = `moe_grouped_tp2`'s helpers verbatim; the impl is the shipped x_impl),
the prefix up to the x_impl copied verbatim (helpers), the x_impl by anchored substitution (asserted once each).
  python3 tools/gen_fn151_nvfp4p1.py          # write
  python3 tools/gen_fn151_nvfp4p1.py --check  # the committed file == the derivation
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "crates/saber-kernels/hip/moe_grouped_far.hip")
OUT = os.path.join(REPO, "crates/saber-kernels/hip/moe_fn151.hip")

# (dbk, ks, tm, gather, SC, suffix): SC bit 0 = live-row ladder, bit 1 = scales in registers (else LDS)
CELLS = (
    [(128, ks, 1, 1, 1, "") for ks in (1, 2, 4)]
    + [(128, ks, tm, 0, 1, "") for ks in (1, 2) for tm in (1, 2, 4)]
    + [(64, ks, 1, 0, 1, "") for ks in (1, 2)]
    # attribution arms at the served ks1 cells: no ladder (8 FMAs a WMMA), register scales
    + [(128, 1, 1, 1, 0, "_f8"), (128, 1, 1, 1, 3, "_r"), (64, 1, 1, 0, 0, "_f8"), (64, 1, 1, 0, 3, "_r")]
    + [(128, 4, 1, 1, 0, "_f8"), (64, 2, 1, 0, 0, "_f8")]
    # SC bit 2: global loads + LDS-scope slab barriers (no per-slab release drain), at the served cells
    + [(128, ks, 1, 1, 5, "_l") for ks in (1, 2, 4)] + [(128, ks, 1, 1, 7, "_lr") for ks in (1, 2, 4)]
    + [(64, ks, 1, 0, 5, "_l") for ks in (1, 2)] + [(64, ks, 1, 0, 7, "_lr") for ks in (1, 2)]
    + [(128, 1, 1, 0, 5, "_l"), (128, 1, 1, 0, 7, "_lr")]
)


def sub(text, old, new, count=1):
    n = text.count(old)
    assert n == count, (n, count, old[:80])
    return text.replace(old, new)


PFD1 = r'''  static_assert(WPERM && PFD == 1, "FN151: the _p1 (WPERM, PFD 1) body only");
  constexpr int KSTEPS_P = DBK / 16, NTILES_P = BND / 16;
  constexpr int NWI = NTILES_P * KSTEPS_P * 32 / (DNTHREADS * 2);
  static_assert(NWI * DNTHREADS * 2 == NTILES_P * KSTEPS_P * 32, "WPERM slot pairs tile the block");
  static_assert(KSTEPS_P == 4 || KSTEPS_P == 8, "FN151: bk 64 / 128");
  // FN151: the slab's E4M3 g16 scale bytes, [BND rows][KSTEPS_P groups] (SC bit 1 clear).
  __shared__ __attribute__((aligned(8))) unsigned char s_sc151[BND * KSTEPS_P];
  const unsigned int *__restrict__ Wp = (const unsigned int *)W;
  const int ksteps_g = K / 16;
  const int ncol = n0 + wave * 16 + col < N - 1 ? n0 + wave * 16 + col : N - 1;
  auto pf_load_w = [&](int k0, uint2_t (&wv)[NWI], unsigned int (&ws2)[NWI]) {
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NWI; ++it) {
      const int sl = it * DNTHREADS * 2 + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_P, ntl = rest / KSTEPS_P;
      const int r = ntl * 16 + (lane_ & 15);
      const int gn = n0 + r;
      const int gc = gn < N - 2 ? gn : N - 2;
      const int lanec = (lane_ & 16) | (gc & 15);
      const uint2_t *pw_ = (const uint2_t *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
      const unsigned short *ps_ = (const unsigned short *)&Ws[(size_t)(kstep0 + kst) * N + gc];
      if constexpr ((SC & 4) != 0) {   // global (address-space-1) loads: LOADcnt only, never DScnt
        wv[it] = mx_ldg<NT>(pw_);
        if constexpr ((SC & 2) == 0) ws2[it] = (unsigned int)mx_ldg<false>(ps_);
      } else {
        wv[it] = ld_w<NT>(pw_);
        if constexpr ((SC & 2) == 0) ws2[it] = (unsigned int)*ps_;
      }
    }
  };
  auto pf_load_s = [&](int k0, unsigned int (&sr)[KSTEPS_P / 4]) {
    if constexpr ((SC & 2) != 0) {
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int j = 0; j < KSTEPS_P / 4; ++j) {
        unsigned int v = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b)
          v |= (unsigned int)((SC & 4) != 0 ? mx_ldg<false>(&Ws[(size_t)(kstep0 + 4 * j + b) * N + ncol])
                                            : Ws[(size_t)(kstep0 + 4 * j + b) * N + ncol]) << (8 * b);
        sr[j] = v;
      }
    }
  };
  auto pf_load_a = [&](int k0, uint4_t (&av)[NITA]) {
#pragma unroll
    for (int i = 0; i < NITA; ++i) {
      const int idx = i * DNTHREADS * 16 + tid * 16;
      if constexpr ((SC & 4) != 0)   // unpredicated (arow_base of a spare slot = row 0): no execz join
        av[i] = mx_ldg<false>((const uint4_t *)(arow_base[i] + k0 + idx % DBK));
      else if (idx < DEC_MTILE * DTM * DBK) av[i] = *(const uint4_t *)(arow_base[i] + k0 + idx % DBK);
    }
  };
  // SC bit 2: the slab barriers are LDS-scope (FN31's PFD 2 barrier: no release drain of the prefetch)
  auto bar = [&]() {
    if constexpr ((SC & 4) != 0) mx_lds_barrier();
    else __syncthreads();
  };
  auto pf_put = [&](const uint2_t (&wv)[NWI], const unsigned int (&ws2)[NWI],
                    const uint4_t (&av)[NITA]) {
#pragma unroll
    for (int i = 0; i < NITA; ++i) {
      const int idx = i * DNTHREADS * 16 + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        *(uint4_t *)(&sA[r * DASTR + c]) = av[i];
      }
    }
    // the plain e2m1 magnitudes as e4m3 bytes (the d = 0 row of the fold table): constant, no table read
    const unsigned int t0 = 0x3c383000u, t1 = 0x4c484440u;
#pragma unroll
    for (int it = 0; it < NWI; ++it) {
      const int sl = it * DNTHREADS * 2 + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_P, ntl = rest / KSTEPS_P;
      const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const unsigned int ev = wq_dec(wv[it], q) & 0x0F0F0F0Fu;
        const unsigned int od = (wq_dec(wv[it], q) >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
            uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                    __builtin_amdgcn_perm(bo, be, 0x07030602u)};
      }
      if constexpr ((SC & 2) == 0) {
        if ((lane_ & 16) == 0) {   // the k-half 0 slot of a (row pair, group) writes its two scale bytes
          s_sc151[r * KSTEPS_P + kst] = (unsigned char)(ws2[it] & 0xFFu);
          s_sc151[(r + 1) * KSTEPS_P + kst] = (unsigned char)((ws2[it] >> 8) & 0xFFu);
        }
      }
    }
  };
  // One slab: NE live accumulator elements (the ladder; NE == 8 = FN148's chain op for op).
  auto mma_slab = [&](auto ne_c, const unsigned int (&sr)[KSTEPS_P / 4]) {
    constexpr int NE = decltype(ne_c)::value;
    constexpr int NT_ = NE == 8 ? DTM : 1;   // NE < 8 => M <= 4: the tiles i >= 1 are wholly dead
    bar();
    unsigned int scw[KSTEPS_P / 4];
    if constexpr ((SC & 2) == 0) {
      if constexpr (KSTEPS_P == 8) {
        const uint2_t v = *(const uint2_t *)&s_sc151[(wave * 16 + col) * 8];
        scw[0] = v[0];
        scw[1] = v[1];
      } else {
        scw[0] = *(const unsigned int *)&s_sc151[(wave * 16 + col) * 4];
      }
    } else {
#pragma unroll
      for (int j = 0; j < KSTEPS_P / 4; ++j) scw[j] = sr[j];
    }
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < NT_; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
      const float scv = __builtin_amdgcn_cvt_f32_fp8((int)(scw[step >> 2] >> (8 * (step & 3))), 0);
      const floatx8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < NT_; ++i) {
        const floatx8 t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, z);
#pragma unroll
        for (int e = 0; e < NE; ++e) acc[i][e] = __builtin_fmaf(t[e], scv, acc[i][e]);
      }
    }
    bar();
  };
  auto run = [&](auto ne_c) {
    uint2_t wv0[NWI], wv1[NWI];
    unsigned int ws0[NWI], ws1[NWI];
    unsigned int sr0[KSTEPS_P / 4], sr1[KSTEPS_P / 4];
    uint4_t av0[NITA], av1[NITA];
#pragma unroll
    for (int i = 0; i < NITA; ++i) {
      av0[i] = uint4_t{0u, 0u, 0u, 0u};
      av1[i] = av0[i];
    }
#pragma unroll
    for (int j = 0; j < KSTEPS_P / 4; ++j) sr0[j] = sr1[j] = 0u;
#pragma unroll
    for (int it = 0; it < NWI; ++it) ws0[it] = ws1[it] = 0u;
    if (s_lo < s_hi) { pf_load_w(s_lo * DBK, wv0, ws0); pf_load_s(s_lo * DBK, sr0); }
    const bool pre2 = mx_pre2(wait) && s_lo + 1 < s_hi;
    if (pre2) { pf_load_w((s_lo + 1) * DBK, wv1, ws1); pf_load_s((s_lo + 1) * DBK, sr1); }
    wait();
    if (s_lo < s_hi) pf_load_a(s_lo * DBK, av0);
    for (int s = s_lo; s < s_hi; s += 2) {
      const bool m1 = s + 1 < s_hi;
      if (m1) {
        if (!pre2 || s != s_lo) { pf_load_w((s + 1) * DBK, wv1, ws1); pf_load_s((s + 1) * DBK, sr1); }
        pf_load_a((s + 1) * DBK, av1);
      }
      pf_put(wv0, ws0, av0);
      mma_slab(ne_c, sr0);
      if (!m1) break;
      if (s + 2 < s_hi) {
        pf_load_w((s + 2) * DBK, wv0, ws0);
        pf_load_s((s + 2) * DBK, sr0);
        pf_load_a((s + 2) * DBK, av0);
      }
      pf_put(wv1, ws1, av1);
      mma_slab(ne_c, sr1);
    }
  };
  if constexpr ((SC & 1) != 0) {
    if (M <= 1) run(fn151_ic<1>{});
    else if (M <= 2) run(fn151_ic<2>{});
    else if (M <= 4) run(fn151_ic<4>{});
    else run(fn151_ic<8>{});
  } else {
    run(fn151_ic<8>{});
  }
'''


def derive():
    src = open(SRC).read()
    tmpl = ("template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool GATHER, bool DIRECT, bool NT,\n"
            "          int PFD, typename WaitF>\n__device__ __forceinline__ void moe_grouped_w4a8_x_impl(")
    lo = src.index(tmpl)
    hi = src.index("// The silu-quant of ONE expert-major row with a WAVE")
    prefix, impl = src[:lo], src[lo:hi].rstrip() + "\n"
    assert impl.count("moe_grouped_w4a8_x_impl(") == 1
    impl = sub(impl, tmpl, "template <int SC, int DWN, int DBK, int DKS, int DTM, bool WPERM, bool GATHER, bool DIRECT,\n"
                           "          bool NT, int PFD, typename WaitF>\n__device__ __forceinline__ void fn151_x_impl(")
    impl = sub(impl, "    __bf16 *__restrict__ C, int N, int K, int Pmax, unsigned char *__restrict__ smem,\n",
               "    __bf16 *__restrict__ C, int N, int K, int Pmax, int gsplit, unsigned char *__restrict__ smem,\n")
    impl = sub(impl, "  const unsigned char *__restrict__ Wref =\n"
                     "      (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 2];\n",
               "  const float *__restrict__ Gs = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];\n")
    # the staging lambdas of the other loops (Wref / kMag[d] readers) and stage_a: dead here, dropped
    a = impl.index("  auto stage_a = [&](int k0) {")
    b = impl.index("  if constexpr (WPERM && PFD == 1) {\n")
    impl = impl[:a] + PFD1 + impl[impl.index("\n  const int n = n0 + wave * 16 + col;", b) + 1:]
    impl = sub(impl, "      const float rf = __int_as_float((int)Wref[n] << 23);\n",
               "      const float rf = Gs[n < gsplit ? 0 : 1];\n")
    impl = sub(impl, "    const float rf = __int_as_float((int)Wref[nn] << 23);\n",
               "    const float rf = Gs[nn < gsplit ? 0 : 1];\n")
    assert "Wref" not in impl and "kMag" not in impl, "a Wref / kMag use survived"
    prefix = sub(prefix, "// GENERATED by tools/gen_fn137b_far.py -- do not edit by hand, re-run the generator.\n",
                 "// GENERATED by tools/gen_fn151_nvfp4p1.py from moe_grouped_far.hip -- do not edit by hand.\n"
                 "// FN151: FN148's NVFP4 form (b) in the served `_p1` (PFD 1) class -- see the generator's docstring.\n"
                 "// The text up to `fn151_x_impl` is moe_grouped_far.hip's (helpers, verbatim; FN137b's header kept):\n"
                 "// (was) GENERATED by tools/gen_fn137b_far.py -- do not edit by hand, re-run the generator.\n")
    mid = ("\n// ---------------------------------------------------------------- FN151\n"
           "template <int V> struct fn151_ic { static constexpr int value = V; };\n"
           "template <int DBK, int DTM>\n__host__ __device__ constexpr int fn151_lds_bytes() {\n"
           "  return DEC_MTILE * DTM * (DBK + DEC_PAD) + DEC_DWN * 16 * (DBK + DEC_PAD) + 4 * DEC_MTILE * DTM + 16;\n}\n")
    w = ["\n// ---------------------------------------------------------------- FN151: the grouped cells\n"]
    for dbk, ks, tm, g, sc, sfx in CELLS:
        nm = f"fn151_w4a8b_bk{dbk}_ks{ks}_tm{tm}_g{g}{sfx}"
        w.append(f'''extern "C" __global__ __launch_bounds__(256) void {nm}(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,
    const int *__restrict__ meta, const float *__restrict__ As, float *__restrict__ P,
    int *__restrict__ cnt, __bf16 *__restrict__ C, int N, int K, int Pmax, int gsplit) {{
  __shared__ __attribute__((aligned(16))) unsigned char smem[fn151_lds_bytes<{dbk}, {tm}>()];
  fn151_x_impl<{sc}, DEC_DWN, {dbk}, {ks}, {tm}, true, {"true" if g else "false"}, true, true, 1>(
      A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, As, P, cnt, C, N, K, Pmax, gsplit, smem,
      (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)gridDim.x, mx_nowait{{}});
}}
''')
    return prefix + mid + impl + "".join(w)


def main():
    text = derive()
    if "--check" in sys.argv:
        ok = os.path.exists(OUT) and open(OUT).read() == text
        print(f"gen_fn151_nvfp4p1: {OUT} {'== the derivation' if ok else 'DIFFERS from the derivation'}")
        return 0 if ok else 1
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} B, {len(CELLS)} kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
