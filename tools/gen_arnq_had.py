#!/usr/bin/env python3
"""MACaQ: derive crates/saber-kernels/hip/mxfp4_gemm_decode_arnq_had.hip from the shipped
mxfp4_gemm_decode_arnq.hip (row 3: decode GEMM + exact reduce + `_nq` epilogue): the epilogue's normed value
goes through `macaq_had32` before its amax, `had_t` is appended LAST to the impl and every symbol, and every
symbol is renamed `mxfp4_gemm_decode_arnq_had32_*`. `--check` fails if the committed file differs.
    gen_arnq_had.py [--check]"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'crates/saber-kernels/hip/mxfp4_gemm_decode_arnq.hip')
DST = os.path.join(ROOT, 'crates/saber-kernels/hip/mxfp4_gemm_decode_arnq_had.hip')


def sub1(s, a, b):
    n = s.count(a)
    if n != 1:
        raise SystemExit(f'gen_arnq_had: {n} matches for {a[:70]!r}')
    return s.replace(a, b)


def derive(s):
    hdr = ('// Saber source (2026-11-11): DERIVED by tools/gen_arnq_had.py from\n'
           '// mxfp4_gemm_decode_arnq.hip -- the `--macaq-had on` twin of row 3 (the `_nq` epilogue\'s normed value\n'
           '// through macaq_had32 before its amax; `had_t` appended last). Re-run the script after any parent edit.\n')
    s = hdr + s
    inc = s.index('#include <hip/hip_runtime.h>')
    eol = s.index('\n', inc)
    s = s[:eol + 1] + '#include "macaq_had.h"\n' + s[eol + 1:]
    s = sub1(s, 'int k16, int slot_stride16, float eps, int drain, int acq) {\n#pragma clang fp contract(off)',
             'int k16, int slot_stride16, float eps, int drain, int acq, const float* __restrict__ had_t) {\n'
             '#pragma clang fp contract(off)')
    s = sub1(s, '''      float nv = (float)(__hip_bfloat16)(sv[mg][j] * inv * ((float)wh[j] + 1.f));
      sv[mg][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
''', '''      float nv = (float)(__hip_bfloat16)(sv[mg][j] * inv * ((float)wh[j] + 1.f));
      sv[mg][j] = nv;
    }
    macaq_had32(sv[mg], had_t + 8 * (k - start), tid);
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(sv[mg][j]));
''')
    s = sub1(s, '    int acq, int ep_rows) {\n  constexpr int BND',
             '    int acq, int ep_rows, const float *__restrict__ had_t) {\n  constexpr int BND')
    s = sub1(s, 'peer_flags, my_flags, seq_ctrs, k16, slot_stride16, eps, drain, acq);\n}',
             'peer_flags, my_flags, seq_ctrs, k16, slot_stride16, eps, drain, acq, had_t);\n}')
    n = s.count('void mxfp4_gemm_decode_arnq_bk')
    s = s.replace('void mxfp4_gemm_decode_arnq_bk', 'void mxfp4_gemm_decode_arnq_had32_bk')
    s = s.replace('    int slot_stride16, float eps, int drain, int acq, int ep_rows) {\n  saber_',
                  '    int slot_stride16, float eps, int drain, int acq, int ep_rows, '
                  'const float *__restrict__ had_t) {\n  saber_')
    k = s.count('      ep_rows);\n}')
    s = s.replace('      ep_rows);\n}', '      ep_rows, had_t);\n}')
    if not (n == k == 3 and s.count('const float *__restrict__ had_t) {\n  saber_') == 3):
        raise SystemExit(f'gen_arnq_had: {n} symbols, {k} calls')
    return s


def main():
    out = derive(open(SRC).read())
    if '--check' in sys.argv:
        if open(DST).read() != out:
            raise SystemExit('gen_arnq_had --check: mxfp4_gemm_decode_arnq_had.hip differs from its derivation')
        print('gen_arnq_had --check: ok')
        return
    open(DST, 'w').write(out)
    print('wrote', DST)


if __name__ == '__main__':
    main()
