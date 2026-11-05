// Saber source (CMP27T, 2026-10-28): the TP > 1 local argmax that ALSO writes the winning
// logit's bf16 bits, so the host's pick read is ONE table copy instead of the table + a batch of
// `rows` two-byte reads at the columns the table names (`read_picks` -> `download_scattered`,
// ~3.5 us a row: LEDGER CMP27Q's picks term, 0.10 / 0.17 / 0.26 ms a b1 / conc-4 / conc-8 step).
//
// The index is `saber_argmax_bf16`'s (head.hip), operation for operation: the same strided
// per-thread scan, the same (value descending, index ascending) total order, the same LDS tree,
// `cols` when every logit was NaN. The VALUE word is the logit the winner won with, read back
// from the row at the winning column after the reduction -- the very two bytes the host's
// scattered read fetched -- zero-extended to 32 bits (0 when nothing won).
//
// Layout of `out` (2 x gridDim.x ints): `out[row]` = the index (the shipped table, unchanged),
// `out[gridDim.x + row]` = the value word. The host copies both halves in one read.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define ARGMAX_PV_T 256
extern "C" __global__ void saber_argmax_pv_bf16(const __bf16 *__restrict__ logits,
                                                int *__restrict__ out, int cols, long stride) {
  __shared__ float sv[ARGMAX_PV_T];
  __shared__ int si[ARGMAX_PV_T];
  const int row = blockIdx.x;
  const __bf16 *p = logits + (long)row * stride;
  const int tid = threadIdx.x;

  float best = -__builtin_inff();
  int bi = cols;  // "nothing won yet"
  for (int j = tid; j < cols; j += ARGMAX_PV_T) {
    const float v = (float)p[j];
    if (v > best || (v == best && j < bi)) {
      best = v;
      bi = j;
    }
  }
  sv[tid] = best;
  si[tid] = bi;
  __syncthreads();
  for (int s = ARGMAX_PV_T / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const float ov = sv[tid + s];
      const int oi = si[tid + s];
      if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) {
        sv[tid] = ov;
        si[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int w = si[0];
    out[row] = w;
    unsigned bits = 0u;
    if (w >= 0 && w < cols) bits = (unsigned)((const unsigned short *)p)[w];
    out[gridDim.x + row] = (int)bits;
  }
}
