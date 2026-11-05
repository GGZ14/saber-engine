// Flash-Next (FN119d, `--fn-device-step next`): the NEXT verify's input embedding rows gathered on
// the card while the drafter round that drafts them is still in flight. Its own code object
// (`flashnext_ahead` / `flashnext_ahead2` keep their bytes).
//
//   fn119d_vemb  grid (bucket x w), block 256: verify row r = s x w + t of the next step feeds
//                t == 0: the walk's last committed token of sequence s (the result block:
//                        out[s * 16 + 3 + acc_s], the token the host walk checked equal),
//                t >= 1: the chain's draft t of sequence s (ids[(t - 1) * ids_stride + s]),
//                a pad sequence (s >= nseq): token 0 (the host's law);
//                -> the payload row: the embedding row from this rank's half (UVA, `embh`) or
//                -0.0 (the other rank's id); the certified EXACT reduce then sums the pair
//                (x + -0.0 == x: FN22's law) == the host gather's row, byte for byte.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define FN119D_H 2560

extern "C" __global__ __launch_bounds__(256) void fn119d_vemb(
    const uint32_t* __restrict__ out, const uint32_t* __restrict__ ids,
    const uint16_t* __restrict__ embh, uint16_t* __restrict__ pay, int w, int nseq, int half,
    int rank, int ids_stride) {
  const int r = blockIdx.x, s = r / w, t = r % w;
  uint32_t tok = 0u;
  if (s < nseq) {
    if (t == 0) {
      const uint32_t* o = out + (size_t)s * 16;
      tok = o[3 + o[1]];
    } else {
      tok = ids[(size_t)(t - 1) * (size_t)ids_stride + (size_t)s];
    }
  }
  const uint32_t base = (uint32_t)rank * (uint32_t)half;
  const bool mine = tok >= base && tok < base + (uint32_t)half;
  uint4* dst = (uint4*)(pay + (size_t)r * FN119D_H);
  if (mine) {
    const uint4* src = (const uint4*)(embh + (size_t)(tok - base) * FN119D_H);
    for (int i = threadIdx.x; i < FN119D_H / 8; i += 256) dst[i] = src[i];
  } else {
    for (int i = threadIdx.x; i < FN119D_H / 8; i += 256)
      dst[i] = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  }
}
