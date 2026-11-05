// P27-W2-4 (R2): the tree verify's CONV WINDOW put back in the chain's place (Saber's own HIP).
//
// After a tree verify the conv cache holds, per channel, `[w1, w2, x_0 .. x_{rows-1}]` (entry
// `2 + r` = verify row r's input; `p27_gdn_conv_tree`'s rewrite is the chain update's), and the
// accepted path's window is the last three inputs ALONG THE PATH. The chain's conv update reads its
// window at entries `naccept - 1 ..` = `depth .. depth + 2`, so a branched path's rows must move:
// entry `2 + path[k]` -> entry `2 + k` for the last three depths k with `path[k] != k`. Every source
// is at or above its destination and above every earlier destination (`path[k] >= k`, increasing),
// so the moves applied IN ORDER by one thread a channel never read an entry already written.
// Pure 16-bit moves: no arithmetic.
//
//   grid (GDN layers, ceil(dims / 256)), block 256. `layer_base` [layers] u64 conv pool bases;
//   `moves` [n][2] i32 entry indices (src, dst); the sequence's slot is `seq_off` elements in.
#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void p27_tree_conv_commit(
    const unsigned long long* __restrict__ layer_base, const int* __restrict__ moves, int n_moves,
    long seq_off, long cs_dim, long cs_tok, int dims) {
  const int d = blockIdx.y * 256 + threadIdx.x;
  if (d >= dims) return;
  unsigned short* p =
      (unsigned short*)layer_base[blockIdx.x] + seq_off + (size_t)d * cs_dim;
  for (int m = 0; m < n_moves; ++m) {
    const int s = moves[2 * m], t = moves[2 * m + 1];
    p[(size_t)t * cs_tok] = p[(size_t)s * cs_tok];
  }
}
