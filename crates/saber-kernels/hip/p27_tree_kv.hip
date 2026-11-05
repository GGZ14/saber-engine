// P27-15b item (b): the KV COMPACTION of a tree verify's accepted path (Saber's own HIP).
//
// A tree verify writes every row's K/V at the slot of its ROW (anchor + r) with rope by DEPTH
// (`saber_engine::spec_tree::TreePlan::positions`), so after the verdict the accepted path's rows
// must sit at the slots of their DEPTHS (anchor + d) before the next step reads them as context.
// `TreeVerdict::kv_moves` lists `(from_row, to_row)` with `to_row < from_row`; applied IN ORDER no
// copy reads a slot an earlier one wrote, but a later copy's destination may be an earlier copy's
// SOURCE (path rows 0,1,3,4 -> moves 3->2, then 4->3 overwrites slot 3 after it was read) -- so
// the copies are not independent: one workgroup per (layer, kv head) walks the moves in order
// with a barrier between them. Pure byte movement: the token row of a head is `row_bytes`
// contiguous bytes (K then V, the reference's paged layout [block][kv head][block_size][2 x head_dim]).
//
//   grid (layers, kv_heads), block 64. `moves` [n][2] = absolute pool token slots (src, dst),
//   i.e. block_table[pos / bs] * bs + pos % bs, resolved on the host per sequence.
#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void p27_tree_kv_compact(
    const unsigned long long* __restrict__ layer_base, const int* __restrict__ moves, int n_moves,
    long block_stride, long head_stride, int row_bytes, int block_size) {
  const int layer = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
  unsigned char* base = (unsigned char*)layer_base[layer] + (size_t)head * head_stride;
  for (int m = 0; m < n_moves; ++m) {
    const int s = moves[2 * m], d = moves[2 * m + 1];
    const unsigned char* src =
        base + (size_t)(s / block_size) * block_stride + (size_t)(s % block_size) * row_bytes;
    unsigned char* dst =
        base + (size_t)(d / block_size) * block_stride + (size_t)(d % block_size) * row_bytes;
    // 16-byte lanes; row_bytes is 512 (fp8) or 1024 (bf16) at head_dim 256
    for (int i = tid * 16; i < row_bytes; i += 64 * 16)
      *(uint4*)(dst + i) = *(const uint4*)(src + i);
    __syncthreads();  // the next move may read this move's destination's slot
  }
}
