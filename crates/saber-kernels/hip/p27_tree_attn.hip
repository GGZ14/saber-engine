// Saber paged attention, CLEANROOM-2 (CR2-2c): the 27B tree-verify decode: ragged rows, the step's causal tail replaced by a per-row ancestor word.
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/p27_tree_attn.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_decode.h"

namespace {
using namespace sattn;
template <int NW, bool KV8>
using TreeDecode = DecodeSpec<6, NW, KV8, KeyMap::Pages, RowSrc::Ragged, MaskBy::Ancestor, Tail::Partial>;
}  // namespace

extern "C" __global__ void __launch_bounds__(32)
p27_attn_decode_anc_h256_gqa6_fp8kv_nw1(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<1, true>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
p27_attn_decode_anc_h256_gqa6_fp8kv_nw2(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<2, true>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
p27_attn_decode_anc_h256_gqa6_fp8kv_nw3(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<3, true>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
p27_attn_decode_anc_h256_gqa6_fp8kv_nw4(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<4, true>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(32)
p27_attn_decode_anc_h256_gqa6_bf16kv_nw1(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<1, false>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
p27_attn_decode_anc_h256_gqa6_bf16kv_nw2(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<2, false>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
p27_attn_decode_anc_h256_gqa6_bf16kv_nw3(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<3, false>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
p27_attn_decode_anc_h256_gqa6_bf16kv_nw4(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<TreeDecode<4, false>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}
