// Saber paged attention, CLEANROOM-2 (CR2-2c): the 27B ragged-row object (per-sequence query rows from cu_q, GQA 6).
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/saber_attn_ragged.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_combine.h"
#include "sattn_decode.h"

namespace {
using namespace sattn;
template <int NW, bool KV8>
using RaggedDecode = DecodeSpec<6, NW, KV8, KeyMap::Pages, RowSrc::Ragged, MaskBy::Causal, Tail::Partial>;
}  // namespace

extern "C" __global__ void __launch_bounds__(32)
saber_attn_decode_ragged_h256_gqa6_fp8kv_nw1(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<1, true>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
saber_attn_decode_ragged_h256_gqa6_fp8kv_nw2(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<2, true>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
saber_attn_decode_ragged_h256_gqa6_fp8kv_nw3(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<3, true>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
saber_attn_decode_ragged_h256_gqa6_fp8kv_nw4(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<4, true>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(32)
saber_attn_decode_ragged_h256_gqa6_bf16kv_nw1(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<1, false>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
saber_attn_decode_ragged_h256_gqa6_bf16kv_nw2(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<2, false>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
saber_attn_decode_ragged_h256_gqa6_bf16kv_nw3(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<3, false>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
saber_attn_decode_ragged_h256_gqa6_bf16kv_nw4(sattn::Params args, int splits, const int* cu_q) {
  decode_body<RaggedDecode<4, false>>(args, splits, cu_q, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(256)
saber_attn_splitkv_combine_ragged_h256_pf1(sattn::Params args, int splits, int tile, const int* cu_q) {
  combine_job<true, true>(args, splits, tile, cu_q);
}

extern "C" __global__ void __launch_bounds__(256)
saber_attn_splitkv_combine_ragged_h256_pf0(sattn::Params args, int splits, int tile, const int* cu_q) {
  combine_job<true, false>(args, splits, tile, cu_q);
}
