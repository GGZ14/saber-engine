// Saber paged attention, CLEANROOM-2 (CR2-2c): the Flash-Next bf16-KV object (GQA 12): prefill, decode, last-arriver decode, combine.
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/flashnext_attn.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_combine.h"
#include "sattn_decode.h"
#include "sattn_prefill.h"

namespace {
using namespace sattn;
using Prefill = PrefillSpec<12, false, false, false, false, Skip::None, false>;
template <int NW>
using PairDecode = DecodeSpec<12, NW, false, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, Tail::Partial>;
template <int NW, Tail TL>
using MergeDecode = DecodeSpec<12, NW, false, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, TL>;
}  // namespace

extern "C" __global__ void __launch_bounds__(768)
fn_attn_prefill_h256_gqa12_bf16kv(sattn::Params args) {
  prefill_body<Prefill>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(32)
fn_attn_decode_h256_gqa12_bf16kv_nw1(sattn::Params args, int splits) {
  decode_body<PairDecode<1>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
fn_attn_decode_h256_gqa12_bf16kv_nw2(sattn::Params args, int splits) {
  decode_body<PairDecode<2>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
fn_attn_decode_h256_gqa12_bf16kv_nw3(sattn::Params args, int splits) {
  decode_body<PairDecode<3>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
fn_attn_decode_h256_gqa12_bf16kv_nw4(sattn::Params args, int splits) {
  decode_body<PairDecode<4>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(32)
fn_attn_declc_h256_gqa12_bf16kv_nw1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<1, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(32)
fn_attn_declc_h256_gqa12_bf16kv_nw1_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<1, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(64)
fn_attn_declc_h256_gqa12_bf16kv_nw2(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<2, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(64)
fn_attn_declc_h256_gqa12_bf16kv_nw2_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<2, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(96)
fn_attn_declc_h256_gqa12_bf16kv_nw3(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<3, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(96)
fn_attn_declc_h256_gqa12_bf16kv_nw3_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<3, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(128)
fn_attn_declc_h256_gqa12_bf16kv_nw4(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<4, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(128)
fn_attn_declc_h256_gqa12_bf16kv_nw4_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<4, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(256)
fn_attn_splitkv_combine_h256_pf1(sattn::Params args, int splits, int tile) {
  combine_job<false, true>(args, splits, tile, nullptr);
}
