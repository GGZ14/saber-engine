// Saber paged attention, CLEANROOM-2 (CR2-2c): the Flash-Next fp8-KV object (GQA 12): prefill and decode (merged by fn_attn_splitkv_combine_h256_pf1).
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/flashnext_attn_kv8.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_decode.h"
#include "sattn_prefill.h"

namespace {
using namespace sattn;
using Prefill = PrefillSpec<12, true, false, false, false, Skip::None, false>;
template <int NW>
using PairDecode = DecodeSpec<12, NW, true, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, Tail::Partial>;
}  // namespace

extern "C" __global__ void __launch_bounds__(768)
fn_attn_prefill_h256_gqa12_fp8kv(sattn::Params args) {
  prefill_body<Prefill>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(32)
fn_attn_decode_h256_gqa12_fp8kv_nw1(sattn::Params args, int splits) {
  decode_body<PairDecode<1>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
fn_attn_decode_h256_gqa12_fp8kv_nw2(sattn::Params args, int splits) {
  decode_body<PairDecode<2>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
fn_attn_decode_h256_gqa12_fp8kv_nw3(sattn::Params args, int splits) {
  decode_body<PairDecode<3>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
fn_attn_decode_h256_gqa12_fp8kv_nw4(sattn::Params args, int splits) {
  decode_body<PairDecode<4>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}
