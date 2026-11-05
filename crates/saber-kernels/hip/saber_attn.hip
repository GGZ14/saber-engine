// Saber paged attention, CLEANROOM-2 (CR2-2c): the 27B uniform-row object (GQA 6): prefill legs, decode, last-arriver decode, combines.
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/saber_attn.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_combine.h"
#include "sattn_decode.h"
#include "sattn_prefill.h"

namespace {
using namespace sattn;
// prefill over fp8 pages: QK8 / PV8 legs, V prefetch on or off (measured per leg)
template <bool QK8, bool PV8, bool AHEAD>
using Fp8Prefill = PrefillSpec<6, true, QK8, PV8, false, Skip::None, AHEAD>;
using Bf16Prefill = PrefillSpec<6, false, false, false, false, Skip::None, false>;
template <int NW, bool KV8>
using PairDecode = DecodeSpec<6, NW, KV8, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, Tail::Partial>;
template <int NW, Tail TL>
using MergeDecode = DecodeSpec<6, NW, true, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, TL>;
}  // namespace

extern "C" __global__ void __launch_bounds__(768)
saber_attn_prefill_h256_gqa6_fp8kv(sattn::Params args) {
  prefill_body<Fp8Prefill<false, false, false>>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(768)
saber_attn_prefill_h256_gqa6_fp8kv_qk8(sattn::Params args) {
  prefill_body<Fp8Prefill<true, false, false>>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(768)
saber_attn_prefill_h256_gqa6_fp8kv_pv8(sattn::Params args) {
  prefill_body<Fp8Prefill<false, true, false>>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(768)
saber_attn_prefill_h256_gqa6_fp8kv_qk8pv8(sattn::Params args) {
  prefill_body<Fp8Prefill<true, true, true>>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(768)
saber_attn_prefill_h256_gqa6_bf16kv(sattn::Params args) {
  prefill_body<Bf16Prefill>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(32)
saber_attn_decode_h256_gqa6_fp8kv_nw1(sattn::Params args, int splits) {
  decode_body<PairDecode<1, true>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
saber_attn_decode_h256_gqa6_fp8kv_nw2(sattn::Params args, int splits) {
  decode_body<PairDecode<2, true>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
saber_attn_decode_h256_gqa6_fp8kv_nw3(sattn::Params args, int splits) {
  decode_body<PairDecode<3, true>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
saber_attn_decode_h256_gqa6_fp8kv_nw4(sattn::Params args, int splits) {
  decode_body<PairDecode<4, true>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(32)
saber_attn_decode_h256_gqa6_bf16kv_nw1(sattn::Params args, int splits) {
  decode_body<PairDecode<1, false>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
saber_attn_decode_h256_gqa6_bf16kv_nw2(sattn::Params args, int splits) {
  decode_body<PairDecode<2, false>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
saber_attn_decode_h256_gqa6_bf16kv_nw3(sattn::Params args, int splits) {
  decode_body<PairDecode<3, false>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
saber_attn_decode_h256_gqa6_bf16kv_nw4(sattn::Params args, int splits) {
  decode_body<PairDecode<4, false>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(32)
saber_attn_declc_h256_gqa6_fp8kv_nw1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<1, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(32)
saber_attn_declc_h256_gqa6_fp8kv_nw1_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<1, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(64)
saber_attn_declc_h256_gqa6_fp8kv_nw2(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<2, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(64)
saber_attn_declc_h256_gqa6_fp8kv_nw2_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<2, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(96)
saber_attn_declc_h256_gqa6_fp8kv_nw3(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<3, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(96)
saber_attn_declc_h256_gqa6_fp8kv_nw3_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<3, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(128)
saber_attn_declc_h256_gqa6_fp8kv_nw4(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<4, Tail::Merge>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(128)
saber_attn_declc_h256_gqa6_fp8kv_nw4_drop1(sattn::Params args, int splits, unsigned* attn_decode_counter) {
  decode_body<MergeDecode<4, Tail::MergeDrop>>(args, splits, nullptr, nullptr, UnionKeys{}, attn_decode_counter);
}

extern "C" __global__ void __launch_bounds__(256)
saber_attn_splitkv_combine_h256_pf1(sattn::Params args, int splits, int tile) {
  combine_job<false, true>(args, splits, tile, nullptr);
}

extern "C" __global__ void __launch_bounds__(256)
saber_attn_splitkv_combine_h256_pf0(sattn::Params args, int splits, int tile) {
  combine_job<false, false>(args, splits, tile, nullptr);
}
