// Saber paged attention, CLEANROOM-2 (CR2-2c): the Flash-Next QSA gather object, bf16 KV (keys from the union slot list, mask from its bits).
// Bodies: the parametrised templates of sattn_*.h (written from the CR2-2a specification). This file only
// names the instantiations the host plans launch; symbol names, argument names and launch bounds are fixed
// ABI (prebuilt/gfx1201/flashnext_qsa_attn.manifest.toml, crates/saber-kernels/rules/).
#include "sattn_decode.h"
#include "sattn_prefill.h"

namespace {
using namespace sattn;
using GatherPrefill = PrefillSpec<12, false, false, false, true, Skip::None, false>;
using GatherDecode = DecodeSpec<12, 4, false, KeyMap::List, RowSrc::Uniform, MaskBy::Bits, Tail::Partial>;
}  // namespace

extern "C" __global__ void __launch_bounds__(768)
fn_qsa_attn_prefill_h256_gqa12_bf16kv(sattn::Params args, sattn::UnionKeys qa) {
  prefill_body<GatherPrefill>(args, qa);
}

extern "C" __global__ void __launch_bounds__(128)
fn_qsa_attn_decode_h256_gqa12_bf16kv_nw4(sattn::Params args, int splits, sattn::UnionKeys qa) {
  decode_body<GatherDecode>(args, splits, nullptr, nullptr, qa, nullptr);
}
