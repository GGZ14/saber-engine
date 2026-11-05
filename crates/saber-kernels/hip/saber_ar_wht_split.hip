// Saber two-rank rotated all-reduce, split halves (CLEANROOM-2, CR2-4c): the instantiation table of code object
// `saber_ar_wht_split` (`--engine-form split`, an arm). Bodies: sar_wpar.h. The push halves take flag_val (the
// host's monotone split value); `_push_nb` runs on P <= nbv blocks; the reduce runs at the one-shot's grid.
#include "sar_wpar.h"

extern "C" __global__ void __launch_bounds__(1024)
    saber_ar_split_2rank_wht6_bf16_push(SAR_ROT_PARAMS, unsigned flag_val) {
  (void)out; (void)my_flags; (void)acq;
  sar::split_push(in, peer_scratch, my_scratch, loc_pack, peer_flags, seq_ctrs, n_groups, n_chunks,
                  slot_stride_bytes, scale_off_bytes, drain, flag_val);
}

extern "C" __global__ void __launch_bounds__(1024)
    saber_ar_split_2rank_wht6_bf16_push_nb(SAR_ROT_PARAMS, unsigned flag_val, int nbv) {
  (void)out; (void)my_flags; (void)acq;
  sar::split_push_nb(in, peer_scratch, my_scratch, loc_pack, peer_flags, seq_ctrs, n_groups, n_chunks,
                     slot_stride_bytes, scale_off_bytes, drain, flag_val, nbv);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_split_2rank_wht6_bf16_reduce(SAR_ROT_PARAMS) {
  (void)in; (void)peer_scratch; (void)peer_flags; (void)my_flags; (void)drain;
  sar::split_reduce(my_scratch, loc_pack, out, seq_ctrs, n_groups, n_chunks, slot_stride_bytes, scale_off_bytes,
                    acq);
}
