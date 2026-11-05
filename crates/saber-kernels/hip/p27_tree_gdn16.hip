// p27_tree_gdn16: the 16-row GDN tree objects (named by the `--engine-spec-tree best16` arm).
//
// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the WY tree verify at up to 16 rows (the `best16` tree arm)
// and its standalone advance (spec 2.9; the 10-row tree with TM = 16). Body in sgdn_tree.h; this file is the
// instantiation table.
#define SGDN_TREE_ROWS16
#include "p27_tree_gdn.hip"

SGDN_TREE(16, p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs1_adv0, 1, 0, 0)
SGDN_TREE(16, p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs1_adv1, 1, 0, 1)
SGDN_TREE_ADVANCE(16, p27_gdn_tree16_advance_f16s_k128_v128)
