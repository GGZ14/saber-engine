//! Packing a saber descriptor: the by-value struct, byte for byte, on the host.
//!
//! The two paged-attention kernels take `const SABERArgs a` **by value**. The code object reports one
//! 136-byte `by_value` argument; the host has to write nine pointers and eleven scalars into it.
//! Those are two views of the same bytes, and "a kernarg layout the host and the kernel disagree
//! about is memory corruption, not an error" (ARCHITECTURE.md, *Kernel plane*).
//!
//! So this test builds the image twice, from two independent derivations, and requires them to be
//! identical:
//!
//! 1. **The engine's path** — `saber::resolve` (the same function `saber-coreplan` calls, so this is
//!    that binary's output without the JSON round trip) into [`saber_exec::pack_kernarg`], which is
//!    what `Launcher::launch` hands to `hipModuleLaunchKernel`.
//! 2. **The ABI's path** — `saber_kernels::abi::pack_fields_at(SABER_ARGS_FIELDS, 0)` for the field
//!    offsets and `abi::pack(SIG_SABER_ATTN_DECODE)` for the struct's own slot, with every value
//!    written by hand here.
//!
//! If either derivation drifts, the two images stop matching. A test that only checked "the
//! descriptor packs without error" would have caught nothing.

use std::collections::BTreeMap;

use saber_exec::{pack_kernarg, ArgBindings};
use saber_hip::DeviceAddr;
use saber_kernels::abi;
use saber_kernels::rules::Shape as GemmShape;
use saber_kernels::saber::{resolve, AttnShape, KvDtype, Op, SaberRules, Shape};

/// Distinct, recognisable addresses: a wrong field lands as a visibly wrong pointer rather than as
/// a plausible one. Not real device memory — nothing is launched here.
const Q: u64 = 0x1000_0000;
const KV: u64 = 0x2000_0000;
const BLOCK_TABLE: u64 = 0x3000_0000;
const SEQUSED_K: u64 = 0x4000_0000;
const OUT: u64 = 0x5000_0000;
const SCRATCH: u64 = 0x6000_0000;

fn shape() -> AttnShape {
    // One decode step at prod's TP=2 attention geometry: 12 q heads / 2 kv heads (GQA 6),
    // head_dim 256, paged block 16, a 5-token speculative window.
    AttnShape {
        num_seqs: 2,
        q_len: 5,
        live_rows: 5,
        q_heads: 12,
        kv_heads: 2,
        head_dim: 256,
        block_size: 16,
        max_blocks: 2048,
        max_ctx: 8192,
        splits: 0,
        kv_dtype: KvDtype::Fp8E4m3,
        kv_block_stride: 8192,
        kv_head_stride: 4096,
        scale_bits: (0.0625f32).to_bits(),
        has_descales: false,
        ragged_rows: 0,
    }
}

fn bindings() -> (ArgBindings, BTreeMap<String, DeviceAddr>) {
    // `ArgBindings` carries a GEMM shape for the Persistent workspace key; the attention path has
    // no Persistent workspace, so it is only along for the ride here.
    let mut b = ArgBindings::new(GemmShape::new(10, 0, 0));
    for (name, addr) in [
        ("q", Q),
        ("kv", KV),
        ("block_table", BLOCK_TABLE),
        ("seqused_k", SEQUSED_K),
        ("out", OUT),
    ] {
        b.bind(name, addr_of(addr));
    }
    let mut ws = BTreeMap::new();
    ws.insert("attn_decode_scratch".to_string(), addr_of(SCRATCH));
    (b, ws)
}

/// A `DeviceAddr` cannot be forged from an integer — that is the point of the type. The only
/// constructors outside `saber-hip` are `NULL` and `offset`, so a test address is built from NULL.
fn addr_of(v: u64) -> DeviceAddr {
    DeviceAddr::NULL.offset(v).expect("no overflow")
}

#[test]
fn the_decode_descriptor_packs_byte_identically_to_the_abi_field_walk() {
    let rules = SaberRules::prod_saber().expect("the prod saber rules parse");
    let s = shape();
    let desc = resolve(Op::AttnDecode, Shape::Attn(s), &rules).expect("attn_decode resolves");
    assert!(
        desc.kernel
            .starts_with("saber_attn_decode_h256_gqa6_fp8kv_nw"),
        "{}",
        desc.kernel
    );
    // The split-KV combine is a second dispatch, declared here and asserted at capture.
    assert_eq!(desc.n_launches, 2);

    let (args, ws) = bindings();
    let got = pack_kernarg(&desc, &args, &ws, None).expect("the descriptor packs");

    // ---- the independent derivation -------------------------------------------------------
    let (fields, fields_end) = abi::pack_fields_at(abi::SABER_ARGS_FIELDS, 0);
    let (slots, size) = abi::pack(abi::SIG_SABER_ATTN_DECODE);
    assert_eq!(size, desc.kernarg_bytes, "segment size");
    assert_eq!(
        slots[0],
        (0, abi::SABER_ARGS_BYTES),
        "the struct's own slot"
    );

    let splits = splits_for(&desc);
    let mut want = vec![0u8; size as usize];
    let mut put = |index: usize, bytes: &[u8]| {
        let (off, sz) = fields[index];
        assert_eq!(sz as usize, bytes.len(), "field {index} size");
        want[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
    };
    put(0, &Q.to_le_bytes()); // q
    put(1, &KV.to_le_bytes()); // kv
    put(2, &BLOCK_TABLE.to_le_bytes()); // block_table
    put(3, &SEQUSED_K.to_le_bytes()); // seqused_k
    put(4, &OUT.to_le_bytes()); // out
    put(5, &0u64.to_le_bytes()); // k_descale: folded away, so a null the kernel reads as "none"
    put(6, &0u64.to_le_bytes()); // v_descale
    put(7, &0u64.to_le_bytes()); // q_descale: documented unused, the query is always bf16
    put(8, &SCRATCH.to_le_bytes()); // scratch: the split-KV partials workspace
    put(9, &s.num_seqs.to_le_bytes());
    put(10, &s.q_len.to_le_bytes());
    put(11, &s.q_heads.to_le_bytes());
    put(12, &s.kv_heads.to_le_bytes());
    put(13, &s.head_dim.to_le_bytes());
    put(14, &s.block_size.to_le_bytes());
    put(15, &s.max_blocks.to_le_bytes());
    put(16, &s.kv_block_stride.to_le_bytes());
    put(17, &s.kv_head_stride.to_le_bytes());
    put(18, &s.scale_bits.to_le_bytes());
    // The struct's OWN `splits` is the caller's request (0 = "use the split law"); the second
    // `splits`, outside the struct, is what the law answered.
    put(19, &s.splits.to_le_bytes());
    put(20, &s.max_ctx.to_le_bytes());
    let (off, sz) = slots[1];
    assert_eq!(sz, 4);
    want[off as usize..off as usize + 4].copy_from_slice(&splits.to_le_bytes());

    assert_eq!(
        got, want,
        "the engine's kernarg image and the ABI's field walk disagree"
    );

    // The 4 bytes between the last field and the end of the struct are its own tail padding: real
    // memory the host must leave alone, and zero rather than whatever the staging buffer held.
    assert_eq!(fields_end, 132);
    assert_eq!(&got[132..136], &[0u8; 4], "SABERArgs tail padding");
}

#[test]
fn the_prefill_descriptor_writes_the_struct_and_nothing_after_it() {
    let rules = SaberRules::prod_saber().unwrap();
    let mut s = shape();
    s.q_len = 512; // above MAX_DECODE_QLEN = 10, so this is the prefill kernel
    s.live_rows = 512; // a chunk that FILLS its bucket
    let desc = resolve(Op::AttnPrefill { fp8_mode: 3 }, Shape::Attn(s), &rules)
        .expect("attn_prefill resolves");
    assert_eq!(desc.kernarg_bytes, abi::SABER_ARGS_BYTES);
    assert_eq!(desc.n_launches, 1);

    let (args, ws) = bindings();
    let got = pack_kernarg(&desc, &args, &ws, None).unwrap();
    assert_eq!(got.len(), 136);
    // Prefill takes no split-KV scratch, so the ninth pointer is a null the kernel reads as
    // "no partials".
    let (fields, _) = abi::pack_fields_at(abi::SABER_ARGS_FIELDS, 0);
    let (off, _) = fields[8];
    assert_eq!(&got[off as usize..off as usize + 8], &[0u8; 8]);

    // ws-h27, AT THE KERNARG IMAGE: a PADDED chunk writes the CHUNK's row count into field 10,
    // which is the byte `live = qpos < q_len` and `klimit = ctx - q_len + qpos` are read from.
    // This is the one assertion that is about the bytes the card sees rather than about a field
    // on the way to them.
    let (q_off, q_sz) = fields[10];
    assert_eq!(q_sz, 4);
    assert_eq!(
        &got[q_off as usize..q_off as usize + 4],
        &512u32.to_le_bytes()
    );
    let mut padded = s;
    padded.live_rows = 234;
    let pdesc = resolve(Op::AttnPrefill { fp8_mode: 3 }, Shape::Attn(padded), &rules)
        .expect("a padded chunk resolves");
    let pgot = pack_kernarg(&pdesc, &args, &ws, None).unwrap();
    assert_eq!(
        &pgot[q_off as usize..q_off as usize + 4],
        &234u32.to_le_bytes()
    );
    // ...and NOTHING else in the image moves with it: the padding is a launch-geometry fact, not
    // an addressing one.
    let mut without_q = (got.clone(), pgot.clone());
    for b in q_off as usize..q_off as usize + 4 {
        without_q.0[b] = 0;
        without_q.1[b] = 0;
    }
    assert_eq!(without_q.0, without_q.1);
}

#[test]
fn a_thirty_six_argument_scalar_descriptor_still_packs_the_ordinary_way() {
    // The fused GDN update is the opposite shape of problem: no struct, 36 separate arguments,
    // 256 bytes. It shares `pack_kernarg` with the attention path, so it is the control that the
    // struct support did not change the scalar case.
    let rules = SaberRules::prod_saber().unwrap();
    let g = saber_kernels::saber::GdnShape {
        n_seqs: 1,
        h: 24,
        hg: 8,
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        maxq: 5,
        o_rows: 8,
        xpitch: 5120,
        cs_seq: 512,
        cs_dim: 4,
        cs_tok: 1,
        ci_stride: 1,
        ab_stride: 24,
        st_slot: 393_216,
        st_head: 16_384,
        sidx_stride: 1,
        scale_bits: (0.088_388_35f32).to_bits(),
        softplus_thr_bits: (20.0f32).to_bits(),
        has_naccept: true,
        has_bias: true,
        state_f16: false,
    };
    let desc = resolve(Op::GdnFusedUpdate { ab_bf16: true }, Shape::Gdn(g), &rules)
        .expect("gdn_fused_update resolves");
    assert_eq!(desc.kernarg_bytes, 256);
    assert_eq!(desc.n_launches, 1);
    assert_eq!(desc.args.len(), 36);

    let mut b = ArgBindings::new(GemmShape::new(1, 0, 0));
    let mut next = 0x1000u64;
    for slot in &desc.args {
        if let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value {
            next += 0x1000;
            b.bind(buffer, addr_of(next));
        }
    }
    let mut ws = BTreeMap::new();
    for w in &desc.workspaces {
        ws.insert(w.name.clone(), addr_of(0xdead_0000));
    }
    let image = pack_kernarg(&desc, &b, &ws, None).expect("the fused update packs");
    assert_eq!(image.len(), 256);

    // Every declared slot lands where `abi::pack` says, and the bytes between slots are padding
    // the host leaves zero.
    let (slots, size) = abi::pack(abi::SIG_SABER_GDN_FUSED_UPDATE);
    assert_eq!(size, 256);
    for (i, slot) in desc.args.iter().enumerate() {
        assert_eq!(
            (slot.offset, slot.size),
            slots[i],
            "slot {i} `{}`",
            slot.name
        );
    }
    // `state_len_max` at offset 64 is followed by 4 bytes of padding before the 8-aligned
    // `cache_idx` pointer at 72.
    assert_eq!(&image[68..72], &[0u8; 4], "alignment padding stays zero");
}

/// The split count the descriptor's second `splits` argument carries, read back out of it.
fn splits_for(desc: &saber_kernels::LaunchDescriptor) -> i32 {
    let slot = desc
        .args
        .iter()
        .rev()
        .find(|s| s.name == "splits")
        .expect("the decode descriptor carries `splits` outside the struct");
    match slot.value {
        saber_kernels::rules::ArgValue::I32 { ival } => i32::try_from(ival).unwrap(),
        ref other => panic!("`splits` is {other:?}"),
    }
}
