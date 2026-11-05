//! FN45a host simulation (NO GPU): replay FN19's real QSA selections through `saber_mem::kv_tier`
//! and print hit %, gather bytes per step and the VRAM hot pool per quota.
//!
//!   fn45_kv_sim [--block 16] [--window 1024] [--step 1] [--quotas 0,32,64,...] <idx_sel.npy>...
//!
//! Each `idx_sel.npy` is `[12, T - 2,051, 512]` i32 (row r = the query at position 2,051 + r,
//! groups ascending, -1 = padding). One prompt = one sequence: the first 2,051 tokens are
//! prefilled (64-token appends, so the plan's append headroom stays small), then every `--step`
//! rows are one decode step (append, gather the rows' union per layer, commit). The tier runs the
//! 12 target QSA layers (the MTP layer's selections are not captured: scale bytes by 13/12).
//! Lines: `sim <file> B W step quota | hit % | copied/step | uva/step | MB/step (12 L) | hot MiB`.

use std::collections::BTreeMap;

use saber_mem::kv_tier::{replay, KvTier, KvTierPlan, KV_TOKEN_LAYER_BYTES};

const P0: u32 = 2051;
const LAYERS: u32 = 12;
const APPEND: u32 = 64;

/// A C-order little-endian int32 `.npy` (format 1.0 / 2.0): shape and values.
fn read_npy_i32(path: &str) -> (Vec<usize>, Vec<i32>) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&raw[..6], b"\x93NUMPY", "{path}: not an npy");
    let (hlen, off) = if raw[6] == 1 {
        (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&raw[off..off + hlen]).unwrap();
    assert!(
        header.contains("'<i4'"),
        "{path}: dtype is not <i4: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "{path}: {header}"
    );
    let shape: Vec<usize> = {
        let s = &header[header.find("'shape': (").unwrap() + 10..];
        s[..s.find(')').unwrap()]
            .split(',')
            .filter(|x| !x.trim().is_empty())
            .map(|x| x.trim().parse().unwrap())
            .collect()
    };
    let data: Vec<i32> = raw[off + hlen..]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{path}: truncated"
    );
    (shape, data)
}

fn main() {
    let mut a = std::env::args().skip(1).peekable();
    let mut opts: BTreeMap<String, String> = BTreeMap::new();
    let mut files = Vec::new();
    while let Some(x) = a.next() {
        if let Some(k) = x.strip_prefix("--") {
            opts.insert(k.to_string(), a.next().expect("flag value"));
        } else {
            files.push(x);
        }
    }
    let get = |k: &str, d: &str| opts.get(k).cloned().unwrap_or_else(|| d.to_string());
    let block: u32 = get("block", "16").parse().unwrap();
    let window: u32 = get("window", "1024").parse().unwrap();
    let step: u32 = get("step", "1").parse().unwrap();
    let quotas: Vec<u32> = get("quotas", "0,16,32,48,64,96,128,192,256")
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let bb = u64::from(block) * KV_TOKEN_LAYER_BYTES;
    for f in &files {
        let (shape, data) = read_npy_i32(f);
        assert_eq!((shape[0], shape[2]), (12, 512), "{f}: {shape:?}");
        let rows = shape[1] as u32;
        let ctx = P0 + rows;
        let sel = |layer: u32, row: u32| -> &[i32] {
            let o = (layer as usize * shape[1] + row as usize) * 512;
            &data[o..o + 512]
        };
        for &q in &quotas {
            let wb = u64::from(window / block) + 1;
            let cb = u64::from(APPEND.div_ceil(block)) + 1;
            let floor = u64::from(LAYERS) * (wb + cb);
            let hot = (floor + u64::from(LAYERS) * u64::from(q)) * bb;
            let ram = u64::from(LAYERS) * u64::from(ctx.div_ceil(block)) * bb;
            let plan = KvTierPlan::new(LAYERS, block, window, APPEND, 1, ctx, hot, ram, 0).unwrap();
            assert_eq!(plan.quota, q);
            let mut t = KvTier::new(plan);
            let st = replay(&mut t, 0, P0, rows, step, LAYERS, sel).unwrap();
            t.check().unwrap();
            let mb = st.gather_bytes_per_step(bb) / 1e6;
            println!(
                "sim {f} B {block} W {window} step {step} quota {q} | hit {:.2} % | copied/step {:.2} | \
                 uva/step {:.2} | MB/step {:.3} | hot MiB {:.1} | steps {}",
                100.0 * st.hit(),
                st.copied as f64 / st.steps as f64,
                st.uva as f64 / st.steps as f64,
                mb,
                hot as f64 / f64::from(1 << 20),
                st.steps
            );
        }
    }
}
