//! FN144's CSF census (the Rust side of the byte law): decode every routed-expert scale plane of
//! the FP4-Local checkpoint present on disk and write one TSV row per plane.
//!
//!     saber-fp4local-csf --checkpoint /mnt/nvme/models/Qwen3.8-Flash-Next-FP4-Local --out RUST.tsv
//!     saber-fp4local-csf --plain /mnt/nvme/models/Qwen3.8-Flash-Next-NVFP4 --out PLAIN.tsv
//!
//! Columns: `name rows cols exceptions redundant repeated base_max sha_exact sha_nibble
//! sha_wrong` -- `sha_nibble` decodes with the codes read high nibble first and `sha_wrong` with
//! the NEIGHBOUR expert's exception list (expert `(e + 1) % 512`, same layer and projection): the
//! two negative controls, which must differ from `sha_exact` on every row.
//! `tools/pack/fp4local_csf.py census` writes the same columns independently (numpy) and
//! `fp4local_csf.py compare` holds the two files equal.
//!
//! `--plain DIR` is the round-trip law over a checkpoint with PLAIN E4M3 scales (the NVFP4 PTQ
//! shard on disk): every routed-expert `weight_scale` plane present is encoded with
//! `nvfp4_csf::encode` and decoded back; a row is written per plane with its sha256 before and after.
//!
//! A shard not complete on disk (a running download) is skipped and counted; never guessed.

use std::io::Write;
use std::process::ExitCode;

use saber_model::qwen4_exp::nvfp4_csf::{decode_variant, encode, Variant};
use saber_model::qwen4_exp::st::ShardSet;
use sha2::{Digest, Sha256};

const LAYERS: usize = 48;
const EXPERTS: usize = 512;
const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn prefix(l: usize, e: usize, p: &str) -> String {
    format!("model.language_model.layers.{l}.mlp.experts.{e}.{p}")
}

fn csf(dir: &str, out: &mut dyn Write, layers: (usize, usize)) -> Result<(), String> {
    let mut s = ShardSet::open(std::path::Path::new(dir))?;
    let (mut done, mut skipped) = (0usize, 0usize);
    let mut exc = [0u64; 3];
    let mut scales = [0u64; 3];
    let mut base_max = 0u8;
    for l in layers.0..layers.1 {
        for e in 0..EXPERTS {
            for (pi, p) in PROJ.iter().enumerate() {
                let x = prefix(l, e, p);
                let nb = prefix(l, (e + 1) % EXPERTS, p);
                let fname = format!("{x}.weight_scale.nvfp4_csf_fixed");
                let ename = format!("{x}.weight_scale.nvfp4_csf_exceptions");
                let nname = format!("{nb}.weight_scale.nvfp4_csf_exceptions");
                let wname = format!("{x}.weight");
                if !(s.present(&fname)
                    && s.present(&ename)
                    && s.present(&nname)
                    && s.present(&wname))
                {
                    skipped += 1;
                    continue;
                }
                let w = s.info(&wname).expect("present");
                let (rows, cols) = (w.shape[0] as usize, w.shape[1] as usize * 2 / 16);
                let (_, fixed) = s.read(&fname)?;
                let (_, ex) = s.read_u32(&ename)?;
                let (_, nex) = s.read_u32(&nname)?;
                let d = decode_variant(&fixed, &ex, rows, cols, Variant::Exact)
                    .map_err(|er| format!("{fname}: {er}"))?;
                let nib = decode_variant(&fixed, &ex, rows, cols, Variant::NibbleSwap)
                    .map_err(|er| format!("{fname}: {er}"))?;
                let wrong = decode_variant(&fixed, &nex, rows, cols, Variant::Exact)
                    .map_err(|er| format!("{fname} x {nname}: {er}"))?;
                writeln!(
                    out,
                    "{x}\t{rows}\t{cols}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    d.exceptions,
                    d.redundant,
                    d.repeated,
                    d.base_max,
                    hex(&d.scales),
                    hex(&nib.scales),
                    hex(&wrong.scales)
                )
                .map_err(|er| er.to_string())?;
                exc[pi] += d.exceptions as u64;
                scales[pi] += (rows * cols) as u64;
                base_max = base_max.max(d.base_max);
                done += 1;
            }
        }
    }
    for (pi, p) in PROJ.iter().enumerate() {
        eprintln!(
            "fn144 csf {p}: {} exceptions of {} scales ({:.4} %)",
            exc[pi],
            scales[pi],
            100.0 * exc[pi] as f64 / scales[pi].max(1) as f64
        );
    }
    eprintln!("fn144 csf: {done} planes decoded, {skipped} skipped (shard not complete), base max {base_max}");
    Ok(())
}

fn plain(dir: &str, out: &mut dyn Write) -> Result<(), String> {
    let mut s = ShardSet::open(std::path::Path::new(dir))?;
    let (mut done, mut skipped, mut exc, mut n) = (0usize, 0usize, 0u64, 0u64);
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            for p in PROJ {
                let name = format!("{}.weight_scale", prefix(l, e, p));
                if !s.present(&name) {
                    skipped += 1;
                    continue;
                }
                let (i, b) = s.read(&name)?;
                let (rows, cols) = (i.shape[0] as usize, i.shape[1] as usize);
                let (fixed, ex) = encode(&b, rows, cols).map_err(|er| format!("{name}: {er}"))?;
                let d = decode_variant(&fixed, &ex, rows, cols, Variant::Exact)
                    .map_err(|er| format!("{name}: {er}"))?;
                writeln!(
                    out,
                    "{name}\t{rows}\t{cols}\t{}\t{}\t{}",
                    ex.len(),
                    hex(&b),
                    hex(&d.scales)
                )
                .map_err(|er| er.to_string())?;
                if d.scales != b {
                    return Err(format!("{name}: round trip differs"));
                }
                exc += ex.len() as u64;
                n += (rows * cols) as u64;
                done += 1;
            }
        }
    }
    eprintln!(
        "fn144 plain round trip: {done} planes == after encode+decode, {skipped} absent; \
         min-base encoder exceptions {exc} of {n} ({:.3} %)",
        100.0 * exc as f64 / n.max(1) as f64
    );
    Ok(())
}

fn main() -> ExitCode {
    let mut ckpt = None;
    let mut plain_dir = None;
    let mut out_path = None;
    let mut layers = (0usize, LAYERS);
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--checkpoint" => ckpt = a.next(),
            "--plain" => plain_dir = a.next(),
            "--out" => out_path = a.next(),
            "--layers" => {
                let v = a.next().unwrap_or_default();
                let mut it = v.split(':').filter_map(|t| t.parse().ok());
                layers = (it.next().unwrap_or(0), it.next().unwrap_or(LAYERS));
            }
            _ => {
                eprintln!("usage: saber-fp4local-csf (--checkpoint DIR | --plain DIR) --out TSV [--layers A:B]");
                return ExitCode::from(2);
            }
        }
    }
    let Some(op) = out_path else {
        eprintln!("saber-fp4local-csf: --out TSV is required");
        return ExitCode::from(2);
    };
    let mut f = match std::fs::File::create(&op) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            eprintln!("{op}: {e}");
            return ExitCode::from(1);
        }
    };
    let r = match (ckpt, plain_dir) {
        (Some(d), None) => csf(&d, &mut f, layers),
        (None, Some(d)) => plain(&d, &mut f),
        _ => Err("exactly one of --checkpoint / --plain".into()),
    };
    match r.and_then(|()| f.flush().map_err(|e| e.to_string())) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("saber-fp4local-csf: {e}");
            ExitCode::from(1)
        }
    }
}
