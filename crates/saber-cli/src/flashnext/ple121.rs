//! FN121: the cold PLE drive-read tail -- the PLE prefetch law, the keep-awake's coverage, the
//! embedding rows from host RAM (`--fn-ple-prefetch <form>`), and the per-rank step / read records
//! that attribute a served step's tail (`--fn-ple-trace DIR`, a diagnostic).
//!
//! The forms (a comma set; `off` = today = the control, `all` = every part):
//! - `cand`: THE PLE PREFETCH LAW. At the end of a drafter round (`Cmd::Commit`) and of a first
//!   round (`Cmd::First`) the next verify step's candidate tokens are known -- every live
//!   sequence's committed token and ALL its drafts -- and their n-gram rows (hashed exactly as
//!   `decode_step` will, from the sequence's context over `[next, d1, d2, ..]`) are submitted to
//!   the drive in ONE batch WITHOUT waiting ([`saber_mem::ple_tier::PleTier::prefetch`]: the
//!   pool's workers' reads in flight together). They land in the RAM row cache before the step's
//!   PLE gate (every gather settles a prefetch in flight first), so a prefetched row is exactly
//!   the bytes a gather would read. A row NOT resident when the gate gathers is a counted stall
//!   (`fn121 ple-miss`: rows, steps, ms), never a wrong byte.
//! - `keep`: the keep-awake's coverage -- the period drops to `--fn-ple-keep-ms` (default
//!   [`DEFAULT_FN_PLE_KEEP_MS`]) and the boot checks that the keep-awake's file, every PLE file
//!   and the embedding file sit on ONE block device (both ranks' drive paths are that device),
//!   starting one reader per distinct device otherwise; the end line reports the largest gap
//!   between the reader's commands.
//! - `embed`: the host embedding gather (`store::embed_rows`) reads a host-RAM copy of the whole
//!   table (read once at boot, rows checked against the file) instead of a page-cache read of the
//!   checkpoint file -- a cold page there is a drive read inside the step's host span.

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use anyhow::Context as _;
use saber_mem::ple_tier::{PleTier, ReadBatch};
use serde_json::json;

/// FN121: `--fn-ple-prefetch` when absent -- `off` (today; the control).
pub const DEFAULT_FN_PLE_PREFETCH: &str = "off";
/// FN121: `--fn-ple-keep-ms` when absent (the `keep` part's keep-awake period).
pub const DEFAULT_FN_PLE_KEEP_MS: u64 = 5;

/// FN121: the parts of `--fn-ple-prefetch`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PleForm {
    /// The PLE prefetch law (the candidate tokens' rows submitted after every drafter round).
    pub cand: bool,
    /// The keep-awake's coverage (period `--fn-ple-keep-ms`, one reader per drive).
    pub keep: bool,
    /// The embedding rows from host RAM.
    pub embed: bool,
}

impl PleForm {
    /// Parse `off` | `all` | a comma set of `cand`, `keep`, `embed`.
    pub fn parse(s: &str) -> anyhow::Result<PleForm> {
        let mut f = PleForm::default();
        match s {
            "off" => return Ok(f),
            "all" => {
                return Ok(PleForm {
                    cand: true,
                    keep: true,
                    embed: true,
                })
            }
            _ => {}
        }
        for p in s.split(',') {
            match p {
                "cand" => f.cand = true,
                "keep" => f.keep = true,
                "embed" => f.embed = true,
                _ => anyhow::bail!(
                    "--fn-ple-prefetch off | all | a comma set of cand, keep, embed, not {s:?}"
                ),
            }
        }
        Ok(f)
    }

    /// The canonical name (`off`, `all`, or the set in order).
    #[must_use]
    pub fn name(&self) -> String {
        if *self == PleForm::default() {
            return "off".into();
        }
        if self.cand && self.keep && self.embed {
            return "all".into();
        }
        [
            ("cand", self.cand),
            ("keep", self.keep),
            ("embed", self.embed),
        ]
        .iter()
        .filter(|(_, on)| *on)
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(",")
    }
}

/// FN121: a rank's counters + its optional record writer.
#[derive(Debug, Default)]
pub struct F121 {
    pub form: PleForm,
    w: Option<std::io::BufWriter<std::fs::File>>,
    /// Decode steps recorded / drafter rounds recorded.
    steps: u64,
    rounds: u64,
    /// At the PLE gate (the late gather of a decode step): gates, distinct rows, rows resident,
    /// rows NOT resident (the counted stall), steps with a stall, the stall's gather ms.
    pub gates: u64,
    pub rows: u64,
    pub resident: u64,
    pub miss_rows: u64,
    pub miss_steps: u64,
    pub miss_ms: f64,
    /// Settles at the gate that waited on a prefetch still in flight, and their ms.
    pub inflight_waits: u64,
    pub inflight_ms: f64,
    /// The prefetch submissions (calls, rows submitted, host ms of the submissions).
    pub pf_calls: u64,
    pub pf_rows: u64,
    pub pf_ms: f64,
}

impl F121 {
    /// The rank's FN121 state (`dir` = `--fn-ple-trace`: `rank<r>.jsonl` there).
    pub fn new(form: PleForm, dir: Option<&Path>, rank: u32) -> anyhow::Result<F121> {
        let w = match dir {
            Some(d) => {
                std::fs::create_dir_all(d)?;
                let p = d.join(format!("rank{rank}.jsonl"));
                let f = std::fs::File::create(&p).with_context(|| p.display().to_string())?;
                let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
                writeln!(
                    w,
                    "{}",
                    json!({"format": "fn121-ple/1", "rank": rank, "form": form.name(),
                        "t0_us": saber_mem::ple_tier::unix_us(Instant::now())})
                )?;
                Some(w)
            }
            None => None,
        };
        Ok(F121 {
            form,
            w,
            ..F121::default()
        })
    }

    /// Whether records are written (the PLE tier then logs its read batches).
    #[must_use]
    pub fn tracing(&self) -> bool {
        self.w.is_some()
    }

    /// One record line (a no-op without the trace).
    fn put(&mut self, v: &serde_json::Value) -> anyhow::Result<()> {
        if let Some(w) = self.w.as_mut() {
            writeln!(w, "{v}")?;
            if (self.steps + self.rounds) % 256 == 0 {
                w.flush()?;
            }
        }
        Ok(())
    }

    /// The read batches since the last record, as JSON (per-read durations + start offsets µs).
    fn batches(ple: &mut PleTier) -> serde_json::Value {
        let b: Vec<ReadBatch> = ple.take_read_log();
        serde_json::Value::Array(
            b.iter()
                .map(|x| {
                    json!({"k": x.kind, "t": x.submit_unix_us,
                        "idle": if x.idle_us == u64::MAX { -1.0 } else { x.idle_us as f64 },
                        "wait": (x.wait_us * 10.0).round() / 10.0,
                        "dur": x.reads.iter().map(|r| (r.dur_us * 10.0).round() / 10.0).collect::<Vec<_>>(),
                        "st": x.reads.iter().map(|r| (r.start_us * 10.0).round() / 10.0).collect::<Vec<_>>()})
                })
                .collect(),
        )
    }

    /// A decode step's record (`k: s`).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        ple: &mut PleTier,
        t0_us: u64,
        bucket: u32,
        w: u32,
        n: usize,
        dev_ms: f64,
        host_ms: f64,
        host: &std::collections::BTreeMap<&'static str, f64>,
        nvme: u64,
        hits: u64,
        gate_ms: f64,
    ) -> anyhow::Result<()> {
        self.steps += 1;
        if !self.tracing() {
            return Ok(());
        }
        let h = |k: &str| (host.get(k).copied().unwrap_or(0.0) * 1e3).round() / 1e3;
        let reads = Self::batches(ple);
        let v = json!({"k": "s", "i": self.steps, "t": t0_us, "b": bucket, "w": w, "n": n,
            "dev": (dev_ms * 1e3).round() / 1e3, "host": (host_ms * 1e3).round() / 1e3,
            "emb": h("embed"), "tables": h("tables"), "ple": h("ple"), "l0": h("ple_l0"),
            "gate": h("ple_gate_wait"), "tier": h("tier"), "issue": h("issue"), "wait": h("wait"),
            "tier_end": h("tier_end"), "pick": h("pick"), "under": h("under"),
            "pfw": h("fn121_inflight_ms"), "miss": h("fn121_miss_rows"), "gr": h("fn121_rows"),
            "tgate": (gate_ms * 1e3).round() / 1e3, "nv": nvme, "hit": hits, "reads": reads});
        self.put(&v)
    }

    /// A drafter round's record (`k: r`; `parts` = the round's `rd_*` ms deltas).
    #[allow(clippy::too_many_arguments)]
    pub fn round(
        &mut self,
        ple: &mut PleTier,
        t0_us: u64,
        wall_ms: f64,
        dev_ms: f64,
        parts: &[(&'static str, f64)],
        pf_rows: u64,
        pf_ms: f64,
    ) -> anyhow::Result<()> {
        self.rounds += 1;
        if !self.tracing() {
            return Ok(());
        }
        let reads = Self::batches(ple);
        let mut m = serde_json::Map::new();
        for (k, v) in parts {
            m.insert((*k).to_string(), json!((v * 1e3).round() / 1e3));
        }
        let v = json!({"k": "r", "i": self.rounds, "t": t0_us,
            "wall": (wall_ms * 1e3).round() / 1e3, "dev": (dev_ms * 1e3).round() / 1e3,
            "parts": m, "pf": pf_rows, "pf_ms": (pf_ms * 1e3).round() / 1e3, "reads": reads});
        self.put(&v)
    }

    /// The gate's reading before a decode step's late gather of `ids`: a prefetch in flight
    /// settled (its wait counted), then the rows resident vs not (the counted stall). Returns
    /// (in-flight wait ms, distinct rows, rows not resident).
    pub fn gate(&mut self, ple: &mut PleTier, ids: &[u64]) -> anyhow::Result<(f64, u64, u64)> {
        let w = ple.settle().map_err(|e| anyhow::anyhow!("{e}"))?;
        if w > 0.02 {
            self.inflight_waits += 1;
            self.inflight_ms += w;
        }
        let mut d: Vec<u64> = ids.to_vec();
        d.sort_unstable();
        d.dedup();
        let res = ple.resident(&d);
        let miss = d.len() as u64 - res;
        self.gates += 1;
        self.rows += d.len() as u64;
        self.resident += res;
        self.miss_rows += miss;
        self.miss_steps += u64::from(miss > 0);
        Ok((w, d.len() as u64, miss))
    }

    /// The end line (`fnple121 rank<r>:`).
    #[must_use]
    pub fn end_line(&mut self, rank: u32, ple: &PleTier) -> String {
        if let Some(w) = self.w.as_mut() {
            let _ = w.flush();
        }
        let p = ple.prefetch_counts();
        format!(
            "fnple121 rank{rank}: form {} | gates {} rows {} resident {} | fn121 ple-miss rows {} steps {} ({:.1} ms gathered at the gate) | in-flight waits {} ({:.1} ms) | prefetch calls {} rows submitted {} (cached {} pending {}) batches {} submit {:.1} ms | steps {} rounds {} recorded{}",
            self.form.name(),
            self.gates,
            self.rows,
            self.resident,
            self.miss_rows,
            self.miss_steps,
            self.miss_ms,
            self.inflight_waits,
            self.inflight_ms,
            p.calls,
            p.submitted,
            p.cached,
            p.pending,
            p.batches,
            self.pf_ms,
            self.steps,
            self.rounds,
            if self.tracing() { " (traced)" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn121_the_form_parses_off_all_and_sets_and_off_is_the_default() {
        assert_eq!(
            PleForm::parse(DEFAULT_FN_PLE_PREFETCH).unwrap(),
            PleForm::default()
        );
        let a = PleForm::parse("all").unwrap();
        assert!(a.cand && a.keep && a.embed);
        assert_eq!(a.name(), "all");
        let c = PleForm::parse("cand,embed").unwrap();
        assert!(c.cand && !c.keep && c.embed);
        assert_eq!(c.name(), "cand,embed");
        assert_eq!(PleForm::parse("keep").unwrap().name(), "keep");
        assert_eq!(PleForm::parse("cand,keep,embed").unwrap().name(), "all");
        // the control: an unknown part is refused by name
        assert!(PleForm::parse("cand,warm")
            .unwrap_err()
            .to_string()
            .contains("cand, keep, embed"));
        assert!(PleForm::parse("").is_err());
    }
}
