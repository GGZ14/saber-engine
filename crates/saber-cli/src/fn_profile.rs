//! `saber flashnext-profile` (FN18): the memory plane's placement intelligence on the host, over the
//! serve plane's route trace (`--fn-route-trace`, `saber_mem::route_trace`):
//!
//! * `merge`   -- the real-traffic routing profile per window (corpus) and step kind
//!   (`fn7-route-profile/1` files the tier warm-starts from) + its concentration (top-N share per
//!   layer), entropy and rank correlation with a reference profile (FN3's);
//! * `replay`  -- the real `ExpertTier` over the trace's target steps (a counting mover): the
//!   per-layer VRAM hit rate, RAM reads / inserts / link bytes per step by bucket, for any warm
//!   start, slot count, insert law or owner permutation; `--metrics` checks the replay against a
//!   served boot's `saber_xtier_layer_routed_total` counters (the self-test: exact);
//! * `ceiling` -- the residency ceiling (the best static placement at the slot count);
//! * `owner`   -- the EP owner permutation estimate (per-layer max-rank distinct experts and the
//!   RAM reads per rank, identity vs a frequency-balanced owner);
//! * `ple`     -- the PLE row cache sized on the trace's gathers (cold and warm passes);
//! * `prefetch` -- FN47a: the step-ahead expert prefetch study (`saber_mem::step_ahead`): per
//!   predictor the recall of each decode step's misses, the copies, the staging pool and the
//!   step under the timing law (step-start and `--lead`-layers-ahead issue), by bucket.
//! * `rebuild` -- FN65: the USER'S warm-start profile from recorded served traffic: every target
//!   step of the traces (files, or the directories `--fn-route-record` / `serve_flashnext.sh
//!   --record-routes` fills) merged into one `fn7-route-profile/1` file in the pack's unit (steps),
//!   written as the ADDED pack member `profile/route_profile.user.json` (`--pack DIR`) or `--out
//!   FILE`; the pack's own `profile/route_profile.json` is never rewritten. Served with
//!   `--fn-profile user`.
//! * `tail` -- FN53a: the expert long tail on the SSD (a three-tier placement): the threshold
//!   tails and their held-out stability, and per tail size the drive reads a prefill chunk and a
//!   decode step add (serialized / one-layer-ahead prefetch) over the exact tier replay.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod admit;
mod ringorder;

use anyhow::{bail, Context};
use saber_mem::expert_tier::{
    Demote, PathCounts, RouteProfile, TierParams, VictimKind, VictimPolicy,
};
use saber_mem::route_trace::{
    balanced_owner, coldest_ram_tail, entropy_bits, ep_balance, learn_table, min_replay,
    ple_replay, replay, spearman, static_ceiling, tail_replay, top_share, Counts, InsertLaw,
    MinForm, Perm, ReplayOut, StepKind, TableMode, TraceReader, Window,
};

const STRIDE: u64 = 2_617_344;
const PER_RANK: u32 = 256;
/// Link time of one record crossing (FN8: 90-94 us over UVA, 96 us a DMA copy).
const US_PER_CROSSING: f64 = 94.0;

/// The subcommand's arguments.
#[derive(clap::Args, Debug, Clone)]
pub struct ProfileArgs {
    /// merge | replay | ceiling | owner | ple | ple-device | prefetch | tail | learn | rebuild
    pub mode: String,
    /// The route trace(s) (`fn18-route-trace/1`); `replay`/`owner`/`ple` read the first;
    /// `rebuild` also takes directories (every `*.bin` / `*.bin.part` in them).
    #[arg(long, required = true, alias = "traces", num_args = 1..)]
    pub trace: Vec<PathBuf>,
    /// rebuild (FN65): the pack the user profile is added to (`<pack>/profile/route_profile.user.json`;
    /// its own `route_profile.json` is the rank-correlation reference).
    #[arg(long)]
    pub pack: Option<PathBuf>,
    /// The capture driver's windows file (`[{name, t0, t1}]`).
    #[arg(long)]
    pub windows: Option<PathBuf>,
    /// Keep only these windows (comma list; `name:a` / `name:b` = the first / second half in time).
    #[arg(long, default_value = "")]
    pub only: String,
    /// Output directory (merge: the profiles + stats.json).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// A reference profile for the rank correlation (merge) / the warm start (replay; `none` =
    /// uniform, id order).
    #[arg(long)]
    pub profile: Option<PathBuf>,
    /// VRAM slots per rank.
    #[arg(long, default_value_t = 7404)]
    pub vram: u32,
    /// RAM slots per rank (default: the complement).
    #[arg(long)]
    pub ram: Option<u32>,
    /// Rank(s) to replay.
    #[arg(long, default_value = "0,1")]
    pub ranks: String,
    /// The default insert cap (prefill chunks + unlisted buckets).
    #[arg(long, default_value_t = 2)]
    pub max_inserts: u32,
    /// The per-bucket law (`1:2,8:1`).
    #[arg(long, default_value = "")]
    pub by_bucket: String,
    /// FN54: the per-bucket promotion byte budget a step (`8:96`, MiB; empty = none).
    #[arg(long, default_value = "")]
    pub promote_budget_mib: String,
    /// FN54: second-chance insertion window in steps, `W` or `4:W,8:W` (0 = off).
    #[arg(long, default_value = "0")]
    pub second_chance: String,
    /// Read-through fraction.
    #[arg(long, default_value_t = 0.5)]
    pub read_through: f64,
    /// LRFU decay (0 = LRU).
    #[arg(long, default_value_t = 0.0)]
    pub lambda: f64,
    /// Admission window (0 = off).
    #[arg(long, default_value_t = 0)]
    pub admit_window: u32,
    /// FN41: replay's victim policy -- `tier` (the real tier: LRU / LRFU / admission as flagged),
    /// `min` (the MIN / Belady oracle at today's cap + cadence), `min-pure` (MIN, no cap, no
    /// read-through: the bound at the same per-layer partitions), `min-global` (MIN over one
    /// rank-wide pool), `min-admit-all` (MIN's victims, every capped miss promoted).
    #[arg(long, default_value = "tier")]
    pub policy: String,
    /// FN39: LRFU admission margin (0 = off).
    #[arg(long, default_value_t = 0.0)]
    pub admit_margin: f64,
    /// FN39: RAM uses before a promotion (0 = off).
    #[arg(long, default_value_t = 0)]
    pub admit_uses: u32,
    /// FN36: replay the demotion-free law (victims re-read from NVMe, not copied D2H).
    #[arg(long)]
    pub demote_none: bool,
    /// FN42: replay with this many LENT slots beside `--vram` (the KV lending law's inclusive
    /// slots, lent at the start; 0 = today).
    #[arg(long, default_value_t = 0)]
    pub lend: u32,
    /// FN48: the victim ranking (`lrfu` = today, `reuse` = farthest predicted next use).
    #[arg(long, default_value = "lrfu")]
    pub victim: String,
    /// FN48: LRFU score a read-through layer's route adds (1 = today); reuse: 0 = skip them.
    #[arg(long, default_value_t = 1.0)]
    pub rt_weight: f64,
    /// FN48: LRFU lambda per layer (48 comma values; empty = `--lambda`).
    #[arg(long, default_value = "")]
    pub lambda_by_layer: String,
    /// FN48: LRFU + beta * ln(1 + warm profile count).
    #[arg(long, default_value_t = 0.0)]
    pub prior_beta: f64,
    /// FN48: the capped misses fill the cap in the policy's order.
    #[arg(long)]
    pub miss_rank: bool,
    /// FN48: reuse EWMA weight.
    #[arg(long, default_value_t = 0.3)]
    pub reuse_alpha: f64,
    /// FN48: reuse overdue factor.
    #[arg(long, default_value_t = 1.0)]
    pub reuse_age: f64,
    /// FN48: reuse gap of a once-seen record.
    #[arg(long, default_value_t = 64.0)]
    pub reuse_first_gap: f64,
    /// FN48: the learned victim table (`learn` writes it, `replay --victim table` reads it).
    #[arg(long)]
    pub table: Option<PathBuf>,
    /// FN48: LRFU bump = (rows routing)^gamma (0 = 1 a record, today); the replay feeds live rows.
    #[arg(long, default_value_t = 0.0)]
    pub row_gamma: f64,
    /// FN48: the table's class-counter decay.
    #[arg(long, default_value_t = 0.98)]
    pub table_lambda: f64,
    /// FN48: the table's horizon in steps.
    #[arg(long, default_value_t = 256)]
    pub horizon: u64,
    /// FN48: the table's value (`etr` | `prob`).
    #[arg(long, default_value = "etr")]
    pub table_mode: String,
    /// A served boot's /metrics (the self-test: the replay must equal its per-layer counters).
    #[arg(long)]
    pub metrics: Option<PathBuf>,
    /// A served boot's log with its `fntierend rank<r>:` line (the exact self-test).
    #[arg(long)]
    pub tier_log: Option<PathBuf>,
    /// owner: the windows the permutation is trained on (comma list, `only` syntax).
    #[arg(long, default_value = "")]
    pub train: String,
    /// ple: cache sizes in rows.
    #[arg(long, default_value = "0,65536,262144,1048576,3355443")]
    pub sizes: String,
    /// FN58 ple-device: the host's RAM row cache behind the device table (MiB a rank).
    #[arg(long, default_value_t = 512)]
    pub ple_ram_mib: u64,
    /// FN58 ple-device: host PLE ms of a step with no drive read (FN56's census; the most a
    /// host-free step saves).
    #[arg(long, default_value_t = 0.03)]
    pub ple_free_ms: f64,
    /// Label printed on every result line.
    #[arg(long, default_value = "")]
    pub label: String,
    /// prefetch: predictors (`batch1`, `batch4`, `top8`, `batch1+top8`, `oracle`).
    #[arg(
        long,
        default_value = "batch1,batch2,batch4,top8,top16,batch1+top8,oracle"
    )]
    pub preds: String,
    /// prefetch: compute ms of a step with no link time, by bucket (`b:ms,...`).
    #[arg(long, default_value = "1:28.4,2:34,4:42,8:55")]
    pub compute_ms: String,
    /// prefetch: the demand (UVA) link rate, GB/s (Gen5 x8 = 27.7).
    #[arg(long, default_value_t = 27.7)]
    pub link_gbps: f64,
    /// prefetch: the copy-engine rate, GB/s.
    #[arg(long, default_value_t = 21.4)]
    pub copy_gbps: f64,
    /// prefetch: the trace the cross-layer table (`xlayerK` predictors) is trained on (held out:
    /// the OTHER capture's).
    #[arg(long)]
    pub xtrain: Option<PathBuf>,
    /// prefetch: the issue lead in layers of the layer-ahead form.
    #[arg(long, default_value_t = 2)]
    pub lead: usize,
    /// tail (FN53a): tail sizes in records per rank (the coldest RAM-placed records by the
    /// `--xtrain` trace's decode step share, then prefill row share).
    #[arg(long, default_value = "0,256,512,1024,2048")]
    pub tails: String,
    /// tail (FN53a): the wall of a 2,048-row prefill chunk, ms (the served 665).
    #[arg(long, default_value_t = 665.0)]
    pub chunk_ms: f64,
    /// tail (FN53a): the LAYER-BALANCED tail (at most ceil(T / 48) records a layer first).
    #[arg(long)]
    pub tail_balanced: bool,
    /// pfcensus (FN94): prefill-chunk row bands (`lo-hi,lo-hi`, inclusive).
    #[arg(long, default_value = "100-140,141-200,100-200")]
    pub census_rows: String,
    /// pfcensus (FN94): staging-ring slots a layer to price (profile order, FN43's `stage_order`).
    #[arg(long, default_value = "8,16,32,48,64,128")]
    pub census_slots: String,
    /// ringorder (LR-H): staged-pass row bands (`lo-hi,lo-hi`, inclusive).
    #[arg(long, default_value = "2000-3000,3001-4500,4501-1000000")]
    pub ring_bands: String,
    /// ringorder (LR-H): prefill chunks of at least this many rows stage (the served law 2,049).
    #[arg(long, default_value_t = 2049)]
    pub ring_min_rows: u32,
    /// ringorder (LR-H): ring slots a half (the served 128).
    #[arg(long, default_value_t = 128)]
    pub ring_slots: u32,
    /// ringorder (LR-H): local experts of the replayed rank (512 = the tp2 expert law).
    #[arg(long, default_value_t = 512)]
    pub ring_experts: u32,
    /// ringorder (LR-H): the record stride (tp2 half records, 1.25 MiB).
    #[arg(long, default_value_t = 1_310_720)]
    pub ring_stride: u64,
    /// sched (FN104): the scheduler's law over the tier (`saber_mem::sched_signal::SchedLaw`:
    /// `fd=F,alive=H,prot=N,left=L,ant=K,beta=B,prewarm=P,dead=D,census`; empty = LRFU exactly).
    #[arg(long, default_value = "")]
    pub sched: String,
    /// sched (FN104): the boot's widest prefill chunk (a chunk this wide continues its request).
    #[arg(long, default_value_t = 2048)]
    pub max_chunk: u32,
    /// admcensus (FN103): the admission law replayed (`today` | `slow:HI:LO:THR` |
    /// `xexp:TOTAL:MAX` | `miss:TOTAL:MAX`; `fn_profile/admit.rs`).
    #[arg(long, default_value = "today")]
    pub admit_law: String,
    /// admcensus (FN103): the planner's EWMA weight a step.
    #[arg(long, default_value_t = 0.1)]
    pub admit_alpha: f64,
    /// admcensus (FN103): us of link a read EXPOSED on the slower rank, by bucket (0 = a prefill
    /// chunk: FN83's 102; b1 FN84's T 4 65; b8 FN47b's 88).
    #[arg(long, default_value = "0:102,1:65,2:65,4:75,8:88")]
    pub exposed_us: String,
}

fn windows(a: &ProfileArgs, only: &str) -> anyhow::Result<Option<Vec<Window>>> {
    let Some(p) = &a.windows else {
        return Ok(None);
    };
    let all = Window::parse_all(&std::fs::read_to_string(p)?).map_err(anyhow::Error::msg)?;
    if only.is_empty() {
        return Ok(Some(all));
    }
    let mut out = Vec::new();
    for want in only.split(',') {
        let (name, half) = match want.split_once(':') {
            Some((n, h)) => (n, Some(h)),
            None => (want, None),
        };
        let mut hit = false;
        for w in all.iter().filter(|w| w.name == name || name == "*") {
            hit = true;
            let mid = (w.t0 + w.t1) / 2.0;
            out.push(match half {
                None => w.clone(),
                Some("a") => Window {
                    t1: mid,
                    ..w.clone()
                },
                Some("b") => Window {
                    t0: mid,
                    ..w.clone()
                },
                Some(h) => bail!("window half '{h}' (a | b)"),
            });
        }
        if !hit {
            bail!("no window named {name}");
        }
    }
    Ok(Some(out))
}

fn load_profile(p: &Option<PathBuf>) -> anyhow::Result<Option<RouteProfile>> {
    match p {
        None => Ok(None),
        Some(p) if p.as_os_str() == "none" => Ok(None),
        Some(p) => Ok(Some(
            RouteProfile::from_json(&std::fs::read_to_string(p)?)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )),
    }
}

/// Run the subcommand.
pub fn run(a: &ProfileArgs) -> anyhow::Result<()> {
    match a.mode.as_str() {
        "merge" => merge(a),
        "replay" => replay_mode(a),
        "learn" => learn_mode(a),
        "ceiling" => ceiling(a),
        "owner" => owner(a),
        "ple" => ple(a),
        "prefetch" => prefetch(a),
        "tail" => tail(a),
        "ple-device" => ple_device(a),
        "rebuild" => rebuild(a).map(|_| ()),
        "pfcensus" => pfcensus(a),
        "sched" => sched_mode(a),
        "admcensus" => admit::admcensus(a),
        "ringorder" => ringorder::ringorder(a),
        m => {
            bail!("flashnext-profile mode {m} (merge | replay | ceiling | owner | ple | ple-device | prefetch | tail | rebuild | pfcensus | sched | admcensus | ringorder)")
        }
    }
}

/// Counts per (window name, kind) over the traces; kinds: prefill / decode / mtp.
fn count(
    a: &ProfileArgs,
    ws: Option<&[Window]>,
) -> anyhow::Result<BTreeMap<(String, String), Counts>> {
    let mut out: BTreeMap<(String, String), Counts> = BTreeMap::new();
    for t in &a.trace {
        let mut r = TraceReader::open(t)?;
        while let Some(s) = r.next_step()? {
            let kind = match s.kind {
                StepKind::Prefill => "prefill",
                StepKind::Decode => "decode",
                StepKind::MtpDecode | StepKind::MtpPrefill => "mtp",
                StepKind::Ple => continue,
            };
            let name = match ws {
                None => "all".to_string(),
                Some(ws) => match ws.iter().find(|w| w.holds(s.t)) {
                    Some(w) => w.name.clone(),
                    None => continue,
                },
            };
            let layers = if kind == "mtp" { 1 } else { r.layers as usize };
            out.entry((name, kind.to_string()))
                .or_insert_with(|| Counts::new(layers, r.experts as usize))
                .add(&s);
        }
    }
    Ok(out)
}

fn stats_line(c: &Counts, reference: Option<&RouteProfile>) -> serde_json::Value {
    let rows = &c.routes;
    let n = rows.len() as f64;
    let mean = |f: &dyn Fn(&Vec<u64>) -> f64| rows.iter().map(f).sum::<f64>() / n;
    let mut v = serde_json::json!({
        "tokens": c.tokens, "steps": c.steps,
        "entropy_bits_mean": mean(&|r| entropy_bits(r)),
        "top32_share": mean(&|r| top_share(r, 32)),
        "top64_share": mean(&|r| top_share(r, 64)),
        "top128_share": mean(&|r| top_share(r, 128)),
        "top256_share": mean(&|r| top_share(r, 256)),
        "top308_share": mean(&|r| top_share(r, 308)),
        "entropy_bits_per_layer": rows.iter().map(|r| (entropy_bits(r) * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        "top308_share_per_layer": rows.iter().map(|r| (top_share(r, 308) * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
    });
    if let Some(p) = reference {
        if p.counts.len() == rows.len() {
            let mut sp: Vec<f64> = rows
                .iter()
                .zip(&p.counts)
                .map(|(x, y)| spearman(x, y))
                .collect();
            v["spearman_vs_reference_mean"] = serde_json::json!(sp.iter().sum::<f64>() / n);
            sp.sort_by(f64::total_cmp);
            v["spearman_vs_reference_median"] = serde_json::json!(sp[sp.len() / 2]);
            v["spearman_vs_reference_min"] = serde_json::json!(sp[0]);
        }
    }
    v
}

fn merge(a: &ProfileArgs) -> anyhow::Result<()> {
    let out = a.out.clone().context("merge needs --out")?;
    std::fs::create_dir_all(&out)?;
    let ws = windows(a, &a.only)?;
    let counts = count(a, ws.as_deref())?;
    let reference = load_profile(&a.profile)?;
    let mut total: BTreeMap<String, Counts> = BTreeMap::new();
    let mut stats = serde_json::Map::new();
    for ((w, k), c) in &counts {
        total
            .entry(k.clone())
            .or_insert_with(|| Counts::new(c.routes.len(), c.routes[0].len()))
            .merge(c);
        let src = format!("fn18 served trace, window {w}, {k}");
        std::fs::write(
            out.join(format!("profile_{w}_{k}.json")),
            c.profile_json(&src, false),
        )?;
        let r = if k == "mtp" { None } else { reference.as_ref() };
        let s = stats_line(c, r);
        println!(
            "fnprofile {w:>10} {k:>7}: tokens {:>9} steps {:>7} | entropy {:.3} bits | top-64/128/256/308 of 512 {:.3}/{:.3}/{:.3}/{:.3}{}",
            c.tokens,
            c.steps,
            s["entropy_bits_mean"].as_f64().unwrap_or(0.0),
            s["top64_share"].as_f64().unwrap_or(0.0),
            s["top128_share"].as_f64().unwrap_or(0.0),
            s["top256_share"].as_f64().unwrap_or(0.0),
            s["top308_share"].as_f64().unwrap_or(0.0),
            s.get("spearman_vs_reference_median")
                .map(|x| format!(" | spearman vs ref median {:.3}", x.as_f64().unwrap_or(0.0)))
                .unwrap_or_default()
        );
        stats.insert(format!("{w}/{k}"), s);
    }
    // the served mix: every window, prefill + decode (the warm-start candidate), both units
    let mut served = Counts::new(48, 512);
    for k in ["prefill", "decode"] {
        if let Some(c) = total.get(k) {
            served.merge(c);
        }
    }
    for (k, c) in total
        .iter()
        .chain(std::iter::once((&"served".to_string(), &served)))
    {
        let src = format!("fn18 served trace, all windows, {k}");
        std::fs::write(
            out.join(format!("profile_all_{k}.json")),
            c.profile_json(&src, false),
        )?;
        std::fs::write(
            out.join(format!("profile_all_{k}_steps.json")),
            c.profile_json(&src, true),
        )?;
        let r = if k == "mtp" { None } else { reference.as_ref() };
        let s = stats_line(c, r);
        println!(
            "fnprofile {:>10} {k:>7}: tokens {:>9} steps {:>7} | entropy {:.3} bits | top-64/128/256/308 of 512 {:.3}/{:.3}/{:.3}/{:.3}{}",
            "ALL",
            c.tokens,
            c.steps,
            s["entropy_bits_mean"].as_f64().unwrap_or(0.0),
            s["top64_share"].as_f64().unwrap_or(0.0),
            s["top128_share"].as_f64().unwrap_or(0.0),
            s["top256_share"].as_f64().unwrap_or(0.0),
            s["top308_share"].as_f64().unwrap_or(0.0),
            s.get("spearman_vs_reference_median")
                .map(|x| format!(" | spearman vs ref median {:.3}", x.as_f64().unwrap_or(0.0)))
                .unwrap_or_default()
        );
        stats.insert(format!("ALL/{k}"), s);
    }
    std::fs::write(
        out.join("stats.json"),
        serde_json::to_string_pretty(&stats)?,
    )?;
    Ok(())
}

/// FN65: `rebuild` -- the user's profile from recorded traffic (see the module doc). Returns the
/// file written.
pub fn rebuild(a: &ProfileArgs) -> anyhow::Result<PathBuf> {
    use crate::flashnext::run::{DEFAULT_PROFILE_IN_PACK, USER_PROFILE_IN_PACK};
    let out = match (&a.out, &a.pack) {
        (Some(o), _) => o.clone(),
        (None, Some(p)) => p.join(USER_PROFILE_IN_PACK),
        (None, None) => {
            bail!("rebuild needs --pack DIR (writes profile/route_profile.user.json) or --out FILE")
        }
    };
    let own = Path::new(DEFAULT_PROFILE_IN_PACK)
        .file_name()
        .expect("a file name");
    if out.file_name() == Some(own) {
        bail!(
            "{}: the pack's own profile is never rewritten (FN65) -- the user's goes to {}",
            out.display(),
            USER_PROFILE_IN_PACK
        );
    }
    let files = saber_mem::route_trace::trace_files(&a.trace)?;
    if files.is_empty() {
        bail!("rebuild: no trace files under {:?}", a.trace);
    }
    let mut c: Option<Counts> = None;
    let (mut t0, mut t1) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut truncated = 0;
    for f in &files {
        let mut r = TraceReader::open(f).with_context(|| format!("{}", f.display()))?;
        let cc = c.get_or_insert_with(|| Counts::new(r.layers as usize, r.experts as usize));
        if cc.routes.len() != r.layers as usize || cc.routes[0].len() != r.experts as usize {
            bail!(
                "{}: {} layers x {} experts, the others differ",
                f.display(),
                r.layers,
                r.experts
            );
        }
        loop {
            match r.next_step() {
                Ok(Some(s)) => {
                    if s.kind.target() {
                        t0 = t0.min(s.t);
                        t1 = t1.max(s.t);
                        cc.add(&s);
                    }
                }
                Ok(None) => break,
                // a stopped recorder's open file ends mid-record: keep what is whole
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    truncated += 1;
                    break;
                }
                Err(e) => return Err(anyhow::anyhow!("{}: {e}", f.display())),
            }
        }
    }
    let c = c.context("no trace")?;
    if c.steps == 0 {
        bail!("rebuild: the traces hold no target step");
    }
    let src = format!(
        "fn65 user traffic: {} files, {} target steps, unix {:.0}..{:.0}",
        files.len(),
        c.steps,
        t0,
        t1
    );
    let reference = match (&a.profile, &a.pack) {
        (Some(_), _) => load_profile(&a.profile)?,
        (None, Some(p)) if p.join(DEFAULT_PROFILE_IN_PACK).is_file() => {
            load_profile(&Some(p.join(DEFAULT_PROFILE_IN_PACK)))?
        }
        _ => None,
    };
    // the pack's unit (steps): compare occurrence counts with the reference's
    let mut by_steps = c.clone();
    by_steps.routes.clone_from(&c.occ);
    let st = stats_line(&by_steps, reference.as_ref());
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, c.profile_json(&src, true))?;
    std::fs::rename(&tmp, &out)?;
    println!(
        "fnrebuild: {} files ({} truncated tails), {} target steps, {} tokens | top-308 of 512 {:.3} | spearman vs the pack's profile median {} -> {}",
        files.len(),
        truncated,
        c.steps,
        c.tokens,
        st["top308_share"].as_f64().unwrap_or(0.0),
        st.get("spearman_vs_reference_median")
            .and_then(serde_json::Value::as_f64)
            .map_or("-".to_string(), |x| format!("{x:.3}")),
        out.display()
    );
    Ok(out)
}

fn params(a: &ProfileArgs) -> TierParams {
    TierParams {
        vram_slots: a.vram,
        ram_slots: a.ram.unwrap_or(48 * PER_RANK - a.vram),
        read_through: a.read_through,
        max_inserts: a.max_inserts.max(1),
        lrfu_lambda: a.lambda,
        admit_window: a.admit_window,
        admit_margin: a.admit_margin,
        admit_uses: a.admit_uses,
        demote: if a.demote_none {
            Demote::None
        } else {
            Demote::Ram
        },
        refill_spares: if a.demote_none {
            2 * a.max_inserts.max(1) * 48
        } else {
            0
        },
        lend_boot: a.lend,
        policy: VictimPolicy {
            kind: match a.victim.as_str() {
                "reuse" => VictimKind::Reuse,
                "table" => VictimKind::Table,
                _ => VictimKind::Lrfu,
            },
            rt_weight: a.rt_weight,
            lambda_by_layer: a
                .lambda_by_layer
                .split(',')
                .filter(|x| !x.trim().is_empty())
                .map(|x| x.trim().parse::<f64>().expect("--lambda-by-layer"))
                .collect(),
            prior_beta: a.prior_beta,
            miss_rank: a.miss_rank,
            reuse_alpha: a.reuse_alpha,
            reuse_age: a.reuse_age,
            reuse_first_gap: a.reuse_first_gap,
            table: a
                .table
                .as_ref()
                .filter(|_| a.victim == "table")
                .map(|p| {
                    let v: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(p).expect("--table"))
                            .expect("--table json");
                    v["values"]
                        .as_array()
                        .expect("values")
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0))
                        .collect()
                })
                .unwrap_or_default(),
            table_lambda: a.table_lambda,
            row_gamma: a.row_gamma,
        },
        ..TierParams::default()
    }
}

fn ranks(a: &ProfileArgs) -> anyhow::Result<Vec<u32>> {
    a.ranks
        .split(',')
        .map(|r| {
            r.trim()
                .parse::<u32>()
                .map_err(|e| anyhow::anyhow!("--ranks: {e}"))
        })
        .collect()
}

fn report(label: &str, rank: u32, o: &ReplayOut) -> serde_json::Value {
    let mut b = serde_json::Map::new();
    for (k, t) in &o.buckets {
        let n = t.steps.max(1) as f64;
        let routed = t.paths.routed() as f64;
        let ram = t.paths.ram as f64 / n;
        let ins = t.inserts as f64 / n;
        let (d2h, nv) = (t.demotions as f64 / n, t.refills as f64 / n);
        let name = if *k == 0 {
            "prefill".to_string()
        } else {
            format!("b{k}")
        };
        println!(
            "fnreplay {label} rank{rank} {name:>7}: steps {:>7} | routed/step {:>7.1} | VRAM {:>6.2} % | RAM reads/step {:>6.2} | inserts/step {:>5.2} | link H2D {:>6.1} MB/step = {:>6.2} ms | D2H {:>5.2}/step {:>6.1} MB | NVMe refills {:>5.2}/step {:>6.1} MB",
            t.steps,
            routed / n,
            100.0 * t.paths.resident as f64 / routed.max(1.0),
            ram,
            ins,
            (ram + ins) * STRIDE as f64 / 1e6,
            (ram + ins) * US_PER_CROSSING / 1e3,
            d2h,
            d2h * STRIDE as f64 / 1e6,
            nv,
            nv * STRIDE as f64 / 1e6
        );
        b.insert(
            name,
            serde_json::json!({"steps": t.steps, "routed": t.paths.routed(), "resident": t.paths.resident,
                "ram": t.paths.ram, "inserts": t.inserts, "demotions": t.demotions, "refills": t.refills}),
        );
    }
    let per: Vec<f64> = o
        .per_layer
        .iter()
        .map(|c| (1000.0 * c.resident as f64 / c.routed().max(1) as f64).round() / 10.0)
        .collect();
    let tot = o.per_layer.iter().fold(PathCounts::default(), |mut x, c| {
        x.resident += c.resident;
        x.ram += c.ram;
        x
    });
    let worst = per.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "fnreplay {label} rank{rank}     ALL: VRAM {:.2} % of {} routed | per-layer min {worst:.1} % | per layer {:?}",
        100.0 * tot.resident as f64 / tot.routed().max(1) as f64,
        tot.routed(),
        per
    );
    serde_json::json!({"buckets": b, "per_layer_vram_pct": per,
        "resident": tot.resident, "ram": tot.ram})
}

fn metrics_layers(p: &Path, rank: u32) -> anyhow::Result<BTreeMap<(u32, String), u64>> {
    let mut m = BTreeMap::new();
    for line in std::fs::read_to_string(p)?.lines() {
        let Some(rest) = line.strip_prefix("saber_xtier_layer_routed_total{") else {
            continue;
        };
        let (labels, v) = rest.split_once("} ").context("metrics line")?;
        let get = |k: &str| {
            labels.split(',').find_map(|kv| {
                kv.strip_prefix(&format!("{k}=\""))
                    .map(|x| x.trim_end_matches('"').to_string())
            })
        };
        if get("rank").as_deref() != Some(&rank.to_string()) {
            continue;
        }
        let l: u32 = get("layer").context("layer")?.parse()?;
        m.insert((l, get("path").context("path")?), v.trim().parse()?);
    }
    Ok(m)
}

/// FN48: learn the victim table on `--trace` (rank 0) and write it to `--table`.
fn learn_mode(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    let prof = load_profile(&a.profile)?;
    let mode = match a.table_mode.as_str() {
        "etr" => TableMode::Etr,
        "prob" => TableMode::Prob,
        "density" => TableMode::Density,
        m => bail!("--table-mode {m} (etr | prob | density)"),
    };
    let v = learn_table(
        &a.trace[0],
        0,
        PER_RANK,
        STRIDE,
        &params(a),
        prof.as_ref(),
        ws.as_deref(),
        a.table_lambda,
        a.horizon,
        mode,
    )
    .map_err(anyhow::Error::msg)?;
    let out = a.table.as_ref().context("--table PATH")?;
    std::fs::write(
        out,
        serde_json::to_string(&serde_json::json!({
            "schema": "fn48-victim-table/1",
            "trace": a.trace[0].display().to_string(),
            "table_lambda": a.table_lambda,
            "horizon": a.horizon,
            "mode": a.table_mode,
            "classes": saber_mem::expert_tier::TABLE_CLASSES,
            "ages": saber_mem::expert_tier::TABLE_AGES,
            "values": v,
        }))?,
    )?;
    println!(
        "fnlearn {}: {} values -> {}",
        a.label,
        v.len(),
        out.display()
    );
    Ok(())
}

fn replay_mode(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    let mut prof = load_profile(&a.profile)?;
    let law = InsertLaw::parse(a.max_inserts, &a.by_bucket)
        .and_then(|l| l.with_budget_mib(&a.promote_budget_mib))
        .and_then(|l| l.with_second_chance(&a.second_chance))
        .map_err(anyhow::Error::msg)?;
    let perm = perm_from(a)?;
    if let (Some(p), Some(pr)) = (&perm, prof.as_mut()) {
        // the warm start follows its records to their new owners
        for (l, row) in pr.counts.iter_mut().enumerate() {
            let old = row.clone();
            for (e, c) in old.into_iter().enumerate() {
                row[p[l][e] as usize] = c;
            }
        }
    }
    let mut all = serde_json::Map::new();
    for rank in ranks(a)? {
        let form = match a.policy.as_str() {
            "tier" => None,
            "min" => Some(MinForm::Capped),
            "min-pure" => Some(MinForm::Pure),
            "min-global" => Some(MinForm::Global),
            "min-admit-all" => Some(MinForm::CappedAdmitAll),
            p => bail!("--policy {p} (tier | min | min-pure | min-global | min-admit-all)"),
        };
        let o = match form {
            None => replay(
                &a.trace[0],
                rank,
                PER_RANK,
                STRIDE,
                &params(a),
                prof.as_ref(),
                &law,
                ws.as_deref(),
                perm.as_ref(),
            ),
            Some(f) if perm.is_none() => min_replay(
                &a.trace[0],
                rank,
                PER_RANK,
                STRIDE,
                &params(a),
                prof.as_ref(),
                &law,
                ws.as_deref(),
                f,
            ),
            Some(_) => bail!("--policy min* takes no owner permutation"),
        }
        .map_err(anyhow::Error::msg)?;
        all.insert(format!("rank{rank}"), report(&a.label, rank, &o));
        if let Some(lp) = &a.tier_log {
            let want = format!("fntierend rank{rank}: ");
            let text = std::fs::read_to_string(lp)?;
            let line = text
                .lines()
                .find(|l| l.starts_with(&want))
                .context("no fntierend line")?;
            let arr = |key: &str| -> anyhow::Result<Vec<u64>> {
                let at = line.find(key).context("key")? + key.len();
                let end = line[at..].find(']').context("]")? + at;
                line[at..end]
                    .split(',')
                    .map(|x| Ok(x.trim().parse::<u64>()?))
                    .collect()
            };
            let (res, ram) = (arr("resident [")?, arr("ram [")?);
            let bad = o
                .per_layer
                .iter()
                .enumerate()
                .filter(|(l, c)| c.resident != res[*l] || c.ram != ram[*l])
                .count();
            println!(
                "fnreplay SELFTEST rank{rank}: {} ({bad} of {} layers differ from the served tier's end-of-run counters; {} steps replayed)",
                if bad == 0 { "PASS -- the replay == the served tier" } else { "FAIL" },
                o.per_layer.len(),
                o.steps
            );
        }
        if let Some(mp) = &a.metrics {
            let m = metrics_layers(mp, rank)?;
            if m.is_empty() {
                println!(
                    "fnreplay SELFTEST rank{rank}: no per-layer counters for this rank in {}",
                    mp.display()
                );
                continue;
            }
            let mut bad = 0;
            for (l, c) in o.per_layer.iter().enumerate() {
                for (path, v) in [("resident", c.resident), ("ram", c.ram), ("nvme", c.nvme)] {
                    let served = m.get(&(l as u32, path.to_string())).copied().unwrap_or(0);
                    if served != v {
                        bad += 1;
                        if bad <= 5 {
                            println!("fnreplay SELFTEST rank{rank} layer {l} {path}: served {served} replay {v}");
                        }
                    }
                }
            }
            println!(
                "fnreplay SELFTEST rank{rank}: {} ({} of {} per-layer counters differ)",
                if bad == 0 {
                    "PASS -- the replay == the served tier"
                } else {
                    "FAIL"
                },
                bad,
                o.per_layer.len() * 3
            );
        }
    }
    if let Some(out) = &a.out {
        std::fs::create_dir_all(out)?;
        let name = if a.label.is_empty() {
            "replay".to_string()
        } else {
            a.label.clone()
        };
        std::fs::write(
            out.join(format!("replay_{name}.json")),
            serde_json::to_string_pretty(&all)?,
        )?;
    }
    Ok(())
}

/// FN104: the scheduler's signal (derived from the trace's step structure) over FN18's replay.
fn sched_mode(a: &ProfileArgs) -> anyhow::Result<()> {
    use saber_mem::sched_signal::{infer, infer_line, sched_replay, InferOpts, SchedLaw};
    if !a.only.is_empty() || a.windows.is_some() || !a.train.is_empty() {
        bail!("sched: no windows / owner permutation (the stream needs the whole trace)");
    }
    let prof = load_profile(&a.profile)?;
    let ins = InsertLaw::parse(a.max_inserts, &a.by_bucket)
        .and_then(|l| l.with_budget_mib(&a.promote_budget_mib))
        .and_then(|l| l.with_second_chance(&a.second_chance))
        .map_err(anyhow::Error::msg)?;
    let law = SchedLaw::parse(&a.sched).map_err(anyhow::Error::msg)?;
    let annot = infer(
        &a.trace[0],
        &InferOpts {
            max_chunk: a.max_chunk,
            ..InferOpts::default()
        },
    )
    .map_err(anyhow::Error::msg)?;
    println!("{} [{}]", infer_line(&annot), a.label);
    for rank in ranks(a)? {
        let (o, _, lines) = sched_replay(
            &a.trace[0],
            &annot,
            rank,
            PER_RANK,
            STRIDE,
            &params(a),
            prof.as_ref(),
            &ins,
            &law,
        )
        .map_err(anyhow::Error::msg)?;
        report(&a.label, rank, &o);
        for l in lines {
            println!("{l} [{} rank{rank}]", a.label);
        }
    }
    Ok(())
}

fn ceiling(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    let counts = count(a, ws.as_deref())?;
    for kind in ["decode", "prefill"] {
        let mut c = Counts::new(48, 512);
        for ((_, k), x) in &counts {
            if k == kind {
                c.merge(x);
            }
        }
        for rank in ranks(a)? {
            let (all, per) = static_ceiling(&c.occ, rank, PER_RANK, a.vram, 16);
            let worst = per.iter().copied().fold(f64::INFINITY, f64::min);
            println!(
                "fnceiling {} {kind:>7} rank{rank} at {} slots: {:.2} % of routed (layer, expert, step) | per-layer min {:.1} % | per layer {:?}",
                a.label,
                a.vram,
                100.0 * all,
                100.0 * worst,
                per.iter().map(|x| (x * 1000.0).round() / 10.0).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

fn perm_from(a: &ProfileArgs) -> anyhow::Result<Option<Perm>> {
    if a.train.is_empty() {
        return Ok(None);
    }
    let tw = windows(a, &a.train)?;
    let counts = count(a, tw.as_deref())?;
    let mut c = Counts::new(48, 512);
    for ((_, k), x) in &counts {
        if k == "decode" {
            c.merge(x);
        }
    }
    Ok(Some(balanced_owner(&c.occ, PER_RANK, 2)))
}

fn owner(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    let perm = perm_from(a)?.context("owner needs --train windows")?;
    for (name, p) in [("identity", None), ("balanced", Some(&perm))] {
        let (steps, smax, smean) =
            ep_balance(&a.trace[0], PER_RANK, ws.as_deref(), p).map_err(anyhow::Error::msg)?;
        let n = (steps * 48).max(1) as f64;
        println!(
            "fnowner {} {name:>9}: decode steps {steps} | per layer: max-rank distinct {:.3} vs mean {:.3} (+{:.2} %, the wait)",
            a.label,
            smax as f64 / n,
            smean / n,
            100.0 * (smax as f64 / smean.max(1e-9) - 1.0)
        );
    }
    Ok(())
}

fn ple(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    // decode steps in the same windows (the per-step read rate)
    let mut dsteps = 0u64;
    let mut r = TraceReader::open(&a.trace[0])?;
    while let Some(s) = r.next_step()? {
        if s.kind == StepKind::Decode && ws.as_ref().is_none_or(|w| w.iter().any(|w| w.holds(s.t)))
        {
            dsteps += 1;
        }
    }
    for size in a.sizes.split(',') {
        let rows: u32 = size.trim().parse()?;
        let v = ple_replay(&a.trace[0], rows, ws.as_deref(), 2).map_err(anyhow::Error::msg)?;
        for (pass, c) in v.iter().enumerate() {
            println!(
                "fnple {} cache {rows:>8} rows ({:>7.1} MiB) pass {} ({}): lookups {} distinct/gather {} | hits {:.2} % of distinct | drive reads {} ({:.2} % of lookups) | decode: {} reads over {} steps = {:.2} rows/step",
                a.label,
                f64::from(rows) * 320.0 / 1048576.0,
                pass,
                if pass == 0 { "cold" } else { "warm" },
                c[0],
                c[1],
                100.0 * c[2] as f64 / c[1].max(1) as f64,
                c[3],
                100.0 * c[3] as f64 / c[0].max(1) as f64,
                c[4],
                dsteps,
                c[4] as f64 / dsteps.max(1) as f64
            );
        }
    }
    Ok(())
}

/// FN58: a device-resident PLE hot table sized on the trace. `--sizes` = table MiB a rank (rows =
/// MiB / (320 B row + 24 B of open-addressing slots at load 0.5)), `--ple-ram-mib` = the host's
/// RAM row cache behind it, `--xtrain` = the trace the STATIC set is chosen on (held out); the LRU
/// arm needs no training. `--ple-free-ms` = the host PLE time a step with no drive read costs
/// (FN56's per-step census) -- the most a host-free step can save.
fn ple_device(a: &ProfileArgs) -> anyhow::Result<()> {
    use saber_mem::route_trace::{ple_device_replay, ple_hot_rows, PleDevPolicy};
    let ws = windows(a, &a.only)?;
    let ram_rows = ((a.ple_ram_mib << 20) / 320) as u32;
    let l = &a.label;
    let mut sizes: Vec<u64> = vec![0];
    for s in a.sizes.split(',') {
        sizes.push(s.trim().parse()?);
    }
    for mib in sizes {
        let rows = ((mib << 20) / 344) as u32;
        let mut arms = vec![("lru".to_string(), PleDevPolicy::Lru)];
        if let Some(t) = &a.xtrain {
            let set = ple_hot_rows(t, rows as usize).map_err(anyhow::Error::msg)?;
            arms.push(("static-held".to_string(), PleDevPolicy::Static(set)));
        }
        for (name, pol) in &arms {
            let c = ple_device_replay(&a.trace[0], ram_rows, rows, pol, ws.as_deref())
                .map_err(anyhow::Error::msg)?;
            for (cls, x) in ["b1", "b8", "all"].iter().zip(c.iter()) {
                let n = x.steps.max(1) as f64;
                let free = x.host_free as f64 / n;
                println!(
                    "fnpledev {l} dev {mib:>5} MiB ({rows:>8} rows, {:>6.1} expert slots) {name:<11} {cls:>3}: steps {:>6} | rows/step {:6.2} | on-card hit {:6.2} % | host rows/step {:6.2} | drive reads/step {:5.2} | host-free steps {:6.2} % | drive-free steps {:6.2} % | saved <= {:.4} ms/step",
                    (mib << 20) as f64 / STRIDE as f64,
                    x.steps,
                    x.distinct as f64 / n,
                    100.0 * x.dev_hits as f64 / x.distinct.max(1) as f64,
                    (x.distinct - x.dev_hits) as f64 / n,
                    x.drive as f64 / n,
                    100.0 * free,
                    100.0 * x.drive_free as f64 / n,
                    free * a.ple_free_ms
                );
            }
        }
    }
    Ok(())
}

/// FN47a: the step-ahead prefetch study.
fn prefetch(a: &ProfileArgs) -> anyhow::Result<()> {
    use saber_mem::step_ahead::{study, train_xlayer, Cell, LinkModel, Pred};
    let ws = windows(a, &a.only)?;
    let prof = load_profile(&a.profile)?;
    let law = InsertLaw::parse(a.max_inserts, &a.by_bucket)
        .and_then(|l| l.with_budget_mib(&a.promote_budget_mib))
        .and_then(|l| l.with_second_chance(&a.second_chance))
        .map_err(anyhow::Error::msg)?;
    let preds: Vec<Pred> = a
        .preds
        .split(',')
        .map(|p| Pred::parse(p.trim()))
        .collect::<Result<_, _>>()
        .map_err(anyhow::Error::msg)?;
    let m = LinkModel {
        stride: STRIDE,
        link_gbps: a.link_gbps,
        copy_gbps: a.copy_gbps,
        compute_ms: LinkModel::parse_compute(&a.compute_ms).map_err(anyhow::Error::msg)?,
        lead: a.lead,
    };
    let rk = ranks(a)?;
    let xt = match &a.xtrain {
        Some(p) => Some(train_xlayer(p, None, 64).map_err(anyhow::Error::msg)?),
        None => None,
    };
    let st = study(
        &a.trace[0],
        &rk,
        PER_RANK,
        &params(a),
        prof.as_ref(),
        &law,
        ws.as_deref(),
        &preds,
        &m,
        xt.as_ref(),
    )
    .map_err(anyhow::Error::msg)?;
    let l = &a.label;
    println!(
        "fnprefetch {l}: {} decode steps, ranks {:?}, link {} GB/s, copy {} GB/s, lead {}, compute {}",
        st.decode_steps, rk, a.link_gbps, a.copy_gbps, a.lead, a.compute_ms
    );
    for (&(pi, b), c) in &st.cells {
        let bn = if b == 9 {
            "b8full".to_string()
        } else {
            format!("b{b}")
        };
        let rs = c.rank_steps.max(1) as f64;
        let n = c.steps.max(1) as f64;
        println!(
            "fnprefetch {l} {bn:>6} {:<20} steps {:>6} | miss/rank-step {:7.2} | copies {:7.2} | recall {:6.2} % | precision {:6.2} % | step ms today {:6.2} start {:6.2} lead {:6.2} floor {:6.2} | pool start p50/p99/max {}/{}/{} lead {}/{}/{} | late/rank-step {:.2}",
            preds[pi].name(),
            c.steps,
            c.misses as f64 / rs,
            c.copies as f64 / rs,
            100.0 * c.useful as f64 / (c.misses.max(1) as f64),
            100.0 * c.useful as f64 / (c.copies.max(1) as f64),
            c.ms_today / n,
            c.ms_start / n,
            c.ms_lead / n,
            c.ms_compute / n,
            Cell::pct(&c.pool_start, 0.5),
            Cell::pct(&c.pool_start, 0.99),
            Cell::pct(&c.pool_start, 1.0),
            Cell::pct(&c.pool_lead, 0.5),
            Cell::pct(&c.pool_lead, 0.99),
            Cell::pct(&c.pool_lead, 1.0),
            c.late as f64 / rs,
        );
    }
    for (&b, d) in &st.seq {
        let bn = if b == 9 {
            "b8full".to_string()
        } else {
            format!("b{b}")
        };
        let mm = d.misses.max(1) as f64;
        println!(
            "fnprefetch {l} {bn:>6} per-sequence (same-shape steps {}): seq misses {} | same-seq t-1 recall {:.2} % | whole-batch t-1 recall {:.2} %",
            d.steps,
            d.misses,
            100.0 * d.by_seq as f64 / mm,
            100.0 * d.by_batch as f64 / mm
        );
    }
    Ok(())
}

/// FN53a: the drive's time for `n` concurrent record reads (us): FN7's O_DIRECT thread pool
/// measured 386 / 268 / 250 us a 2,617,344-B record at QD 1 / 8 / 32 -> 135 + 251 n (both ranks
/// share the one drive).
fn drive_us(n: u32) -> f64 {
    if n == 0 {
        0.0
    } else {
        135.0 + 251.0 * f64::from(n)
    }
}

/// FN53a: per kind (prefill rows / decode steps) counts of one trace.
fn kind_counts(t: &Path) -> anyhow::Result<(Counts, Counts)> {
    let mut r = TraceReader::open(t)?;
    let (l, e) = (r.layers as usize, r.experts as usize);
    let (mut pre, mut dec) = (Counts::new(l, e), Counts::new(l, e));
    while let Some(s) = r.next_step()? {
        match s.kind {
            StepKind::Prefill => pre.add(&s),
            StepKind::Decode => dec.add(&s),
            _ => {}
        }
    }
    Ok((pre, dec))
}

/// FN53a: (decode step share, prefill row share) of every record of `rank` (`layer * 256 + local`).
fn tail_key(pre: &Counts, dec: &Counts, rank: u32) -> Vec<(f64, f64)> {
    let mut k = Vec::with_capacity(pre.routes.len() * PER_RANK as usize);
    for l in 0..pre.routes.len() {
        for x in 0..PER_RANK {
            let g = (rank * PER_RANK + x) as usize;
            k.push((
                dec.occ[l][g] as f64 / dec.steps.max(1) as f64,
                pre.routes[l][g] as f64 / pre.tokens.max(1) as f64,
            ));
        }
    }
    k
}

/// FN53a: the SSD expert tail study (`flashnext-profile tail`): `--trace` = the replayed (held-out)
/// trace, `--xtrain` = the trace the tail is DEFINED on, `--profile` = the warm start (the
/// defining capture's served profile). Prints (1) the threshold tails and their stability, (2) per
/// tail size the drive reads and their cost on prefill chunks (serialized / one-layer-ahead
/// prefetch of the layer's whole tail) and decode buckets, and the hit split.
fn tail(a: &ProfileArgs) -> anyhow::Result<()> {
    let def = a.xtrain.as_ref().context("--xtrain (the defining trace)")?;
    let (pa, da) = kind_counts(&a.trace[0])?;
    let (pb, db) = kind_counts(def)?;
    let layers = pa.routes.len();
    let label = &a.label;
    println!(
        "fntail {label} replayed: prefill {} rows / decode {} steps | defining: prefill {} rows / decode {} steps",
        pa.tokens, da.steps, pb.tokens, db.steps
    );
    // (1) the threshold tails
    for rank in [0u32, 1] {
        let (ka, kb) = (tail_key(&pa, &da, rank), tail_key(&pb, &db, rank));
        for (what, thr) in [
            ("prefill rows", 0.005),
            ("prefill rows", 0.01),
            ("prefill rows", 0.02),
            ("decode steps", 0.005),
            ("decode steps", 0.01),
        ] {
            let pick = |k: &[(f64, f64)]| -> Vec<bool> {
                k.iter()
                    .map(|&(d, p)| {
                        if what == "decode steps" {
                            d < thr
                        } else {
                            p < thr
                        }
                    })
                    .collect()
            };
            let (sa, sb) = (pick(&ka), pick(&kb));
            let inter = sa.iter().zip(&sb).filter(|(x, y)| **x && **y).count();
            let uni = sa.iter().zip(&sb).filter(|(x, y)| **x || **y).count();
            let na = sa.iter().filter(|x| **x).count();
            let nb = sb.iter().filter(|x| **x).count();
            let mut per: Vec<usize> = (0..layers)
                .map(|l| sb[l * 256..(l + 1) * 256].iter().filter(|x| **x).count())
                .collect();
            per.sort_unstable();
            // held out: the share of the replayed trace's traffic the defining set takes
            let (mut pre_hit, mut dec_hit) = (0u64, 0u64);
            for (i, on) in sb.iter().enumerate() {
                if *on {
                    let (l, g) = (i / 256, rank as usize * 256 + i % 256);
                    pre_hit += pa.routes[l][g];
                    dec_hit += da.occ[l][g];
                }
            }
            let pre_tot: u64 = (0..layers)
                .map(|l| {
                    pa.routes[l][rank as usize * 256..(rank as usize + 1) * 256]
                        .iter()
                        .sum::<u64>()
                })
                .sum();
            let dec_tot: u64 = (0..layers)
                .map(|l| {
                    da.occ[l][rank as usize * 256..(rank as usize + 1) * 256]
                        .iter()
                        .sum::<u64>()
                })
                .sum();
            println!(
                "fntail {label} rank{rank} {what} < {:.1} %: defining {nb} records ({:.2} GiB) | replayed {na} | Jaccard {:.3} | per layer min/med/max {}/{}/{} | held out: the set takes {:.2} % of prefill routes, {:.2} % of decode step-routes",
                thr * 100.0,
                nb as f64 * STRIDE as f64 / (1u64 << 30) as f64,
                inter as f64 / uni.max(1) as f64,
                per[0],
                per[layers / 2],
                per[layers - 1],
                100.0 * pre_hit as f64 / pre_tot.max(1) as f64,
                100.0 * dec_hit as f64 / dec_tot.max(1) as f64
            );
        }
    }
    // (2) the replay per tail size
    let prof = load_profile(&a.profile)?;
    let law = InsertLaw::parse(a.max_inserts, &a.by_bucket)
        .and_then(|l| l.with_budget_mib(&a.promote_budget_mib))
        .and_then(|l| l.with_second_chance(&a.second_chance))
        .map_err(anyhow::Error::msg)?;
    let p = params(a);
    let chunk_ms = a.chunk_ms;
    let layer_us = chunk_ms * 1e3 / layers as f64;
    for n in a
        .tails
        .split(',')
        .map(|x| x.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?
    {
        let mut tails = Vec::new();
        let mut per_layer_tail = vec![0u32; layers];
        for rank in [0u32, 1] {
            let h = saber_mem::route_trace::replay_header(rank, layers as u32, PER_RANK, STRIDE);
            let tier =
                saber_mem::expert_tier::ExpertTier::new(&h, p.clone(), prof.as_ref(), 0, 1 << 40)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            let cap = if a.tail_balanced {
                n.div_ceil(layers)
            } else {
                0
            };
            let t = coldest_ram_tail(&tier, &tail_key(&pb, &db, rank), n, cap);
            for (i, on) in t.iter().enumerate() {
                if *on {
                    per_layer_tail[i / 256] += 1;
                }
            }
            tails.push(t);
        }
        // prefetch of a full chunk: the drive reads layer l+1's whole tail while l computes
        // (two layers buffered); layer 0 is exposed
        let pf_ms = {
            let (mut drive_end, mut comp_start_prev, mut comp_end) = (0.0f64, 0.0f64, 0.0f64);
            for (l, &k) in per_layer_tail.iter().enumerate() {
                let ds = if l == 0 {
                    0.0
                } else {
                    drive_end.max(comp_start_prev)
                };
                drive_end = ds + drive_us(k);
                let cs = comp_end.max(drive_end);
                comp_start_prev = cs;
                comp_end = cs + layer_us;
            }
            (comp_end - chunk_ms * 1e3) / 1e3
        };
        let max_layer = per_layer_tail.iter().copied().max().unwrap_or(0);
        #[derive(Default)]
        struct Acc {
            steps: u64,
            reads: u64,
            ser_us: f64,
            promoted: u64,
            routed_tail: u64,
            worst_us: f64,
            stepped: u64,
        }
        let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
        let outs = tail_replay(
            &a.trace[0],
            &[0, 1],
            PER_RANK,
            STRIDE,
            &p,
            prof.as_ref(),
            &law,
            None,
            &tails,
            |s| {
                let key = match s.bucket {
                    0 if s.rows >= 2048 => "prefill 2k".to_string(),
                    0 if s.rows >= 1024 => "prefill 1k-2k".to_string(),
                    0 if s.rows >= 256 => "prefill 256-1k".to_string(),
                    0 => "prefill <256".to_string(),
                    b => format!("b{b}"),
                };
                let us: f64 = s
                    .reads
                    .iter()
                    .map(|r| drive_us(r.iter().map(|&x| u32::from(x)).sum()))
                    .sum();
                let e = acc.entry(key).or_default();
                e.steps += 1;
                e.reads += s.reads.iter().flatten().map(|&x| u64::from(x)).sum::<u64>();
                e.ser_us += us;
                e.worst_us = e.worst_us.max(us);
                e.stepped += u64::from(us > 0.0);
                e.promoted += s.tail_promoted.iter().map(|&x| u64::from(x)).sum::<u64>();
                e.routed_tail += s.tail_routed.iter().map(|&x| u64::from(x)).sum::<u64>();
            },
        )
        .map_err(anyhow::Error::msg)?;
        println!(
            "fntail {label} T={n}/rank: pinned RAM freed {:.2} GiB (both ranks {:.2}) | tail per layer (both ranks) max {max_layer} | full-chunk tail {:.2} GB = {:.1} ms on the drive at QD32 | one-layer-ahead prefetch of a 2k chunk ({chunk_ms} ms): +{:.2} ms ({:.2} %), drive needs {:.2} GB/s",
            n as f64 * STRIDE as f64 / (1u64 << 30) as f64,
            2.0 * n as f64 * STRIDE as f64 / (1u64 << 30) as f64,
            2.0 * n as f64 * STRIDE as f64 / 1e9,
            per_layer_tail.iter().map(|&k| drive_us(k)).sum::<f64>() / 1e3,
            pf_ms.max(0.0),
            100.0 * pf_ms.max(0.0) / chunk_ms,
            2.0 * n as f64 * STRIDE as f64 / 1e9 / (chunk_ms / 1e3)
        );
        for (k, e) in &acc {
            let m = e.steps.max(1) as f64;
            println!(
                "fntail {label} T={n}/rank {k:>14}: steps {:>6} | tail routed/step {:>7.2} | drive reads/step {:>7.3} ({:>7.1} MB) | serialized +{:>7.3} ms/step (worst {:>6.1} ms; {:.2} % of steps read) | tail promoted/step {:.3}",
                e.steps,
                e.routed_tail as f64 / m,
                e.reads as f64 / m,
                e.reads as f64 / m * STRIDE as f64 / 1e6,
                e.ser_us / m / 1e3,
                e.worst_us / 1e3,
                100.0 * e.stepped as f64 / m,
                e.promoted as f64 / m
            );
        }
        // the hit split per bucket (both ranks): VRAM / RAM / drive
        let mut by: BTreeMap<u32, (u64, u64, u64)> = BTreeMap::new();
        for o in &outs {
            for (b, t) in &o.buckets {
                let x = by.entry(*b).or_default();
                x.0 += t.paths.routed();
                x.1 += t.paths.resident;
                x.2 += t.paths.ram;
            }
        }
        let mut drive_by: BTreeMap<u32, u64> = BTreeMap::new();
        for (k, e) in &acc {
            let b = if k.starts_with("prefill") {
                0
            } else {
                k[1..].parse::<u32>().unwrap_or(0)
            };
            *drive_by.entry(b).or_default() += e.reads;
        }
        for (b, (routed, res, ram)) in &by {
            let d = drive_by.get(b).copied().unwrap_or(0);
            let r = (*routed).max(1) as f64;
            println!(
                "fntail {label} T={n}/rank hits {:>7}: VRAM {:.2} % | RAM {:.2} % | drive {:.3} %",
                if *b == 0 {
                    "prefill".to_string()
                } else {
                    format!("b{b}")
                },
                100.0 * *res as f64 / r,
                100.0 * (*ram - d) as f64 / r,
                100.0 * d as f64 / r
            );
        }
    }
    Ok(())
}

/// FN94: one prefill chunk's per-layer census on one rank.
#[derive(Debug, Clone, Default)]
struct PfLayer {
    routed: u32,
    resident: u32,
    ram: u32,
    /// Per `--census-slots` entry: routed RAM records the profile-ordered ring of that size holds.
    ring_hit: Vec<u32>,
    /// Per entry: records the ring copies (min(slots, RAM records of the layer on this rank)).
    ring_copy: Vec<u32>,
}

/// FN94: the short-chunk census -- every prefill chunk whose rows fall in a `--census-rows` band,
/// replayed through the real tier (both ranks, paired by chunk index): per layer the distinct
/// routed experts, the VRAM hits, the RAM reads (each rank, the slower), the profile-ordered
/// ring's hits / copies at each `--census-slots`, and the chunk timing under FN83's fit
/// (13.8 ms intercept + 5.2 us a VRAM-hit expert + 102 us a RAM read, NOT overlapped) for: in
/// place (today), the ring (copies at `--copy-gbps` on one copy stream a rank, two buffers: layer
/// L's copies start when layer L-2's MoE ended), the ROUTE-EXACT form on layer L's own misses
/// (copied after L's router, on the critical path) and its ORACLE bound (layer L+1's exact misses
/// copied from L's router on, recall 100 %, no link contention).
#[allow(clippy::too_many_lines)]
fn pfcensus(a: &ProfileArgs) -> anyhow::Result<()> {
    use saber_mem::expert_tier::{ExpertTier, Placement};
    let ws = windows(a, &a.only)?;
    let prof = load_profile(&a.profile)?;
    let law = InsertLaw::parse(a.max_inserts, &a.by_bucket).map_err(anyhow::Error::msg)?;
    let bands: Vec<(u32, u32)> = a
        .census_rows
        .split(',')
        .map(|b| {
            let (lo, hi) = b.split_once('-').context("band lo-hi")?;
            Ok((lo.trim().parse()?, hi.trim().parse()?))
        })
        .collect::<anyhow::Result<_>>()?;
    let slots: Vec<usize> = a
        .census_slots
        .split(',')
        .map(|x| Ok(x.trim().parse()?))
        .collect::<anyhow::Result<_>>()?;
    let in_band = |rows: u32| bands.iter().any(|&(lo, hi)| rows >= lo && rows <= hi);
    let mut per_rank: Vec<Vec<(u32, Vec<PfLayer>)>> = Vec::new();
    for rank in [0u32, 1] {
        let mut r = TraceReader::open(&a.trace[0])?;
        let h = saber_mem::route_trace::replay_header(rank, r.layers, PER_RANK, STRIDE);
        let mut tier = ExpertTier::new(&h, params(a), prof.as_ref(), 0, 1 << 40)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut mv = saber_mem::route_trace::CountMover::default();
        let mut chunks = Vec::new();
        while let Some(s) = r.next_step()? {
            if !s.kind.target() {
                continue;
            }
            if let Some(ws) = &ws {
                if !ws.iter().any(|w| w.holds(s.t)) {
                    continue;
                }
            }
            let pf = s.kind == StepKind::Prefill;
            tier.set_max_inserts(law.cap(if pf { 0 } else { s.bucket }));
            tier.begin_step();
            let want = pf && in_band(s.rows);
            let mut layers = Vec::new();
            for l in 0..s.layers() {
                let lu = l as u32;
                let routed: Vec<u32> = s.layer(l).iter().map(|x| u32::from(x.e)).collect();
                let mut cell = PfLayer::default();
                if want {
                    let local: Vec<u32> = routed
                        .iter()
                        .copied()
                        .filter(|&e| tier.index(lu, e).is_some())
                        .collect();
                    cell.routed = local.len() as u32;
                    let order = tier.stage_order(lu);
                    for &n in &slots {
                        let top: std::collections::HashSet<u32> = order
                            .iter()
                            .take(n)
                            .map(|x| x.0 + rank * PER_RANK)
                            .collect();
                        cell.ring_copy.push(top.len() as u32);
                        cell.ring_hit.push(
                            local
                                .iter()
                                .filter(|&&e| {
                                    top.contains(&e)
                                        && matches!(tier.placement(lu, e), Some(Placement::Ram(_)))
                                })
                                .count() as u32,
                        );
                    }
                }
                let c = tier
                    .resolve(lu, &routed, &mut mv)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if want {
                    cell.resident = c.resident as u32;
                    cell.ram = (c.ram + c.nvme) as u32;
                    layers.push(cell);
                }
            }
            tier.refill(&mut mv).map_err(|e| anyhow::anyhow!("{e}"))?;
            tier.end_step(&mut mv).map_err(|e| anyhow::anyhow!("{e}"))?;
            if want {
                chunks.push((s.rows, layers));
            }
        }
        per_rank.push(chunks);
    }
    anyhow::ensure!(
        per_rank[0].len() == per_rank[1].len(),
        "ranks disagree on chunks"
    );
    let nl = per_rank[0].first().map_or(48, |c| c.1.len());
    let icpt = 13.8 / nl as f64;
    let (hit_ms, read_ms) = (0.0052, 0.102);
    let copy_ms = STRIDE as f64 / (a.copy_gbps * 1e9) * 1e3;
    // the non-MoE share of a layer's intercept: the window before the layer's MoE
    let nonmoe = icpt * 0.7;
    let moe_fixed = icpt - nonmoe;
    let rank_moe = |c: &PfLayer, staged: u32| -> f64 {
        hit_ms * f64::from(c.resident + staged) + read_ms * f64::from(c.ram - staged)
    };
    let mut out = serde_json::Map::new();
    for &(lo, hi) in &bands {
        let idx: Vec<usize> = (0..per_rank[0].len())
            .filter(|&i| per_rank[0][i].0 >= lo && per_rank[0][i].0 <= hi)
            .collect();
        let n = idx.len().max(1) as f64;
        println!(
            "== FN94 pfcensus band {lo}-{hi} rows: {} chunks (mean rows {:.1}), vram {} a rank, copy {:.0} us a record",
            idx.len(),
            idx.iter().map(|&i| f64::from(per_rank[0][i].0)).sum::<f64>() / n,
            a.vram,
            copy_ms * 1e3
        );
        println!(
            "layer | routed r0 r1 | RAM r0 r1 slower | link ms | compute ms | ring hits/copies (slower rank) at {slots:?}"
        );
        let mut tot = [0f64; 4];
        let mut layer_rows = Vec::new();
        for l in 0..nl {
            let mut acc = [0f64; 6];
            let mut rh = vec![0f64; slots.len()];
            let mut rc = vec![0f64; slots.len()];
            for &i in &idx {
                let (c0, c1) = (&per_rank[0][i].1[l], &per_rank[1][i].1[l]);
                let slow = if c0.ram >= c1.ram { c0 } else { c1 };
                acc[0] += f64::from(c0.routed);
                acc[1] += f64::from(c1.routed);
                acc[2] += f64::from(c0.ram);
                acc[3] += f64::from(c1.ram);
                acc[4] += f64::from(slow.ram);
                let comp = |c: &PfLayer| icpt + hit_ms * f64::from(c.resident);
                acc[5] += comp(c0).max(comp(c1));
                for k in 0..slots.len() {
                    rh[k] += f64::from(slow.ring_hit[k]);
                    rc[k] += f64::from(slow.ring_copy[k]);
                }
            }
            let m: Vec<f64> = acc.iter().map(|x| x / n).collect();
            let link = m[4] * read_ms;
            println!(
                "L{l:02} | {:.1} {:.1} | {:.1} {:.1} {:.1} | {:.2} | {:.2} | {}",
                m[0],
                m[1],
                m[2],
                m[3],
                m[4],
                link,
                m[5],
                rh.iter()
                    .zip(&rc)
                    .map(|(h, c)| format!("{:.1}/{:.1}", h / n, c / n))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            tot[0] += m[4];
            tot[1] += link;
            tot[2] += m[5];
            tot[3] += m[0].max(m[1]);
            layer_rows.push(serde_json::json!({"layer": l, "routed": [m[0], m[1]],
                "ram": [m[2], m[3]], "ram_slower": m[4], "link_ms": link, "compute_ms": m[5],
                "ring_hit": rh.iter().map(|x| x / n).collect::<Vec<_>>(),
                "ring_copy": rc.iter().map(|x| x / n).collect::<Vec<_>>()}));
        }
        let mut inplace = 0.0;
        let mut ring = vec![0.0f64; slots.len()];
        let mut bown = 0.0;
        let mut bnext = 0.0;
        for &i in &idx {
            let (ch0, ch1) = (&per_rank[0][i].1, &per_rank[1][i].1);
            for l in 0..nl {
                inplace += icpt + rank_moe(&ch0[l], 0).max(rank_moe(&ch1[l], 0));
                let own = |c: &PfLayer| {
                    copy_ms * f64::from(c.ram) + hit_ms * f64::from(c.resident + c.ram)
                };
                bown += icpt + own(&ch0[l]).max(own(&ch1[l]));
            }
            for (k, acc) in ring.iter_mut().enumerate() {
                let mut free = [0f64; 2];
                let mut moe_end = vec![0f64; nl];
                let mut t = 0.0f64;
                for l in 0..nl {
                    let mut ready = 0.0f64;
                    for (r, c) in [&ch0[l], &ch1[l]].into_iter().enumerate() {
                        let start = free[r].max(if l >= 2 { moe_end[l - 2] } else { 0.0 });
                        free[r] = start + copy_ms * f64::from(c.ring_copy[k]);
                        ready = ready.max(free[r]);
                    }
                    let ms = (t + nonmoe).max(ready);
                    let body = rank_moe(&ch0[l], ch0[l].ring_hit[k])
                        .max(rank_moe(&ch1[l], ch1[l].ring_hit[k]));
                    moe_end[l] = ms + moe_fixed + body;
                    t = moe_end[l];
                }
                *acc += t;
            }
            // oracle: layer L+1's exact misses copied from L's router (= L's MoE start) on
            let mut free = [0f64; 2];
            let mut t = 0.0f64;
            for l in 0..nl {
                let mut ready = 0.0f64;
                for (r, c) in [&ch0[l], &ch1[l]].into_iter().enumerate() {
                    let start = if l == 0 { t + nonmoe } else { free[r] };
                    free[r] = start + copy_ms * f64::from(c.ram);
                    ready = ready.max(free[r]);
                }
                let ms = (t + nonmoe).max(ready);
                for f in &mut free {
                    *f = f.max(ms);
                }
                let body = (hit_ms * f64::from(ch0[l].resident + ch0[l].ram))
                    .max(hit_ms * f64::from(ch1[l].resident + ch1[l].ram));
                t = ms + moe_fixed + body;
            }
            bnext += t;
        }
        println!(
            "chunk (mean): RAM reads slower rank {:.0} = link {:.1} ms in place, compute {:.1} ms, max-rank distinct routed {:.0}",
            tot[0], tot[1], tot[2], tot[3]
        );
        println!("arm in place (today, FN83 fit): {:.1} ms", inplace / n);
        for (k, s) in slots.iter().enumerate() {
            println!(
                "arm A ring {s} slots a half (profile order, 2 buffers): {:.1} ms",
                ring[k] / n
            );
        }
        println!(
            "arm B own layer (copy after L's router, on the critical path): {:.1} ms",
            bown / n
        );
        println!(
            "arm B next-layer ORACLE (L+1's exact misses copied under L, recall 100 %, no contention): {:.1} ms",
            bnext / n
        );
        out.insert(
            format!("{lo}-{hi}"),
            serde_json::json!({"chunks": idx.len(), "layers": layer_rows,
                "inplace_ms": inplace / n,
                "ring_ms": slots.iter().zip(&ring).map(|(s, x)| (s.to_string(), x / n)).collect::<BTreeMap<_, _>>(),
                "b_own_ms": bown / n, "b_next_oracle_ms": bnext / n,
                "ram_slower": tot[0], "link_ms": tot[1], "compute_ms": tot[2]}),
        );
    }
    if let Some(o) = &a.out {
        std::fs::create_dir_all(o)?;
        let name = if a.label.is_empty() {
            "pfcensus".to_string()
        } else {
            a.label.clone()
        };
        std::fs::write(
            o.join(format!("{name}.json")),
            serde_json::to_string_pretty(&out)?,
        )?;
    }
    Ok(())
}
