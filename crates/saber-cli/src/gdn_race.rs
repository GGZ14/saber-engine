//! `saber gdn-race` — T9's repeat-and-compare harness for the GDN prefill path, on ONE card.
//!
//! T8 named the fault's place (LEDGER T8): three events in 336 `Lmo` boots, each first moving ONE
//! GDN layer's RECURRENT state after a prefill chunk (p384 twice, p256 once, one of them
//! rank-local) while that layer's conv tail and every earlier layer are the modal bytes. The same
//! inputs giving a different answer is a race (or a hardware-level non-determinism), and a boot is
//! a slow way to sample it: this command runs the SAME prefill chunk over and over on fixed inputs
//! and diffs every op's output bytes against the first run.
//!
//! **What runs is the engine's own plan, not a re-derivation.** The body is planned exactly as
//! `saber engine` plans a prefill rung (`PlanOptions::BODY`, the rank's embedding window, the
//! batch-invariant attention, the mid-chunk / A-tiled-mid / skinny-gate rules the binary's defaults
//! bind, the f16 GDN state unless `--state f32`), over the first `--layers` layers of a TP=2 rank's
//! pack, at a `--rows` rung with `--live` live rows (`retarget_prefill_rows`, the chunk's `cu`) and
//! its recurrent state pointed at slot 1 (`retarget_prefill_state`). The one difference is the
//! all-reduce: a single process has no peer, so `layer.allreduce` is the identity alias
//! (`Collectives::Identity`) -- every GDN op is rank-local, which is what T8's boot 82 says the
//! fault is.
//!
//! **Per rep:** the recurrent + conv state and the K/V cache are zeroed (what admission does),
//! then the plan runs op by op, and after each op (a carrier and the ops fused into it) the
//! T6 digest kernel (`saber_race_digest_regions`, order-free, on the card) digests every arena
//! range and state range the op binds; one read-back a rep. `--digest end` is the undisturbed arm:
//! the whole chunk back to back, then one digest of every state buffer and the whole arena.
//! A rep whose record differs from rep 0's names its FIRST differing op and binding -- the op
//! whose inputs were identical and whose output was not.
//!
//! An INSTRUMENT: no serve path reads it, and it writes nothing but its own report.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use saber_exec::layer::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, Collectives, PlanOptions, Route,
    RuntimeParams, Source, StateAt,
};
use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, StreamFlags};
use saber_kernels::race_digest::{self, PARTS};
use saber_kernels::rules::{LaunchRules, Shape};
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Bucket, Geometry, Graph, ModelConfig, Phase};

use crate::layer::{pack_file, to_device};

/// Everything the command takes.
pub struct GdnRaceArgs {
    /// The device-final tier pack (directory or `rank<r>.pack`).
    pub pack: PathBuf,
    /// The pack's TP world size (2 = the served TP=2 rank shapes).
    pub world: u32,
    /// TP rank of the pack.
    pub rank: u32,
    /// Plan layers `0..layers`.
    pub layers: u32,
    /// The prefill rung (bucket rows).
    pub rows: u32,
    /// The chunk's live rows (`<= rows`).
    pub live: u32,
    /// A JSON id array or a `.jsonl` corpus; `None` = deterministic synthetic ids.
    pub prompt_ids: Option<PathBuf>,
    /// Corpus row.
    pub prompt_index: usize,
    /// Reps (rep 0 is the reference).
    pub reps: u64,
    /// `f16` (the default) or `f32` GDN state.
    pub state: String,
    /// `step` (a digest after every op) or `end` (undisturbed, one digest a rep).
    pub digest: String,
    /// Print a progress line every this many reps.
    pub every: u64,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// Device buffers budget.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// A label for the report lines.
    pub label: String,
    /// Idle milliseconds before every rep (the load step a boot's first request sees).
    pub gap_ms: u64,
    /// NEGATIVE CONTROL: one non-tail token id changed on every odd rep.
    pub control: bool,
}

/// One digested group: a carrier step and the ops fused into it.
struct Group {
    /// `steps[lo..hi]`.
    lo: usize,
    hi: usize,
    /// The carrier op's name.
    name: String,
    /// (binding name, address, bytes).
    regions: Vec<(String, DeviceAddr, u64)>,
    /// Offset of this group's partials in the record, in u64 words.
    rec_word: usize,
    /// Offset of this group's table in the descriptor buffer, in bytes.
    desc_off: u64,
}

fn synthetic_ids(n: u32, seed: u64) -> Vec<i32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // ordinary text ids: below the 151,643 special tokens
            (x % 150_000) as i32 + 100
        })
        .collect()
}

/// Run the harness. Returns the number of reps whose record differed from rep 0's.
#[allow(clippy::too_many_lines)]
pub fn run(hip: &HipThread, args: &GdnRaceArgs) -> anyhow::Result<u64> {
    hip.set_device(args.device)?;
    if args.live == 0 || args.live > args.rows {
        anyhow::bail!("--live {} must be in 1..={}", args.live, args.rows);
    }
    let f16 = match args.state.as_str() {
        "f16" => true,
        "f32" => false,
        s => anyhow::bail!("--state {s}: f16 or f32"),
    };
    let step_digest = match args.digest.as_str() {
        "step" => true,
        "end" => false,
        s => anyhow::bail!("--digest {s}: step or end"),
    };
    let g = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), args.world, 8)?;
    let graph = if f16 { g.with_gdn_state_f16()? } else { g };
    // The engine's prefill rules at the binary's defaults (G5 mid-chunk, G6 A-tiled mid band and
    // the skinny gate; `saber_engine::device::run`).
    let mut mxfp4 = LaunchRules::prod_mxfp4()?;
    mxfp4.knobs.folded_mt = true;
    mxfp4.knobs.atiled_mid = true;
    mxfp4.knobs.skinny = true;
    let saber = SaberRules::prod_saber()?;

    let layers: Vec<u32> = (0..args.layers).collect();
    let params = RuntimeParams {
        max_ctx: args.rows,
        gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
    };
    let kv_blocks = args.rows.div_ceil(saber_exec::forward::KV_BLOCK).max(1);
    let opts = PlanOptions::BODY
        .rank(args.rank)
        .with_collectives(Collectives::Identity)
        .with_batch_invariant_attn()
        .with_table_blocks(kv_blocks);
    let mut plan = BlockPlan::with_options(
        &graph,
        Bucket::Prefill {
            tokens: args.rows,
            seqs: 1,
        },
        &layers,
        params,
        &mxfp4,
        &saber,
        opts,
    )?;
    plan.retarget_prefill_rows(args.live, &saber)?;
    plan.retarget_prefill_state(saber_exec::forward::FIRST_STATE_SLOT)?;

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    // ---- the weights: the plan's window of the rank's pack --------------------------------
    let want: BTreeSet<String> = plan.weight_keys();
    let reader = PackReader::open(&pack_file(&args.pack, args.rank))?;
    let mut layout_of: BTreeMap<String, u32> = BTreeMap::new();
    let mut idx: Vec<usize> = Vec::new();
    for (i, rec) in reader.records().iter().enumerate() {
        let name = reader.name(i);
        if want.contains(name) {
            idx.push(i);
            layout_of.insert(name.to_owned(), rec.layout_id);
        }
    }
    if layout_of.len() != want.len() {
        anyhow::bail!(
            "{} of {} records the plan binds are not in {}",
            want.len() - layout_of.len(),
            want.len(),
            args.pack.display()
        );
    }
    let (lo, hi) = (idx[0], *idx.last().expect("non-empty"));
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    let mut budget = args.budget_mib << 20;
    if span > budget {
        anyhow::bail!("the weight window is {} MiB", span >> 20);
    }
    budget -= span;
    let cfg = UploadConfig {
        start_record: lo,
        limit_bytes: Some(span),
        mode: OpenMode::Direct,
        ..UploadConfig::default()
    };
    let upload = Upload::open(&args.pack, args.rank, cfg)?;
    let uplan = upload.plan();
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(args.rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut weights: BTreeMap<String, BoundWeight> = BTreeMap::new();
    upload.run(&mut sink, &ring, |r| {
        if let Some(layout) = layout_of.get(r.name) {
            weights.insert(r.name.to_string(), BoundWeight::new(r.addr, *layout));
        }
    })?;
    if weights.len() != want.len() {
        anyhow::bail!("{} record(s) did not land", want.len() - weights.len());
    }

    // ---- state: every GDN slot pool + the K/V cache, owned here ---------------------------
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights,
        ..BlockInputs::default()
    };
    let mut state_bytes: BTreeMap<String, (u64, Option<u64>)> = BTreeMap::new();
    let mut state_ptrs: Vec<usize> = Vec::new();
    for step in &plan.steps {
        for st in &step.op.state {
            if inputs.states.contains_key(&st.name) {
                continue;
            }
            let bytes = match st.bytes_per_slot {
                Some(b) => b * u64::from(params.gdn_slots),
                None => match step.op.geometry {
                    Geometry::Attn {
                        head_dim,
                        block_size,
                        kv_heads,
                        ..
                    } => {
                        u64::from(kv_blocks)
                            * u64::from(kv_heads)
                            * u64::from(block_size)
                            * 2
                            * u64::from(head_dim)
                    }
                    _ => anyhow::bail!("state `{}` has no size", st.name),
                },
            };
            if bytes > budget {
                anyhow::bail!("state `{}` exceeds the budget", st.name);
            }
            budget -= bytes;
            let p = hip.malloc(bytes.max(1))?;
            inputs.states.insert(st.name.clone(), p.addr());
            state_bytes.insert(st.name.clone(), (bytes, st.bytes_per_slot));
            state_ptrs.push(owned.len());
            owned.push(p);
        }
    }

    // ---- the index tables: one sequence, `live` rows of a `rows`-row chunk, slot 1 --------
    let ids: Vec<i32> = match &args.prompt_ids {
        Some(p) => {
            let mut v = crate::forward::read_prompt_ids(p, args.prompt_index)?;
            if v.len() < args.live as usize {
                anyhow::bail!(
                    "prompt row {} has {} ids, fewer than --live {}",
                    args.prompt_index,
                    v.len(),
                    args.live
                );
            }
            v.truncate(args.live as usize);
            v
        }
        None => synthetic_ids(args.live, 0x7439),
    };
    let mut padded = ids.clone();
    padded.resize(args.rows as usize, *ids.last().unwrap_or(&0));
    let blocks: Vec<u32> = (0..kv_blocks).collect();
    let seqs = [saber_exec::forward::SeqFacts {
        state_slot: saber_exec::forward::FIRST_STATE_SLOT,
        done_before: 0,
        kv_blocks: &blocks,
        live_rows: args.live,
        rope: None,
    }];
    let step_idx = saber_exec::forward::StepIndex {
        phase: Phase::Prefill,
        rows: args.rows,
        max_blocks: kv_blocks,
        seqs: &seqs,
        ids: &padded,
        band_width: None,
    };
    step_idx.check().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut control_ids: Vec<(String, DeviceAddr, DeviceAddr)> = Vec::new();
    for key in plan.index_keys() {
        let bytes = saber_exec::forward::index_table_for(
            &key,
            &step_idx,
            saber_exec::forward::Accepted::AllRows,
            saber_exec::forward::AttnQLen::Exact(args.live),
        )?;
        let p = to_device(hip, &stream, &bytes, &mut budget)?;
        inputs.indices.insert(key.clone(), p.addr());
        if args.control && key.ends_with(".ids") {
            // THE NEGATIVE CONTROL: the same table with ONE non-tail id changed, swapped in on
            // every odd rep -- the harness must report every one of those reps.
            let mut alt = padded.clone();
            let at = (args.live as usize) / 3;
            alt[at] = if alt[at] == 1000 { 1001 } else { 1000 };
            let q = to_device(hip, &stream, &crate::forward::i32_bytes(&alt), &mut budget)?;
            control_ids.push((key.clone(), p.addr(), q.addr()));
            owned.push(q);
        }
        owned.push(p);
    }

    if plan.arena_bytes > budget {
        anyhow::bail!("the arena is {} MiB", plan.arena_bytes >> 20);
    }
    let mut exec = BlockExecutor::new(hip, &plan, 8 << 20)?;
    exec.zero(hip, &stream)?;

    // ---- the digest groups ------------------------------------------------------------------
    let state_size = |key: &str, at: StateAt| -> u64 {
        let (whole, per) = state_bytes.get(key).copied().unwrap_or((0, None));
        match (at, per) {
            (StateAt::Slot { .. }, Some(p)) => p,
            _ => whole,
        }
    };
    let mut groups: Vec<Group> = Vec::new();
    let mut words = 0usize;
    let mut desc_bytes: Vec<u8> = Vec::new();
    let mut push_group =
        |lo: usize, hi: usize, name: String, regions: Vec<(String, DeviceAddr, u64)>| {
            let g = Group {
                lo,
                hi,
                name,
                rec_word: words,
                desc_off: desc_bytes.len() as u64,
                regions,
            };
            for (_, a, b) in &g.regions {
                desc_bytes.extend_from_slice(&a.as_u64().to_le_bytes());
                desc_bytes.extend_from_slice(&b.to_le_bytes());
            }
            words += (g.regions.len() + 1) * PARTS as usize;
            groups.push(g);
        };
    let n_steps = plan.steps.len();
    if step_digest {
        let mut i = 0usize;
        while i < n_steps {
            let mut j = i + 1;
            while j < n_steps
                && matches!(&plan.steps[j].route,
                    Route::Fused { into, .. } if *into == plan.steps[i].op.name)
            {
                j += 1;
            }
            let mut regions: Vec<(String, DeviceAddr, u64)> = Vec::new();
            let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();
            for s in &plan.steps[i..j] {
                for (bname, src) in &s.bindings {
                    let (a, b) = match src {
                        Source::Arena { bytes, .. } => (exec.address(src, &inputs)?, *bytes),
                        Source::State { key, at } => {
                            (exec.address(src, &inputs)?, state_size(key, *at))
                        }
                        _ => continue,
                    };
                    if b == 0 || !seen.insert((a.as_u64(), b)) {
                        continue;
                    }
                    regions.push((bname.clone(), a, b));
                }
            }
            if matches!(plan.steps[i].route, Route::Kernel { .. }) && !regions.is_empty() {
                push_group(i, j, plan.steps[i].op.name.to_string(), regions);
            } else {
                push_group(i, j, plan.steps[i].op.name.to_string(), Vec::new());
            }
            i = j;
        }
    } else {
        let mut regions: Vec<(String, DeviceAddr, u64)> = Vec::new();
        for (k, (bytes, _)) in &state_bytes {
            regions.push((k.clone(), inputs.states[k], *bytes));
        }
        regions.push(("arena".to_owned(), exec.arena_base(), plan.arena_bytes));
        push_group(0, n_steps, "chunk".to_owned(), regions);
    }
    let desc = to_device(hip, &stream, &desc_bytes, &mut budget)?;
    let rec = hip.malloc((words * 8).max(8) as u64)?;
    let launches: Vec<saber_kernels::rules::LaunchDescriptor> = groups
        .iter()
        .map(|g| race_digest::resolve_regions(g.regions.len() as u32, 0))
        .collect::<Result<_, _>>()?;
    let digested: u64 = groups
        .iter()
        .flat_map(|g| g.regions.iter().map(|r| r.2))
        .sum();
    println!(
        "gdnrace   {} tp{} rank {} layers 0..{} p{} live {} state {} digest {} -- {} steps, {} \
         groups, {} MiB digested a rep, arena {} MiB",
        args.label,
        args.world,
        args.rank,
        args.layers,
        args.rows,
        args.live,
        args.state,
        args.digest,
        n_steps,
        groups.len(),
        digested >> 20,
        plan.arena_bytes >> 20
    );
    let gdn_ops: BTreeMap<&str, u32> = plan
        .steps
        .iter()
        .filter_map(|s| match &s.route {
            Route::Kernel { descs, .. } => descs.first().map(|d| d.kernel.as_str()),
            _ => None,
        })
        .filter(|k| k.contains("gdn") || k.contains("skinny"))
        .fold(BTreeMap::new(), |mut m, k| {
            *m.entry(k).or_default() += 1;
            m
        });
    for (k, n) in &gdn_ops {
        println!("kernel    {k} x{n}");
    }

    // ---- the reps -----------------------------------------------------------------------------
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new("gdn-race", &counters);
    let mut reference: Vec<u8> = Vec::new();
    let mut got = vec![0u8; words * 8];
    let mut first_by_group: BTreeMap<String, u64> = BTreeMap::new();
    let mut diff_reps = 0u64;
    let t0 = Instant::now();
    for rep in 0..args.reps {
        if args.gap_ms > 0 {
            // an idle card before the chunk: the load step a boot's first request sees
            std::thread::sleep(std::time::Duration::from_millis(args.gap_ms));
        }
        for (key, base, alt) in &control_ids {
            let use_alt = rep % 2 == 1;
            inputs
                .indices
                .insert(key.clone(), if use_alt { *alt } else { *base });
        }
        for &i in &state_ptrs {
            hip.memset(&owned[i], 0, &stream)?;
        }
        for (g, d) in groups.iter().zip(&launches) {
            exec.run_steps(
                &ctx,
                &mut launcher,
                &stream,
                &plan.steps[g.lo..g.hi],
                &inputs,
                false,
            )?;
            if g.regions.is_empty() {
                continue;
            }
            let a = ArgBindings::new(Shape::new(0, 0, 0))
                .with(
                    "desc",
                    desc.addr()
                        .offset(g.desc_off)
                        .ok_or_else(|| anyhow::anyhow!("desc offset"))?,
                )
                .with("extra", desc.addr())
                .with(
                    "out",
                    rec.addr()
                        .offset(g.rec_word as u64 * 8)
                        .ok_or_else(|| anyhow::anyhow!("rec offset"))?,
                );
            launcher.launch_op(&ctx, &[d], &a, &stream)?;
        }
        hip.memcpy_d2h_blocking(&mut got, &rec, &stream)?;
        if rep == 0 {
            reference.clone_from(&got);
        } else if got != reference {
            diff_reps += 1;
            // the first group, then the first region in it, whose partials differ
            'find: for g in &groups {
                for (ri, (bname, _, bytes)) in g.regions.iter().enumerate() {
                    let o = (g.rec_word + ri * PARTS as usize) * 8;
                    let e = o + PARTS as usize * 8;
                    if got[o..e] != reference[o..e] {
                        let key = format!("{} {}", g.name, bname);
                        println!(
                            "DIFF      {} rep {rep}: first differing op `{}` binding `{bname}` \
                             ({bytes} B)",
                            args.label, g.name
                        );
                        *first_by_group.entry(key).or_default() += 1;
                        break 'find;
                    }
                }
            }
        }
        if args.every > 0 && (rep + 1) % args.every == 0 {
            println!(
                "progress  {} {} reps, {} differing, {:.1} ms/rep",
                args.label,
                rep + 1,
                diff_reps,
                t0.elapsed().as_secs_f64() * 1000.0 / (rep + 1) as f64
            );
        }
    }
    println!(
        "result    {} tp{} p{} live {} state {} digest {} layers {}: {} reps, {} differing from \
         rep 0 ({:.1} ms/rep)",
        args.label,
        args.world,
        args.rows,
        args.live,
        args.state,
        args.digest,
        args.layers,
        args.reps,
        diff_reps,
        t0.elapsed().as_secs_f64() * 1000.0 / args.reps.max(1) as f64
    );
    for (k, n) in &first_by_group {
        println!("firstdiff {} {k}: {n}", args.label);
    }
    drop(exec);
    drop(owned);
    drop(sink);
    Ok(diff_reps)
}
