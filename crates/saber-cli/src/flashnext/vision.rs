//! FN55: Flash-Next's vision tower as an ADDED pack member (the pack is never rewritten in
//! place). `saber flashnext-pack --vision bf16` copies the checkpoint's 333 `model.visual.*`
//! records verbatim (bf16, checkpoint shapes, replicated) into `vision/rank<r>.pack` (one
//! `saber_mem::pack` file per rank: the header carries the rank, the upload checks it) plus
//! `vision/vision.json` (the member's entry: records, sha256, bytes, source, `out_hidden`).
//!
//! The tower is the 27B's ARCHITECTURE (`vision_config` equal key for key but the merger's
//! `out_hidden_size` 2,560) with Flash-Next's OWN weights: the two checkpoints' visual tensors
//! differ byte for byte (measured FN55: `blocks.0.attn.qkv.weight` sha256 `fa403b68..` vs the 27B's
//! `b8c86101..`), so the 27B's tower is an ARITHMETIC reference (same kernels, same rules but one
//! width), never a byte reference for Flash-Next's rows. The rules are
//! `saber_kernels::vision::VisionRules::flashnext()`; the engine boots the member with
//! `saber_engine::vision::boot_vision_with` (`saber-vision --rules flashnext --pack <pack>/vision`).

use std::path::Path;

use anyhow::{ensure, Context};
use saber_kernels::manifest::sha256_hex;
use saber_kernels::vision::VisionRules;
use saber_mem::pack::writer::{write_pack, PackItem};
use saber_mem::pack::{Dtype, ShardKind, LAYOUT_RAW};
use serde_json::json;

use super::store::Ckpt;

/// The member's directory in the pack.
pub const DIR: &str = "vision";
/// The member's entry.
pub const ENTRY: &str = "vision.json";
/// Ranks the member is written for (EP=2 / TP=2: the tower is replicated on both).
pub const RANKS: u32 = 2;

/// One member record: name, checkpoint shape, bf16 bytes.
pub type Record = (String, Vec<u64>, Vec<u8>);

/// The member's records in the rules' order.
///
/// # Errors
/// A record missing from the checkpoint, not bf16, or a count off the rules'.
pub fn records(ck: &Ckpt, rules: &VisionRules) -> anyhow::Result<Vec<Record>> {
    let names = rules.records();
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        let t = ck.loc(&n)?;
        let b = ck.bf16(&n)?;
        out.push((n, t.shape, b));
    }
    if let Some((_, s, _)) = out
        .iter()
        .find(|(n, _, _)| n == "model.visual.merger.linear_fc2.weight")
    {
        ensure!(
            s.first().copied() == Some(u64::from(rules.model.out_hidden)),
            "merger.linear_fc2.weight {s:?}: the rules' out_hidden is {}",
            rules.model.out_hidden
        );
    }
    Ok(out)
}

/// Write the member under `pack` (refuses if any of its files exists). Returns the bytes of one
/// rank's file.
///
/// # Errors
/// An existing member, the checkpoint, or the writes.
pub fn build(pack: &Path, ckpt: &Path) -> anyhow::Result<u64> {
    let rules = VisionRules::flashnext()?;
    let dir = pack.join(DIR);
    let files: Vec<_> = (0..RANKS)
        .map(|r| dir.join(format!("rank{r}.pack")))
        .chain(std::iter::once(dir.join(ENTRY)))
        .collect();
    for f in &files {
        ensure!(
            !f.exists(),
            "{}: the vision member exists (never rewritten in place)",
            f.display()
        );
    }
    let ck = Ckpt::open(ckpt)?;
    let recs = records(&ck, &rules)?;
    std::fs::create_dir_all(&dir)?;
    let items: Vec<PackItem> = recs
        .iter()
        .map(|(n, s, b)| PackItem {
            name: n.clone(),
            dtype: Dtype::BF16,
            shape: s.clone(),
            shard_kind: ShardKind::Replicated,
            layout_id: LAYOUT_RAW,
            data: b.clone(),
        })
        .collect();
    let mut file_bytes = 0u64;
    for r in 0..RANKS {
        let f = dir.join(format!("rank{r}.pack"));
        let tmp = dir.join(format!("rank{r}.pack.tmp"));
        file_bytes = write_pack(&tmp, RANKS, r, [0u8; 32], [0u8; 32], &items)
            .with_context(|| tmp.display().to_string())?;
        std::fs::rename(&tmp, &f)?;
    }
    let payload: u64 = recs.iter().map(|(_, _, b)| b.len() as u64).sum();
    let doc = json!({
        "member": DIR, "class": "vision", "fn": "FN55",
        "rules": rules.name, "out_hidden": rules.model.out_hidden,
        "source": ckpt, "dtype": "bf16", "replicated": true,
        "records": recs.iter().map(|(n, s, b)| json!({
            "name": n, "shape": s, "length": b.len(), "sha256": sha256_hex(b)})).collect::<Vec<_>>(),
        "payload_bytes": payload, "file_bytes_per_rank": file_bytes,
        "files": (0..RANKS).map(|r| format!("{DIR}/rank{r}.pack")).collect::<Vec<_>>(),
    });
    std::fs::write(dir.join(ENTRY), serde_json::to_vec_pretty(&doc)?)?;
    println!(
        "flashnext-pack vision: {} records, payload {payload} B, {file_bytes} B a rank -> {}",
        recs.len(),
        dir.display()
    );
    Ok(file_bytes)
}

// =================================================================================== FN55b
// The ENGINE half: `saber engine --fn-vision`. The tower is resident on EACH rank (replicated: the
// residual stream is replicated at TP = 2, so both ranks need the rows; the 27B's D2 residency
// argument), billed to the ledger as `fn.vision` before the expert slots are sized.

/// The ledger class of the resident tower (weights + arena).
pub const CLASS: &str = "fn.vision";

/// The VRAM the tower bills on `rank`: (the weight window's bytes, the arena's at `max_patches`),
/// read off the member's own pack table (no HIP).
///
/// # Errors
/// No vision member in the pack (`saber flashnext-pack --vision bf16` first), or the rules.
pub fn demand(pack: &Path, rank: u32, max_patches: u32) -> anyhow::Result<(u64, u64)> {
    let rules = VisionRules::flashnext()?;
    let f = pack.join(DIR).join(format!("rank{rank}.pack"));
    ensure!(
        f.exists(),
        "--fn-vision: {} is missing (build the member with `saber flashnext-pack --vision bf16`)",
        f.display()
    );
    let rd = saber_mem::pack::PackReader::open(&f).with_context(|| f.display().to_string())?;
    let (w, a) = saber_engine::vision::vision_demand(&rd, &rules, max_patches)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((w.span, a))
}

/// FN55c: the REGION a lazy tower (`--fn-vision lazy`) holds on `rank`: (the weight upload's
/// arena, the tower's arena at `max_patches`) -- allocated at boot, lent to the expert tier, the
/// tower uploaded into it on the first image (host only: the pack's table).
///
/// # Errors
/// As [`demand`].
pub fn region_bytes(pack: &Path, rank: u32, max_patches: u32) -> anyhow::Result<(u64, u64)> {
    let rules = VisionRules::flashnext()?;
    let f = pack.join(DIR).join(format!("rank{rank}.pack"));
    ensure!(
        f.exists(),
        "--fn-vision: {} is missing (build the member with `saber flashnext-pack --vision bf16`)",
        f.display()
    );
    saber_engine::vision::vision_region_bytes(&f, rank, RANKS, max_patches, &rules)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// One sequence's image facts as the pair's protocol carries them (rank 0 builds them from the
/// expanded prompt + the request's grids; both ranks rebuild the same M-RoPE table): the prompt
/// length, each image span `[start, tokens, grid_h, grid_w, src_row]`, the tower run (epoch)
/// whose rows the prompt scatters.
///
/// FN125 (`--fn-image-cache on`): + each image's content hash (`ImageBatch::content_hashes`, in
/// the spans' order -- the key the prefix tree / prompt cache folds into every piece the image's
/// run overlaps) and the DEFERRED tower run (`defer` = the pending pixels' ticket, `epoch` 0):
/// the tower runs only when a prefill chunk that is COMPUTED (not reloaded) touches an image
/// run. Both fields are absent from the frame when empty / 0 (FN55b's frame byte for byte).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImgFacts {
    pub prompt_len: u32,
    pub spans: Vec<[u32; 5]>,
    pub epoch: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hashes: Vec<u64>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub defer: u64,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(x: &u64) -> bool {
    *x == 0
}

impl ImgFacts {
    /// FN125: [`Self::from_prompt`] + the images' content hashes (one per grid, in order) and the
    /// deferred tower's ticket (0 = the tower already ran: `epoch`).
    ///
    /// # Errors
    /// As [`Self::from_prompt`], or a hash count off the grids'.
    pub fn keyed(
        ids: &[u32],
        grids: &[[u32; 3]],
        epoch: u64,
        hashes: &[u64],
        defer: u64,
    ) -> anyhow::Result<ImgFacts> {
        ensure!(
            hashes.is_empty() || hashes.len() == grids.len(),
            "{} content hashes for {} images",
            hashes.len(),
            grids.len()
        );
        let mut f = Self::from_prompt(ids, grids, epoch)?;
        f.hashes = hashes.to_vec();
        f.defer = defer;
        Ok(f)
    }

    /// FN125: the prefix keys' image runs (`saber_sched::prefix_key::ImageKey`, prompt order);
    /// empty when the facts carry no hashes (the pre-FN125 frame: such a prompt keeps the
    /// FN55b bypass).
    #[must_use]
    pub fn image_keys(&self) -> Vec<saber_sched::prefix_key::ImageKey> {
        if self.hashes.len() != self.spans.len() {
            return Vec::new();
        }
        self.spans
            .iter()
            .zip(&self.hashes)
            .map(|(s, &hash)| saber_sched::prefix_key::ImageKey {
                start: s[0],
                tokens: s[1],
                hash,
            })
            .collect()
    }

    /// From the expanded prompt `ids` and the request's grids.
    ///
    /// # Errors
    /// A prompt whose `<|image_pad|>` runs the grids do not account for (by name).
    pub fn from_prompt(ids: &[u32], grids: &[[u32; 3]], epoch: u64) -> anyhow::Result<ImgFacts> {
        let spans = saber_exec::mrope::image_spans(ids, saber_vision_input::IMAGE_PAD_ID, grids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(ImgFacts {
            prompt_len: ids.len() as u32,
            spans: spans
                .iter()
                .map(|s| [s.start, s.tokens, s.grid_h, s.grid_w, s.src_row])
                .collect(),
            epoch,
            hashes: Vec::new(),
            defer: 0,
        })
    }

    /// The sequence's M-RoPE table (the 27B's law: `saber_exec::mrope`).
    #[must_use]
    pub fn rope(&self) -> saber_exec::mrope::MropePositions {
        let spans: Vec<saber_exec::mrope::ImageSpan> = self
            .spans
            .iter()
            .map(|s| saber_exec::mrope::ImageSpan {
                start: s[0],
                tokens: s[1],
                grid_h: s[2],
                grid_w: s[3],
                src_row: s[4],
            })
            .collect();
        saber_exec::mrope::mrope_positions(self.prompt_len, &spans)
    }
}

/// A sequence's image state on a rank: its M-RoPE table and the tower run its rows came from.
/// FN125: + the prefix keys' image runs (empty = no hashes: the FN55b bypass) and the deferred
/// tower's ticket (`epoch` 0 until the run, [`Self::ran`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImgSeq {
    pub rope: saber_exec::mrope::MropePositions,
    pub epoch: u64,
    pub keys: Vec<saber_sched::prefix_key::ImageKey>,
    pub defer: u64,
}

impl ImgSeq {
    #[must_use]
    pub fn new(f: &ImgFacts) -> ImgSeq {
        ImgSeq {
            rope: f.rope(),
            epoch: f.epoch,
            keys: f.image_keys(),
            defer: f.defer,
        }
    }

    /// FN125: whether a COMPUTED prefill over prompt positions `lo..hi` needs the deferred tower
    /// first: the tower has not run for this sequence and an image run overlaps the range (the
    /// drafter's prompt rows embed tokens inside the same chunk, so the chunk's range is the
    /// whole read set).
    #[must_use]
    pub fn tower_due(&self, lo: u32, hi: u32) -> bool {
        self.epoch == 0
            && self.defer != 0
            && self
                .rope
                .scatter_runs(lo, hi.saturating_sub(lo))
                .iter()
                .any(|r| r.rows > 0)
    }

    /// FN125: this sequence after its deferred tower ran as run `epoch`.
    #[must_use]
    pub fn ran(&self, epoch: u64) -> ImgSeq {
        ImgSeq {
            epoch,
            defer: 0,
            ..self.clone()
        }
    }
}

/// FN126: THE PER-IMAGE TOWER-RUN LAW. A request's images run through the tower in GROUPS of
/// consecutive images (prompt order), each group as large as fits the arena (greedy: an image
/// joins the open group while the group's patches stay <= `max_patches`, else it opens the
/// next) -- so only the LARGEST SINGLE IMAGE must fit the arena, never the request's sum. A
/// request whose images fit together is ONE group = the one-batch run every request took before
/// FN126 (its rows byte for byte). The tower is per-image math (per-row GEMMs at one tile class
/// whatever M, per-row norms, attention inside each image's `cu_seqlens` segment, per-image
/// position / rope tables: `fn126_a_two_image_request_plans_as_its_images`), so a group's rows
/// are the rows the same images get in any other group.
///
/// # Errors
/// An image of more patches than the arena alone (by name; after FN126's clamp the serve plane
/// never sends one).
pub fn tower_groups(
    grids: &[[u32; 3]],
    max_patches: u32,
) -> Result<Vec<std::ops::Range<usize>>, String> {
    let cap = u64::from(max_patches);
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    let mut open = 0u64;
    for (i, g) in grids.iter().enumerate() {
        let p = u64::from(g[0]) * u64::from(g[1]) * u64::from(g[2]);
        if p > cap {
            return Err(format!(
                "vision: image {i} of {p} patches ({}x{}) exceeds this engine's tower arena of \
                 {max_patches} patches (`--fn-vision-max-patches`; the serve plane clamps an \
                 image to the arena, FN126)",
                g[1], g[2]
            ));
        }
        match out.last_mut() {
            Some(r) if open + p <= cap => {
                r.end = i + 1;
                open += p;
            }
            _ => {
                out.push(i..i + 1);
                open = p;
            }
        }
    }
    Ok(out)
}

/// FN125: the tower's host refusals for one image set, WITHOUT the tower (a deferred run must
/// refuse a request at encode time exactly where the run would have: the patch cap, the plan,
/// the pixel count) -- `max_patches` = the boot's tower arena. FN126: the cap is PER IMAGE
/// ([`tower_groups`]); every group's plan is checked.
///
/// # Errors
/// The refusal's text.
pub fn tower_precheck(grids: &[[u32; 3]], pixels: usize, max_patches: u32) -> Result<(), String> {
    let rules = VisionRules::flashnext().map_err(|e| format!("{e:#}"))?;
    let g: Vec<saber_kernels::vision::Grid> = grids
        .iter()
        .map(|x| saber_kernels::vision::Grid {
            t: x[0],
            h: x[1],
            w: x[2],
        })
        .collect();
    let groups = tower_groups(grids, max_patches)?;
    if groups.is_empty() {
        // the planner's own refusal ("no image grid")
        saber_kernels::vision::plan_tower(&rules, &g).map_err(|e| format!("{e}"))?;
    }
    let mut patches = 0usize;
    for r in groups {
        let plan = saber_kernels::vision::plan_tower(&rules, &g[r]).map_err(|e| format!("{e}"))?;
        patches += plan.patches as usize;
    }
    let m = &rules.model;
    let kin = (m.in_channels * m.temporal_patch * m.patch * m.patch) as usize;
    let want = patches * kin;
    if pixels != want {
        return Err(format!(
            "vision: {pixels} pixel values for {patches} patches x {kin} (want {want})"
        ));
    }
    Ok(())
}

/// The `[3, rows]` i64 position table of rows at token indices `idx` (row-major by axis), each
/// row through `rope` when the row's sequence has one, else the index on every axis (the text
/// law, byte for byte).
#[must_use]
pub fn pos3_of(rows: &[(Option<&saber_exec::mrope::MropePositions>, u32)]) -> Vec<i64> {
    let m = rows.len();
    let mut v = vec![0i64; 3 * m];
    for (r, (rope, i)) in rows.iter().enumerate() {
        let p = rope.map_or([i64::from(*i); 3], |x| x.at(*i));
        v[r] = p[0];
        v[m + r] = p[1];
        v[2 * m + r] = p[2];
    }
    v
}

/// The resident tower on a rank: the member's records + arena, the `vision` plugin's functions,
/// the launcher, the last run (its epoch and where its rows are).
pub struct FnVision<'h> {
    pub tier: saber_engine::vision::VisionTier<'h>,
    table: saber_exec::loader::FunctionTable<'h>,
    launcher: saber_exec::Launcher,
    /// The tower run whose rows the arena holds (0 = none yet).
    pub epoch: u64,
    out: Option<saber_exec::vision::VisionOut>,
    dump: Option<std::path::PathBuf>,
    rank: u32,
    /// Runs, tokens produced, rows scattered, device ms of the runs.
    pub runs: u64,
    pub tokens: u64,
    pub scattered: u64,
    pub ms: f64,
    /// FN126: a request SPLIT into tower groups ([`tower_groups`]) leaves its rows HERE (each
    /// group's run downloaded in prompt order; the arena holds one group at a time): `(rows,
    /// tokens)` of the current epoch, scattered by a host -> card copy. `None` = the one-group
    /// run, its rows in the arena's `out` slot (every request before FN126).
    host: Option<(Vec<u8>, u32)>,
    /// `pixel_values` row width (3 x T x p x p = 1,536).
    kin: usize,
    /// FN126 (the which-path counters of the split route): requests split, the group runs they
    /// took, rows scattered from the host copy.
    pub split: u64,
    pub group_runs: u64,
    pub host_scattered: u64,
}

impl std::fmt::Debug for FnVision<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FnVision({:?}, epoch {})", self.tier, self.epoch)
    }
}

/// Bytes of one tower output row (`[tokens, 2560]` bf16).
pub const ROW_BYTES: u64 = 2560 * 2;

impl<'h> FnVision<'h> {
    /// Upload the member's `rank` file and build the tower's executor.
    ///
    /// # Errors
    /// The member, the upload, the arena.
    pub fn boot(
        hip: &'h saber_hip::HipThread,
        table: saber_exec::loader::FunctionTable<'h>,
        pack: &Path,
        rank: u32,
        max_patches: u32,
        dump: Option<&Path>,
    ) -> anyhow::Result<FnVision<'h>> {
        Self::boot_in(hip, table, pack, rank, max_patches, dump, None, 0)
    }

    /// FN55c: [`Self::boot`] into `regions` (weights, arena) when given -- the lazy tower's
    /// region ([`region_bytes`]) -- with the tower-run counter continuing at `epoch` (a reload
    /// never reuses an earlier run's number).
    ///
    /// # Errors
    /// As [`Self::boot`], or a region smaller than its plan.
    #[allow(clippy::too_many_arguments)]
    pub fn boot_in(
        hip: &'h saber_hip::HipThread,
        table: saber_exec::loader::FunctionTable<'h>,
        pack: &Path,
        rank: u32,
        max_patches: u32,
        dump: Option<&Path>,
        regions: Option<(
            std::rc::Rc<saber_hip::DevicePtr>,
            std::rc::Rc<saber_hip::DevicePtr>,
        )>,
        epoch: u64,
    ) -> anyhow::Result<FnVision<'h>> {
        let rules = VisionRules::flashnext()?;
        ensure!(
            u64::from(rules.model.out_hidden) * 2 == ROW_BYTES,
            "the tower's rows are {} wide, the trunk's 2,560",
            rules.model.out_hidden
        );
        let f = pack.join(DIR).join(format!("rank{rank}.pack"));
        let kin = (rules.model.in_channels
            * rules.model.temporal_patch
            * rules.model.patch
            * rules.model.patch) as usize;
        let tier =
            saber_engine::vision::boot_vision_in(hip, &f, rank, RANKS, max_patches, rules, regions)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        let counters = saber_exec::LaunchCounters::new();
        Ok(FnVision {
            tier,
            table,
            launcher: saber_exec::Launcher::new(format!("fnvision.r{rank}"), &counters),
            epoch,
            out: None,
            dump: dump.map(Path::to_path_buf),
            rank,
            runs: 0,
            tokens: 0,
            scattered: 0,
            ms: 0.0,
            host: None,
            kin,
            split: 0,
            group_runs: 0,
            host_scattered: 0,
        })
    }

    /// Run the tower over one request's images (eager: the graph replays the same launches, D2's
    /// gate) on `stream`, synchronously; the rows stay in the arena under the new epoch. The epoch
    /// moves on whatever the answer (a failed run may have written part of the out slot).
    ///
    /// FN126: the images run in [`tower_groups`] -- one group (the images fit the arena together)
    /// is the run above, unchanged; several groups run one after another and each group's rows
    /// are downloaded, in prompt order, to the host copy the scatter reads ([`Self::scatter`]).
    ///
    /// # Errors
    /// The plan (an IMAGE past the arena), the upload, the launches.
    pub fn encode(
        &mut self,
        hip: &saber_hip::HipThread,
        stream: &saber_hip::Stream,
        grids: &[[u32; 3]],
        pixels: &[f32],
    ) -> anyhow::Result<u32> {
        self.epoch += 1;
        self.out = None;
        self.host = None;
        let groups = tower_groups(grids, self.tier.exec.layout().max_patches)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if groups.len() > 1 {
            return self.encode_split(hip, stream, grids, pixels, &groups);
        }
        let g: Vec<saber_kernels::vision::Grid> = grids
            .iter()
            .map(|x| saber_kernels::vision::Grid {
                t: x[0],
                h: x[1],
                w: x[2],
            })
            .collect();
        let t0 = std::time::Instant::now();
        let ctx = saber_exec::LaunchCtx::new(hip, &self.table);
        let o = self
            .tier
            .exec
            .run(&ctx, &mut self.launcher, stream, &g, pixels, false)
            .map_err(|e| anyhow::anyhow!("vision: {e}"))?;
        hip.sync_stream(stream)?;
        self.ms += t0.elapsed().as_secs_f64() * 1e3;
        self.runs += 1;
        self.tokens += u64::from(o.tokens);
        if let Some(d) = &self.dump {
            let d = d.join(format!("rank{}", self.rank));
            std::fs::create_dir_all(&d)?;
            let mut rows = vec![0u8; o.bytes as usize];
            self.tier
                .exec
                .download(hip, stream, o.offset, &mut rows)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            std::fs::write(d.join(format!("enc{}.bin", self.epoch)), rows)?;
            std::fs::write(
                d.join(format!("enc{}.json", self.epoch)),
                serde_json::to_vec(&json!({"grids": grids, "tokens": o.tokens}))?,
            )?;
        }
        self.out = Some(o);
        Ok(o.tokens)
    }

    /// FN126: [`Self::encode`] over SEVERAL tower groups: each group planned, uploaded and run
    /// alone (the arena holds one), its rows downloaded behind the previous group's; the request's
    /// rows end in prompt order in the host copy -- the same layout as one run's `out` slot.
    fn encode_split(
        &mut self,
        hip: &saber_hip::HipThread,
        stream: &saber_hip::Stream,
        grids: &[[u32; 3]],
        pixels: &[f32],
        groups: &[std::ops::Range<usize>],
    ) -> anyhow::Result<u32> {
        let g: Vec<saber_kernels::vision::Grid> = grids
            .iter()
            .map(|x| saber_kernels::vision::Grid {
                t: x[0],
                h: x[1],
                w: x[2],
            })
            .collect();
        let all: usize = g.iter().map(|x| x.patches() as usize).sum();
        ensure!(
            pixels.len() == all * self.kin,
            "vision: {} pixel values for {all} patches x {} (want {})",
            pixels.len(),
            self.kin,
            all * self.kin
        );
        let t0 = std::time::Instant::now();
        let ctx = saber_exec::LaunchCtx::new(hip, &self.table);
        let mut rows: Vec<u8> = Vec::new();
        let mut tokens = 0u32;
        let mut px0 = 0usize;
        for r in groups {
            let gg = &g[r.clone()];
            let n = gg.iter().map(|x| x.patches() as usize).sum::<usize>() * self.kin;
            let o = self
                .tier
                .exec
                .run(
                    &ctx,
                    &mut self.launcher,
                    stream,
                    gg,
                    &pixels[px0..px0 + n],
                    false,
                )
                .map_err(|e| anyhow::anyhow!("vision: {e}"))?;
            hip.sync_stream(stream)?;
            let at = rows.len();
            rows.resize(at + o.bytes as usize, 0);
            self.tier
                .exec
                .download(hip, stream, o.offset, &mut rows[at..])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            px0 += n;
            tokens += o.tokens;
            self.group_runs += 1;
        }
        ensure!(
            rows.len() as u64 == u64::from(tokens) * ROW_BYTES,
            "vision: the split run's rows are {} B for {tokens} tokens",
            rows.len()
        );
        self.ms += t0.elapsed().as_secs_f64() * 1e3;
        self.runs += 1;
        self.split += 1;
        self.tokens += u64::from(tokens);
        if let Some(d) = &self.dump {
            let d = d.join(format!("rank{}", self.rank));
            std::fs::create_dir_all(&d)?;
            std::fs::write(d.join(format!("enc{}.bin", self.epoch)), &rows)?;
            std::fs::write(
                d.join(format!("enc{}.json", self.epoch)),
                serde_json::to_vec(&json!({"grids": grids, "tokens": tokens,
                    "groups": groups.iter().map(|r| [r.start, r.end]).collect::<Vec<_>>()}))?,
            )?;
        }
        self.host = Some((rows, tokens));
        Ok(tokens)
    }

    /// FN125: whether the arena holds tower run `epoch`'s rows (what [`Self::scatter`] checks).
    /// FN126: or the host copy of a split run does.
    #[must_use]
    pub fn holds(&self, epoch: u64) -> bool {
        self.epoch == epoch && (self.out.is_some() || self.host.is_some())
    }

    /// Copy the rows of `runs` (tower run `epoch`) into `dst` at byte `dst_off` (+ `dst_row` rows
    /// of [`ROW_BYTES`]) -- the prompt's embedding rows, before layer 0's HC expansion.
    ///
    /// # Errors
    /// The arena holds another run (refused by name, never a wrong row), a run past the tower's
    /// output, or the copies.
    pub fn scatter(
        &mut self,
        hip: &saber_hip::HipThread,
        stream: &saber_hip::Stream,
        epoch: u64,
        runs: &[saber_exec::mrope::ScatterRun],
        dst: &saber_hip::DevicePtr,
        dst_off: u64,
    ) -> anyhow::Result<()> {
        if runs.is_empty() {
            return Ok(());
        }
        // FN126: a split run's rows come from its host copy (the arena held one group at a time)
        if let Some((rows, tokens)) = self.host.as_ref().filter(|_| self.epoch == epoch) {
            for r in runs {
                ensure!(
                    r.src_row + r.rows <= *tokens,
                    "vision: rows {}..{} past the split run's {tokens} tokens",
                    r.src_row,
                    r.src_row + r.rows
                );
                let a = (u64::from(r.src_row) * ROW_BYTES) as usize;
                let n = (u64::from(r.rows) * ROW_BYTES) as usize;
                hip.memcpy_h2d_at_blocking(
                    dst,
                    dst_off + u64::from(r.dst_row) * ROW_BYTES,
                    &rows[a..a + n],
                    stream,
                )?;
                self.scattered += u64::from(r.rows);
                self.host_scattered += u64::from(r.rows);
            }
            return Ok(());
        }
        let o = self.out.filter(|_| self.epoch == epoch).with_context(|| {
            format!(
                "vision: this chunk scatters tower run {epoch} and the arena holds run {} -- \
                     another request's images overwrote it before this prompt was in",
                self.epoch
            )
        })?;
        for r in runs {
            ensure!(
                r.src_row + r.rows <= o.tokens,
                "vision: rows {}..{} past the run's {} tokens",
                r.src_row,
                r.src_row + r.rows,
                o.tokens
            );
            hip.memcpy_d2d_blocking(
                dst,
                dst_off + u64::from(r.dst_row) * ROW_BYTES,
                self.tier.exec.arena(),
                o.offset + u64::from(r.src_row) * ROW_BYTES,
                u64::from(r.rows) * ROW_BYTES,
                stream,
            )?;
            self.scattered += u64::from(r.rows);
        }
        Ok(())
    }

    /// FN55c: the tower leaves (the lazy release): its function table back (the plugin set
    /// outlives every load), the last tower-run number and the counters `(runs, tokens,
    /// scattered, ms)`; the region's memory returns to its owner when the executor and the
    /// upload sink drop here (the sink drains its stream first).
    #[must_use]
    pub fn release(
        self,
    ) -> (
        saber_exec::loader::FunctionTable<'h>,
        u64,
        (u64, u64, u64, f64),
    ) {
        (
            self.table,
            self.epoch,
            (self.runs, self.tokens, self.scattered, self.ms),
        )
    }

    /// The census line.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "fnvision rank{}: {} tower runs, {} image tokens, {} rows scattered, {:.1} ms in the \
             tower; weights {} B, arena {} B ({} patches) | FN126 split runs {} ({} group runs, {} \
             rows scattered from the host copy)",
            self.rank,
            self.runs,
            self.tokens,
            self.scattered,
            self.ms,
            self.tier.weight_bytes,
            self.tier.exec.arena_bytes(),
            self.tier.exec.layout().max_patches,
            self.split,
            self.group_runs,
            self.host_scattered
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN55b: the protocol's image facts rebuild the 27B's M-RoPE table exactly (the D1 image's
    /// 192 tokens span 16 positions: delta -176), a text index reads the identity law, and
    /// `pos3_of` lays rows out `[3, m]` by axis.
    #[test]
    fn fn55b_image_facts_rebuild_the_mrope_table() {
        let pad = saber_vision_input::IMAGE_PAD_ID;
        let mut ids: Vec<u32> = vec![1, 2, 3];
        ids.extend(std::iter::repeat_n(pad, 192));
        ids.extend([4, 5]);
        let f = ImgFacts::from_prompt(&ids, &[[1, 24, 32]], 7).unwrap();
        let back: ImgFacts = serde_json::from_slice(&serde_json::to_vec(&f).unwrap()).unwrap();
        assert_eq!(back, f);
        let want = saber_exec::mrope::for_prompt(&ids, pad, &[[1, 24, 32]]).unwrap();
        let s = ImgSeq::new(&back);
        assert_eq!(s.rope, want);
        assert_eq!(s.epoch, 7);
        assert_eq!(s.rope.delta, -176);
        // image token 13 = merged (r 0, c 13): (3, 3, 16)
        assert_eq!(s.rope.at(3 + 13), [3, 3, 16]);
        let p = pos3_of(&[(Some(&s.rope), 3 + 13), (None, 9), (Some(&s.rope), 300)]);
        assert_eq!(p, vec![3, 9, 124, 3, 9, 124, 16, 9, 124]);
        // a text-only table is three equal rows (the pre-FN55b law)
        let t = pos3_of(&[(None, 4), (None, 5)]);
        assert_eq!(t, vec![4, 5, 4, 5, 4, 5]);
        // a prompt whose runs the grids do not account for is refused by name
        assert!(ImgFacts::from_prompt(&ids, &[[1, 24, 30]], 7).is_err());
        // FN55b's protocol: a TEXT sequence's `Zero` frame is the pre-FN55b bytes
        let z = super::super::serve::Cmd::Zero {
            slot: 3,
            tokens: 4096,
            img: None,
        };
        assert_eq!(
            String::from_utf8(super::super::serve::enc(&z).unwrap()).unwrap(),
            r#"{"Zero":{"slot":3,"tokens":4096}}"#
        );
        let zi = super::super::serve::Cmd::Zero {
            slot: 3,
            tokens: 4096,
            img: Some(f.clone()),
        };
        assert_eq!(
            super::super::serve::dec(&super::super::serve::enc(&zi).unwrap()).unwrap(),
            zi
        );
    }

    /// FN125: facts without hashes are FN55b's frame byte for byte (no `hashes` / `defer` key)
    /// and keep the bypass (no image keys); keyed facts round-trip, carry one key per image in
    /// prompt order, and a hash count off the grids' is refused by name.
    #[test]
    fn fn125_keyed_facts_ride_the_frame_and_unkeyed_ones_are_fn55bs() {
        let pad = saber_vision_input::IMAGE_PAD_ID;
        let mut ids: Vec<u32> = vec![1, 2, 3];
        ids.extend(std::iter::repeat_n(pad, 192));
        ids.extend([4, 5]);
        ids.extend(std::iter::repeat_n(pad, 16));
        let grids = [[1, 24, 32], [1, 8, 8]];
        let f = ImgFacts::from_prompt(&ids, &grids, 7).unwrap();
        let j = String::from_utf8(serde_json::to_vec(&f).unwrap()).unwrap();
        assert!(!j.contains("hashes") && !j.contains("defer"), "{j}");
        assert!(f.image_keys().is_empty());
        assert!(ImgSeq::new(&f).keys.is_empty());
        let k = ImgFacts::keyed(&ids, &grids, 0, &[u64::MAX, 5], 3).unwrap();
        let back: ImgFacts = serde_json::from_slice(&serde_json::to_vec(&k).unwrap()).unwrap();
        assert_eq!(back, k, "u64 hashes survive the JSON frame exactly");
        let keys = back.image_keys();
        assert_eq!(
            keys,
            vec![
                saber_sched::prefix_key::ImageKey {
                    start: 3,
                    tokens: 192,
                    hash: u64::MAX
                },
                saber_sched::prefix_key::ImageKey {
                    start: 197,
                    tokens: 16,
                    hash: 5
                }
            ]
        );
        let s = ImgSeq::new(&back);
        assert_eq!((s.epoch, s.defer), (0, 3));
        assert!(s.tower_due(195, 200) && !s.tower_due(195, 197) && !s.tower_due(0, 3));
        assert_eq!(s.ran(9).epoch, 9);
        assert!(ImgFacts::keyed(&ids, &grids, 0, &[1], 3).is_err());
    }

    /// FN125: the deferred encode refuses where the tower's run would (the patch cap, the pixel
    /// count, a bad grid), on the host, without the tower.
    #[test]
    fn fn125_the_tower_precheck_refuses_where_the_run_would() {
        let g = [[1, 24, 32]];
        let px = 24 * 32 * 1536;
        assert!(tower_precheck(&g, px, 16_384).is_ok());
        let cap = tower_precheck(&g, px, 512).unwrap_err();
        // FN126: the cap is per image (the text names the image and the arena)
        assert!(
            cap.contains("image 0 of 768 patches (24x32) exceeds this engine's tower arena"),
            "{cap}"
        );
        let short = tower_precheck(&g, px - 1, 16_384).unwrap_err();
        assert!(short.contains("pixel values"), "{short}");
        assert!(tower_precheck(&[[1, 24, 31]], 24 * 31 * 1536, 16_384).is_err());
    }

    /// FN126: the per-image tower-run law. Images that fit the arena together are ONE group (the
    /// pre-FN126 one-batch run); past it they run in consecutive greedy groups, so only the
    /// largest single image must fit: FN125 h0's agent request (5 x 1280x704 = 17,600 patches),
    /// the brief's 5 x 3,072 + 12,000 and the user's 31,440-patch requests are answered at the
    /// served 16,384 arena; a single image past the arena is still refused, by the arena's name
    /// (the HTTP plane's `arena` count reads that text).
    #[test]
    fn fn126_the_tower_groups_law() {
        let a = 16_384u32;
        let s1280 = [1, 44, 80]; // 1280 x 704 = 3,520 patches
        let s1024 = [1, 48, 64]; // 1024 x 768 = 3,072 patches
        assert_eq!(tower_groups(&[s1024; 5], a).unwrap(), vec![0..5]);
        assert_eq!(tower_groups(&[s1280; 5], a).unwrap(), vec![0..4, 4..5]);
        let big = [1, 100, 120]; // 12,000 patches
        let req = [s1024, s1024, s1024, s1024, s1024, big];
        assert_eq!(tower_groups(&req, a).unwrap(), vec![0..5, 5..6]);
        // greedy in prompt order: a small image after a big one joins it when it fits
        assert_eq!(
            tower_groups(&[big, s1024, s1024], a).unwrap(),
            vec![0..2, 2..3]
        );
        // the user's 31,440 patches: 2 x 15,720 (an 8 MP pair downscaled to the arena)
        let half = [1, 60, 262];
        assert_eq!(half[1] * half[2], 15_720);
        assert_eq!(tower_groups(&[half, half], a).unwrap(), vec![0..1, 1..2]);
        assert!(tower_groups(&[], a).unwrap().is_empty());
        // exactly the arena is one group; one patch more is refused by name
        assert_eq!(tower_groups(&[[1, 128, 128]], a).unwrap(), vec![0..1]);
        let e = tower_groups(&[s1024, [1, 128, 130]], a).unwrap_err();
        assert!(
            e.contains("image 1 of 16640 patches")
                && e.contains(saber_serve::images::ARENA_REFUSAL_MARK),
            "{e}"
        );
        // the precheck: a summed request past the arena now passes, an image past it does not
        let px = |g: &[[u32; 3]]| g.iter().map(|x| (x[1] * x[2]) as usize).sum::<usize>() * 1536;
        assert!(tower_precheck(&req, px(&req), a).is_ok());
        assert!(tower_precheck(&[s1280; 5], px(&[s1280; 5]), a).is_ok());
        assert!(tower_precheck(&[half, half], px(&[half, half]), a).is_ok());
        let one = [[1, 128, 130]];
        assert!(tower_precheck(&one, px(&one), a)
            .unwrap_err()
            .contains(saber_serve::images::ARENA_REFUSAL_MARK));
        assert!(tower_precheck(&req, px(&req) - 1536, a)
            .unwrap_err()
            .contains("pixel values"));
        // a bad grid inside a later group is the planner's refusal
        assert!(tower_precheck(&[big, [1, 24, 31]], px(&[big, [1, 24, 31]]), a).is_err());
        // the boot line prints the budget
        let l = super::super::serve::budget_line(16_384);
        assert!(
            l.contains("pixel budget 4194304 px") && l.contains("CLAMPED"),
            "{l}"
        );
        assert!(super::super::serve::budget_line(65_536).contains("prod's own"));
    }

    /// FN126: a two-image request's batch run and its per-image runs are the SAME kernels on the
    /// same per-row inputs -- every launch the same symbol (the GEMM's one tile class at any M),
    /// block and kernel-argument layout; the attention runs each image in its own `cu_seqlens`
    /// segment; the pixel rows, the position-embedding gather (per corner) and the 2-D rope rows
    /// of the batch are the images' own, concatenated -- so a group's rows are the one-batch
    /// run's rows (the card's bytes: `tools/gates/fn126_hold.sh`, the dumps of the forced split
    /// vs the one-batch run).
    #[test]
    fn fn126_a_two_image_request_plans_as_its_images() {
        use saber_kernels::vision::{plan_tower, Grid};
        let r = VisionRules::flashnext().unwrap();
        let a = Grid { t: 1, h: 48, w: 64 };
        let b = Grid { t: 1, h: 44, w: 80 };
        let both = plan_tower(&r, &[a, b]).unwrap();
        let pa = plan_tower(&r, &[a]).unwrap();
        let pb = plan_tower(&r, &[b]).unwrap();
        assert_eq!(both.launches.len(), pa.launches.len());
        assert_eq!(both.launches.len(), pb.launches.len());
        assert_eq!(both.tokens, pa.tokens + pb.tokens);
        for ((x, y), z) in both.launches.iter().zip(&pa.launches).zip(&pb.launches) {
            assert_eq!((&x.op, &x.kernel), (&y.op, &y.kernel));
            assert_eq!((&x.op, &x.kernel), (&z.op, &z.kernel));
            assert_eq!((x.block, x.kernarg_bytes), (y.block, y.kernarg_bytes));
            assert_eq!((x.block, x.kernarg_bytes), (z.block, z.kernarg_bytes));
            if x.op.ends_with(".attn") {
                // one segment per image, x over the longest image's query tiles
                assert_eq!(x.grid[2], 2);
                assert_eq!((y.grid[2], z.grid[2]), (1, 1));
                assert_eq!(x.grid[0], y.grid[0].max(z.grid[0]));
            }
        }
        assert_eq!(both.cu_seqlens, vec![0, 3072, 3072 + 3520]);
        // the host inputs: the batch's tables are the images' own, concatenated
        let pix = |n: u32, s: f32| -> Vec<f32> {
            (0..n as usize * 1536)
                .map(|i| s + (i % 977) as f32 / 977.0)
                .collect()
        };
        let (xa, xb) = (pix(3072, 0.0), pix(3520, 1.0));
        let xab: Vec<f32> = xa.iter().chain(&xb).copied().collect();
        let hb = saber_exec::vision::host_inputs(&r, &both, &xab).unwrap();
        let ha = saber_exec::vision::host_inputs(&r, &pa, &xa).unwrap();
        let hz = saber_exec::vision::host_inputs(&r, &pb, &xb).unwrap();
        let get = |h: &[(&'static str, Vec<u8>)], n: &str| {
            h.iter().find(|(k, _)| *k == n).unwrap().1.clone()
        };
        for row_major in ["rope_cos", "rope_sin", "pixels"] {
            let cat: Vec<u8> = get(&ha, row_major)
                .into_iter()
                .chain(get(&hz, row_major))
                .collect();
            assert_eq!(get(&hb, row_major), cat, "{row_major}");
        }
        // pos tables are [4 corners][L] 4-byte values: per corner, image a's rows then b's
        for corner_major in ["pos_idx", "pos_wts"] {
            let (bb, ab, zb) = (
                get(&hb, corner_major),
                get(&ha, corner_major),
                get(&hz, corner_major),
            );
            let (la, lb) = (3072 * 4, 3520 * 4);
            for c in 0..4 {
                let lab = la + lb;
                assert_eq!(&bb[c * lab..c * lab + la], &ab[c * la..(c + 1) * la]);
                assert_eq!(&bb[c * lab + la..(c + 1) * lab], &zb[c * lb..(c + 1) * lb]);
            }
        }
    }

    /// FN55b: the ledger class off the served pack's member (when this host has it): weights
    /// ~0.84 GiB + the arena at 16,384 patches (the 27B's 0.253 GiB at the same cap).
    #[test]
    fn fn55b_the_vision_class_of_the_served_pack() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        if !pack.join(DIR).join("rank0.pack").exists() {
            eprintln!("skipped: no vision member on this host");
            return;
        }
        for r in 0..RANKS {
            let (w, a) = demand(pack, r, 16_384).unwrap();
            assert!((897_862_112..898_367_488 + (1 << 20)).contains(&w), "{w}");
            assert!(a > 0 && a < 300 << 20, "{a}");
        }
        assert!(demand(pack, 0, 1 << 30).is_err());
    }

    /// The builder on a scratch checkpoint carrying the 333 records at toy sizes is refused
    /// twice: once for a wrong merger width (the 27B's 5,120 -- the control), once on a second
    /// build over the written member (never rewritten); the right one round-trips through the
    /// engine's window finder.
    #[test]
    fn fn55_the_vision_member_is_added_never_rewritten() {
        let scratch = saber_test_scratch::ScratchDir::new("saber-fn55-vision");
        let ck_dir = scratch.path().join("ckpt");
        let pack = scratch.path().join("pack");
        std::fs::create_dir_all(&ck_dir).unwrap();
        std::fs::create_dir_all(&pack).unwrap();
        let rules = VisionRules::flashnext().unwrap();
        let write_ckpt = |out_hidden: u64| {
            let mut hdr = serde_json::Map::new();
            let mut data = Vec::new();
            for n in rules.records() {
                let shape: Vec<u64> = if n == "model.visual.merger.linear_fc2.weight" {
                    vec![out_hidden, 2]
                } else {
                    vec![2, 2]
                };
                let len = 2 * shape.iter().product::<u64>() as usize;
                let off = data.len();
                data.extend((0..len).map(|i| (i + off) as u8));
                hdr.insert(
                    n,
                    json!({"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len]}),
                );
            }
            let h = serde_json::to_vec(&hdr).unwrap();
            let mut f = (h.len() as u64).to_le_bytes().to_vec();
            f.extend(&h);
            f.extend(&data);
            std::fs::write(ck_dir.join("m.safetensors"), f).unwrap();
            let wm: serde_json::Map<String, serde_json::Value> = rules
                .records()
                .into_iter()
                .map(|n| (n, json!("m.safetensors")))
                .collect();
            std::fs::write(
                ck_dir.join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map": wm})).unwrap(),
            )
            .unwrap();
        };
        write_ckpt(5120);
        let e = build(&pack, &ck_dir).unwrap_err().to_string();
        assert!(e.contains("out_hidden is 2560"), "{e}");
        write_ckpt(2560);
        let bytes = build(&pack, &ck_dir).unwrap();
        assert!(bytes > 0);
        for r in 0..RANKS {
            let f = pack.join(DIR).join(format!("rank{r}.pack"));
            let rd = saber_mem::pack::PackReader::open(&f).unwrap();
            assert_eq!(rd.header().rank, r);
            let w = saber_engine::vision::vision_window(&rd, &rules).unwrap();
            assert_eq!((w.first, w.records), (0, 333));
        }
        let e = build(&pack, &ck_dir).unwrap_err().to_string();
        assert!(e.contains("never rewritten"), "{e}");
    }
}
