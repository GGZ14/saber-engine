//! `saber sample-check` — the served stream against the rows the head actually wrote.
//!
//! `saber serve --engine real --sample-dump` records, for every row it SAMPLED, the seam the head
//! produced, the parameters the request resolved to, the noise key the draw used and the token that
//! came out. This replays that: it rebuilds each row, re-runs the shipped sampler against the
//! recorded key, and compares ids. A disagreement means the token the client received is not the
//! token this row implies — which is the one thing no wire-level gate can see, because the row and
//! the reply are separated by the whole serving plane.
//!
//! It also DRAWS: ten thousand samples from a handful of those real rows, written out as a
//! histogram. The expected distribution is computed independently (`tools/gates/sampler_stream.py`,
//! in numpy, from the same dumped row) and the two are compared with a chi-square. That is the
//! shape the V1 corpus gate took and the reason it is worth taking: two implementations of one
//! distribution, rather than one implementation compared with itself.
//!
//! Host only. No card, no lock: the rows are already off the device.

use std::collections::BTreeMap;
use std::path::PathBuf;

use saber_serve::engine::TokenId;
use saber_serve::sampler::{DeviceRow, PenaltyContext, RowParams, Sampler, SeededNoise};
use saber_serve::sampling::SamplingParams;

/// Arguments, as the binary declares them.
#[derive(Debug, Clone)]
pub struct SampleCheckArgs {
    /// The JSONL `--sample-dump` wrote.
    pub rows: PathBuf,
    /// Draws per histogram row.
    pub draws: u64,
    /// How many of the dumped rows to draw from.
    pub histogram_rows: usize,
    /// Where to write the histograms.
    pub out: Option<PathBuf>,
    /// Where to write every seam row's truncated distribution `p'` (ws-f41), for the prod oracle
    /// (`the internal reference harness`) to compare row for row.
    pub law_out: Option<PathBuf>,
}

/// One dumped row, as the executor wrote it.
#[derive(Debug, Clone, serde::Deserialize)]
struct Dumped {
    seq: u64,
    choice: u32,
    step: u64,
    /// The output position the draw was keyed by (ws-f40). A dump written before F40 has none and
    /// was keyed by `step`.
    #[serde(default)]
    position: Option<u64>,
    seed: Option<u64>,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    presence: f32,
    frequency: f32,
    repetition: f32,
    vocab: usize,
    path: String,
    token: TokenId,
    /// The draft token this row was asked to accept (ws-f41); `None` on a bonus row, a row with no
    /// draft, or a dump written before F41.
    #[serde(default)]
    draft: Option<TokenId>,
    lse: Option<f32>,
    lse_t: Option<f32>,
    top_ids: Vec<TokenId>,
    top_values: Vec<f32>,
    penalised: Vec<(TokenId, f32)>,
}

impl Dumped {
    /// The noise key's second coordinate: the position since ws-f40, the step before it.
    fn key(&self) -> u64 {
        self.position.unwrap_or(self.step)
    }

    fn device_row(&self) -> DeviceRow {
        DeviceRow {
            top_values: self.top_values.clone(),
            top_ids: self.top_ids.clone(),
            logsumexp: self.lse.unwrap_or(f32::NAN),
            logsumexp_at_temperature: self.lse_t,
            penalised: self.penalised.clone(),
            vocab_size: self.vocab,
        }
    }

    /// The top candidate's share of the whole temperature-scaled distribution: 1.0 when the model
    /// is certain, and small on a row with many plausible continuations. The flatter the row, the
    /// more cells its truncation leaves and the more a chi-square over it can say.
    fn peak_share(&self) -> f64 {
        let (Some(lse_t), Some(v0)) = (self.lse_t, self.top_values.first().copied()) else {
            return 1.0;
        };
        let t = f64::from(self.temperature).max(1e-6);
        (f64::from(v0) / t - f64::from(lse_t)).exp()
    }

    /// The knobs, ALREADY RESOLVED on the serving side. They are rebuilt through
    /// `RowParams::resolve` anyway so the replay runs the same normalisation the step did, and the
    /// two are asserted to agree — a dump whose `top_k` had been resolved differently would
    /// otherwise replay a different pipeline and call the agreement a pass.
    fn params(&self) -> RowParams {
        let p = SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: i32::try_from(self.top_k).unwrap_or(0),
            min_p: self.min_p,
            presence_penalty: self.presence,
            frequency_penalty: self.frequency,
            repetition_penalty: self.repetition,
            ..SamplingParams::default()
        };
        RowParams::resolve(&p, self.vocab)
    }
}

/// Run the check. Returns the number of rows whose replay disagreed with the served token.
///
/// # Errors
/// The dump could not be read or parsed, or the histogram could not be written.
pub fn run(args: &SampleCheckArgs) -> anyhow::Result<usize> {
    let text = std::fs::read_to_string(&args.rows)?;
    let mut rows: Vec<Dumped> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("{}:{}: {e}", args.rows.display(), n + 1))?,
        );
    }
    println!(
        "sample-check: {} sampled row(s) in {}",
        rows.len(),
        args.rows.display()
    );

    let sampler = Sampler::new();
    let (mut agreed, mut skipped_dense, mut skipped_unseeded, mut skipped_penalised) = (0, 0, 0, 0);
    let mut disagreed: Vec<(usize, TokenId, String)> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.path != "seam" {
            // A row read densely has no seam to replay: the dense logits are not in the dump and
            // putting 485 KiB per row in one would be a file nobody reads twice.
            skipped_dense += 1;
            continue;
        }
        let Some(seed) = row.seed else {
            // An unseeded row draws from entropy and is not reproducible by design.
            skipped_unseeded += 1;
            continue;
        };
        if !row.penalised.is_empty() {
            // The dump carries the penalised LOGITS but not the token history behind them, so the
            // pipeline cannot be re-run exactly. Counted rather than approximated.
            skipped_penalised += 1;
            continue;
        }
        let mut noise = SeededNoise::new(seed, row.key(), row.choice);
        match sampler.sample(
            &row.device_row(),
            &row.params(),
            &PenaltyContext::default(),
            &mut noise,
        ) {
            Ok(out) if out.token_id == row.token => agreed += 1,
            Ok(out) => disagreed.push((
                i,
                row.token,
                format!("the replay drew {} from the same row", out.token_id),
            )),
            Err(e) => disagreed.push((i, row.token, e.to_string())),
        }
    }
    println!(
        "replay        {agreed} of {} seam rows agree with the served token \
         ({} dense, {} unseeded, {} penalised were skipped)",
        rows.len() - skipped_dense - skipped_unseeded - skipped_penalised,
        skipped_dense,
        skipped_unseeded,
        skipped_penalised
    );
    for (i, token, why) in disagreed.iter().take(8) {
        println!("  row {i}: the stream committed {token} and {why}");
    }

    // ---- the draws ---------------------------------------------------------------------------
    if let Some(out) = &args.out {
        let mut json = String::from("{\n \"draws\": ");
        json.push_str(&args.draws.to_string());
        json.push_str(",\n \"rows\": [\n");
        // The rows a histogram can be drawn from at all, and then the FLATTEST of them.
        //
        // Not an arbitrary stride: at prod's own defaults most real rows truncate to ONE surviving
        // token — the model is sure — and a chi-square over one cell is a statistic with nothing
        // in it (the first run of this gate read `chi2 0.0 at df 1, p 1` on two of three rows and
        // that number was not evidence of anything). The flattest rows are the ones with cells,
        // and they are also the ones a seam has to work hardest on, so the test is where the
        // information is. `exp(v0/T - lse_t)` is the top candidate's share of the whole
        // temperature-scaled distribution, straight off the device's own log-sum-exp.
        let mut replayable: Vec<&Dumped> = rows
            .iter()
            .filter(|r| r.path == "seam" && r.seed.is_some() && r.penalised.is_empty())
            .collect();
        replayable.sort_by(|a, b| {
            a.peak_share()
                .partial_cmp(&b.peak_share())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let chosen: Vec<&Dumped> = replayable.into_iter().take(args.histogram_rows).collect();
        for (n, row) in chosen.iter().enumerate() {
            let params = row.params();
            let device = row.device_row();
            let seed = row.seed.unwrap_or(0);
            let mut counts: BTreeMap<TokenId, u64> = BTreeMap::new();
            let mut refusals = 0u64;
            for d in 0..args.draws {
                // The noise key varies by POSITION, exactly as it does between two tokens of one
                // request: the draws are from the same family the serve path uses.
                let mut noise = SeededNoise::new(seed, row.key().wrapping_add(d), row.choice);
                match sampler.sample(&device, &params, &PenaltyContext::default(), &mut noise) {
                    Ok(o) => *counts.entry(o.token_id).or_insert(0) += 1,
                    Err(_) => refusals += 1,
                }
            }
            let ids: Vec<String> = device.top_ids.iter().map(ToString::to_string).collect();
            let values: Vec<String> = device.top_values.iter().map(|v| format!("{v:e}")).collect();
            let hist: Vec<String> = counts.iter().map(|(id, c)| format!("[{id},{c}]")).collect();
            json.push_str(&format!(
                "  {{\"seq\":{},\"step\":{},\"seed\":{seed},\"temperature\":{:e},\"top_p\":{:e},\
                 \"top_k\":{},\"min_p\":{:e},\"vocab\":{},\"lse_t\":{},\"refusals\":{refusals},\
                 \"top_ids\":[{}],\"top_values\":[{}],\"counts\":[{}]}}{}\n",
                row.seq,
                row.step,
                params.temperature,
                params.top_p,
                params.top_k,
                params.min_p,
                params.vocab_size,
                device
                    .logsumexp_at_temperature
                    .map_or("null".to_owned(), |v| format!("{v:e}")),
                ids.join(","),
                values.join(","),
                hist.join(","),
                if n + 1 == chosen.len() { "" } else { "," }
            ));
        }
        json.push_str(" ]\n}\n");
        std::fs::write(out, json)?;
        println!(
            "draws         {} row(s) x {} draws -> {}",
            chosen.len(),
            args.draws,
            out.display()
        );
    }
    if let Some(out) = &args.law_out {
        law(&rows, &sampler, out)?;
    }
    Ok(disagreed.len())
}

/// THE LAW ON THE DUMPED ROWS (ws-f41): every seam row's `p'` as the shipped sampler computes it,
/// written for the prod oracle, and the acceptance law's own books kept on the host.
///
/// A verify row the walk reached with a draft `d` accepts `d` with probability `p'(d)` — the rate
/// prod's rejection sampler uses for a greedy drafter — so over every such row the accepted count
/// and `Σ p'(d)` are equal in expectation (a martingale: each row's outcome is a coin of its own
/// `p'(d)` whatever selected the row). A walk that accepts fewer than its rows' own rates is a
/// law defect; one that accepts what its rates say is the law, and any gap to prod lives in `p'`
/// or in the drafts. The mean of `p'(argmax)` is the single-row acceptance of an argmax drafter.
fn law(rows: &[Dumped], sampler: &Sampler, out: &std::path::Path) -> anyhow::Result<()> {
    use std::fmt::Write as _;
    let mut text = String::new();
    let (mut n, mut refused, mut skipped) = (0usize, 0usize, 0usize);
    let (mut sum_argmax, mut drafted, mut accepted) = (0.0f64, 0usize, 0usize);
    let (mut rate, mut var, mut draft_is_argmax) = (0.0f64, 0.0f64, 0usize);
    for (i, row) in rows.iter().enumerate() {
        if row.path != "seam" || !row.penalised.is_empty() {
            skipped += 1;
            continue;
        }
        let device = row.device_row();
        let dist = match sampler.distribution(&device, &row.params(), &PenaltyContext::default()) {
            Ok(Some(d)) => d,
            Ok(None) => {
                skipped += 1;
                continue;
            }
            Err(e) => {
                refused += 1;
                writeln!(
                    text,
                    "{{\"i\":{i},\"refused\":\"{}\"}}",
                    e.to_string().replace('"', "'")
                )?;
                continue;
            }
        };
        n += 1;
        let p_of = |id: TokenId| -> f64 {
            dist.iter()
                .find(|(t, _)| *t == id)
                .map_or(0.0, |(_, p)| f64::from(*p))
        };
        let argmax = device.top_ids[0];
        sum_argmax += p_of(argmax);
        if let Some(d) = row.draft {
            let p = p_of(d);
            drafted += 1;
            rate += p;
            var += p * (1.0 - p);
            accepted += usize::from(row.token == d);
            draft_is_argmax += usize::from(d == argmax);
        }
        let ids: Vec<String> = dist.iter().map(|(t, _)| t.to_string()).collect();
        let probs: Vec<String> = dist.iter().map(|(_, p)| format!("{p:e}")).collect();
        writeln!(
            text,
            "{{\"i\":{i},\"seq\":{},\"position\":{},\"token\":{},\"draft\":{},\"argmax\":{argmax},\
             \"ids\":[{}],\"probs\":[{}]}}",
            row.seq,
            row.key(),
            row.token,
            row.draft.map_or("null".to_owned(), |d| d.to_string()),
            ids.join(","),
            probs.join(",")
        )?;
    }
    std::fs::write(out, text)?;
    println!(
        "law           {n} seam row(s) -> {} ({refused} refused, {skipped} skipped: dense, \
         penalised or greedy); mean p'(argmax) {:.4}",
        out.display(),
        if n > 0 {
            sum_argmax / n as f64
        } else {
            f64::NAN
        }
    );
    if drafted > 0 {
        println!(
            "walk          {drafted} drafted row(s) reached: accepted {accepted}, expected \
             sum p'(draft) {rate:.1} (sd {:.1}, z {:+.2}); mean p'(draft) {:.4}; draft = argmax \
             on {draft_is_argmax}",
            var.sqrt(),
            (accepted as f64 - rate) / var.sqrt().max(1e-9),
            rate / drafted as f64
        );
    }
    Ok(())
}
