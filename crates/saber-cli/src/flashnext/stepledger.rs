//! FN75: the SERVED-STEP LEDGER -- one record per scheduler step on the Flash-Next serve plane,
//! joined from its two halves: the scheduler's clock and shape (`saber_sched::StepTiming`, the step
//! thread) and the engine's (rank 0's job: when it was picked up, its device spans by part, the
//! host phases of the target step, the reply, the post-reply drafter round). The halves arrive in
//! step order on two threads and are joined FIFO (one engine record per `Step` / `Mixed` job, one
//! scheduler record per executed step: the same sequence).
//!
//! Byte-neutral: it reads clocks and the device spans the engine already measures (its two step
//! events); nothing it records feeds a decision. `/metrics` carries the aggregate
//! (`saber_fn_step_*`, by step class), the shutdown line `fnstep75` the same, and
//! `--fn-step-trace PATH` writes every joined record as one JSON line (`fn75-step/1`).

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::io::Write as _;
use std::time::Instant;

use saber_plan::Phase;
use saber_sched::StepTiming;

/// The engine's half of one step job (rank 0).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngRec {
    /// The job's pick-up (the rank thread's `recv` returned).
    pub pick: Option<Instant>,
    /// The reply's send.
    pub reply: Option<Instant>,
    /// Wall of the NON-step jobs (admit, release, encode, valve) since the previous step job.
    pub other_ms: f64,
    /// Non-step jobs run since the previous step job.
    pub other_n: u32,
    /// The target decode / verify step: device span (its two events), wall, host before the
    /// wait (embed, tables, PLE, tier, issue, refill), the wait, host after (tier end, pick).
    pub tgt_dev_ms: f64,
    pub tgt_wall_ms: f64,
    pub tgt_pre_ms: f64,
    pub tgt_wait_ms: f64,
    pub tgt_post_ms: f64,
    /// Prefill chunks: device span and wall (the drafter's chunk rows excluded).
    pub pf_dev_ms: f64,
    pub pf_wall_ms: f64,
    /// The drafter (MTP passes in this job: after-chunk rows, First, the post-reply round):
    /// device span and wall.
    pub dr_dev_ms: f64,
    pub dr_wall_ms: f64,
    /// The drafter's wall AFTER the reply (the round that overlaps the scheduler's commit).
    pub post_ms: f64,
    /// Dense-row reads (a seam that could not decide a row).
    pub dense: u32,
    /// The verify width the engine ran (decode), the ready rows.
    pub w: u32,
    pub n: u32,
    /// FN93: the target step's host phases by name ([`SUB93`]): what `tgt_pre` and `tgt_post` are
    /// made of (records only).
    pub sub: [f64; SUB93_N],
    /// FN90a (the tail census, trace only): the target step's host phases by name (`h_embed`,
    /// `h_ple_l0`, `h_tier`, ..), the PLE rows it read (`ple_nvme` from the drive, `ple_hits`), its
    /// expert reads (`x_res`, `x_ram`) and rank 0's lazy capture passes (`cap`), summed per job.
    pub fx: Vec<(&'static str, f64)>,
}

impl EngRec {
    /// FN90a: add `v` to the extra field `k` (created at its first use).
    pub fn fx_add(&mut self, k: &'static str, v: f64) {
        match self.fx.iter_mut().find(|(n, _)| *n == k) {
            Some(e) => e.1 += v,
            None => self.fx.push((k, v)),
        }
    }
}

/// FN93: the number of named host phases a target step carries ([`SUB93`]).
pub const SUB93_N: usize = 16;

/// FN93: the target step's named host phases (ms), in [`EngRec::sub`] order. After the wait
/// (`tgt_post`): the tier end's fence, route-log read, resolve, victim PLANNING, copy ISSUE and
/// the rest of its lap (the op-timing read, the census), then the pick's logits read (the device
/// seam and the rows a sampled or logprob row needs), its host seam (frames, the log-sum-exps),
/// the rank exchange and merge, and the rest. Before the launch (`tgt_pre`): embed, tables, PLE
/// (gather, rows, the layer-1 copy), tier begin, graph issue, refill.
pub const SUB93: [&str; SUB93_N] = [
    "te_fence",
    "te_log",
    "te_resolve",
    "te_plan",
    "te_issue",
    "te_rest",
    "pick_read",
    "pick_seam",
    "pick_xchg",
    "pick_rest",
    "pre_embed",
    "pre_tables",
    "pre_ple",
    "pre_tier",
    "pre_issue",
    "pre_refill",
];

/// One joined record, in ms (the derived spans).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    pub class: String,
    /// The previous step's end -> this step's entry (the step loop's turn, idle polls).
    pub gap_ms: f64,
    /// Whether the gap had work (a request running or queued at the previous step's end).
    pub gap_work: bool,
    /// Entry -> the executor call (admission, batch build).
    pub prep_ms: f64,
    /// The executor call -> the engine picked the job up (the previous job's post-reply drafter
    /// round, the channel's trip, non-step jobs ahead of it).
    pub hand_ms: f64,
    /// Pick-up -> reply (everything rank 0 runs for the step before answering).
    pub job_ms: f64,
    /// Reply -> the executor call returned (the channel's trip back).
    pub trip_ms: f64,
    /// The commit (token delivery to the requests, the scheduler's bookkeeping).
    pub commit_ms: f64,
    /// The whole step from the previous step's end (a gap without work counted out).
    pub wall_ms: f64,
    pub tokens: u32,
    pub rows: u32,
    pub width: u32,
    pub chunk_rows: u32,
    pub chunks: u32,
    pub eng: EngRec,
}

fn ms(a: Instant, b: Instant) -> f64 {
    b.saturating_duration_since(a).as_secs_f64() * 1e3
}

/// The step's class: `d<rows>` decode, `p` a prefill chunk alone, `m<rows>` a mixed step.
fn class(t: &StepTiming) -> String {
    match (t.phase, t.chunks) {
        (Phase::Prefill, _) => "p".to_owned(),
        (_, 0) => format!("d{}", t.decode_rows),
        _ => format!("m{}", t.decode_rows),
    }
}

/// Join one scheduler record with its engine record.
#[must_use]
pub fn join(t: &StepTiming, e: EngRec) -> Row {
    let gap_ms = t.prev_end.map_or(0.0, |p| ms(p, t.start));
    let gap_work = t.prev_end.is_some() && t.work_at_prev_end;
    let pick = e.pick.unwrap_or(t.exec_start);
    let reply = e.reply.unwrap_or(t.exec_end);
    let from = if gap_work {
        t.prev_end.unwrap_or(t.start)
    } else {
        t.start
    };
    Row {
        class: class(t),
        gap_ms,
        gap_work,
        prep_ms: ms(t.start, t.exec_start),
        hand_ms: ms(t.exec_start, pick),
        job_ms: ms(pick, reply),
        trip_ms: ms(reply, t.exec_end),
        commit_ms: ms(t.exec_end, t.end),
        wall_ms: ms(from, t.end),
        tokens: t.committed_tokens,
        rows: t.decode_rows,
        width: t.width,
        chunk_rows: t.chunk_rows,
        chunks: t.chunks,
        eng: e,
    }
}

/// The per-class sums (`/metrics`, the shutdown line).
#[derive(Clone, Debug, Default)]
struct Agg {
    n: u64,
    sums: [f64; 14],
    tokens: u64,
    /// FN93: the named host phases' sums.
    sub: [f64; SUB93_N],
}

const FIELDS: [&str; 14] = [
    "wall", "gap", "prep", "hand", "job", "trip", "commit", "tgt_dev", "tgt_pre", "tgt_post",
    "pf_dev", "dr_dev", "dr_wall", "post",
];

impl Agg {
    fn add(&mut self, r: &Row) {
        self.n += 1;
        self.tokens += u64::from(r.tokens);
        let g = if r.gap_work { r.gap_ms } else { 0.0 };
        let v = [
            r.wall_ms,
            g,
            r.prep_ms,
            r.hand_ms,
            r.job_ms,
            r.trip_ms,
            r.commit_ms,
            r.eng.tgt_dev_ms,
            r.eng.tgt_pre_ms,
            r.eng.tgt_post_ms,
            r.eng.pf_dev_ms,
            r.eng.dr_dev_ms,
            r.eng.dr_wall_ms,
            r.eng.post_ms,
        ];
        for (s, x) in self.sums.iter_mut().zip(v) {
            *s += x;
        }
        for (s, x) in self.sub.iter_mut().zip(r.eng.sub) {
            *s += x;
        }
    }
}

/// The ledger both threads share (`Arc<Mutex<_>>`; one short lock a record).
#[derive(Debug, Default)]
pub struct StepLedger {
    sched: VecDeque<StepTiming>,
    eng: VecDeque<EngRec>,
    by: BTreeMap<String, Agg>,
    /// Client-idle gaps (no request running or queued): count, ms.
    idle: (u64, f64),
    trace: Option<std::io::BufWriter<std::fs::File>>,
    /// The trace's clock: an `Instant` and the unix ms it was taken at (records carry unix ms).
    epoch: Option<(Instant, f64)>,
    joined: u64,
}

impl StepLedger {
    /// A ledger writing every joined record to `trace` (JSON lines) when given.
    pub fn new(trace: Option<&std::path::Path>) -> anyhow::Result<Self> {
        let trace = match trace {
            Some(p) => {
                let mut w = std::io::BufWriter::new(std::fs::File::create(p)?);
                writeln!(w, "{{\"format\":\"fn75-step/1\"}}")?;
                Some(w)
            }
            None => None,
        };
        Ok(Self {
            trace,
            epoch: Some((
                Instant::now(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0.0, |d| d.as_secs_f64() * 1e3),
            )),
            ..Self::default()
        })
    }

    pub fn push_sched(&mut self, t: StepTiming) {
        self.sched.push_back(t);
        self.drain();
    }

    pub fn push_eng(&mut self, e: EngRec) {
        self.eng.push_back(e);
        self.drain();
    }

    fn drain(&mut self) {
        while !self.sched.is_empty() && !self.eng.is_empty() {
            let (Some(t), Some(e)) = (self.sched.pop_front(), self.eng.pop_front()) else {
                break;
            };
            let r = join(&t, e);
            if !r.gap_work && t.prev_end.is_some() {
                self.idle.0 += 1;
                self.idle.1 += r.gap_ms;
            }
            self.by.entry(r.class.clone()).or_default().add(&r);
            self.joined += 1;
            if let Some(w) = self.trace.as_mut() {
                let at = self.epoch.map_or(0.0, |(i0, u0)| u0 + ms(i0, t.start));
                let e = &r.eng;
                let mut line = format!(
                    "{{\"i\":{},\"t\":{at:.3},\"c\":\"{}\",\"rows\":{},\"w\":{},\"ew\":{},\"en\":{},\"crows\":{},\"chunks\":{},\"tok\":{},\"gw\":{},\"gap\":{:.3},\"idle_polls\":{},\"prep\":{:.3},\"hand\":{:.3},\"job\":{:.3},\"trip\":{:.3},\"commit\":{:.3},\"wall\":{:.3},\"other\":{:.3},\"other_n\":{},\"tgt_dev\":{:.3},\"tgt_wall\":{:.3},\"tgt_pre\":{:.3},\"tgt_wait\":{:.3},\"tgt_post\":{:.3},\"pf_dev\":{:.3},\"pf_wall\":{:.3},\"dr_dev\":{:.3},\"dr_wall\":{:.3},\"post\":{:.3},\"dense\":{}{}",
                    self.joined,
                    r.class,
                    r.rows,
                    r.width,
                    e.w,
                    e.n,
                    r.chunk_rows,
                    r.chunks,
                    r.tokens,
                    u8::from(r.gap_work),
                    r.gap_ms,
                    t.idle_polls,
                    r.prep_ms,
                    r.hand_ms,
                    r.job_ms,
                    r.trip_ms,
                    r.commit_ms,
                    r.wall_ms,
                    e.other_ms,
                    e.other_n,
                    e.tgt_dev_ms,
                    e.tgt_wall_ms,
                    e.tgt_pre_ms,
                    e.tgt_wait_ms,
                    e.tgt_post_ms,
                    e.pf_dev_ms,
                    e.pf_wall_ms,
                    e.dr_dev_ms,
                    e.dr_wall_ms,
                    e.post_ms,
                    e.dense,
                    SUB93.iter().zip(e.sub).fold(String::new(), |mut o, (k, v)| {
                        let _ = write!(o, ",\"{k}\":{v:.3}");
                        o
                    }),
                );
                // FN90a: the extra fields (none unless the engine filled them)
                for (k, v) in &e.fx {
                    let _ = write!(line, ",\"{k}\":{v:.3}");
                }
                // FN157 (the admit-slack census): the step's entry -> end, the sequences running,
                // the queue left after admission, the sequences the step finished, and each
                // admitted request's arrival -> entry (ms)
                let _ = write!(
                    line,
                    ",\"dur\":{:.3},\"run\":{},\"wq\":{},\"fin\":{},\"adm\":{},\"alag\":[{}]",
                    ms(t.start, t.end),
                    t.running,
                    t.waiting,
                    t.finished,
                    t.admitted,
                    t.admit_lag_us
                        .iter()
                        .take(t.admitted.min(8) as usize)
                        .map(|u| format!("{:.3}", f64::from(*u) / 1e3))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                let _ = writeln!(w, "{line}}}");
            }
        }
    }

    /// Flush the trace (the shutdown).
    pub fn flush(&mut self) {
        if let Some(w) = self.trace.as_mut() {
            let _ = w.flush();
        }
    }

    /// `saber_fn_step_*` families, by step class.
    #[must_use]
    pub fn render(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(
            o,
            "# HELP saber_fn_step_total FN75 served steps by class (d<rows> decode, p prefill alone, m<rows> mixed).\n# TYPE saber_fn_step_total counter"
        );
        for (c, a) in &self.by {
            let _ = writeln!(o, "saber_fn_step_total{{class=\"{c}\"}} {}", a.n);
        }
        let _ = writeln!(o, "# TYPE saber_fn_step_tokens_total counter");
        for (c, a) in &self.by {
            let _ = writeln!(
                o,
                "saber_fn_step_tokens_total{{class=\"{c}\"}} {}",
                a.tokens
            );
        }
        let _ = writeln!(
            o,
            "# HELP saber_fn_step_ms_total FN75 summed ms by class and span.\n# TYPE saber_fn_step_ms_total counter"
        );
        for (c, a) in &self.by {
            for (k, s) in FIELDS.iter().zip(a.sums) {
                let _ = writeln!(
                    o,
                    "saber_fn_step_ms_total{{class=\"{c}\",span=\"{k}\"}} {s:.3}"
                );
            }
            // FN93: the target step's named host phases
            for (k, s) in SUB93.iter().zip(a.sub) {
                let _ = writeln!(
                    o,
                    "saber_fn_step_ms_total{{class=\"{c}\",span=\"{k}\"}} {s:.3}"
                );
            }
        }
        let _ = writeln!(
            o,
            "# TYPE saber_fn_step_client_idle_ms_total counter\nsaber_fn_step_client_idle_ms_total {:.3}\nsaber_fn_step_client_idle_total {}",
            self.idle.1, self.idle.0
        );
        o
    }

    /// The shutdown line: per class, steps, tokens and the mean of every span.
    #[must_use]
    pub fn line(&self) -> String {
        let mut o = format!(
            "fnstep75 rank0: {} steps joined ({} sched / {} engine unjoined), client-idle gaps {} ({:.1} ms)",
            self.joined,
            self.sched.len(),
            self.eng.len(),
            self.idle.0,
            self.idle.1
        );
        for (c, a) in &self.by {
            let _ = write!(o, " | {c} n {} tok {}", a.n, a.tokens);
            for (k, s) in FIELDS.iter().zip(a.sums) {
                let _ = write!(o, " {k} {:.3}", s / a.n.max(1) as f64);
            }
        }
        o
    }

    /// FN93: the shutdown line of the target step's named host phases, decode classes only
    /// (means a step: what `tgt_post` and `tgt_pre` are made of).
    #[must_use]
    pub fn line93(&self) -> String {
        let mut o = String::from("fnstep93 rank0:");
        for (c, a) in self.by.iter().filter(|(c, _)| c.starts_with('d')) {
            let n = a.n.max(1) as f64;
            let _ = write!(o, " | {c} n {}", a.n);
            for (k, s) in SUB93.iter().zip(a.sub) {
                let _ = write!(o, " {k} {:.3}", s / n);
            }
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn timing(prev: Option<Instant>, t0: Instant, rows: u32, chunks: u32) -> StepTiming {
        StepTiming {
            prev_end: prev,
            work_at_prev_end: true,
            idle_polls: 0,
            start: t0,
            exec_start: t0 + Duration::from_millis(1),
            exec_end: t0 + Duration::from_millis(30),
            end: t0 + Duration::from_millis(31),
            phase: if rows == 0 {
                Phase::Prefill
            } else {
                Phase::Decode
            },
            decode_rows: rows,
            width: 3,
            chunk_rows: if chunks > 0 { 300 } else { 0 },
            chunks,
            waiting_decoders: 0,
            running: rows.max(1),
            committed_tokens: rows * 2,
            finished: 0,
            waiting: 0,
            admitted: 0,
            admit_lag_us: [0; 8],
        }
    }

    #[test]
    fn fn75_the_halves_join_in_order_and_the_spans_add_up_to_the_wall() {
        let t0 = Instant::now();
        let mut l = StepLedger::default();
        let a = timing(None, t0, 8, 0);
        let b = timing(Some(a.end), a.end + Duration::from_millis(2), 0, 1);
        // the engine's halves first for one, the scheduler's first for the other
        l.push_eng(EngRec {
            pick: Some(a.exec_start + Duration::from_millis(3)),
            reply: Some(a.exec_end - Duration::from_millis(1)),
            tgt_dev_ms: 20.0,
            ..EngRec::default()
        });
        l.push_sched(a);
        l.push_sched(b);
        assert_eq!(l.joined, 1, "b waits for its engine half");
        l.push_eng(EngRec {
            pick: Some(b.exec_start),
            reply: Some(b.exec_end),
            pf_dev_ms: 25.0,
            ..EngRec::default()
        });
        assert_eq!(l.joined, 2);
        let d8 = &l.by["d8"];
        let p = &l.by["p"];
        assert_eq!((d8.n, p.n, d8.tokens), (1, 1, 16));
        let r = join(&b, EngRec::default());
        let parts = r.gap_ms + r.prep_ms + r.hand_ms + r.job_ms + r.trip_ms + r.commit_ms;
        assert!((parts - r.wall_ms).abs() < 1e-6, "{parts} vs {}", r.wall_ms);
        assert!((r.gap_ms - 2.0).abs() < 1e-6);
        // the first step has no gap: its wall is entry -> end
        let ra = join(&a, EngRec::default());
        assert!((ra.wall_ms - 31.0).abs() < 1e-6);
        assert!(l.render().contains("saber_fn_step_total{class=\"d8\"} 1"));
        assert!(l.line().contains("d8 n 1 tok 16"));
        // FN93: the named phases ride the decode classes only
        assert!(l.line93().contains("| d8 n 1 te_fence 0.000"));
        assert!(!l.line93().contains("| p n"));
        assert!(l
            .render()
            .contains("saber_fn_step_ms_total{class=\"d8\",span=\"pick_seam\"} 0.000"));
        // the control: an engine record that is not this step's moves the spans
        let wrong = join(
            &a,
            EngRec {
                pick: Some(a.end),
                ..EngRec::default()
            },
        );
        assert!(wrong.hand_ms > 29.0);
    }

    #[test]
    fn fn93_the_named_host_phases_ride_the_trace_line_and_the_class_means() {
        let dir = saber_test_scratch::ScratchDir::new("fn93-trace");
        let p = dir.path().join("trace.jsonl");
        let t0 = Instant::now();
        let mut l = StepLedger::new(Some(&p)).unwrap();
        let a = timing(None, t0, 8, 0);
        let mut sub = [0.0; SUB93_N];
        sub[3] = 1.5; // te_plan
        sub[7] = 3.25; // pick_seam
        l.push_eng(EngRec {
            pick: Some(a.exec_start),
            reply: Some(a.exec_end),
            sub,
            ..EngRec::default()
        });
        l.push_sched(a);
        l.flush();
        let text = std::fs::read_to_string(&p).unwrap();
        let rec = text.lines().nth(1).unwrap();
        assert!(rec.contains(",\"pre_refill\":0.000,"), "{rec}");
        assert!(rec.contains(",\"te_plan\":1.500,") && rec.contains(",\"pick_seam\":3.250,"));
        assert!(l.line93().contains("te_plan 1.500") && l.line93().contains("pick_seam 3.250"));
        // the control: a record without the phases carries zeros, not the previous step's
        assert_eq!(EngRec::default().sub, [0.0; SUB93_N]);
    }

    #[test]
    fn fn90a_the_extra_fields_ride_the_trace_line_as_json() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fn90a-tests");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("trace.jsonl");
        let t0 = Instant::now();
        {
            let mut l = StepLedger::new(Some(p.as_path())).unwrap();
            let a = timing(None, t0, 1, 0);
            let mut e = EngRec {
                pick: Some(a.exec_start),
                reply: Some(a.exec_end),
                ..EngRec::default()
            };
            e.fx_add("h_ple_l0", 8.5);
            e.fx_add("ple_nvme", 4.0);
            e.fx_add("h_ple_l0", 0.25);
            l.push_eng(e);
            l.push_sched(a);
            // the control: a record without extra fields writes the FN75 line unchanged
            let b = timing(Some(a.end), a.end, 1, 0);
            l.push_eng(EngRec::default());
            l.push_sched(b);
            l.flush();
        }
        let text = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert!((lines[1]["h_ple_l0"].as_f64().unwrap() - 8.75).abs() < 1e-9);
        assert_eq!(lines[1]["ple_nvme"].as_f64(), Some(4.0));
        assert!(lines[2].get("h_ple_l0").is_none());
        assert_eq!(lines[2]["dense"].as_u64(), Some(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// FN157: the admit-slack census rides the trace line (running, queue, finished, each admitted
    /// request's arrival -> entry); a step that admitted nothing carries an empty list.
    #[test]
    fn fn157_the_admit_census_rides_the_trace_line() {
        let dir = saber_test_scratch::ScratchDir::new("fn157-trace");
        let p = dir.path().join("trace.jsonl");
        let t0 = Instant::now();
        let mut l = StepLedger::new(Some(&p)).unwrap();
        let mut a = timing(None, t0, 7, 1);
        a.running = 8;
        a.waiting = 1;
        a.finished = 2;
        a.admitted = 2;
        a.admit_lag_us = [1500, 31250, 0, 0, 0, 0, 0, 0];
        l.push_eng(EngRec::default());
        l.push_sched(a);
        let b = timing(Some(a.end), a.end, 8, 0);
        l.push_eng(EngRec::default());
        l.push_sched(b);
        l.flush();
        let text = std::fs::read_to_string(&p).unwrap();
        let v: Vec<serde_json::Value> = text
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect();
        assert_eq!(v[1]["run"].as_u64(), Some(8));
        assert_eq!(v[1]["wq"].as_u64(), Some(1));
        assert_eq!(v[1]["fin"].as_u64(), Some(2));
        assert_eq!(v[1]["adm"].as_u64(), Some(2));
        let lag: Vec<f64> = v[1]["alag"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(lag, vec![1.5, 31.25]);
        assert!((v[1]["dur"].as_f64().unwrap() - 31.0).abs() < 1e-6);
        assert_eq!(v[2]["alag"].as_array().map(Vec::len), Some(0));
    }
}
