//! FN64e: the pack's recorded lazy run set (`profile/lazy_runs.rank<r>.json`, FN64c's
//! `fn64c-lazy-runs/1`) as a pack MEMBER -- `saber flashnext-pack --lazy-runs DIR` merges one or
//! more served boots' records (`--fn-graph-lazy-record DIR`) with the pack's current set (a
//! union: a run's served uses summed over every input; forms and runs most used first) and writes
//! the result into the pack's `profile/` (the old files kept as `lazy_runs.rank<r>.json.<backup>`)
//! or into `--out-dir` (the pack untouched). Refresh it after a defaults change that moves the
//! served step forms (a context band, the MTP table, the graph quantum): record one served boot
//! under the new defaults, then merge it here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context};
use saber_kernels::manifest::sha256_hex;
use serde_json::{json, Value};

/// FN64c's format tag (the worker's `LAZY_SET_FORMAT`).
pub const FORMAT: &str = "fn64c-lazy-runs/1";

/// A form key: (bucket, width, parity, short rows, context band).
type Key = (u64, u64, u64, bool, u64);
/// A form's runs: `[a, b)` -> summed served uses.
type Runs = BTreeMap<(u64, u64), u64>;
/// Per rank: (forms, runs, the old sha256, the new sha256).
pub type RankSummary = (usize, usize, Option<String>, String);

/// One rank's merged set: the grid (q, PLE cut), the replays, per form its runs' summed uses.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Merged {
    pub q: Option<u64>,
    pub ple: Option<Option<u64>>,
    pub replays: u64,
    pub forms: BTreeMap<Key, Runs>,
}

impl Merged {
    /// Add one `fn64c-lazy-runs/1` document of rank `rank`; refuses another format, rank or grid.
    /// # Errors
    /// A malformed document, another rank's, another grid's than the documents already merged.
    pub fn add(&mut self, v: &Value, rank: u32, what: &str) -> anyhow::Result<()> {
        ensure!(
            v["format"] == FORMAT,
            "{what}: format {} is not {FORMAT}",
            v["format"]
        );
        ensure!(
            v["rank"].as_u64() == Some(u64::from(rank)),
            "{what}: rank {} is not {rank}",
            v["rank"]
        );
        let q = v["q"].as_u64().with_context(|| format!("{what}: q"))?;
        let ple = v["ple"].as_u64();
        ensure!(
            self.q.is_none_or(|x| x == q) && self.ple.is_none_or(|x| x == ple),
            "{what}: grid q {q} / PLE {ple:?} differs from the merged q {:?} / PLE {:?}",
            self.q,
            self.ple
        );
        self.q = Some(q);
        self.ple = Some(ple);
        self.replays += v["replays"].as_u64().unwrap_or(0);
        for fm in v["forms"]
            .as_array()
            .with_context(|| format!("{what}: forms"))?
        {
            let g = |k: &str| {
                fm[k]
                    .as_u64()
                    .with_context(|| format!("{what}: form key {k}"))
            };
            let key: Key = (
                g("bucket")?,
                g("width")?,
                g("parity")?,
                fm["short"].as_bool().context("form key short")?,
                g("band")?,
            );
            let runs = self.forms.entry(key).or_default();
            for r in fm["runs"].as_array().context("runs")? {
                let x = |i: usize| r[i].as_u64().context("run [a, b, uses]");
                let (a, b, u) = (x(0)?, x(1)?, x(2)?);
                ensure!(a < b, "{what}: run [{a}, {b}) is empty");
                *runs.entry((a, b)).or_default() += u;
            }
        }
        Ok(())
    }

    /// The runs held over every form.
    #[must_use]
    pub fn runs(&self) -> usize {
        self.forms.values().map(BTreeMap::len).sum()
    }

    /// The merged document: forms by total uses (descending, then key), runs by uses
    /// (descending, then range) -- the order the boot's pre-capture and selection read.
    #[must_use]
    pub fn to_json(&self, rank: u32, from: &[String]) -> Value {
        let mut forms: Vec<(&Key, &Runs)> = self.forms.iter().collect();
        forms.sort_by(|x, y| {
            let t = |m: &Runs| m.values().sum::<u64>();
            t(y.1).cmp(&t(x.1)).then(x.0.cmp(y.0))
        });
        let forms: Vec<Value> = forms
            .into_iter()
            .map(|(&(b, w, p, sh, band), runs)| {
                let mut rs: Vec<(&(u64, u64), &u64)> = runs.iter().collect();
                rs.sort_by(|x, y| y.1.cmp(x.1).then(x.0.cmp(y.0)));
                json!({"bucket": b, "width": w, "parity": p, "short": sh, "band": band,
                    "runs": rs.iter().map(|(r, u)| json!([r.0, r.1, u])).collect::<Vec<_>>()})
            })
            .collect();
        json!({"format": FORMAT, "rank": rank, "q": self.q, "ple": self.ple.flatten(),
            "replays": self.replays, "merged_from": from, "forms": forms})
    }
}

/// FN64e (`saber flashnext-pack --lazy-runs DIR...`): merge per rank the pack's current set
/// (unless `fresh`) with each `DIR/lazy_runs.rank<r>.json`, write it to `out_dir` (or the pack's
/// `profile/`, the old file renamed `lazy_runs.rank<r>.json.<backup>`; an existing backup is
/// refused). Returns per rank (forms, runs, old sha256, new sha256).
/// # Errors
/// A missing rank file in a source, a format / rank / grid mismatch, an existing backup, I/O.
pub fn build(
    pack: &Path,
    srcs: &[PathBuf],
    out_dir: Option<&Path>,
    backup: &str,
    fresh: bool,
) -> anyhow::Result<Vec<RankSummary>> {
    ensure!(
        !srcs.is_empty(),
        "--lazy-runs: at least one record directory"
    );
    ensure!(
        !backup.is_empty() && !backup.contains('/'),
        "--lazy-runs-backup: a plain suffix, not {backup:?}"
    );
    let profile = pack.join(super::run::LAZY_SET_IN_PACK);
    let dest = match out_dir {
        Some(d) => {
            std::fs::create_dir_all(d)?;
            d.to_path_buf()
        }
        None => profile.canonicalize().with_context(|| {
            format!("{}: the pack has no profile/ directory", profile.display())
        })?,
    };
    let mut out = Vec::new();
    for rank in 0..2u32 {
        let mut m = Merged::default();
        let mut from = Vec::new();
        let cur = super::run::lazy_set_file(&profile, rank);
        let old = if cur.exists() {
            let b = std::fs::read(&cur)?;
            let sha = sha256_hex(&b);
            if !fresh {
                m.add(
                    &serde_json::from_slice(&b)?,
                    rank,
                    &cur.display().to_string(),
                )?;
                from.push(format!("{} sha256 {sha}", cur.display()));
            }
            Some(sha)
        } else {
            None
        };
        for s in srcs {
            let p = super::run::lazy_set_file(s, rank);
            let b = std::fs::read(&p).with_context(|| format!("{}", p.display()))?;
            m.add(&serde_json::from_slice(&b)?, rank, &p.display().to_string())?;
            from.push(format!("{} sha256 {}", p.display(), sha256_hex(&b)));
        }
        let text = serde_json::to_string_pretty(&m.to_json(rank, &from))? + "\n";
        let target = super::run::lazy_set_file(&dest, rank);
        if out_dir.is_none() && target.exists() {
            let bk = dest.join(format!("lazy_runs.rank{rank}.json.{backup}"));
            ensure!(
                !bk.exists(),
                "{}: the backup exists (pick another --lazy-runs-backup)",
                bk.display()
            );
            std::fs::copy(&target, &bk)?;
        }
        let tmp = dest.join(format!(".lazy_runs.rank{rank}.json.tmp"));
        std::fs::write(&tmp, &text)?;
        std::fs::rename(&tmp, &target)?;
        let new = sha256_hex(text.as_bytes());
        println!(
            "flashnext-pack lazy-runs rank{rank}: {} forms, {} runs, {} replays merged from {} file(s) -> {} (sha256 {} -> {new})",
            m.forms.len(),
            m.runs(),
            m.replays,
            from.len(),
            target.display(),
            old.as_deref().unwrap_or("none")
        );
        out.push((m.forms.len(), m.runs(), old, new));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(rank: u32, q: u64, forms: Value) -> Value {
        json!({"format": FORMAT, "rank": rank, "q": q, "ple": 1, "replays": 10, "forms": forms})
    }

    /// FN64e: the union sums a run's uses over every input, keeps runs one input lacks, orders
    /// forms and runs most used first, and refuses another rank / grid / format by name; the
    /// merged file reads back through the same law (a re-merge of one file is the identity on
    /// order and doubles every count).
    #[test]
    fn fn64e_the_merge_is_a_union_most_used_first_and_refuses_another_grid() {
        let a = doc(
            0,
            4,
            json!([{"bucket": 1, "width": 4, "parity": 0, "short": true, "band": 10240,
                    "runs": [[0, 1, 50], [48, 49, 50], [1, 48, 40]]}]),
        );
        let b = doc(
            0,
            4,
            json!([{"bucket": 1, "width": 4, "parity": 0, "short": true, "band": 10240,
                    "runs": [[1, 48, 30], [1, 12, 5]]},
                   {"bucket": 8, "width": 3, "parity": 1, "short": false, "band": 262144,
                    "runs": [[0, 1, 400], [48, 49, 400]]}]),
        );
        let mut m = Merged::default();
        m.add(&a, 0, "a").unwrap();
        m.add(&b, 0, "b").unwrap();
        assert_eq!(m.runs(), 6);
        assert_eq!(m.replays, 20);
        let k1 = (1, 4, 0, true, 10240);
        assert_eq!(m.forms[&k1][&(1, 48)], 70);
        assert_eq!(m.forms[&k1][&(1, 12)], 5);
        let v = m.to_json(0, &[]);
        // the long b8 form (800 uses) before the short b1 form (175)
        assert_eq!(v["forms"][0]["bucket"], 8);
        assert_eq!(v["forms"][1]["runs"][0], json!([1, 48, 70]));
        assert_eq!(v["forms"][1]["runs"][3], json!([1, 12, 5]));
        // it reads back: merged twice = the same forms, doubled counts
        let mut r = Merged::default();
        r.add(&v, 0, "v").unwrap();
        r.add(&v, 0, "v").unwrap();
        assert_eq!(r.forms[&k1][&(1, 48)], 140);
        assert_eq!(r.runs(), 6);
        // refusals: another rank, another grid, another format
        assert!(Merged::default().add(&a, 1, "a").is_err());
        let mut g = Merged::default();
        g.add(&a, 0, "a").unwrap();
        assert!(g.add(&doc(0, 2, json!([])), 0, "q2").is_err());
        let mut bad = a.clone();
        bad["format"] = json!("fn64c-lazy-runs/0");
        assert!(Merged::default().add(&bad, 0, "bad").is_err());
    }

    /// FN64e: `build` merges the pack's set with a record into `--out-dir` (the pack untouched)
    /// and into the pack's `profile/` with the backup kept; a second install with the same backup
    /// suffix is refused; `fresh` drops the pack's set.
    #[test]
    fn fn64e_build_installs_with_a_backup_and_refuses_to_overwrite_it() {
        let scratch = saber_test_scratch::ScratchDir::new("saber-fn64e");
        let root = scratch.path().to_path_buf();
        let (pack, rec, out) = (root.join("pack"), root.join("rec"), root.join("out"));
        std::fs::create_dir_all(pack.join("profile")).unwrap();
        std::fs::create_dir_all(&rec).unwrap();
        for r in 0..2u32 {
            let p = doc(
                r,
                4,
                json!([{"bucket": 1, "width": 4, "parity": 0, "short": true, "band": 10240,
                        "runs": [[0, 1, 5], [48, 49, 5]]}]),
            );
            let n = doc(
                r,
                4,
                json!([{"bucket": 1, "width": 4, "parity": 0, "short": false, "band": 32768,
                        "runs": [[0, 1, 9], [48, 49, 9]]}]),
            );
            std::fs::write(
                pack.join(format!("profile/lazy_runs.rank{r}.json")),
                p.to_string(),
            )
            .unwrap();
            std::fs::write(rec.join(format!("lazy_runs.rank{r}.json")), n.to_string()).unwrap();
        }
        let before = std::fs::read(pack.join("profile/lazy_runs.rank0.json")).unwrap();
        let s = build(
            &pack,
            std::slice::from_ref(&rec),
            Some(&out),
            "fn64c",
            false,
        )
        .unwrap();
        assert_eq!((s[0].0, s[0].1), (2, 4));
        assert_eq!(
            std::fs::read(pack.join("profile/lazy_runs.rank0.json")).unwrap(),
            before
        );
        let s = build(&pack, std::slice::from_ref(&rec), None, "fn64c", false).unwrap();
        assert_eq!(s[1].1, 4);
        assert_eq!(
            std::fs::read(pack.join("profile/lazy_runs.rank0.json.fn64c")).unwrap(),
            before
        );
        assert!(build(&pack, std::slice::from_ref(&rec), None, "fn64c", false).is_err());
        let s = build(&pack, std::slice::from_ref(&rec), Some(&out), "x", true).unwrap();
        assert_eq!((s[0].0, s[0].1), (1, 2));
        drop(scratch);
    }
}
