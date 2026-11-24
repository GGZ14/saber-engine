//! INK32: copy drafting (`--ink-ngram on`, an ARM; `off` = THE DEFAULT, the control).
//!
//! A request that repeats its own context (an edited file returned whole, quotes, reformatted
//! tables, echoed tool output) is drafted by LOOKUP instead of by the drafter: the trailing
//! n-gram of the sequence's tokens (prompt + every committed output) is searched for its most
//! recent earlier occurrence -- matched over at least `min` tokens, extended backwards up to `max`,
//! the longest wins (ties: the most recent) -- and the tokens that followed it are the drafts.
//!
//! Exactness: the drafts are one-hot proposals walked by the serve plane's drawn-row walk (row a
//! is the target's own draw at output position `position + a`; a draft survives only where the
//! draw equals it), so the committed tokens are the non-speculative stream whatever the drafts --
//! greedy and seeded alike. The drafts only move how many tokens a round carries.
//!
//! Price law (`depth_for`): a copy round verifies `1 + d` rows on the fused chained verify
//! (`ink_xsdc`, T <= 8) and skips the drafter's round; the census (`tools/capture/ink32/
//! ngram_census.py`) found the full depth pays at every qualifying match length.
use std::collections::HashMap;

/// The flags' resolved form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramOpt {
    /// matched tokens a proposal needs (>= 1)
    pub min: usize,
    /// the longest match compared (>= min)
    pub max: usize,
    /// drafts a single-sequence round may verify (1..7: T = 1 + depth <= 8)
    pub depth: usize,
}

impl NgramOpt {
    /// `--ink-ngram on|off` + `--ink-ngram-{min,max,depth}`; `None` = off.
    pub fn parse(on: &str, min: u64, max: u64, depth: u64) -> anyhow::Result<Option<Self>> {
        match on {
            "off" => Ok(None),
            "on" => {
                anyhow::ensure!(
                    (1..=32).contains(&min) && (min..=64).contains(&max),
                    "--ink-ngram-min {min} / --ink-ngram-max {max}: 1 <= min <= 32, min <= max <= 64"
                );
                anyhow::ensure!(
                    (1..=7).contains(&depth),
                    "--ink-ngram-depth {depth}: 1..7 (a verify is <= 8 rows)"
                );
                Ok(Some(Self {
                    min: min as usize,
                    max: max as usize,
                    depth: depth as usize,
                }))
            }
            o => anyhow::bail!("--ink-ngram {o:?}: on | off (the control)"),
        }
    }
}

/// One sequence's token history + the index of its `min`-grams: key = the `min` tokens, value =
/// the END positions (exclusive) of their earlier occurrences, ascending.
#[derive(Debug, Default)]
pub struct History {
    pub toks: Vec<u32>,
    tab: HashMap<Vec<u32>, Vec<u32>>,
    min: usize,
}

/// How many of the most recent occurrences a lookup compares (bounds the host cost).
const SCAN: usize = 64;

impl History {
    pub fn new(min: usize) -> Self {
        Self {
            toks: Vec::new(),
            tab: HashMap::new(),
            min: min.max(1),
        }
    }

    /// Append committed tokens; the `min`-gram ending just before each new token is indexed
    /// (its continuation starts at that token), so the current suffix never matches itself.
    pub fn push(&mut self, ts: &[u32]) {
        for &t in ts {
            self.toks.push(t);
            let p = self.toks.len();
            if p > self.min {
                let e = p - 1;
                let key = self.toks[e - self.min..e].to_vec();
                self.tab.entry(key).or_default().push(e as u32);
            }
        }
    }

    /// The proposal for the next tokens: (match length, up to `depth` drafts), or (0, []) when
    /// no earlier occurrence of the trailing `min` tokens exists.
    pub fn propose(&self, max: usize, depth: usize) -> (usize, Vec<u32>) {
        let c = &self.toks;
        let p = c.len();
        let n = self.min;
        if p <= n || depth == 0 {
            return (0, Vec::new());
        }
        let Some(ends) = self.tab.get(&c[p - n..p]) else {
            return (0, Vec::new());
        };
        let (mut best, mut be) = (0usize, 0usize);
        for &e in ends.iter().rev().take(SCAN) {
            let e = e as usize;
            let mut m = n;
            while m < max && e > m && c[e - m - 1] == c[p - m - 1] {
                m += 1;
            }
            if m > best {
                best = m;
                be = e;
                if m >= max {
                    break;
                }
            }
        }
        (best, c[be..(be + depth).min(p)].to_vec())
    }
}

/// INK32 census (`ngram_census.py` over the copy set + the 40 smoke prompts, the price table of
/// INK19 h8 + the served round overhead): once a match qualifies (>= `min`, 5 at the default:
/// 3- and 4-token matches are mostly coincidences that LOSE to the drafter on structured output,
/// e.g. CSV -> JSON 0.97 vs 1.13 at min 5), verifying the full depth pays at every match length
/// (by-length depth tables tied within 1 %): `depth_for` = the flag's depth.
pub fn depth_for(o: &NgramOpt, matched: usize) -> usize {
    if matched >= o.min {
        o.depth
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ink32_the_lookup_takes_the_longest_most_recent_match_and_never_itself() {
        let mut h = History::new(3);
        // 1 2 3 4 5 | 9 2 3 4 7 | 1 2 3 4 -> the suffix [1 2 3 4] matches at 0..4 (len 4, then 5)
        // and [9 2 3 4] at 5..9 (len 3, then 7): the longer wins
        h.push(&[1, 2, 3, 4, 5, 9, 2, 3, 4, 7, 1, 2, 3, 4]);
        let (m, d) = h.propose(8, 3);
        assert_eq!(m, 4);
        assert_eq!(d, vec![5, 9, 2]);
        // equal length -> the most recent
        let mut h = History::new(3);
        h.push(&[2, 3, 4, 5, 0, 2, 3, 4, 6, 0, 2, 3, 4]);
        let (m, d) = h.propose(8, 2);
        assert_eq!((m, d), (4, vec![6, 0]));
        // a suffix seen only as itself: no match
        let mut h = History::new(3);
        h.push(&[1, 2, 3]);
        assert_eq!(h.propose(8, 3), (0, vec![]));
        // the copy is capped at the end of the history (it may run into the suffix)
        let mut h = History::new(3);
        h.push(&[7, 7, 7, 7]);
        let (m, d) = h.propose(8, 5);
        assert_eq!(m, 3);
        assert_eq!(d, vec![7]);
    }

    #[test]
    fn ink32_the_flags_and_the_depth_law() {
        assert_eq!(NgramOpt::parse("off", 3, 8, 7).unwrap(), None);
        let o = NgramOpt::parse("on", 3, 8, 7).unwrap().unwrap();
        assert!(NgramOpt::parse("on", 3, 8, 8).is_err());
        assert!(NgramOpt::parse("on", 0, 8, 7).is_err());
        assert!(NgramOpt::parse("maybe", 3, 8, 7).is_err());
        assert_eq!(depth_for(&o, 2), 0);
        assert_eq!(depth_for(&o, 3), 7);
        assert_eq!(depth_for(&o, 8), 7);
        let o2 = NgramOpt { depth: 4, ..o };
        assert_eq!(depth_for(&o2, 8), 4);
    }
}
