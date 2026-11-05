//! **P27-W2-3: THE SAMPLED TREE ACCEPTANCE LAW** (`--spec-tree-sampled on`), gated on the host.
//!
//! The law ([`TreePlan::drawn_verdict`]): at each verify row the request DRAWS the position that
//! row decides with ws-f40's keyed race (`SeededNoise(seed, position, choice)` through the
//! production `Sampler`), descends into the child whose token is the draw, else commits the draw.
//!
//! What is checked, on a toy vocabulary (6 tokens, T 0.7, a target and an imperfect drafter that
//! are hash functions of the prefix, best-first 9-node trees of <= 3 children, depth <= 7):
//!
//! (a) **stream identity**: for every seed the tree's committed stream == the width-1 stream
//!     (each position drawn once under its own key) token for token -- the law commits the
//!     target's own draws, the tree moves only step boundaries;
//! (b) **distribution**: a chi-square of the first 3 committed tokens (20,000 seeds) against the
//!     target's exact joint law PASSES for the drawn law and for recursive rejection sampling
//!     (RRS, the literature's multi-draft law with residual renormalisation), and FAILS for the
//!     CONTROL -- RRS with the residual omitted (a rejected child's mass not removed, the bonus
//!     drawn from `p`), a deliberately biased law;
//! (c) **acceptance**: the first step's mean committed tokens equals the exact enumeration
//!     `1 + sum over nodes prod p(path)` within 4 sigma, and the tree beats the chain of 7
//!     (the drafter's greedy path) at the same law.

use saber_engine::spec_tree::{Node, SpecTree, TreePlan};
use saber_serve::sampler::{NoiseSource, PenaltyContext, RowParams, Sampler, SeededNoise};
use saber_serve::sampling::SamplingParams;

/// One step's law: (plan, committed prefix, seed) -> the committed tokens.
type StepLaw<'a> = dyn FnMut(&TreePlan, &[i64], u64) -> Vec<i64> + 'a;

const V: usize = 6;
const T: f32 = 0.7;
const SEEDS: u64 = 20_000;
const L: usize = 3;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn key(prefix: &[i64], salt: u64) -> u64 {
    prefix
        .iter()
        .fold(splitmix(salt), |h, t| splitmix(h ^ (*t as u64 + 1)))
}

fn unit(h: u64) -> f32 {
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

/// The toy target's raw logits after `prefix`.
fn target_logits(prefix: &[i64]) -> Vec<f32> {
    let h = key(prefix, 7);
    (0..V)
        .map(|v| {
            let base = unit(splitmix(h ^ (v as u64 * 0x1234_5678))) * 4.0 - 2.0;
            base + if (h % V as u64) as usize == v {
                1.5
            } else {
                0.0
            }
        })
        .collect()
}

/// The toy drafter: the target's logits plus its own error.
fn drafter_logits(prefix: &[i64]) -> Vec<f32> {
    let h = key(prefix, 99);
    target_logits(prefix)
        .iter()
        .enumerate()
        .map(|(v, l)| l + (unit(splitmix(h ^ v as u64)) * 2.0 - 1.0) * 1.2)
        .collect()
}

fn softmax_t(l: &[f32]) -> Vec<f64> {
    let m = l.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let e: Vec<f64> = l.iter().map(|x| f64::from((x - m) / T).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn params() -> RowParams {
    let sp = SamplingParams {
        temperature: T,
        ..SamplingParams::default()
    };
    RowParams::resolve(&sp, V)
}

/// The production draw: position `pos` of request `seed` over the target's row after `prefix`.
fn draw(sampler: &Sampler, seed: u64, pos: u64, prefix: &[i64]) -> i64 {
    let mut l = target_logits(prefix);
    let mut noise = SeededNoise::new(seed, pos, 0);
    i64::from(
        sampler
            .sample_dense(&mut l, &params(), &PenaltyContext::default(), &mut noise)
            .expect("a dense toy row samples")
            .token_id,
    )
}

/// Best-first 9 nodes, <= 3 children, depth <= 7, by the drafter's path probability.
fn best_first_tree(prefix: &[i64]) -> TreePlan {
    // frontier: (score, parent node, depth, token, path tokens)
    let mut nodes: Vec<Node> = Vec::new();
    let mut toks: Vec<i64> = Vec::new();
    let mut paths: Vec<Vec<i64>> = Vec::new();
    let mut kids = vec![0usize; 1];
    let mut frontier: Vec<(f64, Option<usize>, u32, i64)> = Vec::new();
    let push_children = |frontier: &mut Vec<(f64, Option<usize>, u32, i64)>,
                         parent: Option<usize>,
                         depth: u32,
                         score: f64,
                         path: &[i64]| {
        let mut full = prefix.to_vec();
        full.extend_from_slice(path);
        let q = softmax_t(&drafter_logits(&full));
        for (v, p) in q.iter().enumerate() {
            frontier.push((score * p, parent, depth, v as i64));
        }
    };
    push_children(&mut frontier, None, 1, 1.0, &[]);
    while nodes.len() < 9 && !frontier.is_empty() {
        let (bi, _) =
            frontier
                .iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |(bi, bs), (i, f)| {
                    if f.0 > bs {
                        (i, f.0)
                    } else {
                        (bi, bs)
                    }
                });
        let (score, parent, depth, tok) = frontier.swap_remove(bi);
        let slot = parent.map_or(0, |p| p + 1);
        if kids[slot] >= 3 {
            continue;
        }
        kids[slot] += 1;
        let mut path = parent.map_or_else(Vec::new, |p| paths[p].clone());
        path.push(tok);
        nodes.push(Node {
            depth,
            cand: 0,
            parent,
        });
        toks.push(tok);
        paths.push(path.clone());
        kids.push(0);
        if depth < 7 {
            push_children(
                &mut frontier,
                Some(nodes.len() - 1),
                depth + 1,
                score,
                &path,
            );
        }
    }
    plan_of_tokens(SpecTree::new(nodes).expect("a well-formed tree"), &toks)
}

/// A plan whose row `i + 1` embeds `toks[i]` (`TreePlan::new` visits the nodes in row order).
fn plan_of_tokens(tree: SpecTree, toks: &[i64]) -> TreePlan {
    let at = std::cell::Cell::new(0usize);
    TreePlan::new(tree, -1, |_| {
        let i = at.get();
        at.set(i + 1);
        toks[i]
    })
    .expect("<= 10 rows")
}

/// The drafter's greedy chain of 7 as a plan (the chain law's shape).
fn chain_tree(prefix: &[i64]) -> TreePlan {
    let mut path = prefix.to_vec();
    let mut toks = Vec::new();
    for _ in 0..7 {
        let q = drafter_logits(&path);
        let t = (0..V).max_by(|a, b| q[*a].total_cmp(&q[*b])).unwrap_or(0) as i64;
        toks.push(t);
        path.push(t);
    }
    plan_of_tokens(SpecTree::chain(&[0; 7]).expect("a chain"), &toks)
}

/// One sequence of `n` tokens under a plan builder and a step law; returns (stream, tokens of
/// the first step).
fn generate(
    seed: u64,
    n: usize,
    plan_of: &dyn Fn(&[i64]) -> TreePlan,
    law: &mut StepLaw<'_>,
) -> (Vec<i64>, usize) {
    let mut out: Vec<i64> = Vec::new();
    let mut first = 0;
    while out.len() < n {
        let plan = plan_of(&out);
        let c = law(&plan, &out, seed);
        if first == 0 {
            first = c.len();
        }
        out.extend(c);
    }
    out.truncate(n);
    (out, first)
}

/// THE LAW under test: the engine's own walk with the production keyed draw.
fn drawn_step(sampler: &Sampler) -> impl FnMut(&TreePlan, &[i64], u64) -> Vec<i64> + '_ {
    move |plan, prefix, seed| {
        let at = prefix.len() as u64;
        let w = plan.drawn_verdict(
            &mut |_row, depth, path| {
                let mut full = prefix.to_vec();
                full.extend_from_slice(path);
                Some((draw(sampler, seed, at + depth as u64, &full), false))
            },
            &|_| -1,
        );
        assert!(!w.undecided);
        assert_eq!(w.decided, w.verdict.committed.len());
        // the path's node tokens are the committed draws before the last
        for (j, r) in w.verdict.path_rows.iter().enumerate().skip(1) {
            assert_eq!(plan.tokens()[*r], w.verdict.committed[j - 1]);
        }
        w.verdict.committed
    }
}

/// Uniform in (0, 1] and a race over `p` from one keyed source.
fn uniform(src: &mut SeededNoise, i: usize) -> f64 {
    (-f64::from(src.exponential(i))).exp()
}

fn race(p: &[f64], src: &mut SeededNoise) -> i64 {
    (0..p.len())
        .filter(|v| p[*v] > 0.0)
        .map(|v| (v, p[v] / f64::from(src.exponential(1000 + v))))
        .fold(
            (0usize, f64::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a },
        )
        .0 as i64
}

/// Recursive rejection sampling over the tree (deterministic distinct children): child `c` is
/// accepted with probability `r(c) / sum r` of the current residual `r`; a rejected child's mass
/// is removed; all rejected -> a draw from the residual. `residual = false` is THE CONTROL: the
/// rejected mass is not removed and the bonus is drawn from `p` (biased).
fn rrs_step(residual: bool) -> impl FnMut(&TreePlan, &[i64], u64) -> Vec<i64> {
    move |plan, prefix, seed| {
        let mut row = 0usize;
        let mut path: Vec<i64> = Vec::new();
        loop {
            let pos = prefix.len() + path.len();
            let mut full = prefix.to_vec();
            full.extend_from_slice(&path);
            let p = softmax_t(&target_logits(&full));
            let mut r = p.clone();
            let mut src = SeededNoise::new(seed ^ 0xA5A5, pos as u64, 0);
            let mut next = None;
            for (i, c) in plan.children(row).into_iter().enumerate() {
                let t = plan.tokens()[c] as usize;
                let s: f64 = r.iter().sum();
                let acc = if residual { r[t] / s } else { p[t] };
                if uniform(&mut src, i) <= acc {
                    next = Some(c);
                    break;
                }
                if residual {
                    r[t] = 0.0;
                }
            }
            match next {
                Some(c) => {
                    path.push(plan.tokens()[c]);
                    row = c;
                }
                None => {
                    path.push(race(if residual { &r } else { &p }, &mut src));
                    return path;
                }
            }
        }
    }
}

/// The exact joint law of the first `L` tokens.
fn exact_joint() -> Vec<f64> {
    let mut out = vec![0.0; V.pow(L as u32)];
    for (cell, o) in out.iter_mut().enumerate() {
        let mut prefix = Vec::new();
        let mut pr = 1.0;
        let mut c = cell;
        for _ in 0..L {
            let t = c % V;
            c /= V;
            pr *= softmax_t(&target_logits(&prefix))[t];
            prefix.push(t as i64);
        }
        *o = pr;
    }
    out
}

fn cell_of(s: &[i64]) -> usize {
    s.iter().rev().fold(0, |a, t| a * V + *t as usize)
}

/// Pearson chi-square over cells with expected count >= 5 (the rest pooled); returns (stat, df,
/// the 0.9995 critical value by Wilson-Hilferty).
fn chi_square(counts: &[u64], probs: &[f64], n: u64) -> (f64, usize, f64) {
    let (mut stat, mut df) = (0.0, 0usize);
    let (mut po, mut pe) = (0.0, 0.0);
    for (c, p) in counts.iter().zip(probs) {
        let e = p * n as f64;
        if e >= 5.0 {
            stat += (*c as f64 - e).powi(2) / e;
            df += 1;
        } else {
            po += *c as f64;
            pe += e;
        }
    }
    if pe > 0.0 {
        stat += (po - pe).powi(2) / pe.max(1e-9);
        df += 1;
    }
    let k = (df - 1) as f64;
    let z = 3.29;
    let crit = k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3);
    (stat, df - 1, crit)
}

fn joint_counts(law: &mut StepLaw<'_>) -> Vec<u64> {
    let mut counts = vec![0u64; V.pow(L as u32)];
    for seed in 0..SEEDS {
        let (s, _) = generate(seed, L, &best_first_tree, law);
        counts[cell_of(&s)] += 1;
    }
    counts
}

#[test]
fn p27w2_the_drawn_tree_commits_the_width1_stream_draw_for_draw() {
    let sampler = Sampler::new();
    let mut law = drawn_step(&sampler);
    let mut steps_tree = 0usize;
    for seed in 0..2_000u64 {
        let n = 24;
        let (tree, _) = generate(seed, n, &best_first_tree, &mut law);
        // width 1: every position drawn once under its own key
        let mut w1: Vec<i64> = Vec::new();
        while w1.len() < n {
            let t = draw(&sampler, seed, w1.len() as u64, &w1);
            w1.push(t);
        }
        assert_eq!(
            tree, w1,
            "seed {seed}: the tree's stream is not the width-1 stream"
        );
        steps_tree += 1;
    }
    assert_eq!(steps_tree, 2_000);
    // CONTROL: the same walk keyed by the STEP (the pre-F40 key class) must leave the stream.
    let mut moved = 0;
    for seed in 0..200u64 {
        let mut step = 0u64;
        let mut ctl = |plan: &TreePlan, prefix: &[i64], seed: u64| {
            step += 1;
            let s = step;
            plan.drawn_verdict(
                &mut |_row, depth, path| {
                    let mut full = prefix.to_vec();
                    full.extend_from_slice(path);
                    Some((draw(&sampler, seed, s * 64 + depth as u64, &full), false))
                },
                &|_| -1,
            )
            .verdict
            .committed
        };
        let (c, _) = generate(seed, 24, &best_first_tree, &mut ctl);
        let mut w1: Vec<i64> = Vec::new();
        while w1.len() < 24 {
            let t = draw(&sampler, seed, w1.len() as u64, &w1);
            w1.push(t);
        }
        moved += usize::from(c != w1);
    }
    assert!(
        moved > 150,
        "the step-keyed control kept the stream on {} of 200 seeds",
        200 - moved
    );
}

#[test]
fn p27w2_the_drawn_tree_and_rrs_pass_the_chi_square_and_the_biased_control_fails() {
    let probs = exact_joint();
    let sampler = Sampler::new();
    let (s, df, crit) = chi_square(&joint_counts(&mut drawn_step(&sampler)), &probs, SEEDS);
    eprintln!("drawn tree: chi2 {s:.1} df {df} crit {crit:.1}");
    assert!(
        s < crit,
        "the drawn tree law fails the chi-square: {s} >= {crit}"
    );
    let (s, df, crit) = chi_square(&joint_counts(&mut rrs_step(true)), &probs, SEEDS);
    eprintln!("rrs: chi2 {s:.1} df {df} crit {crit:.1}");
    assert!(
        s < crit,
        "recursive rejection sampling fails the chi-square: {s} >= {crit}"
    );
    let (s, df, crit) = chi_square(&joint_counts(&mut rrs_step(false)), &probs, SEEDS);
    eprintln!("CONTROL (residual omitted): chi2 {s:.1} df {df} crit {crit:.1}");
    assert!(
        s > 4.0 * crit,
        "the biased control PASSED the chi-square: the test has no power"
    );
}

#[test]
fn p27w2_first_step_acceptance_is_the_exact_enumeration_and_beats_the_chain() {
    let sampler = Sampler::new();
    let exact = |plan_of: &dyn Fn(&[i64]) -> TreePlan| {
        let plan = plan_of(&[]);
        // 1 + sum over nodes of prod p(token | path) along the node's path
        let mut reach = vec![0.0f64; plan.rows()];
        reach[0] = 1.0;
        let mut e = 1.0;
        for r in 1..plan.rows() {
            let pr = plan.parent_row(r).unwrap_or(0);
            let path: Vec<i64> = plan.path_rows(pr)[1..]
                .iter()
                .map(|x| plan.tokens()[*x])
                .collect();
            let p = softmax_t(&target_logits(&path));
            reach[r] = reach[pr] * p[plan.tokens()[r] as usize];
            e += reach[r];
        }
        e
    };
    let mut law = drawn_step(&sampler);
    for (name, plan_of) in [
        ("tree", &best_first_tree as &dyn Fn(&[i64]) -> TreePlan),
        ("chain", &chain_tree as &dyn Fn(&[i64]) -> TreePlan),
    ] {
        let mut sum = 0.0;
        let mut sq = 0.0;
        for seed in 0..SEEDS {
            let plan = plan_of(&[]);
            let c = law(&plan, &[], seed).len() as f64;
            sum += c;
            sq += c * c;
        }
        let n = SEEDS as f64;
        let mean = sum / n;
        let sd = ((sq / n - mean * mean) / n).sqrt();
        let want = exact(plan_of);
        eprintln!("{name}: first-step tokens mean {mean:.4} exact {want:.4} (se {sd:.4})");
        assert!(
            (mean - want).abs() < 4.0 * sd + 1e-9,
            "{name}: {mean} vs exact {want}"
        );
    }
    assert!(exact(&best_first_tree) > exact(&chain_tree));
}
