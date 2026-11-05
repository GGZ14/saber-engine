//! The autotune cache: the warm-up discard, the loud miss, and persistence.

mod common;

use saber_plan::{
    medians_after_warmup, noise_floor_us, AutotuneCache, AutotuneKey, AutotuneRecord, Autotuner,
    Measurement, PlanError, ShapeKeys, TieBreak, VariantParams,
};

fn shape(m: i64) -> ShapeKeys {
    ShapeKeys::new()
        .with("m", m)
        .with("n", 5120i64)
        .with("k", 5120i64)
        .with("w_dtype", "mxfp4")
}

fn key(m: i64, cap: u32) -> AutotuneKey {
    AutotuneKey::new(
        "gemm_w4a8",
        shape(m),
        cap,
        "planhash",
        "pluginhash",
        "policy",
    )
    .unwrap()
}

fn tuner() -> Autotuner {
    Autotuner::new(AutotuneCache::new(), common::fixture_provenance())
}

#[test]
fn a_key_without_m_is_refused() {
    let no_m = ShapeKeys::new().with("n", 5120i64).with("k", 5120i64);
    let err = AutotuneKey::new("gemm_w4a8", no_m, 210, "p", "b", "policy").unwrap_err();
    assert!(matches!(err, PlanError::ShapeMissingM { .. }), "{err}");
    assert!(
        err.to_string().contains("flips per shape AND per M"),
        "{err}"
    );
}

#[test]
fn the_first_sample_of_the_first_variant_is_discarded() {
    // The 09-xx measurement this protocol exists for: the first-measured variant per shape reads
    // ~17% inflated. Variant 1 is genuinely the faster kernel (100 us) but its first sample is
    // 117; variant 2 is a flat 105. Without the discard the sweep would pick the slower kernel and
    // cache that answer forever.
    let sweep = vec![
        Measurement::new(
            "gemm_w4a8_mxfp4_decode",
            VariantParams::new(1),
            vec![117.0, 100.0],
        ),
        Measurement::new(
            "gemm_w4a8_mxfp4_wperm",
            VariantParams::new(2),
            vec![105.0, 105.0],
        ),
    ];

    let naive_first_variant_median = 108.5; // (117 + 100) / 2, what no discard would give
    assert!(
        naive_first_variant_median > 105.0,
        "the trap this protocol avoids"
    );

    let medians = medians_after_warmup(&sweep).unwrap();
    assert_eq!(medians, vec![100.0, 105.0]);

    let mut tuner = tuner();
    let record = tuner
        .record_measurements(key(40, 210), sweep)
        .unwrap()
        .clone();
    assert_eq!(record.impl_name, "gemm_w4a8_mxfp4_decode");
    assert_eq!(record.variant.variant_id, 1);
    assert_eq!(record.measured_us, 100.0);
    assert_eq!(record.samples_kept, 1, "the warm-up sample is not counted");
    let runner_up = record.runner_up.as_ref().unwrap();
    assert_eq!(runner_up.variant_id, 2);
    assert_eq!(runner_up.measured_us, 105.0);
    assert!((record.margin_pct().unwrap() - 5.0).abs() < 1e-9);
}

#[test]
fn only_the_first_variant_loses_a_sample() {
    let sweep = vec![
        Measurement::new("a", VariantParams::new(1), vec![200.0, 100.0, 100.0, 300.0]),
        Measurement::new("b", VariantParams::new(2), vec![50.0, 90.0, 70.0]),
    ];
    // a: drop 200, median(100, 100, 300) = 100.  b: keeps all three, median(50, 70, 90) = 70.
    assert_eq!(medians_after_warmup(&sweep).unwrap(), vec![100.0, 70.0]);
    let mut tuner = tuner();
    let record = tuner.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.impl_name, "b");
    assert_eq!(record.samples_kept, 3);
}

#[test]
fn an_even_sample_count_takes_the_midpoint() {
    let sweep = vec![
        Measurement::new(
            "a",
            VariantParams::new(1),
            vec![999.0, 10.0, 20.0, 30.0, 40.0],
        ),
        Measurement::new("b", VariantParams::new(2), vec![100.0, 200.0]),
    ];
    assert_eq!(medians_after_warmup(&sweep).unwrap(), vec![25.0, 150.0]);
}

#[test]
fn a_single_sample_on_the_first_variant_is_an_error() {
    let sweep = vec![
        Measurement::new("a", VariantParams::new(1), vec![117.0]),
        Measurement::new("b", VariantParams::new(2), vec![105.0]),
    ];
    let err = medians_after_warmup(&sweep).unwrap_err();
    assert!(matches!(err, PlanError::Autotune(_)), "{err}");
    assert!(err.to_string().contains("at least two samples"), "{err}");
    assert!(err.to_string().contains("~17% inflated"), "{err}");
}

#[test]
fn an_empty_sweep_is_an_error() {
    let err = medians_after_warmup(&[]).unwrap_err();
    assert!(err.to_string().contains("decides nothing"), "{err}");
}

#[test]
fn a_single_variant_sweep_has_no_runner_up() {
    let sweep = vec![Measurement::new(
        "a",
        VariantParams::new(1),
        vec![117.0, 100.0, 101.0, 99.0],
    )];
    let mut tuner = tuner();
    let record = tuner.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.measured_us, 100.0);
    assert!(record.runner_up.is_none());
    assert!(record.margin_pct().is_none());
}

#[test]
fn a_miss_is_not_tuned_and_says_what_to_measure() {
    let cache = AutotuneCache::new();
    let k = key(40, 210);
    let err = cache.lookup(&k).unwrap_err();
    assert_eq!(*err.key, k);

    let err = cache
        .require(&k, &["gemm_w4a8_mxfp4_decode", "gemm_w4a8_mxfp4_wperm"])
        .unwrap_err();
    let PlanError::NotTuned(not_tuned) = err else {
        panic!("expected NotTuned")
    };
    let rendered = not_tuned.to_string();
    assert!(
        rendered.starts_with("not tuned: op `gemm_w4a8`"),
        "{rendered}"
    );
    assert!(rendered.contains("m=40"), "{rendered}");
    assert!(rendered.contains("power cap 210 W"), "{rendered}");
    assert!(
        rendered.contains("gemm_w4a8_mxfp4_decode, gemm_w4a8_mxfp4_wperm"),
        "{rendered}"
    );
    assert!(rendered.contains("error if not tuned"), "{rendered}");
}

#[test]
fn every_key_component_is_load_bearing() {
    let mut cache = AutotuneCache::new();
    let record = AutotuneRecord::new(
        "gemm_w4a8_mxfp4_decode",
        VariantParams::new(1),
        100.0,
        common::fixture_provenance(),
    );
    cache.insert(key(40, 210), record);
    assert!(cache.lookup(&key(40, 210)).is_ok());

    // A different M is a different record: the optimum flips per shape AND per M.
    assert!(cache.lookup(&key(48, 210)).is_err());
    // A record measured at a different cap is not valid here.
    assert!(cache.lookup(&key(40, 250)).is_err());
    // So is a different plan input hash or plugin build hash.
    let other_plan = AutotuneKey::new(
        "gemm_w4a8",
        shape(40),
        210,
        "otherplan",
        "pluginhash",
        "policy",
    )
    .unwrap();
    assert!(cache.lookup(&other_plan).is_err());
    let other_plugin = AutotuneKey::new(
        "gemm_w4a8",
        shape(40),
        210,
        "planhash",
        "otherplugin",
        "policy",
    )
    .unwrap();
    assert!(cache.lookup(&other_plugin).is_err());
}

#[test]
fn json_persistence_round_trips() {
    let mut tuner = tuner();
    tuner
        .record_measurements(
            key(40, 210),
            vec![
                Measurement::new(
                    "a",
                    VariantParams::new(1).param("split_k", 4),
                    vec![117.0, 100.0],
                ),
                Measurement::new("b", VariantParams::new(2), vec![105.0, 105.0]),
            ],
        )
        .unwrap();
    tuner
        .record_measurements(
            key(8, 210),
            vec![Measurement::new(
                "a",
                VariantParams::new(7),
                vec![99.0, 50.0, 52.0],
            )],
        )
        .unwrap();
    let cache = tuner.into_cache();
    assert_eq!(cache.len(), 2);

    let (_dir_cache, path) = common::temp_path("cache");
    cache.save(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"schema_version\": 1"), "{text}");
    assert!(text.contains("\"power_cap_w\": 210"), "{text}");

    let loaded = AutotuneCache::load(&path).unwrap();
    assert_eq!(loaded, cache);
    assert_eq!(loaded.lookup(&key(40, 210)).unwrap().variant.split_k(), 4);
    // The file is a stable diff: saving it again is byte-identical.
    let (_dir_cache2, second) = common::temp_path("cache2");
    loaded.save(&second).unwrap();
    assert_eq!(std::fs::read_to_string(&second).unwrap(), text);

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(&second).unwrap();
}

#[test]
fn a_missing_cache_file_loads_empty_and_then_fails_loudly() {
    let (_dir_absent, path) = common::temp_path("absent");
    let cache = AutotuneCache::load(&path).unwrap();
    assert!(cache.is_empty());
    assert!(cache.lookup(&key(40, 210)).is_err());
}

#[test]
fn a_corrupt_cache_file_is_an_error_not_an_empty_cache() {
    let (_dir_corrupt, path) = common::temp_path("corrupt");
    std::fs::write(&path, "{ this is not json").unwrap();
    let err = AutotuneCache::load(&path).unwrap_err();
    assert!(matches!(err, PlanError::CacheJson { .. }), "{err}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn split_k_defaults_to_one_and_is_never_zero() {
    assert_eq!(VariantParams::new(1).split_k(), 1);
    assert_eq!(VariantParams::new(1).param("split_k", 0).split_k(), 1);
    assert_eq!(VariantParams::new(1).param("split_k", 4).split_k(), 4);
}

// ---------------------------------------------------------------- the below-floor tie-break
//
// Most margins in a real sweep are tiny: of the 32 shapes tuned on this box, 11 were decided by
// 0.0-1.6%, which the +-2.2% / +-0.5 us floor says is nothing. Taking the fastest median anyway
// makes the binding a coin flip, and the plan hash churns with the weather.

/// Two variants at `slow`/`fast`, the second one flagged as the reference launcher's pick when
/// `reference_is_second` is set. Three samples each after the first variant's warm-up.
fn tied_sweep(first_us: f64, first_id: u32, second_us: f64, second_id: u32) -> Vec<Measurement> {
    vec![
        Measurement::new(
            "gemm_w4a8_mxfp4_decode",
            VariantParams::new(first_id),
            vec![first_us * 1.17, first_us, first_us, first_us],
        ),
        Measurement::new(
            "gemm_w4a8_mxfp4_decode",
            VariantParams::new(second_id),
            vec![second_us, second_us, second_us],
        ),
    ]
}

#[test]
fn the_noise_floor_is_the_larger_of_the_band_and_the_absolute_floor() {
    // At 100 us the 2.2% band is what binds; at 17 us (the smallest kernels in the gate) the
    // 0.5 us absolute floor does, which is the rule the ledger adopted after two 17 us cells
    // landed on opposite sides of 1.000 between runs.
    assert!((noise_floor_us(100.0) - 2.2).abs() < 1e-9);
    assert!((noise_floor_us(17.0) - 0.5).abs() < 1e-9);
}

#[test]
fn a_tie_inside_the_floor_goes_to_the_reference_launcher_even_though_it_measured_slower() {
    // 100.0 vs 100.9: 0.9%, inside the 2.2% band. The measurement decided nothing.
    let mut sweep = tied_sweep(100.0, 7, 100.9, 9);
    sweep[1] = sweep[1].clone().reference();

    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap().clone();
    assert_eq!(
        record.variant.variant_id, 9,
        "the reference launcher's variant"
    );
    assert_eq!(record.measured_us, 100.9);
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher)
    );
    // The runner-up is the faster one, so the margin is negative -- and inside the floor.
    let margin = record.margin_pct().unwrap();
    assert!(margin < 0.0, "{margin}");
    assert!(margin.abs() < saber_plan::NOISE_FLOOR_PCT, "{margin}");
}

#[test]
fn a_tie_with_no_reference_variant_takes_the_lowest_variant_id() {
    // Same numbers, nothing flagged: the winner must still be the same on every run and every
    // machine, so it is the lowest id -- arbitrary, but total.
    let mut t = tuner();
    let record = t
        .record_measurements(key(40, 210), tied_sweep(100.0, 9, 100.9, 7))
        .unwrap()
        .clone();
    assert_eq!(record.variant.variant_id, 7);
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::LowestVariantId)
    );

    // And it does not depend on the order the sweep measured them in.
    let mut swapped = tuner();
    let other = swapped
        .record_measurements(key(40, 210), tied_sweep(100.9, 7, 100.0, 9))
        .unwrap();
    assert_eq!(other.variant.variant_id, 7);
}

#[test]
fn a_margin_that_clears_the_floor_is_decided_by_the_measurement_not_by_the_reference() {
    // 100.0 vs 103.0: 3%, outside the band. The reference launcher does not get a vote.
    let mut sweep = tied_sweep(100.0, 7, 103.0, 9);
    sweep[1] = sweep[1].clone().reference();

    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.variant.variant_id, 7);
    assert_eq!(record.measured_us, 100.0);
    assert_eq!(record.provenance.tie_broken_by, None, "no tie to break");
    assert!(record.margin_pct().unwrap() > saber_plan::NOISE_FLOOR_PCT);
}

#[test]
fn on_a_small_kernel_the_absolute_floor_decides_what_counts_as_a_tie() {
    // 17.0 vs 17.4 is 2.35% -- outside the percentage band -- but 0.4 us, inside the absolute
    // floor. Tie: the floor is the larger of the two, which is what makes small kernels judgeable.
    let mut sweep = tied_sweep(17.0, 7, 17.4, 9);
    sweep[1] = sweep[1].clone().reference();
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.variant.variant_id, 9);
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher)
    );

    // Negative control: 0.6 us at the same size clears both halves of the floor, so it is a
    // result and the slower reference loses.
    let mut sweep = tied_sweep(17.0, 7, 17.6, 9);
    sweep[1] = sweep[1].clone().reference();
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.variant.variant_id, 7);
    assert_eq!(record.provenance.tie_broken_by, None);
}

#[test]
fn a_tie_that_the_fastest_variant_wins_still_records_how_it_was_broken() {
    // The reference launcher's pick IS the fastest, by less than the floor. The winner would be
    // the same either way -- but the record has to say the margin did not decide it, or a reader
    // concludes the sweep measured a difference that it did not.
    let mut sweep = tied_sweep(100.0, 7, 100.9, 9);
    sweep[0] = sweep[0].clone().reference();
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.variant.variant_id, 7);
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher)
    );
}

#[test]
fn a_single_variant_sweep_is_not_a_tie() {
    let sweep = vec![Measurement::new(
        "a",
        VariantParams::new(1),
        vec![117.0, 100.0, 100.0],
    )];
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.provenance.tie_broken_by, None);
}

/// The paired estimator is the whole reason the sweep interleaves.
///
/// Both variants drift upward together over the rounds -- the machine warms, prod takes a
/// request -- and `b` is a flat 1.5 us behind `a` in every single round. Compared as two
/// independent medians that difference reads as whatever the drift happens to leave; compared per
/// round it reads 1.5, which is what it is, and 1.5 is inside the 2.2 us floor at 100 us.
#[test]
fn a_tie_is_judged_on_the_paired_difference_not_on_two_independent_medians() {
    // Round 0 carries the first variant's warm-up and is dropped from the pairing.
    let a = vec![117.0, 100.0, 101.0, 102.0, 103.0, 104.0];
    let b: Vec<f64> = a.iter().map(|x| x + 1.5).collect();
    let sweep = vec![
        Measurement::new("a", VariantParams::new(9), a),
        Measurement::new("b", VariantParams::new(7), b).reference(),
    ];
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher),
        "1.5 us at 100 us is inside the 2.2% floor"
    );
    assert_eq!(record.variant.variant_id, 7);
}

/// The negative control for the same mechanism: a *paired* gap that clears the floor is a result,
/// however the two medians are dragged around by drift.
#[test]
fn a_paired_gap_above_the_floor_is_a_result_however_the_medians_drift() {
    let a = vec![117.0, 100.0, 102.0, 104.0, 106.0];
    let b: Vec<f64> = a.iter().map(|x| x + 3.0).collect();
    let sweep = vec![
        Measurement::new("a", VariantParams::new(9), a),
        Measurement::new("b", VariantParams::new(7), b).reference(),
    ];
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(
        record.provenance.tie_broken_by, None,
        "3 us > the 2.2 us floor"
    );
    assert_eq!(record.variant.variant_id, 9, "the measurement decided");
}

/// The boundary itself: a hair inside the floor is a tie, a hair past it is not.
///
/// This is the edge the plan hash turns on, so it is pinned. It is also why the ledger's residual
/// exists: a pair whose true difference sits *at* this boundary gets classified by whichever side
/// the run's noise lands on, and no threshold can fix that.
#[test]
fn the_tie_boundary_is_the_ledgers_floor_and_nothing_else() {
    let a = vec![117.0, 100.0, 100.0, 100.0, 100.0];
    let floor = noise_floor_us(100.0);
    let just_inside: Vec<f64> = a.iter().map(|x| x + floor - 0.01).collect();
    let sweep = vec![
        Measurement::new("a", VariantParams::new(9), a.clone()),
        Measurement::new("b", VariantParams::new(7), just_inside).reference(),
    ];
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher),
        "a hair inside the floor is still indistinguishable"
    );

    let past_it: Vec<f64> = a.iter().map(|x| x + floor + 0.01).collect();
    let sweep = vec![
        Measurement::new("a", VariantParams::new(9), a),
        Measurement::new("b", VariantParams::new(7), past_it).reference(),
    ];
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(record.provenance.tie_broken_by, None, "past it is a result");
    assert_eq!(record.variant.variant_id, 9);
}

/// Ragged rows cannot be paired, and the fallback is the difference of medians rather than a
/// panic or a silently skipped comparison.
#[test]
fn rows_that_cannot_be_paired_fall_back_to_the_medians() {
    let sweep = vec![
        Measurement::new("a", VariantParams::new(9), vec![117.0, 100.0, 100.0, 100.0]),
        Measurement::new("b", VariantParams::new(7), vec![100.9, 100.9]).reference(),
    ];
    let mut t = tuner();
    let record = t.record_measurements(key(40, 210), sweep).unwrap();
    assert_eq!(
        record.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher)
    );
}

/// Two passes that DISAGREE: the cache does not decide, and says so.
///
/// This is I17's `attn_decode_paged` at batch 4 in miniature — one variant measuring 81.3 µs in
/// one pass and 73.96 in the next, the winner flipping, and a tie-break silently deciding a 9%
/// swing. The record must bind what the RULES would pick and carry both passes.
#[test]
fn two_passes_that_disagree_leave_the_cell_to_the_rules() {
    let mut t = tuner();
    // Pass 1: variant A wins outright. Pass 2: variant B does, by more than the floor.
    let a1 = Measurement::new("impl_a", VariantParams::new(7), vec![100.0; 6]);
    let b1 = Measurement::new("impl_b", VariantParams::new(9), vec![140.0; 6]).reference();
    let a2 = Measurement::new("impl_a", VariantParams::new(7), vec![150.0; 6]);
    let b2 = Measurement::new("impl_b", VariantParams::new(9), vec![100.0; 6]).reference();

    let rec = t
        .record_two_passes(key(5, 210), vec![a1, b1], vec![a2, b2])
        .expect("two passes record")
        .clone();
    let u = rec
        .unstable
        .expect("the passes disagreed, so the record must say so");
    assert_eq!(u.first.variant_id, 7);
    assert_eq!(u.second.variant_id, 9);
    // 100 -> 150 on the SAME variant is 50%, and that is the number that says how far outside
    // the floor the cell sits.
    assert!(
        (u.worst_swing_pct - 50.0).abs() < 0.001,
        "{}",
        u.worst_swing_pct
    );
    // The rules keep the cell: `impl_b` is the reference launcher's, so it is what binds — even
    // though it lost the first pass by 40%.
    assert_eq!(rec.impl_name, "impl_b");
    assert_eq!(rec.variant.variant_id, 9);
    assert_eq!(
        rec.provenance.tie_broken_by,
        Some(TieBreak::ReferenceLauncher)
    );
}

/// Two passes that AGREE cost the GPU time and buy the claim: the record is the ordinary one.
#[test]
fn two_passes_that_agree_record_what_one_pass_would_have() {
    let mut t = tuner();
    let mk = || {
        (
            Measurement::new("impl_a", VariantParams::new(7), vec![100.0; 6]),
            Measurement::new("impl_b", VariantParams::new(9), vec![140.0; 6]),
        )
    };
    let (a1, b1) = mk();
    let (a2, b2) = mk();
    let two = t
        .record_two_passes(key(5, 210), vec![a1, b1], vec![a2, b2])
        .expect("record")
        .clone();
    assert!(two.unstable.is_none());
    assert_eq!(two.variant.variant_id, 7);

    // NEGATIVE CONTROL on the instrument itself: the SINGLE-pass protocol cannot tell the two
    // situations apart. Fed the disagreeing pair's first pass alone it produces a confident
    // record for variant 7, with no hint that a second pass would have named 9.
    let mut one = tuner();
    let rec = one
        .record_measurements(
            key(5, 210),
            vec![
                Measurement::new("impl_a", VariantParams::new(7), vec![100.0; 6]),
                Measurement::new("impl_b", VariantParams::new(9), vec![140.0; 6]).reference(),
            ],
        )
        .expect("record");
    assert_eq!(rec.variant.variant_id, 7);
    assert!(
        rec.unstable.is_none(),
        "a single pass has nothing to compare against"
    );
}

/// Two passes over different variant sets are not comparable, and are refused rather than
/// compared by position.
#[test]
fn two_passes_over_different_variants_are_refused() {
    let mut t = tuner();
    let first = vec![Measurement::new(
        "impl_a",
        VariantParams::new(7),
        vec![100.0; 6],
    )];
    let second = vec![Measurement::new(
        "impl_b",
        VariantParams::new(9),
        vec![100.0; 6],
    )];
    assert!(t.record_two_passes(key(5, 210), first, second).is_err());
}
