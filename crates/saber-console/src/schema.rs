//! The flag schema the UI renders, generated from the CLI's clap structs.
//!
//! [`build`] walks a `clap::Command` (the CLI passes `Cli::command()`), takes the global flags
//! and the named subcommands' flags, and annotates each with a group (by the name rules in
//! [`GROUP_RULES`]) and a tier: `basic` for the hand-picked flags in [`BASIC`], `diagnostic`
//! for hidden flags and gate/diagnostic arms, `advanced` for the rest. Every engine flag is a
//! boot flag (`requires_restart`): the engine has no live configuration.
//!
//! A flag that matches no group rule lands in [`Schema::unclassified`]; the CLI's test
//! `console_schema_covers_every_flag` fails on any, so a new flag cannot reach the UI without a
//! place in it.

use std::any::TypeId;

use clap::builder::ValueParser;
use clap::{Arg, ArgAction, Command};
use serde::Serialize;

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Basic,
    Advanced,
    Diagnostic,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlagSpec {
    /// `--fn-max-ctx`
    pub name: String,
    /// `global`, or the subcommand (`serve`, `engine`).
    pub section: String,
    /// `bool`, `int`, `uint`, `float`, `path`, `string` or `enum`.
    pub kind: &'static str,
    pub default: Option<String>,
    pub choices: Vec<String>,
    /// Takes more than one value (a list flag).
    pub multiple: bool,
    /// A list flag's value delimiter (`,`): its values go in ONE argument. A list flag without
    /// one repeats (`--api-key a --api-key b`).
    pub delimiter: Option<String>,
    /// The value range the console enforces ([`RANGES`]); `None` = the type's own.
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub help: String,
    pub long_help: Option<String>,
    pub group: String,
    pub tier: Tier,
    pub requires_restart: bool,
    pub hidden: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Schema {
    pub version: u32,
    pub sections: Vec<String>,
    pub groups: Vec<&'static str>,
    pub flags: Vec<FlagSpec>,
    /// `section --flag` of every flag no rule placed (must be empty).
    pub unclassified: Vec<String>,
    /// Basic annotations that name no flag (must be empty).
    pub stale_annotations: Vec<String>,
}

/// The hand-annotated basic flags: (section, flag, group). The first screen of settings.
pub const BASIC: &[(&str, &str, &str)] = &[
    // Flash-Next (`saber engine --fn-serve`)
    ("engine", "--pack", "Model"),
    ("engine", "--tp", "Parallelism"),
    ("engine", "--fn-serve", "Server"),
    ("engine", "--fn-served-name", "Server"),
    ("engine", "--fn-max-ctx", "Context and KV cache"),
    ("engine", "--fn-thinking", "Chat and parsing"),
    ("engine", "--fn-prefix-tree", "Prefix cache"),
    ("engine", "--fn-kv-dtype", "Context and KV cache"),
    ("engine", "--fn-kv-tier", "Context and KV cache"),
    ("engine", "--fn-kv-dir", "Context and KV cache"),
    ("engine", "--fn-image-cache", "Vision"),
    ("engine", "--fn-experts", "Parallelism"),
    ("engine", "--fn-trunk", "Model"),
    // 27B (`saber serve --engine real`)
    ("serve", "--engine", "Model"),
    ("serve", "--pack", "Model"),
    ("serve", "--engine-tp", "Parallelism"),
    ("serve", "--engine-max-ctx", "Context and KV cache"),
    ("serve", "--engine-max-seqs", "Batching"),
    ("serve", "--host", "Server"),
    ("serve", "--port", "Server"),
    ("serve", "--served-model-name", "Server"),
    ("serve", "--enable-prefix-caching", "Prefix cache"),
    ("serve", "--vision", "Vision"),
    ("serve", "--tool-call-parser", "Chat and parsing"),
    ("serve", "--reasoning-parser", "Chat and parsing"),
    ("serve", "--chat-template", "Chat and parsing"),
    ("serve", "--engine-draft-pack", "Speculative decoding"),
];

/// Group rules: (section or `*`, name prefix, group). The FIRST match wins, so specific
/// prefixes come before general ones. There is deliberately no catch-all.
pub const GROUP_RULES: &[(&str, &str, &str)] = &[
    ("global", "--", "Logging and process"),
    // diagnostics and gates first: they share prefixes with everything
    ("*", "--fn-dump", "Diagnostics"),
    ("*", "--fn-tap", "Diagnostics"),
    ("*", "--fn-control", "Diagnostics"),
    ("*", "--fn-poison", "Diagnostics"),
    ("*", "--fn-route-trace", "Diagnostics"),
    ("*", "--fn-route-record", "Diagnostics"),
    ("*", "--fail-rank", "Diagnostics"),
    ("*", "--sample-dump", "Diagnostics"),
    ("*", "--fn-out", "Diagnostics"),
    ("*", "--fn-mtp-ref", "Diagnostics"),
    ("*", "--keep-stale-anchor-blocks", "Diagnostics"),
    ("*", "--step-keyed-draws", "Diagnostics"),
    ("*", "--macaq6-resid", "Diagnostics"),
    ("*", "--poison-boot", "Diagnostics"),
    ("*", "--race-check", "Diagnostics"),
    ("*", "--frame-digest", "Diagnostics"),
    ("*", "--whole-step-frame", "Diagnostics"),
    ("*", "--constant-seed-control", "Diagnostics"),
    // `saber engine`'s 27B harness (the gate runs, not the served boot)
    ("engine", "--prompt-ids", "Harness and gates"),
    ("engine", "--prompt-index", "Harness and gates"),
    ("engine", "--tokens", "Harness and gates"),
    ("engine", "--reference", "Harness and gates"),
    ("engine", "--min-margin", "Harness and gates"),
    ("engine", "--gen", "Harness and gates"),
    ("engine", "--warmup", "Harness and gates"),
    ("engine", "--batch-gen", "Harness and gates"),
    ("engine", "--gate-seqs", "Harness and gates"),
    ("engine", "--conc", "Harness and gates"),
    ("engine", "--messages", "Harness and gates"),
    ("engine", "--continuity-tokens", "Harness and gates"),
    ("engine", "--fork-", "Harness and gates"),
    ("engine", "--chunk-bisect", "Harness and gates"),
    ("engine", "--attribute", "Harness and gates"),
    ("engine", "--accept-", "Harness and gates"),
    ("engine", "--stop-after-accept", "Harness and gates"),
    ("engine", "--solo-", "Harness and gates"),
    ("engine", "--first-block-ids", "Harness and gates"),
    // the mock executor
    ("serve", "--mock", "Mock engine"),
    ("serve", "--step-delay-ms", "Mock engine"),
    ("serve", "--kv-gib", "Mock engine"),
    ("serve", "--state-gib", "Mock engine"),
    // speculative decoding
    ("*", "--propose-in-step", "Speculative decoding"),
    ("*", "--selector-rank", "Speculative decoding"),
    ("*", "--fn-mtp", "Speculative decoding"),
    ("*", "--fn-draft", "Speculative decoding"),
    ("*", "--fn-verify", "Speculative decoding"),
    ("*", "--engine-draft", "Speculative decoding"),
    ("*", "--spec-", "Speculative decoding"),
    ("*", "--draft", "Speculative decoding"),
    ("*", "--drafter", "Speculative decoding"),
    // memory plane
    ("*", "--fn-kv", "Context and KV cache"),
    ("*", "--fn-max-ctx", "Context and KV cache"),
    ("*", "--fn-vram", "Memory plane"),
    ("*", "--fn-ple", "Memory plane"),
    ("*", "--fn-ledger", "Memory plane"),
    ("*", "--fn-arena", "Memory plane"),
    ("*", "--xtier", "Expert tier"),
    ("*", "--fn-profile", "Expert tier"),
    ("*", "--fn-prefix", "Prefix cache"),
    ("*", "--prefix", "Prefix cache"),
    ("*", "--enable-prefix", "Prefix cache"),
    ("*", "--park", "Session parking"),
    ("*", "--engine-park", "Session parking"),
    ("*", "--fn-image", "Vision"),
    ("*", "--fn-vision", "Vision"),
    ("*", "--vision", "Vision"),
    ("*", "--engine-vision", "Vision"),
    // serving plane
    ("*", "--fn-serve", "Server"),
    ("*", "--fn-served", "Server"),
    ("*", "--served", "Server"),
    ("*", "--host", "Server"),
    ("*", "--port", "Server"),
    ("*", "--api-", "Server"),
    ("*", "--request-ring", "Server"),
    ("*", "--max-logprobs", "Server"),
    ("*", "--warm-prompts", "Prefix cache"),
    ("*", "--fn-thinking", "Chat and parsing"),
    ("*", "--chat-template", "Chat and parsing"),
    ("*", "--tokenizer", "Chat and parsing"),
    ("*", "--tool-call", "Chat and parsing"),
    ("*", "--reasoning", "Chat and parsing"),
    ("*", "--enable-auto-tool", "Chat and parsing"),
    ("*", "--no-enable-auto-tool", "Chat and parsing"),
    // model and parallelism
    ("*", "--pack", "Model"),
    ("*", "--fn-trunk", "Model"),
    ("*", "--prebuilt", "Model"),
    ("*", "--hip-lib", "Model"),
    ("*", "--manifest", "Model"),
    ("*", "--autotune", "Model"),
    ("*", "--engine-tp", "Parallelism"),
    ("*", "--tp", "Parallelism"),
    ("*", "--fn-experts", "Parallelism"),
    ("*", "--fn-one-process", "Parallelism"),
    ("*", "--engine-device", "Parallelism"),
    ("*", "--engine-peer", "Parallelism"),
    ("*", "--peer", "Parallelism"),
    ("*", "--fn-rank", "Parallelism"),
    ("*", "--fn-peer", "Parallelism"),
    ("*", "--rank", "Parallelism"),
    ("*", "--device", "Parallelism"),
    ("*", "--limit-mm-images", "Vision"),
    ("*", "--parser-defaults", "Chat and parsing"),
    // the 27B engine's memory plane (`saber engine` spelling)
    ("*", "--max-ctx", "Context and KV cache"),
    ("*", "--kv-headroom-mib", "Memory plane"),
    ("*", "--kv-pool-blocks", "Memory plane"),
    ("*", "--ledger", "Memory plane"),
    ("*", "--budget-mib", "Memory plane"),
    ("*", "--ple-", "Memory plane"),
    ("*", "--deadline-s", "Logging and process"),
    // scheduling and batching
    ("*", "--max-seqs", "Batching"),
    ("*", "--prefill-rows", "Batching"),
    ("*", "--decode-batches", "Batching"),
    ("*", "--decode-widths", "Batching"),
    ("*", "--width-rule", "Batching"),
    ("*", "--ragged-", "Batching"),
    ("*", "--row-bands", "Batching"),
    ("*", "--rectangular-widths", "Batching"),
    ("*", "--mixed-", "Batching"),
    ("*", "--max-batched-tokens", "Batching"),
    ("*", "--prefill-fill-above", "Batching"),
    ("*", "--fixed-step-wait", "Batching"),
    ("*", "--gather-per-row", "Batching"),
    ("*", "--engine-max-seqs", "Batching"),
    ("*", "--engine-decode", "Batching"),
    ("*", "--engine-prefill", "Batching"),
    ("*", "--engine-rectangular", "Batching"),
    ("*", "--fn-max-chunk", "Batching"),
    ("*", "--fn-wide", "Batching"),
    ("*", "--fn-prefill", "Batching"),
    ("*", "--fn-busy", "Batching"),
    // the 27B engine's remaining knobs and Flash-Next's kernel arms
    ("*", "--prefill-ar-overlap", "All-reduce"),
    ("*", "--engine-", "Engine kernels and arms"),
    ("*", "--gdn-", "Engine kernels and arms"),
    ("*", "--macaq", "Engine kernels and arms"),
    ("*", "--bf16-head", "Engine kernels and arms"),
    ("*", "--mxfp4-head", "Engine kernels and arms"),
    ("*", "--fp8-head", "Engine kernels and arms"),
    ("*", "--capture", "Engine kernels and arms"),
    ("*", "--live-rows", "Engine kernels and arms"),
    ("*", "--small-chunk", "Engine kernels and arms"),
    ("*", "--shared-prefill-arena", "Engine kernels and arms"),
    ("*", "--shared-head-arena", "Engine kernels and arms"),
    ("*", "--mid-chunk", "Engine kernels and arms"),
    ("*", "--skinny-gate", "Engine kernels and arms"),
    ("*", "--decode-pf", "Engine kernels and arms"),
    ("*", "--head-m64", "Engine kernels and arms"),
    ("*", "--b1-arms", "Engine kernels and arms"),
    ("*", "--c4-arms", "Engine kernels and arms"),
    ("*", "--step-arms", "Engine kernels and arms"),
    ("*", "--decode-fusion", "Engine kernels and arms"),
    ("*", "--fuse", "Engine kernels and arms"),
    ("*", "--ar-", "All-reduce"),
    ("*", "--fn-ar", "All-reduce"),
    ("*", "--fn-", "Engine kernels and arms"),
];

/// Value ranges the console enforces before a boot (flag, min, max): the engine refuses these
/// itself, later and in the boot log; the console says it at the field. Ports are 1..=65535.
pub const RANGES: &[(&str, f64, f64)] = &[
    ("--port", 1.0, 65_535.0),
    ("--fn-serve", 1.0, 65_535.0),
    ("--tp", 1.0, 2.0),
    ("--engine-tp", 1.0, 2.0),
    ("--fn-max-ctx", 256.0, 1_048_576.0),
    ("--engine-max-ctx", 256.0, 1_048_576.0),
    ("--max-ctx", 256.0, 1_048_576.0),
    ("--engine-max-seqs", 1.0, 64.0),
    ("--max-seqs", 1.0, 64.0),
    ("--fn-budget-mib", 1_024.0, 262_144.0),
    ("--engine-budget-mib", 1_024.0, 262_144.0),
    ("--budget-mib", 1_024.0, 262_144.0),
    ("--fn-kv-hot-gib", 0.0, 256.0),
    ("--fn-kv-ram-gib", 0.0, 1_024.0),
    ("--fn-kv-ssd-gib", 0.0, 16_384.0),
    ("--fn-kv-pcache-gib", 0.0, 16_384.0),
    ("--fn-prefix-tree-ram-gib", 0.0, 1_024.0),
    ("--fn-ple-drive-keepalive-ms", 0.0, 60_000.0),
    ("--engine-draft-block", 0.0, 64.0),
    ("--draft-block", 0.0, 64.0),
    ("--limit-mm-images", 0.0, 10_000.0),
    ("--step-delay-ms", 0.0, 60_000.0),
    ("--kv-gib", 1.0, 1_024.0),
    ("--state-gib", 1.0, 1_024.0),
];

/// The enforced range of `name`.
#[must_use]
pub fn range_of(name: &str) -> Option<(f64, f64)> {
    RANGES
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, a, b)| (*a, *b))
}

/// Group display order.
pub const GROUPS: &[&str] = &[
    "Model",
    "Server",
    "Parallelism",
    "Context and KV cache",
    "Batching",
    "Prefix cache",
    "Chat and parsing",
    "Vision",
    "Speculative decoding",
    "Memory plane",
    "Expert tier",
    "Session parking",
    "All-reduce",
    "Engine kernels and arms",
    "Mock engine",
    "Logging and process",
    "Harness and gates",
    "Diagnostics",
];

/// Groups whose flags are all diagnostic-tier.
pub const DIAGNOSTIC_GROUPS: &[&str] = &["Diagnostics", "Harness and gates"];

fn group_for(section: &str, name: &str) -> Option<&'static str> {
    if let Some((_, _, g)) = BASIC.iter().find(|(s, n, _)| *s == section && *n == name) {
        return Some(g);
    }
    let rule = |name: &str| {
        GROUP_RULES
            .iter()
            .find(|(s, prefix, _)| {
                (*s == "*" && section != "global" || *s == section) && name.starts_with(prefix)
            })
            .map(|(_, _, g)| *g)
    };
    // `--no-X` (the negative switch of an arm) lives with `--X`
    rule(name).or_else(|| {
        name.strip_prefix("--no-")
            .and_then(|r| rule(&format!("--{r}")))
    })
}

fn is_diagnostic(arg: &Arg, help: &str, group: &str) -> bool {
    let h = help.to_ascii_lowercase();
    arg.is_hide_set()
        || DIAGNOSTIC_GROUPS.contains(&group)
        || h.contains("diagnostic")
        || h.starts_with("the control")
        || h.contains(", the control for")
        || h.contains("gate arm")
        || h.contains("harness")
}

fn kind_of(arg: &Arg) -> &'static str {
    if matches!(
        arg.get_action(),
        ArgAction::SetTrue | ArgAction::SetFalse | ArgAction::Count
    ) {
        return "bool";
    }
    if !arg.get_possible_values().is_empty() {
        return "enum";
    }
    let vp: &ValueParser = arg.get_value_parser();
    let id = vp.type_id();
    let is = |t: &[TypeId]| t.iter().any(|t| id == *t);
    if is(&[TypeId::of::<std::path::PathBuf>()]) {
        "path"
    } else if is(&[TypeId::of::<f64>(), TypeId::of::<f32>()]) {
        "float"
    } else if is(&[
        TypeId::of::<i64>(),
        TypeId::of::<i32>(),
        TypeId::of::<i16>(),
        TypeId::of::<i8>(),
    ]) {
        "int"
    } else if is(&[
        TypeId::of::<u64>(),
        TypeId::of::<u32>(),
        TypeId::of::<u16>(),
        TypeId::of::<u8>(),
        TypeId::of::<usize>(),
    ]) {
        "uint"
    } else {
        "string"
    }
}

fn spec(section: &str, arg: &Arg) -> Option<FlagSpec> {
    let long = arg.get_long()?;
    if long == "help" || long == "version" {
        return None;
    }
    let name = format!("--{long}");
    let help = arg.get_help().map(|h| h.to_string()).unwrap_or_default();
    let long_help = arg.get_long_help().map(|h| h.to_string());
    let group = group_for(section, &name).unwrap_or("").to_string();
    let basic = BASIC.iter().any(|(s, n, _)| *s == section && *n == name);
    let full_help = long_help.clone().unwrap_or_else(|| help.clone());
    let tier = if basic {
        Tier::Basic
    } else if is_diagnostic(arg, &full_help, &group) {
        Tier::Diagnostic
    } else {
        Tier::Advanced
    };
    let defaults: Vec<String> = arg
        .get_default_values()
        .iter()
        .map(|v| v.to_string_lossy().into_owned())
        .collect();
    let multiple = matches!(arg.get_action(), ArgAction::Append)
        || arg.get_num_args().is_some_and(|r| r.max_values() > 1);
    let range = range_of(&name);
    Some(FlagSpec {
        delimiter: arg.get_value_delimiter().map(|c| c.to_string()),
        min: range.map(|r| r.0),
        max: range.map(|r| r.1),
        name,
        section: section.to_string(),
        kind: kind_of(arg),
        default: (!defaults.is_empty()).then(|| defaults.join(",")),
        choices: arg
            .get_possible_values()
            .iter()
            .filter(|p| !p.is_hide_set())
            .map(|p| p.get_name().to_string())
            .collect(),
        multiple,
        help,
        long_help,
        group,
        tier,
        requires_restart: true,
        hidden: arg.is_hide_set(),
    })
}

/// The schema of `root`'s global flags plus the `sections` subcommands' flags.
#[must_use]
pub fn build(root: &Command, sections: &[&str]) -> Schema {
    let mut root = root.clone();
    root.build();
    let mut flags = Vec::new();
    for a in root.get_arguments() {
        if a.is_global_set() {
            flags.extend(spec("global", a));
        }
    }
    let mut stale = Vec::new();
    for sec in sections {
        let Some(sub) = root.find_subcommand(sec) else {
            stale.push(format!("no subcommand `{sec}`"));
            continue;
        };
        for a in sub.get_arguments() {
            if a.is_global_set() {
                continue;
            }
            flags.extend(spec(sec, a));
        }
    }
    let unclassified = flags
        .iter()
        .filter(|f| f.group.is_empty())
        .map(|f| format!("{} {}", f.section, f.name))
        .collect();
    for (s, n, _) in BASIC {
        if !flags.iter().any(|f| f.section == *s && f.name == *n) {
            stale.push(format!("{s} {n}"));
        }
    }
    Schema {
        version: SCHEMA_VERSION,
        sections: std::iter::once("global".to_string())
            .chain(sections.iter().map(|s| s.to_string()))
            .collect(),
        groups: GROUPS.to_vec(),
        flags,
        unclassified,
        stale_annotations: stale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Parser, Subcommand};

    #[derive(Parser)]
    struct T {
        #[arg(long, global = true)]
        verbose: bool,
        #[command(subcommand)]
        cmd: C,
    }
    #[derive(Subcommand)]
    enum C {
        Serve {
            /// The port.
            #[arg(long, default_value_t = 8080)]
            port: u16,
            /// Ratio.
            #[arg(long, default_value_t = 0.5)]
            fn_kv_ram_gib: f64,
            /// Which.
            #[arg(long, value_parser = ["on", "off"], default_value = "on")]
            fn_prefix_tree: String,
            /// DIAGNOSTIC: dump.
            #[arg(long)]
            fn_dump: Option<std::path::PathBuf>,
            /// A flag no rule knows.
            #[arg(long)]
            zzz_unknown: bool,
        },
    }

    #[test]
    fn kinds_groups_tiers_and_the_unclassified_net() {
        use clap::CommandFactory;
        let s = build(&T::command(), &["serve"]);
        let g = |n: &str| s.flags.iter().find(|f| f.name == n).unwrap();
        assert_eq!(g("--port").kind, "uint");
        assert_eq!(g("--port").default.as_deref(), Some("8080"));
        assert_eq!(g("--port").group, "Server");
        assert_eq!(g("--fn-kv-ram-gib").kind, "float");
        assert_eq!(g("--fn-prefix-tree").kind, "enum");
        assert_eq!(g("--fn-prefix-tree").choices, ["on", "off"]);
        assert_eq!(g("--fn-dump").kind, "path");
        assert_eq!(g("--fn-dump").tier, Tier::Diagnostic);
        assert_eq!(g("--verbose").section, "global");
        assert_eq!(s.unclassified, ["serve --zzz-unknown"]);
        // the real BASIC table names flags this toy CLI lacks
        assert!(!s.stale_annotations.is_empty());
        let _ = T::parse_from(["t", "serve"]).cmd;
    }
}
