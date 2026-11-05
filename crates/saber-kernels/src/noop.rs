//! The null plugin: `resolve` for two kernels that do (almost) nothing.
//!
//! Every other rules module in this crate ports a decision some prod launcher makes in C++. This
//! one ports nothing, because prod never launches a kernel that does nothing. It exists because
//! `saber chain-bench` has to measure what a *node* costs, and that measurement needs a dispatch
//! whose kernel time and memory traffic are as close to zero as the hardware allows:
//!
//! * [`NoopOp::Noop`] — `saber_noop()`, no arguments, no memory traffic. A chain of them has no
//!   drain to pay, so its wall per node is the command processor's own per-dispatch work (plus,
//!   under a graph, the node's share of replay).
//! * [`NoopOp::Touch`] — `saber_noop_touch(p)`, one byte read-modify-written. A chain of them on
//!   the same byte is a real RAW dependency of the smallest possible footprint, so the difference
//!   between the two chains is the drain at one byte.
//!
//! The geometry is in `rules/noop_gfx1201.toml` like every other launch geometry in the engine,
//! and the argument layout comes from [`crate::abi::signature_for`] like every other kernarg
//! image. Nothing here is a literal at a call site.

use serde::Deserialize;

use crate::abi;
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

const PROD_RULES: &str = include_str!("../rules/noop_gfx1201.toml");

/// The same text, for the plan input hash: the policy this module enforces.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

/// The one code object both kernels live in.
const PLUGIN: &str = "noop";

/// One row: an op name, the symbol it launches, and the geometry it launches at.
#[derive(Debug, Clone, Deserialize)]
pub struct NoopRow {
    pub op: String,
    pub symbol: String,
    pub grid: u32,
    pub block: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoopRules {
    pub schema_version: u32,
    pub name: String,
    #[serde(default)]
    pub kernel: Vec<NoopRow>,
}

impl NoopRules {
    /// The committed rules for this instrument.
    pub fn prod_noop() -> Result<Self> {
        Self::from_toml(PROD_RULES)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: NoopRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "noop rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        if self.kernel.is_empty() {
            return Err(Error::Rule("noop rules name no kernels at all".into()));
        }
        for (i, row) in self.kernel.iter().enumerate() {
            if self.kernel[..i].iter().any(|r| r.op == row.op) {
                return Err(Error::Rule(format!(
                    "noop rules state op `{}` twice",
                    row.op
                )));
            }
            // One workgroup of one thread, or it is not measuring a dispatch any more. Stated
            // here rather than assumed at the call site: a row that widened would silently turn
            // the instrument into an occupancy benchmark.
            if row.grid != 1 || row.block != 1 {
                return Err(Error::Rule(format!(
                    "{}: the null plugin launches grid 1 x block 1, not {} x {}",
                    row.op, row.grid, row.block
                )));
            }
            let sig = abi::signature_for(&row.symbol)
                .ok_or_else(|| Error::Rule(format!("{}: no declared signature", row.symbol)))?;
            // Pointers only. A scalar argument would mean the rules file has to carry a VALUE,
            // and a value is a decision this instrument is not entitled to make.
            if let Some(bad) = sig.iter().find(|a| a.kind != abi::ArgKind::Buffer) {
                return Err(Error::Rule(format!(
                    "{}: argument `{}` is not a buffer; the null plugin takes pointers only",
                    row.symbol, bad.name
                )));
            }
        }
        Ok(())
    }

    /// Every symbol these rules name.
    pub fn symbols(&self) -> Vec<&str> {
        self.kernel.iter().map(|r| r.symbol.as_str()).collect()
    }

    /// The plugin these symbols live in. Asked of the rules, like [`crate::paro::ParoRules::owner`],
    /// so the planner never has to guess it from a file stem.
    pub fn owner(&self) -> &'static str {
        PLUGIN
    }

    pub fn row(&self, op: NoopOp) -> Result<&NoopRow> {
        self.kernel
            .iter()
            .find(|r| r.op == op.as_str())
            .ok_or_else(|| Error::Rule(format!("the noop rules describe no op `{}`", op.as_str())))
    }
}

/// The two null ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoopOp {
    /// No arguments, no memory traffic.
    Noop,
    /// One pointer; one byte read-modify-written.
    Touch,
}

impl NoopOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            NoopOp::Noop => "noop",
            NoopOp::Touch => "noop_touch",
        }
    }
}

/// The descriptor for a null op: geometry from the rules, kernarg layout from the ABI.
///
/// Every buffer argument binds to the argument's own name, so a caller hands
/// `ArgBindings::with("p", addr)` and nothing in between invents a name.
pub fn resolve(op: NoopOp, rules: &NoopRules) -> Result<LaunchDescriptor> {
    let row = rules.row(op)?;
    let sig = abi::signature_for(&row.symbol)
        .ok_or_else(|| Error::Rule(format!("{}: no declared signature", row.symbol)))?;
    let (packed, bytes) = abi::pack(sig);
    let args = sig
        .iter()
        .zip(packed)
        .map(|(a, (offset, size))| Slot {
            name: a.name.to_owned(),
            offset,
            size,
            value: ArgValue::Buffer {
                buffer: a.name.to_owned(),
            },
        })
        .collect();
    Ok(LaunchDescriptor {
        op: row.op.clone(),
        plugin: PLUGIN.into(),
        kernel: row.symbol.clone(),
        grid: [row.grid, 1, 1],
        block: [row.block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_rules_resolve_both_ops() {
        let r = NoopRules::prod_noop().expect("the noop rules parse");
        assert_eq!(r.owner(), "noop");
        assert_eq!(r.symbols(), vec!["saber_noop", "saber_noop_touch"]);

        let d = resolve(NoopOp::Noop, &r).expect("noop resolves");
        assert_eq!(d.kernel, "saber_noop");
        assert_eq!(d.grid, [1, 1, 1]);
        assert_eq!(d.block, [1, 1, 1]);
        assert_eq!(d.n_launches, 1);
        // The whole point of this row: the host writes NO kernarg bytes at all.
        assert_eq!(d.kernarg_bytes, 0);
        assert!(d.args.is_empty());

        let t = resolve(NoopOp::Touch, &r).expect("noop_touch resolves");
        assert_eq!(t.kernel, "saber_noop_touch");
        assert_eq!(t.kernarg_bytes, 8);
        assert_eq!(t.args.len(), 1);
        assert_eq!(t.args[0].name, "p");
        assert_eq!(t.args[0].offset, 0);
    }

    /// NEGATIVE CONTROL: a row that widened past one workgroup is refused at load. Without this
    /// the instrument could silently become an occupancy benchmark and still parse.
    #[test]
    fn a_row_that_is_not_one_workgroup_of_one_thread_is_refused() {
        let bad = PROD_RULES.replace("block = 1", "block = 256");
        match NoopRules::from_toml(&bad) {
            Err(Error::Rule(m)) => assert!(m.contains("grid 1 x block 1"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// NEGATIVE CONTROL: a symbol no signature describes cannot be resolved, so a typo in the
    /// rules file is a load failure rather than a "named symbol not found" at launch.
    #[test]
    fn a_symbol_with_no_declared_signature_is_refused() {
        let bad = PROD_RULES.replace("saber_noop_touch", "saber_noop_touchh");
        match NoopRules::from_toml(&bad) {
            Err(Error::Rule(m)) => assert!(m.contains("no declared signature"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}
