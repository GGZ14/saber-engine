//! P27-7: the 27B's two byte-ledger reserves, as ONE named law.
//!
//! The engine's fence is `--engine-budget-mib` (a cap on every device allocation the engine bills)
//! and the KV pool is derived as `budget - everything billed before it - GDN slots - headroom`, where
//! `--engine-kv-headroom-mib` is what the bills taken AFTER the pool is cut (the index blob, the
//! drafter's aux tap, the two-rank IPC workspace) come out of. Both numbers were set long before
//! ROCm 10.1: 30,500 of a 32,624 MiB card (2,124 MiB never billed) and 1,024 MiB of headroom whose
//! post-pool bills are 825 MiB at the served 262k boot. P27-7 measured the card (sysfs
//! `mem_info_vram_used`, both cards, 0.5 s, boot + 250k probe + BetterBench conc-8) and the law below
//! is that measurement: the budget leaves [`LEAN_FREE_MIB`] free at the measured peak, and the
//! headroom is the post-pool bills plus 16 MiB.
//!
//! `today` (the control) keeps the two flags exactly as given. `lean` REPLACES them with the
//! measured pair, said at boot (`engledger` line); every arm is a byte COUNT (the pool's size), never
//! a byte VALUE: no plan, kernel, table layout or page address changes (the allocators hand out the
//! lowest free block, so a larger pool never moves a page a smaller one used).

use anyhow::{bail, Result};

/// The binary's default ledger law: `lean` since P27-7 (2026-10-07, every gate green: ids + the
/// standing paged shadow signature at 4,096, served bytes at 1.5k..250k, BetterBench 2 + 2 tied);
/// `today` is the control.
pub const DEFAULT_LEDGER: &str = "lean";

/// `lean`'s fence, MiB: the card (32,624 MiB) minus the runtime the fence does not see at the
/// measured peak minus [`LEAN_FREE_MIB`], plus the headroom's own slack above the post-pool bills.
/// P27-7 hold 1 (2026-10-07, ROCm 10.1, 262k served boot): fenced 30,301 MiB, peak 30,838 ->
/// 537 MiB outside the fence; budget 31,854 / headroom 848 peaked at 32,386 of 32,624 on card 1
/// (238 MiB free), so the budget is 30 MiB lower than that arm's. **P27-9d (2026-10-08): 31,776**
/// (-48): the drafter's rope table is now ALLOCATED at `max_ctx` (64 MiB a rank at 262k; the fence
/// billed it before, the card never held it), and the 31,824 boot peaked at 32,415 MiB on card 1
/// with the share off (209 free) / 32,390 with it on (234 free). **31,744 (-32, orchestrator,
/// 2026-10-08 05:30 CDT)**: P27-9d's share-on boot at 31,776 left 284 / 250 MiB free -- card 1 six
/// MiB under [`LEAN_FREE_MIB`] -- and the 48 MiB cut had moved its peak only 16 MiB (the peak's
/// run-to-run spread), so the fence takes another 32 to keep the law on both cards.
pub const LEAN_BUDGET_MIB: u64 = 31_744;

/// `lean`'s headroom, MiB: the post-pool bills at the served 262k boot (825 MiB) + 16.
pub const LEAN_HEADROOM_MIB: u64 = 848;

/// What `lean` leaves free on the card at the measured peak, MiB.
pub const LEAN_FREE_MIB: u64 = 256;

/// FN71b: the ledger class name of the kernel's P2PDMA pool (`amdgpu.p2pmem_mb`, LEDGER FN71b).
pub const P2PMEM_CLASS: &str = "kernel.p2pmem";

/// Where the PCI devices live in sysfs (the host root; a container sees the same read-only tree).
pub const SYSFS_PCI_DEVICES: &str = "/sys/bus/pci/devices";

/// FN71b: the kernel's P2PDMA pool a card, MiB (rounded up), read from `<root>/<bdf>/p2pmem/size`.
/// The pool is pinned VRAM at boot that no allocator sees, so every device budget that counts the
/// card's bytes must give it up. `bdf = None` = the largest pool over every device under `root`
/// (the ledgers resolve before HIP opens, and `amdgpu.p2pmem_mb` sizes every dGPU's pool alike);
/// 0 when absent, unreadable or 0 (a boot without the pool).
#[must_use]
pub fn p2pmem_pool_mib(root: &std::path::Path, bdf: Option<&str>) -> u64 {
    let one = |dir: std::path::PathBuf| -> u64 {
        std::fs::read_to_string(dir.join("p2pmem").join("size"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map_or(0, |b| b.div_ceil(1 << 20))
    };
    match bdf {
        Some(b) => one(root.join(b)),
        None => std::fs::read_dir(root)
            .map(|rd| rd.flatten().map(|e| one(e.path())).max().unwrap_or(0))
            .unwrap_or(0),
    }
}

/// FN71b: this box's pool a card ([`p2pmem_pool_mib`] over [`SYSFS_PCI_DEVICES`], the largest).
#[must_use]
pub fn box_p2pmem_mib() -> u64 {
    p2pmem_pool_mib(std::path::Path::new(SYSFS_PCI_DEVICES), None)
}

/// FN71b: `budget` less the pool, said at boot as the named class (`what` = the line's tag).
#[must_use]
pub fn discount_p2pmem(what: &str, budget_mib: u64, pool_mib: u64) -> u64 {
    let b = budget_mib.saturating_sub(pool_mib);
    if pool_mib > 0 {
        println!(
            "{what}: {P2PMEM_CLASS} {pool_mib} MiB (the kernel's P2PDMA pool, pinned at boot, \
             invisible to the allocators) -> budget {budget_mib} - {pool_mib} = {b} MiB (FN71b)"
        );
    }
    b
}

/// The two ledger laws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ledger {
    /// The flags as given (`--engine-budget-mib`, `--engine-kv-headroom-mib`): the control.
    Today,
    /// The measured pair ([`LEAN_BUDGET_MIB`], [`LEAN_HEADROOM_MIB`]).
    Lean,
}

impl Ledger {
    /// Parse `today` / `lean`; anything else is refused by name.
    ///
    /// # Errors
    /// An unknown law.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "today" => Ok(Self::Today),
            "lean" => Ok(Self::Lean),
            other => {
                bail!("--engine-ledger {other}: the laws are `today` (the control) and `lean`")
            }
        }
    }

    /// The (budget, headroom) MiB the engine runs with, from the flags as given.
    #[must_use]
    pub fn resolve(self, budget_mib: u64, headroom_mib: u64) -> (u64, u64) {
        match self {
            Self::Today => (budget_mib, headroom_mib),
            Self::Lean => (LEAN_BUDGET_MIB, LEAN_HEADROOM_MIB),
        }
    }

    /// Resolve and say so (one `engledger` line at boot).
    #[must_use]
    pub fn resolve_said(self, budget_mib: u64, headroom_mib: u64, pool_mib: u64) -> (u64, u64) {
        let (b0, h) = self.resolve(budget_mib, headroom_mib);
        // FN71b: the kernel's P2PDMA pool comes off EITHER law's fence (it is a fact of the boot,
        // not a choice of law: the fences were measured on a card without it)
        let b = discount_p2pmem("engledger", b0, pool_mib);
        match self {
            Self::Today => println!(
                "engledger today: budget {b} MiB, KV headroom {h} MiB as given (less {P2PMEM_CLASS} \
                 {pool_mib} MiB) (P27-7: `--engine-ledger lean` = budget {LEAN_BUDGET_MIB} / \
                 headroom {LEAN_HEADROOM_MIB})"
            ),
            Self::Lean => println!(
                "engledger lean (P27-7): budget {b} MiB, KV headroom {h} MiB (the flags' {budget_mib} \
                 / {headroom_mib} are replaced; {LEAN_BUDGET_MIB} less {P2PMEM_CLASS} {pool_mib} \
                 MiB; measured on ROCm 10.1, {LEAN_FREE_MIB} MiB left free at the served peak); \
                 `--engine-ledger today` is the control"
            ),
        }
        (b, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p27_7_lean_is_the_default_today_keeps_the_flags_lean_replaces_both() {
        assert_eq!(Ledger::parse(DEFAULT_LEDGER).unwrap(), Ledger::Lean);
        assert_eq!(Ledger::Today.resolve(30_500, 1_024), (30_500, 1_024));
        assert_eq!(Ledger::Today.resolve(26_000, 7), (26_000, 7));
        assert_eq!(
            Ledger::Lean.resolve(30_500, 1_024),
            (LEAN_BUDGET_MIB, LEAN_HEADROOM_MIB)
        );
        assert!(Ledger::parse("tight").is_err());
    }

    #[test]
    fn fn71b_the_ledgers_discount_the_kernels_p2pdma_pool_read_from_sysfs() {
        let dir = saber_test_scratch::ScratchDir::new("saber-fn71b-p2pmem");
        let root = dir.path();
        // no devices at all, then devices without a pool: 0
        assert_eq!(p2pmem_pool_mib(&root.join("absent"), None), 0);
        for bdf in ["0000:03:00.0", "0000:07:00.0", "0000:7f:00.0"] {
            std::fs::create_dir_all(root.join(bdf)).unwrap();
        }
        assert_eq!(p2pmem_pool_mib(root, None), 0);
        assert_eq!(p2pmem_pool_mib(root, Some("0000:07:00.0")), 0);
        // the FN71b boot: 2 GiB on both dGPUs, the iGPU without a pool, one card's size = 0 elsewhere
        for bdf in ["0000:03:00.0", "0000:07:00.0"] {
            let p = root.join(bdf).join("p2pmem");
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("size"), "2147483648\n").unwrap();
        }
        assert_eq!(p2pmem_pool_mib(root, Some("0000:07:00.0")), 2_048);
        assert_eq!(p2pmem_pool_mib(root, Some("0000:7f:00.0")), 0);
        assert_eq!(p2pmem_pool_mib(root, None), 2_048);
        // a partial MiB rounds UP (the fence never under-bills); garbage = no pool
        std::fs::write(root.join("0000:03:00.0/p2pmem/size"), "3145729").unwrap();
        assert_eq!(p2pmem_pool_mib(root, Some("0000:03:00.0")), 4);
        std::fs::write(root.join("0000:03:00.0/p2pmem/size"), "x").unwrap();
        assert_eq!(p2pmem_pool_mib(root, Some("0000:03:00.0")), 0);
        // both laws take it off their fence; no pool = the laws unchanged
        assert_eq!(
            Ledger::Lean.resolve_said(30_500, 1_024, 2_048),
            (LEAN_BUDGET_MIB - 2_048, LEAN_HEADROOM_MIB)
        );
        assert_eq!(
            Ledger::Today.resolve_said(30_500, 1_024, 2_048),
            (28_452, 1_024)
        );
        assert_eq!(
            Ledger::Lean.resolve_said(30_500, 1_024, 0),
            (LEAN_BUDGET_MIB, LEAN_HEADROOM_MIB)
        );
        assert_eq!(discount_p2pmem("t", 30_500, 2_048), 28_452);
        assert_eq!(discount_p2pmem("t", 100, 2_048), 0);
        assert_eq!(P2PMEM_CLASS, "kernel.p2pmem");
    }

    #[test]
    fn p27_7_lean_never_leaves_less_than_the_post_pool_bills_or_overruns_the_card() {
        // The post-pool bills at the served 262k boot (indices 53 + aux tap 100 + IPC 672 MiB).
        const POST_POOL_MIB: u64 = 825;
        const CARD_MIB: u64 = 32_624;
        const {
            assert!(LEAN_HEADROOM_MIB >= POST_POOL_MIB);
            assert!(
                LEAN_BUDGET_MIB + LEAN_FREE_MIB <= CARD_MIB + (LEAN_HEADROOM_MIB - POST_POOL_MIB)
            );
            // The control's own pair is not lean's: an arm that changed nothing would gate nothing.
            assert!(LEAN_BUDGET_MIB > 30_500);
            assert!(LEAN_HEADROOM_MIB < 1_024);
        }
    }
}
