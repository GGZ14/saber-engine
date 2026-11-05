//! The byte ledger: declared capacities against named reservations.
//!
//! From `docs/ARCHITECTURE.md` § Memory plane: *"Over-commit is a startup error, not an OOM at token
//! 4000."* Every subsystem that intends to hold bytes — weights per rank, pinned staging, the KV
//! arena, the (separate) GDN-state arena, parked sessions on NVMe — declares them here **before**
//! allocating, by name. A reservation that does not fit is an error at reservation time, with both
//! numbers in the message, so the failure names the thing that did not fit instead of surfacing as a
//! HIP OOM three layers down.
//!
//! The ledger counts *intent*, not allocations: it never allocates and never talks to HIP. That is
//! what lets it run at startup, before a single arena exists, and what lets `saber explain` print
//! the whole budget without a GPU.
//!
//! NVMe is a special case: session parking is *write* traffic, so the drive gets both a space
//! capacity (parked bytes resident) and a monotonic write counter, which is the number
//! `/metrics` needs against a 2400 TBW-class drive.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

/// A pool of bytes with one capacity and one set of reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pool {
    /// Device memory on one TP rank.
    Vram(u32),
    /// Pinned (page-locked) host memory. Scarce; every pinned byte is a byte the kernel cannot
    /// reclaim.
    HostPinned,
    /// Ordinary host memory the engine intends to hold.
    HostPageable,
    /// Bytes resident on NVMe (parked sessions). Write traffic is counted separately.
    Nvme,
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pool::Vram(r) => write!(f, "vram.rank{r}"),
            Pool::HostPinned => write!(f, "host.pinned"),
            Pool::HostPageable => write!(f, "host.pageable"),
            Pool::Nvme => write!(f, "nvme"),
        }
    }
}

/// Why a ledger operation failed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LedgerError {
    /// A pool was used before its capacity was declared.
    #[error("pool {pool} has no declared capacity; declare it before reserving against it")]
    Undeclared {
        /// The pool.
        pool: Pool,
    },

    /// A pool's capacity was declared twice.
    #[error("pool {pool} capacity already declared as {existing} bytes")]
    AlreadyDeclared {
        /// The pool.
        pool: Pool,
        /// The capacity already on record.
        existing: u64,
    },

    /// The same name was reserved twice in one pool.
    #[error("{pool}: reservation {name:?} already exists ({existing} bytes)")]
    DuplicateReservation {
        /// The pool.
        pool: Pool,
        /// The name.
        name: String,
        /// Bytes already reserved under that name.
        existing: u64,
    },

    /// No such reservation to release.
    #[error("{pool}: no reservation named {name:?}")]
    NoSuchReservation {
        /// The pool.
        pool: Pool,
        /// The name.
        name: String,
    },

    /// The reservation does not fit. This is the startup error the whole ledger exists for.
    #[error(
        "{pool} over-commit: {name:?} wants {want} bytes, {free} free of {capacity} \
             ({used} already reserved by {holders} reservations)"
    )]
    OverCommit {
        /// The pool.
        pool: Pool,
        /// The reservation that did not fit.
        name: String,
        /// Bytes it asked for.
        want: u64,
        /// Bytes still free.
        free: u64,
        /// Declared capacity.
        capacity: u64,
        /// Bytes already reserved.
        used: u64,
        /// How many reservations are holding them.
        holders: usize,
    },
}

#[derive(Debug, Default, Clone)]
struct PoolState {
    capacity: u64,
    used: u64,
    written: u64,
    reservations: BTreeMap<String, u64>,
}

/// The byte ledger.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    pools: BTreeMap<Pool, PoolState>,
}

impl Ledger {
    /// An empty ledger with nothing declared.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a pool's capacity. Each pool may be declared once.
    pub fn declare(&mut self, pool: Pool, capacity_bytes: u64) -> Result<(), LedgerError> {
        if let Some(existing) = self.pools.get(&pool) {
            return Err(LedgerError::AlreadyDeclared {
                pool,
                existing: existing.capacity,
            });
        }
        self.pools.insert(
            pool,
            PoolState {
                capacity: capacity_bytes,
                ..PoolState::default()
            },
        );
        Ok(())
    }

    /// Reserve `bytes` under `name`. Fails if the pool is undeclared, the name is taken, or the
    /// reservation would over-commit the pool. A failed reservation changes nothing.
    pub fn reserve(&mut self, pool: Pool, name: &str, bytes: u64) -> Result<(), LedgerError> {
        let st = self
            .pools
            .get_mut(&pool)
            .ok_or(LedgerError::Undeclared { pool })?;
        if let Some(&existing) = st.reservations.get(name) {
            return Err(LedgerError::DuplicateReservation {
                pool,
                name: name.to_string(),
                existing,
            });
        }
        let free = st.capacity.saturating_sub(st.used);
        if bytes > free {
            return Err(LedgerError::OverCommit {
                pool,
                name: name.to_string(),
                want: bytes,
                free,
                capacity: st.capacity,
                used: st.used,
                holders: st.reservations.len(),
            });
        }
        st.used += bytes;
        st.reservations.insert(name.to_string(), bytes);
        Ok(())
    }

    /// Release a reservation, returning the bytes it held.
    pub fn release(&mut self, pool: Pool, name: &str) -> Result<u64, LedgerError> {
        let st = self
            .pools
            .get_mut(&pool)
            .ok_or(LedgerError::Undeclared { pool })?;
        let bytes = st
            .reservations
            .remove(name)
            .ok_or_else(|| LedgerError::NoSuchReservation {
                pool,
                name: name.to_string(),
            })?;
        st.used -= bytes;
        Ok(bytes)
    }

    /// Count bytes written to a pool (NVMe parking traffic). Monotonic; not a reservation.
    pub fn record_written(&mut self, pool: Pool, bytes: u64) -> Result<(), LedgerError> {
        let st = self
            .pools
            .get_mut(&pool)
            .ok_or(LedgerError::Undeclared { pool })?;
        st.written += bytes;
        Ok(())
    }

    /// Declared capacity of a pool.
    #[must_use]
    pub fn capacity(&self, pool: Pool) -> Option<u64> {
        self.pools.get(&pool).map(|s| s.capacity)
    }

    /// Bytes reserved in a pool.
    #[must_use]
    pub fn used(&self, pool: Pool) -> u64 {
        self.pools.get(&pool).map_or(0, |s| s.used)
    }

    /// Bytes still free in a pool.
    #[must_use]
    pub fn free(&self, pool: Pool) -> Option<u64> {
        self.pools
            .get(&pool)
            .map(|s| s.capacity.saturating_sub(s.used))
    }

    /// Bytes written to a pool since startup.
    #[must_use]
    pub fn written(&self, pool: Pool) -> u64 {
        self.pools.get(&pool).map_or(0, |s| s.written)
    }

    /// Point-in-time copy of the whole ledger, for `/metrics` and `saber explain`.
    #[must_use]
    pub fn snapshot(&self) -> LedgerSnapshot {
        LedgerSnapshot {
            pools: self
                .pools
                .iter()
                .map(|(&pool, st)| PoolSnapshot {
                    pool,
                    label: pool.to_string(),
                    capacity_bytes: st.capacity,
                    used_bytes: st.used,
                    written_bytes: st.written,
                    reservations: st
                        .reservations
                        .iter()
                        .map(|(name, &bytes)| Reservation {
                            name: name.clone(),
                            bytes,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

/// One named reservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    /// Reservation name, e.g. `weights`, `kv`, `gdn_state`, `pack_staging`.
    pub name: String,
    /// Bytes held.
    pub bytes: u64,
}

/// One pool's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolSnapshot {
    /// The pool.
    pub pool: Pool,
    /// Its metrics label, e.g. `vram.rank0`.
    pub label: String,
    /// Declared capacity.
    pub capacity_bytes: u64,
    /// Bytes reserved.
    pub used_bytes: u64,
    /// Bytes written since startup.
    pub written_bytes: u64,
    /// Every reservation, by name.
    pub reservations: Vec<Reservation>,
}

/// A point-in-time copy of the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerSnapshot {
    /// One entry per declared pool.
    pub pools: Vec<PoolSnapshot>,
}

impl LedgerSnapshot {
    /// Render as Prometheus text. The serving plane exposes this on `/metrics`.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP saber_ledger_capacity_bytes Declared capacity of a byte pool.\n");
        out.push_str("# TYPE saber_ledger_capacity_bytes gauge\n");
        for p in &self.pools {
            out.push_str(&format!(
                "saber_ledger_capacity_bytes{{pool=\"{}\"}} {}\n",
                p.label, p.capacity_bytes
            ));
        }
        out.push_str("# HELP saber_ledger_reserved_bytes Bytes reserved in a byte pool.\n");
        out.push_str("# TYPE saber_ledger_reserved_bytes gauge\n");
        for p in &self.pools {
            out.push_str(&format!(
                "saber_ledger_reserved_bytes{{pool=\"{}\"}} {}\n",
                p.label, p.used_bytes
            ));
            for r in &p.reservations {
                out.push_str(&format!(
                    "saber_ledger_reservation_bytes{{pool=\"{}\",name=\"{}\"}} {}\n",
                    p.label, r.name, r.bytes
                ));
            }
        }
        out.push_str("# HELP saber_ledger_written_bytes_total Bytes written to a pool.\n");
        out.push_str("# TYPE saber_ledger_written_bytes_total counter\n");
        for p in &self.pools {
            out.push_str(&format!(
                "saber_ledger_written_bytes_total{{pool=\"{}\"}} {}\n",
                p.label, p.written_bytes
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn boot() -> Ledger {
        let mut l = Ledger::new();
        l.declare(Pool::Vram(0), 32 * GIB).unwrap();
        l.declare(Pool::Vram(1), 32 * GIB).unwrap();
        l.declare(Pool::HostPinned, 4 * GIB).unwrap();
        l.declare(Pool::HostPageable, 48 * GIB).unwrap();
        l.declare(Pool::Nvme, 64 * GIB).unwrap();
        l
    }

    #[test]
    fn reservations_add_up() {
        let mut l = boot();
        l.reserve(Pool::Vram(0), "weights", 11 * GIB).unwrap();
        l.reserve(Pool::Vram(0), "kv", 18 * GIB).unwrap();
        l.reserve(Pool::Vram(0), "gdn_state", GIB).unwrap();
        assert_eq!(l.used(Pool::Vram(0)), 30 * GIB);
        assert_eq!(l.free(Pool::Vram(0)), Some(2 * GIB));
        assert_eq!(l.used(Pool::Vram(1)), 0);
    }

    #[test]
    fn over_commit_is_an_error_and_changes_nothing() {
        let mut l = boot();
        l.reserve(Pool::Vram(0), "weights", 30 * GIB).unwrap();
        let err = l
            .reserve(Pool::Vram(0), "kv", 4 * GIB)
            .expect_err("must not fit");
        match err {
            LedgerError::OverCommit {
                pool,
                want,
                free,
                capacity,
                used,
                holders,
                ..
            } => {
                assert_eq!(pool, Pool::Vram(0));
                assert_eq!(want, 4 * GIB);
                assert_eq!(free, 2 * GIB);
                assert_eq!(capacity, 32 * GIB);
                assert_eq!(used, 30 * GIB);
                assert_eq!(holders, 1);
            }
            other => panic!("wrong error: {other}"),
        }
        assert_eq!(
            l.used(Pool::Vram(0)),
            30 * GIB,
            "failed reserve must not mutate"
        );
        l.reserve(Pool::Vram(0), "kv", 2 * GIB)
            .expect("exactly the free bytes must fit");
    }

    #[test]
    fn undeclared_pool_is_an_error() {
        let mut l = Ledger::new();
        assert_eq!(
            l.reserve(Pool::Vram(3), "weights", 1),
            Err(LedgerError::Undeclared {
                pool: Pool::Vram(3)
            })
        );
        l.declare(Pool::Vram(3), 8).unwrap();
        assert!(matches!(
            l.declare(Pool::Vram(3), 16),
            Err(LedgerError::AlreadyDeclared { existing: 8, .. })
        ));
    }

    #[test]
    fn duplicate_and_missing_reservations() {
        let mut l = boot();
        l.reserve(Pool::HostPinned, "pack_staging", GIB).unwrap();
        assert!(matches!(
            l.reserve(Pool::HostPinned, "pack_staging", 1),
            Err(LedgerError::DuplicateReservation { .. })
        ));
        assert_eq!(l.release(Pool::HostPinned, "pack_staging"), Ok(GIB));
        assert_eq!(l.used(Pool::HostPinned), 0);
        assert!(matches!(
            l.release(Pool::HostPinned, "pack_staging"),
            Err(LedgerError::NoSuchReservation { .. })
        ));
        // Released bytes are reusable.
        l.reserve(Pool::HostPinned, "pack_staging", 4 * GIB)
            .unwrap();
    }

    #[test]
    fn nvme_write_bytes_are_counted_not_reserved() {
        let mut l = boot();
        l.reserve(Pool::Nvme, "session:7", 144 << 20).unwrap();
        l.record_written(Pool::Nvme, 144 << 20).unwrap();
        l.record_written(Pool::Nvme, 144 << 20).unwrap();
        assert_eq!(l.used(Pool::Nvme), 144 << 20);
        assert_eq!(l.written(Pool::Nvme), 288 << 20);
        l.release(Pool::Nvme, "session:7").unwrap();
        assert_eq!(l.used(Pool::Nvme), 0);
        assert_eq!(l.written(Pool::Nvme), 288 << 20, "writes are monotonic");
    }

    #[test]
    fn snapshot_and_metrics() {
        let mut l = boot();
        l.reserve(Pool::Vram(0), "weights", 11 * GIB).unwrap();
        l.reserve(Pool::Vram(1), "weights", 11 * GIB).unwrap();
        l.record_written(Pool::Nvme, 1024).unwrap();
        let snap = l.snapshot();
        assert_eq!(snap.pools.len(), 5);
        let r0 = snap
            .pools
            .iter()
            .find(|p| p.pool == Pool::Vram(0))
            .expect("rank 0");
        assert_eq!(r0.label, "vram.rank0");
        assert_eq!(r0.used_bytes, 11 * GIB);
        assert_eq!(r0.reservations.len(), 1);
        assert_eq!(r0.reservations[0].name, "weights");

        let text = snap.render_prometheus();
        assert!(text.contains("saber_ledger_capacity_bytes{pool=\"vram.rank0\"} 34359738368"));
        assert!(text.contains(
            "saber_ledger_reservation_bytes{pool=\"vram.rank1\",name=\"weights\"} 11811160064"
        ));
        assert!(text.contains("saber_ledger_written_bytes_total{pool=\"nvme\"} 1024"));
    }
}
