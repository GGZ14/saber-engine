//! FN63b — the SDMA COPY LANE: `hsa_amd_memory_async_copy(_on_engine)` between allocations HIP
//! made (the expert tier's pinned RAM slab and its VRAM slots), each copy with its own completion
//! signal from a ring the lane owns, dependencies as SIGNALS (the SDMA engine polls them; no host
//! wait, no stream order), host→device copies on one SDMA engine and device→host on another.
//!
//! What the lane is for (LEDGER FN63): FN59's 45 x 2.6 MB batch lands in 4,372 us on one in-order
//! HIP copy stream and in 2,957 us (-32 %) as unchained HSA copies -- the link's two directions at
//! once. The expert mover issues exactly such a batch every step; the order it needs is a per-slot
//! law (`saber_mem::device_mover::SignalPlanner`), which the lane takes as ticket dependencies.
//!
//! **Tickets.** Every operation is a TICKET (1, 2, 3, .. in issue order; 0 = none). Ticket `t`
//! owns ring entry `t % ring`: its completion signal and, for a word write, its 4-byte source. An
//! entry is reused by ticket `t + ring` only after `t` has landed AND every ticket that named `t`
//! as a dependency has landed (their engines polled `t`'s signal before they ran); the reuse waits
//! for that on the host when it must (counted: [`LaneCounts::ring_waits`]).
//!
//! **Regions (the safety law).** A copy names absolute addresses; each must fall inside a region
//! registered from a `saber-hip` allocation ([`SdmaLane::register_device`] /
//! [`SdmaLane::register_host`]) or the lane's own word table. The lane does not own those
//! allocations: their owner keeps them alive until the lane is dropped (its `Drop` waits for every
//! copy in flight) -- `saber_mem::device_mover::HipCopyQueue` declares its lane FIRST so the lane
//! drops before the slot arena and the slab.
//!
//! **Signals.** Created with the GPU as the only consumer (no interrupt event: ROCr's
//! `AsyncEventsLoop` never sees them); the host reads them (`signal_load`) and, for a drain or a
//! ring reuse, busy-waits them (a drain is rare: law 4's host writes, a reclaim).

use std::sync::atomic::{fence, Ordering};

use saber_hip::{DevicePtr, PinnedBuf};

use crate::{Agent, Error, Hsa, Result, Signal, WaitState};

/// Which SDMA engines carry the copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engines {
    /// `hsa_amd_memory_async_copy`: ROCr picks (FN63's unchained arm measured this form).
    Runtime,
    /// `hsa_amd_memory_async_copy_on_engine` (SDMA forced): H2D on `h2d`, D2H on `d2h` (one
    /// `HSA_AMD_SDMA_ENGINE_*` bit each).
    Explicit {
        /// The host→device engine bit.
        h2d: u32,
        /// The device→host engine bit.
        d2h: u32,
    },
}

/// How [`SdmaLane::open`] picks [`Engines`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineChoice {
    /// The lowest available engine for H2D, the lowest OTHER available engine for D2H; the
    /// runtime's choice when the entry points are missing or fewer than two engines exist.
    Auto,
    /// Engine indices (0-based) for H2D and D2H; refused when unavailable.
    Pair(u32, u32),
    /// Always the runtime's choice.
    Runtime,
}

impl std::str::FromStr for EngineChoice {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<EngineChoice, String> {
        match s {
            "auto" => Ok(EngineChoice::Auto),
            "runtime" => Ok(EngineChoice::Runtime),
            _ => {
                let (a, b) = s
                    .split_once(':')
                    .ok_or_else(|| format!("engines `{s}`: auto | runtime | H2D:D2H (e.g. 0:1)"))?;
                let p = |x: &str| {
                    x.parse::<u32>()
                        .ok()
                        .filter(|v| *v < 16)
                        .ok_or_else(|| format!("engines `{s}`: an engine index is 0..15"))
                };
                let (a, b) = (p(a)?, p(b)?);
                if a == b {
                    return Err(format!("engines `{s}`: H2D and D2H need different engines"));
                }
                Ok(EngineChoice::Pair(a, b))
            }
        }
    }
}

/// The lane's which-path counters (rule 5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneCounts {
    /// Host→device copies issued.
    pub h2d: u64,
    /// Device→host copies issued.
    pub d2h: u64,
    /// FN71c: device→device copies issued (the P2PDMA pool -> a VRAM slot).
    pub d2d: u64,
    /// 4-byte word writes (layer / all marks) issued.
    pub words: u64,
    /// Dependency signals handed to the engines (each = two SDMA polls).
    pub deps: u64,
    /// Dependencies dropped because they had already landed when the op was issued.
    pub deps_landed: u64,
    /// Ring reuses that had to wait on the host (an entry's ticket or a dependent still in flight).
    pub ring_waits: u64,
    /// Host waits ([`SdmaLane::wait`]) that found the ticket still in flight.
    pub host_waits: u64,
}

#[derive(Debug, Clone, Copy)]
struct Region {
    base: u64,
    len: u64,
    host: bool,
}

/// The lane (see the module docs).
pub struct SdmaLane {
    hsa: Hsa,
    cpu: Agent,
    gpu: Agent,
    engines: Engines,
    masks: (Option<u32>, Option<u32>),
    sigs: Vec<Signal>,
    /// Per ring entry: the highest ticket that named the entry's ticket as a dependency (0: none).
    dep_hi: Vec<u64>,
    /// The word-write sources, one u32 per ring entry (ROCr fine-grained system memory).
    vals: *mut u32,
    regions: Vec<Region>,
    next: u64,
    hz: f64,
    profiling: bool,
    /// The counters.
    pub counts: LaneCounts,
}

impl std::fmt::Debug for SdmaLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SdmaLane({:?}, ring {}, next {})",
            self.engines,
            self.sigs.len(),
            self.next
        )
    }
}

/// A busy wait gives up after this long (a copy that never lands is an error, never a hang).
const WAIT_SECS: f64 = 10.0;

impl SdmaLane {
    /// Open ROCr (`soname`), find the GPU agent at PCI `bdfid` (`(bus << 8) | (dev << 3) | fn`)
    /// and the first CPU agent, pick the engines, create `ring` signals and the word table.
    pub fn open(soname: &str, bdfid: u32, ring: usize, choice: EngineChoice) -> Result<SdmaLane> {
        if !(16..=1 << 20).contains(&ring) {
            return Err(Error::Invalid(format!("ring {ring}: 16 ..= 1 Mi entries")));
        }
        let hsa = Hsa::open(soname)?;
        let agents = hsa.agents()?;
        let gpu = agents
            .iter()
            .find(|a| a.gpu && a.bdfid == bdfid)
            .ok_or_else(|| Error::Invalid(format!("no HSA GPU agent at bdfid {bdfid:#x}")))?
            .agent;
        let cpu = agents
            .iter()
            .find(|a| !a.gpu)
            .ok_or_else(|| Error::Invalid("no HSA CPU agent".into()))?
            .agent;
        let masks = (
            hsa.copy_engine_status(gpu, cpu).ok().flatten(),
            hsa.copy_engine_status(cpu, gpu).ok().flatten(),
        );
        // the status masks are the engines IDLE right now (HIP's own copies may hold one), so
        // they only order the choice; the boot self-test is what proves an engine
        let low = |m: u32| (m != 0).then(|| m & m.wrapping_neg());
        let engines = match choice {
            EngineChoice::Runtime => Engines::Runtime,
            EngineChoice::Auto if !hsa.has_copy_on_engine() => Engines::Runtime,
            EngineChoice::Auto => {
                let u = masks.0.unwrap_or(0) | masks.1.unwrap_or(0);
                match low(u).and_then(|a| low(u & !a).map(|b| (a, b))) {
                    Some((a, b)) => Engines::Explicit { h2d: a, d2h: b },
                    None => Engines::Explicit { h2d: 1, d2h: 2 },
                }
            }
            EngineChoice::Pair(a, b) => {
                if !hsa.has_copy_on_engine() {
                    return Err(Error::Invalid(format!(
                        "engines {a}:{b}: this ROCr has no hsa_amd_memory_async_copy_on_engine"
                    )));
                }
                Engines::Explicit {
                    h2d: 1u32 << a,
                    d2h: 1u32 << b,
                }
            }
        };
        let hz = hsa.timestamp_hz()? as f64;
        let mut sigs = Vec::with_capacity(ring);
        for _ in 0..ring {
            match hsa.signal(0, &[gpu]) {
                Ok(s) => sigs.push(s),
                Err(e) => {
                    for s in sigs {
                        let _ = hsa.signal_destroy(s);
                    }
                    return Err(e);
                }
            }
        }
        let vals = match hsa.alloc_fine_sys(cpu, gpu, ring * 4) {
            Ok(p) => p.cast::<u32>(),
            Err(e) => {
                for s in sigs {
                    let _ = hsa.signal_destroy(s);
                }
                return Err(e);
            }
        };
        Ok(SdmaLane {
            hsa,
            cpu,
            gpu,
            engines,
            masks,
            dep_hi: vec![0; ring],
            sigs,
            vals,
            regions: vec![Region {
                base: vals as u64,
                len: ring as u64 * 4,
                host: true,
            }],
            next: 1,
            hz,
            profiling: false,
            counts: LaneCounts::default(),
        })
    }

    /// The engines in use.
    #[must_use]
    pub fn engines(&self) -> Engines {
        self.engines
    }

    /// Switch to the runtime's engine choice (before the first ticket: the `auto` fallback when
    /// the explicit pair fails its self-test).
    pub fn set_runtime_engines(&mut self) -> Result<()> {
        if self.next != 1 {
            return Err(Error::Invalid(
                "engines change after the first ticket".into(),
            ));
        }
        self.engines = Engines::Runtime;
        Ok(())
    }

    /// The availability masks ROCr reported (H2D, D2H; `None` = no entry point).
    #[must_use]
    pub fn masks(&self) -> (Option<u32>, Option<u32>) {
        self.masks
    }

    /// The ring size.
    #[must_use]
    pub fn ring(&self) -> usize {
        self.sigs.len()
    }

    /// The next ticket [`Self::copy`] / [`Self::write_word`] expects.
    #[must_use]
    pub fn next_ticket(&self) -> u64 {
        self.next
    }

    /// Register a HIP device allocation as a copy region (its owner keeps it alive while the
    /// lane lives -- the module's safety law).
    pub fn register_device(&mut self, p: &DevicePtr) {
        self.regions.push(Region {
            base: p.addr().as_u64(),
            len: p.len(),
            host: false,
        });
    }

    /// Register a HIP pinned host allocation as a copy region (same law).
    pub fn register_host(&mut self, p: &PinnedBuf) {
        self.regions.push(Region {
            base: p.addr() as u64,
            len: p.len() as u64,
            host: true,
        });
    }

    /// FN70: register a region this crate allocated itself (an NVMe reader's staging or its
    /// host-visible VRAM window); the allocation outlives the lane by the same law.
    pub(crate) fn register_raw(&mut self, base: u64, len: u64, host: bool) {
        self.regions.push(Region { base, len, host });
    }

    /// FN70: the runtime and the (CPU, GPU) agents the lane opened.
    pub(crate) fn runtime(&self) -> (&Hsa, Agent, Agent) {
        (&self.hsa, self.cpu, self.gpu)
    }

    fn region(&self, addr: u64, bytes: u64) -> Result<bool> {
        self.regions
            .iter()
            .find(|r| {
                addr >= r.base && addr.checked_add(bytes).is_some_and(|e| e <= r.base + r.len)
            })
            .map(|r| r.host)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "copy range {addr:#x} + {bytes} lies in no registered region"
                ))
            })
    }

    /// Whether ticket `t` has landed (0 = none: true; a negative signal = a failed copy: true
    /// here, and [`Self::wait`] reports it).
    #[must_use]
    pub fn done(&self, t: u64) -> bool {
        if t == 0 || t + self.sigs.len() as u64 <= self.next {
            return true;
        }
        if t >= self.next {
            return false;
        }
        self.hsa
            .signal_load(self.sigs[(t % self.sigs.len() as u64) as usize])
            < 1
    }

    /// The host waits (busy) until ticket `t` has landed; a failed or never-landing copy is an
    /// error.
    pub fn wait(&mut self, t: u64) -> Result<()> {
        if t == 0 || t + self.sigs.len() as u64 <= self.next {
            return Ok(());
        }
        if t >= self.next {
            return Err(Error::Invalid(format!(
                "wait on ticket {t}, never issued (next {})",
                self.next
            )));
        }
        let s = self.sigs[(t % self.sigs.len() as u64) as usize];
        let mut v = self.hsa.signal_load(s);
        if v >= 1 {
            self.counts.host_waits += 1;
            v = self
                .hsa
                .wait_done(s, WaitState::Active, (self.hz * WAIT_SECS) as u64);
        }
        if v < 0 {
            return Err(Error::Invalid(format!(
                "SDMA copy ticket {t} FAILED (signal {v})"
            )));
        }
        if v >= 1 {
            return Err(Error::Invalid(format!(
                "SDMA copy ticket {t} did not land in {WAIT_SECS} s"
            )));
        }
        Ok(())
    }

    /// The host waits for every ticket in flight.
    pub fn wait_all(&mut self) -> Result<()> {
        let lo = self.next.saturating_sub(self.sigs.len() as u64).max(1);
        for t in lo..self.next {
            self.wait(t)?;
        }
        Ok(())
    }

    /// Ticket `k`'s ring entry is free: its previous ticket and every dependent of it landed.
    fn reuse(&mut self, k: u64) -> Result<usize> {
        let ring = self.sigs.len() as u64;
        let i = (k % ring) as usize;
        if k > ring {
            let (p, h) = (k - ring, self.dep_hi[i]);
            let waits = self.counts.host_waits;
            // the previous occupant first: `wait` treats a ticket `ring` behind `next` as landed
            // (it is about to be reused), so check its signal here directly
            let s = self.sigs[i];
            let mut v = self.hsa.signal_load(s);
            if v >= 1 {
                self.counts.host_waits += 1;
                v = self
                    .hsa
                    .wait_done(s, WaitState::Active, (self.hz * WAIT_SECS) as u64);
            }
            if v != 0 {
                return Err(Error::Invalid(format!(
                    "ring reuse: ticket {p} {}",
                    if v < 0 { "FAILED" } else { "never landed" }
                )));
            }
            if h > p {
                self.wait(h)?;
            }
            if self.counts.host_waits != waits {
                self.counts.ring_waits += 1;
            }
            self.dep_hi[i] = 0;
        }
        Ok(i)
    }

    fn dep_signals(&mut self, k: u64, deps: &[u64]) -> Result<Vec<Signal>> {
        let ring = self.sigs.len() as u64;
        let mut out = Vec::with_capacity(deps.len());
        for &d in deps {
            if d == 0 {
                continue;
            }
            if d >= k {
                return Err(Error::Invalid(format!(
                    "ticket {k} depends on {d}, not issued before it"
                )));
            }
            if self.done(d) {
                self.counts.deps_landed += 1;
                continue;
            }
            let i = (d % ring) as usize;
            self.dep_hi[i] = self.dep_hi[i].max(k);
            out.push(self.sigs[i]);
        }
        self.counts.deps += out.len() as u64;
        Ok(out)
    }

    /// Issue ticket `t` (must be [`Self::next_ticket`]): copy `bytes` from `src` to `dst` (one
    /// host, one device address, both in registered regions) after every ticket in `deps` landed.
    pub fn copy(&mut self, t: u64, dst: u64, src: u64, bytes: u64, deps: &[u64]) -> Result<()> {
        if t != self.next {
            return Err(Error::Invalid(format!(
                "ticket {t} issued out of order (next {})",
                self.next
            )));
        }
        let (dh, sh) = (self.region(dst, bytes)?, self.region(src, bytes)?);
        if dh == sh {
            return Err(Error::Invalid(format!(
                "copy {src:#x} -> {dst:#x}: the lane carries host<->device copies only"
            )));
        }
        self.issue(t, dst, src, bytes, sh, deps)?;
        if sh {
            self.counts.h2d += 1;
        } else {
            self.counts.d2h += 1;
        }
        Ok(())
    }

    /// FN71c: issue ticket `t`: copy `bytes` from device address `src` to device address `dst`
    /// (both in registered DEVICE regions -- the P2PDMA pool's GPU view into a VRAM slot) on
    /// ROCr's own same-device route (`hsa_amd_memory_async_copy`: its blit, not the lane's SDMA
    /// engine pair, which is chosen for host<->device), after every ticket in `deps` landed; the
    /// ticket's signal and the ordering law are the lane's.
    pub fn copy_d2d(&mut self, t: u64, dst: u64, src: u64, bytes: u64, deps: &[u64]) -> Result<()> {
        if t != self.next {
            return Err(Error::Invalid(format!(
                "ticket {t} issued out of order (next {})",
                self.next
            )));
        }
        if self.region(dst, bytes)? || self.region(src, bytes)? {
            return Err(Error::Invalid(format!(
                "d2d copy {src:#x} -> {dst:#x}: both ends must be device regions"
            )));
        }
        let i = self.reuse(t)?;
        self.issue_agents(t, i, (dst, self.gpu), (src, self.gpu), bytes, true, deps)?;
        self.counts.d2d += 1;
        Ok(())
    }

    /// Issue ticket `t`: write the 32-bit `value` to device address `dst` (a registered device
    /// region) on the H2D engine, after every ticket in `deps` landed.
    pub fn write_word(&mut self, t: u64, dst: u64, value: u32, deps: &[u64]) -> Result<()> {
        if t != self.next {
            return Err(Error::Invalid(format!(
                "ticket {t} issued out of order (next {})",
                self.next
            )));
        }
        if self.region(dst, 4)? {
            return Err(Error::Invalid(format!(
                "word write to {dst:#x}: not a device region"
            )));
        }
        let i = (t % self.sigs.len() as u64) as usize;
        // the entry's source word is free once its previous ticket landed (`issue` -> `reuse`
        // checks that BEFORE this store would be visible to an engine: do the reuse first)
        let i2 = self.reuse(t)?;
        debug_assert_eq!(i, i2);
        // SAFETY: `vals` holds `ring` u32s (ROCr allocation owned by the lane) and `i < ring`;
        // no engine reads entry `i` now (its previous ticket landed: `reuse`).
        unsafe { self.vals.add(i).write_volatile(value) };
        let src = self.vals as u64 + 4 * i as u64;
        self.issue_reused(t, i, dst, src, 4, true, deps)?;
        self.counts.words += 1;
        Ok(())
    }

    fn issue(
        &mut self,
        t: u64,
        dst: u64,
        src: u64,
        bytes: u64,
        h2d: bool,
        deps: &[u64],
    ) -> Result<()> {
        let i = self.reuse(t)?;
        self.issue_reused(t, i, dst, src, bytes, h2d, deps)
    }

    #[allow(clippy::too_many_arguments)]
    fn issue_reused(
        &mut self,
        t: u64,
        i: usize,
        dst: u64,
        src: u64,
        bytes: u64,
        h2d: bool,
        deps: &[u64],
    ) -> Result<()> {
        let (da, sa) = if h2d {
            (self.gpu, self.cpu)
        } else {
            (self.cpu, self.gpu)
        };
        self.issue_agents(t, i, (dst, da), (src, sa), bytes, h2d, deps)
    }

    /// Issue one copy with explicit (address, agent) ends on the H2D (`h2d_engine`) or the D2H
    /// engine.
    #[allow(clippy::too_many_arguments)]
    fn issue_agents(
        &mut self,
        t: u64,
        i: usize,
        (dst, da): (u64, Agent),
        (src, sa): (u64, Agent),
        bytes: u64,
        h2d: bool,
        deps: &[u64],
    ) -> Result<()> {
        let dep = self.dep_signals(t, deps)?;
        let sig = self.sigs[i];
        self.hsa.signal_store(sig, 1);
        fence(Ordering::SeqCst);
        let n = usize::try_from(bytes).map_err(|_| Error::Invalid("copy size".into()))?;
        // SAFETY: both ranges lie in registered regions (checked by the callers), whose owners
        // keep them alive while the lane lives; the completion signal is the ring entry `reuse`
        // freed; every dependency is an issued ticket's signal.
        let r = unsafe {
            match self.engines {
                // FN71c: a same-device copy (both ends the GPU) takes ROCr's own route
                _ if da == sa => self.hsa.async_copy(dst, da, src, sa, n, &dep, sig),
                Engines::Runtime => self.hsa.async_copy(dst, da, src, sa, n, &dep, sig),
                Engines::Explicit { h2d: he, d2h: de } => self.hsa.async_copy_on_engine(
                    dst,
                    da,
                    src,
                    sa,
                    n,
                    &dep,
                    sig,
                    if h2d { he } else { de },
                ),
            }
        };
        if let Err(e) = r {
            // the entry never carried a copy: leave it landed
            self.hsa.signal_store(sig, 0);
            return Err(e);
        }
        self.next = t + 1;
        Ok(())
    }

    /// FN63b's boot self-test over a registered device word: H2D `magic` into it on the H2D
    /// engine, D2H it back on the D2H engine (the second after the first, by a signal), compare.
    /// Uses ring entries 0..2's sources and its own signals; call before the first ticket.
    pub fn self_test(&mut self, word: u64) -> Result<u32> {
        if self.next != 1 {
            return Err(Error::Invalid("self_test after the first ticket".into()));
        }
        if self.region(word, 4)? {
            return Err(Error::Invalid("self_test needs a device word".into()));
        }
        let magic = 0x5EB0_0000_u32
            | (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
                & 0xffff);
        let a = self.hsa.signal(1, &[self.gpu])?;
        let b = self.hsa.signal(1, &[self.gpu])?;
        let r = (|| {
            // SAFETY: entries 0 and 1 of the lane's own table; no ticket uses them yet.
            unsafe {
                self.vals.write_volatile(magic);
                self.vals.add(1).write_volatile(0);
            }
            fence(Ordering::SeqCst);
            let v0 = self.vals as u64;
            // SAFETY: the word is a registered device region; the table is the lane's; both
            // copies are waited for below before anything is freed.
            unsafe {
                match self.engines {
                    Engines::Runtime => {
                        self.hsa
                            .async_copy(word, self.gpu, v0, self.cpu, 4, &[], a)?;
                    }
                    Engines::Explicit { h2d, .. } => {
                        self.hsa.async_copy_on_engine(
                            word,
                            self.gpu,
                            v0,
                            self.cpu,
                            4,
                            &[],
                            a,
                            h2d,
                        )?;
                    }
                }
            }
            // SAFETY: as above.
            let second = unsafe {
                match self.engines {
                    Engines::Runtime => {
                        self.hsa
                            .async_copy(v0 + 4, self.cpu, word, self.gpu, 4, &[a], b)
                    }
                    Engines::Explicit { d2h, .. } => self.hsa.async_copy_on_engine(
                        v0 + 4,
                        self.cpu,
                        word,
                        self.gpu,
                        4,
                        &[a],
                        b,
                        d2h,
                    ),
                }
            };
            if let Err(e) = second {
                // the first copy is in flight: it lands before its signal is destroyed
                let _ = self
                    .hsa
                    .wait_done(a, WaitState::Active, (self.hz * WAIT_SECS) as u64);
                return Err(e);
            }
            for s in [a, b] {
                let v = self
                    .hsa
                    .wait_done(s, WaitState::Active, (self.hz * WAIT_SECS) as u64);
                if v != 0 {
                    return Err(Error::Invalid(format!("self-test copy signal {v}")));
                }
            }
            // SAFETY: entry 1, written by the D2H that just landed.
            let back = unsafe { self.vals.add(1).read_volatile() };
            if back != magic {
                return Err(Error::Invalid(format!(
                    "self-test read back {back:#x}, wrote {magic:#x}"
                )));
            }
            Ok(magic)
        })();
        let _ = self.hsa.signal_destroy(a);
        let _ = self.hsa.signal_destroy(b);
        r
    }

    /// Enable / disable ROCr's async-copy profiling (process-wide: the census arm only).
    pub fn set_profiling(&mut self, on: bool) -> Result<()> {
        self.hsa.copy_profiling(on)?;
        self.profiling = on;
        Ok(())
    }

    /// A landed ticket's (start, end) in the system timestamp domain (profiling on; `None` when
    /// the ticket is not landed or its entry was reused).
    #[must_use]
    pub fn times(&self, t: u64) -> Option<(u64, u64)> {
        if !self.profiling || t == 0 || t >= self.next || t + self.sigs.len() as u64 <= self.next {
            return None;
        }
        let s = self.sigs[(t % self.sigs.len() as u64) as usize];
        if self.hsa.signal_load(s) != 0 {
            return None;
        }
        self.hsa.copy_time(s).ok().map(|x| (x.start, x.end))
    }

    /// The system timestamp now (the profiling domain).
    pub fn now(&self) -> Result<u64> {
        self.hsa.timestamp()
    }

    /// Ticks per second of [`Self::now`].
    #[must_use]
    pub fn hz(&self) -> f64 {
        self.hz
    }
}

impl Drop for SdmaLane {
    fn drop(&mut self) {
        // every copy lands before the regions' owners may free them (the safety law)
        if let Err(e) = self.wait_all() {
            eprintln!("saber-hsa lane: drain at drop: {e}");
        }
        if self.profiling {
            let _ = self.hsa.copy_profiling(false);
        }
        for s in std::mem::take(&mut self.sigs) {
            let _ = self.hsa.signal_destroy(s);
        }
        // SAFETY: the table came from `alloc_fine_sys`; nothing in flight reads it (drained).
        let _ = unsafe { self.hsa.free(self.vals.cast()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_choices_parse() {
        assert_eq!("auto".parse::<EngineChoice>(), Ok(EngineChoice::Auto));
        assert_eq!("runtime".parse::<EngineChoice>(), Ok(EngineChoice::Runtime));
        assert_eq!("0:1".parse::<EngineChoice>(), Ok(EngineChoice::Pair(0, 1)));
        assert_eq!("1:0".parse::<EngineChoice>(), Ok(EngineChoice::Pair(1, 0)));
        assert!("1:1".parse::<EngineChoice>().is_err());
        assert!("0:16".parse::<EngineChoice>().is_err());
        assert!("fast".parse::<EngineChoice>().is_err());
    }

    #[test]
    fn a_lane_without_a_runtime_is_not_available_not_a_panic() {
        let e = SdmaLane::open("libsaber-no-such-hsa.so", 0x300, 64, EngineChoice::Auto)
            .expect_err("no runtime");
        assert!(matches!(e, Error::NotAvailable(_)), "{e}");
        let e = SdmaLane::open("libsaber-no-such-hsa.so", 0x300, 8, EngineChoice::Auto)
            .expect_err("ring too small");
        assert!(matches!(e, Error::Invalid(_)), "{e}");
    }
}
