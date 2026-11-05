//! `RankPeer` — one rank's side of the two-rank IPC exchange.
//!
//! The one-shot all-reduce is the first route in the engine whose *correctness depends on another
//! rank*. Everything else is a kernel over buffers this process owns; this one writes into the
//! peer's scratch and then spins on a flag the peer wrote. Three consequences shape this module:
//!
//! 1. **The exchange happens once, before capture.** Addresses obtained from
//!    `hipIpcOpenMemHandle` are baked into kernarg images and into graph nodes, so they must be
//!    established at plan-bind time and outlive every replay (invariant I1). A handle exchanged
//!    mid-capture would be an allocation inside a capture, which is illegal, and an address that
//!    changed between replays, which is corruption.
//! 2. **Disagreement between ranks must be a refusal, not a hang.** A kernel that spins for a
//!    push that will never come does not return a wrong answer — it never retires, and it takes
//!    the queue with it. So the two ranks compare a digest of everything the protocol depends on
//!    *before* either launches, and [`RankPeer::agree`] refuses when they differ. This is the
//!    cheap check that turns the worst failure mode into a message.
//! 3. **The transport is abstracted, because the deployment is not decided.** The gate runs two
//!    ranks as two OS threads in one process, each with its own [`saber_hip::HipThread`] bound to
//!    its own card. A served TP=2 engine may use two processes. Both are the same protocol, so the
//!    transport is a trait and the protocol is not.
//!
//! What is deliberately NOT here: any notion of a collective schedule, of more than two ranks, or
//! of dynamic membership. The reference's kernel is `2rank` in its name and in its body.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

use saber_hip::IpcMemHandle;

use crate::error::{Error, Result};

/// How long a rendezvous waits before deciding the peer is not coming.
///
/// A rendezvous that blocks forever is the same bug as a kernel that spins forever, moved to the
/// host. Every wait in this module is bounded.
pub const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// What one rank publishes to the other, and what it needs back.
///
/// Implementors carry no HIP state: a handle is 64 opaque bytes and a digest is a `u64`, so the
/// transport can be a channel, a socket or a file without knowing anything about the device.
pub trait IpcExchange: Send + Sync {
    /// This rank's ordinal.
    fn rank(&self) -> u32;
    /// The other rank's ordinal. Two ranks, so this is `1 - rank`.
    fn peer_rank(&self) -> u32;
    /// Publish `mine` under `name` and return the peer's handle for the same name.
    fn swap_handle(&self, name: &str, mine: IpcMemHandle) -> Result<IpcMemHandle>;
    /// Publish `mine` under `name` and return the peer's value.
    fn swap_digest(&self, name: &str, mine: u64) -> Result<u64>;
    /// Publish `mine` under `name` and return the peer's bytes.
    ///
    /// **Both ranks must send the same number of bytes**, because the reader sizes its read from
    /// what it is sending: these are shard exchanges of a known, symmetric width — a vocabulary
    /// shard of logits, a rank's partial embedding — not a general message channel. A length
    /// mismatch is a protocol bug and this returns one rather than blocking forever.
    ///
    /// This is a HOST path and it is not on the serving path: the two uses are the logits
    /// all-gather (`logits.gather` has no code object) and cross-rank reference checks in a gate.
    fn swap_bytes(&self, name: &str, mine: &[u8]) -> Result<Vec<u8>>;
    /// Publish `mine` under `name` and return the peer's bytes, **with no symmetry requirement**.
    ///
    /// The frame carries its own length, so the two ranks may send different widths and neither
    /// has to be told the other's in advance. That is the whole difference from [`swap_bytes`],
    /// and it is what ws-h29 asked for when it counted what is left on this wire:
    ///
    /// * `swap_bytes` is SYMMETRIC by contract, so the side that has nothing to say answers with a
    ///   buffer of zeros as wide as the side that does — the engine's step frame was **1,932 KiB
    ///   in each direction** to carry a leader-to-follower payload (ws-h29's own table);
    /// * and it agrees the length over the socket first, which is a whole extra round trip per
    ///   call, five times a decode step.
    ///
    /// This is still a RENDEZVOUS — neither side returns until both have sent — because the two
    /// ranks' captured graphs contain each other's one-shot all-reduces, whose bounded spin is
    /// what a pair that drifts apart exhausts (`docs/EXEC.md` §4). A one-way send would let the
    /// leader run ahead; an exchange where one side's frame is eight bytes costs one round trip
    /// and keeps the lockstep.
    ///
    /// # Errors
    /// The transport, or a peer answering a frame this call did not ask for.
    fn xchg_bytes(&self, name: &str, mine: &[u8]) -> Result<Vec<u8>>;
    /// Both ranks reach `tag` before either returns from it.
    fn rendezvous(&self, tag: &str) -> Result<()>;
}

/// One rank's view of the pair.
pub struct RankPeer {
    exchange: Arc<dyn IpcExchange>,
}

impl RankPeer {
    /// Wrap a transport.
    #[must_use]
    pub fn new(exchange: Arc<dyn IpcExchange>) -> Self {
        RankPeer { exchange }
    }

    /// This rank's ordinal.
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.exchange.rank()
    }

    /// The other rank's ordinal.
    #[must_use]
    pub fn peer_rank(&self) -> u32 {
        self.exchange.peer_rank()
    }

    /// The transport, so a second holder (the [`crate::Launcher`], say) can share it rather than
    /// opening a second one. Two transports between the same pair of ranks would interleave their
    /// messages and pair the wrong handles.
    #[must_use]
    pub fn exchange(&self) -> &Arc<dyn IpcExchange> {
        &self.exchange
    }

    /// Exchange one region's handle.
    pub fn swap_handle(&self, name: &str, mine: IpcMemHandle) -> Result<IpcMemHandle> {
        self.exchange.swap_handle(name, mine)
    }

    /// Both ranks reach this point before either goes on.
    pub fn rendezvous(&self, tag: &str) -> Result<()> {
        self.exchange.rendezvous(tag)
    }

    /// Refuse unless both ranks computed the same digest for `what`.
    ///
    /// This is the negative control for the protocol, and it is what keeps a mismatch from
    /// becoming a hang. The digest covers everything the two ranks must agree on — the kernel
    /// symbol, the launch geometry, the byte size of every IPC region, and the **reset epoch** of
    /// the sequence counters, because `slot = seq & 1` picks which half of the double-buffered
    /// scratch both ranks use and a rank that re-zeroed its counters while the other did not is
    /// reading the wrong half.
    pub fn agree(&self, what: &str, mine: u64) -> Result<()> {
        let theirs = self.exchange.swap_digest(what, mine)?;
        if theirs == mine {
            return Ok(());
        }
        Err(Error::PeerDisagreement {
            what: what.to_owned(),
            rank: self.rank(),
            mine,
            peer_rank: self.peer_rank(),
            theirs,
        })
    }
}

impl std::fmt::Debug for RankPeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RankPeer(rank {} of 2, peer {})",
            self.rank(),
            self.peer_rank()
        )
    }
}

// ------------------------------------------------------------------------------------------- //
// The in-process transport                                                                      //
// ------------------------------------------------------------------------------------------- //

/// What crosses the channel. Keyed by name so a mis-ordered exchange is caught rather than
/// silently pairing a scratch handle with a flags handle.
#[derive(Debug)]
enum Msg {
    Handle(String, IpcMemHandle),
    Digest(String, u64),
    Bytes(String, Vec<u8>),
}

/// Two ranks as two OS threads in one process.
///
/// Enough for the bit-exact gate, where each rank is a thread with its own `HipThread` and its own
/// `hipSetDevice`. It is NOT a shortcut around the IPC path: the handles still go through
/// `hipIpcGetMemHandle` / `hipIpcOpenMemHandle`, because that is what the kernel's addressing
/// assumes and because a same-process pointer would test nothing.
pub struct ThreadPairExchange {
    rank: u32,
    tx: Sender<Msg>,
    rx: Mutex<Receiver<Msg>>,
    barrier: Arc<Barrier>,
    timeout: Duration,
}

impl ThreadPairExchange {
    /// Build both ends with the default [`RENDEZVOUS_TIMEOUT`]. Hand one to each rank's thread.
    #[must_use]
    pub fn pair() -> (ThreadPairExchange, ThreadPairExchange) {
        Self::pair_with_timeout(RENDEZVOUS_TIMEOUT)
    }

    /// [`ThreadPairExchange::pair`] with an explicit deadline, so a test can prove the wait is
    /// bounded without waiting the full production timeout to find out.
    #[must_use]
    pub fn pair_with_timeout(timeout: Duration) -> (ThreadPairExchange, ThreadPairExchange) {
        let (tx0, rx1) = std::sync::mpsc::channel();
        let (tx1, rx0) = std::sync::mpsc::channel();
        let barrier = Arc::new(Barrier::new(2));
        (
            ThreadPairExchange {
                rank: 0,
                tx: tx0,
                rx: Mutex::new(rx0),
                barrier: Arc::clone(&barrier),
                timeout,
            },
            ThreadPairExchange {
                rank: 1,
                tx: tx1,
                rx: Mutex::new(rx1),
                barrier,
                timeout,
            },
        )
    }

    fn recv(&self, want: &str) -> Result<Msg> {
        let rx = self.rx.lock().map_err(|_| {
            Error::Refused("the peer exchange receiver was poisoned by a panicking rank".to_owned())
        })?;
        rx.recv_timeout(self.timeout).map_err(|e| {
            Error::Refused(format!(
                "rank {}: waiting for the peer's `{want}` timed out after {:?} ({e})",
                self.rank, self.timeout
            ))
        })
    }
}

impl IpcExchange for ThreadPairExchange {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn peer_rank(&self) -> u32 {
        1 - self.rank
    }

    fn swap_handle(&self, name: &str, mine: IpcMemHandle) -> Result<IpcMemHandle> {
        self.tx
            .send(Msg::Handle(name.to_owned(), mine))
            .map_err(|_| Error::Refused(format!("rank {}: the peer is gone", self.rank)))?;
        match self.recv(name)? {
            Msg::Handle(got, h) if got == name => Ok(h),
            Msg::Handle(got, _) => Err(Error::Refused(format!(
                "rank {}: asked the peer for `{name}` and got `{got}`; the two ranks are \
                 exchanging handles in different orders",
                self.rank
            ))),
            Msg::Digest(got, _) => Err(Error::Refused(format!(
                "rank {}: asked the peer for handle `{name}` and got digest `{got}`",
                self.rank
            ))),
            Msg::Bytes(got, _) => Err(Error::Refused(format!(
                "rank {}: asked the peer for handle `{name}` and got bytes `{got}`",
                self.rank
            ))),
        }
    }

    fn swap_digest(&self, name: &str, mine: u64) -> Result<u64> {
        self.tx
            .send(Msg::Digest(name.to_owned(), mine))
            .map_err(|_| Error::Refused(format!("rank {}: the peer is gone", self.rank)))?;
        match self.recv(name)? {
            Msg::Digest(got, v) if got == name => Ok(v),
            Msg::Digest(got, _) => Err(Error::Refused(format!(
                "rank {}: asked the peer for `{name}` and got `{got}`",
                self.rank
            ))),
            Msg::Handle(got, _) => Err(Error::Refused(format!(
                "rank {}: asked the peer for digest `{name}` and got handle `{got}`",
                self.rank
            ))),
            Msg::Bytes(got, _) => Err(Error::Refused(format!(
                "rank {}: asked the peer for digest `{name}` and got bytes `{got}`",
                self.rank
            ))),
        }
    }

    fn swap_bytes(&self, name: &str, mine: &[u8]) -> Result<Vec<u8>> {
        self.tx
            .send(Msg::Bytes(name.to_owned(), mine.to_vec()))
            .map_err(|_| Error::Refused(format!("rank {}: the peer is gone", self.rank)))?;
        match self.recv(name)? {
            Msg::Bytes(got, v) if got == name => {
                if v.len() != mine.len() {
                    return Err(Error::Refused(format!(
                        "rank {}: `{name}` is {} bytes here and {} on the peer; swap_bytes is \
                         for symmetric shards",
                        self.rank,
                        mine.len(),
                        v.len()
                    )));
                }
                Ok(v)
            }
            other => Err(Error::Refused(format!(
                "rank {}: asked the peer for bytes `{name}` and got {other:?}",
                self.rank
            ))),
        }
    }

    fn xchg_bytes(&self, name: &str, mine: &[u8]) -> Result<Vec<u8>> {
        // A channel carries a `Vec` whatever its length, so the only thing `swap_bytes` adds over
        // this one is the symmetry CHECK — and that is exactly what an asymmetric frame drops.
        self.tx
            .send(Msg::Bytes(name.to_owned(), mine.to_vec()))
            .map_err(|_| Error::Refused(format!("rank {}: the peer is gone", self.rank)))?;
        match self.recv(name)? {
            Msg::Bytes(got, v) if got == name => Ok(v),
            other => Err(Error::Refused(format!(
                "rank {}: asked the peer for bytes `{name}` and got {other:?}",
                self.rank
            ))),
        }
    }

    fn rendezvous(&self, _tag: &str) -> Result<()> {
        // `Barrier` has no timed form. It is safe here because every other wait in this module is
        // bounded, so a rank that is going to fail fails at a timed step and drops its channel,
        // which unblocks the peer's next `recv` with an error rather than leaving it here.
        self.barrier.wait();
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------- //
// The cross-process transport                                                                   //
// ------------------------------------------------------------------------------------------- //

/// Two ranks as two PROCESSES, over a loopback TCP stream.
///
/// **This is the only transport that can exercise the IPC path.** `hipIpcOpenMemHandle` opens "an
/// interprocess memory handle exported from *another process*"; handing it a handle this process
/// exported returns `hipErrorInvalidContext`, measured on 2026-09-18. So the obvious shape — two
/// threads, two `HipThread`s, two devices, one process — allocates and exports fine and then fails
/// at the open, and no amount of peer-access enabling changes it.
///
/// Loopback TCP rather than a socketpair or a FIFO because it needs no `libc` in this crate, and
/// because "the two ranks are two processes that find each other over a socket" is also what a
/// served TP=2 engine will do.
pub struct TcpExchange {
    rank: u32,
    stream: Mutex<TcpStream>,
}

impl TcpExchange {
    /// Rank 0: listen on `127.0.0.1:port` and accept the peer.
    pub fn accept(listener: &TcpListener, timeout: Duration) -> Result<TcpExchange> {
        let (stream, _) = listener
            .accept()
            .map_err(|e| Error::Refused(format!("rank 0: accepting the peer: {e}")))?;
        Self::wrap(0, stream, timeout)
    }

    /// Rank 1: connect to the port rank 0 is listening on.
    pub fn connect(port: u16, timeout: Duration) -> Result<TcpExchange> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(stream) => return Self::wrap(1, stream, timeout),
                Err(e) if std::time::Instant::now() >= deadline => {
                    return Err(Error::Refused(format!(
                        "rank 1: could not reach rank 0 on port {port} within {timeout:?}: {e}"
                    )))
                }
                Err(_) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    fn wrap(rank: u32, stream: TcpStream, timeout: Duration) -> Result<TcpExchange> {
        // Bounded reads, for the same reason every other wait here is bounded.
        stream
            .set_read_timeout(Some(timeout))
            .and_then(|()| stream.set_write_timeout(Some(timeout)))
            .and_then(|()| stream.set_nodelay(true))
            .map_err(|e| Error::Refused(format!("rank {rank}: socket options: {e}")))?;
        Ok(TcpExchange {
            rank,
            stream: Mutex::new(stream),
        })
    }

    /// One frame: a kind byte, a length-prefixed name, then the payload.
    fn exchange(&self, kind: u8, name: &str, payload: &[u8], want: usize) -> Result<Vec<u8>> {
        let mut s = self.stream.lock().map_err(|_| {
            Error::Refused("the peer socket was poisoned by a panicking rank".to_owned())
        })?;
        let mut frame = Vec::with_capacity(5 + name.len() + payload.len());
        frame.push(kind);
        frame.extend_from_slice(&(name.len() as u32).to_le_bytes());
        frame.extend_from_slice(name.as_bytes());
        frame.extend_from_slice(payload);
        let io = |e: std::io::Error, what: &str| {
            Error::Refused(format!("rank {}: {what} `{name}`: {e}", self.rank))
        };
        s.write_all(&frame).map_err(|e| io(e, "sending"))?;
        s.flush().map_err(|e| io(e, "flushing"))?;

        let mut head = [0u8; 5];
        s.read_exact(&mut head).map_err(|e| io(e, "waiting for"))?;
        let got_len = u32::from_le_bytes([head[1], head[2], head[3], head[4]]) as usize;
        let mut got_name = vec![0u8; got_len];
        s.read_exact(&mut got_name).map_err(|e| io(e, "reading"))?;
        let mut body = vec![0u8; want];
        s.read_exact(&mut body).map_err(|e| io(e, "reading"))?;
        if head[0] != kind || got_name != name.as_bytes() {
            return Err(Error::Refused(format!(
                "rank {}: asked the peer for `{name}` (kind {kind}) and got `{}` (kind {}); the \
                 two ranks are exchanging in different orders",
                self.rank,
                String::from_utf8_lossy(&got_name),
                head[0]
            )));
        }
        Ok(body)
    }
}

/// The frame kind [`IpcExchange::xchg_bytes`] uses. 1 and 2 are the handle and digest swaps and 0
/// is `swap_bytes`'s; a kind byte a peer does not expect is refused by name rather than read as a
/// length.
const KIND_XCHG: u8 = 3;

impl IpcExchange for TcpExchange {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn peer_rank(&self) -> u32 {
        1 - self.rank
    }

    fn swap_handle(&self, name: &str, mine: IpcMemHandle) -> Result<IpcMemHandle> {
        let bytes: Vec<u8> = mine.reserved.iter().map(|c| *c as u8).collect();
        let got = self.exchange(1, name, &bytes, saber_hip::IPC_HANDLE_BYTES)?;
        let mut out = IpcMemHandle::default();
        for (slot, b) in out.reserved.iter_mut().zip(got) {
            *slot = b as std::ffi::c_char;
        }
        Ok(out)
    }

    fn swap_digest(&self, name: &str, mine: u64) -> Result<u64> {
        let got = self.exchange(2, name, &mine.to_le_bytes(), 8)?;
        Ok(u64::from_le_bytes([
            got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7],
        ]))
    }

    fn swap_bytes(&self, name: &str, mine: &[u8]) -> Result<Vec<u8>> {
        // The length goes over first, so a rank that disagrees is told which one it is instead of
        // reading half of its peer's next frame.
        let theirs = self.swap_digest(&format!("{name}.len"), mine.len() as u64)?;
        if theirs != mine.len() as u64 {
            return Err(Error::Refused(format!(
                "rank {}: `{name}` is {} bytes here and {theirs} on the peer; swap_bytes is for \
                 symmetric shards",
                self.rank,
                mine.len()
            )));
        }
        // RANK-ORDERED, not symmetric. `exchange` writes the whole frame before reading one, which
        // deadlocks as soon as the payload exceeds the socket buffers: both ranks block in
        // `write_all` and neither ever reads. (20 MiB of embedding partials did exactly that on
        // 2026-09-19 — `EAGAIN` on one side, `ECONNRESET` on the other.) So rank 0 sends then
        // receives and rank 1 receives then sends, which streams cleanly at any size.
        let mut s = self.stream.lock().map_err(|_| {
            Error::Refused("the peer socket was poisoned by a panicking rank".to_owned())
        })?;
        let io = |e: std::io::Error, what: &str| {
            Error::Refused(format!("rank {}: {what} `{name}` bytes: {e}", self.rank))
        };
        let mut got = vec![0u8; mine.len()];
        if self.rank == 0 {
            s.write_all(mine).map_err(|e| io(e, "sending"))?;
            s.flush().map_err(|e| io(e, "flushing"))?;
            s.read_exact(&mut got).map_err(|e| io(e, "reading"))?;
        } else {
            s.read_exact(&mut got).map_err(|e| io(e, "reading"))?;
            s.write_all(mine).map_err(|e| io(e, "sending"))?;
            s.flush().map_err(|e| io(e, "flushing"))?;
        }
        Ok(got)
    }

    fn xchg_bytes(&self, name: &str, mine: &[u8]) -> Result<Vec<u8>> {
        // THE FRAME CARRIES ITS OWN LENGTH, so there is no length round trip and no requirement
        // that the two sides be the same width. RANK-ORDERED for the same reason `swap_bytes` is:
        // a symmetric `write_all` of a payload larger than the socket buffers deadlocks, both
        // ranks blocked in the write and neither reading (20 MiB of embedding partials did exactly
        // that on 2026-09-19).
        let mut s = self.stream.lock().map_err(|_| {
            Error::Refused("the peer socket was poisoned by a panicking rank".to_owned())
        })?;
        let io = |e: std::io::Error, what: &str| {
            Error::Refused(format!("rank {}: {what} `{name}` bytes: {e}", self.rank))
        };
        let mut frame = Vec::with_capacity(9 + name.len() + mine.len());
        frame.push(KIND_XCHG);
        frame.extend_from_slice(&(name.len() as u32).to_le_bytes());
        frame.extend_from_slice(name.as_bytes());
        frame.extend_from_slice(&(mine.len() as u32).to_le_bytes());
        frame.extend_from_slice(mine);
        let read_theirs = |s: &mut std::net::TcpStream| -> Result<Vec<u8>> {
            let mut head = [0u8; 5];
            s.read_exact(&mut head).map_err(|e| io(e, "waiting for"))?;
            let name_len = u32::from_le_bytes([head[1], head[2], head[3], head[4]]) as usize;
            let mut got_name = vec![0u8; name_len];
            s.read_exact(&mut got_name).map_err(|e| io(e, "reading"))?;
            let mut len = [0u8; 4];
            s.read_exact(&mut len).map_err(|e| io(e, "reading"))?;
            if head[0] != KIND_XCHG || got_name != name.as_bytes() {
                return Err(Error::Refused(format!(
                    "rank {}: asked the peer for `{name}` (kind {KIND_XCHG}) and got `{}` (kind \
                     {}); the two ranks are exchanging in different orders",
                    self.rank,
                    String::from_utf8_lossy(&got_name),
                    head[0]
                )));
            }
            let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
            s.read_exact(&mut body).map_err(|e| io(e, "reading"))?;
            Ok(body)
        };
        if self.rank == 0 {
            s.write_all(&frame).map_err(|e| io(e, "sending"))?;
            s.flush().map_err(|e| io(e, "flushing"))?;
            read_theirs(&mut s)
        } else {
            let got = read_theirs(&mut s)?;
            s.write_all(&frame).map_err(|e| io(e, "sending"))?;
            s.flush().map_err(|e| io(e, "flushing"))?;
            Ok(got)
        }
    }

    fn rendezvous(&self, tag: &str) -> Result<()> {
        // A swap IS a rendezvous: neither side returns until both have sent.
        self.swap_digest(tag, 0).map(|_| ())
    }
}

// ------------------------------------------------------------------------------------------- //
// Protocol digest                                                                               //
// ------------------------------------------------------------------------------------------- //

/// A stable digest of the things both ranks must agree on before a collective launch.
///
/// FNV-1a over a canonical byte encoding: small, dependency-free, and deterministic across runs
/// and machines, which a `DefaultHasher` is explicitly not.
#[derive(Debug, Clone, Default)]
pub struct ProtocolDigest {
    state: u64,
}

impl ProtocolDigest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;

    /// An empty digest.
    #[must_use]
    pub fn new() -> Self {
        ProtocolDigest {
            state: Self::OFFSET,
        }
    }

    /// Fold in raw bytes.
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        for byte in b {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
        self
    }

    /// Fold in a string, length-prefixed so `("ab", "c")` and `("a", "bc")` differ.
    pub fn text(&mut self, s: &str) -> &mut Self {
        self.bytes(&(s.len() as u64).to_le_bytes());
        self.bytes(s.as_bytes())
    }

    /// Fold in a number.
    pub fn num(&mut self, v: u64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    /// The digest so far.
    #[must_use]
    pub fn finish(&self) -> u64 {
        self.state
    }
}

/// The reset epoch of every IPC region this rank owns, by name.
///
/// `slot = seq & 1` picks which half of the double-buffered scratch both ranks write, so the two
/// ranks must have zeroed their sequence counters the same number of times. Counting the resets is
/// how that becomes checkable without reading device memory.
pub type ResetEpochs = BTreeMap<String, u64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_is_deterministic_and_order_sensitive() {
        let mut a = ProtocolDigest::new();
        a.text("kernel").num(4).num(256);
        let mut b = ProtocolDigest::new();
        b.text("kernel").num(4).num(256);
        assert_eq!(a.finish(), b.finish());

        let mut c = ProtocolDigest::new();
        c.text("kernel").num(256).num(4);
        assert_ne!(a.finish(), c.finish(), "order must matter");

        // Length prefixing: a split that concatenates to the same bytes must still differ.
        let mut d = ProtocolDigest::new();
        d.text("ker").text("nel");
        let mut e = ProtocolDigest::new();
        e.text("kernel");
        assert_ne!(d.finish(), e.finish());
    }

    #[test]
    fn two_threads_swap_handles_and_digests() {
        let (a, b) = ThreadPairExchange::pair();
        let (a, b) = (Arc::new(a), Arc::new(b));
        let p0 = RankPeer::new(a);
        let peer1 = Arc::clone(&b);
        let t = std::thread::spawn(move || {
            let p1 = RankPeer::new(peer1);
            p1.rendezvous("start").unwrap();
            let got = p1
                .swap_handle("ar_scratch", IpcMemHandle::default())
                .unwrap();
            p1.agree("launch", 0xabcd).unwrap();
            got
        });
        p0.rendezvous("start").unwrap();
        p0.swap_handle("ar_scratch", IpcMemHandle::default())
            .unwrap();
        p0.agree("launch", 0xabcd).unwrap();
        t.join().unwrap();
    }

    #[test]
    fn negative_control_a_disagreement_is_refused_and_names_both_sides() {
        let (a, b) = ThreadPairExchange::pair();
        let (a, b) = (Arc::new(a), Arc::new(b));
        let p0 = RankPeer::new(a);
        let peer1 = Arc::clone(&b);
        let t = std::thread::spawn(move || {
            let p1 = RankPeer::new(peer1);
            // Rank 1 zeroed its sequence counters once more than rank 0 did, so `slot = seq & 1`
            // would pick the other half of the scratch. This is the bug that hangs the GPU.
            p1.agree("launch", 0x1111).err().map(|e| e.to_string())
        });
        let mine = p0
            .agree("launch", 0x2222)
            .expect_err("rank 0 must refuse too");
        let theirs = t.join().unwrap().expect("rank 1 must refuse");
        // BOTH ranks refuse, and each message names both digests, so a reader can tell which
        // side drifted without correlating two logs.
        for text in [mine.to_string(), theirs] {
            assert!(text.contains("disagree"), "{text}");
            assert!(text.contains("1111"), "{text}");
            assert!(text.contains("2222"), "{text}");
            assert!(text.contains("Nothing was launched"), "{text}");
        }
    }

    /// **THE REAL TRANSPORT, and the frame that is not symmetric.**
    ///
    /// `ThreadPairExchange` carries a `Vec` and cannot tell an asymmetric frame from a symmetric
    /// one; a loopback socket can, and it is what the engine runs on. This drives a real
    /// `TcpExchange` pair through `xchg_bytes` with **different widths on the two sides** — a
    /// 4 KiB step frame one way and an 8-byte ack the other, which is the engine's own shape — and
    /// then does it again in the other direction, because the seam's output frame goes
    /// follower-to-leader and the two rank orderings are two code paths.
    ///
    /// The CONTROL is `swap_bytes` on the same pair: it must REFUSE the same two widths, which is
    /// the property this new call is defined by.
    #[test]
    fn a_tcp_pair_exchanges_frames_of_two_different_widths() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let port = listener.local_addr().expect("the port").port();
        let t = std::thread::spawn(move || {
            let r1 = TcpExchange::connect(port, Duration::from_secs(5)).expect("rank 1 connects");
            // Leader -> follower: a wide frame out, an ack back.
            let got = r1
                .xchg_bytes("engine.step", &[7u8; 8])
                .expect("rank 1 step");
            assert_eq!(got.len(), 4096, "rank 1 read the leader's whole frame");
            assert!(got.iter().all(|b| *b == 3), "and its contents");
            // Follower -> leader: the seam's outputs, answered with an ack.
            let out = vec![9u8; 512];
            let ack = r1.xchg_bytes("engine.seam.out", &out).expect("rank 1 seam");
            assert_eq!(ack, vec![1u8; 8], "rank 1 got the leader's ack");
            // And the control, on the same socket: two widths through `swap_bytes` is a refusal.
            let _ = r1.swap_bytes("engine.control", &[0u8; 8]);
        });
        let r0 = TcpExchange::accept(&listener, Duration::from_secs(5)).expect("rank 0 accepts");
        let ack = r0
            .xchg_bytes("engine.step", &[3u8; 4096])
            .expect("rank 0 step");
        assert_eq!(ack, vec![7u8; 8], "rank 0 got the follower's ack");
        let got = r0
            .xchg_bytes("engine.seam.out", &[1u8; 8])
            .expect("rank 0 seam");
        assert_eq!(got.len(), 512, "rank 0 read the follower's whole frame");
        let e = r0
            .swap_bytes("engine.control", &[0u8; 4096])
            .expect_err("swap_bytes accepted two different widths");
        assert!(
            e.to_string().contains("symmetric shards"),
            "the control must name the symmetry law: {e}"
        );
        t.join().expect("rank 1");
    }

    #[test]
    fn a_peer_that_never_answers_times_out_rather_than_blocking_forever() {
        let short = Duration::from_millis(150);
        let (a, _b) = ThreadPairExchange::pair_with_timeout(short);
        // `_b` is alive but nobody is driving it, which is what a wedged rank looks like. The wait
        // is bounded, so this returns; a `recv()` without a deadline would hang the test suite.
        let p0 = RankPeer::new(Arc::new(a));
        let started = std::time::Instant::now();
        let err = p0
            .swap_handle("ar_scratch", IpcMemHandle::default())
            .expect_err("no peer is answering");
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(started.elapsed() >= short);
    }
}
