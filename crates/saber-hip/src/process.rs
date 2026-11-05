//! FN73: the fatal teardown's hard exit.

/// `_exit(code)`: leave NOW without the C runtime's atexit handlers. The HIP runtime registers
/// one, and running it under a live `HipThread` turns any exit into a SIGSEGV (139, 09-18);
/// the fatal teardown (`saber_serve::fatal`) must leave with its own code. Flush what must be
/// read before calling it (stdio buffers are not flushed).
pub fn hard_exit(code: i32) -> ! {
    // SAFETY: `_exit` takes a plain int, never returns and touches no Rust state.
    unsafe { libc::_exit(code) }
}
