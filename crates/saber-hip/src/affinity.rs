//! Host-thread placement (FN57): which CPUs a thread may run on, and its kernel thread id.
//!
//! Lives here because it is the crate that may call `extern "C"` (structural rule 2); the engine
//! crates name CPUs, never the syscall. A mask is a decision the caller made from a measured law
//! (the Flash-Next worker's `pin_law`), not an environment variable.

use crate::error::{Error, Result};

/// The calling thread's kernel thread id (`gettid`).
#[must_use]
pub fn current_tid() -> i32 {
    // SAFETY: `gettid` takes no arguments and cannot fail.
    unsafe { libc::gettid() }
}

/// Pin the CALLING thread to `cpus` (`sched_setaffinity(0, ..)`). Threads it creates afterwards
/// inherit the mask.
///
/// # Errors
/// An empty set, a CPU id past `CPU_SETSIZE`, or the kernel refusing (a CPU outside the process's
/// allowed set, e.g. a container's cpuset).
pub fn pin_current_thread(cpus: &[usize]) -> Result<()> {
    if cpus.is_empty() {
        return Err(Error::Invalid("pin: an empty CPU set".into()));
    }
    // SAFETY: a zeroed cpu_set_t is the empty set; CPU_SET writes inside it for ids below
    // CPU_SETSIZE, which is checked first.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cpus {
        if c >= libc::CPU_SETSIZE as usize {
            return Err(Error::Invalid(format!("pin: cpu {c} is past CPU_SETSIZE")));
        }
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // SAFETY: pid 0 = the calling thread; `set` is a valid cpu_set_t of the size passed.
    let rc = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if rc != 0 {
        return Err(Error::Invalid(format!(
            "pin: sched_setaffinity({cpus:?}) failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

/// Pin ANOTHER thread of this process (kernel thread id `tid`) to `cpus`.
///
/// # Errors
/// As [`pin_current_thread`].
pub fn pin_thread(tid: i32, cpus: &[usize]) -> Result<()> {
    if cpus.is_empty() {
        return Err(Error::Invalid("pin: an empty CPU set".into()));
    }
    // SAFETY: as in `pin_current_thread`.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cpus {
        if c >= libc::CPU_SETSIZE as usize {
            return Err(Error::Invalid(format!("pin: cpu {c} is past CPU_SETSIZE")));
        }
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // SAFETY: `tid` names a thread of this process (or the call fails with ESRCH).
    let rc = unsafe { libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if rc != 0 {
        return Err(Error::Invalid(format!(
            "pin: sched_setaffinity(tid {tid}, {cpus:?}) failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

/// The kernel thread ids of this process's threads whose name (`comm`) is `name` -- e.g. the HIP
/// runtime's own threads, which inherit `saber-hip` from the thread that created them.
#[must_use]
pub fn threads_named(name: &str) -> Vec<i32> {
    let Ok(rd) = std::fs::read_dir("/proc/self/task") else {
        return Vec::new();
    };
    let mut out: Vec<i32> = rd
        .filter_map(|e| {
            let e = e.ok()?;
            let tid: i32 = e.file_name().to_str()?.parse().ok()?;
            let comm = std::fs::read_to_string(e.path().join("comm")).ok()?;
            (comm.trim_end() == name).then_some(tid)
        })
        .collect();
    out.sort_unstable();
    out
}

/// The calling thread's allowed CPUs (`sched_getaffinity(0, ..)`).
///
/// # Errors
/// The kernel refusing.
pub fn current_cpus() -> Result<Vec<usize>> {
    // SAFETY: a zeroed cpu_set_t of the size passed; the kernel fills it.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let rc =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if rc != 0 {
        return Err(Error::Invalid(format!(
            "sched_getaffinity failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&c| unsafe { libc::CPU_ISSET(c, &set) })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thread pinned to one of its allowed CPUs reports exactly that CPU, and restoring the old
    /// set restores it; an empty set is refused before the syscall.
    #[test]
    fn a_pinned_thread_reports_its_mask() {
        let all = current_cpus().expect("the test thread has a mask");
        assert!(!all.is_empty());
        let one = *all.last().expect("one cpu");
        std::thread::spawn(move || {
            pin_current_thread(&[one]).expect("pin to an allowed cpu");
            assert_eq!(current_cpus().unwrap(), vec![one]);
            pin_current_thread(&all).expect("restore");
            assert_eq!(current_cpus().unwrap(), all);
        })
        .join()
        .unwrap();
        assert!(pin_current_thread(&[]).is_err());
        assert!(current_tid() > 0);
    }
}
