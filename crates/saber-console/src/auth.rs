//! The console's credentials: the admin token and the `/v1` proxy keys at rest, the
//! constant-time checks, and the failed-auth rate limit.
//!
//! - A secret is stored as a SALTED SHA-256 (`sha256$<salt hex>$<digest hex>`), never as text:
//!   [`Config::save`](crate::config::Config::save) hashes any plaintext it is handed (a legacy
//!   file, a wizard step, `--gen-admin-token`) and the console shows a generated secret ONCE.
//!   The secrets are random 192-bit strings, so a fast hash is enough (no password stretching:
//!   there is nothing to guess).
//! - Every check compares in constant time in the digest's bytes, and a presented value is
//!   checked against EVERY candidate (no early exit that would time which key matched).
//! - [`Limiter`]: a client address that presents a WRONG token or key [`Limiter::MAX_FAILS`]
//!   times inside [`Limiter::WINDOW`] is refused (429) for [`Limiter::LOCKOUT`], even with a
//!   right token. A call with no credential at all (a signed-out page polling) is refused (401)
//!   but not counted.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// The prefix of a stored hash.
pub const HASH_SCHEME: &str = "sha256";
/// What the API shows in place of a stored secret (and what a client sends back to keep it).
pub const REDACTED: &str = "(set)";

/// Constant-time equality (length leaks; the stored side is a fixed-length digest).
#[must_use]
pub fn eq_ct(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn digest(salt: &[u8], secret: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"saber-console-v1\0");
    h.update(salt);
    h.update(secret);
    h.finalize().into()
}

/// Random bytes from the kernel.
pub fn random_bytes(n: usize) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    let mut b = vec![0u8; n];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(b)
}

/// `sha256$<salt>$<digest>` for `secret` with a fresh 16-byte salt.
pub fn hash_secret(secret: &str) -> anyhow::Result<String> {
    let salt = random_bytes(16)?;
    Ok(hash_with_salt(secret, &salt))
}

#[must_use]
pub fn hash_with_salt(secret: &str, salt: &[u8]) -> String {
    format!(
        "{HASH_SCHEME}${}${}",
        hex::encode(salt),
        hex::encode(digest(salt, secret.as_bytes()))
    )
}

/// True when `s` is a well-formed stored hash.
#[must_use]
pub fn is_hash(s: &str) -> bool {
    parse(s).is_some()
}

fn parse(s: &str) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut it = s.split('$');
    if it.next()? != HASH_SCHEME {
        return None;
    }
    let salt = hex::decode(it.next()?).ok()?;
    let d = hex::decode(it.next()?).ok()?;
    (it.next().is_none() && !salt.is_empty() && d.len() == 32).then_some((salt, d))
}

/// Check `presented` against a stored hash in constant time (a malformed hash never matches).
#[must_use]
pub fn verify(stored: &str, presented: &str) -> bool {
    let Some((salt, want)) = parse(stored) else {
        return false;
    };
    eq_ct(&digest(&salt, presented.as_bytes()), &want)
}

/// The console's admin credential at run time.
#[derive(Clone, PartialEq, Eq)]
pub enum Admin {
    /// No token: the console is open (loopback binds only; [`crate::check_bind`]).
    None,
    /// `--admin-token` on the command line (kept in memory only).
    Flag(String),
    /// The config's stored hash (`console.admin_token_hash`).
    Hash(String),
}

impl std::fmt::Debug for Admin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // never print a secret
        f.write_str(match self {
            Admin::None => "Admin::None",
            Admin::Flag(_) => "Admin::Flag(..)",
            Admin::Hash(_) => "Admin::Hash(..)",
        })
    }
}

impl Admin {
    #[must_use]
    pub fn is_set(&self) -> bool {
        !matches!(self, Admin::None)
    }

    /// `presented` is the admin token.
    #[must_use]
    pub fn check(&self, presented: &str) -> bool {
        match self {
            Admin::None => false,
            Admin::Flag(t) => {
                // hash both sides so the compare is over fixed-length digests
                !presented.is_empty()
                    && eq_ct(
                        &digest(b"", t.as_bytes()),
                        &digest(b"", presented.as_bytes()),
                    )
            }
            Admin::Hash(h) => !presented.is_empty() && verify(h, presented),
        }
    }

    /// Where it came from (the Security page shows it).
    #[must_use]
    pub fn source(&self) -> &'static str {
        match self {
            Admin::None => "none",
            Admin::Flag(_) => "flag",
            Admin::Hash(_) => "config",
        }
    }
}

/// (window start, failures in it, locked until).
type FailState = (Instant, u32, Option<Instant>);

/// The failed-auth limiter, per client address.
pub struct Limiter {
    fails: Mutex<HashMap<IpAddr, FailState>>,
    window: Duration,
    lockout: Duration,
    max_fails: u32,
}

impl Default for Limiter {
    fn default() -> Self {
        Limiter::new(Self::MAX_FAILS, Self::WINDOW, Self::LOCKOUT)
    }
}

impl Limiter {
    pub const MAX_FAILS: u32 = 10;
    pub const WINDOW: Duration = Duration::from_secs(60);
    pub const LOCKOUT: Duration = Duration::from_secs(60);

    #[must_use]
    pub fn new(max_fails: u32, window: Duration, lockout: Duration) -> Limiter {
        Limiter {
            fails: Mutex::new(HashMap::new()),
            window,
            lockout,
            max_fails,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<IpAddr, FailState>> {
        self.fails.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `Some(seconds left)` while `ip` is locked out.
    #[must_use]
    pub fn blocked(&self, ip: IpAddr) -> Option<u64> {
        let now = Instant::now();
        let mut g = self.lock();
        let e = g.get_mut(&ip)?;
        match e.2 {
            Some(until) if until > now => Some((until - now).as_secs().max(1)),
            Some(_) => {
                g.remove(&ip);
                None
            }
            None => None,
        }
    }

    /// Count a failure; true when it started a lockout.
    pub fn fail(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let mut g = self.lock();
        if g.len() > 10_000 {
            // a flood of addresses: forget the stale ones
            let w = self.window;
            g.retain(|_, e| e.2.is_some_and(|u| u > now) || now.duration_since(e.0) < w);
        }
        let e = g.entry(ip).or_insert((now, 0, None));
        if now.duration_since(e.0) > self.window {
            *e = (now, 0, None);
        }
        e.1 += 1;
        if e.1 >= self.max_fails && e.2.is_none() {
            e.2 = Some(now + self.lockout);
            return true;
        }
        false
    }

    /// A success clears the address's count (not a running lockout).
    pub fn ok(&self, ip: IpAddr) {
        let mut g = self.lock();
        if g.get(&ip).is_some_and(|e| e.2.is_none()) {
            g.remove(&ip);
        }
    }
}

/// The remember-me cookie: a signed session the browser keeps after one sign-in.
pub const SESSION_COOKIE: &str = "saber_session";
/// How long a session lasts.
pub const SESSION_TTL: Duration = Duration::from_secs(30 * 24 * 3600);

/// The run-time auth state the HTTP plane shares: the admin credential (swapped in place when
/// the token is rotated), the limiter and the session secret.
///
/// A session is `<expiry ms>.<mac>`, the mac a SHA-256 over a per-install secret
/// (`<data dir>/session.key`, 0600, never leaves the machine), the admin credential's
/// fingerprint and the expiry. Rotating or removing the admin token changes the fingerprint, so
/// every session ends at once; the secret keeps them valid across a console restart.
pub struct Auth {
    admin: RwLock<Admin>,
    pub limiter: Limiter,
    session_secret: [u8; 32],
}

impl Auth {
    #[must_use]
    pub fn new(admin: Admin) -> Auth {
        // a per-process secret until `with_session_secret` hands over the persisted one
        let mut secret = [0u8; 32];
        if let Ok(b) = random_bytes(32) {
            secret.copy_from_slice(&b);
        }
        Auth {
            admin: RwLock::new(admin),
            limiter: Limiter::default(),
            session_secret: secret,
        }
    }

    /// Use `secret` (the persisted per-install one) to sign sessions.
    #[must_use]
    pub fn with_session_secret(mut self, secret: [u8; 32]) -> Auth {
        self.session_secret = secret;
        self
    }

    fn session_mac(&self, exp_ms: u64) -> [u8; 32] {
        let fp: Vec<u8> = match self.admin() {
            Admin::None => Vec::new(),
            Admin::Flag(t) => digest(b"fp", t.as_bytes()).to_vec(),
            Admin::Hash(h) => digest(b"fp", h.as_bytes()).to_vec(),
        };
        let mut h = Sha256::new();
        h.update(b"saber-session-v1\0");
        h.update(self.session_secret);
        h.update(&fp);
        h.update(exp_ms.to_be_bytes());
        h.finalize().into()
    }

    /// A signed session valid for [`SESSION_TTL`] from `now_ms`; `None` on an open console.
    #[must_use]
    pub fn session_mint(&self, now_ms: u64) -> Option<String> {
        if !self.required() {
            return None;
        }
        let exp = now_ms + SESSION_TTL.as_millis() as u64;
        Some(format!("{exp}.{}", hex::encode(self.session_mac(exp))))
    }

    /// True when `value` is an unexpired session signed for the current admin credential.
    #[must_use]
    pub fn session_check(&self, value: &str, now_ms: u64) -> bool {
        if !self.required() {
            return false;
        }
        let Some((exp, mac)) = value.split_once('.') else {
            return false;
        };
        let (Ok(exp), Ok(mac)) = (exp.parse::<u64>(), hex::decode(mac)) else {
            return false;
        };
        exp > now_ms && eq_ct(&self.session_mac(exp), &mac)
    }

    /// No token (a loopback console).
    #[must_use]
    pub fn open() -> Auth {
        Auth::new(Admin::None)
    }

    #[must_use]
    pub fn with_limiter(admin: Admin, limiter: Limiter) -> Auth {
        Auth {
            limiter,
            ..Auth::new(admin)
        }
    }

    #[must_use]
    pub fn admin(&self) -> Admin {
        self.admin.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn set_admin(&self, a: Admin) {
        *self.admin.write().unwrap_or_else(|e| e.into_inner()) = a;
    }

    #[must_use]
    pub fn required(&self) -> bool {
        self.admin().is_set()
    }

    #[must_use]
    pub fn check_admin(&self, presented: &str) -> bool {
        self.admin().check(presented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_are_signed_expire_and_die_with_the_token() {
        let a = Auth::new(Admin::Flag("tok-tok-tok-tok-tok".into())).with_session_secret([7; 32]);
        let now = 1_000_000;
        let v = a.session_mint(now).unwrap();
        assert!(a.session_check(&v, now + 1));
        assert!(!a.session_check(&v, now + SESSION_TTL.as_millis() as u64 + 1), "expired");
        let (exp, mac) = v.split_once('.').unwrap();
        let later = exp.parse::<u64>().unwrap() + 1;
        assert!(!a.session_check(&format!("{later}.{mac}"), now), "expiry is signed");
        assert!(!a.session_check("garbage", now));
        // another install's secret, or a rotated token, does not honour it
        let b = Auth::new(Admin::Flag("tok-tok-tok-tok-tok".into())).with_session_secret([8; 32]);
        assert!(!b.session_check(&v, now + 1));
        a.set_admin(Admin::Flag("a-different-token-0000".into()));
        assert!(!a.session_check(&v, now + 1), "a new token ends every session");
        // an open console has no sessions
        assert!(Auth::open().session_mint(now).is_none());
        assert!(!Auth::open().session_check(&v, now));
    }

    #[test]
    fn hashes_verify_and_never_hold_the_secret() {
        let h = hash_secret("sa-admin-0123456789abcdef").unwrap();
        assert!(is_hash(&h), "{h}");
        assert!(!h.contains("0123456789abcdef"));
        assert!(verify(&h, "sa-admin-0123456789abcdef"));
        assert!(!verify(&h, "sa-admin-0123456789abcdeF"));
        assert!(!verify(&h, ""));
        // two hashes of one secret differ (salted) and both verify
        let h2 = hash_secret("sa-admin-0123456789abcdef").unwrap();
        assert_ne!(h, h2);
        assert!(verify(&h2, "sa-admin-0123456789abcdef"));
        // malformed hashes never match
        for bad in [
            "",
            "sha256$$",
            "md5$00$00",
            "sha256$zz$00",
            "sha256$00$0011",
        ] {
            assert!(!is_hash(bad), "{bad}");
            assert!(!verify(bad, ""), "{bad}");
        }
        assert_eq!(
            hash_with_salt("x", &[1, 2]),
            hash_with_salt("x", &[1, 2]),
            "deterministic for a salt"
        );
    }

    #[test]
    fn admin_modes() {
        assert!(!Admin::None.check("x"));
        let f = Admin::Flag("flag-token-0123456".into());
        assert!(f.check("flag-token-0123456") && !f.check("flag-token-012345") && !f.check(""));
        let h = Admin::Hash(hash_secret("cfg-token-0123456").unwrap());
        assert!(h.check("cfg-token-0123456") && !h.check("flag-token-0123456"));
        assert!(!format!("{f:?}{h:?}").contains("token"), "Debug hides it");
    }

    #[test]
    fn the_limiter_locks_after_max_fails_and_expires() {
        let ip: IpAddr = "10.0.0.7".parse().unwrap();
        let other: IpAddr = "10.0.0.8".parse().unwrap();
        let l = Limiter::new(3, Duration::from_secs(60), Duration::from_millis(150));
        assert!(!l.fail(ip) && !l.fail(ip));
        assert_eq!(l.blocked(ip), None);
        assert!(l.fail(ip), "the third failure locks");
        assert!(l.blocked(ip).is_some());
        assert_eq!(l.blocked(other), None, "per address");
        l.ok(ip);
        assert!(l.blocked(ip).is_some(), "a success does not lift a lockout");
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(l.blocked(ip), None, "the lockout expires");
        assert!(!l.fail(ip), "and the count starts again");
    }
}
