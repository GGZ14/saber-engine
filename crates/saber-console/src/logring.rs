//! The engine's stdout/stderr as a bounded ring of lines (the boot output is mostly `println!`),
//! with a broadcast for the live tail.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Clone, Debug, Serialize)]
pub struct LogLine {
    /// Monotonic sequence number (a client resumes with `since`).
    pub seq: u64,
    /// Unix milliseconds the console read the line.
    pub t_ms: u64,
    /// `out`, `err` or `console` (the supervisor's own notes).
    pub stream: &'static str,
    pub text: String,
}

pub struct LogRing {
    inner: Mutex<(u64, VecDeque<LogLine>)>,
    cap: usize,
    tx: broadcast::Sender<LogLine>,
}

impl LogRing {
    #[must_use]
    pub fn new(cap: usize) -> LogRing {
        let (tx, _) = broadcast::channel(1024);
        LogRing {
            inner: Mutex::new((0, VecDeque::with_capacity(cap.min(4096)))),
            cap: cap.max(1),
            tx,
        }
    }

    pub fn push(&self, stream: &'static str, text: impl Into<String>) {
        let line = {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.0 += 1;
            let line = LogLine {
                seq: g.0,
                t_ms: crate::now_ms(),
                stream,
                text: text.into(),
            };
            if g.1.len() == self.cap {
                g.1.pop_front();
            }
            g.1.push_back(line.clone());
            line
        };
        let _ = self.tx.send(line);
    }

    /// Lines after `since`, at most `limit` of the newest.
    #[must_use]
    pub fn since(&self, since: u64, limit: usize) -> Vec<LogLine> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let v: Vec<LogLine> = g.1.iter().filter(|l| l.seq > since).cloned().collect();
        let skip = v.len().saturating_sub(limit);
        v.into_iter().skip(skip).collect()
    }

    /// The last line matching `pred`, newest first.
    pub fn last_matching(&self, pred: impl Fn(&str) -> bool) -> Option<String> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.1.iter()
            .rev()
            .find(|l| pred(&l.text))
            .map(|l| l.text.clone())
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<LogLine> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_bounded_and_resumable() {
        let r = LogRing::new(3);
        for i in 0..5 {
            r.push("out", format!("l{i}"));
        }
        let all = r.since(0, 100);
        assert_eq!(
            all.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(),
            ["l2", "l3", "l4"]
        );
        assert_eq!(r.since(4, 100).len(), 1);
        assert_eq!(r.since(0, 1)[0].text, "l4");
        assert_eq!(
            r.last_matching(|t| t.starts_with("l3")).as_deref(),
            Some("l3")
        );
    }
}
