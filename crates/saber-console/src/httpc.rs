//! A minimal HTTP/1.1 GET for the engine's loopback endpoints (`/metrics`, `/health`,
//! `/v1/models`). The engine is always on this host; a client crate would be more code than this.

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// GET `http://127.0.0.1:{port}{path}` -> (status, body). Errors on connect/timeout/parse.
pub async fn get(port: u16, path: &str, timeout: Duration) -> anyhow::Result<(u16, String)> {
    get_auth(port, path, timeout, None).await
}

/// [`get`] with `Authorization: Bearer <bearer>` when the engine's API keys are on (
/// the console's polls keep working with engine auth on).
pub async fn get_auth(
    port: u16,
    path: &str,
    timeout: Duration,
    bearer: Option<&str>,
) -> anyhow::Result<(u16, String)> {
    let auth = bearer
        .filter(|b| !b.is_empty() && !b.contains(['\r', '\n']))
        .map_or(String::new(), |b| format!("Authorization: Bearer {b}\r\n"));
    tokio::time::timeout(timeout, async {
        let mut s = TcpStream::connect(("127.0.0.1", port)).await?;
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUser-Agent: saber-console\r\n\
             {auth}Accept: */*\r\nConnection: close\r\n\r\n"
        );
        s.write_all(req.as_bytes()).await?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).await?;
        parse_response(&buf)
    })
    .await
    .map_err(|_| anyhow::anyhow!("GET {path}: timed out"))?
}

fn parse_response(buf: &[u8]) -> anyhow::Result<(u16, String)> {
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("no header terminator"))?;
    let head = std::str::from_utf8(&buf[..split])?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow::anyhow!("bad status line"))?;
    let body = &buf[split + 4..];
    let chunked = head.lines().any(|l| {
        let l = l.to_ascii_lowercase();
        l.starts_with("transfer-encoding:") && l.contains("chunked")
    });
    let body = if chunked {
        dechunk(body)?
    } else {
        body.to_vec()
    };
    Ok((status, String::from_utf8_lossy(&body).into_owned()))
}

fn dechunk(mut b: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = b
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| anyhow::anyhow!("bad chunk"))?;
        let size_str = std::str::from_utf8(&b[..eol])?;
        let size = usize::from_str_radix(size_str.split(';').next().unwrap_or("").trim(), 16)?;
        b = &b[eol + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size {
            anyhow::bail!("short chunk");
        }
        out.extend_from_slice(&b[..size]);
        b = b.get(size + 2..).unwrap_or(&[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_chunked() {
        let r = b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok";
        assert_eq!(parse_response(r).unwrap(), (200, "ok".into()));
        let r =
            b"HTTP/1.1 503 X\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n";
        assert_eq!(parse_response(r).unwrap(), (503, "abcde".into()));
    }
}
