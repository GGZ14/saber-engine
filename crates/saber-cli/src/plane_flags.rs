//! The serving plane's operator flags, shared by `saber serve` (mock and 27B)
//! and `saber engine --fn-serve` (Flash-Next). They configure the plane only -- no engine or
//! step-path behaviour -- and are installed once into `saber_serve::state` before a server is
//! built (the engine reads no env vars: keys come from these flags or the file they name).

use std::path::PathBuf;

use anyhow::Context as _;

#[derive(clap::Args, Debug, Clone, Default)]
pub struct PlaneFlags {
    /// Require `Authorization: Bearer <KEY>` on the /v1 routes (repeatable, `LABEL=KEY`; the
    /// label is recorded in `/v1/requests`). OFF when no key is given (the default). Prefer
    /// `--api-key-file`: a key on the command line is visible in `ps`.
    #[arg(long = "api-key", value_name = "LABEL=KEY")]
    pub api_key: Vec<String>,

    /// API keys from a file, one `LABEL=KEY` (or `LABEL KEY`) per line, `#` comments.
    #[arg(long)]
    pub api_key_file: Option<PathBuf>,

    /// With API keys on, also require a key on `/health` and `/metrics` (open by default so
    /// probes and scrapers keep working).
    #[arg(long)]
    pub api_auth_all: bool,

    /// Recent finished requests kept for `GET /v1/requests` (counts and timings only, never a
    /// prompt or a reply).
    #[arg(long, default_value_t = saber_serve::requests::DEFAULT_RING)]
    pub request_ring: usize,
}

impl PlaneFlags {
    /// Resolve the keys and install the options for every server this process builds.
    pub fn install(&self) -> anyhow::Result<()> {
        let file = match &self.api_key_file {
            Some(p) => Some(
                std::fs::read_to_string(p)
                    .with_context(|| format!("--api-key-file {}", p.display()))?,
            ),
            None => None,
        };
        let keys = saber_serve::auth::ApiKeys::from_specs(
            self.api_key.iter().map(String::as_str),
            file.as_deref(),
            self.api_auth_all,
        )
        .map_err(anyhow::Error::msg)?;
        if let Some(k) = &keys {
            println!(
                "apiauth        ON: {} key(s) ({}), open routes: {}",
                k.labels().len(),
                k.labels().join(", "),
                if self.api_auth_all {
                    "none".to_owned()
                } else {
                    saber_serve::auth::OPEN_ROUTES.join(" ")
                }
            );
        }
        if self.request_ring == 0 {
            anyhow::bail!("--request-ring must be at least 1");
        }
        let _ = saber_serve::state::install_plane_options(saber_serve::state::PlaneOptions {
            auth: keys.map(std::sync::Arc::new),
            request_ring: self.request_ring,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct T {
        #[command(flatten)]
        p: PlaneFlags,
    }

    #[test]
    fn defaults_are_auth_off_and_a_1024_ring() {
        let t = T::parse_from(["t"]);
        assert!(t.p.api_key.is_empty() && t.p.api_key_file.is_none() && !t.p.api_auth_all);
        assert_eq!(t.p.request_ring, 1024);
        let t = T::parse_from([
            "t",
            "--api-key",
            "a=0123456789",
            "--api-key",
            "b=abcdefghij",
            "--api-auth-all",
            "--request-ring",
            "16",
        ]);
        assert_eq!(t.p.api_key.len(), 2);
        assert!(t.p.api_auth_all);
        assert_eq!(t.p.request_ring, 16);
    }
}
