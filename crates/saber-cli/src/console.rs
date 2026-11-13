//! `saber ui`: resolve the console's paths and the flag schema, then hand off to
//! `saber-console`. The schema is read from THIS binary's clap structs (`Cli`, its `serve` and
//! `engine` subcommands with the flattened `FnFlags`), so the UI never drifts from the code.

use std::path::PathBuf;

use clap::CommandFactory;

pub struct UiArgs {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub admin_token: Option<String>,
    pub gen_admin_token: bool,
    pub config: Option<PathBuf>,
    pub engine: Option<String>,
    pub mock_port: u16,
    pub dump_schema: bool,
}

/// The subcommands whose flags the console configures (the two serving boots).
pub const SCHEMA_SECTIONS: &[&str] = &["serve", "engine"];

/// The flag schema of this binary.
pub fn schema() -> saber_console::schema::Schema {
    saber_console::schema::build(&crate::Cli::command(), SCHEMA_SECTIONS)
}

pub fn run(a: UiArgs) -> anyhow::Result<()> {
    let schema = schema();
    if a.dump_schema {
        println!("{}", serde_json::to_string_pretty(&schema)?);
        return Ok(());
    }
    // The console's own locations only (the engine still reads no environment variable: the
    // console passes it everything as arguments).
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME is not set; pass --config"))?;
    let config_path = a
        .config
        .unwrap_or_else(|| home.join(".config/saber/config.toml"));
    if a.gen_admin_token {
        let token = saber_console::gen_admin_token(&config_path)?;
        println!("saber ui: a new admin token (shown once; the config keeps only its hash):");
        println!("{token}");
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    // A binary built inside a checkout (`<repo>/.cache/target-container/release/saber`, or
    // `<repo>/target/...`) roots the default paths at that checkout, wherever it was cloned.
    let checkout = checkout_of(&exe);
    let rooted = |mut c: saber_console::config::Config| {
        if let Some(repo) = &checkout {
            c.engine.paths = c.engine.paths.rooted_at(repo);
        }
        c
    };
    let override_config = match a.engine.as_deref() {
        Some("mock") => Some(rooted(saber_console::config::Config::mock(a.mock_port))),
        Some(other) => anyhow::bail!("--engine {other}: the console's override is `mock`"),
        None => None,
    };
    saber_console::run(saber_console::ConsoleArgs {
        host: a.host,
        port: a.port,
        admin_token: a.admin_token,
        admin: None,
        config_path,
        default_config: rooted(saber_console::config::Config::default()),
        override_config,
        home,
        exe,
        schema: serde_json::to_value(&schema)?,
    })
}

/// The Saber checkout `exe` was built in: the nearest ancestor that holds the workspace's
/// `Cargo.toml` and `tools/serve_console.sh`.
fn checkout_of(exe: &std::path::Path) -> Option<PathBuf> {
    exe.ancestors()
        .skip(1)
        .find(|d| d.join("Cargo.toml").is_file() && d.join("tools/serve_console.sh").is_file())
        .map(std::path::Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checkout_is_found_from_a_binary_inside_it() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let repo = std::fs::canonicalize(repo).unwrap();
        let exe = repo.join(".cache/target-container/release/saber");
        assert_eq!(checkout_of(&exe).as_deref(), Some(repo.as_path()));
        assert_eq!(checkout_of(std::path::Path::new("/usr/bin/saber")), None);
        let p = saber_console::config::Paths::default().rooted_at(&repo);
        assert_eq!(p.repo, repo);
        assert_eq!(p.chat_template, repo.join(".cache/template_corpus.jinja"));
        assert_eq!(p.nvme, saber_console::config::Paths::default().nvme);
    }

    /// Every flag of every serving boot has a group and a tier, and every hand-annotated basic
    /// flag still exists. A new flag whose name matches no group rule fails HERE: give it a
    /// group in `saber_console::schema::GROUP_RULES` (or a basic annotation).
    #[test]
    fn console_schema_covers_every_flag() {
        // building the whole `Cli` command overflows the default 2 MiB test stack in debug
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(covers_every_flag)
            .unwrap()
            .join()
            .unwrap();
    }

    fn covers_every_flag() {
        let s = schema();
        assert!(
            s.unclassified.is_empty(),
            "flags without a console group/tier (add a rule to saber_console::schema::GROUP_RULES): {:#?}",
            s.unclassified
        );
        assert!(
            s.stale_annotations.is_empty(),
            "basic annotations naming no flag: {:?}",
            s.stale_annotations
        );
        let n_basic = s
            .flags
            .iter()
            .filter(|f| f.tier == saber_console::schema::Tier::Basic)
            .count();
        assert!((20..=40).contains(&n_basic), "{n_basic} basic flags");
        for f in &s.flags {
            assert!(
                !f.group.is_empty() && !f.help.is_empty() || f.hidden,
                "{f:?}"
            );
        }
        // both serving boots are in it, with the big ones
        for (sec, name) in [
            ("engine", "--fn-max-ctx"),
            ("serve", "--engine-max-ctx"),
            ("global", "--verbose"),
        ] {
            assert!(
                s.flags.iter().any(|f| f.section == sec && f.name == name),
                "{sec} {name}"
            );
        }
    }
}
