//! `saber models` and `saber serve --model` end to end, as a person runs them: the real binary, a
//! user registry with a synthetic model on a local mirror directory, a scratch pack cache. No
//! network, no GPU (every command here is a dry run, a list, or a pull from the mirror).

use std::path::Path;
use std::process::{Command, Output};

use saber_test_scratch::ScratchDir;

struct Box_ {
    dir: ScratchDir,
}

impl Box_ {
    fn new(name: &str) -> Box_ {
        let dir = ScratchDir::new(name);
        let m = dir.join("mirror/me/tiny");
        for (tp, sub) in [(1u32, "tp1"), (2, "tp2")] {
            let d = m.join(sub);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(
                d.join("manifest.json"),
                format!(r#"{{"tp_size":{tp},"plan":{{"descriptor":{{"transforms":["macaq_iu8/1 synthetic"]}}}}}}"#),
            )
            .unwrap();
            for r in 0..tp {
                std::fs::write(
                    d.join(format!("rank{r}.pack")),
                    b"SBRPACK1 a tiny synthetic pack",
                )
                .unwrap();
            }
        }
        std::fs::write(m.join("tok.json"), b"{}").unwrap();
        std::fs::write(m.join("t.jinja"), b"{{ x }}").unwrap();
        std::fs::create_dir_all(dir.join("cfg/saber")).unwrap();
        std::fs::create_dir_all(dir.join("prebuilt/gfx1201")).unwrap();
        std::fs::write(
            dir.join("cfg/saber/models.toml"),
            format!(
                r#"
default = "tiny"
[[model]]
name = "Tiny-SABER"
aliases = ["tiny"]
kind = "target"
repo = "me/tiny"
endpoint = "file://{}"
served_id = "Tiny"
requires_tag = "macaq_iu8/1"
files = ["tok.json", "t.jinja"]
tokenizer = "tok.json"
chat_template = "t.jinja"
[model.tp.1]
path = "tp1"
bytes = 30
max_ctx = 4096
max_seqs = 1
decode_batches = "1"
decode_widths = "1"
"#,
                dir.join("mirror").display()
            ),
        )
        .unwrap();
        Box_ { dir }
    }

    fn saber(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_saber"))
            .args(args)
            .env("XDG_CONFIG_HOME", self.dir.join("cfg"))
            .env("SABER_PACKS", self.dir.join("packs"))
            .env("HF_HOME", self.dir.join("hf"))
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN")
            .env("NO_COLOR", "1")
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap()
    }

    fn prebuilt(&self) -> String {
        self.dir.join("prebuilt/gfx1201").display().to_string()
    }
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

#[test]
fn list_pull_serve_dry_run_rm() {
    let b = Box_::new("models-cli");
    let o = b.saber(&["models"]);
    let t = text(&o);
    assert!(o.status.success(), "{t}");
    assert!(
        t.contains("* Tiny-SABER") && t.contains("Qwen3.8-27B-MACaQ-W4A8-SABER"),
        "{t}"
    );

    // json is parseable and says not downloaded
    let o = b.saber(&["models", "info", "tiny", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["tps"][0]["status"], "not downloaded");

    // a dry run before the pull names the download and never touches the cache
    let o = b.saber(&[
        "serve",
        "--tp",
        "1",
        "--dry-run",
        "--prebuilt",
        &b.prebuilt(),
    ]);
    let t = text(&o);
    assert!(o.status.success(), "{t}");
    assert!(
        t.contains("would download these first (saber models pull tiny --tp 1)"),
        "{t}"
    );
    assert!(
        t.contains("Engine command:") && t.contains("Dry run: nothing was started or downloaded."),
        "{t}"
    );
    assert!(!b.dir.join("packs/me").exists());

    // pull from the mirror, then again (idempotent)
    let o = b.saber(&["models", "pull", "tiny", "--tp", "1"]);
    let t = text(&o);
    assert!(o.status.success(), "{t}");
    assert!(t.contains("verified"), "{t}");
    let o = b.saber(&["models", "pull", "tiny", "--tp", "1"]);
    assert!(text(&o).contains("Nothing to do"), "{}", text(&o));

    // serve resolves the cached packs; argv is the explicit engine command line
    let o = b.saber(&[
        "serve",
        "--model",
        "tiny",
        "--tp",
        "1",
        "--print-argv",
        "--prebuilt",
        &b.prebuilt(),
        "--port",
        "9123",
    ]);
    let t = text(&o);
    assert!(o.status.success(), "{t}");
    let argv: Vec<&str> = std::str::from_utf8(&o.stdout).unwrap().lines().collect();
    let at = |f: &str| {
        argv.iter()
            .position(|a| *a == f)
            .map(|i| argv[i + 1])
            .unwrap_or("")
    };
    assert_eq!(argv[1], "serve");
    assert_eq!(at("--engine"), "real");
    assert_eq!(at("--engine-tp"), "1");
    assert_eq!(Path::new(at("--pack")), b.dir.join("packs/me/tiny/tp1"));
    assert_eq!(
        Path::new(at("--tokenizer")),
        b.dir.join("packs/me/tiny/tok.json")
    );
    assert_eq!(at("--served-model-name"), "Tiny");
    assert_eq!(at("--port"), "9123");
    assert!(!argv.contains(&"--model"));

    // an engine-flag typo is caught by the engine's own parser, with clap's suggestion
    let o = b.saber(&[
        "serve",
        "--tp",
        "1",
        "--dry-run",
        "--prebuilt",
        &b.prebuilt(),
        "--engine-max-ctxx",
        "5",
    ]);
    let t = text(&o);
    assert_eq!(o.status.code(), Some(2), "{t}");
    assert!(
        t.contains("--engine-max-ctx'") && t.contains("fix:") && !t.contains("Usage:"),
        "{t}"
    );

    // unknown model: did you mean
    let o = b.saber(&["models", "info", "tiy"]);
    assert!(text(&o).contains("did you mean: tiny?"), "{}", text(&o));

    // rm without a terminal asks for --yes; with it, deletes and cleans up
    let o = b.saber(&["models", "rm", "tiny"]);
    assert!(
        !o.status.success() && text(&o).contains("--yes"),
        "{}",
        text(&o)
    );
    let o = b.saber(&["models", "rm", "tiny", "--yes"]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(!b.dir.join("packs/me").exists());
    let o = b.saber(&["models", "rm", "tiny", "--yes"]);
    assert!(text(&o).contains("nothing to remove"), "{}", text(&o));
}

#[test]
fn a_broken_user_registry_names_the_file_and_line() {
    let b = Box_::new("models-cli-bad");
    std::fs::write(
        b.dir.join("cfg/saber/models.toml"),
        "[[model]]\nname = \"x\"\nkind = \"tagret\"\n",
    )
    .unwrap();
    let o = b.saber(&["models"]);
    let t = text(&o);
    assert!(!o.status.success(), "{t}");
    assert!(
        t.contains("models.toml is not a valid registry file") && t.contains("line 3"),
        "{t}"
    );
    assert!(
        t.contains("fix:") && !t.contains("panicked") && !t.contains("RUST_BACKTRACE"),
        "{t}"
    );
}
