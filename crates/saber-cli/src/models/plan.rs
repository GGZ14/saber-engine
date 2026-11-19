//! What a pull has to do: which packs and files are already here, which to download or verify,
//! how many bytes, where they go. Printed by `--dry-run`, executed by `pull` (and by `serve`
//! when it pulls first).

use std::path::PathBuf;

use super::cache::{self, Status};
use super::fetch::{self, Endpoint, Progress, RemoteFile, Repo};
use super::registry::Entry;
use super::ui::{human, tilde, Res, Style, UserError};
use super::Ctx;

/// Parts of a model the caller supplies itself (`saber serve --pack ...` etc.).
#[derive(Clone, Copy, Debug, Default)]
pub struct Skip {
    pub target_pack: bool,
    pub drafter_pack: bool,
    pub files: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Already here (verified, or an older download served in place).
    Have,
    /// Here but never verified (put there by hand): hash it, no download.
    Verify,
    /// Download (resuming whatever is there).
    Get,
}

/// One pack (an entry at a TP) or an entry's shared files.
#[derive(Clone, Debug)]
pub struct Item {
    pub label: String,
    pub repo: String,
    pub revision: String,
    pub endpoint: String,
    /// `None` = the shared files.
    pub tp: Option<u32>,
    pub tp_path: String,
    /// Where it is or goes.
    pub dir: PathBuf,
    pub action: Action,
    /// (remote file, local destination), when listed.
    pub files: Vec<(RemoteFile, PathBuf)>,
    pub bytes_total: u64,
    /// Bytes already on disk (complete files and `.part`s).
    pub bytes_have: u64,
    pub requires_tag: Option<String>,
    pub note: Option<String>,
}

/// The whole pull.
pub struct Plan {
    pub items: Vec<Item>,
    /// Set when the listing was skipped (`online = false`) or failed (not strict): the sizes are
    /// the registry's.
    pub estimate: Option<String>,
    pub retry: String,
}

fn existing(dest: &std::path::Path) -> u64 {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    std::fs::metadata(dest)
        .or_else(|_| std::fs::metadata(part))
        .map(|m| m.len())
        .unwrap_or(0)
}

impl Plan {
    /// Plan `e` (and its drafter `d`) at `tp`. `online` = may list the endpoint; `strict` = a
    /// failed listing is an error (a real pull) rather than a note (a dry run).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &Ctx,
        e: &Entry,
        d: Option<&Entry>,
        tp: u32,
        online: bool,
        strict: bool,
        retry: &str,
        skip: &Skip,
    ) -> Res<Plan> {
        let mut p = Plan {
            items: Vec::new(),
            estimate: if online {
                None
            } else {
                Some("not checked online".into())
            },
            retry: retry.to_string(),
        };
        if !skip.target_pack {
            p.pack(ctx, e, tp, online, strict)?;
        }
        if e.kind == super::registry::Kind::Target && !skip.files && !e.files.is_empty() {
            p.files(ctx, e, online, strict)?;
        }
        if let Some(d) = d {
            if !skip.drafter_pack {
                p.pack(ctx, d, tp, online, strict)?;
            }
        }
        Ok(p)
    }

    fn repo<'a>(&self, ctx: &Ctx, endpoint: &str, repo: &'a str, rev: &'a str) -> Repo<'a> {
        let ep = Endpoint::parse(endpoint);
        let token = if matches!(ep, Endpoint::Http(_)) {
            ctx.env.hf_token()
        } else {
            None
        };
        Repo {
            ep,
            repo,
            revision: rev,
            token,
            retry: self.retry.clone(),
        }
    }

    fn listing_failed(&mut self, strict: bool, err: UserError) -> Res<()> {
        if strict {
            return Err(err);
        }
        self.estimate = Some(err.what);
        Ok(())
    }

    fn pack(&mut self, ctx: &Ctx, e: &Entry, tp: u32, online: bool, strict: bool) -> Res<()> {
        let spec = e.at(tp).expect("the TP was chosen from the entry");
        let root = ctx.root();
        let managed = cache::tp_dir(root, e, tp).expect("has the TP");
        let st = cache::locate(root, e, tp);
        let (action, dir, note) = match &st {
            Status::Ready {
                dir,
                verified: true,
                ..
            } => (Action::Have, dir.clone(), None),
            Status::Ready {
                dir, legacy: true, ..
            } => (
                Action::Have,
                dir.clone(),
                Some("an older download, served in place".to_string()),
            ),
            Status::Ready { dir, .. } => (
                Action::Verify,
                dir.clone(),
                Some("complete but never verified: checking its hashes".to_string()),
            ),
            Status::Partial { .. } => (Action::Get, managed.clone(), Some("resuming".into())),
            Status::Missing => (Action::Get, managed.clone(), None),
        };
        let endpoint = ctx.reg.endpoint_of(e).to_string();
        let mut item = Item {
            label: format!("{} TP={tp}", e.name),
            repo: e.repo.clone(),
            revision: e.revision.clone(),
            endpoint: endpoint.clone(),
            tp: Some(tp),
            tp_path: spec.path.clone(),
            dir,
            action,
            files: Vec::new(),
            bytes_total: spec.bytes,
            bytes_have: if action == Action::Get {
                cache::dir_bytes(&managed)
            } else {
                spec.bytes
            },
            requires_tag: e.requires_tag.clone(),
            note,
        };
        if action != Action::Have && online {
            let r = self.repo(ctx, &endpoint, &e.repo, &e.revision);
            match r.list(&spec.path) {
                Ok(list) if list.is_empty() => {
                    let err =
                        UserError::new(format!("{} has no files under `{}/`", e.repo, spec.path))
                            .fix(format!(
                                "check the entry's `path` (saber models info {})",
                                super::short(e)
                            ));
                    self.listing_failed(strict, err)?;
                }
                Ok(list) => {
                    let sums = r.sums();
                    item.files = list
                        .into_iter()
                        .map(|mut f| {
                            if f.sha256.is_none() {
                                f.sha256 = sums
                                    .iter()
                                    .find(|(p, _)| *p == f.rel)
                                    .map(|(_, s)| s.clone());
                            }
                            let dest = cache::repo_dir(root, e).join(&f.rel);
                            let dest = if action == Action::Verify {
                                item.dir.join(f.rel.rsplit('/').next().unwrap_or(&f.rel))
                            } else {
                                dest
                            };
                            (f, dest)
                        })
                        .collect();
                    item.bytes_total = item.files.iter().map(|(f, _)| f.bytes.unwrap_or(0)).sum();
                    if action == Action::Get {
                        item.bytes_have = item.files.iter().map(|(_, d)| existing(d)).sum();
                    }
                }
                Err(err) => self.listing_failed(strict, err)?,
            }
        }
        self.items.push(item);
        Ok(())
    }

    fn files(&mut self, ctx: &Ctx, e: &Entry, online: bool, strict: bool) -> Res<()> {
        let root = ctx.root();
        let rd = cache::repo_dir(root, e);
        let missing = cache::missing_files(root, e);
        let endpoint = ctx.reg.endpoint_of(e).to_string();
        let mut item = Item {
            label: format!("{} tokenizer + chat template", e.name),
            repo: e.repo.clone(),
            revision: e.revision.clone(),
            endpoint: endpoint.clone(),
            tp: None,
            tp_path: String::new(),
            dir: rd.clone(),
            action: if missing.is_empty() {
                Action::Have
            } else {
                Action::Get
            },
            files: Vec::new(),
            bytes_total: 13_000_000,
            bytes_have: 0,
            requires_tag: None,
            note: None,
        };
        if item.action == Action::Have {
            item.bytes_total = e.files.iter().map(|f| existing(&rd.join(f))).sum();
            item.bytes_have = item.bytes_total;
        } else if online {
            let r = self.repo(ctx, &endpoint, &e.repo, &e.revision);
            match r.list("") {
                Ok(list) => {
                    let sums = r.sums();
                    for f in &missing {
                        match list.iter().find(|x| &x.rel == f) {
                            Some(x) => {
                                let mut x = x.clone();
                                if x.sha256.is_none() {
                                    x.sha256 =
                                        sums.iter().find(|(p, _)| p == f).map(|(_, s)| s.clone());
                                }
                                item.files.push((x, rd.join(f)));
                            }
                            None => {
                                let err =
                                    UserError::new(format!("{} has no `{f}` at its root", e.repo))
                                        .fix(format!(
                                            "check the entry's `files` (saber models info {})",
                                            super::short(e)
                                        ));
                                self.listing_failed(strict, err)?;
                            }
                        }
                    }
                    item.bytes_total = item.files.iter().map(|(f, _)| f.bytes.unwrap_or(0)).sum();
                    item.bytes_have = item.files.iter().map(|(_, d)| existing(d)).sum();
                }
                Err(err) => self.listing_failed(strict, err)?,
            }
        }
        self.items.push(item);
        Ok(())
    }

    /// Bytes still to download.
    #[must_use]
    pub fn need(&self) -> u64 {
        self.items
            .iter()
            .filter(|i| i.action == Action::Get)
            .map(|i| i.bytes_total.saturating_sub(i.bytes_have))
            .sum()
    }

    #[must_use]
    pub fn has_work(&self) -> bool {
        self.items.iter().any(|i| i.action != Action::Have)
    }

    /// The plan, one line an item, then the total and the free space.
    pub fn print(&self, ctx: &Ctx) {
        let st = Style::stderr();
        let w = self.items.iter().map(|i| i.label.len()).max().unwrap_or(0);
        for i in &self.items {
            let (verb, size) = match i.action {
                Action::Have => (st.green("have  "), human(i.bytes_total)),
                Action::Verify => (st.yellow("verify"), human(i.bytes_total)),
                Action::Get if i.bytes_have > 0 => (
                    st.cyan("get   "),
                    format!(
                        "{} left of {}",
                        human(i.bytes_total - i.bytes_have.min(i.bytes_total)),
                        human(i.bytes_total)
                    ),
                ),
                Action::Get => (st.cyan("get   "), human(i.bytes_total)),
            };
            let arrow = if i.action == Action::Get { "->" } else { "at" };
            eprintln!(
                "  {verb} {:<w$}  {:>9}  {arrow} {}{}",
                i.label,
                size,
                tilde(&i.dir),
                i.note
                    .as_deref()
                    .map_or(String::new(), |n| format!("  ({n})")),
            );
        }
        let need = self.need();
        let (fs, free, _) = saber_console::system::disk_space(ctx.root());
        let free_s = free.map_or("unknown".into(), human);
        if need > 0 {
            eprintln!(
                "  {} {}{} to {} ({free_s} free on {})",
                st.bold("download"),
                if self.estimate.is_some() {
                    "about "
                } else {
                    ""
                },
                human(need),
                tilde(ctx.root()),
                fs.display()
            );
        }
        if let Some(why) = &self.estimate {
            eprintln!(
                "  {}",
                st.dim(&format!("sizes are the registry's estimates ({why})"))
            );
        }
    }

    /// Enough free space for the download (plus 2 GB of slack)?
    pub fn check_disk(&self, ctx: &Ctx) -> Res<()> {
        let need = self.need();
        let (fs, free, _) = saber_console::system::disk_space(ctx.root());
        let Some(free) = free else {
            return Ok(());
        };
        let slack = 2_000_000_000u64;
        if need + slack <= free {
            return Ok(());
        }
        let mut others: Vec<String> = ctx
            .reg
            .entries
            .iter()
            .filter(|x| {
                x.tps().iter().any(|t| {
                    matches!(
                        cache::locate(ctx.root(), x, *t),
                        Status::Ready { legacy: false, .. }
                    )
                })
            })
            .filter(|x| !self.items.iter().any(|i| i.repo == x.repo))
            .map(|x| format!("saber models rm {}", super::short(x)))
            .collect();
        others.truncate(2);
        let mut err = UserError::new(format!(
            "not enough disk space: the download needs {} (+2 GB of room to work), {} has {} free",
            human(need),
            fs.display(),
            human(free)
        ));
        for o in others {
            err = err.fix(format!("free space: {o}"));
        }
        Err(err.fix(format!("or download to a bigger disk: {} --packs-dir /path/to/disk  (or export SABER_PACKS=/path/to/disk)", self.retry)))
    }

    /// Download, verify and mark everything the plan says.
    pub fn execute(&self, ctx: &Ctx) -> Res<()> {
        let st = Style::stderr();
        let gets: Vec<&Item> = self
            .items
            .iter()
            .filter(|i| i.action == Action::Get)
            .collect();
        if !gets.is_empty() {
            let total: u64 = gets.iter().map(|i| i.bytes_total).sum();
            eprintln!(
                "  {}",
                st.dim("downloads resume if interrupted: run the same command again")
            );
            let mut prog = Progress::new(total, "downloading");
            for i in &gets {
                let r = self.repo(ctx, &i.endpoint, &i.repo, &i.revision);
                for (f, dest) in &i.files {
                    r.download(f, dest, &mut prog)?;
                }
            }
            prog.finish(&format!(
                "{} file{}",
                gets.iter().map(|i| i.files.len()).sum::<usize>(),
                if gets.len() == 1 && gets[0].files.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ));
        }
        let checks: Vec<&Item> = self
            .items
            .iter()
            .filter(|i| i.action != Action::Have)
            .collect();
        let hashed: u64 = checks
            .iter()
            .flat_map(|i| i.files.iter())
            .filter(|(f, _)| f.sha256.is_some())
            .map(|(f, _)| f.bytes.unwrap_or(0))
            .sum();
        let mut prog = Progress::new(hashed, "verifying  ");
        for i in &checks {
            let mut marked = Vec::new();
            for (f, dest) in &i.files {
                if let Some(want) = &f.sha256 {
                    let got = fetch::sha256_with_progress(dest, &mut prog).map_err(|e| {
                        UserError::new(format!("cannot read {} ({e})", dest.display()))
                            .fix(self.retry.clone())
                    })?;
                    if &got != want {
                        prog.clear_line();
                        let _ = std::fs::remove_file(dest);
                        return Err(UserError::new(format!(
                            "{} failed its checksum (a corrupt or changed download); it was removed",
                            f.rel
                        ))
                        .fix(format!("run `{}` again to fetch it afresh", self.retry)));
                    }
                }
                let bytes = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
                if f.bytes.is_some_and(|b| b != bytes) {
                    prog.clear_line();
                    return Err(UserError::new(format!(
                        "{} is {bytes} bytes; the repo says {}",
                        f.rel,
                        f.bytes.unwrap_or(0)
                    ))
                    .fix(format!(
                        "delete {} and run `{}` again",
                        dest.display(),
                        self.retry
                    )));
                }
                marked.push(cache::MarkedFile {
                    path: f.rel.clone(),
                    bytes,
                    sha256: f.sha256.clone(),
                });
            }
            if let Some(tp) = i.tp {
                if let Err(problem) = cache::check_pack(&i.dir, tp, i.requires_tag.as_deref()) {
                    prog.clear_line();
                    let name = i
                        .label
                        .rsplit_once(" TP=")
                        .map_or(i.label.as_str(), |x| x.0);
                    let handle = ctx.reg.find(name).map_or(name, super::short);
                    let rm = format!("saber models rm {handle} --tp {tp} --yes");
                    let err = UserError::new(format!(
                        "the downloaded {} is not usable: {problem}",
                        i.label
                    ));
                    return Err(match problem {
                        cache::PackProblem::MissingRank(_) | cache::PackProblem::BadMagic(_) => err
                            .fix(format!("{rm}, then {} (a fresh download)", self.retry)),
                        _ => err
                            .note("the repo does not hold the pack this registry entry describes")
                            .fix(format!("check the entry's `repo`, `path` and `requires_tag`: saber models info {handle}"))
                            .fix(format!("clean up: {rm}")),
                    });
                }
                let m = cache::Marker {
                    repo: i.repo.clone(),
                    revision: i.revision.clone(),
                    path: i.tp_path.clone(),
                    tp,
                    files: marked,
                    verified_unix: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs()),
                };
                cache::write_marker(&i.dir, &m).map_err(|e| {
                    UserError::new(format!(
                        "cannot write {} in {} ({e})",
                        cache::MARKER,
                        i.dir.display()
                    ))
                })?;
            }
        }
        if !checks.is_empty() {
            prog.clear_line();
            let n_hashed = checks
                .iter()
                .flat_map(|i| i.files.iter())
                .filter(|(f, _)| f.sha256.is_some())
                .count();
            eprintln!(
                "  {} {} checked{}",
                st.green("verified"),
                checks
                    .iter()
                    .map(|i| i.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                if n_hashed > 0 {
                    format!(" ({n_hashed} sha256 hashes, pack headers and format tags)")
                } else {
                    " (sizes and pack headers)".into()
                }
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cache::{testkit::fake_pack, Env};
    use crate::models::registry::Registry;
    use crate::models::Gpus;
    use saber_test_scratch::ScratchDir;

    fn ctx(root: &std::path::Path, user: Option<&str>) -> Ctx {
        let reg = Registry::from_texts(
            crate::models::registry::SHIPPED,
            user.map(|t| (t, std::path::Path::new("/u/models.toml"))),
        )
        .unwrap();
        Ctx {
            reg,
            env: Env {
                packs_root: root.to_path_buf(),
                ..Env::default()
            },
            gpus: Gpus::default(),
        }
    }

    #[test]
    fn an_offline_plan_uses_the_registry_sizes() {
        let s = ScratchDir::new("models-plan-offline");
        let c = ctx(&s, None);
        let e = c.reg.resolve("w4a8").unwrap();
        let d = c.reg.resolve("drafter-w4a8").unwrap();
        let p = Plan::build(&c, e, Some(d), 2, false, false, "r", &Skip::default()).unwrap();
        assert_eq!(p.items.len(), 3);
        assert!(p.items.iter().all(|i| i.action == Action::Get));
        assert_eq!(p.need(), 20_543_143_513 + 7_260_439_156 + 13_000_000);
        assert!(p.estimate.is_some());
        // a legacy drafter folder is used in place
        fake_pack(&s.join("qwen38-dflash2-drafter-w4a8"), 2, &["macaq_iu8d/1"]);
        let p = Plan::build(&c, e, Some(d), 2, false, false, "r", &Skip::default()).unwrap();
        assert_eq!(p.items[2].action, Action::Have);
        assert_eq!(p.need(), 20_543_143_513 + 13_000_000);
    }

    /// A synthetic model on a mirror directory: pull, verify, mark, then nothing to do.
    #[test]
    fn a_mirror_pull_end_to_end() {
        let s = ScratchDir::new("models-plan-mirror");
        let m = s.join("mirror/me/tiny/tp1");
        fake_pack(&m, 1, &["macaq_iu8/1"]);
        std::fs::write(s.join("mirror/me/tiny/tok.json"), b"{}").unwrap();
        std::fs::write(s.join("mirror/me/tiny/t.jinja"), b"x").unwrap();
        let sha = saber_console::downloads::sha256_file(&m.join("rank0.pack")).unwrap();
        std::fs::write(
            s.join("mirror/me/tiny/SHA256SUMS"),
            format!("{sha}  tp1/rank0.pack\n"),
        )
        .unwrap();
        let user = format!(
            r#"
[[model]]
name = "tiny"
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
bytes = 1
max_ctx = 4096
max_seqs = 1
decode_batches = "1"
decode_widths = "1"
"#,
            s.join("mirror").display()
        );
        let root = s.join("cache");
        let c = ctx(&root, Some(&user));
        let e = c.reg.resolve("tiny").unwrap();
        let p = Plan::build(&c, e, None, 1, true, true, "r", &Skip::default()).unwrap();
        assert!(p.estimate.is_none());
        assert_eq!(p.items[0].files.len(), 2);
        assert_eq!(
            p.items[0]
                .files
                .iter()
                .filter(|(f, _)| f.sha256.is_some())
                .count(),
            1
        );
        p.execute(&c).unwrap();
        assert!(matches!(
            cache::locate(&root, e, 1),
            Status::Ready {
                verified: true,
                legacy: false,
                ..
            }
        ));
        assert!(cache::missing_files(&root, e).is_empty());
        let again = Plan::build(&c, e, None, 1, true, true, "r", &Skip::default()).unwrap();
        assert!(!again.has_work());
        assert_eq!(again.need(), 0);

        // a corrupt file on the mirror fails its checksum and is removed
        let root2 = s.join("cache2");
        std::fs::write(m.join("rank0.pack"), b"SBRPACK1-changed").unwrap();
        let c2 = ctx(&root2, Some(&user));
        let e2 = c2.reg.resolve("tiny").unwrap();
        let p = Plan::build(
            &c2,
            e2,
            None,
            1,
            true,
            true,
            "saber models pull tiny",
            &Skip::default(),
        )
        .unwrap();
        let err = p.execute(&c2).unwrap_err().render(false);
        assert!(
            err.contains("failed its checksum") && err.contains("saber models pull tiny"),
            "{err}"
        );
        assert!(!cache::tp_dir(&root2, e2, 1)
            .unwrap()
            .join("rank0.pack")
            .exists());
    }

    #[test]
    fn a_full_disk_is_refused_before_downloading() {
        let s = ScratchDir::new("models-plan-disk");
        let c = ctx(&s, None);
        let e = c.reg.resolve("w4a8").unwrap();
        let mut p = Plan::build(
            &c,
            e,
            None,
            2,
            false,
            false,
            "saber models pull w4a8",
            &Skip::default(),
        )
        .unwrap();
        p.items[0].bytes_total = u64::MAX / 4;
        let err = p.check_disk(&c).unwrap_err().render(false);
        assert!(
            err.contains("not enough disk space") && err.contains("--packs-dir"),
            "{err}"
        );
    }
}
