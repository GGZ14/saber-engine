//! Read the AMDGPU code-object v5 metadata out of a `.hsaco` produced by `hipcc --genco`.
//!
//! `hipcc --genco` emits a *clang offload bundle* (`__CLANG_OFFLOAD_BUNDLE__` magic) holding an
//! empty host entry plus one `hipv4-amdgcn-amd-amdhsa--gfx1201` ELF. `hipModuleLoad` unbundles it
//! at load time; so do we, then read the `NT_AMDGPU_METADATA` (type 32, owner `AMDGPU`) note,
//! which is a msgpack map with `amdhsa.kernels[]`.
//!
//! This is the only source of truth for a kernel's kernarg layout, LDS, wave size and register
//! counts. The manifest *declares* them and `Manifest::validate_against_hsaco` refuses on any
//! disagreement -- the code-object equivalent of the partition-mismatched tuple that silently
//! hangs the GPU (docs/ARCHITECTURE.md, Kernel plane).

use crate::msgpack::{self, Value};
use crate::{Error, Result};

const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
const NT_AMDGPU_METADATA: u32 = 32;
const SHT_NOTE: u32 = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgMeta {
    pub name: Option<String>,
    pub offset: u32,
    pub size: u32,
    pub value_kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelMeta {
    pub name: String,
    pub symbol: String,
    pub kernarg_segment_size: u32,
    pub kernarg_segment_align: u32,
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub wavefront_size: u32,
    pub max_flat_workgroup_size: u32,
    pub args: Vec<ArgMeta>,
}

impl KernelMeta {
    /// Args the host must supply. The trailing `hidden_*` args are filled by the HIP runtime.
    pub fn explicit_args(&self) -> impl Iterator<Item = &ArgMeta> {
        self.args
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden_"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeObject {
    pub target: String,
    pub kernels: Vec<KernelMeta>,
}

impl CodeObject {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let elf = if bytes.starts_with(BUNDLE_MAGIC) {
            unbundle(bytes)?
        } else {
            bytes
        };
        let note = amdgpu_note(elf)?;
        let (v, _) = msgpack::read(note, 0).map_err(|e| Error::Parse(e.to_string()))?;
        from_metadata(&v)
    }

    pub fn kernel(&self, name: &str) -> Option<&KernelMeta> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

/// The bare amdgcn ELF of a committed `.hsaco` (a clang offload bundle) or of a bare ELF -- what a
/// non-HIP loader (`hsa_executable_load_agent_code_object`, FN63) takes. HIP unbundles itself.
pub fn amdgcn_elf(bytes: &[u8]) -> Result<&[u8]> {
    if bytes.starts_with(BUNDLE_MAGIC) {
        unbundle(bytes)
    } else {
        Ok(bytes)
    }
}

/// Pick the (single) amdgcn entry out of a clang offload bundle.
fn unbundle(bytes: &[u8]) -> Result<&[u8]> {
    let mut p = BUNDLE_MAGIC.len();
    let n = rd_u64(bytes, p)? as usize;
    p += 8;
    let mut found: Option<(usize, usize)> = None;
    for _ in 0..n {
        let off = rd_u64(bytes, p)? as usize;
        let size = rd_u64(bytes, p + 8)? as usize;
        let tlen = rd_u64(bytes, p + 16)? as usize;
        let triple = bytes
            .get(p + 24..p + 24 + tlen)
            .ok_or_else(|| Error::Parse("offload bundle: truncated triple".into()))?;
        let triple = String::from_utf8_lossy(triple).into_owned();
        p += 24 + tlen;
        if triple.contains("amdgcn") {
            if found.is_some() {
                return Err(Error::Parse(
                    "offload bundle holds more than one amdgcn entry; one arch per blob".into(),
                ));
            }
            found = Some((off, size));
        }
    }
    let (off, size) =
        found.ok_or_else(|| Error::Parse("offload bundle has no amdgcn entry".into()))?;
    bytes
        .get(off..off + size)
        .ok_or_else(|| Error::Parse("offload bundle: entry out of range".into()))
}

fn rd_u64(b: &[u8], at: usize) -> Result<u64> {
    let s = b
        .get(at..at + 8)
        .ok_or_else(|| Error::Parse(format!("truncated u64 at {at}")))?;
    let mut a = [0u8; 8];
    a.copy_from_slice(s);
    Ok(u64::from_le_bytes(a))
}

fn rd_u32(b: &[u8], at: usize) -> Result<u32> {
    let s = b
        .get(at..at + 4)
        .ok_or_else(|| Error::Parse(format!("truncated u32 at {at}")))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_u16(b: &[u8], at: usize) -> Result<u16> {
    let s = b
        .get(at..at + 2)
        .ok_or_else(|| Error::Parse(format!("truncated u16 at {at}")))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

/// The `NT_AMDGPU_METADATA` note payload of a 64-bit little-endian ELF.
fn amdgpu_note(elf: &[u8]) -> Result<&[u8]> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" {
        return Err(Error::Parse("not an ELF".into()));
    }
    if elf[4] != 2 || elf[5] != 1 {
        return Err(Error::Parse("expected a 64-bit little-endian ELF".into()));
    }
    let shoff = rd_u64(elf, 0x28)? as usize;
    let shentsize = rd_u16(elf, 0x3a)? as usize;
    let shnum = rd_u16(elf, 0x3c)? as usize;
    for i in 0..shnum {
        let sh = shoff + i * shentsize;
        if rd_u32(elf, sh + 4)? != SHT_NOTE {
            continue;
        }
        let off = rd_u64(elf, sh + 0x18)? as usize;
        let size = rd_u64(elf, sh + 0x20)? as usize;
        let seg = elf
            .get(off..off + size)
            .ok_or_else(|| Error::Parse("note section out of range".into()))?;
        let mut p = 0usize;
        while p + 12 <= seg.len() {
            let namesz = rd_u32(seg, p)? as usize;
            let descsz = rd_u32(seg, p + 4)? as usize;
            let ntype = rd_u32(seg, p + 8)?;
            let name_at = p + 12;
            let desc_at = name_at + namesz.div_ceil(4) * 4;
            let end = desc_at + descsz.div_ceil(4) * 4;
            let name = seg
                .get(name_at..name_at + namesz)
                .ok_or_else(|| Error::Parse("note name out of range".into()))?;
            if ntype == NT_AMDGPU_METADATA && name.starts_with(b"AMDGPU") {
                return seg
                    .get(desc_at..desc_at + descsz)
                    .ok_or_else(|| Error::Parse("note desc out of range".into()));
            }
            p = end;
        }
    }
    Err(Error::Parse(
        "no NT_AMDGPU_METADATA note (code object v5?)".into(),
    ))
}

fn u32_field(v: &Value, key: &str, ctx: &str) -> Result<u32> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Parse(format!("{ctx}: missing or non-integer `{key}`")))
        .map(|x| x as u32)
}

fn from_metadata(v: &Value) -> Result<CodeObject> {
    let target = v
        .get("amdhsa.target")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let ks = v
        .get("amdhsa.kernels")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Parse("metadata has no amdhsa.kernels array".into()))?;
    let mut kernels = Vec::with_capacity(ks.len());
    for k in ks {
        let name = k
            .get(".name")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Parse("kernel without .name".into()))?
            .to_owned();
        let symbol = k
            .get(".symbol")
            .and_then(Value::as_str)
            .unwrap_or(&name)
            .to_owned();
        let mut args = Vec::new();
        if let Some(list) = k.get(".args").and_then(Value::as_array) {
            for a in list {
                args.push(ArgMeta {
                    name: a.get(".name").and_then(Value::as_str).map(str::to_owned),
                    offset: u32_field(a, ".offset", &name)?,
                    size: u32_field(a, ".size", &name)?,
                    value_kind: a
                        .get(".value_kind")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::Parse(format!("{name}: arg without .value_kind")))?
                        .to_owned(),
                });
            }
        }
        kernels.push(KernelMeta {
            kernarg_segment_size: u32_field(k, ".kernarg_segment_size", &name)?,
            kernarg_segment_align: u32_field(k, ".kernarg_segment_align", &name)?,
            group_segment_fixed_size: u32_field(k, ".group_segment_fixed_size", &name)?,
            private_segment_fixed_size: u32_field(k, ".private_segment_fixed_size", &name)?,
            vgpr_count: u32_field(k, ".vgpr_count", &name)?,
            sgpr_count: u32_field(k, ".sgpr_count", &name)?,
            wavefront_size: u32_field(k, ".wavefront_size", &name)?,
            max_flat_workgroup_size: u32_field(k, ".max_flat_workgroup_size", &name)?,
            name,
            symbol,
            args,
        });
    }
    Ok(CodeObject { target, kernels })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage() {
        assert!(CodeObject::parse(b"not an elf at all").is_err());
    }

    #[test]
    fn reads_the_committed_blobs() {
        for (path, sample) in crate::prebuilt::ALL {
            let bytes = std::fs::read(crate::prebuilt::dir().join(path))
                .unwrap_or_else(|e| panic!("read {path}: {e}"));
            let co = CodeObject::parse(&bytes).unwrap_or_else(|e| panic!("parse {path}: {e}"));
            assert!(
                co.target.contains("gfx1201"),
                "{path}: target {}",
                co.target
            );
            let k = co
                .kernel(sample)
                .unwrap_or_else(|| panic!("{path}: no kernel {sample}"));
            assert_eq!(k.wavefront_size, 32, "{path}/{sample} wave size");
            // The null plugin is the one code object whose sample kernel takes NO argument: it
            // exists so that a dispatch can be measured with no kernarg image at all. Asserting
            // the opposite bound for it keeps the >= 8 below a statement about the plugins that
            // really do carry pointers, rather than one loosened until everything passes.
            if *path == "noop.hsaco" {
                assert_eq!(k.kernarg_segment_size, 0, "{path}/{sample}");
                assert_eq!(k.explicit_args().count(), 0, "{path}/{sample}");
                continue;
            }
            assert!(k.kernarg_segment_size >= 8);
            // At least one explicit argument -- the reference's paged attention kernels take exactly
            // one, a 136-byte `SABERArgs` BY VALUE, so a count bound any higher would be a fact
            // about the MXFP4 plugin rather than about reading a code object.
            assert!(k.explicit_args().count() >= 1);
            assert!(
                k.explicit_args().map(|a| a.size).sum::<u32>() > 0,
                "{path}/{sample}: the explicit arguments occupy no bytes"
            );
        }
    }
}
