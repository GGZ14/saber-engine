//! Just enough `.npy` to read what `the internal reference harness` writes.
//!
//! Every captured buffer is a **flat `uint8`** array — the capture script views each tensor as
//! bytes before saving it. That is deliberate and it is the whole tolerance policy of the gate:
//! with no dtype there is nothing to compare approximately, so the only possible verdict is
//! byte-identical or not.
//!
//! A file that is not flat `|u1` is an error, never a conversion. The reference capture is the
//! contract; reinterpreting it here would make the gate a test of this parser.

use std::path::Path;

/// Read a flat `uint8` `.npy` written by NumPy (format 1.0, 2.0 or 3.0).
pub fn read_u8(path: &Path) -> anyhow::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        anyhow::bail!("{}: not a .npy file", path.display());
    }
    let major = bytes[6];
    let (header_len, header_at) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        other => anyhow::bail!("{}: .npy format version {other}", path.display()),
    };
    let end = header_at + header_len;
    if end > bytes.len() {
        anyhow::bail!("{}: truncated .npy header", path.display());
    }
    let header = String::from_utf8_lossy(&bytes[header_at..end]).to_string();
    if !header.contains("'|u1'") && !header.contains("\"|u1\"") {
        anyhow::bail!(
            "{}: the gate only reads flat uint8 (header was {header})",
            path.display()
        );
    }
    if header.contains("'fortran_order': True") {
        anyhow::bail!("{}: fortran order", path.display());
    }
    Ok(bytes[end..].to_vec())
}

/// Read it only if it is there. A missing buffer is a normal outcome — a name may be an input, an
/// expected output, a buffer shared with another case, or a workspace the launcher owns — so the
/// gate asks for each in turn rather than treating absence as failure.
pub fn read_u8_opt(path: &Path) -> anyhow::Result<Option<Vec<u8>>> {
    if !path.exists() {
        return Ok(None);
    }
    read_u8(path).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_npy(dir: &Path, name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        let mut h = header.as_bytes().to_vec();
        while (10 + h.len()) % 64 != 0 {
            h.push(b' ');
        }
        h.push(b'\n');
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        f.write_all(&(h.len() as u16).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(data).unwrap();
        p
    }

    #[test]
    fn reads_a_flat_uint8_array() {
        // Under `target/`, removed on drop -- never `/tmp` (FN51h).
        let dir = saber_test_scratch::ScratchDir::new("saber-npy-test");
        let p = write_npy(
            &dir,
            "ok.npy",
            "{'descr': '|u1', 'fortran_order': False, 'shape': (4,), }",
            &[1, 2, 3, 4],
        );
        assert_eq!(read_u8(&p).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn negative_control_a_typed_array_is_refused_rather_than_reinterpreted() {
        // Under `target/`, removed on drop -- never `/tmp` (FN51h).
        let dir = saber_test_scratch::ScratchDir::new("saber-npy-test");
        let p = write_npy(
            &dir,
            "f4.npy",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }",
            &[0, 0, 0, 0],
        );
        assert!(read_u8(&p).is_err());
    }
}
