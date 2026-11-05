//! A minimal MessagePack reader, enough for the AMDGPU code-object v5 metadata note.
//!
//! Hand-written rather than pulled in as a dependency: this runs on the host with no ROCm and no
//! container, it is the only msgpack in the engine, and the note is a closed schema (maps of
//! strings to unsigned ints, strings and arrays). Anything the note cannot contain is an error
//! rather than a silently-skipped value.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Uint(u64),
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<Value>),
    Map(Vec<(Value, Value)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "msgpack: {}", self.0)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

fn need(buf: &[u8], at: usize, n: usize) -> Result<&[u8]> {
    buf.get(at..at + n).ok_or_else(|| {
        Error(format!(
            "truncated: want {n} bytes at {at}, have {}",
            buf.len()
        ))
    })
}

fn be_uint(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |a, &x| (a << 8) | x as u64)
}

/// Decode one value at `at`; returns the value and the offset just past it.
pub fn read(buf: &[u8], at: usize) -> Result<(Value, usize)> {
    let tag = *need(buf, at, 1)?.first().expect("1 byte");
    let p = at + 1;
    match tag {
        0x00..=0x7f => Ok((Value::Uint(tag as u64), p)),
        0xe0..=0xff => Ok((Value::Int(tag as i8 as i64), p)),
        0x80..=0x8f => read_map(buf, p, (tag & 0x0f) as usize),
        0x90..=0x9f => read_array(buf, p, (tag & 0x0f) as usize),
        0xa0..=0xbf => read_str(buf, p, (tag & 0x1f) as usize),
        0xc0 => Ok((Value::Nil, p)),
        0xc2 => Ok((Value::Bool(false), p)),
        0xc3 => Ok((Value::Bool(true), p)),
        0xc4..=0xc6 => {
            let w = 1 << (tag - 0xc4);
            let n = be_uint(need(buf, p, w)?) as usize;
            let b = need(buf, p + w, n)?.to_vec();
            Ok((Value::Bin(b), p + w + n))
        }
        0xca => {
            let b = need(buf, p, 4)?;
            Ok((
                Value::F32(f32::from_be_bytes([b[0], b[1], b[2], b[3]])),
                p + 4,
            ))
        }
        0xcb => {
            let b = need(buf, p, 8)?;
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            Ok((Value::F64(f64::from_be_bytes(a)), p + 8))
        }
        0xcc..=0xcf => {
            let w = 1usize << (tag - 0xcc);
            Ok((Value::Uint(be_uint(need(buf, p, w)?)), p + w))
        }
        0xd0..=0xd3 => {
            let w = 1usize << (tag - 0xd0);
            let raw = be_uint(need(buf, p, w)?);
            let shift = 64 - 8 * w;
            Ok((Value::Int(((raw << shift) as i64) >> shift), p + w))
        }
        0xd9..=0xdb => {
            let w = 1usize << (tag - 0xd9);
            let n = be_uint(need(buf, p, w)?) as usize;
            read_str(buf, p + w, n)
        }
        0xdc..=0xdd => {
            let w = if tag == 0xdc { 2 } else { 4 };
            let n = be_uint(need(buf, p, w)?) as usize;
            read_array(buf, p + w, n)
        }
        0xde..=0xdf => {
            let w = if tag == 0xde { 2 } else { 4 };
            let n = be_uint(need(buf, p, w)?) as usize;
            read_map(buf, p + w, n)
        }
        other => Err(Error(format!("unsupported tag 0x{other:02x} at {at}"))),
    }
}

fn read_str(buf: &[u8], at: usize, n: usize) -> Result<(Value, usize)> {
    let b = need(buf, at, n)?;
    let s = std::str::from_utf8(b).map_err(|e| Error(format!("bad utf8 at {at}: {e}")))?;
    Ok((Value::Str(s.to_owned()), at + n))
}

fn read_array(buf: &[u8], at: usize, n: usize) -> Result<(Value, usize)> {
    let mut v = Vec::with_capacity(n.min(1024));
    let mut p = at;
    for _ in 0..n {
        let (x, np) = read(buf, p)?;
        v.push(x);
        p = np;
    }
    Ok((Value::Array(v), p))
}

fn read_map(buf: &[u8], at: usize, n: usize) -> Result<(Value, usize)> {
    let mut v = Vec::with_capacity(n.min(1024));
    let mut p = at;
    for _ in 0..n {
        let (k, np) = read(buf, p)?;
        let (x, np2) = read(buf, np)?;
        v.push((k, x));
        p = np2;
    }
    Ok((Value::Map(v), p))
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m.iter().find_map(|(k, v)| match k {
                Value::Str(s) if s == key => Some(v),
                _ => None,
            }),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::Uint(u) => Some(u),
            Value::Int(i) if i >= 0 => Some(i as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixmap_of_scalars() {
        // {"a": 1, "b": "xy"}
        let buf = [0x82, 0xa1, b'a', 0x01, 0xa1, b'b', 0xa2, b'x', b'y'];
        let (v, end) = read(&buf, 0).expect("decode");
        assert_eq!(end, buf.len());
        assert_eq!(v.get("a").and_then(Value::as_u64), Some(1));
        assert_eq!(v.get("b").and_then(Value::as_str), Some("xy"));
    }

    #[test]
    fn wide_uints_and_arrays() {
        // [0xcd 0x0100 (256), 0xce 0x00010000 (65536)]
        let buf = [0x92, 0xcd, 0x01, 0x00, 0xce, 0x00, 0x01, 0x00, 0x00];
        let (v, end) = read(&buf, 0).expect("decode");
        assert_eq!(end, buf.len());
        let a = v.as_array().expect("array");
        assert_eq!(a[0].as_u64(), Some(256));
        assert_eq!(a[1].as_u64(), Some(65536));
    }

    #[test]
    fn negative_control_truncated_input_is_an_error() {
        // A fixarray of 2 with only one element present must NOT decode.
        let buf = [0x92, 0x01];
        assert!(read(&buf, 0).is_err());
    }
}
