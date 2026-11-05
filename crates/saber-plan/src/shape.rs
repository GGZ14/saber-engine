//! Shape keys — the `{name: value}` geometry a route is selected on.
//!
//! Modelled on the reference's `select(op, **geometry)`: a shape key set is what constraints are tested
//! against and what the autotune key is built from. Values are either numbers or short strings
//! (dtype names), and the map is ordered so that every serialization of it is canonical — the plan
//! hash depends on that.

use serde::{Deserialize, Serialize};
use std::collections::btree_map::Iter;
use std::collections::BTreeMap;
use std::fmt;

/// One geometry value: a number, or a short name such as a dtype.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShapeValue {
    /// A numeric parameter (`m`, `n`, `k`, `head_dim`, `causal` as 0/1).
    Num(i64),
    /// A named parameter (`w_dtype = "mxfp4"`).
    Str(String),
}

impl ShapeValue {
    /// The number, when this value is numeric.
    pub fn as_num(&self) -> Option<i64> {
        match self {
            ShapeValue::Num(v) => Some(*v),
            ShapeValue::Str(_) => None,
        }
    }

    /// The string, when this value is a name.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ShapeValue::Num(_) => None,
            ShapeValue::Str(s) => Some(s),
        }
    }
}

impl fmt::Display for ShapeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeValue::Num(v) => write!(f, "{v}"),
            ShapeValue::Str(s) => f.write_str(s),
        }
    }
}

impl From<i64> for ShapeValue {
    fn from(v: i64) -> Self {
        ShapeValue::Num(v)
    }
}

impl From<u32> for ShapeValue {
    fn from(v: u32) -> Self {
        ShapeValue::Num(i64::from(v))
    }
}

impl From<&str> for ShapeValue {
    fn from(v: &str) -> Self {
        ShapeValue::Str(v.to_string())
    }
}

impl From<String> for ShapeValue {
    fn from(v: String) -> Self {
        ShapeValue::Str(v)
    }
}

/// An ordered set of geometry parameters, e.g. `m=40 n=5120 k=5120 w_dtype=mxfp4`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ShapeKeys(BTreeMap<String, ShapeValue>);

impl ShapeKeys {
    /// An empty key set.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Builder-style insert.
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<ShapeValue>) -> Self {
        self.insert(key, value);
        self
    }

    /// Inserts or replaces one key.
    pub fn insert(&mut self, key: &str, value: impl Into<ShapeValue>) {
        self.0.insert(key.to_string(), value.into());
    }

    /// The value under `key`, if any.
    pub fn get(&self, key: &str) -> Option<&ShapeValue> {
        self.0.get(key)
    }

    /// The numeric value under `key`, if it is present and numeric.
    pub fn num(&self, key: &str) -> Option<i64> {
        self.0.get(key).and_then(ShapeValue::as_num)
    }

    /// `m`, the row count. Every autotune key must carry it: the tile and split-K optimum flips
    /// per shape *and per M*, so a record keyed without M is a record for the wrong call.
    pub fn m(&self) -> Option<i64> {
        self.num("m")
    }

    /// Iterates the keys in canonical (sorted) order.
    pub fn iter(&self) -> Iter<'_, String, ShapeValue> {
        self.0.iter()
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the key set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> IntoIterator for &'a ShapeKeys {
    type Item = (&'a String, &'a ShapeValue);
    type IntoIter = Iter<'a, String, ShapeValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl fmt::Display for ShapeKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (k, v) in &self.0 {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "{k}={v}")?;
        }
        Ok(())
    }
}
