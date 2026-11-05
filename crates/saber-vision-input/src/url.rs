//! `image_url.url` -> bytes, as the reference server's `MediaConnector.load_from_url` classifies it:
//!
//! | form | prod (the reference 0.27.1 defaults) | Saber |
//! |---|---|---|
//! | `data:<type>;base64,<b64>` | decoded (`pybase64`, `validate=True`) | decoded, prod's refusal texts |
//! | `data:` without `;base64` | 501 `Only base64 data URLs are supported for now.` | same |
//! | `http://`, `https://` | **fetched** (no `--allowed-media-domains` by default) | **refused by name** ([`InputError::RemoteFetch`]) |
//! | `file://` | 500 `Cannot load local files without --allowed-local-media-path` (default `""`) | same |
//! | anything else | 400 `The URL must be either a HTTP, data or file URL.` | same |

use base64::Engine as _;

use crate::{pyfmt, InputError};

/// Load one URL's bytes.
///
/// # Errors
/// See the module table.
pub fn load(url: &str) -> Result<Vec<u8>, InputError> {
    if url.len() >= 5 && url.as_bytes()[..5].eq_ignore_ascii_case(b"data:") {
        return data_url(url);
    }
    // urllib3's `parse_url`: the scheme is what precedes "://" (lower-cased).
    let scheme = url
        .split_once("://")
        .map(|(s, _)| s.to_ascii_lowercase())
        .unwrap_or_default();
    if scheme.starts_with("http") {
        return Err(InputError::RemoteFetch);
    }
    if scheme == "file" {
        return Err(InputError::FileUrl);
    }
    Err(InputError::UnknownScheme)
}

/// RFC 2397, the reference server's `_load_data_url`: `data_spec, sep, data = url[5:].partition(",")`,
/// `media_type, sep, encoding = data_spec.rpartition(";")`.
fn data_url(url: &str) -> Result<Vec<u8>, InputError> {
    let rest = &url[5..];
    let Some((spec, data)) = rest.split_once(',') else {
        let head: String = url.chars().take(32).collect();
        return Err(InputError::DataUrlNoComma(format!(
            "Invalid data URL {}: missing ',' separator.",
            pyfmt::str_repr(&head)
        )));
    };
    match spec.rsplit_once(';') {
        Some((_, "base64")) => {}
        _ => return Err(InputError::DataUrlNotBase64),
    }
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| InputError::BadBase64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prods_texts() {
        assert_eq!(load("data:image/png;base64,aGk=").unwrap(), b"hi");
        assert_eq!(load("DATA:image/png;base64,aGk=").unwrap(), b"hi");
        assert_eq!(
            load("data:image/png;base64,iVBOR!!").unwrap_err(),
            InputError::BadBase64
        );
        assert_eq!(
            load("data:image/png;base64,aGk").unwrap_err(),
            InputError::BadBase64
        );
        assert_eq!(
            load("data:image/png,abc").unwrap_err(),
            InputError::DataUrlNotBase64
        );
        assert_eq!(
            load("data:image/png;base64").unwrap_err().to_string(),
            "Invalid data URL 'data:image/png;base64': missing ',' separator."
        );
        assert_eq!(load("file:///etc/hosts").unwrap_err(), InputError::FileUrl);
        assert_eq!(
            load("ftp://example.com/a.png").unwrap_err(),
            InputError::UnknownScheme
        );
        assert_eq!(
            load("https://example.com/a.png").unwrap_err(),
            InputError::RemoteFetch
        );
        assert_eq!(
            load("example.com/a.png").unwrap_err(),
            InputError::UnknownScheme
        );
    }
}
