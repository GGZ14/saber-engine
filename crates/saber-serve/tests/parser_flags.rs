//! `saber serve`'s parser flags at the reference server parity (V9): every flag, every refusal, and the reference server's text.
//!
//! The texts are the reference 0.27.1's, read out of prod's image (`the reference image`):
//! `upstream/entrypoints/openai/cli_args.py::validate_parsed_serve_args` (auto tool choice without a
//! parser) and `upstream/entrypoints/openai/api_server.py::validate_api_server_args` (an unknown name,
//! listing `list_registered()`); the lists themselves are `COMPAT_TOOL_PARSERS` /
//! `COMPAT_REASONING_PARSERS`, copied from the image the same day.

use saber_serve::state::{
    ParserDefaults, ParserFlags, ReasoningParserKind, ResolvedParsers, ToolParserKind,
    COMPAT_REASONING_PARSERS, COMPAT_TOOL_PARSERS,
};

fn flags(auto: Option<bool>, tool: Option<&str>, reasoning: Option<&str>) -> ParserFlags {
    ParserFlags {
        enable_auto_tool_choice: auto,
        tool_call_parser: tool.map(str::to_owned),
        reasoning_parser: reasoning.map(str::to_owned),
        defaults: ParserDefaults::Prod,
    }
}

#[test]
fn no_flag_is_prods_profile_and_so_is_prods_own_argv() {
    let absent = flags(None, None, None).resolve().expect("resolves");
    assert!(absent.enable_auto_tool_choice);
    assert_eq!(absent.tool_parser, Some(ToolParserKind::Qwen3Coder));
    assert_eq!(absent.reasoning_parser, Some(ReasoningParserKind::Qwen3));
    // `serve-mxfp4.sh:1071`, verbatim: `--enable-auto-tool-choice --tool-call-parser qwen3_coder
    // --reasoning-parser qwen3`.
    let argv = flags(Some(true), Some("qwen3_coder"), Some("qwen3"))
        .resolve()
        .expect("resolves");
    assert_eq!(argv, absent);
    assert_eq!(absent, ResolvedParsers::prod());
    assert_eq!(
        absent.banner(),
        "parsers        auto tool choice ON, tool call parser qwen3_coder = \
         Qwen3EngineToolParser, reasoning parser qwen3 = Qwen3ParserReasoningAdapter"
    );
}

#[test]
fn qwen3_xml_and_mimo_are_the_one_parser_and_the_boot_line_says_so() {
    for alias in ["qwen3_xml", "mimo"] {
        let r = flags(Some(true), Some(alias), Some("mimo"))
            .resolve()
            .expect("resolves");
        assert_eq!(r.tool_parser, Some(ToolParserKind::Qwen3Coder), "{alias}");
        assert_eq!(r.reasoning_parser, Some(ReasoningParserKind::Qwen3));
        let banner = r.banner();
        assert!(
            banner.contains(&format!(
                "{alias} = Qwen3EngineToolParser -- the same parser as qwen3_coder in prod's image"
            )),
            "{banner}"
        );
        assert!(
            banner.contains("mimo = Qwen3ParserReasoningAdapter -- the same parser as qwen3"),
            "{banner}"
        );
    }
}

#[test]
fn an_unknown_tool_parser_gets_the_reference_error_text() {
    let e = flags(Some(true), Some("bogus"), None)
        .resolve()
        .unwrap_err();
    assert_eq!(
        e,
        format!(
            "invalid tool call parser: bogus (chose from {{ {} }})",
            COMPAT_TOOL_PARSERS.join(",")
        )
    );
    assert!(e.starts_with("invalid tool call parser: bogus (chose from { apertus,cohere_command3,"));
    assert!(e.ends_with(",step3,step3p5,xlam })"));
}

#[test]
fn an_unknown_reasoning_parser_gets_the_reference_error_text() {
    let e = flags(None, None, Some("bogus")).resolve().unwrap_err();
    assert_eq!(
        e,
        format!(
            "invalid reasoning parser: bogus (chose from {{ {} }})",
            COMPAT_REASONING_PARSERS.join(",")
        )
    );
}

#[test]
fn a_registered_parser_saber_does_not_implement_is_refused_by_name() {
    let e = flags(Some(true), Some("hermes"), None)
        .resolve()
        .unwrap_err();
    assert!(
        e.starts_with(
            "--tool-call-parser hermes: registered in the reference 0.27.1 but not implemented"
        ),
        "{e}"
    );
    let e = flags(None, None, Some("deepseek_r1"))
        .resolve()
        .unwrap_err();
    assert!(
        e.starts_with(
            "--reasoning-parser deepseek_r1: registered in the reference 0.27.1 but not implemented"
        ),
        "{e}"
    );
}

#[test]
fn auto_tool_choice_without_a_parser_is_the_reference_type_error() {
    let compat = ParserFlags {
        enable_auto_tool_choice: Some(true),
        defaults: ParserDefaults::Compat,
        ..ParserFlags::default()
    };
    assert_eq!(
        compat.resolve().unwrap_err(),
        "Error: --enable-auto-tool-choice requires --tool-call-parser"
    );
    // `--tool-call-parser ""` is no parser either.
    assert_eq!(
        flags(Some(true), Some(""), None).resolve().unwrap_err(),
        "Error: --enable-auto-tool-choice requires --tool-call-parser"
    );
}

#[test]
fn a_tool_parser_without_auto_tool_choice_is_inert_and_never_looked_up() {
    // The reference server's `ParserManager.get_tool_parser` returns before the lookup when auto tool choice is
    // off, and `validate_api_server_args` checks the name only when it is on: even a name the reference server does
    // not know is accepted. The boot line says it is inert.
    for name in ["qwen3_coder", "bogus", "hermes"] {
        let r = flags(Some(false), Some(name), Some("qwen3"))
            .resolve()
            .expect("accepted, as the reference server accepts it");
        assert!(!r.enable_auto_tool_choice);
        assert_eq!(r.tool_parser, None, "{name}");
        assert!(
            r.banner().contains(&format!("{name} named but INERT")),
            "{}",
            r.banner()
        );
    }
}

#[test]
fn an_empty_reasoning_parser_is_the_reference_default_none() {
    let r = flags(None, None, Some("")).resolve().expect("resolves");
    assert_eq!(r.reasoning_parser, None);
    assert!(r
        .banner()
        .contains("reasoning parser none (the think block stays in `content`)"));
}

#[test]
fn compat_defaults_are_everything_off() {
    let r = ParserFlags {
        defaults: ParserDefaults::Compat,
        ..ParserFlags::default()
    }
    .resolve()
    .expect("resolves");
    assert!(!r.enable_auto_tool_choice);
    assert_eq!(r.tool_parser, None);
    assert_eq!(r.reasoning_parser, None);
    // ...and under the reference server's defaults the prod argv still reads as prod's profile.
    let prod = ParserFlags {
        enable_auto_tool_choice: Some(true),
        tool_call_parser: Some("qwen3_coder".to_owned()),
        reasoning_parser: Some("qwen3".to_owned()),
        defaults: ParserDefaults::Compat,
    }
    .resolve()
    .expect("resolves");
    assert_eq!(prod, ResolvedParsers::prod());
}

/// THE CONTROL (structural rule 8): the lists the error text is built from are prod's, so a name
/// prod registers is never "invalid" and Saber's own names are all registered in prod.
#[test]
fn every_name_saber_accepts_is_one_prod_registers() {
    for name in ["qwen3_coder", "qwen3_xml", "mimo"] {
        assert!(COMPAT_TOOL_PARSERS.contains(&name), "{name}");
        assert!(ToolParserKind::from_name(name).is_some());
    }
    for name in ["qwen3", "mimo"] {
        assert!(COMPAT_REASONING_PARSERS.contains(&name), "{name}");
        assert!(ReasoningParserKind::from_name(name).is_some());
    }
    assert_eq!(COMPAT_TOOL_PARSERS.len(), 45);
    assert_eq!(COMPAT_REASONING_PARSERS.len(), 29);
    let mut sorted = COMPAT_TOOL_PARSERS.to_vec();
    sorted.sort_unstable();
    assert_eq!(
        sorted, COMPAT_TOOL_PARSERS,
        "the reference server lists them sorted"
    );
}
