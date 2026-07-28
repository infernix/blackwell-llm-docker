#!/usr/bin/env python3
"""Verify GLM required-tool structural-tag semantics in XGrammar."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata

import xgrammar as xgr
from xgrammar.builtin_structural_tag import get_model_structural_tag
from xgrammar.testing import _is_grammar_accept_string


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--expected-transformers-version")
    args = parser.parse_args()

    actual_version = metadata.version("xgrammar")
    assert actual_version == args.expected_version, (
        f"xgrammar version mismatch: {actual_version} != {args.expected_version}"
    )

    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        assert tokenizer_info.vocab_size > 0
        if args.expected_transformers_version:
            actual_transformers = metadata.version("transformers")
            assert actual_transformers == args.expected_transformers_version, (
                "transformers version mismatch: "
                f"{actual_transformers} != {args.expected_transformers_version}"
            )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {
                    "type": "object",
                    "properties": {"tz": {"type": "string"}},
                    "required": ["tz"],
                },
            },
        },
    ]
    structural_tag = get_model_structural_tag(
        "glm_4_7",
        tools=tools,
        tool_choice="required",
        reasoning=False,
    )
    grammar = xgr.Grammar.from_structural_tag(structural_tag)
    weather = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Prague</arg_value></tool_call>"
    )
    current_time = (
        "<tool_call>get_time<arg_key>tz</arg_key>"
        "<arg_value>UTC</arg_value></tool_call>"
    )

    # Required means at least one call, not exactly one call. A response may
    # terminate or continue normally after satisfying that requirement.
    assert not _is_grammar_accept_string(grammar, "")
    assert _is_grammar_accept_string(grammar, weather)
    assert _is_grammar_accept_string(grammar, weather + current_time)
    assert _is_grammar_accept_string(grammar, weather + " done")
    print(f"xgrammar GLM required-tool regression: PASS {actual_version}")


if __name__ == "__main__":
    main()
