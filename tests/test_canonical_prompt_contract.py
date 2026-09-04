"""Unit tests verifying the Gemma 4 canonical prompt, termination, and decoder contract.

Verifies:
1. Training (PRLRDomainDataset) and evaluation use the identical canonical prompt format.
2. The thought channel is properly closed: `<|channel>thought\n<channel|>`.
3. Target tokenization terminates in token 106 (<turn|>), never 107 (\n), for Gemma 4.
4. Target truncation preserves the termination token 106.
5. Newline (107) never halts generation in GemmaCausalPrefixDecoder for Gemma 4.
6. ProceduralVerifier handles both raw JSON and markdown-fenced JSON cleanly.
"""

import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer

from prlr.domain.loader import PRLRDomainDataset
from prlr.domain.prompt_format import (
    extract_user_body,
    format_canonical_prompt,
    is_gemma4_tokenizer,
)
from prlr.domain.schema import DomainSample
from prlr.domain.solver_lane import ProceduralVerifier
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.manifest import ModelManifest


@pytest.fixture(scope="module")
def gemma4_tokenizer():
    manifest = ModelManifest.gemma_4_12b_it()
    tok_path = Path(manifest.tokenizer_path)
    tok_dir = tok_path.parent if tok_path.is_file() else tok_path
    if not tok_dir.exists():
        pytest.skip(f"Gemma 4 tokenizer not found at {tok_dir}")
    return AutoTokenizer.from_pretrained(str(tok_dir), fix_mistral_regex=True)


def test_extract_user_body():
    # Raw prompt
    assert extract_user_body("Hello world") == "Hello world"

    # Gemma 2 legacy turn tags
    legacy = "<start_of_turn>user\nPlan the route for goal A.<end_of_turn>\n<start_of_turn>model\n"
    assert extract_user_body(legacy) == "Plan the route for goal A."

    # Gemma 4 turn tags
    g4 = "<bos><|turn>user\nPlan the route.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    assert extract_user_body(g4) == "Plan the route."


def test_closed_thought_channel(gemma4_tokenizer):
    raw = "You are an execution planner."
    formatted = format_canonical_prompt(raw, gemma4_tokenizer, is_gemma4=True)

    assert formatted.startswith("<bos><|turn>user\n")
    assert "<|channel>thought\n<channel|>" in formatted
    assert formatted.endswith("<channel|>")
    assert "<turn|>" in formatted


def test_identical_tokenization_train_and_eval(gemma4_tokenizer):
    legacy_prompt = (
        "<start_of_turn>user\n"
        "You are an autonomous execution planner. Given the available tool registry, "
        "determine the minimal valid sequence.\n"
        "Available Tools:\n- parser: req [a], prod [b]\n- solver: req [b], prod [c]\n\n"
        "Initial State: [a]\nTarget Goal: c\n<end_of_turn>\n<start_of_turn>model\n"
    )

    # 1. Format directly via format_canonical_prompt
    canonical = format_canonical_prompt(legacy_prompt, gemma4_tokenizer, is_gemma4=True)
    eval_tokens = gemma4_tokenizer.encode(canonical, add_special_tokens=False)

    # 2. Tokenize via PRLRDomainDataset
    sample = DomainSample(
        id="sample_001",
        split="train",
        domain="api_workflow",
        difficulty=1,
        num_steps=2,
        prompt=legacy_prompt,
        target_solution='{"route": ["parser", "solver"], "terminal": "solver"}',
        ground_truth='{"route": ["parser", "solver"], "terminal": "solver"}',
        verifier_type="bfs_oracle",
        verifier_config={"expected_route": ["parser", "solver"], "terminal_tool": "solver"},
        seed=42,
    )
    dataset = PRLRDomainDataset(
        samples=[sample],
        tokenizer=gemma4_tokenizer,
        pad_token_id=0,
        pretokenize=True,
    )
    _, train_prompt_tokens, _ = dataset[0]

    # Must be 100% bitwise/token identical
    assert train_prompt_tokens == eval_tokens, (
        f"Mismatch between train tokens ({len(train_prompt_tokens)}) "
        f"and eval tokens ({len(eval_tokens)})"
    )
    assert train_prompt_tokens[0] == 2  # BOS
    assert train_prompt_tokens[-1] == 101  # <channel|>


def test_target_tokenization_terminates_in_106(gemma4_tokenizer):
    sample = DomainSample(
        id="sample_001",
        split="train",
        domain="api_workflow",
        difficulty=1,
        num_steps=2,
        prompt="hello",
        target_solution='{"route": ["schema_parser", "auth_validator"], "terminal": "auth_validator"}',
        ground_truth='{"route": ["schema_parser", "auth_validator"], "terminal": "auth_validator"}',
        verifier_type="bfs_oracle",
        verifier_config={"expected_route": ["schema_parser", "auth_validator"], "terminal_tool": "auth_validator"},
        seed=42,
    )
    dataset = PRLRDomainDataset(
        samples=[sample],
        tokenizer=gemma4_tokenizer,
        pad_token_id=0,
        pretokenize=True,
    )
    _, _, target_tokens = dataset[0]

    # Target must end in token 106 (<turn|>), NOT 107 (\n)
    assert target_tokens[-1] == 106, f"Target ends in {target_tokens[-1]}, expected 106 (<turn|>)"
    assert 106 in dataset.eos_token_ids
    assert 107 not in dataset.eos_token_ids


def test_target_truncation_preserves_106(gemma4_tokenizer):
    sample = DomainSample(
        id="sample_001",
        split="train",
        domain="api_workflow",
        difficulty=1,
        num_steps=4,
        prompt="hello",
        target_solution='{"route": ["schema_parser", "auth_validator", "permission_checker", "db_reader"], "terminal": "db_reader"}',
        ground_truth='{"route": ["schema_parser", "auth_validator", "permission_checker", "db_reader"], "terminal": "db_reader"}',
        verifier_type="bfs_oracle",
        verifier_config={"expected_route": ["schema_parser", "auth_validator"], "terminal_tool": "auth_validator"},
        seed=42,
    )
    # Force severe truncation with max_target_len=8
    dataset = PRLRDomainDataset(
        samples=[sample],
        tokenizer=gemma4_tokenizer,
        pad_token_id=0,
        max_target_len=8,
        pretokenize=True,
    )
    _, _, target_tokens = dataset[0]

    assert len(target_tokens) == 8
    assert target_tokens[-1] == 106, "Truncation dropped termination token 106"


def test_newline_never_halts_gemma4_decoder():
    class MockBackbone:
        manifest = ModelManifest.gemma_4_12b_it()

    backbone = MockBackbone()
    # Even if caller erroneously supplies (1, 107):
    decoder = GemmaCausalPrefixDecoder(backbone=backbone, eos_token_ids=(1, 107))

    assert 107 not in decoder.eos_token_ids, "Token 107 (\\n) must not be in Gemma 4 eos_token_ids"
    assert 106 in decoder.eos_token_ids, "Token 106 (<turn|>) must be in Gemma 4 eos_token_ids"


def test_procedural_verifier_json_and_markdown():
    # 1. Raw JSON
    raw_json = '{"route": ["a", "b"], "terminal": "b"}<turn|>'
    v1 = ProceduralVerifier.verify(raw_json, ("a", "b"))
    assert v1["is_valid"] is True
    assert v1["exact_match"] is True

    # 2. Markdown fenced JSON with turn token
    md_json = '```json\n{\n  "route": [\n    "a",\n    "b"\n  ],\n  "terminal": "b"\n}\n```<turn|>'
    v2 = ProceduralVerifier.verify(md_json, ("a", "b"))
    assert v2["is_valid"] is True
    assert v2["exact_match"] is True
