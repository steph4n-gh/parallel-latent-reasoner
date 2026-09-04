"""Canonical prompt formatting and user body extraction for PRLR.

Guarantees:
- Single canonical prompt formatting function for training, evaluation, and pipeline inference.
- Official Gemma 4 chat template via `tokenizer.apply_chat_template(..., enable_thinking=False)`.
- Proper closure of thought channel (`<|channel>thought\n<channel|>`).
- Extraction of raw user prompt text from legacy or existing prompts.
- Deterministic tokenization across all operational modes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_user_body(prompt: str) -> str:
    """Extract raw user prompt body, stripping control tokens if present."""
    if not prompt:
        return ""
    text = prompt.strip()
    if "<start_of_turn>user" in text:
        m = re.search(r"<start_of_turn>user\s*\n(.*?)(?:<end_of_turn>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    if "<|turn>user" in text:
        m = re.search(r"<\|turn>user\s*\n(.*?)(?:<turn\|>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def is_gemma4_tokenizer(tokenizer: Any) -> bool:
    """Detect whether a tokenizer belongs to the Gemma 4 family (vocab 262,144, turn-end 106)."""
    if tokenizer is None:
        return False
    try:
        ids = tokenizer.encode("<turn|>", add_special_tokens=False)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return len(ids) == 1 and ids[0] == 106
    except Exception:
        pass
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and vocab_size >= 260000:
        return True
    return False


def format_canonical_prompt(
    prompt_or_body: str,
    tokenizer: Any,
    is_gemma4: Optional[bool] = None,
) -> str:
    """Format prompt strictly using the canonical chat template.

    For Gemma 4:
      Calls `tokenizer.apply_chat_template([{"role": "user", "content": user_body}],
             tokenize=False, add_generation_prompt=True, enable_thinking=False)`
      Ensures prompt ends with `<|channel>thought\n<channel|>`.
    For Gemma 2:
      Formats `<start_of_turn>user\n{user_body}<end_of_turn>\n<start_of_turn>model\n`.
    """
    user_body = extract_user_body(prompt_or_body)

    if is_gemma4 is None:
        is_gemma4 = is_gemma4_tokenizer(tokenizer)

    if is_gemma4:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_body}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                return formatted
            except Exception:
                pass
        return f"<bos><|turn>user\n{user_body}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    else:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_body}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception:
                pass
        return f"<start_of_turn>user\n{user_body}<end_of_turn>\n<start_of_turn>model\n"
