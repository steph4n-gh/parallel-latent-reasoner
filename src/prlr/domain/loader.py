"""MLX-native DataLoader and Dataset for PRLR domain reasoning.

Implements:
- PRLRDomainDataset with pretokenization, SentencePiece validation, and Rule 5 enforcement
- PRLRDomainDataLoader yielding DomainBatch (training) or EvaluationBatch (isolated inference)
- Right-padding with pad_token_id=0, float32 prompt_mask and target_mask
- Strict target_mask computation: 1.0 up to and including the first EOS token ({1, 107}), 0.0 on pad/post-EOS
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union

import mlx.core as mx

from prlr.domain.prompt_format import format_canonical_prompt, is_gemma4_tokenizer
from prlr.domain.schema import DomainSample, EvaluationInput
from prlr.manifest import Rule5ViolationError


@dataclass
class DomainBatch:
    """Padded MLX tensor batch formatted for PRLR training."""
    input_ids: mx.array         # (B, P_max) int32
    prompt_mask: mx.array       # (B, P_max) float32 (1.0 for real token, 0.0 for pad)
    prompt_lengths: mx.array    # (B,) int32
    target_ids: mx.array        # (B, T_max) int32
    target_mask: mx.array       # (B, T_max) float32 (1.0 for valid target token, 0.0 for pad/post-EOS)
    target_lengths: mx.array    # (B,) int32
    sample_ids: List[str]       # (B,) strings
    difficulties: mx.array      # (B,) int32
    num_steps: mx.array         # (B,) int32


@dataclass
class EvaluationBatch:
    """Isolated evaluation batch yielding zero ground truth or target tokens (Rule 1)."""
    input_ids: mx.array         # (B, P_max) int32
    prompt_mask: mx.array       # (B, P_max) float32
    prompt_lengths: mx.array    # (B,) int32
    sample_ids: List[str]       # (B,) strings
    prompts_text: List[str]     # (B,) strings


class PRLRDomainDataset:
    """Dataset container with pretokenization, indexing, and validation."""

    def __init__(
        self,
        samples: Sequence[DomainSample],
        tokenizer: Any,
        pad_token_id: int = 0,
        eos_token_ids: Optional[Sequence[int]] = None,
        max_prompt_len: int = 1024,
        max_target_len: int = 128,
        pretokenize: bool = True,
    ):
        if tokenizer is None:
            raise Rule5ViolationError(
                "PRLRDomainDataset requires a verified official tokenizer. "
                "Character-modulo tokenization (ord(c) % vocab) is strictly prohibited under Rule 5."
            )
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self._is_gemma4 = is_gemma4_tokenizer(tokenizer)

        if eos_token_ids is None:
            self.eos_token_ids = {1, 106} if self._is_gemma4 else {1, 107}
        else:
            self.eos_token_ids = set(eos_token_ids)
            if self._is_gemma4 and 107 in self.eos_token_ids:
                self.eos_token_ids.discard(107)
                self.eos_token_ids.add(106)

        self.max_prompt_len = max_prompt_len
        self.max_target_len = max_target_len

        self._tokenized_prompts: Optional[List[List[int]]] = None
        self._tokenized_targets: Optional[List[List[int]]] = None
        if pretokenize:
            self._pretokenize_all()

    def _tokenize_prompt(self, text: str) -> List[int]:
        canonical_text = format_canonical_prompt(text, self.tokenizer, is_gemma4=self._is_gemma4)
        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(canonical_text, add_special_tokens=False)
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
        elif hasattr(self.tokenizer, "encode_as_ids"):
            tokens = self.tokenizer.encode_as_ids(canonical_text)
        else:
            tokens = list(self.tokenizer(canonical_text))

        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is None and hasattr(self.tokenizer, "bos_id"):
            bos_id = self.tokenizer.bos_id()
        if bos_id is not None and (len(tokens) == 0 or tokens[0] != bos_id):
            tokens = [bos_id] + tokens

        if len(tokens) > self.max_prompt_len:
            tokens = tokens[: self.max_prompt_len]
        return tokens

    def _tokenize_target(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
        elif hasattr(self.tokenizer, "encode_as_ids"):
            tokens = self.tokenizer.encode_as_ids(text)
        else:
            tokens = list(self.tokenizer(text))

        term_id = 106 if self._is_gemma4 else 107
        # Ensure termination with EOS / turn-end
        if not any(tok in self.eos_token_ids for tok in tokens):
            tokens.append(term_id)

        # Truncation must never drop the turn-ending token
        if len(tokens) > self.max_target_len:
            last_tok = tokens[-1] if tokens[-1] in self.eos_token_ids else term_id
            tokens = tokens[: self.max_target_len - 1] + [last_tok]
        return tokens

    def _pretokenize_all(self):
        self._tokenized_prompts = [
            self._tokenize_prompt(s.prompt) for s in self.samples
        ]
        self._tokenized_targets = [
            self._tokenize_target(s.target_solution) for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[DomainSample, List[int], List[int]]:
        sample = self.samples[idx]
        p_toks = (
            self._tokenized_prompts[idx]
            if self._tokenized_prompts is not None
            else self._tokenize_prompt(sample.prompt)
        )
        t_toks = (
            self._tokenized_targets[idx]
            if self._tokenized_targets is not None
            else self._tokenize_target(sample.target_solution)
        )
        return sample, p_toks, t_toks


class PRLRDomainDataLoader:
    """Padded batch iterator for MLX training and isolated evaluation."""

    def __init__(
        self,
        dataset: PRLRDomainDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        mode: Literal["train", "eval"] = "train",
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.mode = mode
        self.drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[Union[DomainBatch, EvaluationBatch]]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            import random
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_indices = indices[start:end]
            items = [self.dataset[i] for i in batch_indices]

            if self.mode == "eval":
                yield self._collate_eval(items)
            else:
                yield self._collate_train(items)

    def _collate_train(
        self, items: List[Tuple[DomainSample, List[int], List[int]]]
    ) -> DomainBatch:
        prompts = [it[1] for it in items]
        targets = [it[2] for it in items]

        max_p = max(len(p) for p in prompts)
        max_t = max(len(t) for t in targets)

        padded_prompts = []
        prompt_masks = []
        prompt_lens = []
        for p in prompts:
            l = len(p)
            prompt_lens.append(l)
            padded_prompts.append(p + [self.dataset.pad_token_id] * (max_p - l))
            prompt_masks.append([1.0] * l + [0.0] * (max_p - l))

        padded_targets = []
        target_masks = []
        target_lens = []
        for t in targets:
            l = len(t)
            target_lens.append(l)
            padded_targets.append(t + [self.dataset.pad_token_id] * (max_t - l))

            t_mask = []
            seen_eos = False
            for tok in t:
                if not seen_eos and tok != self.dataset.pad_token_id:
                    t_mask.append(1.0)
                    if tok in self.dataset.eos_token_ids:
                        seen_eos = True
                else:
                    t_mask.append(0.0)
            t_mask.extend([0.0] * (max_t - l))
            target_masks.append(t_mask)

        return DomainBatch(
            input_ids=mx.array(padded_prompts, dtype=mx.int32),
            prompt_mask=mx.array(prompt_masks, dtype=mx.float32),
            prompt_lengths=mx.array(prompt_lens, dtype=mx.int32),
            target_ids=mx.array(padded_targets, dtype=mx.int32),
            target_mask=mx.array(target_masks, dtype=mx.float32),
            target_lengths=mx.array(target_lens, dtype=mx.int32),
            sample_ids=[it[0].id for it in items],
            difficulties=mx.array([it[0].difficulty for it in items], dtype=mx.int32),
            num_steps=mx.array([it[0].num_steps for it in items], dtype=mx.int32),
        )

    def _collate_eval(
        self, items: List[Tuple[DomainSample, List[int], List[int]]]
    ) -> EvaluationBatch:
        prompts = [it[1] for it in items]
        max_p = max(len(p) for p in prompts)

        padded_prompts = []
        prompt_masks = []
        prompt_lens = []
        for p in prompts:
            l = len(p)
            prompt_lens.append(l)
            padded_prompts.append(p + [self.dataset.pad_token_id] * (max_p - l))
            prompt_masks.append([1.0] * l + [0.0] * (max_p - l))

        return EvaluationBatch(
            input_ids=mx.array(padded_prompts, dtype=mx.int32),
            prompt_mask=mx.array(prompt_masks, dtype=mx.float32),
            prompt_lengths=mx.array(prompt_lens, dtype=mx.int32),
            sample_ids=[it[0].id for it in items],
            prompts_text=[it[0].prompt for it in items],
        )


__all__ = [
    "DomainBatch",
    "EvaluationBatch",
    "PRLRDomainDataset",
    "PRLRDomainDataLoader",
]
