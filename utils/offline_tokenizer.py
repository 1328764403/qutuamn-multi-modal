"""
Offline-safe tokenizer used when HuggingFace assets are unavailable locally.

Goal: NEVER download anything from the internet during training/validation.
"""

from __future__ import annotations

from pathlib import Path
import re
import torch


class SimpleTokenizer:
    """
    A tiny tokenizer that turns text into a fixed-length `input_ids` tensor.

    - No vocab files
    - No network access
    - Deterministic across runs
    """

    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = int(vocab_size)

    def __call__(
        self,
        text: str,
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        max_length = int(max_length)
        # Basic tokenization: words + digits
        tokens = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", (text or "").lower())
        ids = [self._token_to_id(t) for t in tokens]
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length" and len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = (input_ids != 0).to(torch.long)

        if return_tensors != "pt":
            # Keep behavior predictable: this project only uses torch tensors
            raise ValueError("SimpleTokenizer only supports return_tensors='pt'")

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _token_to_id(self, token: str) -> int:
        # Stable hashing (avoid Python's randomized hash)
        h = 2166136261
        for ch in token.encode("utf-8", errors="ignore"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h % self.vocab_size)


def load_local_hf_tokenizer(local_dir: Path) -> object:
    """
    Best-effort: load a tokenizer from local HuggingFace folder only.
    Falls back to `SimpleTokenizer` if anything is missing.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore

        if local_dir.exists():
            return AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    except Exception:
        pass

    return SimpleTokenizer()

