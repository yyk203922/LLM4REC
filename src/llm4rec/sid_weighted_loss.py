"""Prefix-aware token weighting for three-level semantic IDs.

The ordinary language-model loss is preserved for every supervised token.
Only target tokens matching ``<s_a_*>``, ``<s_b_*>`` or ``<s_c_*>`` receive
level-specific weights. Earlier levels can therefore receive stronger
supervision without discarding CoT, world-knowledge, or user-understanding
targets.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


SID_LEVEL_RE = re.compile(r"^<s_([abc])_[^<>]+>$")
DOMAIN_TOKENS = (
    "<|video_begin|>",
    "<|prod_begin|>",
    "<|ad_begin|>",
    "<|living_begin|>",
    "<|sid_begin|>",
)


def collect_sid_token_ids_by_level(tokenizer: object) -> dict[str, list[int]]:
    """Collect SID token IDs from a tokenizer vocabulary."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        raise TypeError("tokenizer does not expose get_vocab()")

    grouped: dict[str, list[int]] = {"s_a": [], "s_b": [], "s_c": []}
    for token, token_id in get_vocab().items():
        match = SID_LEVEL_RE.fullmatch(str(token))
        if match:
            grouped[f"s_{match.group(1)}"].append(int(token_id))
    return {level: sorted(set(token_ids)) for level, token_ids in grouped.items()}


def collect_domain_token_ids(tokenizer: object) -> list[int]:
    """Collect domain-prefix token IDs that exist in the vocabulary."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        raise TypeError("tokenizer does not expose get_vocab()")
    vocab = get_vocab()
    return sorted({int(vocab[token]) for token in DOMAIN_TOKENS if token in vocab})


def normalize_level_weights(raw: Mapping[str, float] | float) -> dict[str, float]:
    """Normalize scalar or per-level SID weights to one configuration."""
    if isinstance(raw, Mapping):
        default = float(raw.get("default", 1.0))
        weights = {
            "default": default,
            "domain": float(raw.get("domain", default)),
            "s_a": float(raw.get("s_a", raw.get("a", default))),
            "s_b": float(raw.get("s_b", raw.get("b", default))),
            "s_c": float(raw.get("s_c", raw.get("c", default))),
        }
    else:
        sid_weight = float(raw)
        weights = {
            "default": 1.0,
            "domain": 1.0,
            "s_a": sid_weight,
            "s_b": sid_weight,
            "s_c": sid_weight,
        }
    if any(value <= 0 for value in weights.values()):
        raise ValueError(f"all token weights must be positive: {weights}")
    return weights


def _valid_ids(token_ids: Iterable[int], vocab_size: int) -> list[int]:
    return sorted({int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab_size})


def build_hierarchical_token_weights(
    tokenizer: object,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
    level_weights: Mapping[str, float] | float,
) -> torch.Tensor:
    """Build a vocabulary-sized class-weight vector for weighted CE."""
    weights = normalize_level_weights(level_weights)
    class_weights = torch.full(
        (vocab_size,), weights["default"], dtype=dtype, device=device
    )

    domain_ids = _valid_ids(collect_domain_token_ids(tokenizer), vocab_size)
    if domain_ids:
        class_weights[torch.tensor(domain_ids, device=device)] = weights["domain"]

    grouped = collect_sid_token_ids_by_level(tokenizer)
    if not any(grouped.values()):
        raise ValueError("no <s_a_*>/<s_b_*>/<s_c_*> tokens found in the tokenizer")
    for level in ("s_a", "s_b", "s_c"):
        token_ids = _valid_ids(grouped[level], vocab_size)
        if token_ids:
            class_weights[torch.tensor(token_ids, device=device)] = weights[level]
    return class_weights


def compute_weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Reference implementation of weighted next-token cross entropy."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="mean",
    )


def compute_fused_causal_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
    class_weights: torch.Tensor,
    ignore_index: int = -100,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Memory-efficient weighted CE for long packed sequences.

    Liger's fused linear cross entropy avoids materializing the complete
    ``batch x sequence x vocabulary`` logits tensor. Its weighted mean is
    ``sum(weight[target] * CE) / sum(weight[target])`` over valid labels.
    """
    try:
        from liger_kernel.transformers.functional import (
            liger_fused_linear_cross_entropy,
        )
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise RuntimeError(
            "Long-context training requires liger-kernel. Install it or use "
            "compute_weighted_causal_lm_loss with a shorter cutoff."
        ) from exc

    shift_hidden = hidden_states[..., :-1, :].contiguous().view(
        -1, hidden_states.size(-1)
    )
    shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_hidden.device)
    return liger_fused_linear_cross_entropy(
        shift_hidden,
        lm_head_weight,
        shift_labels,
        bias=bias,
        ce_weight=class_weights,
        ignore_index=ignore_index,
        reduction="mean",
    )
