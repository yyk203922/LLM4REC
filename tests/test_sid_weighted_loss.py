from __future__ import annotations

import torch
import torch.nn.functional as F

from llm4rec.sid_weighted_loss import (
    build_hierarchical_token_weights,
    collect_sid_token_ids_by_level,
    compute_weighted_causal_lm_loss,
)


class FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            "plain": 0,
            "<|video_begin|>": 1,
            "<s_a_7>": 2,
            "<s_b_9>": 3,
            "<s_c_4>": 4,
        }


def test_collects_sid_levels() -> None:
    assert collect_sid_token_ids_by_level(FakeTokenizer()) == {
        "s_a": [2],
        "s_b": [3],
        "s_c": [4],
    }


def test_builds_prefix_heavy_class_weights() -> None:
    weights = build_hierarchical_token_weights(
        FakeTokenizer(),
        vocab_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        level_weights={"default": 1, "domain": 1, "s_a": 5, "s_b": 3, "s_c": 1},
    )
    assert weights.tolist() == [1.0, 1.0, 5.0, 3.0, 1.0]


def test_weighted_loss_matches_cross_entropy() -> None:
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 5)
    labels = torch.tensor([[-100, 0, 2, 3, 4]])
    weights = torch.tensor([1.0, 1.0, 5.0, 3.0, 1.0])

    actual = compute_weighted_causal_lm_loss(logits, labels, weights)
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, 5),
        labels[:, 1:].reshape(-1),
        weight=weights,
    )
    torch.testing.assert_close(actual, expected)
