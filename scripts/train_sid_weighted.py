#!/usr/bin/env python3
"""Run LLaMA-Factory SFT with hierarchical SID token weighting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm4rec.sid_weighted_loss import (  # noqa: E402
    build_hierarchical_token_weights,
    compute_fused_causal_lm_loss,
    normalize_level_weights,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return config


def unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the causal LM beneath DDP and PEFT wrappers."""
    candidate = getattr(model, "module", model)
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    if hasattr(candidate, "model") and hasattr(candidate, "lm_head"):
        return candidate
    base_model = getattr(candidate, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        candidate = base_model.model
    if not (hasattr(candidate, "model") and hasattr(candidate, "lm_head")):
        raise TypeError(f"cannot locate causal LM backbone on {type(model)!r}")
    return candidate


def install_weighted_loss_patch(level_weights: dict[str, float]) -> None:
    """Patch only LLaMA-Factory's SFT loss; data and optimizer stay unchanged."""
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def compute_loss(self, model, inputs, *args, **kwargs):
        if "labels" not in inputs:
            return original_compute_loss(self, model, inputs, *args, **kwargs)

        labels = inputs["labels"]
        causal_lm = unwrap_causal_lm(model)
        transformer_inputs = {
            key: value
            for key, value in inputs.items()
            if key
            in {
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values",
                "inputs_embeds",
            }
        }
        transformer_inputs["use_cache"] = False
        outputs = causal_lm.model(**transformer_inputs)
        hidden_states = outputs.last_hidden_state

        class_weights = getattr(self, "_llm4rec_class_weights", None)
        if class_weights is None or class_weights.device != hidden_states.device:
            vocab_size = int(
                getattr(causal_lm.config, "vocab_size", causal_lm.lm_head.weight.size(0))
            )
            class_weights = build_hierarchical_token_weights(
                self.processing_class,
                vocab_size,
                hidden_states.device,
                torch.float32,
                level_weights,
            )
            self._llm4rec_class_weights = class_weights
            print(f"[llm4rec] hierarchical SID weights: {level_weights}", flush=True)

        loss = compute_fused_causal_lm_loss(
            hidden_states,
            labels,
            causal_lm.lm_head.weight,
            class_weights,
            ignore_index=IGNORE_INDEX,
            bias=getattr(causal_lm.lm_head, "bias", None),
        )
        return (loss, outputs) if kwargs.get("return_outputs", False) else loss

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = compute_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    weights = normalize_level_weights(config.pop("sid_loss_weight", 1.0))
    install_weighted_loss_patch(weights)

    os.environ.setdefault("ALLOW_EXTRA_ARGS", "0")
    from llamafactory.train.tuner import run_exp

    run_exp(config)


if __name__ == "__main__":
    main()
