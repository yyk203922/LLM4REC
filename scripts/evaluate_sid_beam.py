#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FULL_RE = re.compile(r"<\|(video|prod|ad|living|sid)_begin\|><s_a_[^<>]+><s_b_[^<>]+><s_c_[^<>]+>")
DOMAIN_RE = re.compile(r"(<\|(video|prod|ad|living|sid)_begin\|>)")


def read_samples(path: Path, n: int, seed: int, require_output_sid: bool = False, domain: str | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gold = final_full_sid(row.get("output", "")) if require_output_sid else ""
                if require_output_sid and not gold:
                    continue
                if domain is not None and not gold.startswith(domain):
                    continue
                rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def final_full_sid(text: str) -> str:
    matches = [m.group(0) for m in FULL_RE.finditer(text)]
    return matches[-1] if matches else ""


def domain_token(full_sid: str) -> str:
    match = DOMAIN_RE.search(full_sid)
    return match.group(1) if match else "<|video_begin|>"


def user_content(example: dict) -> str:
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    if instruction and input_text:
        return instruction + "\n" + input_text
    return instruction or input_text


def chat_input(tokenizer, example: dict, enable_thinking: bool) -> torch.Tensor:
    messages = []
    system = (example.get("system") or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    content = user_content(example)
    if not enable_thinking:
        content = re.sub(r"/think\s*$", "/no_think", content)
    messages.append({"role": "user", "content": content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )


def append_prompt_token(tokenizer, input_ids: torch.Tensor, prompt_token: str) -> torch.Tensor:
    token_ids = tokenizer(prompt_token, add_special_tokens=False, return_tensors="pt").input_ids
    return torch.cat([input_ids, token_ids], dim=-1)


def beam_candidates(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_token: str,
    beam_width: int,
    max_new_tokens: int,
) -> list[str]:
    input_ids = append_prompt_token(tokenizer, input_ids, prompt_token).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    candidates = []
    for row in outputs:
        suffix = tokenizer.decode(row[input_ids.shape[-1] :], skip_special_tokens=False)
        candidates.append(prompt_token + suffix)
    return candidates


def generate_think(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> str:
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=False)


def input_with_assistant_text(tokenizer, example: dict, assistant_text: str) -> torch.Tensor:
    messages = []
    system = (example.get("system") or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content(example)})
    messages.append({"role": "assistant", "content": assistant_text})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if text.endswith("<|im_end|>\n"):
        text = text[: -len("<|im_end|>\n")]
    elif text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")]
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def eval_material(args, model, tokenizer) -> None:
    samples = read_samples(
        Path(args.material),
        args.material_n,
        args.seed,
        require_output_sid=True,
        domain="<|video_begin|>" if args.material_video_only else None,
    )
    hit = valid = 0
    for idx, ex in enumerate(samples, 1):
        gold = final_full_sid(ex.get("output", ""))
        # Official Task4 is video-only. For mixed training data, use gold domain so this small check is not unfair.
        prompt = domain_token(gold)
        cands = beam_candidates(model, tokenizer, chat_input(tokenizer, ex, False), prompt, args.material_beam, 3)
        valid += int(any(final_full_sid(c) for c in cands))
        hit += int(gold in cands)
        print(f"material {idx}/{len(samples)} beam_hit={int(gold in cands)} gold={gold} top={cands[0] if cands else ''}", flush=True)
    print(f"RESULT material beam{args.material_beam} hit={hit}/{len(samples)} valid={valid}/{len(samples)}")


def eval_rec(args, model, tokenizer) -> None:
    samples = read_samples(Path(args.rec), args.rec_n, args.seed)
    single_hit = two_hit = combined_hit = 0
    for idx, ex in enumerate(samples, 1):
        gold = final_full_sid(ex.get("output", ""))
        prompt = domain_token(gold)
        single = beam_candidates(model, tokenizer, chat_input(tokenizer, ex, False), prompt, args.rec_beam, 3)
        think = generate_think(model, tokenizer, chat_input(tokenizer, ex, True), args.think_max_new)
        two = beam_candidates(model, tokenizer, input_with_assistant_text(tokenizer, ex, think), prompt, args.rec_beam, 3)
        single_hit += int(gold in single)
        two_hit += int(gold in two)
        combined_hit += int(gold in set(single + two))
        print(
            f"rec {idx}/{len(samples)} single={int(gold in single)} two_stage={int(gold in two)} "
            f"combined={int(gold in set(single + two))} gold={gold} top_single={single[0] if single else ''}",
            flush=True,
        )
    n = len(samples)
    print(f"RESULT rec beam{args.rec_beam} single={single_hit}/{n} two_stage={two_hit}/{n} combined64={combined_hit}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--material", required=True)
    parser.add_argument("--rec", required=True)
    parser.add_argument("--material-n", type=int, default=8)
    parser.add_argument("--rec-n", type=int, default=4)
    parser.add_argument("--material-beam", type=int, default=64)
    parser.add_argument("--rec-beam", type=int, default=32)
    parser.add_argument("--think-max-new", type=int, default=512)
    parser.add_argument("--material-video-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.adapter, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    eval_material(args, model, tokenizer)
    eval_rec(args, model, tokenizer)


if __name__ == "__main__":
    main()
