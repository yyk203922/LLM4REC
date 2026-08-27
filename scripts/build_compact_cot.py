#!/usr/bin/env python3
"""Build compact, auditable recommendation CoT from official Caption/Tag data.

Only Caption/Tag entries whose SIDs appear in the user history are used.  The
target SID and any metadata after the user message are never read when creating
the CoT.  The resulting think text intentionally contains no SID token: it is
a short semantic bridge, not a copy of the user's interaction sequence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


FULL_SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
SID_RE = re.compile(r"<s_[abc]_\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-domain", type=int, default=1000)
    parser.add_argument("--min-history-tags", type=int, default=12)
    parser.add_argument("--min-top-root-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--emit-no-think",
        action="store_true",
        help="Also emit a no-think counterpart. The final compact-CoT recipe does not require it.",
    )
    return parser.parse_args()


def flatten(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def parse_messages(raw: str) -> dict[str, str]:
    messages = json.loads(raw)
    parsed: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        if role in {"system", "user", "assistant"}:
            parsed[role] = flatten(message.get("content"))
    return parsed


def clean_tag(value: object) -> str:
    return " ".join(value.split()).strip() if isinstance(value, str) else ""


def final_sid(text: str) -> str:
    matches = [match.group(0) for match in FULL_SID_RE.finditer(text)]
    return matches[-1] if matches else ""


def target_domain(sid: str) -> str:
    return sid.split("><", 1)[0] + ">"


def with_mode(user: str, mode: str) -> str:
    return user.replace("/think", "").replace("/no_think", "").rstrip() + mode


def source_id(record_id: int, user: str, gold: str) -> str:
    payload = f"{record_id}\0{user}\0{gold}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def history_tag_counts(row: dict, user: str) -> tuple[collections.Counter[str], collections.Counter[str], int]:
    """Collect tags for the user-message occurrences only, preserving multiplicity."""
    remaining = collections.Counter(match.group(0) for match in FULL_SID_RE.finditer(user))
    roots: collections.Counter[str] = collections.Counter()
    details: collections.Counter[str] = collections.Counter()
    tag_count = 0
    for sid, tag in zip(row["sid_token_list"], row["tag_list"]):
        if remaining[sid] <= 0:
            continue
        remaining[sid] -= 1
        tag = clean_tag(tag)
        if not tag:
            continue
        pieces = [piece.strip() for piece in tag.split("-") if piece.strip()]
        if not pieces:
            continue
        tag_count += 1
        roots[pieces[0]] += 1
        details["-".join(pieces[:2])] += 1
    return roots, details, tag_count


def compact_semantic_think(roots: collections.Counter[str], details: collections.Counter[str]) -> tuple[str, list[str]]:
    selected: list[str] = []
    for root, _count in roots.most_common(3):
        matching = [(detail, count) for detail, count in details.items() if detail.split("-", 1)[0] == root]
        matching.sort(key=lambda pair: (-pair[1], pair[0]))
        detail = matching[0][0] if matching else root
        selected.append(detail)
    summary = "、".join(selected)
    think = (
        "<think>\n"
        f"用户历史中的高频兴趣集中在：{summary}。"
        "应结合这些稳定偏好与当前目标场景，选择语义匹配的内容。\n"
        "</think>"
    )
    if SID_RE.search(think):
        raise RuntimeError("semantic think unexpectedly contains SID tokens")
    return think, selected


def candidate(row: dict, messages: dict[str, str], args: argparse.Namespace) -> tuple[str, dict, dict] | None:
    system = messages.get("system", "").strip()
    user = messages.get("user", "").strip()
    assistant = messages.get("assistant", "")
    if not system or not user or "</think>" not in assistant:
        return None
    tail = assistant.split("</think>", 1)[1].strip()
    gold = final_sid(tail)
    if not gold or len(FULL_SID_RE.findall(tail)) != 1:
        return None
    roots, details, tag_count = history_tag_counts(row, user)
    if tag_count < args.min_history_tags or not roots or roots.most_common(1)[0][1] < args.min_top_root_count:
        return None
    think, selected_tags = compact_semantic_think(roots, details)
    audit = {
        "record_id": int(row["record_id"]),
        "source_id": source_id(int(row["record_id"]), user, gold),
        "gold": gold,
        "target_domain": target_domain(gold),
        "history_tag_count": tag_count,
        "top_root_count": roots.most_common(1)[0][1],
        "semantic_tags": selected_tags,
        "think": think,
    }
    shared = {"instruction": system, "history": []}
    think_row = {**shared, "input": with_mode(user, "/think"), "output": think + "\n" + tail}
    no_think_row = {**shared, "input": with_mode(user, "/no_think"), "output": tail}
    return target_domain(gold), think_row, {"row": no_think_row, "audit": audit}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_per_domain <= 0:
        raise ValueError("--max-per-domain must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_domain: dict[str, list[tuple[dict, dict]]] = collections.defaultdict(list)
    scanned = 0
    for batch in pq.ParquetFile(args.parquet).iter_batches(
        columns=["record_id", "messages", "sid_token_list", "caption_list", "tag_list"], batch_size=256
    ):
        for row in batch.to_pylist():
            scanned += 1
            messages = parse_messages(row["messages"])
            built = candidate(row, messages, args)
            if built is None:
                continue
            domain, think_row, payload = built
            by_domain[domain].append((think_row, payload))

    rng = random.Random(args.seed)
    output_rows: list[dict] = []
    audit_rows: list[dict] = []
    selected_by_domain: dict[str, int] = {}
    candidate_by_domain: dict[str, int] = {}
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        candidate_by_domain[domain] = len(rows)
        rng.shuffle(rows)
        rows = rows[: args.max_per_domain]
        selected_by_domain[domain] = len(rows)
        for think_row, payload in rows:
            output_rows.append(think_row)
            if args.emit_no_think:
                output_rows.append(payload["row"])
            audit_rows.append(payload["audit"])
    rng.shuffle(output_rows)

    for row in output_rows:
        if row["input"].endswith("/think") and SID_RE.search(row["output"].split("</think>", 1)[0]):
            raise RuntimeError("a think target contains SID tokens")
        if not final_sid(row["output"]):
            raise RuntimeError("missing original gold SID in output")

    suffix = "pairs" if args.emit_no_think else "think"
    data_path = args.output_dir / f"official_caption_tag_rec_semantic_bridge_{suffix}.jsonl"
    audit_path = args.output_dir / "audit.jsonl"
    write_jsonl(data_path, output_rows)
    write_jsonl(audit_path, audit_rows)
    manifest = {
        "format": "official_caption_tag_semantic_bridge_v1",
        "emit_no_think": args.emit_no_think,
        "source": str(args.parquet),
        "scanned_records": scanned,
        "quality_gate": {
            "min_history_tags": args.min_history_tags,
            "min_top_root_count": args.min_top_root_count,
            "no_sid_tokens_in_think": True,
            "metadata_source": "user-message SID occurrences only",
        },
        "candidate_by_target_domain": candidate_by_domain,
        "selected_pairs_by_target_domain": selected_by_domain,
        "pairs": len(audit_rows),
        "rows": len(output_rows),
        "data_path": str(data_path),
        "audit_path": str(audit_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
