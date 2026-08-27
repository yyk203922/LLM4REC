#!/usr/bin/env python3
"""Build high-confidence video ItemIC SID-layer SFT data.

The generated curriculum combines complete SID generation with explicit
layer-wise conditional prediction while keeping every target traceable to the
official PID-to-SID mapping.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


DOMAIN = "video/video"
PREFIX = "<|video_begin|>"

LIST_LIKE_RE = re.compile(r"^\s*\[.*\]\s*$", re.S)
URL_OR_JUNK_RE = re.compile(r"(https?://|www\.|@\w+|^[\d\W_]+$)", re.I)
COMMERCIAL_RE = re.compile(
    r"(点击|下载|安装|下单|私信|链接|优惠|直播间|购买|领取|红包|广告|推广|招商|扫码|APP)",
    re.I,
)

SYSTEM = "你是短视频语义标识生成助手，请根据视频内容理解并输出对应的结构化视频 SID token。"


def iter_parquet_rows(directory: Path, columns: list[str], max_files: int | None = None) -> Iterable[dict]:
    paths = sorted(directory.glob("*.parquet"))
    if max_files is not None:
        paths = paths[:max_files]
    for path in paths:
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            yield row


def normalize_caption(text: object) -> tuple[str, list[str]]:
    if text is None:
        return "", []

    raw = str(text).strip()
    phrases: list[str] = []
    if LIST_LIKE_RE.match(raw):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                phrases = [str(x).strip() for x in parsed if str(x).strip()]
                raw = "，".join(phrases)
        except Exception:
            pass

    raw = raw.replace("\\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    if not phrases:
        phrases = [x.strip() for x in re.split(r"[，,。；;、]", raw) if x.strip()]
    return raw, phrases[:10]


def caption_ok(caption: str, phrases: list[str], drop_commercial: bool) -> bool:
    if not caption or URL_OR_JUNK_RE.search(caption):
        return False
    if drop_commercial and COMMERCIAL_RE.search(caption):
        return False
    length = len(caption)
    if length < 60 or length > 460:
        return False
    if len(set(caption)) < min(16, max(6, length // 15)):
        return False
    if phrases and len([x for x in phrases if len(x) >= 2]) < 3:
        return False
    return True


def sid_parts(sid_three: object) -> tuple[int, int, int] | None:
    if sid_three is None or len(sid_three) != 3:
        return None
    try:
        a, b, c = [int(float(x)) for x in sid_three]
    except Exception:
        return None
    if min(a, b, c) < 0:
        return None
    return a, b, c


def suffix(parts: tuple[int, int, int]) -> str:
    a, b, c = parts
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def full_sid(parts: tuple[int, int, int]) -> str:
    return PREFIX + suffix(parts)


def reservoir_add(bucket: list[dict], item: dict, cap: int, seen: int, rng: random.Random) -> None:
    if len(bucket) < cap:
        bucket.append(item)
        return
    idx = rng.randint(0, seen - 1)
    if idx < cap:
        bucket[idx] = item


def collect_caption_candidates(
    root: Path,
    cap: int,
    rng: random.Random,
    max_caption_files: int | None,
    drop_commercial: bool,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    candidates: list[dict] = []
    seen = 0
    for row in iter_parquet_rows(root / "OneReason_Pid2Caption", ["pid", "domain", "caption"], max_caption_files):
        stats["caption_rows"] += 1
        if row.get("domain") != DOMAIN:
            continue
        stats["video_rows"] += 1
        caption, phrases = normalize_caption(row.get("caption"))
        if not caption_ok(caption, phrases, drop_commercial):
            stats["drop_caption_quality"] += 1
            continue
        seen += 1
        reservoir_add(
            candidates,
            {"pid": int(row["pid"]), "caption": caption, "phrases": phrases},
            cap,
            seen,
            rng,
        )
    stats["caption_candidates_seen"] = seen
    stats["caption_candidates_kept"] = len(candidates)
    return candidates, stats


def collect_sid_map(root: Path, wanted: set[int]) -> dict[int, tuple[int, int, int]]:
    out: dict[int, tuple[int, int, int]] = {}
    if not wanted:
        return out
    for row in iter_parquet_rows(root / "OneReason_Pid2Sid", ["pid", "domain", "sid_three"]):
        if row.get("domain") != DOMAIN:
            continue
        pid = int(row["pid"])
        if pid not in wanted:
            continue
        parts = sid_parts(row.get("sid_three"))
        if parts:
            out[pid] = parts
    return out


def collect_tag_map(root: Path, wanted: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    if not wanted:
        return out
    tag_dir = root / "OneReason_Pid2Tag"
    if not tag_dir.exists():
        return out
    for row in iter_parquet_rows(tag_dir, ["pid", "domain", "tag_lv3"]):
        if row.get("domain") != DOMAIN:
            continue
        pid = int(row["pid"])
        if pid not in wanted:
            continue
        tag = str(row.get("tag_lv3") or "").strip()
        if tag:
            out[pid] = tag
    return out


def concise_caption(caption: str, max_chars: int) -> str:
    if len(caption) <= max_chars:
        return caption
    parts = [x.strip() for x in re.split(r"(?<=[。！？])", caption) if x.strip()]
    kept = ""
    for part in parts:
        if len(kept) + len(part) > max_chars:
            break
        kept += part
    return kept.strip() if len(kept) >= 80 else caption[:max_chars].rstrip("，,。；; ")


def select_balanced_items(
    candidates: list[dict],
    sid_map: dict[int, tuple[int, int, int]],
    tag_map: dict[int, str],
    total_items: int,
    max_per_sa: int,
    max_per_sab: int,
    rng: random.Random,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    by_sa: dict[int, list[dict]] = defaultdict(list)
    seen_sid: set[tuple[int, int, int]] = set()
    seen_caption: set[str] = set()
    sab_counts = Counter()

    rng.shuffle(candidates)
    for row in candidates:
        pid = row["pid"]
        parts = sid_map.get(pid)
        if not parts:
            stats["drop_no_sid"] += 1
            continue
        digest = hashlib.md5(row["caption"].encode("utf-8")).hexdigest()
        if parts in seen_sid or digest in seen_caption:
            stats["drop_duplicate"] += 1
            continue
        a, b, _ = parts
        if sab_counts[(a, b)] >= max_per_sab:
            stats["drop_sab_cap"] += 1
            continue
        seen_sid.add(parts)
        seen_caption.add(digest)
        sab_counts[(a, b)] += 1
        item = dict(row)
        item["sid_parts"] = parts
        item["tag"] = tag_map.get(pid, "")
        by_sa[a].append(item)

    for rows in by_sa.values():
        rng.shuffle(rows)

    selected: list[dict] = []
    sa_take = Counter()
    keys = list(by_sa)
    rng.shuffle(keys)
    cursor = 0
    misses = 0
    while len(selected) < total_items and keys and misses < len(keys) * 2:
        a = keys[cursor % len(keys)]
        rows = by_sa[a]
        if rows and sa_take[a] < max_per_sa:
            selected.append(rows.pop())
            sa_take[a] += 1
            misses = 0
        else:
            misses += 1
        cursor += 1

    rng.shuffle(selected)
    stats["joined_unique_pool"] = sum(len(v) for v in by_sa.values()) + len(selected)
    stats["selected_items"] = len(selected)
    stats["selected_sa"] = len(sa_take)
    stats["max_selected_per_sa"] = max(sa_take.values()) if sa_take else 0
    return selected, stats


def render_source(item: dict, rng: random.Random, tag_ratio: float, max_caption_chars: int) -> str:
    caption = concise_caption(item["caption"], max_caption_chars)
    tag = item.get("tag") or ""
    if tag and rng.random() < tag_ratio:
        return f"短视频描述：{caption}\n类目标签：{tag}"
    return f"短视频描述：{caption}"


def make_layer_examples(item: dict, rng: random.Random, tag_ratio: float, max_caption_chars: int) -> list[dict]:
    parts = item["sid_parts"]
    a, b, c = parts
    src = render_source(item, rng, tag_ratio, max_caption_chars)
    suf = suffix(parts)
    full = full_sid(parts)
    sa = f"<s_a_{a}>"
    sb = f"<s_b_{b}>"
    sc = f"<s_c_{c}>"

    prompt_full = rng.choice(
        [
            "请根据以下短视频内容描述，生成对应的完整视频 SID token。\n{src}",
            "阅读短视频内容描述，输出匹配的视频 token。\n{src}",
            "请把这段短视频内容映射为结构化视频 token。\n{src}",
        ]
    ).format(src=src)
    prompt_sid_only = rng.choice(
        [
            "已知目标属于短视频域，视频 token 前缀为 <|video_begin|>。请只输出后三段 SID token。\n{src}",
            "下面是短视频内容，已给定 domain token <|video_begin|>，请预测 <s_a><s_b><s_c>。\n{src}",
        ]
    ).format(src=src)
    prompt_sid_only_alt = rng.choice(
        [
            "目标 domain token 已给定为 <|video_begin|>。请根据短视频描述，仅生成三段 SID 编码，不要重复 domain token。\n{src}",
            "这是短视频域内容。请根据描述输出对应的 <s_a_*><s_b_*><s_c_*>，不要添加解释。\n{src}",
        ]
    ).format(src=src)
    prompt_a = f"任务：预测短视频 SID 的第一层 <s_a>。\n已知 domain token：{PREFIX}\n{src}"
    prompt_b = f"任务：预测短视频 SID 的第二层 <s_b>。\n已知 domain token：{PREFIX}\n已知第一层 SID：{sa}\n{src}"
    prompt_c = f"任务：预测短视频 SID 的第三层 <s_c>。\n已知 domain token：{PREFIX}\n已知前两层 SID：{sa}{sb}\n{src}"

    common = {"pid": item["pid"], "s_a": a, "s_b": b, "s_c": c}
    return [
        {"task": "full_direct", "system": SYSTEM, "prompt": prompt_full, "response": full, **common},
        {"task": "sid_only", "system": SYSTEM, "prompt": prompt_sid_only, "response": suf, **common},
        {"task": "sid_only_alt", "system": SYSTEM, "prompt": prompt_sid_only_alt, "response": suf, **common},
        {"task": "layer_a", "system": SYSTEM, "prompt": prompt_a, "response": sa, **common},
        {"task": "layer_b", "system": SYSTEM, "prompt": prompt_b, "response": sb, **common},
        {"task": "layer_c", "system": SYSTEM, "prompt": prompt_c, "response": sc, **common},
    ]


def write_jsonl(path: Path, rows: list[dict], fmt: str) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            if fmt == "official":
                fp.write(json.dumps([{"system": row["system"], "prompt": row["prompt"], "response": row["response"]}], ensure_ascii=False) + "\n")
            elif fmt == "alpaca":
                fp.write(json.dumps({"instruction": row["system"], "input": row["prompt"], "output": row["response"]}, ensure_ascii=False) + "\n")
            else:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing OneReason_* parquet folders, or its parent data directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-items", type=int, default=10000, help="Unique video items used for 6-example 1:2:1:1:1 training groups.")
    parser.add_argument("--dev-items", type=int, default=1000, help="Held-out unique video items for diagnostics.")
    parser.add_argument("--oversample", type=int, default=10)
    parser.add_argument("--max-caption-files", type=int, default=None)
    parser.add_argument("--max-per-sa", type=int, default=36)
    parser.add_argument("--max-per-sab", type=int, default=8)
    parser.add_argument("--tag-ratio", type=float, default=0.25)
    parser.add_argument("--max-caption-chars", type=int, default=360)
    parser.add_argument("--keep-commercial", action="store_true", help="Keep captions with commercial/download/click wording.")
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data_root)
    if (root / "data" / "OneReason_Pid2Caption").exists():
        root = root / "data"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_items = args.train_items + args.dev_items
    candidate_cap = max(target_items * args.oversample, target_items + 2000)
    candidates, collect_stats = collect_caption_candidates(
        root,
        candidate_cap,
        rng,
        args.max_caption_files,
        drop_commercial=not args.keep_commercial,
    )
    wanted = {row["pid"] for row in candidates}
    sid_map = collect_sid_map(root, wanted)
    tag_map = collect_tag_map(root, wanted)
    selected, select_stats = select_balanced_items(
        candidates,
        sid_map,
        tag_map,
        target_items,
        args.max_per_sa,
        args.max_per_sab,
        rng,
    )

    dev_items = selected[: args.dev_items]
    train_items = selected[args.dev_items : args.dev_items + args.train_items]

    train_examples: list[dict] = []
    task_counts = Counter()
    response_counts = Counter()
    for item in train_items:
        examples = make_layer_examples(item, rng, args.tag_ratio, args.max_caption_chars)
        for ex in examples:
            train_examples.append(ex)
            task_counts[ex["task"]] += 1
            response_counts[ex["response"][:16]] += 1
    rng.shuffle(train_examples)

    dev_rows: list[dict] = []
    for item in dev_items:
        a, b, c = item["sid_parts"]
        dev_rows.append(
            {
                "pid": item["pid"],
                "domain": DOMAIN,
                "prefix": PREFIX,
                "caption": concise_caption(item["caption"], args.max_caption_chars),
                "tag": item.get("tag") or "",
                "s_a": a,
                "s_b": b,
                "s_c": c,
                "suffix": suffix(item["sid_parts"]),
                "full_sid": full_sid(item["sid_parts"]),
            }
        )

    stem = f"extra_itemic_video_layers_safe_{len(train_examples)}"
    official_path = out_dir / f"{stem}.official.jsonl"
    alpaca_path = out_dir / f"{stem}.alpaca.jsonl"
    raw_path = out_dir / f"{stem}.raw.jsonl"
    dev_path = out_dir / f"extra_itemic_video_diagnostic_{len(dev_rows)}.jsonl"
    report_path = out_dir / f"{stem}.report.json"

    write_jsonl(official_path, train_examples, "official")
    write_jsonl(alpaca_path, train_examples, "alpaca")
    write_jsonl(raw_path, train_examples, "raw")
    write_jsonl(dev_path, dev_rows, "raw")

    sa_counts = Counter(row["s_a"] for row in dev_rows)
    train_sa_counts = Counter(item["sid_parts"][0] for item in train_items)
    train_sab_counts = Counter((item["sid_parts"][0], item["sid_parts"][1]) for item in train_items)
    report = {
        "purpose": "video ItemIC curriculum (full:sid:a:b:c = 1:2:1:1:1)",
        "domain": DOMAIN,
        "prefix": PREFIX,
        "settings": vars(args),
        "candidate_cap": candidate_cap,
        "train_items": len(train_items),
        "dev_items": len(dev_items),
        "train_examples": len(train_examples),
        "task_counts": dict(task_counts),
        "caption_collect_stats": dict(collect_stats),
        "selection_stats": dict(select_stats),
        "train_unique_sa": len(train_sa_counts),
        "train_max_per_sa": max(train_sa_counts.values()) if train_sa_counts else 0,
        "train_unique_sab": len(train_sab_counts),
        "train_max_per_sab": max(train_sab_counts.values()) if train_sab_counts else 0,
        "dev_unique_sa": len(sa_counts),
        "paths": {
            "official": str(official_path),
            "alpaca": str(alpaca_path),
            "raw": str(raw_path),
            "diagnostic": str(dev_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
