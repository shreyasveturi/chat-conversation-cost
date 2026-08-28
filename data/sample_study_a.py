"""Sample long multi-turn conversations from WildChat-1M for Study A.

Reads a deterministic subset of the dataset's parquet shards (every Nth shard),
applies the shared filters, requires a minimum conversation length, de-duplicates
on the first user turn, and writes both the surviving conversations and a full
filter log.

The raw dataset is never committed. What lands in the repo is the filter log and
the derived per-turn token table, which is what a reviewer actually needs.

Usage:
    python data/sample_study_a.py [--config config.yaml] [--out data/out]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filters import check_record, extract_turns  # noqa: E402

WS_RE = re.compile(r"\s+")


def dedupe_key(text: str) -> str:
    """Near-duplicate key: whitespace-collapsed, lowercased, first 300 chars.

    Catches the very common case of the same templated opener being sent by many
    users, without trying to be a full semantic dedupe.
    """
    norm = WS_RE.sub(" ", text.strip().lower())[:300]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def cached_shards(repo: str) -> list[str]:
    """Parquet shards already in the local HF cache, if any.

    Used as a fallback when the repo listing is unavailable (no token, offline).
    Lets a run complete against whatever has already been downloaded rather than
    failing outright — at the cost of a possibly partial shard set, which is
    recorded in the filter log.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    return sorted({
        str(p.relative_to(snap))
        for snap in sorted(root.glob("*"))
        for p in snap.rglob("*.parquet")
    }) if root.exists() else []


def list_shards(repo: str, stride: int) -> tuple[list[str], bool]:
    """Return (shard filenames, used_cache_fallback)."""
    try:
        files = sorted(
            f for f in HfApi().list_repo_files(repo, repo_type="dataset")
            if f.endswith(".parquet")
        )
    except Exception as exc:  # offline, network failure, rate limit
        files = []
        print(f"[shards] could not list {repo} ({type(exc).__name__}).", flush=True)

    if files:
        selected = files[::stride]
        print(f"[shards] {len(files)} parquet files in repo; "
              f"reading {len(selected)} (stride {stride}).", flush=True)
        return selected, False

    selected = cached_shards(repo)
    if not selected:
        raise SystemExit(
            f"Cannot list {repo} and nothing is cached locally.\n"
            "The dataset is public (ODC-BY), so this is most likely a network\n"
            "problem or HF_HUB_OFFLINE being set."
        )
    print(f"[shards] WARNING: using {len(selected)} shard(s) already in the local "
          f"cache; the stride setting is not applied. Set HF_TOKEN for the full "
          f"reproducible sample.", flush=True)
    return selected, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/out")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds = cfg["dataset"]
    sa = cfg["study_a"]
    seed = cfg["seed"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check real credentials, not just the env var: `hf auth login` stores the
    # token on disk, which authenticates fine but leaves the environment empty.
    try:
        from huggingface_hub import get_token
        authed = bool(get_token())
    except Exception:
        authed = bool(os.environ.get("HF_TOKEN")
                      or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not authed:
        print("[note] No Hugging Face credentials found. WildChat-1M is public "
              "(ODC-BY), so this should work anyway.", flush=True)

    shards, used_cache = list_shards(ds["name"], ds["shard_stride"])

    reasons: Counter[str] = Counter()
    seen_keys: set[str] = set()
    kept: list[dict] = []
    scanned = 0

    for shard in shards:
        path = hf_hub_download(ds["name"], shard, repo_type="dataset")
        # Read only the columns we use. WildChat carries per-message
        # `openai_moderation` and `detoxify_moderation` structs that dwarf the
        # conversation text; reading the full table pulls gigabytes into memory
        # for no reason.
        available = set(pq.ParquetFile(path).schema_arrow.names)
        wanted = ["conversation_hash", "conversation", "language", "toxic",
                  "turn", "model"]
        missing = [c for c in ("conversation_hash", "conversation", "language") if c not in available]
        if missing:
            raise SystemExit(
                f"{shard} is missing required column(s) {missing}. "
                f"Available: {sorted(available)}"
            )
        table = pq.read_table(path, columns=[c for c in wanted if c in available])

        for batch in table.to_batches(max_chunksize=2000):
            for record in batch.to_pylist():
                scanned += 1
                keep, reason = check_record(record, cfg)
                if not keep:
                    reasons[reason] += 1
                    continue

                turns = extract_turns(record["conversation"])
                if len(turns) < sa["min_turns"]:
                    reasons["too_few_turns"] += 1
                    continue

                key = dedupe_key(turns[0][0])
                if key in seen_keys:
                    reasons["near_duplicate"] += 1
                    continue
                seen_keys.add(key)

                reasons["ok"] += 1
                kept.append({
                    "conversation_id": record.get("conversation_hash"),
                    "source_model": record.get("model"),
                    "n_turns": len(turns),
                    "turns": turns,
                })

        print(f"[scan] {shard}: scanned={scanned:,} kept={len(kept):,}", flush=True)
        if len(kept) >= sa["target_conversations"] * 3:
            print("[scan] ample pool collected; stopping early.", flush=True)
            break

    if not kept:
        raise SystemExit("No conversations survived filtering. Check dataset access "
                         "and filter settings.")

    # Seeded sample down to target. Sorting first makes the sample independent of
    # dict/shard iteration order, so the seed alone reproduces it.
    kept.sort(key=lambda c: str(c["conversation_id"]))
    rng = random.Random(seed)
    target = min(sa["target_conversations"], len(kept))
    sample = rng.sample(kept, target)

    conv_path = out_dir / "study_a_conversations.jsonl"
    with conv_path.open("w") as fh:
        for conv in sample:
            fh.write(json.dumps(conv) + "\n")

    log = {
        "dataset": ds["name"],
        "shards_read": shards,
        "shard_list_from_local_cache": used_cache,
        "seed": seed,
        "min_turns": sa["min_turns"],
        "records_scanned": scanned,
        "drop_reasons": dict(reasons.most_common()),
        "pool_after_filters": len(kept),
        "sampled": len(sample),
        "filters": cfg["filters"],
    }
    (out_dir / "study_a_filter_log.json").write_text(json.dumps(log, indent=2))

    print(f"\n[done] scanned {scanned:,} conversations")
    for reason, n in reasons.most_common():
        print(f"       {reason:<24} {n:>8,}  ({n / scanned:.1%})")
    print(f"[done] pool={len(kept):,}  sampled={len(sample):,} -> {conv_path}")


if __name__ == "__main__":
    main()
