"""Build a blind validation set for the coding filter.

Eyeballing only the survivors measures nothing: it can reveal false negatives
(coding prompts that slipped through) but says nothing about false positives
(ordinary prompts wrongly discarded). So we sample BOTH sides of the decision
and shuffle them together, with the filter's verdict hidden in a separate file.

The population sampled is the one the filter actually acts on inside the Study A
pipeline: records that already passed the language, toxicity and length filters
and have at least `min_turns` turns — i.e. exactly the pool the final sample was
drawn from.

Outputs (to data/out/):
    coding_validation_blind.csv    first user turn + empty `label` column
    coding_validation_key.json     the filter's verdict for each id

Label each row in the blind CSV as:
    c   coding      — asks for code, debugging, or a programming explanation
    n   not coding  — everything else
    ?   unsure

Then score with:  python data/validate_coding_filter.py --score

Usage:
    python data/validate_coding_filter.py [--per-side 100]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filters import extract_turns, is_coding  # noqa: E402
from sample_study_a import list_shards  # noqa: E402


def collect(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Return (flagged_as_coding, passed_as_clean) first-user-turns."""
    ds, sa, f = cfg["dataset"], cfg["study_a"], cfg["filters"]
    cf = f["coding"]
    shards, _ = list_shards(ds["name"], ds["shard_stride"])

    flagged: list[dict] = []
    clean: list[dict] = []

    for shard in shards:
        path = hf_hub_download(ds["name"], shard, repo_type="dataset")
        table = pq.read_table(
            path, columns=["conversation_hash", "conversation", "language", "toxic"]
        )
        for batch in table.to_batches(max_chunksize=2000):
            for rec in batch.to_pylist():
                # Mirror check_record's ordering, minus the coding rule itself.
                if rec.get("language") != f["language"]:
                    continue
                if f["drop_toxic"] and rec.get("toxic"):
                    continue
                turns = extract_turns(rec.get("conversation") or [])
                if len(turns) < sa["min_turns"]:
                    continue
                first = turns[0][0].strip()
                if not (f["min_user_chars"] <= len(first) <= f["max_user_chars"]):
                    continue

                row = {"id": rec["conversation_hash"], "text": first}
                target = flagged if is_coding(
                    first, cf["min_keyword_hits"], cf["drop_on_code_block"]
                ) else clean
                target.append(row)
        print(f"[scan] {shard}: flagged={len(flagged):,} clean={len(clean):,}",
              flush=True)

    return flagged, clean


def build(cfg: dict, per_side: int, out_dir: Path) -> None:
    flagged, clean = collect(cfg)
    rng = random.Random(cfg["seed"])

    picked = ([dict(r, verdict="coding") for r in
               rng.sample(flagged, min(per_side, len(flagged)))]
              + [dict(r, verdict="clean") for r in
                 rng.sample(clean, min(per_side, len(clean)))])
    rng.shuffle(picked)

    blind = out_dir / "coding_validation_blind.csv"
    with blind.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "first_user_turn"])
        for r in picked:
            # Collapse newlines so one prompt stays on one spreadsheet row.
            w.writerow([r["id"], "", " ".join(r["text"].split())[:1500]])

    key = out_dir / "coding_validation_key.json"
    key.write_text(json.dumps(
        {"seed": cfg["seed"], "per_side": per_side,
         "population_flagged": len(flagged), "population_clean": len(clean),
         "verdicts": {r["id"]: r["verdict"] for r in picked}}, indent=2))

    print(f"\n[build] {len(picked)} rows -> {blind}")
    print(f"[build] verdict key (do not peek while labelling) -> {key}")
    print(f"[build] population: {len(flagged):,} flagged, {len(clean):,} clean")


def score(out_dir: Path, labels_path: Path) -> None:
    key = json.loads((out_dir / "coding_validation_key.json").read_text())
    verdicts = key["verdicts"]

    tp = fp = tn = fn = unsure = 0
    with labels_path.open() as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip().lower()
            verdict = verdicts.get(row["id"])
            if verdict is None:
                continue
            if label not in ("c", "n"):
                unsure += 1
                continue
            truth_coding = label == "c"
            said_coding = verdict == "coding"
            if said_coding and truth_coding:
                tp += 1
            elif said_coding and not truth_coding:
                fp += 1
            elif not said_coding and truth_coding:
                fn += 1
            else:
                tn += 1

    labelled = tp + fp + tn + fn
    if not labelled:
        raise SystemExit("No usable labels found. Fill the `label` column with c/n.")

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")

    # The sample is balanced by design (equal numbers from each side), so these
    # rates are per-side rates, NOT the rate in the corpus. Contamination of the
    # final study sample is estimated from the false-negative rate applied to the
    # clean population.
    fn_rate = fn / (fn + tn) if fn + tn else float("nan")
    contamination = fn_rate * key["population_clean"]

    print(f"labelled {labelled} rows ({unsure} unsure/blank, excluded)\n")
    print("                    filter said coding   filter said clean")
    print(f"  actually coding   {tp:>16}   {fn:>17}")
    print(f"  actually clean    {fp:>16}   {tn:>17}\n")
    print(f"  precision (flagged that really are coding): {precision:.1%}")
    print(f"  recall    (coding prompts it catches):      {recall:.1%}")
    print(f"  false-negative rate among survivors:        {fn_rate:.1%}")
    print(f"  => est. coding prompts remaining in the {key['population_clean']:,}"
          f"-conversation pool: ~{contamination:,.0f}")

    (out_dir / "coding_validation_result.json").write_text(json.dumps({
        "labelled": labelled, "unsure_excluded": unsure,
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "precision": precision, "recall": recall,
        "false_negative_rate_among_survivors": fn_rate,
        "estimated_coding_remaining_in_pool": contamination,
        "population_clean": key["population_clean"],
        "population_flagged": key["population_flagged"],
        "note": ("Sample is balanced by construction, so precision/recall are "
                 "per-side rates and not corpus base rates."),
    }, indent=2))
    print(f"\n[score] wrote {out_dir / 'coding_validation_result.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--per-side", type=int, default=100)
    ap.add_argument("--score", action="store_true",
                    help="score a filled-in label file instead of building one")
    ap.add_argument("--labels", default=None,
                    help="CSV with the label column filled (default: the blind file)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score:
        labels = Path(args.labels) if args.labels else out_dir / "coding_validation_blind.csv"
        score(out_dir, labels)
    else:
        build(yaml.safe_load(Path(args.config).read_text()), args.per_side, out_dir)


if __name__ == "__main__":
    main()
