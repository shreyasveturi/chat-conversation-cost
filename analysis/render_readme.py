"""Render README.md from the template, injecting real numbers from the results.

No result number in the README is typed by hand. If the pipeline has not been
run, placeholders render as "—" and the status line says so, rather than showing
a plausible-looking figure that nobody measured.

Usage:
    python analysis/render_readme.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PENDING = "—"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pct(x) -> str:
    return PENDING if x is None else f"{x * 100:.0f}%"


def times(x) -> str:
    return PENDING if x is None else f"{x:.0f}×"


def thousands(x) -> str:
    if x is None:
        return PENDING
    return f"{x / 1_000_000:.1f}M" if x >= 1_000_000 else f"{x / 1000:.0f}k"


def gap_pp(bal: dict) -> str:
    """Balanced vs unbalanced resent share at the comparison turn, in points."""
    a = bal.get("history_share_unbalanced_at_compare_turn")
    b = bal.get("history_share_balanced_at_compare_turn")
    return PENDING if a is None or b is None else f"{abs(a - b) * 100:.1f}"


def drop_table(log: dict | None) -> str:
    if not log:
        return "_Pending — run the pipeline._"
    scanned = log.get("records_scanned") or 1
    rows = ["| Reason | Conversations | Share of scanned |",
            "|---|---:|---:|"]
    for reason, count in log.get("drop_reasons", {}).items():
        label = "**kept**" if reason == "ok" else reason.replace("_", " ")
        rows.append(f"| {label} | {count:,} | {count / scanned:.1%} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="analysis/readme_template.md")
    ap.add_argument("--out", default="README.md")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--filter-log", default="data/out/study_a_filter_log.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    head = load(Path(args.tables) / "study_a_headline.json")
    flog = load(Path(args.filter_log))
    cfg = yaml.safe_load(Path(args.config).read_text())

    if head:
        hist = head["history_share_cum_at_turn"]
        rel = head["relative_cost_vs_turn1_at_turn"]
        cum = head["cum_input_tokens_at_turn"]
        max_turn = str(head["max_turn_reported"])
        bal = head.get("balanced_panel", {})
        growth = head.get("growth", {})
        values = {
            "status_a": "**complete** — results below",
            "n_conversations": f"{head['n_conversations']:,}",
            "max_turn": max_turn,
            "hist_share_10": pct(hist.get("10")),
            "hist_share_max": pct(hist.get(max_turn)),
            "rel_cost_10": times(rel.get("10")),
            "rel_cost_max": times(rel.get(max_turn)),
            "cum_tokens_max": thousands(cum.get(max_turn)),
            "growth_exponent": f"{growth.get('growth_exponent', 0):.2f}",
            "log_log_r2": f"{growth.get('log_log_r2', 0):.3f}",
            "balanced_n": f"{bal.get('n_conversations', 0):,}",
            "balanced_share": pct(bal.get("history_share_cum_at_final_turn")),
            "balanced_turn": str(bal.get("min_turns", PENDING)),
            "balanced_compare_turn": str(bal.get("compare_turn", PENDING)),
            "balanced_compare": pct(bal.get("history_share_balanced_at_compare_turn")),
            "unbalanced_compare": pct(bal.get("history_share_unbalanced_at_compare_turn")),
            "balanced_gap_pp": gap_pp(bal),
        }
    else:
        values = dict.fromkeys(
            ["n_conversations", "max_turn", "hist_share_10", "hist_share_max",
             "rel_cost_10", "rel_cost_max", "cum_tokens_max", "growth_exponent",
             "log_log_r2", "balanced_n", "balanced_share", "balanced_turn",
             "balanced_compare_turn", "balanced_compare", "unbalanced_compare",
             "balanced_gap_pp"],
            PENDING,
        )
        values["status_a"] = (
            "**not yet run** — the pipeline is complete and tested, but no "
            "results exist yet. Every figure below renders as — until "
            "`./run_study_a.sh` is run"
        )

    # Reproducibility note about which shards were actually read.
    if flog and flog.get("shard_list_from_local_cache"):
        values["shard_note"] = (
            f"- **This run read only the {len(flog['shards_read'])} shard(s) already "
            "in the local Hugging Face cache**, not the configured "
            f"`shard_stride` sample of the full corpus. The shard list is in the "
            "filter log. A run with `HF_TOKEN` set reads the configured stride "
            "across all shards and will produce a larger, more evenly spread "
            "sample."
        )
    elif flog:
        values["shard_note"] = (
            f"- Read {len(flog['shards_read'])} shard(s) at stride "
            f"{cfg['dataset']['shard_stride']}; exact list in the filter log."
        )
    else:
        values["shard_note"] = ""

    values["n_at_max"] = PENDING
    if head:
        try:
            import csv
            with open(Path(args.tables) / "study_a_by_turn_index.csv") as fh:
                rows = list(csv.DictReader(fh))
            values["n_at_max"] = f"{int(rows[-1]['n']):,}"
        except (FileNotFoundError, IndexError, KeyError, ValueError):
            pass

    values["seed"] = str(cfg["seed"])
    values["records_scanned"] = f"{flog['records_scanned']:,}" if flog else PENDING
    values["pool_size"] = f"{flog['pool_after_filters']:,}" if flog else PENDING
    values["drop_table"] = drop_table(flog)

    text = Path(args.template).read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)

    leftover = [line for line in text.splitlines() if "{{" in line]
    if leftover:
        raise SystemExit(f"Unfilled placeholders remain:\n" + "\n".join(leftover))

    Path(args.out).write_text(text)
    print(f"[readme] wrote {args.out} "
          f"({'with results' if head else 'in pending state'})")


if __name__ == "__main__":
    main()
