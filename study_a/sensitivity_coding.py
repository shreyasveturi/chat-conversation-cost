"""Does residual coding contamination change Study A's answer?

Validation (see `data/validate_coding_filter.py`) put the coding filter at
roughly 78% precision and 88% recall, implying that about one in nine surviving
conversations is still coding-flavoured. The question that matters is not
whether the filter is imperfect — it is — but whether that imperfection moves
the numbers.

This re-runs the whole cost computation on an aggressively re-filtered subset
(one coding keyword is enough to drop, versus two in the study filter) and
prints the difference. If the headline metrics barely move, contamination is
not load-bearing for Study A.

Usage:
    python study_a/sensitivity_coding.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from filters import is_coding  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", default="data/out/study_a_conversations.jsonl")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/tables/study_a_sensitivity_coding.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.conversations).open()]
    keep = [r for r in rows if not is_coding(r["turns"][0][0], 1, True)]
    dropped = len(rows) - len(keep)
    print(f"[filter] {len(rows):,} -> {len(keep):,} "
          f"(dropped {dropped:,}, {dropped / len(rows):.1%})", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        subset = Path(tmp) / "strict.jsonl"
        with subset.open("w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        subprocess.run(
            [sys.executable, "study_a/compute_costs.py",
             "--conversations", str(subset), "--tables", tmp],
            check=True, stdout=subprocess.DEVNULL,
        )
        strict = json.loads((Path(tmp) / "study_a_headline.json").read_text())

    base = json.loads((Path(args.tables) / "study_a_headline.json").read_text())

    deltas = {}
    print(f"\n{'metric':<34}{'baseline':>11}{'strict':>11}{'delta':>11}")
    print(f"{'n conversations':<34}{base['n_conversations']:>11,}"
          f"{strict['n_conversations']:>11,}{'':>11}")
    for turn in ("10", "20", "30"):
        if turn not in base["history_share_cum_at_turn"]:
            continue
        x = base["history_share_cum_at_turn"][turn] * 100
        y = strict["history_share_cum_at_turn"][turn] * 100
        deltas[f"history_share_turn_{turn}_pp"] = y - x
        print(f"{'history share @ turn ' + turn:<34}{x:>10.1f}%{y:>10.1f}%{y - x:>+10.2f}pp")
    for turn in ("10", "30"):
        if turn not in base["relative_cost_vs_turn1_at_turn"]:
            continue
        x = base["relative_cost_vs_turn1_at_turn"][turn]
        y = strict["relative_cost_vs_turn1_at_turn"][turn]
        deltas[f"relative_cost_turn_{turn}_x"] = y - x
        print(f"{'rel cost @ turn ' + turn:<34}{x:>10.1f}x{y:>10.1f}x{y - x:>+10.1f}x")
    x = base["growth"]["growth_exponent"]
    y = strict["growth"]["growth_exponent"]
    deltas["growth_exponent"] = y - x
    print(f"{'growth exponent':<34}{x:>11.3f}{y:>11.3f}{y - x:>+11.3f}")

    Path(args.out).write_text(json.dumps({
        "baseline_n": base["n_conversations"],
        "strict_n": strict["n_conversations"],
        "dropped": dropped,
        "dropped_share": dropped / len(rows),
        "deltas": deltas,
        "note": ("Aggressive re-filter uses min_keyword_hits=1 instead of 2. "
                 "Small deltas mean residual coding content is not load-bearing "
                 "for Study A's conclusions."),
    }, indent=2))
    print(f"\n[sensitivity] wrote {args.out}")


if __name__ == "__main__":
    main()
